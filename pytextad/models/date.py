"""DATE: Detecting Anomalies in Text via Self-Supervision of Transformers
(Manolache, Brad & Burceanu, NAACL 2021).

Re-implementation for modern PyTorch / transformers (4.x and 5.x). Logic follows
the official code, https://github.com/bit-ml/date (experiments/train_ag.py defaults
and the modified simpletransformers fork it ships, Apache License 2.0).

Faithful to the official code
  * discriminator: ELECTRA encoder trained from scratch on inlier text
    (embedding 128, hidden 256, 4 layers, 4 heads, FFN 1024, dropout 0.5, gelu),
    RTD head dense-gelu-linear, RMD head on [CLS] linear-relu-linear
  * pretext task: one of K=50 fixed masks (50% of the 128 positions, first position
    always masked) chosen uniformly per sequence; masked positions get uniformly
    random token ids in [5, vocab-1) ("random_generator=1", the default)
  * loss = 50 * BCE(RTD) + 100 * CE(RMD); RTD labels are 0 where the random id
    happens to equal the original token
  * no attention mask is given to the encoder, and the RTD loss is taken over all
    positions including [CLS] and padding, exactly as in the official training loop
    (use_attention_mask=True changes this)
  * AdamW (amsgrad, eps 1e-8), lr 1e-5, weight decay 0.1, and 0.01 (AdamW's default,
    because the official parameter group omits the key) for bias/LayerNorm,
    gradient-norm clipping 1.0, batch 16, 20 epochs; the official "plateau"
    scheduler has patience 1e5 and therefore never changes the LR
  * sliding windows of 128 tokens with stride int(0.8 * 130) = 104 for training and
    testing; tokenizer bert-base-uncased
  * test: no masking; window score = mean over positions 1..128 that are not padding
    (so [SEP] is included unless it falls on position 129) of P(original token);
    document score = mean over its windows. The official AUROC is computed on this
    "PL_RTD" score with inliers as the positive class; decision_function returns
    1 - score so that higher = more anomalous (identical AUROC).

Removed relative to the official code (no effect with the default settings)
  * The ELECTRA generator: with random_generator=1 its samples are overwritten by
    random ids and its MLM loss is never added to the objective.

Not reproduced
  * The official code evaluates on the test set every 500 steps while training.
    Here you get the model after the last epoch.

Extension (not in the paper)
  * token_scores(): per-subword P(replaced), max over overlapping windows, aggregated
    to words with max.
"""

import pickle
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraModel

from .base import BaseTextDetector
from ..utils.embeddings import encode_words, words_from_subwords

Text = Union[str, Sequence[str]]


class _DATENet(nn.Module):
    def __init__(self, vocab_size, n_masks, hidden=256, layers=4, heads=4, intermediate=1024,
                 emb=128, dropout=0.5):
        super().__init__()
        cfg = ElectraConfig(vocab_size=vocab_size, embedding_size=emb, hidden_size=hidden,
                            num_hidden_layers=layers, num_attention_heads=heads,
                            intermediate_size=intermediate, hidden_dropout_prob=dropout,
                            attention_probs_dropout_prob=dropout, max_position_embeddings=512,
                            hidden_act="gelu")
        self.encoder = ElectraModel(cfg)
        self.rtd_dense = nn.Linear(hidden, hidden)        # ElectraDiscriminatorPredictions
        self.rtd_out = nn.Linear(hidden, 1)
        self.rmd_fc1 = nn.Linear(hidden, hidden)          # ElectraRMD
        self.rmd_fc2 = nn.Linear(hidden, n_masks)

    def forward(self, input_ids, attention_mask=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        rtd = self.rtd_out(F.gelu(self.rtd_dense(h))).squeeze(-1)    # [B, L]
        rmd = self.rmd_fc2(F.relu(self.rmd_fc1(h[:, 0])))            # [B, K]
        return rtd, rmd


class DATE(BaseTextDetector):
    supports_token = True


    def __init__(self, tokenizer="bert-base-uncased", max_len=128, stride=0.8,
                 n_masks=50, mask_ratio=0.5, masks=None,
                 hidden=256, layers=4, heads=4, intermediate=1024, emb=128, dropout=0.5,
                 rtd_weight=50.0, rmd_weight=100.0, lr=1e-5, weight_decay=0.1,
                 max_grad_norm=1.0, n_epochs=20, batch_size=16, use_attention_mask=False,
                 cache_dir=None, contamination=0.1, random_state=0, device=None, verbose=False):
        """
        tokenizer : HF name or path, a vocab.txt path, or a fast tokenizer object.
        cache_dir : Hugging Face cache folder, as in from_pretrained(..., cache_dir=...).
        masks     : None -> K random masks from random_state, or the path of the official
                    experiments/pseudo_labels128_p50.pkl to use the exact official set.
        """
        super().__init__(contamination, random_state, device, verbose)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.n_masks = n_masks
        self.mask_ratio = mask_ratio
        self.masks = masks
        self.hidden, self.layers, self.heads = hidden, layers, heads
        self.intermediate, self.emb, self.dropout = intermediate, emb, dropout
        self.rtd_weight, self.rmd_weight = rtd_weight, rmd_weight
        self.lr, self.weight_decay, self.max_grad_norm = lr, weight_decay, max_grad_norm
        self.n_epochs, self.batch_size = n_epochs, batch_size
        self.use_attention_mask = use_attention_mask
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------ setup
    def _load_tokenizer(self):
        from transformers import AutoTokenizer, BertTokenizerFast
        tok = self.tokenizer
        if isinstance(tok, str):
            if tok.endswith(".txt"):
                vocab = {w.rstrip("\n"): i for i, w in enumerate(open(tok, encoding="utf-8"))}
                try:                                   # transformers >= 5
                    tok = BertTokenizerFast(vocab=vocab, do_lower_case=True)
                except TypeError:
                    tok = None
                if tok is None or len(tok) < len(vocab):  # transformers 4.x
                    tok = BertTokenizerFast(vocab_file=self.tokenizer, do_lower_case=True)
            else:
                tok = AutoTokenizer.from_pretrained(tok, use_fast=True, cache_dir=self.cache_dir)
        if not getattr(tok, "is_fast", False):
            raise ValueError("A fast tokenizer is required (for word alignment).")
        self.tok_ = tok

    def _build_masks(self):
        L = self.max_len
        if isinstance(self.masks, str):
            raw = sorted(pickle.load(open(self.masks, "rb")), key=lambda m: m["label"])
            M = np.array([m["mask"] for m in raw], dtype=bool)
            if M.shape[1] != L:
                raise ValueError(f"mask length {M.shape[1]} != max_len {L}")
        else:
            rng = np.random.RandomState(self.random_state)
            k = int(self.mask_ratio * L)
            M = np.zeros((self.n_masks, L), dtype=bool)
            for i in range(self.n_masks):
                M[i, rng.choice(L, k, replace=False)] = True
        M = M.copy()
        M[:, 0] = True                                  # official: first content position always masked
        full = np.zeros((M.shape[0], L + 2), dtype=bool)
        full[:, 1:L + 1] = M
        self.masks_ = torch.from_numpy(full)
        self.n_masks_ = M.shape[0]

    # ------------------------------------------------------------------ windowing
    def _encode(self, texts: List[Text]):
        out = []
        for t in texts:
            if isinstance(t, str):
                enc = self.tok_(t, add_special_tokens=False, truncation=False)
                out.append((enc["input_ids"], enc.word_ids()))
            else:   # list of words: tokenised as running text, sub-words mapped back to words
                out.append(encode_words(self.tok_, list(t)))
        return out

    def _windows(self, ids):
        """encode_sliding_window_custom: windows of max_len, step int((max_len + 2) * stride)."""
        L = self.max_len
        if len(ids) <= L:
            return [(0, ids)]
        step = max(1, int((L + 2) * self.stride))
        return [(s, ids[s:s + L]) for s in range(0, len(ids), step)]

    def _pack(self, chunks):
        cls, sep, pad = self.tok_.cls_token_id, self.tok_.sep_token_id, self.tok_.pad_token_id
        X = torch.full((len(chunks), self.max_len + 2), pad, dtype=torch.long)
        for i, c in enumerate(chunks):
            seq = [cls] + list(c) + [sep]
            X[i, :len(seq)] = torch.tensor(seq)
        return X, X != pad

    # ------------------------------------------------------------------ training
    def _corrupt(self, X, nonpad):
        labels = torch.randint(0, self.n_masks_, (X.shape[0],))
        m = self.masks_[labels] & nonpad                # padding is never replaced
        rand = torch.randint(5, self.vocab_size_ - 1, X.shape)
        Xc = X.clone()
        Xc[m] = rand[m]
        return Xc, (m & (Xc != X)).float(), labels

    def _loss(self, Xc, nonpad, rtd_y, rmd_y):
        am = nonpad.long() if self.use_attention_mask else None
        rtd, rmd = self.net_(Xc, am)
        if self.use_attention_mask:
            l_rtd = F.binary_cross_entropy_with_logits(rtd[nonpad], rtd_y[nonpad])
        else:                                           # official: all positions
            l_rtd = F.binary_cross_entropy_with_logits(rtd, rtd_y)
        l_rmd = F.cross_entropy(rmd, rmd_y)
        return self.rtd_weight * l_rtd + self.rmd_weight * l_rmd

    def _make_optimizer(self):
        no_decay = ("bias", "LayerNorm.weight")
        named = list(self.net_.named_parameters())
        return torch.optim.AdamW(
            [{"params": [p for n, p in named if not any(k in n for k in no_decay)], "weight_decay": self.weight_decay},
             # the official group for bias/LayerNorm has no "weight_decay" key, so it gets
             # torch.optim.AdamW's default 0.01 (not 0)
             {"params": [p for n, p in named if any(k in n for k in no_decay)], "weight_decay": 0.01}],
            lr=self.lr, eps=1e-8, amsgrad=True)

    def _train_step(self, Xc, nonpad, rtd_y, rmd_y, opt):
        loss = self._loss(Xc, nonpad, rtd_y, rmd_y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net_.parameters(), self.max_grad_norm)
        opt.step()
        return loss.item()

    def fit(self, X: List[Text], y=None):
        self._set_seed()
        self._load_tokenizer()
        self.vocab_size_ = len(self.tok_)
        self._build_masks()
        windows = [w for ids, _ in self._encode(X) for _, w in self._windows(ids) if len(w) > 0]
        data, nonpad = self._pack(windows)

        self.net_ = _DATENet(self.vocab_size_, self.n_masks_, self.hidden, self.layers, self.heads,
                             self.intermediate, self.emb, self.dropout).to(self.device)
        opt = self._make_optimizer()

        n = data.shape[0]
        self.history_ = []
        for ep in range(self.n_epochs):
            self.net_.train()
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                Xc, rtd_y, rmd_y = self._corrupt(data[idx], nonpad[idx])
                loss = self._train_step(Xc.to(self.device), nonpad[idx].to(self.device),
                                        rtd_y.to(self.device), rmd_y.to(self.device), opt)
                tot += loss * len(idx)
            self.history_.append(tot / n)
            self._log(f"epoch {ep + 1}/{self.n_epochs} loss {tot / n:.4f} ({n} windows)")
        return self._process_decision_scores(self.decision_function(X))

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _score(self, X: List[Text], batch_size=64):
        self.net_.eval()
        enc = self._encode(X)
        jobs = [(d, s, w) for d, (ids, _) in enumerate(enc) for s, w in self._windows(ids) if len(w) > 0]
        sub = [np.full(len(ids), -np.inf) for ids, _ in enc]
        win = [[] for _ in enc]
        L2 = self.max_len + 2
        for b in range(0, len(jobs), batch_size):
            part = jobs[b:b + batch_size]
            xb, nonpad = self._pack([j[2] for j in part])
            am = nonpad.long().to(self.device) if self.use_attention_mask else None
            rtd, _ = self.net_(xb.to(self.device), am)
            p_rep = torch.sigmoid(rtd).cpu().numpy()
            for (d, s, w), pr, npd in zip(part, p_rep, nonpad.numpy()):
                keep = npd.copy()
                keep[0] = False
                keep[L2 - 1] = False                    # official skips positions 0 and 129
                win[d].append(1.0 - pr[keep].mean())    # PL_RTD = mean P(original)
                sub[d][s:s + len(w)] = np.maximum(sub[d][s:s + len(w)], pr[1:1 + len(w)])
        return win, sub, [e[1] for e in enc]

    def decision_function(self, X: List[Text]):
        win, _, _ = self._score(X)
        return np.array([1.0 - np.mean(w) if w else np.nan for w in win])

    def token_scores(self, X: List[Text], agg="max"):
        """EXTENSION: per-word P(replaced). For word lists the output has one score per word."""
        _, sub, wids = self._score(X)
        return [words_from_subwords(p, wid, len(x) if not isinstance(x, str) else None, agg)
                for x, p, wid in zip(X, sub, wids)]
