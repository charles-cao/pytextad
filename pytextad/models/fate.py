"""FATE: Few-shot Anomaly detection in TExt (Das, Ajay, Saha & Bhuyan, ICONIP 2023).

Re-implementation of the official code, https://github.com/arav1ndajay/fate
(src/main.py, deviation_loss.py, balanced_sampler.py). The repository has no
licence file, so no code is copied; the logic is re-written and was checked
numerically against it (see tests/).

Faithful to the official code
  * encoder: a SentenceTransformer, fine-tuned end to end; default
    sentence-transformers/all-MiniLM-L6-v2 (the one hard-coded in main.py);
    h = its token embeddings (last transformer layer, before pooling)
  * inputs padded to max_length=128 and the attention softmax runs over all 128
    positions, padding included (mask_padding=True changes this)
  * A = softmax_over_tokens(tanh(h W1) W2), attention size 150, 5 heads, no biases;
    S = A h flattened (heads x dim); score = mean of the top 10 % of |S|
    (the code takes absolute values; the paper's formula does not)
  * deviation loss: a new reference sample of 5000 N(0, 1) draws at EVERY batch;
    Z = (score - mean(ref)) / std(ref); loss = mean((1-y)|Z| + y max(0, 5 - Z))
  * regulariser mean((A A^T - I)^2) (a mean, not the paper's Frobenius sum), weight 1
  * balanced batches: 8 inliers + 8 labelled anomalies, each drawn from its own
    endlessly re-shuffled list; steps per epoch = (n_inliers + n_anomalies) // 16
  * Adam, lr 1e-6, batch 16; epochs in main.py: 4 for AG News, 50 for 20 Newsgroups,
    40 for Reuters, 80 whenever there are < 500 training inliers (default here: 4)
  * test score = the network output (no Z-transform), higher = more anomalous

Not reproduced
  * main.py evaluates on the test set every 600 steps during training; here you get
    the model after the last epoch.

Unsupervised use (y=None or no 1 in y)
  The official sampler cannot run without labelled anomalies. We then fill whole
  batches with inliers and train only the |Z| term: this is the "FATE*" variant of
  NLP-ADBench, not the method of the paper, and should be reported as such.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTextDetector


class _FATENet(nn.Module):
    def __init__(self, encoder, attention_size, n_heads, top_k, mask_padding):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.W1 = nn.Linear(d, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_heads, bias=False)
        self.top_k = top_k
        self.mask_padding = mask_padding

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # [B, T, d]
        logits = self.W2(torch.tanh(self.W1(h)))                                                 # [B, T, m]
        if self.mask_padding:
            logits = logits.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
        A = F.softmax(logits, dim=1).transpose(1, 2)                                              # [B, m, T]
        S = (A @ h).flatten(1)                                                                    # [B, m*d]
        k = max(int(S.size(1) * self.top_k), 1)
        return torch.topk(S.abs(), k, dim=1)[0].mean(1).float(), A


def _endless(idx):
    while True:
        for i in np.random.permutation(idx):
            yield i


class FATE(BaseTextDetector):
    """Few-shot Anomaly detection in TExt with deviation learning (Das et al., ICONIP 2023).

    A sentence encoder with multi-head self-attention is fine-tuned so that a few
    labelled anomalies score far above a Gaussian reference while normal documents stay
    close to it.

    Parameters
    ----------
    encoder : str, default="sentence-transformers/all-MiniLM-L6-v2"
        Hugging Face name or path of the encoder, fine-tuned end to end.
    max_length : int, default=128
        Inputs are padded or truncated to this many tokens.
    attention_size : int, default=150
        Hidden size of the self-attention layer.
    n_heads : int, default=5
        Number of attention heads.
    top_k : float, default=0.1
        The score is the mean of the top ``top_k`` fraction of absolute features.
    margin : float, default=5.0
        Deviation margin for anomalies.
    n_ref : int, default=5000
        Size of the Gaussian reference sample, redrawn for every batch.
    include_regularization : bool, default=True
        Add the attention orthogonality penalty.
    mask_padding : bool, default=False
        Exclude padding from the attention (the official code does not).
    lr : float, default=1e-6
        Adam learning rate.
    batch_size : int, default=16
        Mini-batch size (half normal, half labelled anomalies).
    n_epochs : int, default=4
        Training epochs.
    cache_dir : str or None, default=None
        Hugging Face cache folder for the encoder.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=0
        Seed for all random number generators.
    device : str or None, default=None
        "cuda" or "cpu"; None picks CUDA when available.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    decision_scores_ : numpy.ndarray
        Anomaly scores of the training data (higher = more anomalous).
    threshold_ : float
        Score above which ``predict`` returns 1.
    labels_ : numpy.ndarray
        Binary labels of the training data.

    Notes
    -----
    ``X`` is a list of strings; pass ``y`` (1 = labelled anomaly) to ``fit``. Without
    labelled anomalies only the normal-data term is trained (the "FATE*" variant).
    Differences from the official code are listed in :doc:`/faithfulness`.
    """
    supports_token = False


    def __init__(self, encoder="sentence-transformers/all-MiniLM-L6-v2", max_length=128,
                 attention_size=150, n_heads=5, top_k=0.1, margin=5.0, n_ref=5000,
                 include_regularization=True, mask_padding=False, lr=1e-6, batch_size=16,
                 n_epochs=4, cache_dir=None, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device, verbose)
        self.encoder = encoder
        self.max_length = max_length
        self.attention_size = attention_size
        self.n_heads = n_heads
        self.top_k = top_k
        self.margin = margin
        self.n_ref = n_ref
        self.include_regularization = include_regularization
        self.mask_padding = mask_padding
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------ pieces
    def _tok(self, texts):
        pad = "longest" if self.mask_padding else "max_length"     # official: pad to 128
        return self.tokenizer_(list(texts), padding=pad, truncation=True, max_length=self.max_length,
                               return_tensors="pt")

    def _build(self):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.encoder, cache_dir=self.cache_dir)
        enc = AutoModel.from_pretrained(self.encoder, cache_dir=self.cache_dir)
        self.net_ = _FATENet(enc, self.attention_size, self.n_heads, self.top_k, self.mask_padding).to(self.device)

    def _loss(self, input_ids, attention_mask, yb):
        psi, A = self.net_(input_ids, attention_mask)
        ref = torch.normal(mean=0.0, std=torch.full([self.n_ref], 1.0)).to(self.device)   # redrawn every batch
        z = (psi - ref.mean()) / ref.std()
        loss = torch.mean((1 - yb) * z.abs() + yb * (self.margin - z).clamp(min=0.0))
        if self.include_regularization:
            I = torch.eye(self.n_heads, device=self.device)
            loss = loss + torch.mean((A @ A.transpose(1, 2) - I) ** 2)
        return loss

    def _train_steps(self, batches, opt):
        """batches: iterable of (input_ids, attention_mask, labels) tensors."""
        self.net_.train()
        tot, nb = 0.0, 0
        for ids, am, yb in batches:
            opt.zero_grad()
            loss = self._loss(ids.to(self.device), am.to(self.device), yb.float().to(self.device))
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        return tot / max(nb, 1)

    # ------------------------------------------------------------------ API
    def fit(self, X: List[str], y: Optional[np.ndarray] = None):
        """Fit on texts ``X``; ``y`` marks labelled anomalies (1) and inliers (0). Returns ``self``."""
        self._set_seed()
        X = list(X)
        y = np.zeros(len(X), dtype=int) if y is None else np.asarray(y, dtype=int)
        if len(y) != len(X):
            raise ValueError("X and y have different lengths")
        inl, out = np.where(y == 0)[0], np.where(y == 1)[0]
        self.few_shot_ = len(out) > 0
        if not self.few_shot_:
            self._log("no labelled anomalies: training the unsupervised FATE* variant")
        self._build()
        enc = self._tok(X)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        n_in = self.batch_size // 2 if self.few_shot_ else self.batch_size
        gen_in, gen_out = _endless(inl), (_endless(out) if self.few_shot_ else None)
        steps = len(X) // self.batch_size

        def batches():
            for _ in range(steps):
                idx = [next(gen_in) for _ in range(n_in)]
                if self.few_shot_:
                    idx += [next(gen_out) for _ in range(self.batch_size - n_in)]
                idx = torch.tensor(idx)
                yield enc["input_ids"][idx], enc["attention_mask"][idx], torch.from_numpy(y[idx.numpy()])

        self.history_ = []
        for ep in range(self.n_epochs):
            self.history_.append(self._train_steps(batches(), opt))
            self._log(f"epoch {ep + 1}/{self.n_epochs} loss {self.history_[-1]:.4f}")
        return self._process_decision_scores(self.decision_function(X))

    @torch.no_grad()
    def decision_function(self, X: List[str], batch_size=16):
        """Anomaly score of each document in ``X`` (higher = more anomalous)."""
        self.net_.eval()
        X = list(X)
        out = []
        for s in range(0, len(X), batch_size):
            t = self._tok(X[s:s + batch_size])
            out.append(self.net_(t["input_ids"].to(self.device), t["attention_mask"].to(self.device))[0].cpu().numpy())
        return np.concatenate(out)
