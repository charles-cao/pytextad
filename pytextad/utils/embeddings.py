"""Frozen token-embedding extraction (input for CVDD and other token-sequence detectors).

Returns, per document, the last-layer hidden states of a frozen Hugging Face encoder,
as a float32 array of shape [n_tokens, dim], plus the word index of every subword
(so that token scores can be aggregated back to words).
"""

from typing import List, Sequence, Union

import numpy as np
import torch

Text = Union[str, Sequence[str]]


class TokenEmbedder:

    def __init__(self, model_name_or_path, layer=-1, max_length=512, batch_size=32,
                 keep_special_tokens=False, device=None, dtype=torch.float32, cache_dir=None):
        """
        model_name_or_path : any Hugging Face encoder (bert-base-uncased, roberta-base, ...).
        layer              : hidden layer to return (-1 = last, as in official CVDD-BERT).
        max_length         : longer documents are truncated; the number truncated is
                             stored in ``n_truncated_`` after each call.
        keep_special_tokens: keep [CLS]/[SEP] (<s>/</s>) vectors. Default False.
        cache_dir          : Hugging Face cache folder, as in from_pretrained(..., cache_dir=...).
        """
        from transformers import AutoModel, AutoTokenizer
        self._path, self._cache = model_name_or_path, cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=cache_dir)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for word alignment.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:                                   # transformers >= 4.56 / 5.x
            model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, cache_dir=cache_dir)
        except TypeError:
            model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype, cache_dir=cache_dir)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.layer = layer
        self.max_length = max_length
        self.batch_size = batch_size
        self.keep_special_tokens = keep_special_tokens

    @torch.no_grad()
    def transform(self, texts: List[Text]):
        """Returns (embeddings, word_ids): two lists with one entry per document."""
        embs, wids = [], []
        self.n_truncated_ = 0
        for b in range(0, len(texts), self.batch_size):
            batch = list(texts[b:b + self.batch_size])
            split = not isinstance(batch[0], str)
            if split and getattr(self.tokenizer, "add_prefix_space", None) is False:
                # byte-level BPE tokenizers (RoBERTa, GPT-2) need this for pre-split words
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self._path, use_fast=True, add_prefix_space=True,
                                                               cache_dir=self._cache)
            enc = self.tokenizer(batch, is_split_into_words=split, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt",
                                 return_special_tokens_mask=True, return_overflowing_tokens=False)
            full = self.tokenizer(batch, is_split_into_words=split, add_special_tokens=True)["input_ids"]
            self.n_truncated_ += sum(len(f) > self.max_length for f in full)
            special = enc.pop("special_tokens_mask")
            inputs = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**inputs, output_hidden_states=True)
            H = out.hidden_states[self.layer].float().cpu().numpy()
            for i in range(len(batch)):
                keep = enc["attention_mask"][i].bool()
                if not self.keep_special_tokens:
                    keep &= ~special[i].bool()
                idx = keep.nonzero().squeeze(-1).numpy()
                embs.append(H[i, idx].astype(np.float32))
                wi = enc.word_ids(i)
                wids.append([wi[j] for j in idx])
        return embs, wids


def mean_pool(token_embeddings):
    """Document vector = mean of its token vectors (for RSRAE and other vector detectors)."""
    return np.stack([e.mean(0) for e in token_embeddings])


def words_from_subwords(scores, word_ids, n_words=None, agg="max"):
    """Aggregate one document's subword scores to word scores (max or mean)."""
    valid = [w for w in word_ids if w is not None]
    n = n_words if n_words is not None else (max(valid) + 1 if valid else 0)
    out = np.full(n, -np.inf) if agg == "max" else np.zeros(n)
    cnt = np.zeros(n)
    for s, w in zip(scores, word_ids):
        if w is None:
            continue
        out[w] = max(out[w], s) if agg == "max" else out[w] + s
        cnt[w] += 1
    if agg == "mean":
        out = out / np.maximum(cnt, 1)
    out[cnt == 0] = np.nan          # words lost to truncation have no score
    return out
