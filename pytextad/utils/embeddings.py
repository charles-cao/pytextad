"""Frozen embeddings from Hugging Face models.

* ``TokenEmbedder``: one vector per sub-word token, or per word (``word_pooling``).
* ``SentenceEmbedder``: one vector per document (CLS, mean or last-token pooling).

Both accept documents as strings or as lists of words, read models from a custom
Hugging Face cache folder (``cache_dir``), and can store their output on disk
(``transform(..., cache="file.npz")``) so that an encoder runs only once per dataset.
"""

import hashlib
import json
import os
import warnings
from typing import List, Sequence, Union

import numpy as np
import torch

Text = Union[str, Sequence[str]]


# ----------------------------------------------------------------------------- loading
def _load(model_name_or_path, cache_dir, dtype, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=cache_dir)
    if not tok.is_fast:
        raise ValueError("A fast tokenizer is required for word alignment.")
    if tok.pad_token is None:                # GPT-2 style tokenizers have no padding token
        tok.pad_token = tok.eos_token
    # right padding for every model: with left padding, models with absolute position
    # embeddings (GPT-2, ...) would see shifted positions and give different vectors
    tok.padding_side = "right"
    try:                                    # transformers >= 4.56 / 5.x
        model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, cache_dir=cache_dir)
    except TypeError:
        model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype, cache_dir=cache_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model, device


class _HFEmbedder:
    def __init__(self, model_name_or_path, layer, max_length, batch_size, device, dtype, cache_dir):
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.layer = layer
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer, self.model, self.device = _load(model_name_or_path, cache_dir, dtype, device)

    def _tokenize(self, batch):
        """Tokenise a batch. Returns (encoding, word_ids) where word_ids[i][j] is the word index
        of position j of document i (None for special and padding positions).

        Lists of words are joined with single spaces and tokenised as ordinary text; each
        sub-word is then mapped to the word whose characters it covers. This gives every
        tokenizer, including byte-level BPE ones (GPT-2, RoBERTa, Qwen), the same input as for
        running text, which ``is_split_into_words=True`` does not guarantee."""
        split = not isinstance(batch[0], str)
        texts, spans = [], []
        for doc in batch:
            if split:
                text, sp, pos = [], [], 0
                for w in doc:
                    sp.append((pos, pos + len(w)))
                    text.append(w)
                    pos += len(w) + 1
                texts.append(" ".join(text))
                spans.append(sp)
            else:
                texts.append(doc)
        kw = dict(padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
                  return_special_tokens_mask=True, return_offsets_mapping=split)
        enc = self.tokenizer(texts, **kw)
        full = self.tokenizer(texts, add_special_tokens=True)["input_ids"]
        self.n_truncated_ += sum(len(f) > self.max_length for f in full)
        if not split:
            return enc, [enc.word_ids(i) for i in range(len(texts))]
        offsets = enc.pop("offset_mapping").tolist()
        word_ids = []
        for i, sp in enumerate(spans):
            special = enc["special_tokens_mask"][i].tolist()
            mask = enc["attention_mask"][i].tolist()
            row = []
            for (a, b), sp_tok, m in zip(offsets[i], special, mask):
                w = None
                if m and not sp_tok and b > a:
                    for k, (ws, we) in enumerate(sp):
                        if a < we and b > ws:
                            w = k
                            break
                    else:   # a token made only of the separating space belongs to the next word
                        nxt = [k for k, (ws, _) in enumerate(sp) if ws >= a]
                        w = nxt[0] if nxt else None
                row.append(w)
            word_ids.append(row)
        return enc, word_ids

    def _hidden(self, enc):
        inputs = {k: v.to(self.device) for k, v in enc.items() if k not in ("special_tokens_mask", "offset_mapping")}
        out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer].float()

    def _key(self, texts, extra):
        h = hashlib.sha1()
        h.update(json.dumps([str(self.model_name_or_path), self.layer, self.max_length, extra]).encode())
        for t in texts:
            h.update(json.dumps(t if isinstance(t, str) else list(t), ensure_ascii=False).encode())
            h.update(b"\x00")
        return h.hexdigest()


# ----------------------------------------------------------------------------- token level
class TokenEmbedder(_HFEmbedder):
    """Frozen contextual embeddings of every token or word, from any Hugging Face model.

    Parameters
    ----------
    model_name_or_path : str
        Hugging Face name (``bert-base-uncased``, ``roberta-base``,
        ``Qwen/Qwen3-Embedding-0.6B``, ...) or local folder of an encoder or decoder.
    layer : int, default=-1
        Hidden layer to return (-1 = last).
    max_length : int, default=512
        Longer documents are truncated; ``n_truncated_`` counts them.
    batch_size : int, default=32
        Documents per forward pass.
    keep_special_tokens : bool, default=False
        Keep [CLS]/[SEP]-like positions (sub-word mode only).
    word_pooling : {None, "max", "mean", "first"}, default=None
        None returns one vector per sub-word (``TokenDetector`` then combines sub-word
        scores per word); otherwise one vector per word, pooling its sub-word vectors
        (documents must then be word lists; TokenCore as published uses "max").
    device : str or None, default=None
        "cuda" or "cpu"; None picks CUDA when available.
    dtype : torch.dtype, default=torch.float32
        Model precision.
    cache_dir : str or None, default=None
        Hugging Face cache folder.
    """

    def __init__(self, model_name_or_path, layer=-1, max_length=512, batch_size=32,
                 keep_special_tokens=False, word_pooling=None, device=None, dtype=torch.float32,
                 cache_dir=None):
        if word_pooling not in (None, "max", "mean", "first"):
            raise ValueError("word_pooling must be None, 'max', 'mean' or 'first'")
        super().__init__(model_name_or_path, layer, max_length, batch_size, device, dtype, cache_dir)
        self.keep_special_tokens = keep_special_tokens
        self.word_pooling = word_pooling

    @torch.no_grad()
    def transform(self, texts: List[Text], cache=None):
        """Embed documents.

        Parameters
        ----------
        texts : list of str or list of list of str
            Documents as strings or word lists (word lists are required with ``word_pooling``).
        cache : str or None, default=None
            ``.npz`` file; reused when written for the same model, settings and texts.

        Returns
        -------
        embeddings : list of numpy.ndarray
            One ``[n_units, dim]`` array per document (sub-words, or words with ``word_pooling``).
        ids : list of list
            For sub-words, the word index of each sub-word (None for special tokens); for
            words, the indices of the embedded words (words cut off by ``max_length`` are missing).
        """
        texts = list(texts)
        if self.word_pooling is not None and any(isinstance(t, str) for t in texts):
            raise ValueError("word_pooling needs documents given as lists of words")
        key = self._key(texts, ["token", self.keep_special_tokens, self.word_pooling])
        if cache is not None and os.path.exists(cache):
            loaded = _load_ragged(cache, key)
            if loaded is not None:
                self.n_truncated_ = loaded[2]
                return loaded[0], loaded[1]
        embs, ids = [], []
        self.n_truncated_ = 0
        incomplete = []
        for b in range(0, len(texts), self.batch_size):
            batch = texts[b:b + self.batch_size]
            enc, word_ids = self._tokenize(batch)
            H = self._hidden(enc).cpu().numpy()
            for i in range(len(batch)):
                keep = enc["attention_mask"][i].bool()
                wi = word_ids[i]
                if self.word_pooling is None:
                    if not self.keep_special_tokens:
                        keep &= ~enc["special_tokens_mask"][i].bool()
                    pos = keep.nonzero().squeeze(-1).numpy()
                    embs.append(H[i, pos].astype(np.float32))
                    ids.append([wi[j] for j in pos])
                    if not isinstance(batch[i], str) and \
                            len({w for w in ids[-1] if w is not None}) < len(batch[i]):
                        incomplete.append(b + i)
                else:
                    groups = {}
                    for j, w in enumerate(wi):
                        if w is not None and keep[j]:
                            groups.setdefault(w, []).append(j)
                    words = sorted(groups)
                    if self.word_pooling == "max":
                        vec = [H[i, groups[w]].max(0) for w in words]
                    elif self.word_pooling == "mean":
                        vec = [H[i, groups[w]].mean(0) for w in words]
                    else:
                        vec = [H[i, groups[w][0]] for w in words]
                    dim = H.shape[-1]
                    embs.append(np.stack(vec).astype(np.float32) if vec else np.zeros((0, dim), np.float32))
                    ids.append(words)
                    if len(words) < len(batch[i]):
                        incomplete.append(b + i)
        if incomplete:
            warnings.warn(f"{len(incomplete)} document(s) have words without an embedding (truncated, "
                          f"or empty after tokenisation), e.g. documents {incomplete[:5]}; use the "
                          f"returned word indices to align labels", stacklevel=3)
        if cache is not None:
            _save_ragged(cache, key, embs, ids, self.n_truncated_)
        return embs, ids


# ----------------------------------------------------------------------------- sentence level
class SentenceEmbedder(_HFEmbedder):
    """Frozen document embeddings from any Hugging Face model.

    Parameters
    ----------
    model_name_or_path : str
        Hugging Face name or local folder of an encoder or decoder.
    pooling : {"auto", "cls", "mean", "last"}, default="auto"
        "cls": first position; "mean": mean over non-padding positions; "last": last
        non-padding position; "auto": "cls" if the tokenizer has a CLS token (BERT,
        RoBERTa), else "last" (decoders such as GPT and Qwen).
    layer : int, default=-1
        Hidden layer to use (-1 = last).
    max_length : int, default=512
        Longer documents are truncated.
    batch_size : int, default=32
        Documents per forward pass.
    normalize : bool, default=False
        L2-normalise the output vectors.
    device : str or None, default=None
        "cuda" or "cpu"; None picks CUDA when available.
    dtype : torch.dtype, default=torch.float32
        Model precision.
    cache_dir : str or None, default=None
        Hugging Face cache folder.
    """

    def __init__(self, model_name_or_path, pooling="auto", layer=-1, max_length=512, batch_size=32,
                 normalize=False, device=None, dtype=torch.float32, cache_dir=None):
        if pooling not in ("auto", "cls", "mean", "last"):
            raise ValueError("pooling must be 'auto', 'cls', 'mean' or 'last'")
        super().__init__(model_name_or_path, layer, max_length, batch_size, device, dtype, cache_dir)
        if pooling == "auto":
            pooling = "cls" if self.tokenizer.cls_token is not None else "last"
        self.pooling = pooling
        self.normalize = normalize

    @torch.no_grad()
    def transform(self, texts: List[Text], cache=None):
        """Embed documents.

        Parameters
        ----------
        texts : list of str or list of list of str
            Documents as strings or word lists.
        cache : str or None, default=None
            ``.npz`` file; reused when written for the same model, settings and texts.

        Returns
        -------
        numpy.ndarray
            ``[n_documents, dim]``.
        """
        texts = list(texts)
        key = self._key(texts, ["sentence", self.pooling, self.normalize])
        if cache is not None and os.path.exists(cache):
            loaded = _load_ragged(cache, key)
            if loaded is not None:
                self.n_truncated_ = loaded[2]
                return np.stack(loaded[0])[:, 0]
        out = []
        self.n_truncated_ = 0
        for b in range(0, len(texts), self.batch_size):
            enc, _ = self._tokenize(texts[b:b + self.batch_size])
            H = self._hidden(enc)
            mask = enc["attention_mask"].to(H.device)
            if self.pooling == "cls":
                v = H[:, 0]
            elif self.pooling == "mean":
                m = mask.unsqueeze(-1).float()
                v = (H * m).sum(1) / m.sum(1).clamp(min=1)
            else:   # last non-padding position, for left or right padding
                pos = torch.arange(mask.shape[1], device=H.device).expand_as(mask)
                last = (pos * mask).argmax(1)
                v = H[torch.arange(H.shape[0], device=H.device), last]
            if self.normalize:
                v = torch.nn.functional.normalize(v, dim=-1)
            out.append(v.cpu().numpy().astype(np.float32))
        V = np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), np.float32)
        if cache is not None:
            _save_ragged(cache, key, [v[None] for v in V], [[0]] * len(V), self.n_truncated_)
        return V


# ----------------------------------------------------------------------------- helpers
def _save_ragged(path, key, arrays, ids, n_truncated):
    lengths = np.array([len(a) for a in arrays], dtype=np.int64)
    dim = arrays[0].shape[1] if arrays else 0
    data = np.concatenate(arrays) if arrays else np.zeros((0, dim), np.float32)
    flat_ids = np.array([-1 if w is None else w for row in ids for w in row], dtype=np.int64)
    id_len = np.array([len(r) for r in ids], dtype=np.int64)
    np.savez(path, key=key, data=data, lengths=lengths, ids=flat_ids, id_lengths=id_len,
             n_truncated=n_truncated)


def _load_ragged(path, key):
    z = np.load(path, allow_pickle=False)
    if str(z["key"]) != key:
        warnings.warn(f"{path} was written for different texts or settings; recomputing", stacklevel=3)
        return None
    arrays = np.split(z["data"], np.cumsum(z["lengths"])[:-1]) if len(z["lengths"]) else []
    flat = np.split(z["ids"], np.cumsum(z["id_lengths"])[:-1]) if len(z["id_lengths"]) else []
    ids = [[None if w < 0 else int(w) for w in row] for row in flat]
    return list(arrays), ids, int(z["n_truncated"])


def encode_words(tokenizer, words, add_special_tokens=False):
    """Token ids and per-token word indices for one document given as a list of words,
    tokenised as the running text ' '.join(words) (see ``_HFEmbedder._tokenize``)."""
    spans, pos = [], 0
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    enc = tokenizer(" ".join(words), add_special_tokens=add_special_tokens, truncation=False,
                    return_offsets_mapping=True, return_special_tokens_mask=True)
    word_ids = []
    for (a, b), special in zip(enc["offset_mapping"], enc["special_tokens_mask"]):
        w = None
        if not special and b > a:
            for k, (ws, we) in enumerate(spans):
                if a < we and b > ws:
                    w = k
                    break
            else:
                nxt = [k for k, (ws, _) in enumerate(spans) if ws >= a]
                w = nxt[0] if nxt else None
        word_ids.append(w)
    return list(enc["input_ids"]), word_ids


def mean_pool(token_embeddings):
    """Document vector = mean of its token vectors."""
    return np.stack([e.mean(0) for e in token_embeddings])


def words_from_subwords(scores, word_ids, n_words=None, agg="max"):
    """Aggregate one document's sub-word scores to word scores (max or mean).
    Words without any sub-word (e.g. truncated) get NaN."""
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
    out[cnt == 0] = np.nan
    return out


def align_labels(token_labels, ids):
    """Keep the labels of the words that have a vector.

    ``ids`` is the second output of ``TokenEmbedder.transform`` (sub-word or word mode).
    The result is aligned with ``TokenDetector.token_scores`` given the ``(embeddings, ids)``
    pair: words cut off by ``max_length`` are dropped."""
    out = []
    for lab, i in zip(token_labels, ids):
        kept = sorted({w for w in i if w is not None})
        out.append(np.asarray(lab)[kept])
    return out
