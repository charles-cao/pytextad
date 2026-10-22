"""Benchmark datasets with token-level and document-level anomaly labels.

Every dataset is a list of documents; each document is a list of words (``tokens``) with
one 0/1 label per word (``token_labels``, 1 = anomalous). A document is anomalous if any of
its words is (``labels``). Which documents are used for training is left to the user.

>>> from pytextad.datasets import load_dataset
>>> ds = load_dataset("restaurant_review")
>>> ds.tokens[0][:5], ds.token_labels[0][:5], ds.labels[:5]

Built-in datasets are downloaded once from the Hugging Face Hub into the usual Hugging Face
cache (or ``cache_dir``), checked against a SHA-256 checksum, and then read offline. Set
``HF_ENDPOINT`` to use a mirror. Your own data can be read with ``load_local`` or
``TextADDataset.from_lists``.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_CITE_TOKENCORE = """@inproceedings{cao2026tokenlevel,
  title     = {Towards Token-Level Text Anomaly Detection},
  author    = {Cao, Yang and Yu, Bicheng and Yang, Sikun and Liu, Ming and Yang, Yujiu},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  doi       = {10.1145/3774904.3792952}
}"""

# Hub commit of the data files. Every word is a unit of the word segmentation ``\w+|[^\w\s]``
# in normal and anomalous documents alike: an annotated span such as "not fresh" is two words,
# each labelled 1 (until 0.3.x, spans in anomalous documents were stored as single items, so
# the number of sub-words of an item revealed its label). Items that no tokenizer turns into
# a sub-word (lone invisible characters) were removed.
_REVISION = "331ef3b8f4a2c4a60bedf293ccf17f7e717c7643"

# name -> where the file lives and what it must contain. The checksum guarantees the content.
DATASETS: Dict[str, dict] = {
    "sms_spam": dict(
        repo_id="Charles-Cao/pytextad", filename="data/sms_spam.jsonl", revision=_REVISION,
        sha256="e315a7c1881c867bc464307234abb7518734abea36181a570aa83677ad8be16b",
        n_documents=4518, n_anomalous=393, n_tokens=82197, n_anomalous_tokens=424,
        anomaly="meaningless character sequences injected into SMS messages",
        source="SMS Spam Collection (via NLP-ADBench), annotated in Cao et al., WWW 2026",
        citation=_CITE_TOKENCORE),
    "restaurant_review": dict(
        repo_id="Charles-Cao/pytextad", filename="data/restaurant_review.jsonl", revision=_REVISION,
        sha256="fb6c53666488b5e2c7009443fe19077c0184e63cc8a9121b795542387db1e0db",
        n_documents=1100, n_anomalous=50, n_tokens=36222, n_anomalous_tokens=859,
        anomaly="words expressing negative sentiment in otherwise positive reviews",
        source="Google Maps restaurant reviews, collected and annotated in Cao et al., WWW 2026",
        citation=_CITE_TOKENCORE),
    "grammar_correction": dict(
        repo_id="Charles-Cao/pytextad", filename="data/grammar_correction.jsonl", revision=_REVISION,
        sha256="de0ad483497e0cfdc299898599edbfb430aada09e461b0e1be5a68f2b54bd7ba",
        n_documents=300, n_anomalous=30, n_tokens=2746, n_anomalous_tokens=47,
        anomaly="grammatical errors",
        source="Kaggle grammar-correction (satishgunjal), annotated in Cao et al., WWW 2026",
        citation=_CITE_TOKENCORE),
    "hate_speech": dict(
        repo_id="Charles-Cao/pytextad", filename="data/hate_speech.jsonl", revision=_REVISION,
        sha256="d7741418065ac141815df52c62aa957937fbc717cf0ae23f18229e2b99a16341",
        n_documents=4302, n_anomalous=140, n_tokens=99835, n_anomalous_tokens=484,
        anomaly="hateful or offensive words in tweets",
        source="Tweets from the hate speech and offensive language data of Davidson et al. (ICWSM "
               "2017), token-level annotation by Cao et al.",
        citation=_CITE_TOKENCORE),
    "olid": dict(
        repo_id="Charles-Cao/pytextad", filename="data/olid.jsonl", revision=_REVISION,
        sha256="455e4aec3b74e854db6a78c17907cfd086292c7450933ef378edd3a4f08fae1b",
        n_documents=650, n_anomalous=30, n_tokens=21256, n_anomalous_tokens=133,
        anomaly="offensive words and phrases in tweets",
        source="Tweets from OLID, the Offensive Language Identification Dataset of Zampieri et al. "
               "(NAACL 2019), token-level annotation by Cao et al.",
        citation=_CITE_TOKENCORE),
    "restaurant_review2": dict(
        repo_id="Charles-Cao/pytextad", filename="data/restaurant_review2.jsonl", revision=_REVISION,
        sha256="22b8d843415c0ec5c6c35aba318fe8e2efabdc80a75476d398e7bd8a37ffc7a5",
        n_documents=520, n_anomalous=25, n_tokens=33764, n_anomalous_tokens=208,
        anomaly="words expressing negative sentiment in otherwise positive reviews",
        source="Reviews of a second restaurant, collected and annotated by Cao et al.",
        citation=_CITE_TOKENCORE),
}
# short names, including those used in the CA-PTD code (spam, grammar, review1, review2)
ALIASES = {"spam": "sms_spam", "grammar": "grammar_correction", "review": "restaurant_review",
           "review1": "restaurant_review", "review2": "restaurant_review2"}

_WORD_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class TextADDataset:
    """Documents as word lists, with one 0/1 label per word.

    Attributes
    ----------
    tokens : list of list of str
        The words of each document; pass these to detectors and embedders.
    token_labels : list of numpy.ndarray
        One int array per document, aligned with ``tokens``.
    texts : list of str
        The original text of each document (may differ from ``" ".join(tokens)`` in spacing).
    name : str
    info : dict
        Source, citation and statistics.
    """
    tokens: List[List[str]]
    token_labels: List[np.ndarray]
    texts: List[str]
    name: str = "custom"
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.tokens) != len(self.token_labels) or len(self.tokens) != len(self.texts):
            raise ValueError(f"{len(self.tokens)} documents, {len(self.token_labels)} label lists, "
                             f"{len(self.texts)} texts")
        self.tokens = [list(map(str, t)) for t in self.tokens]
        self.token_labels = [np.asarray(l, dtype=np.int64).reshape(-1) for l in self.token_labels]
        for i, (t, l) in enumerate(zip(self.tokens, self.token_labels)):
            if len(t) != len(l):
                raise ValueError(f"document {i}: {len(t)} words but {len(l)} labels")
            if len(l) and not np.isin(l, (0, 1)).all():
                raise ValueError(f"document {i}: labels must be 0 or 1, got {sorted(set(l.tolist()))}")

    # --------------------------------------------------------------------- construction
    @classmethod
    def from_lists(cls, tokens, token_labels, texts=None, name="custom", info=None):
        """Build a dataset from word lists and per-word labels; ``texts`` default to the
        words joined by spaces."""
        texts = [" ".join(map(str, t)) for t in tokens] if texts is None else list(texts)
        return cls(list(tokens), list(token_labels), texts, name, dict(info or {}))

    # --------------------------------------------------------------------- views
    @property
    def labels(self) -> np.ndarray:
        """Document labels: 1 if any word of the document is anomalous."""
        return np.array([int(l.any()) for l in self.token_labels], dtype=np.int64)

    @property
    def normal_indices(self) -> np.ndarray:
        """Indices of the normal documents."""
        return np.flatnonzero(self.labels == 0)

    @property
    def anomalous_indices(self) -> np.ndarray:
        """Indices of the anomalous documents."""
        return np.flatnonzero(self.labels == 1)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        """An int gives one document as a dict; a slice, index array or boolean mask gives a
        new ``TextADDataset``."""
        if isinstance(idx, (int, np.integer)):
            return {"tokens": self.tokens[idx], "token_labels": self.token_labels[idx],
                    "text": self.texts[idx], "label": int(self.token_labels[idx].any())}
        return self.subset(idx)

    def subset(self, indices):
        """New dataset with the documents at ``indices`` (array, list, slice or boolean mask)."""
        if isinstance(indices, slice):
            indices = range(len(self))[indices]
        indices = np.asarray(indices)
        if indices.dtype == bool:
            if len(indices) != len(self):
                raise ValueError("boolean mask has the wrong length")
            indices = np.flatnonzero(indices)
        indices = indices.astype(np.int64)
        return TextADDataset([self.tokens[i] for i in indices], [self.token_labels[i] for i in indices],
                             [self.texts[i] for i in indices], self.name, dict(self.info))

    def stats(self) -> dict:
        """Numbers of documents and words, anomalous ones, and mean document length."""
        lab = self.labels
        n_tok = sum(len(t) for t in self.tokens)
        n_anom_tok = int(sum(l.sum() for l in self.token_labels))
        return {"n_documents": len(self), "n_anomalous": int(lab.sum()),
                "anomaly_rate": float(lab.mean()) if len(self) else 0.0,
                "n_tokens": n_tok, "n_anomalous_tokens": n_anom_tok,
                "mean_words": n_tok / max(len(self), 1)}

    def __repr__(self):
        s = self.stats()
        return (f"TextADDataset({self.name!r}: {s['n_documents']} documents, {s['n_anomalous']} anomalous; "
                f"{s['n_tokens']} words, {s['n_anomalous_tokens']} anomalous)")

    # --------------------------------------------------------------------- transforms
    def retokenize(self):
        """Split every word into ``\\w+`` runs and single punctuation marks (the segmentation
        of the built-in datasets); each piece keeps the label of the word it comes from.

        Use it on your own data when anomalous spans are stored as single items ("not
        fresh"): the evaluation unit must not depend on the labels, otherwise the number of
        sub-words of an item reveals whether it is anomalous."""
        tokens, labels = [], []
        for words, lab in zip(self.tokens, self.token_labels):
            t, l = [], []
            for w, y in zip(words, lab):
                pieces = _WORD_RE.findall(w) or [w]
                t.extend(pieces)
                l.extend([int(y)] * len(pieces))
            tokens.append(t)
            labels.append(np.array(l, dtype=np.int64))
        info = dict(self.info, retokenized=True)
        return TextADDataset(tokens, labels, list(self.texts), self.name, info)

    # --------------------------------------------------------------------- io
    def to_jsonl(self, path):
        """Write one JSON object per line: {"id", "text", "tokens", "labels"}."""
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i, (t, l, x) in enumerate(zip(self.tokens, self.token_labels, self.texts)):
                f.write(json.dumps({"id": i, "text": x, "tokens": t, "labels": l.tolist()},
                                   ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- reading
def _read_records(records, name, info):
    tokens, labels, texts = [], [], []
    for r in records:
        tokens.append(r["tokens"])
        labels.append(r["labels"] if "labels" in r else r["token_labels"])
        texts.append(r.get("text") if r.get("text") is not None else " ".join(r["tokens"]))
    return TextADDataset(tokens, labels, texts, name, info)


def _read_jsonl(path, name, info):
    with open(path, encoding="utf-8") as f:
        return _read_records((json.loads(line) for line in f if line.strip()), name, info)


def load_local(path, name=None):
    """Read a dataset from a local file.

    * ``.jsonl``: one object per line with "tokens", "labels" (or "token_labels") and
      optionally "text";
    * ``.json`` : a list of such objects;
    * ``.npy``  : the TokenCore format, a pickled dict with "tokens", "labels" and optionally
      "sentences". Loading it unpickles the file, so only use ``.npy`` files you trust.
    """
    name = name or os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return _read_jsonl(path, name, {"path": str(path)})
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            return _read_records(json.load(f), name, {"path": str(path)})
    if ext == ".npy":
        d = np.load(path, allow_pickle=True).item()
        texts = None
        if "sentences" in d:
            texts = [s[0] if isinstance(s, (list, tuple)) else s for s in d["sentences"]]
        return TextADDataset.from_lists(d["tokens"], d["labels"], texts, name, {"path": str(path)})
    raise ValueError(f"unsupported file type {ext!r}; use .jsonl, .json or .npy")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_datasets():
    """Names of the built-in datasets."""
    return sorted(DATASETS)


def dataset_info(name):
    """Source, citation and statistics of a built-in dataset."""
    return dict(DATASETS[_resolve(name)])


def _resolve(name):
    key = ALIASES.get(name, name)
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(list_datasets())}")
    return key


def load_dataset(name, cache_dir=None, revision=None):
    """Load a built-in dataset, downloading it from the Hugging Face Hub the first time.

    Parameters
    ----------
    name : str
        One of ``list_datasets()``; the short names "spam", "grammar", "review1"
        and "review2" also work.
    cache_dir : str, optional
        Download folder; default: the Hugging Face cache (``HF_HOME`` / ``HF_HUB_CACHE``).
    revision : str, optional
        Hub branch, tag or commit; default: the revision pinned in ``DATASETS``. With an
        explicit revision the checksum and count checks are skipped.

    Returns
    -------
    TextADDataset
    """
    key = _resolve(name)
    meta = DATASETS[key]
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(meta["repo_id"], meta["filename"], repo_type="dataset",
                               revision=revision or meta["revision"], cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"could not download {meta['repo_id']}/{meta['filename']}: {e}\n"
            "If huggingface.co is not reachable, set the HF_ENDPOINT environment variable to a "
            "mirror, or download the file yourself and use pytextad.datasets.load_local(path).") from e
    if revision is None and not meta["sha256"].startswith("__"):
        got = _sha256(path)
        if got != meta["sha256"]:
            raise RuntimeError(f"{path} has SHA-256 {got}, expected {meta['sha256']}; the file is "
                               "corrupted or was changed on the Hub. Delete it and retry.")
    info = {k: v for k, v in meta.items() if k != "sha256"}
    info["path"] = path
    ds = _read_jsonl(path, key, info)
    expected = {k: meta[k] for k in ("n_documents", "n_anomalous", "n_tokens", "n_anomalous_tokens")}
    got = {k: ds.stats()[k] for k in expected}
    if revision is None and got != expected:
        raise RuntimeError(f"{key}: expected {expected}, read {got}")
    _check_words(ds)
    return ds


def _check_words(ds):
    """Every word is one unit of the segmentation: no whitespace inside, not empty."""
    for i, words in enumerate(ds.tokens):
        for w in words:
            if not w.strip() or any(c.isspace() for c in w):
                raise RuntimeError(f"{ds.name}: document {i} has the word {w!r}; words must be "
                                   "single units without whitespace")
