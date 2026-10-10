"""Turn any embedding detector (SIK, TokenCore, RSRAE, or your own) into a text detector.

An "embedding detector" is any object with ``fit(X)`` and ``decision_function(X)`` working on a
2-D array, higher scores meaning more anomalous.

* ``DocumentDetector``: one vector per document (e.g. from ``SentenceEmbedder``).
* ``TokenDetector``   : one vector per sub-word (or word); the detector is fitted on the
  vectors of all training documents and scores every one; sub-word scores are combined
  into word scores, and word scores into a document score.
"""

import numpy as np

from ..metrics import aggregate
from ..utils.embeddings import words_from_subwords
from ..utils.rng import preserve_rng, seed_everything
from .base import BaseTextDetector


def _is_text(X):
    return len(X) > 0 and (isinstance(X[0], str) or
                           (isinstance(X[0], (list, tuple)) and len(X[0]) > 0 and isinstance(X[0][0], str)))


def _as_float(x):
    """Array of x, keeping its float precision (float64 vectors are not rounded to float32,
    which would change the scores of a detector compared with running it directly)."""
    x = np.asarray(x)
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float64)


def _seed(wrapper):
    """Seed the wrapped detector and the global generators before fitting, as benchmark
    scripts do (most published deep detectors use the global generators)."""
    if wrapper.random_state is None:
        return
    if hasattr(wrapper.detector, "random_state"):
        wrapper.detector.random_state = wrapper.random_state
    seed_everything(wrapper.random_state)


class DocumentDetector(BaseTextDetector):
    """Use any anomaly detector for embeddings (e.g. SIK, or a PyOD detector :cite:`zhao2019pyod`) on document embeddings.

    Parameters
    ----------
    detector : object
        Unfitted detector with ``fit(X)`` and ``decision_function(X)``, for example
        ``pytextad.SIK()``.
    embedder : SentenceEmbedder or None, default=None
        If given, ``fit`` and ``decision_function`` accept raw texts; otherwise ``X`` is
        a ``[n_documents, dim]`` array.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=None
        If not None, copied to ``detector.random_state`` and used to seed the global random
        generators (Python, NumPy, PyTorch) before fitting.
    verbose : bool, default=False
        Print progress.

    Examples
    --------
    >>> from pytextad import SIK, DocumentDetector, SentenceEmbedder
    >>> det = DocumentDetector(SIK(), embedder=SentenceEmbedder("bert-base-uncased"))
    >>> scores = det.fit(train_texts).decision_function(test_texts)
    """
    supports_token = False

    def __init__(self, detector, embedder=None, contamination=0.1, random_state=None, verbose=False):
        super().__init__(contamination, random_state, device="cpu", verbose=verbose)
        self.detector = detector
        self.embedder = embedder

    def _vectors(self, X):
        if _is_text(X):
            if self.embedder is None:
                raise ValueError("texts given but no embedder; pass a SentenceEmbedder or vectors")
            return self.embedder.transform(X)
        return _as_float(X)

    def fit(self, X, y=None):
        """Fit the detector on training documents ``X`` and return ``self``."""
        V = self._vectors(X)
        _seed(self)
        self.detector.fit(V)
        with preserve_rng():
            return self._process_decision_scores(self.detector.decision_function(V))

    def decision_function(self, X):
        """Anomaly score of each document in ``X`` (higher = more anomalous)."""
        return np.asarray(self.detector.decision_function(self._vectors(X)), dtype=float)


class TokenDetector(BaseTextDetector):
    """Use any anomaly detector for embeddings (e.g. TokenCore, or a PyOD detector :cite:`zhao2019pyod`) on token embeddings.

    The detector is fitted on the vectors of all training documents together and scores
    every vector; a document's score aggregates its token scores.

    With sub-word vectors (``TokenEmbedder`` without ``word_pooling``, the default) and
    documents given as word lists, every sub-word is scored and the scores of a word's
    sub-words are combined into the word's score (``subword_aggregation``). The vectors then
    do not depend on how the words are grouped (for example an annotated span "not fresh"
    given as one item); the grouping is only used to combine scores. With word vectors
    (``TokenEmbedder(word_pooling="max")``, as in TokenCore :cite:`cao2026tokencore`) the
    sub-word vectors of each item are pooled first and every item is scored once.

    Parameters
    ----------
    detector : object
        Unfitted detector with ``fit(X)`` and ``decision_function(X)``, for example
        ``pytextad.SIK()``.
    embedder : TokenEmbedder or None, default=None
        If given, ``fit`` and ``decision_function`` accept raw texts or word lists.
        Otherwise ``X`` is either the ``(embeddings, ids)`` pair returned by
        ``TokenEmbedder.transform`` (scores are then combined per word with ``ids``), or a
        list of ``[n_tokens, dim]`` arrays, one per document (one score per row).
    subword_aggregation : {"max", "mean"}, default="max"
        How the scores of a word's sub-words are combined into the word's score.
    aggregation : {"max", "mean", "topk"}, default="max"
        How token scores are combined into the document score.
    k : float, default=0.1
        Fraction of tokens averaged by ``aggregation="topk"``.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=None
        If not None, copied to ``detector.random_state`` and used to seed the global random
        generators (Python, NumPy, PyTorch) before fitting.
    verbose : bool, default=False
        Print progress.

    Examples
    --------
    >>> from pytextad import SIK, TokenDetector, TokenEmbedder
    >>> det = TokenDetector(SIK(), embedder=TokenEmbedder("bert-base-uncased")).fit(train_words)
    >>> word_scores = det.token_scores(test_words)      # one score per word
    """
    supports_token = True

    def __init__(self, detector, embedder=None, subword_aggregation="max", aggregation="max", k=0.1,
                 contamination=0.1, random_state=None, verbose=False):
        super().__init__(contamination, random_state, device="cpu", verbose=verbose)
        self.detector = detector
        self.embedder = embedder
        self.subword_aggregation = subword_aggregation
        self.aggregation = aggregation
        self.k = k

    def _units(self, X):
        """Vectors of every document, and for each document how its row scores become token
        scores: None (one token per row) or (ids, n_words); n_words None keeps only the words
        that have a vector."""
        if isinstance(X, tuple) and len(X) == 2:            # (embeddings, ids) from TokenEmbedder
            T, ids = X
            return [_as_float(x) for x in T], [(list(i), None) for i in ids]
        if _is_text(X):
            if self.embedder is None:
                raise ValueError("texts given but no embedder; pass a TokenEmbedder or token arrays")
            T, ids = self.embedder.transform(X)
            if isinstance(X[0], str):                        # running text: one score per sub-word
                return T, [None] * len(T)
            return T, [(list(i), len(doc)) for doc, i in zip(X, ids)]
        return [_as_float(x) for x in X], [None] * len(X)

    def _token_scores(self, flat_scores, T, layout):
        lengths = [len(t) for t in T]
        rows = np.split(np.asarray(flat_scores, dtype=float), np.cumsum(lengths)[:-1]) if lengths else []
        out = []
        for s, lay in zip(rows, layout):
            if lay is None:
                out.append(s)
                continue
            ids, n_words = lay
            if self.subword_aggregation not in ("max", "mean"):
                raise ValueError("subword_aggregation must be 'max' or 'mean'")
            w = words_from_subwords(s, ids, n_words, self.subword_aggregation)
            if n_words is None:
                w = w[sorted({i for i in ids if i is not None})]
            out.append(w)
        return out

    def fit(self, X, y=None):
        """Fit the detector on training documents ``X`` and return ``self``."""
        T, layout = self._units(X)
        flat = np.vstack(T)
        _seed(self)
        self.detector.fit(flat)
        with preserve_rng():
            train_tokens = self._token_scores(self.detector.decision_function(flat), T, layout)
        return self._process_decision_scores(aggregate(train_tokens, self.aggregation, self.k))

    def token_scores(self, X):
        """One array of token scores per document.

        For word lists given with an embedder: one score per input word, NaN for words
        without a vector (truncated). For an ``(embeddings, ids)`` pair: one score per word
        that has a vector, in word order (see :func:`~pytextad.utils.embeddings.align_labels`).
        Otherwise one score per row."""
        T, layout = self._units(X)
        return self._token_scores(self.detector.decision_function(np.vstack(T)), T, layout)

    def decision_function(self, X):
        """Anomaly score of each document in ``X`` (higher = more anomalous)."""
        return aggregate(self.token_scores(X), self.aggregation, self.k)
