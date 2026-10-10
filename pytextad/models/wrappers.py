"""Turn any vector detector (PyOD, scikit-learn style, SIK, RSRAE, ...) into a text detector.

A "vector detector" is any object with ``fit(X)`` and ``decision_function(X)`` working on a
2-D array, higher scores meaning more anomalous; every PyOD model qualifies.

* ``DocumentDetector``: one vector per document (e.g. from ``SentenceEmbedder``).
* ``TokenDetector``   : one vector per token; the detector is fitted on the tokens of all
  training documents, scores every token, and token scores are aggregated into a
  document score (the TokenCore "score aggregation" protocol).
"""

import numpy as np

from ..metrics import aggregate
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
    scripts do (deep PyOD models and most published code use the global generators)."""
    if wrapper.random_state is None:
        return
    if hasattr(wrapper.detector, "random_state"):
        wrapper.detector.random_state = wrapper.random_state
    seed_everything(wrapper.random_state)


class DocumentDetector(BaseTextDetector):
    """Use any vector anomaly detector (e.g. from PyOD :cite:`zhao2019pyod`) on document embeddings.

    Parameters
    ----------
    detector : object
        Unfitted detector with ``fit(X)`` and ``decision_function(X)``, for example
        ``pyod.models.knn.KNN()``.
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
    >>> from pyod.models.knn import KNN
    >>> det = DocumentDetector(KNN(), embedder=SentenceEmbedder("bert-base-uncased"))
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
    """Use any vector anomaly detector (e.g. from PyOD :cite:`zhao2019pyod`) on token embeddings.

    The detector is fitted on the tokens of all training documents together and scores
    every token; a document's score aggregates its token scores.

    Parameters
    ----------
    detector : object
        Unfitted detector with ``fit(X)`` and ``decision_function(X)``, for example
        ``pyod.models.knn.KNN()``.
    embedder : TokenEmbedder or None, default=None
        If given, ``fit`` and ``decision_function`` accept raw texts; otherwise ``X`` is
        a list of ``[n_tokens, dim]`` arrays, one per document.
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
    >>> from pyod.models.knn import KNN
    >>> emb = TokenEmbedder("bert-base-uncased", word_pooling="max")
    >>> det = TokenDetector(KNN(), embedder=emb).fit(train_words)
    >>> word_scores = det.token_scores(test_words)
    """
    supports_token = True

    def __init__(self, detector, embedder=None, aggregation="max", k=0.1, contamination=0.1,
                 random_state=None, verbose=False):
        super().__init__(contamination, random_state, device="cpu", verbose=verbose)
        self.detector = detector
        self.embedder = embedder
        self.aggregation = aggregation
        self.k = k

    def _tokens(self, X, with_layout=False):
        """Token arrays, plus (when an embedder in word mode is used) where each row goes."""
        layout = None
        if _is_text(X):
            if self.embedder is None:
                raise ValueError("texts given but no embedder; pass a TokenEmbedder or token arrays")
            T, ids = self.embedder.transform(X)
            if getattr(self.embedder, "word_pooling", None) is not None:
                layout = [(len(doc), w) for doc, w in zip(X, ids)]
        else:
            T = [_as_float(x) for x in X]
        return (T, layout) if with_layout else T

    def _split(self, flat, lengths):
        return np.split(np.asarray(flat, dtype=float), np.cumsum(lengths)[:-1]) if len(lengths) else []

    def fit(self, X, y=None):
        """Fit the detector on training documents ``X`` and return ``self``."""
        T = self._tokens(X)
        flat = np.vstack(T)
        _seed(self)
        self.detector.fit(flat)
        with preserve_rng():
            train_tokens = self._split(self.detector.decision_function(flat), [len(t) for t in T])
        return self._process_decision_scores(aggregate(train_tokens, self.aggregation, self.k))

    def token_scores(self, X):
        """One array of token scores per document. With a word-level ``TokenEmbedder`` and word
        lists as input, there is one score per input word; words without an embedding
        (truncated) get NaN."""
        T, layout = self._tokens(X, with_layout=True)
        scores = self._split(self.detector.decision_function(np.vstack(T)), [len(t) for t in T])
        if layout is None:
            return scores
        out = []
        for s, (n_words, kept) in zip(scores, layout):
            full = np.full(n_words, np.nan)
            full[np.asarray(kept, dtype=int)] = s
            out.append(full)
        return out

    def decision_function(self, X):
        """Anomaly score of each document in ``X`` (higher = more anomalous)."""
        return aggregate(self.token_scores(X), self.aggregation, self.k)
