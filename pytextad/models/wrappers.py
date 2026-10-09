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
from .base import BaseTextDetector


def _is_text(X):
    return len(X) > 0 and (isinstance(X[0], str) or
                           (isinstance(X[0], (list, tuple)) and len(X[0]) > 0 and isinstance(X[0][0], str)))


class DocumentDetector(BaseTextDetector):
    supports_token = False

    def __init__(self, detector, embedder=None, contamination=0.1, random_state=None, verbose=False):
        """
        detector : unfitted vector detector, e.g. ``pyod.models.knn.KNN()``.
        embedder : optional ``SentenceEmbedder``; then fit/decision_function accept raw texts.
                   Without it, X must be an array [n_documents, dim].
        random_state : if not None, copied onto ``detector.random_state`` before fitting.
        """
        super().__init__(contamination, random_state, device="cpu", verbose=verbose)
        self.detector = detector
        self.embedder = embedder

    def _vectors(self, X):
        if _is_text(X):
            if self.embedder is None:
                raise ValueError("texts given but no embedder; pass a SentenceEmbedder or vectors")
            return self.embedder.transform(X)
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y=None):
        V = self._vectors(X)
        if self.random_state is not None and hasattr(self.detector, "random_state"):
            self.detector.random_state = self.random_state
        self.detector.fit(V)
        return self._process_decision_scores(self.detector.decision_function(V))

    def decision_function(self, X):
        return np.asarray(self.detector.decision_function(self._vectors(X)), dtype=float)


class TokenDetector(BaseTextDetector):
    supports_token = True

    def __init__(self, detector, embedder=None, aggregation="max", k=0.1, contamination=0.1,
                 random_state=None, verbose=False):
        """
        detector    : unfitted vector detector, fitted on all training tokens pooled together.
        embedder    : optional ``TokenEmbedder``; then fit/decision_function accept raw texts.
                      Without it, X must be a list of [n_tokens, dim] arrays, one per document.
        aggregation : "max", "mean" or "topk" (mean of the top ``k`` fraction of tokens);
                      used by ``decision_function``.
        """
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
            T = [np.asarray(x, dtype=np.float32) for x in X]
        return (T, layout) if with_layout else T

    def _split(self, flat, lengths):
        return np.split(np.asarray(flat, dtype=float), np.cumsum(lengths)[:-1]) if len(lengths) else []

    def fit(self, X, y=None):
        T = self._tokens(X)
        flat = np.vstack(T)
        if self.random_state is not None and hasattr(self.detector, "random_state"):
            self.detector.random_state = self.random_state
        self.detector.fit(flat)
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
        return aggregate(self.token_scores(X), self.aggregation, self.k)
