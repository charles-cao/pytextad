"""Common shell for anomaly detector for embeddingss taken from their authors' code."""
import abc

import numpy as np

from ...utils.rng import preserve_rng, seed_everything, silence
from ..base import BaseTextDetector


class BaseEmbeddingDetector(BaseTextDetector):
    """Base class of the embedding detectors in :mod:`pytextad.models.embedding`.

    The algorithm itself is the authors' code, kept in ``pytextad/models/embedding/_vendor``;
    this shell adds the common detector interface and the conventions of PyTextAD:

    * the global random generators (Python, NumPy, PyTorch) are seeded with
      ``random_state`` before the model is built and fitted, as in the benchmark scripts the
      published results come from;
    * the input dimension is read from the data in ``fit``;
    * ``decision_scores_`` are computed without changing the random generators, so that
      scoring test data afterwards gives the same numbers as a plain
      ``fit(X_train); decision_function(X_test)`` script;
    * printing of the original code is suppressed unless ``verbose=True``.

    These detectors work on embeddings (or any numeric vectors). For text, wrap them in
    :class:`~pytextad.models.wrappers.DocumentDetector` (one vector per document) or
    :class:`~pytextad.models.wrappers.TokenDetector` (one vector per token).
    """

    supports_token = False

    @abc.abstractmethod
    def _build(self, n_features):
        """Return the unfitted original model for data with ``n_features`` columns."""

    def _fit_model(self, model, X):
        model.fit(X)

    def fit(self, X, y=None):
        """Fit on ``X`` (``[n_samples, n_features]``); ``y`` is ignored. Returns ``self``."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array [n_samples, n_features], got shape {X.shape}")
        if self.random_state is not None:
            seed_everything(self.random_state)
        with silence(not self.verbose):
            self.model_ = self._build(X.shape[1])
            self._fit_model(self.model_, X)
        with preserve_rng():
            return self._process_decision_scores(self.decision_function(X))

    def decision_function(self, X):
        """Anomaly score of each row of ``X`` (higher = more anomalous)."""
        if not hasattr(self, "model_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call fit() first.")
        with silence(not self.verbose):
            scores = self.model_.decision_function(np.asarray(X))
        return np.asarray(scores, dtype=float).reshape(-1)
