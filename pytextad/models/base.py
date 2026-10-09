"""Common interface for all detectors, modelled on PyOD's BaseDetector.

Conventions (identical to PyOD):
  * fit(X, y=None) returns self and sets ``decision_scores_`` (scores of the training data),
    ``threshold_`` and ``labels_``.
  * decision_function(X) returns one score per sample; higher = more anomalous.
  * predict(X) returns 0/1 using ``threshold_`` (the (1 - contamination) quantile of
    the training scores).
Detectors that can score tokens set ``supports_token = True`` and implement
``token_scores``; all detectors give document scores through ``decision_function``.
"""

import abc
import random

import numpy as np
import torch


class BaseTextDetector(abc.ABC):
    """Base class of all PyTextAD detectors, with the PyOD interface.

    ``fit(X)`` sets ``decision_scores_``, ``threshold_`` and ``labels_``;
    ``decision_function(X)`` returns one score per document (higher = more anomalous);
    ``predict(X)`` returns 0/1. Detectors with ``supports_token = True`` also implement
    ``token_scores(X)``.
    """

    #: True if the detector implements ``token_scores`` (token-level detection).
    supports_token = False

    def __init__(self, contamination=0.1, random_state=0, device=None, verbose=False):
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        self.contamination = contamination
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    # ------------------------------------------------------------------ to implement
    @abc.abstractmethod
    def fit(self, X, y=None):
        """Fit the detector on (mostly) normal data and set ``decision_scores_``,
        ``threshold_`` and ``labels_``. ``y`` is ignored except by semi-supervised
        detectors (FATE). Returns ``self``."""

    @abc.abstractmethod
    def decision_function(self, X):
        """Anomaly score of every sample in ``X``; higher means more anomalous."""

    def token_scores(self, X):
        """One array of token anomaly scores per document (token-level detectors only)."""
        raise NotImplementedError(f"{type(self).__name__} is a document-level detector; "
                                  "use TokenDetector to score tokens with a vector detector")

    # ------------------------------------------------------------------ shared
    def _set_seed(self):
        if self.random_state is None:
            return
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _process_decision_scores(self, scores):
        self.decision_scores_ = np.asarray(scores, dtype=float)
        self.threshold_ = float(np.percentile(self.decision_scores_, 100 * (1 - self.contamination)))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        return self

    def predict(self, X):
        """Binary labels (1 = anomaly) using ``threshold_`` from the training scores."""
        self._check_fitted()
        return (self.decision_function(X) > self.threshold_).astype(int)

    def fit_predict(self, X, y=None):
        """Fit on ``X`` and return the labels of the training data (``labels_``)."""
        return self.fit(X, y).labels_

    def _check_fitted(self):
        if not hasattr(self, "decision_scores_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call fit() first.")

    def _log(self, msg):
        if self.verbose:
            print(f"[{type(self).__name__}] {msg}")
