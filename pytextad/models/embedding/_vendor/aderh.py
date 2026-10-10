# ADERH, official implementation: https://github.com/Walid10010/ADERH (aderh/_aderh.py,
# commit 5942c450b2f25d79b0d30e35bcee428b5435750d), MIT License, Copyright (c) 2025-2026
# Walid Durani. Included unchanged below this header.
"""ADERH: Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres.

Reference:
    W. Durani, C. Leiber, K. Durani, C. Plant, C. Boehm.
    "Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres."
    NeurIPS 2025. https://neurips.cc/virtual/2025/poster/115418

This implementation is a cleaned, packaged version of the original
reference code. It is *behavior-preserving*: for the same data and
``random_state`` it produces scores identical to the original
``ADERH.py`` shipped with the paper (verified by the equivalence test in
``tests/test_equivalence.py``). See ``IMPLEMENTATION_NOTES.md`` for
documented semantics of the reference implementation.

Score convention (PyOD-style): **higher score = more anomalous.**
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

__all__ = ["ADERH"]

_MAX_INT = np.iinfo(np.int32).max


class ADERH(BaseEstimator):
    """Ensemble of random pairs of hyperspheres for unsupervised anomaly
    detection.

    For each of the ``n_estimators`` ensemble members, ``n`` points are
    sampled as hypersphere anchors. Each anchor is paired with a partner
    point; half the squared pairwise distance defines the (squared)
    hypersphere radius. A point covered by a sphere receives a low score
    when the sphere is densely populated (NDensity) and the point sits
    close to the sphere's center relative to its radius (Pitch); points
    covered by no sphere keep the maximal score. Scores are averaged
    over the ensemble.

    Parameters
    ----------
    n_estimators : int, default=256
        Number of ensemble members.
    n : int, default=18
        Number of hypersphere anchors sampled per ensemble member.
    contamination : float, default=0.1
        Expected proportion of anomalies; used only to set
        ``threshold_`` and ``labels_`` after ``fit``.
    random_state : int, RandomState instance or None, default=None
        Controls all randomness; fixed values give deterministic output.

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        Anomaly scores of the training data (higher = more anomalous).
    threshold_ : float
        Score threshold implied by ``contamination``.
    labels_ : ndarray of shape (n_samples,)
        Binary labels of the training data (1 = anomaly).

    Examples
    --------
    >>> from aderh import ADERH
    >>> det = ADERH(random_state=0).fit(X_train)
    >>> scores = det.decision_function(X_test)   # higher = more anomalous
    >>> labels = det.predict(X_test)             # 1 = anomaly
    """

    def __init__(self, n_estimators=256, n=18, contamination=0.1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.n = n
        self.contamination = contamination
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, X, y=None):
        """Fit the ensemble and score the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        X = check_array(X, accept_sparse=False)
        n_samples, n_features = X.shape
        if self.n > n_samples:
            raise ValueError(
                f"n (={self.n}) must be <= n_samples (={n_samples}).")

        n_anchors = self.n
        # Duplicated sphere set: anchors + their pair partners.
        self._centers = np.empty(
            (self.n_estimators, 2 * n_anchors, n_features))
        self._sq_radii = np.empty((self.n_estimators, 2 * n_anchors))

        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(_MAX_INT, size=self.n_estimators)

        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])

            anchor_idx = rnd.choice(n_samples, n_anchors, replace=False)
            anchors = X[anchor_idx]

            # Random pairing permutation with fixed points removed
            # (in-place adjacent swap, as in the reference code).
            perm = rnd.choice(n_anchors, n_anchors, replace=False)
            for j in range(n_anchors):
                if perm[j] == j:
                    k = (j + 1) % n_anchors
                    perm[j], perm[k] = perm[k], perm[j]

            # NOTE (reference semantics, preserved): partners are taken
            # as X[perm], i.e. rows 0..n-1 of the dataset, not
            # anchors[perm]. See IMPLEMENTATION_NOTES.md, item 1.
            sq_pair_dist = np.sum((anchors - X[perm]) ** 2, axis=1)

            self._centers[i] = np.concatenate(
                [anchors, anchors[perm]], axis=0)
            # NOTE (reference semantics, preserved): squared radii are
            # half the *squared* pair distance, duplicated in
            # interleaved order [r0, r0, r1, r1, ...]. See
            # IMPLEMENTATION_NOTES.md, items 2 and 3.
            self._sq_radii[i] = np.repeat(
                sq_pair_dist.reshape(-1, 1) / 2, 2, axis=1).reshape(-1)

        self.n_features_in_ = n_features
        self.decision_scores_ = self._raw_scores(X)
        self.threshold_ = np.percentile(
            self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _raw_scores(self, X):
        """Ensemble anomaly scores, higher = more anomalous."""
        per_estimator = np.ones((self.n_estimators, X.shape[0]))

        for i in range(self.n_estimators):
            sq_radii = np.where(
                self._sq_radii[i] != 0, self._sq_radii[i], 1.0)
            sq_dists = euclidean_distances(
                X, self._centers[i], squared=True)

            inside = np.where(sq_dists <= sq_radii, sq_dists, np.nan)
            covered = np.where(~np.isnan(inside).all(axis=1))
            if covered[0].size == 0:
                continue

            nearest = np.nanargmin(inside[covered], axis=1)

            # Pitch: boundary-proximity ratio in [0, 1].
            pitch = inside[covered[0], nearest] / sq_radii[nearest]

            # NDensity: sparsity-aware density weight.
            unique, counts = np.unique(nearest, return_counts=True)
            count_of = dict(zip(unique, counts))
            density = np.array(
                [count_of[j] for j in nearest], dtype=float)
            density /= sq_radii[nearest]
            if density.size == 0:
                continue
            density /= density.max()

            per_estimator[i][covered] = (1.0 - density) * pitch

        return np.mean(per_estimator, axis=0)

    def decision_function(self, X):
        """Anomaly scores for ``X`` (higher = more anomalous)."""
        check_is_fitted(self, "_centers")
        X = check_array(X, accept_sparse=False)
        return self._raw_scores(X)

    def score_samples(self, X):
        """Scores with scikit-learn sign convention (higher = more
        normal); equal to ``-decision_function(X)``."""
        return -self.decision_function(X)

    def predict(self, X):
        """Binary labels for ``X`` (1 = anomaly), using ``threshold_``."""
        check_is_fitted(self, "threshold_")
        return (self.decision_function(X) > self.threshold_).astype(int)

    def fit_predict(self, X, y=None):
        """Fit on ``X`` and return training labels (1 = anomaly)."""
        return self.fit(X).labels_

    # ------------------------------------------------------------------ #
    # Backwards compatibility with the original reference code
    # ------------------------------------------------------------------ #
    @property
    def outlier_score(self):
        """Alias of ``decision_scores_`` (name used by the original
        reference implementation)."""
        return self.decision_scores_

    def predit(self):  # noqa: D401  (typo kept as deprecated alias)
        """Deprecated alias from the reference code; recomputes
        ``threshold_`` and ``labels_`` from ``decision_scores_``."""
        warnings.warn(
            "predit() is deprecated; fit() already sets threshold_ and "
            "labels_, and predict(X) scores new data.",
            DeprecationWarning, stacklevel=2)
        self.threshold_ = np.percentile(
            self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
