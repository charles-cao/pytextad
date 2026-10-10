"""ADERH (Durani et al., NeurIPS 2025)."""
from ._vendor.aderh import ADERH as _ADERH
from .base import BaseEmbeddingDetector


class ADERH(BaseEmbeddingDetector):
    """Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres :cite:`durani2025aderh`.

    Each of ``n_estimators`` members samples ``n`` anchor points and pairs each with a partner;
    the pair defines hyperspheres. A vector covered by a sphere scores low if the sphere is
    densely populated and the vector lies near its centre; vectors covered by no sphere keep
    the maximal score. Scores are averaged over the ensemble.

    Parameters
    ----------
    n_estimators : int, default=256
        Number of ensemble members.
    n : int, default=18
        Number of anchors per member.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=0
        Seed of the ensemble.
    verbose : bool, default=False
        Unused.

    Notes
    -----
    Code: the official package (github.com/Walid10010/ADERH, MIT License), unchanged. It gives
    the same scores as the original code used for the paper; its documentation lists where
    that code differs from the description in the paper (kept, so that published results
    are reproduced).
    """

    def __init__(self, n_estimators=256, n=18, contamination=0.1, random_state=0, verbose=False):
        super().__init__(contamination, random_state, device="cpu", verbose=verbose)
        self.n_estimators = n_estimators
        self.n = n

    def _build(self, n_features):
        return _ADERH(n_estimators=self.n_estimators, n=self.n, contamination=self.contamination,
                      random_state=self.random_state)
