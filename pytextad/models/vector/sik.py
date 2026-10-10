"""SIK: Simplified Isolation Kernel (Cao et al., Findings of EMNLP 2025)."""
from sklearn.base import BaseEstimator

from ._vendor.sik import SIK as _SIKOriginal
from .base import BaseVectorDetector


class _SIK(_SIKOriginal, BaseEstimator):
    """The original class with scikit-learn's estimator base added: ``check_is_fitted`` warns
    on non-estimators in scikit-learn 1.7 and fails from 1.8 on. No change to the algorithm."""


class SIK(BaseVectorDetector):
    """Simplified Isolation Kernel :cite:`cao2025sik`.

    An ensemble of ``n_estimators`` random partitions, each made of hyperspheres centred at
    ``max_samples`` randomly drawn training points, with a radius equal to the distance to
    the nearest other centre. A vector's anomaly score counts the partitions in which it
    falls outside its nearest hypersphere.

    Parameters
    ----------
    max_samples : int, default=16
        Number of hypersphere centres per partition.
    n_estimators : int, default=200
        Number of partitions.
    novelty : bool, default=True
        True: training data are normal; the score is the number of partitions in which the
        vector is outside. False: training data may contain anomalies; the score is the
        inner product with the mean feature map of the training data.
    sparse : bool, default=False
        Use a sparse feature map (less memory; CPU only).
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=0
        Seed of the partitions.
    device : {"cpu", "cuda", "mps", "auto"}, default="cpu"
        Where distances are computed. On a GPU they are computed in float32 instead of
        float64, so points lying exactly on a sphere boundary may be assigned differently.
    verbose : bool, default=False
        Show the warnings of the original code.

    Notes
    -----
    Code: the authors' implementation (Yang Cao), unchanged; scikit-learn's ``BaseEstimator``
    is added as a base class so that it runs with scikit-learn 1.8 and later.
    """

    def __init__(self, max_samples=16, n_estimators=200, novelty=True, sparse=False,
                 contamination=0.1, random_state=0, device="cpu", verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.novelty = novelty
        self.sparse = sparse

    def _build(self, n_features):
        return _SIK(max_samples=self.max_samples, n_estimators=self.n_estimators, novelty=self.novelty,
                    sparse=self.sparse, device=str(self.device), random_state=self.random_state)
