"""SLAD wrapper (the algorithm is in ``_vendor/slad.py``)."""
from ._vendor.slad import SLAD as _Original
from .base import BaseVectorDetector


class SLAD(BaseVectorDetector):
    """Scale Learning-based Anomaly Detection :cite:`xu2023slad`.

    Parameters
    ----------
    epochs : int, default=100
        Training epochs.
    batch_size : int, default=128
        Mini-batch size.
    lr : float, default=1e-3
        Adam learning rate.
    hidden_dims : int, default=100
        Hidden size of the network.
    act : str, default="LeakyReLU"
        Activation (a ``torch.nn`` class name).
    distribution_size : int, default=10
        Members per group (c in the paper).
    n_ensemble : int, default=20
        Ensemble size.
    subspace_pool_size : int, default=50
        Number of subspace sizes.
    magnify_factor : float, default=200
        Scale of the supervisory signal.
    n_unified_features : int, default=128
        Dimension after transformation (h in the paper).
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=0
        Seed of the global random generators before the model is built and trained.
    device : str or None, default=None
        "cuda" or "cpu"; None picks CUDA when available.
    verbose : bool, default=False
        Show the original code's training output.

    Notes
    -----
    Code: the copy in the TCCM repository (CC BY-SA 4.0), which adapted the official code (MIT) through the DTE repository; unchanged.
    """

    def __init__(self, epochs=100, batch_size=128, lr=0.001, hidden_dims=100, act="LeakyReLU", distribution_size=10, n_ensemble=20, subspace_pool_size=50, magnify_factor=200, n_unified_features=128, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dims = hidden_dims
        self.act = act
        self.distribution_size = distribution_size
        self.n_ensemble = n_ensemble
        self.subspace_pool_size = subspace_pool_size
        self.magnify_factor = magnify_factor
        self.n_unified_features = n_unified_features

    def _build(self, n_features):
        return _Original(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                         hidden_dims=self.hidden_dims, act=self.act, distribution_size=self.distribution_size,
                         n_ensemble=self.n_ensemble, subspace_pool_size=self.subspace_pool_size,
                         magnify_factor=self.magnify_factor, n_unified_features=self.n_unified_features,
                         device=self.device, random_state=self.random_state)
