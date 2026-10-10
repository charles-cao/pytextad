"""GOAD wrapper (the algorithm is in ``_vendor/goad.py``)."""
from ._vendor.goad import GOAD as _Original
from .base import BaseVectorDetector


class GOAD(BaseVectorDetector):
    """Classification-based anomaly detection with random affine transformations :cite:`bergman2020goad`.

    Parameters
    ----------
    d_out : int, default=32
        Dimension of each random projection.
    m : float, default=1
        Margin of the triplet-centre loss.
    n_rots : int, default=256
        Number of random transformations.
    n_epoch : int, default=1
        Training epochs.
    ndf : int, default=8
        Width of the network.
    batch_size : int, default=64
        Mini-batch size.
    lmbda : float, default=0.1
        Weight of the triplet-centre loss.
    eps : float, default=0
        Lower bound of the squared distances at test time.
    lr : float, default=0.001
        Adam learning rate.
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
    Code: the copy in the TCCM repository (CC BY-SA 4.0), which adapted the official code through the DTE repository; unchanged.

    **Research use only**: the official GOAD code is under the Yissum Software Research
    Licence, which forbids commercial use; this derived code carries that restriction.
    Every vector is projected by all ``n_rots`` transformations, so memory and time grow with
    ``n_rots`` (about 5 minutes for 16,000 token vectors on a CPU).
    """

    def __init__(self, d_out=32, m=1, n_rots=256, n_epoch=1, ndf=8, batch_size=64, lmbda=0.1, eps=0, lr=0.001, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.d_out = d_out
        self.m = m
        self.n_rots = n_rots
        self.n_epoch = n_epoch
        self.ndf = ndf
        self.batch_size = batch_size
        self.lmbda = lmbda
        self.eps = eps
        self.lr = lr

    def _build(self, n_features):
        return _Original(d_out=self.d_out, m=self.m, n_rots=self.n_rots, n_epoch=self.n_epoch, ndf=self.ndf,
                         batch_size=self.batch_size, lmbda=self.lmbda, eps=self.eps, lr=self.lr,
                         device=self.device)
