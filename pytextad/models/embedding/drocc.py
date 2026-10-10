"""DROCC wrapper (the algorithm is in ``_vendor/drocc.py``)."""
from ._vendor.drocc import DROCC as _Original
import numpy as np

from .base import BaseEmbeddingDetector


class DROCC(BaseEmbeddingDetector):
    """Distributionally Robust One-Class Classifier :cite:`goyal2020drocc`.

    Parameters
    ----------
    lamda : float, default=1
        Weight of the adversarial loss.
    radius : float, default=3
        Radius of the region of normal data.
    gamma : float, default=2
        Adversarial points are projected between ``radius`` and ``gamma * radius``.
    lr : float, default=1e-4
        Adam learning rate.
    batch_size : int, default=256
        Mini-batch size.
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
    Trains for 200 epochs (fixed in the code), each with 50 gradient-ascent steps per batch to
    generate adversarial points: slow on many vectors (about 12 minutes for 16,000 token
    vectors on a CPU).
    """

    def __init__(self, lamda=1, radius=3, gamma=2, lr=0.0001, batch_size=256, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size

    def _build(self, n_features):
        return _Original(lamda=self.lamda, radius=self.radius, gamma=self.gamma, lr=self.lr,
                         batch_size=self.batch_size, device=self.device)

    def _fit_model(self, model, X):
        model.fit(np.ascontiguousarray(X))
