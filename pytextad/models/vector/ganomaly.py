"""GANomaly wrapper (the algorithm is in ``_vendor/ganomaly.py``)."""
from ._vendor.ganomaly import GANomaly as _Original
from .base import BaseVectorDetector


class GANomaly(BaseVectorDetector):
    """GANomaly: adversarially trained encoder-decoder-encoder :cite:`akcay2018ganomaly`.

    Parameters
    ----------
    epochs : int, default=50
        Training epochs.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=0.01
        SGD learning rate.
    mom : float, default=0.7
        SGD momentum.
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
    Code: the copy in the TCCM repository (CC BY-SA 4.0), which adapted ADBench's; unchanged.
    """

    def __init__(self, epochs=50, batch_size=64, lr=0.01, mom=0.7, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mom = mom

    def _build(self, n_features):
        return _Original(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, mom=self.mom,
                         device=self.device)
