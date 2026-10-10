"""MCM wrapper (the algorithm is in ``_vendor/mcm.py``)."""
from ._vendor.mcm import MCM as _Original
from .base import BaseEmbeddingDetector


class MCM(BaseEmbeddingDetector):
    """Masked Cell Modeling :cite:`yin2024mcm`.

    Parameters
    ----------
    batch_size : int, default=256
        Mini-batch size.
    epochs : int, default=200
        Training epochs.
    learning_rate : float, default=0.05
        Adam learning rate.
    test_batch_size : int, default=32
        Batch size when scoring.
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
    Code: the copy in the TCCM repository (CC BY-SA 4.0), which adapted the official code; unchanged.
    """

    def __init__(self, batch_size=256, epochs=200, learning_rate=0.05, test_batch_size=32, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.test_batch_size = test_batch_size

    def _build(self, n_features):
        return _Original(n_features, batch_size=self.batch_size, epochs=self.epochs,
                         learning_rate=self.learning_rate, test_batch_size=self.test_batch_size,
                         device=self.device)
