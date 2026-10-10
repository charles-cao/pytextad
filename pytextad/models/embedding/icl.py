"""ICL wrapper (the algorithm is in ``_vendor/icl.py``)."""
from .base import BaseEmbeddingDetector


class ICL(BaseEmbeddingDetector):
    """Internal Contrastive Learning :cite:`shenkar2022icl`.

    Parameters
    ----------
    num_epochs : int, default=2000
        Maximum number of epochs (training stops early when the loss is small).
    no_batchs : int, default=3000
        Mini-batch size (name kept from the original code).
    no_negatives : int, default=1000
        Number of negative pairs.
    temperature : float, default=0.01
        Softmax temperature.
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
    Memory grows with ``no_batchs`` times the square of the input dimension; on 768-dimensional
    token embeddings lower ``no_batchs`` (e.g. 256). The original code imports pandas, so
    pandas must be installed.
    """

    def __init__(self, num_epochs=2000, no_batchs=3000, no_negatives=1000, temperature=0.01, lr=0.001, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.num_epochs = num_epochs
        self.no_batchs = no_batchs
        self.no_negatives = no_negatives
        self.temperature = temperature
        self.lr = lr

    def _build(self, n_features):
        try:
            from ._vendor.icl import ICL as _Original
        except ImportError as e:
            raise ImportError("ICL needs pandas: pip install pandas") from e
        return _Original(num_epochs=self.num_epochs, no_batchs=self.no_batchs, no_negatives=self.no_negatives,
                         temperature=self.temperature, lr=self.lr, device=self.device)
