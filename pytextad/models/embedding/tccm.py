"""TCCM (Li et al., NeurIPS 2025)."""
from ._vendor.tccm import TCCM as _TCCM
from .base import BaseEmbeddingDetector


class TCCM(BaseEmbeddingDetector):
    """Time-Conditioned Contraction Matching :cite:`li2025tccm`.

    A network ``f(x, t)`` is trained to predict the contraction ``-x`` at random times ``t``
    (one-step flow matching towards the origin); the anomaly score of ``x`` is
    ``||f(x, 1) + x||``.

    Parameters
    ----------
    epochs : int, default=100
        Training epochs.
    learning_rate : float, default=0.001
        Adam learning rate.
    batch_size : int, default=64
        Mini-batch size.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=0
        Seed of the global random generators before the network is built and trained.
    device : str or None, default=None
        "cuda" or "cpu"; None picks CUDA when available.
    verbose : bool, default=False
        Unused (the official code prints nothing).

    Notes
    -----
    Code: the official implementation (github.com/ZhongLIFR/TCCM-NIPS, CC BY-SA 4.0), unchanged
    except that training data are moved to the model's device. The paper tunes epochs, batch
    size and learning rate per dataset; the defaults here are those of the class in the
    official code.
    """

    def __init__(self, epochs=100, learning_rate=0.001, batch_size=64, contamination=0.1,
                 random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def _build(self, n_features):
        model = _TCCM(n_features, epochs=self.epochs, learning_rate=self.learning_rate,
                      batch_size=self.batch_size)
        model.model.to(self.device)
        return model
