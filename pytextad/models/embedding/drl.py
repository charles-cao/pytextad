"""DRL wrapper (the algorithm is in ``_vendor/drl.py``)."""
from ._vendor.drl import DRL as _Original
from .base import BaseEmbeddingDetector


class DRL(BaseEmbeddingDetector):
    """Decomposed Representation Learning :cite:`ye2025drl`.

    Representations are decomposed onto learnable orthogonal prototypes; the score is the
    distance between a vector's representation and its decomposition.

    Parameters
    ----------
    hidden_dim : int, default=128
        Representation size.
    prototype_num : int, default=10
        Number of prototypes.
    en_nlayers : int, default=3
        Encoder depth.
    de_nlayers : int, default=2
        Decoder depth (the decoder is built but not used for scoring).
    diversity : bool, default=True
        Initialise the prototypes orthogonally.
    plearn : bool, default=True
        Learn the prototypes.
    epochs : int, default=100
        Training epochs.
    lr : float, default=1e-3
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
        Unused.

    Notes
    -----
    Code: a scikit-learn style rewrite by Yang Cao of the official code
    (github.com/HangtingYe/DRL); the only change is that ``device`` is honoured.
    """

    def __init__(self, hidden_dim=128, prototype_num=10, en_nlayers=3, de_nlayers=2, diversity=True,
                 plearn=True, epochs=100, lr=1e-3, batch_size=256, contamination=0.1, random_state=0,
                 device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.hidden_dim = hidden_dim
        self.prototype_num = prototype_num
        self.en_nlayers = en_nlayers
        self.de_nlayers = de_nlayers
        self.diversity = diversity
        self.plearn = plearn
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def _build(self, n_features):
        return _Original(n_features, hidden_dim=self.hidden_dim, prototype_num=self.prototype_num,
                         en_nlayers=self.en_nlayers, de_nlayers=self.de_nlayers, diversity=self.diversity,
                         plearn=self.plearn, epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
                         device=str(self.device))
