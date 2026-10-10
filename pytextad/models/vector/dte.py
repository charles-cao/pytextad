"""DTE wrappers (the algorithms are in ``_vendor/dte.py``, ``_vendor/dte_nonparametric.py``,
``_vendor/ddpm.py``)."""
from ._vendor import dte as _dte
from .base import BaseVectorDetector

_COMMON = """
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=0
        Seed of the global random generators before the model is built and trained.
    device : str or None, default=None
        "cuda" or "cpu"; None picks CUDA when available.
    verbose : bool, default=False
        Show the original code's training output.
"""


class _DTEBase(BaseVectorDetector):
    _original = None

    def __init__(self, hidden_size=(256, 512, 256), epochs=400, batch_size=64, lr=1e-4,
                 weight_decay=5e-4, T=400, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.T = T

    def _kwargs(self):
        import torch
        return dict(hidden_size=list(self.hidden_size), epochs=self.epochs, batch_size=self.batch_size,
                    lr=self.lr, weight_decay=self.weight_decay, T=self.T, device=torch.device(self.device))

    def _build(self, n_features):
        return type(self)._original(**self._kwargs())

    def _fit_model(self, model, X):
        import numpy as np
        model.fit(np.asarray(X, dtype=np.float32))

    def decision_function(self, X):
        """Anomaly score of each row of ``X`` (higher = more anomalous)."""
        import numpy as np
        return super().decision_function(np.asarray(X, dtype=np.float32))


class DTECategorical(_DTEBase):
    """Diffusion Time Estimation, classification variant :cite:`livernoche2024dte`.

    A network predicts, as a class among ``num_bins`` bins, how much Gaussian noise was added
    to a vector; the anomaly score is the expected predicted noise level of the clean vector.

    Parameters
    ----------
    hidden_size : sequence of int, default=(256, 512, 256)
        Hidden layer sizes.
    epochs : int, default=400
        Training epochs.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=1e-4
        Adam learning rate.
    weight_decay : float, default=5e-4
        Adam weight decay.
    T : int, default=400
        Number of diffusion steps.
    num_bins : int, default=7
        Number of classes the diffusion time is binned into.""" + _COMMON + """
    Notes
    -----
    Code: the official DTE code (github.com/vicliv/DTE, MIT), unchanged.
    """
    _original = _dte.DTECategorical

    def __init__(self, hidden_size=(256, 512, 256), epochs=400, batch_size=64, lr=1e-4,
                 weight_decay=5e-4, T=400, num_bins=7, contamination=0.1, random_state=0, device=None,
                 verbose=False):
        super().__init__(hidden_size, epochs, batch_size, lr, weight_decay, T, contamination,
                         random_state, device, verbose)
        self.num_bins = num_bins

    def _kwargs(self):
        return dict(super()._kwargs(), num_bins=self.num_bins)


class DTEInverseGamma(_DTEBase):
    """Diffusion Time Estimation, inverse-gamma variant :cite:`livernoche2024dte`.

    Parameters
    ----------
    hidden_size : sequence of int, default=(256, 512, 256)
        Hidden layer sizes.
    epochs : int, default=400
        Training epochs.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=1e-4
        Adam learning rate.
    weight_decay : float, default=5e-4
        Adam weight decay.
    T : int, default=400
        Number of diffusion steps.""" + _COMMON + """
    Notes
    -----
    Code: the official DTE code (github.com/vicliv/DTE, MIT), unchanged.
    """
    _original = _dte.DTEInverseGamma


class DTEGaussian(_DTEBase):
    """Diffusion Time Estimation, Gaussian (regression) variant :cite:`livernoche2024dte`.

    Parameters
    ----------
    hidden_size : sequence of int, default=(256, 512, 256)
        Hidden layer sizes.
    epochs : int, default=400
        Training epochs.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=1e-4
        Adam learning rate.
    weight_decay : float, default=5e-4
        Adam weight decay.
    T : int, default=400
        Number of diffusion steps.""" + _COMMON + """
    Notes
    -----
    Code: the official DTE code (github.com/vicliv/DTE, MIT), unchanged.
    """
    _original = _dte.DTEGaussian


class DTENonParametric(BaseVectorDetector):
    """Diffusion Time Estimation, non-parametric variant :cite:`livernoche2024dte`.

    The posterior over diffusion times is estimated from the mean distance of a vector to its
    ``K`` nearest training vectors; the score is its most likely diffusion time.

    Parameters
    ----------
    K : int, default=5
        Number of nearest neighbours.
    T : int, default=1000
        Number of diffusion steps.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=0
        Unused (the method is deterministic); accepted for a uniform interface.
    verbose : bool, default=False
        Unused.

    Notes
    -----
    Code: the official DTE code (github.com/vicliv/DTE, MIT), unchanged. It imports
    matplotlib (for plotting functions that PyTextAD does not use), so matplotlib must be
    installed. Scoring queries the neighbours of one vector at a time and is slow for many
    vectors.
    """

    def __init__(self, K=5, T=1000, contamination=0.1, random_state=0, verbose=False):
        super().__init__(contamination, random_state, device="cpu", verbose=verbose)
        self.K = K
        self.T = T

    def _build(self, n_features):
        try:
            from ._vendor.dte_nonparametric import DTENonParametric as _Original
        except ImportError as e:
            raise ImportError("DTENonParametric needs matplotlib: pip install matplotlib") from e
        return _Original(K=self.K, T=self.T)


class DDPM(BaseVectorDetector):
    """Denoising diffusion model used as a reconstruction-based detector :cite:`livernoche2024dte`.

    Vectors are passed through ``reconstruction_t`` reverse diffusion steps; the score is the
    reconstruction error.

    Parameters
    ----------
    epochs : int, default=400
        Maximum training epochs (training stops early when the loss rises).
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=1e-4
        Adam learning rate.
    weight_decay : float, default=5e-4
        Adam weight decay.
    T : int, default=1000
        Number of diffusion steps.
    reconstruction_t : int, default=250
        Number of reverse steps used for scoring.
    full_path : bool, default=False
        Sum the errors along the whole reverse path instead of using the final one.""" + _COMMON + """
    Notes
    -----
    Code: the DDPM baseline of the official DTE code (github.com/vicliv/DTE, MIT), unchanged.
    Scoring is random (it samples the reverse process).
    """

    def __init__(self, epochs=400, batch_size=64, lr=1e-4, weight_decay=5e-4, T=1000,
                 reconstruction_t=250, full_path=False, contamination=0.1, random_state=0, device=None,
                 verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.T = T
        self.reconstruction_t = reconstruction_t
        self.full_path = full_path

    def _build(self, n_features):
        import torch
        from ._vendor.ddpm import DDPM as _Original
        return _Original(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                         weight_decay=self.weight_decay, T=self.T, reconstruction_t=self.reconstruction_t,
                         full_path=self.full_path, device=torch.device(self.device))

    def _fit_model(self, model, X):
        import numpy as np
        model.fit(np.asarray(X, dtype=np.float32))

    def decision_function(self, X):
        """Anomaly score of each row of ``X`` (higher = more anomalous)."""
        import numpy as np
        return super().decision_function(np.asarray(X, dtype=np.float32))
