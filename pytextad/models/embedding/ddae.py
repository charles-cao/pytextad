"""DDAE wrapper (the algorithm is in ``_vendor/ddae.py``)."""
import torch

from ._vendor.ddae import DDAE as _Original
from .base import BaseEmbeddingDetector


class DDAE(BaseEmbeddingDetector):
    """Diffusion-Scheduled Denoising Autoencoder :cite:`sattarov2025ddae`.

    A denoiser is trained to recover vectors from noise added by a diffusion schedule; the
    score sums the reconstruction errors over all noise levels.

    Parameters
    ----------
    hidden_dim : sequence of int, default=(64, 64)
        Hidden layer sizes of the denoiser.
    activation : {"lrelu", "relu", "tanh", "sigmoid"}, default="lrelu"
        Activation.
    num_timesteps : int, default=100
        Number of diffusion steps.
    beta_start, beta_end : float, default=1e-4, 0.02
        Noise schedule range.
    scheduler : {"linear", "quadratic", "cosine", "sigmoid", "exponential"}, default="linear"
        Noise schedule.
    time_emb_dim : int, default=4
        Size of the time embedding.
    time_emb_type : {"sinusoidal", "learnable"}, default="sinusoidal"
        Time embedding.
    epochs : int, default=100
        Training epochs.
    batch_size : int, default=64
        Mini-batch size.
    learning_rate : float, default=1e-3
        Adam learning rate.
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
    Code: the official code (github.com/sattarov/AnoDDAE, MIT), without its evaluation during
    training. Scoring is random (it adds noise).
    """

    def __init__(self, hidden_dim=(64, 64), activation="lrelu", num_timesteps=100, beta_start=1e-4,
                 beta_end=0.02, scheduler="linear", time_emb_dim=4, time_emb_type="sinusoidal", epochs=100,
                 batch_size=64, learning_rate=1e-3, contamination=0.1, random_state=0, device=None,
                 verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.scheduler = scheduler
        self.time_emb_dim = time_emb_dim
        self.time_emb_type = time_emb_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _build(self, n_features):
        return _Original(n_features, hidden_dim=list(self.hidden_dim), activation=self.activation,
                         num_timesteps=self.num_timesteps, beta_start=self.beta_start, beta_end=self.beta_end,
                         scheduler=self.scheduler, time_emb_dim=self.time_emb_dim,
                         time_emb_type=self.time_emb_type, epochs=self.epochs, batch_size=self.batch_size,
                         learning_rate=self.learning_rate, device=torch.device(self.device))
