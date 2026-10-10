"""DAGMM wrapper (the algorithm is in ``_vendor/dagmm.py``)."""
from ._vendor.dagmm import DAGMM as _Original
from .base import BaseVectorDetector


class DAGMM(BaseVectorDetector):
    """Deep Autoencoding Gaussian Mixture Model :cite:`zong2018dagmm`.

    Parameters
    ----------
    num_epochs : int, default=200
        Training epochs.
    lr : float, default=1e-4
        Adam learning rate.
    batch_size : int, default=1024
        Mini-batch size.
    latent_dim : int, default=1
        Size of the autoencoder's code.
    n_gmm : int, default=4
        Number of mixture components.
    lambda_energy, lambda_cov : float, default=0.1, 0.005
        Weights of the energy and covariance terms.
    detach_loss : bool, default=False
        Reproduce published results: the ADBench code (and the DTE and TCCM copies) wraps the
        loss in ``Variable(loss, requires_grad=True)``, which cuts it from the network, so the
        network is never trained. False trains it.
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
    Code: adapted by ADBench, via the DTE and TCCM repositories (CC BY-SA 4.0) and Yang Cao's
    copy. Changes: ``torch.linalg.cholesky`` (``torch.cholesky`` no longer exists) and the
    loss detachment made optional (``detach_loss``).
    """

    def __init__(self, num_epochs=200, lr=0.0001, batch_size=1024, latent_dim=1, n_gmm=4, lambda_energy=0.1, lambda_cov=0.005, detach_loss=False, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_gmm = n_gmm
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.detach_loss = detach_loss

    def _build(self, n_features):
        return _Original(num_epochs=self.num_epochs, lr=self.lr, batch_size=self.batch_size,
                         latent_dim=self.latent_dim, n_gmm=self.n_gmm, lambda_energy=self.lambda_energy,
                         lambda_cov=self.lambda_cov, contamination=self.contamination, device=self.device)

    def _fit_model(self, model, X):
        from ._vendor import dagmm as _module
        _module.ComputeLoss.detach_loss = self.detach_loss
        try:
            model.fit(X)
        finally:
            _module.ComputeLoss.detach_loss = False
