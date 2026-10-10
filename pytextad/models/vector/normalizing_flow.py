"""NormalizingFlow wrapper (the algorithm is in ``_vendor/normalizing_flow.py``)."""
from ._vendor.normalizing_flow import NormalizingFlow as _Original
import numpy as np

from .base import BaseVectorDetector


class NormalizingFlow(BaseVectorDetector):
    """Planar normalizing flow; the score is the negative log-likelihood :cite:`rezende2015planar`.

    Parameters
    ----------
    K : int, default=10
        Number of planar flow layers.
    epochs : int, default=200
        Training epochs.
    batch_size : int, default=64
        Mini-batch size.
    lr : float, default=2e-3
        Adam learning rate.
    legacy_score : bool, default=False
        Reproduce the published scores. The original ``decision_function`` adds a ``[n]``
        array of base log-densities to an ``[n, 1]`` array of log-determinants, which
        broadcasts to ``[n, n]``, and averages each row; every vector then gets the mean base
        log-density of the whole batch plus its own log-determinant, so its own density
        barely matters and its score depends on the other vectors scored with it. False
        returns the per-vector negative log-likelihood ``-(log N(z) + log|det J|)``.
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
    Code: the baseline of the DTE repository (MIT), via the TCCM repository (CC BY-SA 4.0);
    training unchanged (the broadcasting does not change the training loss). Used as a
    baseline in :cite:`livernoche2024dte`.
    """

    def __init__(self, K=10, epochs=200, batch_size=64, lr=0.002, legacy_score=False, contamination=0.1, random_state=0, device=None, verbose=False):
        super().__init__(contamination, random_state, device=device, verbose=verbose)
        self.K = K
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.legacy_score = legacy_score

    def _build(self, n_features):
        return _Original(K=self.K, device=self.device)

    def _fit_model(self, model, X):
        model.fit(np.asarray(X, dtype=np.float32), epochs=self.epochs, batch_size=self.batch_size, lr=self.lr)

    def decision_function(self, X):
        """Anomaly score of each row of ``X`` (higher = more anomalous)."""
        if self.legacy_score:
            return super().decision_function(np.asarray(X, dtype=np.float32))
        import torch
        flow = self.model_.flow
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 4096):
                x = torch.tensor(np.asarray(X[s:s + 4096], dtype=np.float32), device=self.model_.device)
                z, log_det = flow(x)
                log_p = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=1) + log_det.reshape(-1)
                out.append((-log_p).cpu().numpy())
        return np.concatenate(out).astype(float) if out else np.zeros(0)
