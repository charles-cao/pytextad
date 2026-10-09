"""RSRAE: Robust Subspace Recovery AutoEncoder (Lai, Zou & Lerman, ICLR 2020).

PyTorch port of the official TensorFlow code, https://github.com/dmzou/RSRAE
(MIT License, Copyright (c) 2019-present Chieh-Hsin Lai, Dongmian Zou and
Gilad Lerman); see THIRD_PARTY_NOTICES. Input is a vector per document
(sentence embeddings, TF-IDF, ...); RSRAE itself is not text-specific.

Faithful to the official code (RSRAE/model.py + experiments.py defaults)
  * encoder  d -> 32 -> 64 -> 128 (dense + activation + BN), RSR layer y A with
    A ~ N(0, 1) of shape [128, intrinsic_size], L2 re-normalisation of z,
    decoder 128 -> 64 -> 32 -> d (activation also on the output layer)
  * activation chosen from the training data exactly as experiments.py does:
    relu if min(X) >= 0, else tanh if |max(X)| <= 1, else leaky_relu(0.2)
  * losses with the L2,1 norm (mean over samples of the non-squared L2 norm):
      reconstruction ||x - x~||, PCA ||y - A A^T y||, projection mean((A^T A - I)^2)
  * all_alt=True, enforce_proj=True (defaults): every mini-batch runs three Adam steps
    with three separate optimisers: reconstruction on all weights (lr), projection on
    A (10 lr), PCA error on encoder + A (10 lr)
  * lr 2.5e-4, 200 epochs, batch 128, intrinsic size 10, Glorot-uniform weights,
    zero biases, one shuffle before training (the official code does not reshuffle)
  * anomaly score = -cos(x, x~)
  * bn_mode="official": the official code calls Keras BatchNormalization in a TF1
    graph without a training flag, so it always runs in inference mode with moving
    statistics that are never updated (mean 0, variance 1). We verified this by
    running the official code. The layer is therefore a trainable affine map
    gamma * x / sqrt(1 + 1e-3) + beta, which is what "official" reproduces.
    bn_mode="batch" uses real batch normalisation; bn_mode=None disables it.

Note on protocol: the official experiments fit RSRAE on the (contaminated) test set
and score the same data. Here fit() and decision_function() are separate, so use
fit(X).decision_scores_ for that transductive protocol.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTextDetector


class _FrozenStatBN(nn.Module):
    """Keras BatchNormalization in inference mode with untouched moving statistics."""

    def __init__(self, n, eps=1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n))
        self.beta = nn.Parameter(torch.zeros(n))
        self.scale = 1.0 / np.sqrt(1.0 + eps)

    def forward(self, x):
        return self.gamma * x * self.scale + self.beta


def _bn(n, mode):
    if mode is None:
        return nn.Identity()
    if mode == "official":
        return _FrozenStatBN(n)
    if mode == "batch":
        return nn.BatchNorm1d(n, eps=1e-3, momentum=0.01)   # Keras momentum 0.99
    raise ValueError(f"unknown bn_mode {mode}")


def _dense(i, o):
    layer = nn.Linear(i, o)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _RSRAENet(nn.Module):
    def __init__(self, d, hidden, intrinsic, act, normalize, bn_mode):
        super().__init__()
        h0, h1, h2 = hidden
        self.act = act
        self.normalize = normalize
        self.enc = nn.ModuleList([_dense(d, h0), _dense(h0, h1), _dense(h1, h2)])
        self.enc_bn = nn.ModuleList([_bn(h0, bn_mode), _bn(h1, bn_mode), _bn(h2, bn_mode)])
        self.A = nn.Parameter(torch.randn(h2, intrinsic))
        self.dec = nn.ModuleList([_dense(intrinsic, h2), _dense(h2, h1), _dense(h1, h0)])
        self.dec_bn = nn.ModuleList([_bn(h2, bn_mode), _bn(h1, bn_mode), _bn(h0, bn_mode)])
        self.out = _dense(h0, d)

    def encode(self, x):
        for lin, bn in zip(self.enc, self.enc_bn):
            x = bn(self.act(lin(x)))
        return x

    def forward(self, x):
        y = self.encode(x)
        y_rsr = y @ self.A
        z = F.normalize(y_rsr, p=2, dim=-1, eps=1e-12) if self.normalize else y_rsr
        h = z
        for lin, bn in zip(self.dec, self.dec_bn):
            h = bn(self.act(lin(h)))
        return y, y_rsr, z, self.act(self.out(h))

    def encoder_params(self):
        return list(self.enc.parameters()) + list(self.enc_bn.parameters())


def _norm_loss(diff, kind):
    kind = kind.lower()
    if kind in ("mse", "frob", "f"):
        return diff.norm(dim=1).pow(2).mean()
    if kind == "l1":
        return diff.abs().sum(1).mean()
    if kind in ("l21", "lad", "l2"):
        return diff.norm(dim=1).mean()
    raise ValueError(f"unknown norm {kind}")


class RSRAE(BaseTextDetector):
    supports_token = False


    def __init__(self, hidden_layer_sizes=(32, 64, 128), intrinsic_size=10,
                 loss_norm_type="L21", norm_type="L21", all_alt=True, enforce_proj=True,
                 lambda1=0.0025, lambda2=0.1, lr=2.5e-4, n_epochs=200, batch_size=128,
                 normalize=True, bn_mode="official", activation="auto",
                 contamination=0.1, random_state=0, device=None, verbose=False):
        """
        activation : "auto" (official rule), or "relu" / "tanh" / "leaky_relu".
        lambda1, lambda2 are only used when all_alt=False (joint loss), as in the
        official code.
        """
        super().__init__(contamination, random_state, device, verbose)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.intrinsic_size = intrinsic_size
        self.loss_norm_type = loss_norm_type
        self.norm_type = norm_type
        self.all_alt = all_alt
        self.enforce_proj = enforce_proj
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.normalize = normalize
        self.bn_mode = bn_mode
        self.activation = activation

    def _pick_activation(self, X):
        name = self.activation
        if name == "auto":
            if X.min() >= 0:
                name = "relu"
            elif abs(X.max()) <= 1:          # official rule checks max only
                name = "tanh"
            else:
                name = "leaky_relu"
        self.activation_ = name
        return {"relu": F.relu, "tanh": torch.tanh,
                "leaky_relu": lambda t: F.leaky_relu(t, 0.2)}[name]

    def _pca_error(self, y, y_rsr):
        return _norm_loss(y - y_rsr @ self.net_.A.t(), self.norm_type)

    def _proj_error(self):
        A = self.net_.A
        return torch.mean((A.t() @ A - torch.eye(A.shape[1], device=A.device)) ** 2)

    def fit(self, X, y=None):
        self._set_seed()
        X = np.asarray(X, dtype=np.float32)
        act = self._pick_activation(X)
        self.net_ = _RSRAENet(X.shape[1], self.hidden_layer_sizes, self.intrinsic_size, act,
                              self.normalize, self.bn_mode).to(self.device)
        Xt = torch.from_numpy(X).to(self.device)
        idx = np.random.permutation(len(X))           # official: shuffled once
        self._train_loop(Xt, idx)
        return self._process_decision_scores(self.decision_function(X))

    def _train_loop(self, Xt, idx):
        net = self.net_
        opt_main = torch.optim.Adam(net.parameters(), lr=self.lr, eps=1e-8)
        opt_proj = torch.optim.Adam([net.A], lr=10 * self.lr, eps=1e-8)
        opt_pca = torch.optim.Adam(net.encoder_params() + [net.A], lr=10 * self.lr, eps=1e-8)

        n_batch = (len(Xt) - 1) // self.batch_size + 1
        net.train()
        for epoch in range(self.n_epochs):
            for b in range(n_batch):
                xb = Xt[idx[b * self.batch_size:(b + 1) * self.batch_size]]
                yv, y_rsr, _, x_rec = net(xb)
                loss = _norm_loss(xb - x_rec, self.loss_norm_type)
                if not self.all_alt:
                    loss = loss + self.lambda1 * self._pca_error(yv, y_rsr) + self.lambda2 * self._proj_error()
                opt_main.zero_grad()
                loss.backward()
                opt_main.step()
                if self.all_alt and self.enforce_proj:
                    opt_proj.zero_grad()
                    self._proj_error().backward()
                    opt_proj.step()
                if self.all_alt:
                    yv, y_rsr, _, _ = net(xb)
                    opt_pca.zero_grad()
                    self._pca_error(yv, y_rsr).backward()
                    opt_pca.step()
            if self.verbose and (epoch + 1) % 20 == 0:
                self._log(f"epoch {epoch + 1}/{self.n_epochs} last-batch loss {loss.item():.4f}")

    @torch.no_grad()
    def reconstruct(self, X):
        self.net_.eval()
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        return torch.cat([self.net_(Xt[s:s + 1024])[3] for s in range(0, len(Xt), 1024)]).cpu().numpy()

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        R = self.reconstruct(X)
        cos = (X * R).sum(1) / (np.linalg.norm(R, axis=1) + 1e-6) / (np.linalg.norm(X, axis=1) + 1e-6)
        return -cos
