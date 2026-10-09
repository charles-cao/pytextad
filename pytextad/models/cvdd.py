"""CVDD: Context Vector Data Description (Ruff et al., ACL 2019).

Re-implementation for modern PyTorch. The model and training logic follow the
official code, https://github.com/lukasruff/CVDD-PyTorch (MIT License,
Copyright (c) 2019 lukasruff); see THIRD_PARTY_NOTICES.

Input: a list of frozen token-embedding matrices, one [n_tokens, dim] array per
document (GloVe/fastText vectors, or hidden states of a frozen PLM, e.g. from
``pytextad.embeddings.TokenEmbedder``). Only the self-attention layer and the context
vectors are trained, exactly as in the official code.

Faithful to the official code
  * self-attention  A = softmax_over_tokens(W2 tanh(W1 H)), r heads, no biases
  * M = A H; cosine distance d_k = 0.5 (1 - cos(M_k, c_k))
  * context vectors initialised by k-means on L2-normalised mean token embeddings
    of the training set, then L2-normalised
  * loss = mean_n sum_k softmax_k(-alpha d) d_k + lambda_p * mean((C C^T - I)^2)
  * temperature alpha annealed at 5 equidistant milestones
    (soft / linear / logarithmic / hard, same values as the official code)
  * Adam with weight decay, gradient-norm clipping at 0.5, MultiStepLR(gamma=0.1)
    stepped at the start of each epoch as in the official trainer (so with
    PyTorch >= 1.1 the LR drop happens one epoch before the nominal milestone)
  * anomaly score = mean_k d_k ("context_dist_mean", the official default)
  * defaults = the settings in the official README for Reuters / 20 Newsgroups
    (3 heads, attention size 150, lambda_p 1, logarithmic, 100 epochs, lr 0.01,
    lr milestone 40, batch 64, weight decay 5e-7)

Deliberate deviations
  * Padding is masked out of the attention softmax. The official code pads with index
    0 and does not mask. With static word vectors (GloVe, fastText) the pad vector is
    zero, which only rescales M, so the cosine distances, the loss and the training
    are identical (verified numerically with variable-length documents). With PLM
    hidden states the official "bert" option feeds non-zero pad vectors into M, which
    makes a document's score depend on the length of the others in its batch; we do
    not reproduce that.
  * Mini-batches are drawn uniformly at random and every document is scored. The
    official loaders group documents by length (BucketBatchSampler) with
    drop_last=True for both training and testing, so up to batch_size - 1 test
    documents are never scored in the official evaluation.
  * The official "context_best" score is NOT provided: it selects, per run, the
    head with the highest AUROC on the labelled test set, which uses test labels.
    ``head_scores()`` returns per-head distances if you need them.

Extension (not in the CVDD paper)
  * token_scores(): min over heads of the cosine distance between each token vector
    and the context vectors. Comparable across documents. Use with care: it is our
    definition, not part of CVDD.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .base import BaseTextDetector


def _alpha_schedule(name, n_epochs):
    milestones = list(np.arange(1, 6) * int(n_epochs / 5))
    if name == "soft":
        alphas = [0.0] * 5
    elif name == "linear":
        alphas = list(np.linspace(0.2, 1, 5))
    elif name == "logarithmic":
        alphas = list(np.logspace(-4, 0, 5))
    elif name == "hard":
        alphas = [100.0] * 4            # official list has 4 entries
    else:
        raise ValueError(f"unknown alpha_scheduler {name}")
    return milestones, alphas


class _CVDDNet(nn.Module):
    def __init__(self, dim, attention_size, n_heads):
        super().__init__()
        self.W1 = nn.Linear(dim, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_heads, bias=False)
        self.c = nn.Parameter((torch.rand(n_heads, dim) - 0.5) * 2)
        self.alpha = 0.0

    def attend(self, H, mask):
        # H [B, L, d], mask [B, L] (True = real token)
        logits = self.W2(torch.tanh(self.W1(H)))                    # [B, L, r]
        logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        A = F.softmax(logits, dim=1).transpose(1, 2)                # [B, r, L]
        return A @ H, A                                             # M [B, r, d]

    def forward(self, H, mask):
        M, A = self.attend(H, mask)
        d = 0.5 * (1 - F.cosine_similarity(M, self.c.unsqueeze(0), dim=2))   # [B, r]
        w = F.softmax(-self.alpha * d, dim=1)
        return d, w, A


class CVDD(BaseTextDetector):
    """Context Vector Data Description (Ruff et al., ACL 2019).

    Self-attentive one-class model on frozen token embeddings. Each of ``n_heads``
    attention heads summarises a document into one vector, which is compared with a
    learned context vector; the anomaly score is the mean cosine distance over heads.

    Parameters
    ----------
    n_heads : int, default=3
        Number of attention heads and context vectors.
    attention_size : int, default=150
        Hidden size of the self-attention layer.
    lambda_p : float, default=1.0
        Weight of the orthogonality penalty on the attention matrix.
    alpha_scheduler : {"logarithmic", "soft", "linear", "hard"}, default="logarithmic"
        Annealing schedule of the softmax temperature over heads.
    n_epochs : int, default=100
        Training epochs.
    lr : float, default=0.01
        Adam learning rate.
    lr_milestones : tuple of int, default=(40,)
        Epochs at which the learning rate is multiplied by 0.1.
    batch_size : int, default=64
        Mini-batch size.
    weight_decay : float, default=5e-7
        Adam weight decay.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets ``threshold_`` for ``predict``.
    random_state : int or None, default=0
        Seed for all random number generators.
    device : str or None, default=None
        "cuda" or "cpu"; None picks CUDA when available.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    decision_scores_ : numpy.ndarray
        Anomaly scores of the training data (higher = more anomalous).
    threshold_ : float
        Score above which ``predict`` returns 1.
    labels_ : numpy.ndarray
        Binary labels of the training data.

    Notes
    -----
    ``X`` is a list of ``[n_tokens, dim]`` arrays, one per document, for example from
    :class:`~pytextad.utils.embeddings.TokenEmbedder`. ``token_scores`` is an extension
    that is not part of the paper. Differences from the official code are listed in
    :doc:`/faithfulness`.
    """
    supports_token = True


    def __init__(self, n_heads=3, attention_size=150, lambda_p=1.0,
                 alpha_scheduler="logarithmic", n_epochs=100, lr=0.01, lr_milestones=(40,),
                 batch_size=64, weight_decay=0.5e-6, contamination=0.1, random_state=0,
                 device=None, verbose=False):
        super().__init__(contamination, random_state, device, verbose)
        self.n_heads = n_heads
        self.attention_size = attention_size
        self.lambda_p = lambda_p
        self.alpha_scheduler = alpha_scheduler
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_milestones = tuple(lr_milestones)
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    # ------------------------------------------------------------------ batching
    @staticmethod
    def _check(X):
        X = [np.asarray(x, dtype=np.float32) for x in X]
        if any(x.ndim != 2 or len(x) == 0 for x in X):
            raise ValueError("X must be a list of non-empty [n_tokens, dim] arrays")
        if len({x.shape[1] for x in X}) != 1:
            raise ValueError("all token embeddings must have the same dimension")
        return X

    def _batch(self, X, idx):
        L = max(len(X[i]) for i in idx)
        H = np.zeros((len(idx), L, X[idx[0]].shape[1]), dtype=np.float32)
        mask = np.zeros((len(idx), L), dtype=bool)
        for j, i in enumerate(idx):
            H[j, :len(X[i])] = X[i]
            mask[j, :len(X[i])] = True
        return torch.from_numpy(H).to(self.device), torch.from_numpy(mask).to(self.device)

    def _batches(self, n, shuffle):
        order = np.random.permutation(n) if shuffle else np.arange(n)
        for s in range(0, n, self.batch_size):
            yield order[s:s + self.batch_size]

    # ------------------------------------------------------------------ training
    def fit(self, X, y=None):
        """Fit the detector on training documents ``X`` and return ``self``."""
        self._set_seed()
        X = self._check(X)
        dim = X[0].shape[1]
        self.net_ = _CVDDNet(dim, self.attention_size, self.n_heads).to(self.device)

        # context vector initialisation (official initialize_context_vectors)
        means = np.stack([x.mean(0) for x in X])
        means = means / np.clip(np.linalg.norm(means, axis=1, keepdims=True), 1e-8, None)
        km = KMeans(n_clusters=self.n_heads, n_init=10, random_state=self.random_state).fit(means)
        centers = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        self.net_.c.data = torch.from_numpy(centers.astype(np.float32)).to(self.device)

        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(self.lr_milestones), gamma=0.1)
        milestones, alphas = _alpha_schedule(self.alpha_scheduler, self.n_epochs)
        alpha_i = 0
        I = torch.eye(self.n_heads, device=self.device)

        self.net_.alpha = 0.0
        self.history_ = []
        for epoch in range(self.n_epochs):
            with warnings.catch_warnings():  # official order: step at the START of each epoch
                warnings.simplefilter("ignore", UserWarning)
                sched.step()
            if epoch in milestones and alpha_i < len(alphas):     # official: one step per epoch
                self.net_.alpha = float(alphas[alpha_i])
                alpha_i += 1
            self.net_.train()
            tot, nb = 0.0, 0
            for idx in self._batches(len(X), shuffle=True):
                H, mask = self._batch(X, idx)
                d, w, _ = self.net_(H, mask)
                P = torch.mean((self.net_.c @ self.net_.c.t() - I) ** 2)
                loss = torch.mean(torch.sum(w * d, dim=1)) + self.lambda_p * P
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 0.5)   # as in the official trainer
                opt.step()
                tot += loss.item()
                nb += 1
            self.history_.append(tot / nb)
            self._log(f"epoch {epoch + 1}/{self.n_epochs} loss {tot / nb:.6f} alpha {self.net_.alpha:g}")

        return self._process_decision_scores(self.decision_function(X))

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _forward_all(self, X):
        X = self._check(X)
        self.net_.eval()
        D, A_all = [], []
        for idx in self._batches(len(X), shuffle=False):
            H, mask = self._batch(X, idx)
            d, _, A = self.net_(H, mask)
            D.append(d.cpu().numpy())
            A = A.cpu().numpy()
            A_all += [A[j, :, :len(X[i])] for j, i in enumerate(idx)]
        return np.concatenate(D), A_all

    def head_scores(self, X):
        """Per-head cosine distances, shape [n_docs, n_heads]."""
        return self._forward_all(X)[0]

    def decision_function(self, X):
        """Anomaly score of each document in ``X`` (higher = more anomalous)."""
        return self.head_scores(X).mean(1)

    def attention(self, X):
        """Attention weights per document, each [n_heads, n_tokens] (for inspection)."""
        return self._forward_all(X)[1]

    @torch.no_grad()
    def token_scores(self, X):
        """Token scores: cosine distance of each token to its nearest context vector (extension, not in the paper)."""
        X = self._check(X)
        C = F.normalize(self.net_.c.detach(), dim=1).cpu().numpy()
        out = []
        for x in X:
            xn = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
            out.append((0.5 * (1 - xn @ C.T)).min(1))
        return out
