# TCCM, official implementation: https://github.com/ZhongLIFR/TCCM-NIPS
# (commit 363cfa61fd9e786bcb0b905241106dd4e7f52e6f), released under CC BY-SA 4.0
# (https://creativecommons.org/licenses/by-sa/4.0/); this file is distributed under the
# same licence. Contents: SinusoidalTimeEmbedding and FlowMatching from FMAD/functions.py
# and the TCCM class from FMAD/FlowMatchingAD.py, unchanged except for the one line marked
# "PyTextAD", which moves the training data to the model's device (the official code trains
# on the CPU only).
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Sine/cosine encoding of time (Sinusoidal Positional Encoding), used in Transformers
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freq = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freq", freq)

    def forward(self, t):
        # t: [B, 1]
        t = t.view(-1, 1)
        args = t * self.freq  # [B, half_dim]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]


# The Flow Matching model encoding time t using a set of sine/cosine functions at different frequencies
class FlowMatching(nn.Module):
    def __init__(self, input_dim, time_embed_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)  # [B, time_embed_dim]
        x_t = torch.cat([x, t_emb], dim=1)
        return self.model(x_t)


# Core implementation of the Time-Conditioned Contraction Matching (TCCM) algorithm of scikit-learn API style
class TCCM:
    def __init__(self, n_features, epochs=100, learning_rate=0.001, batch_size=64):
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.model = FlowMatching(input_dim=n_features)

    def fit(self, X_train):
        """
        Train the TCCM
        """
        X = torch.tensor(X_train, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)  # PyTextAD: train on the model's device
        y_train = torch.zeros(X.shape[0], dtype=torch.long).squeeze()
        train_loader = DataLoader(TensorDataset(X, y_train), batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            total_loss = 0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                t = torch.rand(batch_x.shape[0], 1, device=batch_x.device)  # Sampling t, line 6 of Algorithm 1.
                f_xt = self.model(batch_x, t)  # Predict contraction vectors f(x, t) # 

                dx_dt = -batch_x
                loss = criterion(f_xt, dx_dt) # Calculate the batch loss, Equation 4.

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
    def decision_function(self, X_test):
        """
        Compute the anomaly scores of X_test
        """
        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)

        with torch.no_grad():
            t = torch.ones(X.shape[0], 1, device=X.device, dtype=torch.float32)  # Set t to 1
            f_xt = self.model(X, t)  # Predict contraction vectors
            anomaly_scores = torch.norm(f_xt + X, dim=1)  # compute the anomaly score, based on Equation 5.

        anomaly_scores = anomaly_scores.cpu().numpy()
        return anomaly_scores
