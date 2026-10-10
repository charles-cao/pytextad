import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

def random_orthogonal_vectors(num_vectors, vector_dim):
    # ensure linear independent
    while True:
        random_matrix = np.random.randn(num_vectors, vector_dim)
        if np.linalg.matrix_rank(random_matrix) == num_vectors:
            break
    
    # initialize
    orthogonal_vectors = np.zeros((num_vectors, vector_dim))
    
    # Gram-Schmidt process
    for i in range(num_vectors):
        v = random_matrix[i]
        
        for j in range(i):
            v -= np.dot(v, orthogonal_vectors[j]) * orthogonal_vectors[j]
        
        # normalize current vector
        orthogonal_vectors[i] = v / np.linalg.norm(v)
    
    return orthogonal_vectors

class PN(nn.Module):
    def __init__(self, model_config):
        super(PN, self).__init__()
        self.data_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.en_nlayers = model_config['en_nlayers']
        self.de_nlayers = model_config['de_nlayers']
        self.model_config = model_config
        
        if model_config['diversity'] == False:
            if model_config['plearn'] == False:
                self.prototype = nn.Parameter(torch.rand(model_config['prototype_num'], self.hidden_dim), requires_grad=False)
            else:
                self.prototype = nn.Parameter(torch.rand(model_config['prototype_num'], self.hidden_dim), requires_grad=True)
        else:
            if model_config['plearn'] == False:
                self.prototype = nn.Parameter(torch.tensor(random_orthogonal_vectors(model_config['prototype_num'], self.hidden_dim)).float(), requires_grad=False)
            else:
                self.prototype = nn.Parameter(torch.tensor(random_orthogonal_vectors(model_config['prototype_num'], self.hidden_dim)).float(), requires_grad=True)
        phi = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-2):
            phi.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            phi.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        phi.append(nn.Linear(encoder_dim,model_config['prototype_num'],bias=False))
        self.phi = nn.Sequential(*phi)
        encoder = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-1):
            encoder.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        self.encoder = nn.Sequential(*encoder)
        decoder = []
        decoder.append(nn.Linear(self.hidden_dim,self.data_dim,bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x_input):
        h = self.encoder(x_input)
        weight = self.phi(x_input)
        h_ = weight@self.prototype
        mse = F.mse_loss(h, h_, reduction='none')
        l2_norm_square = mse.sum(dim=1,keepdim=True)
        l2_norm_square_normalize = mse.sum(dim=1,keepdim=True) / torch.sum(h**2,dim=1,keepdim=True)
        l2_norm_square = torch.cat([l2_norm_square,l2_norm_square_normalize],dim=1)
        l2_norm_square,_ = torch.max(l2_norm_square, dim=1, keepdim=True)
        return l2_norm_square

    def predict_score(self, x_input):
        h = self.encoder(x_input)
        weight = self.phi(x_input)
        h_ = weight@self.prototype
        mse = F.mse_loss(h, h_, reduction='none')
        l2_norm_square = mse.sum(dim=1,keepdim=True)
        return l2_norm_square


class DRL:
    def __init__(self, n_features, hidden_dim=128, prototype_num=10, en_nlayers=3, de_nlayers=2,
                 diversity=True, plearn=True, epochs=100, lr=1e-3, batch_size=256, device='cuda'):
        self.model_config = {
            'data_dim': n_features,
            'hidden_dim': hidden_dim,
            'prototype_num': prototype_num,
            'en_nlayers': en_nlayers,
            'de_nlayers': de_nlayers,
            'diversity': diversity,
            'plearn': plearn,
        }
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X):
        self.model = PN(self.model_config).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for x in loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                loss = self.model(x).mean()
                loss.backward()
                optimizer.step()
        return self

    def decision_function(self, X):
        self.model.eval()
        dataset = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        scores = []
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                score = self.model.predict_score(x)
                scores.append(score.cpu().numpy())

        return np.concatenate(scores).squeeze()