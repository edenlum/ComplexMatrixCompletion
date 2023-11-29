import numpy as np
import torch
from torch import nn
import random


class MatrixMultiplier(nn.Module):
    def __init__(self, depth, size, mode, init_scale):
        super(MatrixMultiplier, self).__init__()
        self.depth = depth
        self.size = size

        self.real_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale) for _ in range(depth)])
        if mode=='real':
            self.imag_matrices = [torch.zeros(size, size) for _ in range(depth)]
        else:
            self.imag_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale) for _ in range(depth)])

    def forward(self):
        real_e2e = self.real_matrices[0]
        imag_e2e = self.imag_matrices[0]

        for real, imag in zip(self.real_matrices[1:], self.imag_matrices[1:]):
            real_e2e, imag_e2e = self.complex_matmul(real, imag, real_e2e, imag_e2e)

        return real_e2e

    @staticmethod
    def complex_matmul(real1, imag1, real2, imag2):
        real = torch.matmul(real1, real2) - torch.matmul(imag1, imag2)
        imag = torch.matmul(real1, imag2) + torch.matmul(imag1, real2)
        return real, imag
    
    def svd(self):
        return torch.svd(self.forward())
    
    def norm(self):
        return torch.norm(self.forward())
    
    def effective_rank(self):
        V, S, U = self.svd()
        # Effective Rank
        normalized_singular_values = S / torch.sum(S)
        effective_rank = torch.exp(-torch.sum(normalized_singular_values * torch.log(normalized_singular_values)))
        return effective_rank.item()


class Data:
    def __init__(self, n, rank, symmetric=False):
        self.n = n
        self.r = rank
        self.symmetric = symmetric
        self.generate_gt_matrix()

    def generate_gt_matrix(self):
        U = torch.randn(self.n, self.r)
        if self.symmetric:
            V = U
        else:
            V = torch.randn(self.n, self.r)
        w_gt = U.matmul(V.T) / np.sqrt(self.r)
        self.w_gt = w_gt / torch.norm(w_gt, 'fro') * self.n

    def generate_observations(self, n_examples):
        indices = np.random.choice(self.n*self.n, size=(n_examples,), replace=False)
        return self.w_gt, indices
    