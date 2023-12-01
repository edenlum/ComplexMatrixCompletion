import numpy as np
import torch
from torch import nn
import random


def start_direction(depth, size):
    real = np.cos((np.pi/2) / depth)
    imag = np.sin((np.pi/2) / depth)
    return torch.eye(size) * real, torch.eye(size) * imag

class MatrixMultiplier(nn.Module):
    def __init__(self, depth, size, mode, init_scale, smart_init=True):
        super(MatrixMultiplier, self).__init__()
        self.depth = depth
        self.size = size
        noise = torch.randn(size, size) * init_scale
        if mode=="complex" and smart_init:
            real, imag = start_direction(depth, size)
        else:
            real, imag = 0, 0
        self.real_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale + real) for _ in range(depth)])
        if mode=='real':
            self.imag_matrices = [torch.zeros(size, size) for _ in range(depth)]
        else:
            self.imag_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale + imag) for _ in range(depth)])

    def forward(self):
        real_e2e = self.real_matrices[0]
        imag_e2e = self.imag_matrices[0]

        for real, imag in zip(self.real_matrices[1:], self.imag_matrices[1:]):
            real_e2e, imag_e2e = complex_matmul(real, imag, real_e2e, imag_e2e)

        return real_e2e

    def calc_real_parts(self):
        a, c = self.real_matrices
        b, d = self.imag_matrices
        return [torch.matmul(a, c), -torch.matmul(b, d)]


def complex_matmul(real1, imag1, real2, imag2):
    real = torch.matmul(real1, real2) - torch.matmul(imag1, imag2)
    imag = torch.matmul(real1, imag2) + torch.matmul(imag1, real2)
    return real, imag

def effective_rank(mat):
    V, S, U = torch.svd(mat)
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
    