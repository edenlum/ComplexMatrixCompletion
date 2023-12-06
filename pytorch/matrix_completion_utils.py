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
        self.mode = mode
        if mode=="complex" and smart_init:
            real, imag = start_direction(depth, size)
        else:
            real, imag = 0, 0
        self.real_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale + real) for _ in range(depth)])
        self.imag_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale + imag) for _ in range(depth)])
        self.matrices = list(zip(self.real_matrices, self.imag_matrices)) if mode=="complex" else self.real_matrices
    
    def forward(self):
        w_e2e = self.matrices[0]

        for w in self.matrices[1:]:
            w_e2e = complex_matmul(w, w_e2e)

        # real, imag = start_direction(self.depth, self.size)
        # n1_re, n1_im, n2_re, n2_im = self.matrices[0][0] - real, self.matrices[0][1] - imag, self.matrices[-1][0] - real, self.matrices[-1][1] - imag
        # w_e2e = (w_e2e[0] - 1/np.sqrt(2) * (n1_re - n1_im + n2_re - n2_im), w_e2e[1])
            
        return w_e2e[0] if self.mode == "complex" else w_e2e

    def calc_real_parts(self):
        a, c = self.real_matrices
        b, d = self.imag_matrices
        return [torch.matmul(a, c), -torch.matmul(b, d)]

    def calc_balanced(self):
        balanced = []
        for x, y in zip(self.matrices[:-1], self.matrices[1:]):
            xxT = complex_matmul(x, conjugate_transpose(x))
            yTy = complex_matmul(conjugate_transpose(y), y)
            if self.mode == "complex":
                balanced.append(torch.norm(xxT[0] - yTy[0]))
            else:
                balanced.append(torch.norm(xxT - yTy))
        return balanced


def conjugate_transpose(w):
    if isinstance(w, tuple):
        return w[0].T, -w[1].T
    return w.T
    
def complex_matmul(w1, w2):
    if isinstance(w1, tuple):
        real1, imag1 = w1
        real2, imag2 = w2
        real = torch.matmul(real1, real2) - torch.matmul(imag1, imag2)
        imag = torch.matmul(real1, imag2) + torch.matmul(imag1, real2)
        return real, imag
    else:
        return torch.matmul(w1, w2)

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
    