from torch import nn
from matrix_completion_utils import *

class MatrixMultiplier(nn.Module):
    def __init__(self, depth, size, mode, init_scale, diag_init_scale, smart_init=True):
        super(MatrixMultiplier, self).__init__()
        self.depth = depth
        self.size = size
        self.mode = mode

        if smart_init:
            real, imag = start_direction(depth, size, diag_init_scale, mode)
        else:
            real, imag = 0, 0
        init_scale = calc_init_scale(depth, size, init_scale, mode)
        self.real_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale + real) for _ in range(depth)])
        self.imag_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale + imag) for _ in range(depth)])
        self.matrices = list(zip(self.real_matrices, self.imag_matrices)) if mode=="complex" else self.real_matrices

    def forward(self):
        w_e2e = self.matrices[0]

        for w in self.matrices[1:]:
            w_e2e = complex_matmul(w, w_e2e)

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


class QuasiComplex(MatrixMultiplier):
    def __init__(self, *args, **kwargs):
        super().__init__(mode='complex', *args, **kwargs)

    def forward(self):
        real_e2e, imag_e2e = self.matrices[0]

        for real, imag in self.real_matrices[1:]:
            real_e2e = torch.matmul(real, real_e2e)
            imag_e2e = torch.matmul(imag, imag_e2e)

        return real_e2e - imag_e2e
