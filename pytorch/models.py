from torch import nn
from matrix_completion_utils import *

class MatrixMultiplier(nn.Module):
    def __init__(self, depth, size, mode, init_scale, diag_init_scale, diag_noise_std=0):
        super(MatrixMultiplier, self).__init__()
        self.depth = depth
        self.size = size
        self.mode = mode

        real, imag = start_direction(depth, size, diag_init_scale, mode, diag_noise_std)
        init_scale = calc_init_scale(depth, size, init_scale, mode)

        diag_noise_std = calc_init_scale(depth, size, diag_noise_std, mode, diag=True)
    
        self.real_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale + real + noisy_diag(diag_noise_std, size)) for _ in range(depth)])
        self.imag_matrices = nn.ParameterList([nn.Parameter(torch.randn(size, size) * init_scale + imag + noisy_diag(diag_noise_std, size)) for _ in range(depth)])
        self.matrices = self.real_matrices if mode=="real" else list(zip(self.real_matrices, self.imag_matrices))

    def forward(self):
        if self.mode == "quasi_complex":
            return self._quasi_complex_forward()
        
        w_e2e = self.matrices[0]
        for w in self.matrices[1:]:
            w_e2e = complex_matmul(w, w_e2e)

        return w_e2e 
    
    def _quasi_complex_forward(self):
        real_e2e, imag_e2e = self.matrices[0]

        for real, imag in self.matrices[1:]:
            real_e2e = torch.matmul(real, real_e2e)
            imag_e2e = torch.matmul(imag, imag_e2e)

        return real_e2e, imag_e2e

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
