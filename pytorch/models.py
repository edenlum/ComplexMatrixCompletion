from torch import nn
from matrix_completion_utils import *

class MatrixMultiplier(nn.Module):
    def __init__(self, depth, size, mode, init_scale, diag_init_scale, diag_noise_std=0, out_features=None):
        super(MatrixMultiplier, self).__init__()
        self.depth = depth
        self.size = size
        self.mode = mode
        self.in_features = size
        self.out_features = out_features if out_features is not None else size
        self.size = self.out_features

        init_scale = calc_init_scale(depth, self.out_features, init_scale, mode)
        diag_noise_std = calc_init_scale(depth, self.out_features, diag_noise_std, mode, diag=True)
    
        def weights(i, real_imag):
            rows = self.in_features if i==0 else self.out_features
            cols = self.out_features
            return torch.randn(rows, cols) * init_scale + start_direction(depth, rows, diag_init_scale, mode, cols)[real_imag] + noisy_diag(diag_noise_std, rows, cols)
        self.real_matrices = nn.ParameterList([nn.Parameter(weights(i, 0)) for i in range(depth)])
        self.imag_matrices = nn.ParameterList([nn.Parameter(weights(i, 1)) for i in range(depth)])
        self.matrices = self.real_matrices if mode=="real" else list(zip(self.real_matrices, self.imag_matrices))

    def forward(self):
        if self.mode == "quasi_complex":
            return self._quasi_complex_forward()
        
        w_e2e = self.matrices[0]
        for w in self.matrices[1:]:
            w_e2e = complex_matmul(w_e2e, w)

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
