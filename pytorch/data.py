import torch
import numpy as np

from matrix_completion_utils import complex_matmul


class Data:
    def __init__(self, n, rank, symmetric=False, seed=1):
        torch.manual_seed(seed)
        np.random.seed(seed)
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
        indices = np.random.choice(self.w_gt.nelement(), size=(n_examples,), replace=False)
        return self.w_gt, indices


class ComplexData(Data):
    def __init__(self, n, rank, symmetric=False, seed=1, magnitude=False, fourier=False):
        self.magnitude = magnitude
        self.fourier = fourier
        super().__init__(n, rank, symmetric, seed)

    def generate_gt_matrix(self):
        U_real = torch.randn(self.n, self.r)
        U_imag = torch.randn(self.n, self.r)
        if self.symmetric:
            V_real = U_real
            V_imag = U_imag
        else:
            V_real = torch.randn(self.n, self.r)
            V_imag = torch.randn(self.n, self.r)
            
        real_gt, imag_gt = complex_matmul((U_real, U_imag), (V_real.T, V_imag.T))
        if self.fourier:
            # pad the input with zeros to double the size
            padding = (self.n//2, self.n//2, self.n//2, self.n//2)
            real_gt = torch.nn.functional.pad(real_gt, padding, mode='constant', value=0)    
            f = torch.fft.fft2(real_gt)
            real_gt, imag_gt = f.real, f.imag
            
        self.complex_gt = (
          real_gt / torch.norm(real_gt, 'fro') * self.n, 
          imag_gt / torch.norm(imag_gt, 'fro') * self.n
        )
        if self.magnitude:
            self.phase = torch.atan(self.complex_gt[1] / self.complex_gt[0])
            self.w_gt = torch.sqrt(self.complex_gt[0]**2 + self.complex_gt[1]**2)
        else:
            self.w_gt = self.complex_gt[0]

    
def main():
    dataObj = Data(n=10, rank=3)
    dataObj.generate_gt_matrix()
    print(dataObj.generate_observations(10))

if __name__ == "__main__":
    main()
    