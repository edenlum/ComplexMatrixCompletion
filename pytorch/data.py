import torch
import numpy as np

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
        indices = np.random.choice(self.n*self.n, size=(n_examples,), replace=False)
        return self.w_gt, indices
    
def main():
    dataObj = Data(n=10, rank=3)
    dataObj.generate_gt_matrix()
    print(dataObj.generate_observations(10))

if __name__ == "__main__":
    main()
    