import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

def get_convergence_step(arr, threshold=3e-2):
    indices = np.where(arr < threshold)[0]
    # print(indices, len(indices))
    return indices[0] if len(indices) > 0 else 2e4

def get_dataframes(directory_path):
    # Create an empty list to store DataFrames
    data_frames = {}

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)
            
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Append the DataFrame to the list
            data_frames[filename] = df
    
    return data_frames

def dataframe_to_matrix(results, agg_fn):
    x_labels = sorted(results.keys())
    y_labels = sorted(results[x_labels[0]].keys())
    res = np.zeros((len(y_labels), len(x_labels)))
    for i, lr in enumerate(x_labels):
        for j, intscl in enumerate(y_labels):
            res[j, i] = agg_fn(results[lr][intscl])
    return x_labels, y_labels, res

def visualize_results(results, agg_fn):
    x_labels, y_labels, res = dataframe_to_matrix(results, agg_fn)
    # Create a heatmap with custom axis labels
    sns.heatmap(res, annot=True, cmap='viridis', fmt='.2f', linewidths=.5,
                xticklabels=x_labels, yticklabels=y_labels)

    # Display the heatmap
    plt.show()

def parse_results(directory_path, diag_init):
    curr_dir = 'diag_init' if diag_init else 'standard_init'
    init_scale_idx = 7 if diag_init else 9
    dataframes = get_dataframes(os.path.join(directory_path, curr_dir))
    results = defaultdict(lambda: defaultdict(list))
    effective_rank_dict = defaultdict(lambda: defaultdict(list))

    for k, v in dataframes.items():
        exp_name = k.split('_')
        seed = exp_name[2]
        init_scale = exp_name[init_scale_idx]
        lr = exp_name[11][:-4]
        # print(results['{}'.format(lr)]['{}'.format(init_scale)])
        # exit()
        results['{}'.format(lr)]['{}'.format(init_scale)].append(get_convergence_step(v['val_loss'].to_numpy()))
        effective_rank_dict['{}'.format(lr)]['{}'.format(init_scale)].append(v['eff_rank'].tolist()[-1])
        # results['{}'.format(lr)]['{}'.format(init_scale)] = get_convergence_step(v['val_loss'].to_numpy())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x_labels, y_labels, res = dataframe_to_matrix(results, np.mean)
    # Create a heatmap with custom axis labels
    sns.heatmap(res, annot=True, cmap='viridis', fmt='.2f', linewidths=.5,
                xticklabels=x_labels, yticklabels=y_labels, ax=axes[0])
    axes[0].set_title('Mean')

    x_labels, y_labels, res = dataframe_to_matrix(results, np.std)
    # Create a heatmap with custom axis labels
    sns.heatmap(res, annot=True, cmap='viridis', fmt='.2f', linewidths=.5,
                xticklabels=x_labels, yticklabels=y_labels, ax=axes[1])
    axes[1].set_title('Std')

    x_labels, y_labels, res = dataframe_to_matrix(effective_rank_dict, np.mean)
    # Create a heatmap with custom axis labels
    sns.heatmap(res, annot=True, cmap='viridis', fmt='.2f', linewidths=.5,
                xticklabels=x_labels, yticklabels=y_labels, ax=axes[2])
    axes[2].set_title('Effective Rank')

    fig.suptitle(curr_dir)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.savefig('{}_results.png'.format(curr_dir))
    
    return results

def start_direction(depth, size, complex_init_scale, mode):
    real = np.cos((np.pi/2) / depth)
    imag = np.sin((np.pi/2) / depth)
    complex_init_scale = calc_init_scale(depth, size, complex_init_scale, mode, diag=True)
    if mode == "complex":
        return torch.eye(size) * real * complex_init_scale, torch.eye(size) * imag * complex_init_scale
    else:
        return torch.eye(size) * complex_init_scale, 0

def calc_init_scale(depth, n, e2e_scale, mode, diag=False):
    # end to end scale = init_scale ^ depth * sqrt(n) ^ (depth - 1)
    # if complex, then there is another sqrt(2) for every multiplication (i.e sqrt(2)^(depth-1))
    n = n if mode == "real" else 2*n
    if diag:
      return (e2e_scale * n**(1/2))**(1/depth)
    return (e2e_scale / ((n**(1/2)) ** (depth - 1)))**(1/depth)

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
    def __init__(self, n, rank, symmetric=False, seed=1):
        self.n = n
        self.r = rank
        self.symmetric = symmetric
        self.generate_gt_matrix()
        torch.manual_seed(seed)
        np.random.seed(seed)

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
        return self.w_gt, indices
    
def main():
    dataObj = Data(n=10, rank=3)
    dataObj.generate_gt_matrix()
    print(dataObj.generate_observations(10))

if __name__=='__main__':
    results = parse_results('/Users/edocoh/work/ComplexMatrixCompletion/pytorch/results', diag_init=True)
    results = parse_results('/Users/edocoh/work/ComplexMatrixCompletion/pytorch/results', diag_init=False)
    # for k, v in results.items():
    #     print(k, v)
    # print(results)
