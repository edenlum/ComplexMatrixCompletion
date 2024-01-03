import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

def get_convergence_step(arr, threshold=1e-5):
    indices = np.where(arr < threshold)[0]
    # print(indices, len(indices))
    return 10*indices[0] if len(indices) > 0 else 5e4

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

def parse_results(directory_path, diag_init, mode='real'):
    curr_dir = 'diag_init' if diag_init else 'standard_init'
    if mode == 'real':
        init_scale_idx = 7 if diag_init else 9
        lr_idx = 11
    else:
        init_scale_idx = 7
        lr_idx = 9
    dataframes = get_dataframes(os.path.join(directory_path, curr_dir, mode))
    results = defaultdict(lambda: defaultdict(list))
    effective_rank_dict = defaultdict(lambda: defaultdict(list))

    for k, v in dataframes.items():
        exp_name = k.split('_')
        # seed = exp_name[2]
        init_scale = exp_name[init_scale_idx]
        # print(exp_name, init_scale)
        lr = exp_name[lr_idx][:-4]
        results['{}'.format(lr)]['{}'.format(init_scale)].append(get_convergence_step(v['val_loss'].to_numpy()))
        effective_rank_dict['{}'.format(lr)]['{}'.format(init_scale)].append(v['eff_rank'].tolist()[-1])
        # results['{}'.format(lr)]['{}'.format(init_scale)] = get_convergence_step(v['val_loss'].to_numpy())
    
    # print(results['1.0'])
    # exit()
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
    # plt.show()
    plt.savefig('{}_{}_results.png'.format(curr_dir, mode))
    
    return results

def start_direction(depth, size, diag_init_scale, mode, diag_noise_std):
    real = np.cos((np.pi/2) / depth)
    imag = np.sin((np.pi/2) / depth)
    diag_noise_std = calc_init_scale(depth, size, diag_noise_std, mode, diag=True)
    noisy_diag = torch.zeros(size, size)
    for i in range(size):
        noisy_diag[i, i] = torch.randn(1) * diag_noise_std    
    diag_init_scale = calc_init_scale(depth, size, diag_init_scale, mode, diag=True)
    identity = torch.eye(size) * diag_init_scale
    if mode == "complex":
        return (identity * real + noisy_diag,
                identity * imag + noisy_diag)
    elif mode == "quasi_complex":
        return (identity + noisy_diag, 
                identity + noisy_diag)
    else:
        return (identity + noisy_diag, 0)

def calc_init_scale(depth, n, e2e_scale, mode, diag=False):
    # end to end scale = init_scale ^ depth * sqrt(n) ^ (depth - 1)
    # if complex, then there is another sqrt(2) for every multiplication (i.e sqrt(2)^(depth-1))
    n = n if mode == "real" else 2*n
    if diag:
        return (e2e_scale * n**(1/2))**(1/depth)
    return (e2e_scale / ((n**(1/2)) ** (depth - 1)))**(1/depth)

    
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


if __name__=='__main__':
    results = parse_results('/Users/edocoh/work/ComplexMatrixCompletion/pytorch/results', diag_init=True, mode='complex')
    # results = parse_results('/Users/edocoh/work/ComplexMatrixCompletion/pytorch/results', diag_init=False)
    # for k, v in results.items():
    #     print(k, v)
    # print(results)
