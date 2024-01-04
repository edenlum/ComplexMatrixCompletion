from torch import optim, nn
import torch
import wandb
import numpy as np
import itertools
from collections import defaultdict
import pandas as pd
import os

from matrix_completion_utils import effective_rank
from data import Data
from models import MatrixMultiplier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(init_scale, diag_init_scale, diag_noise_std, step_size, mode, n_train, n, rank, depth, 
          epochs=5001, smart_init=True, use_wandb=True, seed=1):
    model = MatrixMultiplier(depth, n, mode, init_scale, diag_init_scale, diag_noise_std, smart_init)
    dataObj = Data(n=n, rank=rank, seed=seed)
    observations_gt, indices = dataObj.generate_observations(n_train)
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.to(device)
    observations_gt = observations_gt.to(device)
    if use_wandb:
        wandb.watch(model)
        wandb.config["data_eff_rank"] = effective_rank(observations_gt)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=step_size)

    singular_values_list = []
    complex_sing_values_list = []
    balanced_diff_list = []
    df_dict = defaultdict(list)
    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model()
        pred_flat, obs_flat = pred.flatten(), observations_gt.flatten()
        train_loss = criterion(pred_flat[indices], obs_flat[indices])
        test_indices = np.setdiff1d(np.arange(obs_flat.nelement()), indices)
        val_loss = criterion(pred_flat[test_indices], obs_flat[test_indices])

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            _, S, _ = torch.svd(pred)
            singular_values_list.append(S.tolist())
            # balanced_diff_list.append(model.calc_balanced())
            eff_rank = effective_rank(pred)

            # if mode=='complex':
                # w_e2e = model.matrices[0]
                # for w in model.matrices[1:]:
                #     w_e2e = complex_matmul(w, w_e2e)
                # _, S, _ = torch.svd(w_e2e[0] + 1j*w_e2e[1])
                # complex_sing_values_list.append(S.tolist())
            if mode=='quasi_complex':
                real_e2e, imag_e2e = model.matrices[0]
                for real, imag in model.matrices[1:]:
                    real_e2e = torch.matmul(real, real_e2e)
                    imag_e2e = torch.matmul(imag, imag_e2e)
                _, S_real, _ = torch.svd(real_e2e)
                _, S_imag, _ = torch.svd(imag_e2e)
                wandb.log({
                "epoch": epoch,
                "singular_values_real": {i: s for i, s in enumerate(S_real.tolist()[:10])},
                "singular_values_imag": {i: s for i, s in enumerate(S_imag.tolist()[:10])}
                })

            if use_wandb:
                wandb.log({
                  "epoch": epoch, 
                  "train_loss": train_loss, 
                  "val_loss": val_loss, 
                  "effective_rank": eff_rank,
                  "standard_deviation": torch.std(pred).item(),
                  "fro_norm/size": torch.norm(pred).item()/n,
                  "singular_values": {i: s for i, s in enumerate(S.tolist()[:10])}
                })
                  
            else:
                # log all relevant variable.
                for var in ['train_loss', 'val_loss']:
                    df_dict[var].append(eval(var).item())
                df_dict['eff_rank'].append(eff_rank)
                df_dict['standard_deviation'].append(torch.std(pred).item()),
                df_dict['fro_norm/size'].append(torch.norm(pred).item()/n)
      
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss.item():.5f}, Val Loss: {val_loss.item():.5f}')
    
    print("Training complete")
    
    return df_dict

def experiments(kwargs):
    # Extract argument names and their corresponding value lists
    arg_names = kwargs.keys()
    value_lists = kwargs.values()

    # Iterate over the Cartesian product of value lists
    for values in itertools.product(*value_lists):
        # Yield a dictionary mapping argument names to values
        yield dict(zip(arg_names, values))

def name(kwargs):
    print('#'*100 + f"\n{kwargs}\n" + "#"*100)
    return f"{kwargs['seed']}_depth_{kwargs['depth']}_{kwargs['mode']}_noise_{kwargs['init_scale']}_diag_{kwargs['diag_init_scale']}_diagnoise_{kwargs['diag_noise_std']}_lr_{kwargs['step_size']}"
            
def main():
    for i, kwargs in enumerate(experiments({
            "init_scale":           [1e-6, 0],
            "diag_init_scale":      [1e-4],
            "diag_noise_std":       [0],
            "step_size":            [3],
            "mode":                 ['quasi_complex'],
            "n_train":              [2000],
            "n":                    [100],
            "rank":                 [5],
            "depth":                [4],
            "smart_init":           [True],
            "use_wandb":            [True],
            "seed":                 np.arange(10),
    })):
        exp_name = name(kwargs)
        if kwargs['use_wandb']:
            wandb.init(
                project="ComplexMatrixCompletion",
                entity="complex-team",
                name=exp_name,
                config=kwargs
            )
            train(**kwargs)
            wandb.finish()
        else:
            results = train(**kwargs)
            results_df = pd.DataFrame(results)
            _curr_dir = 'diag_init' if kwargs['smart_init'] else 'standard_init'
            curr_dir = os.path.join('pytorch/results', _curr_dir, kwargs['mode'], '{}.csv'.format(exp_name))
            results_df.to_csv(curr_dir)
            # results_df.to_csv('pytorch/results/{}/{}.csv'.format(curr_dir, exp_name))

if __name__=='__main__':
    main()
