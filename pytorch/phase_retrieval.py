from torch import optim, nn
import torch
import wandb
import numpy as np
import itertools
from collections import defaultdict
import pandas as pd
import os

from matrix_completion_utils import effective_rank, complex_matmul, process_phase_matrix
from data import Data, ComplexData
from models import MatrixMultiplier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(init_scale, diag_init_scale, diag_noise_std, step_size, n_train, 
          n, rank, depth, epochs=5001, use_wandb=True, seed=1, fourier=False):
    model_size = n if not fourier else 2*n
    model = MatrixMultiplier(depth, model_size, 'magnitude', init_scale, diag_init_scale, diag_noise_std)
    dataObj = ComplexData(n=n, rank=rank, seed=seed, magnitude=True, fourier=fourier)
    observations_gt, indices = dataObj.generate_observations(n_train)
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.to(device)
    observations_gt = observations_gt.to(device)
    phase = dataObj.phase.to(device)
    phase = process_phase_matrix(phase)
    if use_wandb:
        wandb.watch(model)
        wandb.config["data_eff_rank"] = effective_rank(observations_gt)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=step_size)

    df_dict = defaultdict(list)
    for epoch in range(epochs):
        optimizer.zero_grad()

        pred, phase_pred = model()
        phase_pred = process_phase_matrix(phase_pred)
        pred_flat, obs_flat = pred.flatten(), observations_gt.flatten()
        train_loss = criterion(pred_flat[indices], obs_flat[indices])
        test_indices = np.setdiff1d(np.arange(obs_flat.nelement()), indices)
        val_loss = criterion(pred_flat[test_indices], obs_flat[test_indices])
        phase_loss = criterion(phase_pred.flatten(), phase.flatten())

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            _, S, _ = torch.svd(pred)
            # balanced_diff_list.append(model.calc_balanced())
            eff_rank = effective_rank(pred)

            w_e2e = model.matrices[0]
            for w in model.matrices[1:]:
                w_e2e = complex_matmul(w, w_e2e)
            _, S_complex, _ = torch.svd(w_e2e[0] + 1j*w_e2e[1])
            wandb.log({
                "epoch": epoch,
                "singular_values_complex": {i: s for i, s in enumerate(S_complex.tolist()[:10])}
            })

            if use_wandb:
                wandb.log({
                  "epoch": epoch, 
                  "train_loss": train_loss, 
                  "val_loss": val_loss, 
                  "phase_loss": phase_loss,
                  "effective_rank": eff_rank,
                  "standard_deviation": torch.std(pred).item(),
                  "fro_norm/size": torch.norm(pred).item()/model_size,
                  "singular_values": {i: s for i, s in enumerate(S.tolist()[:10])}
                })
                  
            else:
                # log all relevant variable.
                for var in ['train_loss', 'val_loss']:
                    df_dict[var].append(eval(var).item())
                df_dict['eff_rank'].append(eff_rank)
                df_dict['standard_deviation'].append(torch.std(pred).item()),
                df_dict['fro_norm/size'].append(torch.norm(pred).item()/model_size)
      
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss.item():.5f}, Val Loss: {val_loss.item():.5f}, Phase Loss: {phase_loss.item():.5f}')
    
    print("Training complete")
    print(phase)
    print(phase_pred)
    
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
    return f"Magnitude rank {kwargs['rank']}"
            
def main():
    for i, kwargs in enumerate(experiments({
            "init_scale":           [1e-4],
            "diag_init_scale":      [0],
            "diag_noise_std":       [0],
            "step_size":            [1],
            "n_train":              [1000],
            "n":                    [50],
            "rank":                 [1,2,3,4],
            "depth":                [4],
            "use_wandb":            [True],
            "seed":                 np.arange(1),
            "fourier":              [True]
    })):
        exp_name = name(kwargs)
        if kwargs['use_wandb']:
            config = {"comment": "Ignore row and col phase and sign"}
            config.update(kwargs)
            wandb.init(
                project="ComplexMatrixCompletion",
                entity="complex-team",
                name=exp_name,
                config=config
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
