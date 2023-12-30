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

def train(init_scale, diag_init_scale, step_size, mode, n_train, n, rank, depth, epochs=50001, smart_init=True, use_wandb=True, seed=1, quasi_complex=False):
    model = MatrixMultiplier(depth, n, mode, init_scale, diag_init_scale, smart_init)
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
            # _, S, _ = torch.svd(pred)
            # singular_values_list.append(S.tolist())
            # balanced_diff_list.append(model.calc_balanced())
            eff_rank = effective_rank(pred)

            # w_e2e = model.matrices[0]
            # for w in model.matrices[1:]:
                # w_e2e = complex_matmul(w, w_e2e)
            # _, S, _ = torch.svd(w_e2e[0] + 1j*w_e2e[1])
            # complex_sing_values_list.append(S.tolist())

            if use_wandb:
                wandb.log({
                  "epoch": epoch, 
                  "train_loss": train_loss, 
                  "val_loss": val_loss, 
                  "effective_rank": eff_rank,
                  "standard_deviation": torch.std(pred).item(),
                  "fro_norm/size": torch.norm(pred).item()/n
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

    # if use_wandb:
    #     wandb.log({
    #         "singular_values" : wandb.plot.line_series(
    #                     xs=list(range(epochs//10)), 
    #                     ys=list(zip(*singular_values_list)),
    #                     title="Singular Values",
    #                     xname="epoch/10"
    #         ),
    #         "balanced_diff" : wandb.plot.line_series(
    #                     xs=list(range(epochs//10)),
    #                     ys=list(zip(*balanced_diff_list)),
    #                     title="Balanced Difference",
    #                     xname="epoch/10"
    #         ),
    #         "complex_singular_values" : wandb.plot.line_series(
    #                     xs=list(range(epochs//10)), 
    #                     ys=list(zip(*complex_sing_values_list)),
    #                     title="Singular Values",
    #                     xname="epoch/10"
    #     )})
    
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
    if not kwargs['smart_init'] and kwargs['diag_init_scale'] > 0:
        raise ValueError("If 'smart init' is False the 'diag_init_scale' must be set to 0.")
    print('#'*100 + f"\n{kwargs}\n" + "#"*100)
    if kwargs['mode'] == 'quasi_complex' and kwargs['init_scale'] == 0.0:
        return "rnd_init_{}_depth_{}_quasiComplex_diaginitscale_{}_lr_{}".format(kwargs['seed'], kwargs['depth'], kwargs['diag_init_scale'], kwargs['step_size'])
    elif kwargs['mode'] == 'quasi_complex' and not kwargs['smart_init']:
        return "rnd_init_{}_depth_{}_quasiComplex_initscale_{}_lr_{}".format(kwargs['seed'], kwargs['depth'], kwargs['init_scale'], kwargs['step_size'])
    if kwargs['smart_init'] and kwargs['mode'] == 'complex' and kwargs['init_scale'] == 0.0:
        return "rnd_init_{}_depth_{}_complex_diaginitscale_{}_lr_{}".format(kwargs['seed'], kwargs['depth'], kwargs['diag_init_scale'], kwargs['step_size'])
    elif kwargs['mode'] == 'complex':
        return "rnd_init_{}_depth_{}_complex_diaginitscale_{}_initscale_{}_lr_{}".format(kwargs['seed'], kwargs['depth'], kwargs['diag_init_scale'], kwargs['init_scale'], kwargs['step_size'])
    elif kwargs['mode'] == 'real':
        return "rnd_init_{}_depth_{}_real_diaginitscale_{}_initscale_{}_lr_{}".format(kwargs['seed'], kwargs['depth'], kwargs['diag_init_scale'], kwargs['init_scale'], kwargs['step_size'])
    else:
        raise ValueError('Testing only real vs complex with diagonal init.')
            
def main():
    for j in np.arange(1):
        for i, kwargs in enumerate(experiments({
                "init_scale":           [1e-3, 3e-4, 1e-4, 3e-5],
                "diag_init_scale":      [0],
                "step_size":            [3],
                "mode":                 ['quasi_complex'],
                "n_train":              [2000],
                "n":                    [100],
                "rank":                 [5],
                "depth":                [4],
                "smart_init":           [False],
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
