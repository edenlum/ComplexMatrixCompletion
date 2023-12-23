from torch import optim, nn
import torch
import wandb
import numpy as np
import itertools
from collections import defaultdict
import pandas as pd

from matrix_completion_utils import MatrixMultiplier, Data, effective_rank

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(init_scale, diag_init_scale, step_size, mode, n_train, n, rank, depth, epochs=20001, smart_init=True, use_wandb=True, seed=1):
    mm = MatrixMultiplier(depth, n, mode, init_scale, diag_init_scale, smart_init)
    dataObj = Data(n=n, rank=rank, seed=seed)
    observations_gt, indices = dataObj.generate_observations(n_train)
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    mm.to(device)
    observations_gt = observations_gt.to(device)

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(mm.parameters(), lr=step_size)
    optimizer = optim.SGD(mm.parameters(), lr=step_size)

    singular_values_list = []
    balanced_diff_list = []
    df_dict = defaultdict(list)
    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = mm()
        pred_flat, obs_flat = pred.flatten(), observations_gt.flatten()
        train_loss = criterion(pred_flat[indices], obs_flat[indices])
        test_indices = np.setdiff1d(np.arange(obs_flat.nelement()), indices)
        val_loss = criterion(pred_flat[test_indices], obs_flat[test_indices])

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            eff_rank = effective_rank(pred)
            if use_wandb:
                _, S, _ = torch.svd(pred)
                singular_values_list.append(S.tolist())
                balanced_diff_list.append(mm.calc_balanced())

                wandb.log({
                "epoch": epoch, 
                "train_loss": train_loss, 
                "val_loss": val_loss, 
                "effective_rank": eff_rank,
                })
            else:
                # log all relevant variable.
                for var in ['train_loss', 'val_loss']:
                    df_dict[var].append(eval(var).item())
                df_dict['eff_rank'].append(eff_rank)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss.item():.2f}, Val Loss: {val_loss.item():.2f}')

    if use_wandb:
        wandb.log({
            "singular_values" : wandb.plot.line_series(
                        xs=list(range(epochs//10)), 
                        ys=list(zip(*singular_values_list)),
                        title="Singular Values",
                        xname="epoch/10"
            ),
            "balanced_diff" : wandb.plot.line_series(
                        xs=list(range(epochs//10)),
                        ys=list(zip(*balanced_diff_list)),
                        title="Balanced Difference",
                        xname="epoch/10"
        )})
    
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

def main():
    for j in np.arange(1):
        for i, kwargs in enumerate(experiments({
                "init_scale":           [0.0],
                # "diag_init_scale":      [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
                "diag_init_scale":      [1e-1],
                # "step_size":            [5e-2, 1e-1, 5e-1],
                "step_size":            [5e-1],
                # "step_size":            [5e-1],
                "mode":                 ['real'],
                "n_train":              [2000],
                "n":                    [100],
                "rank":                 [5],
                "depth":                [4],
                "smart_init":           [True],
                "use_wandb":            [True],
                "seed":                 np.arange(10),
        })):
            if not kwargs['smart_init'] and kwargs['diag_init_scale'] > 0:
                raise ValueError("If 'smart init' is False the 'diag_init_scale' must be set to 0.")
            print('#'*100 + f"\n{kwargs}\n" + "#"*100)
            if kwargs['smart_init'] and kwargs['mode'] == 'complex' and kwargs['init_scale'] == 0.0:
                exp_name = "rnd_init_{}_depth_{}_complex_diaginitscale_{}_lr_{}".format(kwargs['seed'], kwargs['depth'], kwargs['diag_init_scale'], kwargs['step_size'])
            # elif kwargs['mode'] == 'real' and kwargs['init_scale'] > 0.0 and kwargs['diag_init_scale'] == 0.0:
            elif kwargs['mode'] == 'real':
                exp_name = "rnd_init_{}_depth_{}_real_diaginitscale_{}_initscale_{}_lr_{}".format(kwargs['seed'], kwargs['depth'], kwargs['diag_init_scale'], kwargs['init_scale'], kwargs['step_size'])
            else:
                raise ValueError('Testing only real vs complex with diagonal init.')
            
            if kwargs['use_wandb']:
                wandb.init(
                    project="ComplexMatrixCompletion",
                    # entity="complex-team",
                    # name=f"experiment-{i}",
                    name=exp_name,
                    config=kwargs
                )
                train(**kwargs)
                wandb.finish()
            else:
                results = train(**kwargs)
                results_df = pd.DataFrame(results)
                curr_dir = 'diag_init' if kwargs['smart_init'] else 'standard_init'
                results_df.to_csv('pytorch/results/{}/{}.csv'.format(curr_dir, exp_name))

if __name__=='__main__':
    main()
