from torch import optim, nn
import torch
import wandb
import numpy as np
import itertools
from collections import defaultdict
import pandas as pd
import os

from matrix_completion_utils import effective_rank, complex_matmul
from data import Data, ComplexData
from models import MatrixMultiplier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log_results(epoch, model_output, pred, train_loss, val_loss, use_wandb, mode, df_dict, n):
    _, S, _ = torch.svd(pred)
    eff_rank = effective_rank(pred)
    
    wandb.log({
        "epoch": epoch,
        "singular_values": {i: s for i, s in enumerate(S.tolist()[:10])}
    })

    # log complex singular values
    if mode=='complex' or mode=='magnitude':
        _, S_complex, _ = torch.svd(post_process(model_output, "complex"))
        wandb.log({
            "epoch": epoch,
            "singular_values_complex": {i: s for i, s in enumerate(S_complex.tolist()[:10])}
        })

    # log real and imag singular values
    if mode=='quasi_complex':
        _, S_real, _ = torch.svd(model_output[0])
        _, S_imag, _ = torch.svd(model_output[1])
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

def post_process(pred, mode):
    if mode == "complex":
        return pred[0] + 1j*pred[1]
    elif mode == "quasi_complex":
        return pred[0] - pred[1]
    elif mode == "magnitude":
        return torch.sqrt(pred[0]**2 + pred[1]**2 + 1e-8)
    else:
        return pred


def calc_losses(prediction, ground_truth, indices, criterion):
    train_loss = torch.zeros((1), dtype=torch.float32).to(device)
    val_loss = torch.zeros((1), dtype=torch.float32).to(device)
    prediction = prediction if isinstance(prediction, tuple) else (prediction,)
    ground_truth = ground_truth if isinstance(ground_truth, tuple) else (ground_truth,)
    for pred, gt in zip(prediction, ground_truth):
        pred, gt = pred.flatten(), gt.flatten().to(device)
        train_loss += criterion(pred[indices], gt[indices])

        test_indices = np.setdiff1d(np.arange(gt.nelement()), indices)
        val_loss += criterion(pred[test_indices], gt[test_indices])
    return train_loss, val_loss

def custom_data():
    real_part = torch.tensor([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 0]], dtype=torch.float32)
    # imag_part = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    indices = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10])
    return real_part, indices

def train(model, step_size, epochs, observations_gt, indices, use_wandb, mode, n):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=step_size)
    df_dict = defaultdict(list)
    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model()
        pred = post_process(output, mode)
        train_loss, val_loss = calc_losses(pred, observations_gt, indices, criterion)

        train_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            log_results(epoch, output, pred, train_loss, val_loss, use_wandb, mode, df_dict, n)
      
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss.item():.5f}, Val Loss: {val_loss.item():.5f}')
    
    print("Training complete")
    print(output)
    if mode == "complex" or mode == "magnitude":
        print("Effective rank complex model: ", effective_rank(post_process(output, 'complex')))
    print("Effective rank magnitude model: ", effective_rank(pred))
    print("Effective rank real data: ", effective_rank(observations_gt))
    return df_dict

def run(init_scale, diag_init_scale, diag_noise_std, step_size, mode, n_train, 
          n, rank, depth, epochs=30001, use_wandb=True, seed=1, complex_data=False, magnitude_data=False):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = MatrixMultiplier(depth, n, mode, init_scale, diag_init_scale, diag_noise_std).to(device)
    dataObj = ComplexData(n=n, rank=rank, seed=seed) if complex_data else Data(n=n, rank=rank, seed=seed)
    observations_gt, indices = dataObj.generate_observations(n_train)
    if magnitude_data:
        observations_gt = post_process(observations_gt, 'magnitude')
    # print(observations_gt)
    # model_real = MatrixMultiplier(depth, n, 'real', init_scale, diag_init_scale, diag_noise_std).to(device)
    # model_imag = MatrixMultiplier(depth, n, 'real', init_scale, diag_init_scale, diag_noise_std).to(device)

    if use_wandb:
        wandb.watch(model)
        wandb.config["data_eff_rank"] = effective_rank(observations_gt)

    df_dict = train(model, step_size, epochs, observations_gt, indices, use_wandb, mode, n)
    # df_dict = train(model_imag, step_size, epochs, observations_gt[1], indices, use_wandb, mode, n)

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
            "init_scale":           [5e-5],
            "diag_init_scale":      [0],
            "diag_noise_std":       [0],
            "step_size":            [0.5],
            "mode":                 ['magnitude'],
            "n_train":              [3000],
            "n":                    [100],
            "rank":                 [4],
            "depth":                [5],
            "use_wandb":            [True],
            "seed":                 np.arange(10),
            "complex_data":         [True],
            "magnitude_data":       [True]
    })):
        exp_name = name(kwargs)
        if kwargs['use_wandb']:
            wandb.init(
                project="ComplexMatrixCompletion",
                entity="complex-team",
                name=exp_name,
                config=kwargs
            )
            run(**kwargs)
            wandb.finish()
        else:
            results = run(**kwargs)
            results_df = pd.DataFrame(results)
            _curr_dir = 'diag_init' if kwargs['smart_init'] else 'standard_init'
            curr_dir = os.path.join('pytorch/results', _curr_dir, kwargs['mode'], '{}.csv'.format(exp_name))
            results_df.to_csv(curr_dir)
            # results_df.to_csv('pytorch/results/{}/{}.csv'.format(curr_dir, exp_name))

if __name__=='__main__':
    main()
