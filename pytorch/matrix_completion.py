from torch import optim, nn
import torch
import wandb
import numpy as np
import itertools

from matrix_completion_utils import MatrixMultiplier, Data, effective_rank

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(init_scale, complex_init_scale, step_size, mode, n_train, n, rank, depth, epochs=10001, smart_init=True):
    mm = MatrixMultiplier(depth, n, mode, init_scale, complex_init_scale, smart_init)
    dataObj = Data(n=n, rank=rank)
    observations_gt, indices = dataObj.generate_observations(n_train)
    
    mm.to(device)
    observations_gt = observations_gt.to(device)

    wandb.watch(mm)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(mm.parameters(), lr=step_size)

    singular_values_list = []
    balanced_diff_list = []
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
            _, S, _ = torch.svd(pred)
            singular_values_list.append(S.tolist())
            balanced_diff_list.append(mm.calc_balanced())
            eff_rank = effective_rank(pred)

            wandb.log({
              "epoch": epoch, 
              "train_loss": train_loss, 
              "val_loss": val_loss, 
              "effective_rank": eff_rank,
              "standard_deviation": torch.std(pred).item(),
              })

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss.item():.2f}, Val Loss: {val_loss.item():.2f}')

    
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


def experiments(kwargs):
    # Extract argument names and their corresponding value lists
    arg_names = kwargs.keys()
    value_lists = kwargs.values()

    # Iterate over the Cartesian product of value lists
    for values in itertools.product(*value_lists):
        # Yield a dictionary mapping argument names to values
        yield dict(zip(arg_names, values))

def main():
    np.random.seed(1)
    for i, kwargs in enumerate(experiments({
            "init_scale":           [0],
            "complex_init_scale":   [0.001],
            "step_size":            [0.05],
            "mode":                 ['real'],
            "n_train":              [200],
            "n":                    [20],
            "rank":                 [3],
            "depth":                [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "smart_init":           [True],
    })):
        print('#'*100 + f"\n{kwargs}\n" + "#"*100)
        wandb.init(
            project="ComplexMatrixCompletion",
            entity="complex-team",
            name=f"real_d{kwargs['depth']}_diagonal_{kwargs['complex_init_scale']}",
            config=kwargs
        )
        train(**kwargs)
        wandb.finish()

if __name__=='__main__':
    main()
