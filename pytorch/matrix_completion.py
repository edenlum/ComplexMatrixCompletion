from torch import optim
from torch import nn
import wandb
import numpy as np
import itertools

from matrix_completion_utils import MatrixMultiplier, Data

def train(init_scale, step_size, mode, n_train, n, rank, depth, epochs=10000):
    mm = MatrixMultiplier(depth, n, mode, init_scale)
    dataObj = Data(n=n, rank=rank)
    observations_gt, indices = dataObj.generate_observations(n_train)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(mm.parameters(), lr=step_size)

    wandb.init(
        project="ComplexMatrixCompletion",
        entity="complex-team",
        name=f"i{init_scale}_s{step_size}_{mode}", 
        config={
            "init_scale": init_scale,
            "step_size": step_size,
            "mode": mode,
            "n_train": n_train,
            "matrix_size": n,
            "rank": rank,
            "depth": depth
        }
    )

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = mm()
        train_loss = criterion(pred.reshape(-1)[indices], observations_gt.reshape(-1)[indices])
        val_loss = criterion(pred, observations_gt)

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            eff_rank = mm.effective_rank()
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "effective_rank": eff_rank})

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss.item():.2f}, Val Loss: {val_loss.item():.2f}')

    wandb.finish()
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
    for kwargs in experiments({"init_scale":    [1e-2, 1e-3],
                             "step_size":       [5e-2, 1e-2],
                             "mode":            ['complex'],
                             "n_train":         [200],
                             "n":               [20],
                             "rank":            [5],
                             "depth":           [2]}):
        print('#'*50)
        print(kwargs)
        print('#'*50)
        train(**kwargs)

if __name__=='__main__':
    main()
