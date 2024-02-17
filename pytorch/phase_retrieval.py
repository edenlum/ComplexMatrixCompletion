from torch import optim, nn
import torch
import wandb
import numpy as np
import itertools
from collections import defaultdict
import pandas as pd
import os
from matplotlib import pyplot as plt

from matrix_completion_utils import effective_rank, complex_matmul, process_phase_matrix
from data import Data, ComplexData
from models import MatrixMultiplier
from utils import experiments

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(init_scale, diag_init_scale, diag_noise_std, step_size, n_train, 
          n, rank, depth, epochs=5001, use_wandb=True, seed=1, fourier=False):
    model_size = n if not fourier else n*2
    model = MatrixMultiplier(depth, model_size, 'magnitude', init_scale, diag_init_scale, diag_noise_std)
    dataObj = ComplexData(n=n, rank=rank, seed=seed, magnitude=True, fourier=fourier)
    observations_gt, indices = dataObj.generate_observations(n_train)
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.to(device)
    real_gt, imag_gt = observations_gt[0].to(device), observations_gt[1].to(device)
    phase = dataObj.phase.to(device)
    phase = process_phase_matrix(phase)
    if use_wandb:
        wandb.watch(model)
        wandb.config["data_eff_rank"] = effective_rank(real_gt + 1j*imag_gt)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=step_size)

    df_dict = defaultdict(list)
    for epoch in range(epochs):
        optimizer.zero_grad()


        pred, imag = model()
        phase_pred = torch.abs(torch.atan(pred/imag))
        phase = torch.abs(torch.atan(real_gt/imag_gt))
        pred_flat, obs_flat = pred.flatten(), real_gt.flatten()
        
        train_loss = criterion(pred_flat[indices], obs_flat[indices])
        test_indices = np.setdiff1d(np.arange(obs_flat.nelement()), indices)
        val_loss = criterion(pred_flat[test_indices], obs_flat[test_indices])
        phase_loss = criterion(phase_pred.flatten(), phase.flatten())
        imag_loss = criterion(torch.abs(imag).flatten(), torch.abs(imag_gt).flatten())

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
                  "imag_eff_rank": effective_rank(imag),
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
            print(f'Epoch {epoch}/{epochs}, \
            Train Loss: {train_loss.item():.5f}, \
            Val Loss: {val_loss.item():.5f}, \
            Phase Loss: {phase_loss.item():.5f}, \
            Imag Loss: {imag_loss.item():.5f}')
    
    print("Training complete")

    print("IMAG PRED:", imag)
    print("IMAG GT:  ", imag_gt)
    print(phase)
    print(phase_pred)
    plot(dataObj.origin_gt, observations_gt, phase, pred, phase_pred)

    
    return df_dict

def plot(orig, mag_gt, phase_gt, mag_pred, phase_pred):
    # Reconstruct the signal from magnitude and phase
    complex_signal = mag_gt * torch.exp(1j * phase_gt)
    recovered_signal = torch.fft.ifftn(complex_signal).real

    complex_signal_pred = mag_pred * torch.exp(1j * phase_pred)
    recovered_signal_pred = torch.fft.ifftn(complex_signal_pred).real

    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(orig.cpu().detach().numpy(), cmap='gray')
    axs[0].set_title('Original Signal')
    axs[0].axis('off')

    axs[1].imshow(recovered_signal.cpu().detach().numpy(), cmap='gray')
    axs[1].set_title('Recovered Signal')
    axs[1].axis('off')

    axs[2].imshow(recovered_signal_pred.cpu().detach().numpy(), cmap='gray')
    axs[2].set_title('Recovered Signal Pred')
    axs[2].axis('off')

    # plt.show()
    plt.savefig("bla.png")

def name(kwargs):
    print('#'*100 + f"\n{kwargs}\n" + "#"*100)
    return f"Magnitude rank {kwargs['rank']}"
            
def main():
    for i, kwargs in enumerate(experiments({
            "init_scale":           [1e-4],
            "diag_init_scale":      [0],
            "diag_noise_std":       [0],
            "step_size":            [1],
            "n_train":              [7000],
            "n":                    [50],
            "rank":                 [1],
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
