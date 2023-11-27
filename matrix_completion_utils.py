import numpy as np
import torch
import jax.numpy as jnp
from jax import random

def random_layer_params_complex(m, n, key, scale):
    w_real_key, w_image_key, b_real_key, b_image_key = random.split(key, num=4)
    return scale * random.normal(w_real_key, (n, m)) + 1j * scale * random.normal(w_image_key, (n, m)), scale * random.normal(b_real_key, (n,)) + 1j * scale * random.normal(b_real_key, (n,))

def random_layer_params_real(m, n, key, scale):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def random_layer_params(m, n, key, scale, mode):
    if mode == 'complex':
        return random_layer_params_complex(m, n, key, scale)
    elif mode == 'real':
        return random_layer_params_real(m, n, key, scale)
    else:
        raise ValueError("Invalid mode, should be 'compelx' or 'real'.")
    # w_key, b_key = random.split(key)
    # return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale, mode):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale, mode) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def init_network_params_v2(sizes, key, scale, mode):
    keys = random.split(key, len(sizes))
    _keys = ['w{}'.format(i) for i in range(len(sizes))]
    _vals = [p[0] for p in init_network_params(sizes, key, scale, mode)]
    params = dict(zip(_keys, _vals))
    return params

class Data:
    def __init__(self, n, rank, symmetric=False):
        self.n = n
        self.r = rank
        self.symmetric = symmetric
        self.generate_gt_matrix()

    def generate_gt_matrix(self):
        U = np.random.randn(self.n, self.r).astype(jnp.float32)
        if self.symmetric:
            V = U
        else:
            V = np.random.randn(self.n, self.r).astype(jnp.float32)
        w_gt = U.dot(V.T) / jnp.sqrt(self.r)
        self.w_gt = w_gt / jnp.linalg.norm(w_gt, 'fro') * self.n

    def generate_observations(self, key, n_examples):
        indices = random.choice(key, self.n*self.n, shape=(n_examples,), replace=False)

        # us, vs = indices // self.n, indices % self.n
        # ys_ = self.w_gt[us, vs]
        obvs_vals = self.w_gt.reshape(-1)[indices]
        observations = jnp.zeros((self.n * self.n)).at[indices].set(obvs_vals)
        observations = observations.reshape(self.n, self.n)
        print(observations)
        # print('jnp.sqrt((observations**2).mean()): {}'.format(jnp.sqrt((observations**2).mean())))
        # assert 0.8 <= jnp.sqrt((observations**2).mean()) <= 1.2
        return self.w_gt, indices
        # self.w_gt.at[indices]
        # assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2
        # return [(us, vs), ys_]
        # indices = jnp.random.multinomial()
        # with torch.no_grad():
        # if problem == 'matrix-completion':
        # indices = torch.multinomial(torch.ones(n * n), n_train_samples, replacement=False)
        
        
        
        # torch.save([(us, vs), ys_], obs_path)
        # pass

if __name__ == '__main__':
    dataObj = Data(6, 3)
    print(dataObj.w_gt)
    key = random.PRNGKey(2)
    print(dataObj.generate_observations(key, 13))

# def main(n, rank, gt_path, symmetric):
#     r = rank
#     U = np.random.randn(n, r).astype(np.float32)
#     if symmetric:
#         V = U
#     else:
#         V = np.random.randn(n, r).astype(np.float32)
#     w_gt = U.dot(V.T) / np.sqrt(r)
#     w_gt = w_gt / np.linalg.norm(w_gt, 'fro') * n

#     oracle_sv = np.linalg.svd(w_gt, compute_uv=False)
#     # lz.log.info("singular values = %s, Fro(w) = %.3f", oracle_sv[:r], np.linalg.norm(w_gt, ord='fro'))
#     print(w_gt)
#     # torch.save(torch.from_numpy(w_gt), gt_path)

# def main(n, problem, n_train_samples, gt_path, obs_path, _log):
#     w_gt = torch.load(gt_path)

#     with torch.no_grad():
#         if problem == 'matrix-completion':
#             indices = torch.multinomial(torch.ones(n * n), n_train_samples, replacement=False)
#             us, vs = indices // n, indices % n
#             ys_ = w_gt[us, vs]
#             assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2
#             torch.save([(us, vs), ys_], obs_path)
#         elif problem == 'matrix-sensing':
#             xs = torch.randn(n_train_samples, n, n) / n
#             ys_ = (xs * w_gt).sum(dim=-1).sum(dim=-1)
#             assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2
#             torch.save([xs, ys_], obs_path)
#         else:
#             raise ValueError(f'unexpected problem: {problem}')
#     _log.warning('[%s] Saved %d samples to %s', problem, n_train_samples, obs_path)


# if __name__ == '__main__':
#     main()

# if __name__=='__main__':
#     main(6, 1, None, None)

