import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.special import logsumexp
# from sklearn.decomposition import PCA
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
from jax import device_put


from matrix_completion_utils import init_network_params, init_network_params_v2, Data


def get_device():
    try: 
        device = jax.devices('gpu')[0]
    except RuntimeError:
        device = jax.devices()[0]
    return device

def predict(params):
    params = [params['w{}'.format(i)] for i in range(len(params.keys()))]
    # _params = params
    e2e = params[0]
    for w in params[1:]:
        e2e = jnp.dot(w, e2e)
    
    # if mode == 'complex':
    if not jnp.isrealobj(e2e):
        e2e = e2e.real
    return e2e

def get_svd(params):
    e2e = predict(params)
    U, S, Vh = jnp.linalg.svd(e2e, full_matrices=False)
    return S

def get_norm(params):
    e2e = predict(params)
    return jnp.linalg.norm(e2e, 'fro')

def train(init_scale, step_size, mode, n_train, n, rank, num_epochs=10000):
    params = init_network_params_v2([n, n, n, n], random.PRNGKey(0), scale=init_scale, mode=mode)
    dataObj = Data(n=n, rank=rank)
    observations_gt, indices = dataObj.generate_observations(random.PRNGKey(1), n_train)
    device = get_device()
    @jax.jit
    def loss(params, observations, indices):
        # obvs_vals = observations.reshape(-1)[indices]
        # observations_zeros = jnp.zeros(observations.size).at[indices].set(obvs_vals).reshape(observations.shape)
        preds = predict(params)
        return jnp.array(optax.l2_loss(preds, observations)).reshape(-1)[indices].mean()
    
    @jax.jit
    def run_epoch(params, opt_state):
        grads = jax.grad(loss)(params, observations_gt, indices)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        loss_arr.append(loss(params, observations_gt, indices=np.arange(observations_gt.size)))
        val_loss = loss(params, observations_gt, indices=np.arange(observations_gt.size))
        return params, val_loss, opt_state
 
    loss_arr = []
    optimizer = optax.adam(step_size)
    # Obtain the `opt_state` that contains statistics for the optimizer.
    observations_gt = device_put(observations_gt, device)
    params = device_put(params, device)
    nrm = get_norm(params)
    S = get_svd(params)
    print('initial SVD: {}\n'.format(S))
    print('initial norm: {}\n'.format(nrm))
    
    opt_state = optimizer.init(params)

    for epoch in range(num_epochs):
        params, val_loss, opt_state = run_epoch(params, opt_state)
        if epoch % 1000 == 0:
            print('Epoch: {}, Train loss: {}, Test loss: {}'.format(epoch, loss_arr[-1], val_loss))
    return loss_arr, params

def main():
    # activation_type_arr = ['relu', 'linear']
    activation_type_arr = ['linear']
    # init_scale_arr = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    init_scale_arr = [1e-2, 1e-3]
    # init_scale_arr = [1e-7, 5e-7, 1e-6, 5e-6]
    # step_size_arr = [1e-3, 5e-3, 1e-2, 5e-2]
    # step_size_arr = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    # step_size_arr = [1e-4, 5e-3, 1e-3]
    step_size_arr = [5e-3, 1e-3, 1e-4]
    # mode = 'real'
    # dataset = 'cifar10'
    # for activation_type in activation_type_arr:
    # activation_fn = relu if activation_type == 'relu' else lambda x: x
    for step_size in step_size_arr:
        for init_scale in init_scale_arr:
            for mode in ['real', 'complex']:
                print('#'*50)
                print('# init scale={}, lr={}, mode={}'.format(init_scale, step_size, mode))
                print('#'*50)
                loss_arr, _ = train(init_scale, step_size, mode, n_train=3000, n=100, rank=5)
                plt.plot(np.log(loss_arr), label="init_scale={}, mode={}".format(init_scale, mode))
        plt.legend()
        # plt.title('activation_{}_step_size_{}'.format(activation_type, step_size))
        plt.title('step_size_{}'.format(step_size))
        plt.savefig('matrix_completion_results/step_size_{}.png'.format(step_size))
        plt.close()

def run():
    loss_arr, params = train(
        init_scale=1e-4, step_size=1e-4, mode='real', n_train=500, n=50, rank=5
    )
    # final_e2e = predict(params)
    S = get_svd(params)
    nrm = get_norm(params)
    # print('e2e rank: {}'.format(jnp.linalg.matrix_rank(final_e2e, tol=0.01)))
    # U, S, Vh = jnp.linalg.svd(final_e2e, full_matrices=False)
    print('Final SVD:')
    print(S)
    print('Final norm: {}'.format(nrm))

if __name__=='__main__':
    run()
    # main()