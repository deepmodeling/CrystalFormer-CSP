import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises

@partial(jax.vmap, in_axes=(None, None, 0, 0, 0), out_axes=(0, 0, 0))
def inference(model, params, L, X, A):
    dim = X.shape[-1]
    outputs = model(params, L, X, A)
    mu, kappa, logit = jnp.split(outputs[-1], [dim, 2*dim], axis=-1) # only use the last one
    return mu, kappa, logit

@partial(jax.jit, static_argnums=(1, 2, 4, 5, 6))
def sample_crystal(key, flow_sample_fn, transformer, params, n_max, dim, batchsize):
    
    flow_params, transformer_params = params
    key, key_l = jax.random.split(key)
    L, _ = flow_sample_fn(key_l, flow_params, batchsize) # (batchsize, 6)
    X = jnp.zeros((batchsize, 1, dim))
    A = jnp.ones((batchsize, 1))
    
    #TODO replace this with a lax.scan
    for i in range(1, n_max):
        mu, kappa, logit = inference(transformer, transformer_params, L, X[:, :i], A[:, :i])
        key, key_x, key_a = jax.random.split(key, 3)

        x = sample_von_mises(key_x, mu, kappa, (batchsize, dim)) # [-pi, pi]
        x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into [0, 1]
        X = jnp.concatenate([X, x[:, None, :]], axis=1)
            
        a = jax.random.categorical(key_a, logit, axis=1)
        A = jnp.concatenate([A, a[:, None]], axis = 1)

    return L, X, A
