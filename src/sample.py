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

@partial(jax.jit, static_argnums=(1, 3, 4, 6))
def sample_crystal(key, model, params, n_max, dim, atom_types, batchsize, train_data):
    
    L, _, _ = train_data
    L = L[:batchsize] # TODO: replace with a flow model sampling
    X = jnp.zeros((batchsize, 1, dim))
    A = jnp.ones((batchsize, 1))
    
    #TODO replace this with a lax.scan
    for i in range(1, n_max):
        mu, kappa, logit = inference(model, params, L, X[:, :i], A[:, :i])
        key, key_x, key_a = jax.random.split(key, 3)

        x = sample_von_mises(key_x, mu, kappa, (batchsize, dim))
        x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into 0-1
        X = jnp.concatenate([X, x[:, None, :]], axis = 1)

        a = jax.random.categorical(key_a, logit, axis=1)
        A = jnp.concatenate([A, a[:, None]], axis = 1)

    return L, X, A
