import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises

@partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0))
def inference(model, params, G, L, X, A, M):
    dim = X.shape[-1]
    atom_types = A.shape[-1]
    outputs = model(params, G, L, X, A, M)
    mu, kappa, atom_logit, mult_logit = jnp.split(outputs[-1], [dim, 2*dim, 2*dim+atom_types], axis=-1) # only use the last one
    return mu, kappa, atom_logit, mult_logit

@partial(jax.jit, static_argnums=(1, 2, 4, 5, 6, 7, 8))
def sample_crystal(key, lattice_mlp, transformer, params, n_max, dim, batchsize, atom_types, mult_types, G):
    
    mlp_params, transformer_params = params
    
    # L ~ p(L|G)
    G = jnp.array([G]*batchsize).reshape(batchsize, 1)
    G = jax.nn.one_hot(G, 230).reshape(batchsize, 230)
    mu, sigma = jax.vmap(lattice_mlp, (None, 0), (0, 0))(mlp_params, G) # (batchsize, 6)

    key, key_l = jax.random.split(key)
    L = jax.random.normal(key_l, (batchsize, 6)) * sigma + mu # (batchsize, 6)

    X = jnp.zeros((batchsize, 0, dim))
    A = jnp.zeros((batchsize, 0, atom_types))
    M = jnp.zeros((batchsize, 0, mult_types))

    #TODO replace this with a lax.scan
    for i in range(n_max):
        mu, kappa, atom_logit, mult_logit = inference(transformer, transformer_params, G, L, X, A, M)
        key, key_x, key_a, key_m = jax.random.split(key, 4)

        x = sample_von_mises(key_x, mu, kappa, (batchsize, dim)) # [-pi, pi]
        x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into [0, 1]

        a = jax.random.categorical(key_a, atom_logit, axis=1)  # atom_logit.shape : (batchsize, atom_types)
        a = jax.nn.one_hot(a, atom_types) # (batchsize, atom_types)

        m = jax.random.categorical(key_m, mult_logit, axis=1)  # mult_logit.shape: (batchsize, mult_types)
        m = jax.nn.one_hot(m, mult_types) # (batchsize, mult_types)

        X = jnp.concatenate([X, x[:, None, :]], axis=1)
        A = jnp.concatenate([A, a[:, None, :]], axis=1)
        M = jnp.concatenate([M, m[:, None, :]], axis=1)

    return L, X, A, M
