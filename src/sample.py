import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises
from utils import to_A_M, mult_table
from lattice import make_spacegroup_lattice

@partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0), out_axes=(0, 0, 0, 0))
def inference(model, params, K, G, L, X, AM):
    dim = X.shape[-1]
    outputs = model(params, G, L, X, AM)
    x_logit, mu, kappa, am_logit = jnp.split(outputs[-1], [K, K+K*dim, K+2*K*dim], axis=-1)
    return x_logit, mu, kappa, am_logit

@partial(jax.jit, static_argnums=(1, 2, 4, 5, 6, 7, 8, 9))
def sample_crystal(key, lattice_mlp, transformer, params, n_max, dim, batchsize, atom_types, mult_types, K, G):
    
    mlp_params, transformer_params = params
    
    # L ~ p(L|G)
    G = jax.nn.one_hot(G-1, 230).reshape(batchsize, 230)
    mu, sigma = jax.vmap(lattice_mlp, (None, 0))(mlp_params, G) # (batchsize, 6)
    #print ('mu, sigma for lattice', mu, sigma)

    key, key_l = jax.random.split(key)

    L = jax.random.normal(key_l, (batchsize, 6)) * sigma + mu # (batchsize, 6)
    L = jax.vmap(make_spacegroup_lattice)(jnp.argmax(G, axis=-1), L) # impose space group constraint to lattice params 

    X = jnp.zeros((batchsize, 0, dim))

    am_types = (atom_types -1)*(mult_types -1) + 1
    AM = jnp.zeros((batchsize, 0, am_types))

    #TODO replace this with a lax.scan
    for i in range(n_max):
        x_logit, mu, kappa, am_logit = inference(transformer, transformer_params, K, G, L, X, AM)
        key, key_k, key_x, key_am = jax.random.split(key, 4)
        
        #sample coordinate from mixture of von-mises distribution 
        # k is (batchsize, ) integer array whose value in [0, K) 
        k = jax.random.categorical(key_k, x_logit, axis=1)  # x_logit.shape : (batchsize, K)

        mu = mu.reshape(batchsize, K, dim)
        mu = mu[jnp.arange(batchsize), k]
        kappa = kappa.reshape(batchsize, K, dim)
        kappa = kappa[jnp.arange(batchsize), k]

        x = sample_von_mises(key_x, mu, kappa, (batchsize, dim)) # [-pi, pi]
        x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into [0, 1]
        #x = jax.random.normal(key_x, (batchsize, dim)) * kappa + mu

        am = jax.random.categorical(key_am, am_logit, axis=1)  # am_logit.shape : (batchsize, )
        m = jnp.where(am==0, jnp.zeros_like(am), am//(atom_types-1)+1)
        a = jnp.where(am==0, jnp.zeros_like(am), am%(atom_types-1)+1)

        am = jax.nn.one_hot(am, am_types) # (batchsize, am_types)

        X = jnp.concatenate([X, x[:, None, :]], axis=1)
        AM = jnp.concatenate([AM, am[:, None, :]], axis=1)

    A, M = jax.vmap(to_A_M, (0, None))(AM, atom_types)
    
    #re-normalize lattice constants according to N
    N = mult_table[M].sum(axis=1)
    L = jnp.concatenate([L[:, :3]*N[:, None]**(1/3), L[:, 3:]], axis=1)

    return L, X, A, M
