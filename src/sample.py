import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises
from utils import to_A_M, mult_list
from lattice import make_spacegroup_lattice

@partial(jax.vmap, in_axes=(None, None, None, None, 0, 0, 0), out_axes=(0, 0, 0, 0, 0))
def inference(model, params, am_types, K, G, X, AM):
    dim = X.shape[-1]
    outputs = model(params, G, X, AM)
    x_logit, loc, kappa, am_logit, lattice_params = jnp.split(outputs[-1], [K, K+K*dim, 
                                                                K+2*K*dim, 
                                                                K+2*K*dim+am_types, 
                                                                ], axis=-1)
    return x_logit, loc, kappa, am_logit, lattice_params

@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6, 7, 8))
def sample_crystal(key, transformer, params, n_max, dim, batchsize, atom_types, mult_types, K, G):
    
    mult_table = jnp.array(mult_list[:mult_types])
    am_types = (atom_types -1)*(mult_types -1) + 1

    G = jax.nn.one_hot(G-1, 230).reshape(batchsize, 230)
    X = jnp.zeros((batchsize, 0, dim))
    AM = jnp.zeros((batchsize, 0, am_types))
    L = jnp.zeros((batchsize, 0, 12)) # we accumulate lattice params and sample lattice in the end

    #TODO replace this with a lax.scan
    for i in range(n_max):
        x_logit, loc, kappa, am_logit, lattice_params = inference(transformer, params, am_types, K, G, X, AM)
        key, key_k, key_x, key_am = jax.random.split(key, 4)
        
        #sample coordinate from mixture of von-mises distribution 
        # k is (batchsize, ) integer array whose value in [0, K) 
        k = jax.random.categorical(key_k, x_logit, axis=1)  # x_logit.shape : (batchsize, K)

        loc = loc.reshape(batchsize, K, dim)
        loc = loc[jnp.arange(batchsize), k]
        kappa = kappa.reshape(batchsize, K, dim)
        kappa = kappa[jnp.arange(batchsize), k]

        x = sample_von_mises(key_x, loc, kappa, (batchsize, dim)) # [-pi, pi]
        x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into [0, 1]

        am = jax.random.categorical(key_am, am_logit, axis=1)  # am_logit.shape : (batchsize, )
        m = jnp.where(am==0, jnp.zeros_like(am), am//(atom_types-1)+1)
        a = jnp.where(am==0, jnp.zeros_like(am), am%(atom_types-1)+1)
        am = jax.nn.one_hot(am, am_types) # (batchsize, am_types)

        X = jnp.concatenate([X, x[:, None, :]], axis=1)
        AM = jnp.concatenate([AM, am[:, None, :]], axis=1)

        L = jnp.concatenate([L, lattice_params[:, None, :]], axis=1)

    A, M = jax.vmap(to_A_M, (0, None))(AM, atom_types)
    num_sites, num_atoms = jnp.sum(A!=0, axis=1), jnp.sum(mult_table[M], axis=1)

    #scale length according to atom number
    length, angle, sigma = jnp.split(L[jnp.arange(batchsize), num_sites, :], [3, 6], axis=-1)
    length = length*num_atoms[:, None]**(1/3)
    mu = jnp.concatenate([length, angle], axis=1)

    key, key_l = jax.random.split(key)
    L = jax.random.normal(key_l, (batchsize, 6)) * sigma + mu # (batchsize, 6)
    L = jax.vmap(make_spacegroup_lattice)(jnp.argmax(G, axis=-1)+1, L) # impose space group constraint to lattice params 

    return X, A, M, L
