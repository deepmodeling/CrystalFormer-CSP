import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises
from utils import to_A_M
from lattice import make_spacegroup_lattice
from wyckoff import wyckoff_table
from symmetrize import apply_wyckoff_condition

@partial(jax.vmap, in_axes=(None, None, None, None, None, 0, 0), out_axes=0)
def inference(model, params, am_types, K, G, X, AM):
    return model(params, G, X, AM)

@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6, 7, 8, 9))
def sample_crystal(key, transformer, params, n_max, dim, batchsize, atom_types, mult_types, K, spacegroup, am_mask, temperature):
    
    am_types = (atom_types -1)*(mult_types -1) + 1
    xl_types = K+2*K*dim+K+2*6*K

    G = jax.nn.one_hot(spacegroup-1, 230) # (230,)
    X = jnp.zeros((batchsize, 0, dim))
    AM = jnp.zeros((batchsize, 0, am_types))
    L = jnp.zeros((batchsize, 0, K+2*6*K)) # we accumulate lattice params and sample lattice after

    #TODO replace this with a lax.scan
    for i in range(2*n_max):

        if i%2 ==0:
            outputs = inference(transformer, params, am_types, K, G, X, AM)[:, -2:]
            outputs = outputs.reshape(batchsize, 2, am_types)
            am_logit = outputs[:, 0, :] # (batchsize, am_types)

            key, key_am = jax.random.split(key)

            am_logit = am_logit + jnp.where(am_mask, 1e10, 0.0) # enhance the probability of masked atoms (do not need to normalize since we only use it for sampling, not computing logp)

            am = jax.random.categorical(key_am, am_logit/temperature, axis=1)  # am_logit.shape : (batchsize, )
            am = jax.nn.one_hot(am, am_types) # (batchsize, am_types)

            AM = jnp.concatenate([AM, am[:, None, :]], axis=1)
 
        else:
            # pad another zero to match the dimensionality of AM
            Xpad = jnp.concatenate([X, jnp.zeros((batchsize, 1, dim))], axis=1)
            outputs = inference(transformer, params, am_types, K, G, Xpad, AM)[:, -4:-2]
            outputs = outputs.reshape(batchsize, 2, am_types)
            hXL = outputs[:, 1, :] # (batchsize, am_types)

            key, key_k, key_x = jax.random.split(key, 3)
            x_logit, loc, kappa, lattice_params = jnp.split(hXL[:, :xl_types], 
                                                                     [K, 
                                                                      K+K*dim, 
                                                                      K+2*K*dim, 
                                                                    ], axis=-1)

            #sample coordinate from mixture of von-mises distribution 
            # k is (batchsize, ) integer array whose value in [0, K) 
            k = jax.random.categorical(key_k, x_logit/temperature, axis=1)  # x_logit.shape : (batchsize, K)

            loc = loc.reshape(batchsize, K, dim)
            loc = loc[jnp.arange(batchsize), k]
            kappa = kappa.reshape(batchsize, K, dim)
            kappa = kappa[jnp.arange(batchsize), k]

            x = sample_von_mises(key_x, loc, kappa*jnp.sqrt(temperature), (batchsize, dim)) # [-pi, pi]
            x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into [0, 1]

            # impose constrain based on space group and wyckhoff index
            am = jnp.argmax(am, axis=-1) 
            m = jnp.where(am==0, jnp.zeros_like(am), (am-1)//(atom_types-1)+1) # (batchsize, )
            x = jax.vmap(apply_wyckoff_condition, (None, 0, 0))(spacegroup, m-1, x) 
            X = jnp.concatenate([X, x[:, None, :]], axis=1)

            L = jnp.concatenate([L, lattice_params[:, None, :]], axis=1)
    
    A, M = jax.vmap(to_A_M, (0, None))(AM, atom_types)
    num_sites = jnp.sum(A!=0, axis=1)
    num_atoms = jnp.sum(wyckoff_table[spacegroup, M], axis=1)
    
    l_logit, mu, sigma = jnp.split(L[jnp.arange(batchsize), num_sites, :], [K, K+6*K], axis=-1)

    key, key_k, key_l = jax.random.split(key, 3)
    # k is (batchsize, ) integer array whose value in [0, K) 
    k = jax.random.categorical(key_k, l_logit/temperature, axis=1)  # x_logit.shape : (batchsize, K)

    mu = mu.reshape(batchsize, K, 6)
    mu = mu[jnp.arange(batchsize), k]
    sigma = sigma.reshape(batchsize, K, 6)
    sigma = sigma[jnp.arange(batchsize), k]
    L = jax.random.normal(key_l, (batchsize, 6)) * sigma/temperature + mu # (batchsize, 6)
    
    #scale length according to atom number since we did reverse of that when loading data
    length, angle = jnp.split(L, 2, axis=-1)
    length = length*num_atoms[:, None]**(1/3)
    L = jnp.concatenate([length, angle], axis=-1)

    #impose space group constraint to lattice params
    L = jax.vmap(make_spacegroup_lattice, (None, 0))(spacegroup, L)  

    return X, A, M, L
