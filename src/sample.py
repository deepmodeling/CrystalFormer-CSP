import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises
from lattice import symmetrize_lattice
from wyckoff import mult_table, symops
from augmentation import project_x

@partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0), out_axes=0) # batch 
def inference(model, params, g, X, A, W):
    M = mult_table[g-1, W]  
    return model(params, None, g, X, A, W, M, False)

@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6, 7, 8, 9, 10, 13))
def sample_crystal(key, transformer, params, n_max, dim, batchsize, atom_types, wyck_types, Kx, Kl, g, atom_mask, temperature, map_aug):
    
    xl_types = Kx+2*Kx*dim+Kl+2*6*Kl

    W = jnp.zeros((batchsize, 0), dtype=int)
    X = jnp.zeros((batchsize, 0, dim))
    A = jnp.zeros((batchsize, 0), dtype=int)
    L = jnp.zeros((batchsize, 0, Kl+2*6*Kl)) # we accumulate lattice params and sample lattice after

    for i in range(3*n_max):

        if i%3 ==0: # W_n ~ p(W_n | ...)
            w_logit = inference(transformer, params, g, X, A, W)[:, -1] # (batchsize, output_size)
            w_logit = w_logit[:, :wyck_types]

            key, subkey = jax.random.split(key)
            w = jax.random.categorical(subkey, w_logit/temperature, axis=1) 

            W = jnp.concatenate([W, w[:, None]], axis=1)

        elif i%3==1: # X_n ~ p(X_n | ...)

            # pad another zero to match the dimensionality 
            Xpad = jnp.concatenate([X, jnp.zeros((batchsize, 1, dim))], axis=1)
            Apad = jnp.concatenate([A, jnp.zeros((batchsize, 1), dtype=int)], axis=1)

            hXL = inference(transformer, params, g, Xpad, Apad, W)[:, -3] # (batchsize, output_size)

            x_logit, loc, kappa, lattice_params = jnp.split(hXL[:, :xl_types], 
                                                                     [Kx, 
                                                                      Kx+Kx*dim, 
                                                                      Kx+2*Kx*dim, 
                                                                    ], axis=-1)
            #sample coordinate from mixture of von-mises distribution 
            # k is (batchsize, ) integer array whose value in [0, Kx) 
            key, key_k, key_x, key_op = jax.random.split(key, 4)
            k = jax.random.categorical(key_k, x_logit/temperature, axis=1)  # x_logit.shape : (batchsize, Kx)

            loc = loc.reshape(batchsize, Kx, dim)
            loc = loc[jnp.arange(batchsize), k]
            kappa = kappa.reshape(batchsize, Kx, dim)
            kappa = kappa[jnp.arange(batchsize), k]

            x = sample_von_mises(key_x, loc, kappa*temperature, (batchsize, dim)) # [-pi, pi]
            x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into [0, 1]
            
            if map_aug: 
                # randomly project to a wyckoff position according to g and w
                idx = jax.random.randint(key_op, (batchsize,), 0, symops.shape[2]) # (batchsize, ) 
                x = jax.vmap(project_x, in_axes=(None, 0, 0, 0), out_axes=0)(g, w, x, idx) 
            else:
                # always project to the first WP
                x = jax.vmap(project_x, in_axes=(None, 0, 0, None), out_axes=0)(g, w, x, 0) 

            X = jnp.concatenate([X, x[:, None, :]], axis=1)
            L = jnp.concatenate([L, lattice_params[:, None, :]], axis=1)
 
        else: # A_n ~ p(A_n | ...)
            Apad = jnp.concatenate([A, jnp.zeros((batchsize, 1), dtype=int)], axis=1)
            a_logit = inference(transformer, params, g, X, Apad, W)[:, -2] # (batchsize, output_size)
            a_logit = a_logit[:, :atom_types]

            key, subkey = jax.random.split(key)
            a_logit = a_logit + jnp.where(atom_mask, 1e10, 0.0) # enhance the probability of masked atoms (do not need to normalize since we only use it for sampling, not computing logp)
            a = jax.random.categorical(subkey, a_logit/temperature, axis=1)  
            A = jnp.concatenate([A, a[:, None]], axis=1)
    
    M = mult_table[g-1, W]
    num_sites = jnp.sum(A!=0, axis=1)
    num_atoms = jnp.sum(M, axis=1)
    
    l_logit, mu, sigma = jnp.split(L[jnp.arange(batchsize), num_sites, :], [Kl, Kl+6*Kl], axis=-1)

    key, key_k, key_l = jax.random.split(key, 3)
    # k is (batchsize, ) integer array whose value in [0, Kl) 
    k = jax.random.categorical(key_k, l_logit/temperature, axis=1)  # l_logit.shape : (batchsize, Kl)

    mu = mu.reshape(batchsize, Kl, 6)
    mu = mu[jnp.arange(batchsize), k]       # (batchsize, 6)
    sigma = sigma.reshape(batchsize, Kl, 6)
    sigma = sigma[jnp.arange(batchsize), k] # (batchsize, 6)
    L = jax.random.normal(key_l, (batchsize, 6)) * sigma/jnp.sqrt(temperature) + mu # (batchsize, 6)
    
    #scale length according to atom number since we did reverse of that when loading data
    length, angle = jnp.split(L, 2, axis=-1)
    length = length*num_atoms[:, None]**(1/3)
    angle = angle * (180.0 / jnp.pi) # to deg
    L = jnp.concatenate([length, angle], axis=-1)

    #impose space group constraint to lattice params
    L = jax.vmap(symmetrize_lattice, (None, 0))(g, L)  

    return X, A, W, M, L
