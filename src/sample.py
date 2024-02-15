import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises
from utils import to_A_W
from lattice import symmetrize_lattice
from wyckoff import mult_table, symops
from augmentation import project_x

@partial(jax.vmap, in_axes=(None, None, None, None, 0, 0), out_axes=0) # batch 
def inference(model, params, atom_types, g, X, AW):
    A, W = to_A_W(AW, atom_types) 
    M = mult_table[g-1, W]  
    return model(params, None, g, X, A, W, M, False)

@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6, 7, 8, 9, 10, 13))
def sample_crystal(key, transformer, params, n_max, dim, batchsize, atom_types, wyck_types, Kx, Kl, g, aw_mask, temperature, map_aug):
    
    aw_types = (atom_types -1)*(wyck_types -1) + 1
    xl_types = Kx+2*Kx*dim+Kl+2*6*Kl

    X = jnp.zeros((batchsize, 0, dim))
    AW = jnp.zeros((batchsize, 0), dtype=int)
    L = jnp.zeros((batchsize, 0, Kl+2*6*Kl)) # we accumulate lattice paraws and sawple lattice after

    for i in range(2*n_max):

        if i%2 ==0: # AW_n ~ p(AW_n | AW_1, X_1, ..., AW_(n-1), X_(n-1))
            aw_logit = inference(transformer, params, atom_types, g, X, AW)[:, -1] # (batchsize, aw_types)
            key, key_aw = jax.random.split(key)

            aw_logit = aw_logit + jnp.where(aw_mask, 1e10, 0.0) # enhance the probability of masked atoms (do not need to normalize since we only use it for sawpling, not computing logp)
            aw = jax.random.categorical(key_aw, aw_logit/temperature, axis=1)  # aw_logit.shape : (batchsize, )
            AW = jnp.concatenate([AW, aw[:, None]], axis=1)
 
        else: # X_n ~ p(X_n | AW_1, X_1, ... AW_(n-1), X_(n-1), AW_n)

            # pad another zero to match the dimensionality of AW
            Xpad = jnp.concatenate([X, jnp.zeros((batchsize, 1, dim))], axis=1)
            hXL = inference(transformer, params, atom_types, g, Xpad, AW)[:, -2] # (batchsize, aw_types)

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
            
            w = jnp.where(aw==0, jnp.zeros_like(aw), (aw-1)//(atom_types-1)+1) # (batchsize, )
            if map_aug: 
                # randomly project to a wyckoff position according to g and w
                idx = jax.random.randint(key_op, (batchsize,), 0, symops.shape[2]) # (batchsize, ) 
                x = jax.vmap(project_x, in_axes=(None, 0, 0, 0), out_axes=0)(g, w, x, idx) 
            else:
                # always project to the first WP
                x = jax.vmap(project_x, in_axes=(None, 0, 0, None), out_axes=0)(g, w, x, 0) 


            X = jnp.concatenate([X, x[:, None, :]], axis=1)

            L = jnp.concatenate([L, lattice_params[:, None, :]], axis=1)
    
    A, W = jax.vmap(to_A_W, (0, None))(AW, atom_types)
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

    return X, A, W, M, L, AW
