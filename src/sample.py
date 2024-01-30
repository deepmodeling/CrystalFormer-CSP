import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises
from utils import to_A_W
from lattice import symmetrize_lattice
from wyckoff import mult_table
from symops import symops 

@partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)      # batch
def map_x(G, w, x, idx):
    op = symops[G-1, w-1, idx].reshape(3, 4)
    affine_point = jnp.array([*x, 1]) # (4, )
    return jnp.dot(op, affine_point)  # (3, )

@partial(jax.vmap, in_axes=(None, None, None, None, 0, 0), out_axes=0) # batch 
def inference(model, params, atom_types, G, X, AW):
    A, W = to_A_W(AW, atom_types) 
    M = mult_table[G-1, W]  
    return model(params, None, G, X, A, W, M, False)

@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6, 7, 8, 9, 10))
def sample_crystal(key, transformer, params, n_max, dim, batchsize, atom_types, wyck_types, Kx, Kl, G, aw_mask, temperature):
    
    aw_types = (atom_types -1)*(wyck_types -1) + 1
    xl_types = Kx+2*Kx*dim+Kl+2*6*Kl

    X = jnp.zeros((batchsize, 0, dim))
    AW = jnp.zeros((batchsize, 0), dtype=int)
    L = jnp.zeros((batchsize, 0, Kl+2*6*Kl)) # we accumulate lattice paraws and sawple lattice after

    #TODO replace this with a lax.scan
    for i in range(2*n_max):

        if i%2 ==0: # AW_n ~ p(AW_n | AW_1, X_1, ..., AW_(n-1), X_(n-1))
            outputs = inference(transformer, params, atom_types, G, X, AW)[:, -2:]
            outputs = outputs.reshape(batchsize, 2, aw_types)
            aw_logit = outputs[:, 0, :] # (batchsize, aw_types)

            key, key_aw = jax.random.split(key)

            aw_logit = aw_logit + jnp.where(aw_mask, 1e10, 0.0) # enhance the probability of masked atoms (do not need to normalize since we only use it for sawpling, not computing logp)
            aw = jax.random.categorical(key_aw, aw_logit/temperature, axis=1)  # aw_logit.shape : (batchsize, )
            AW = jnp.concatenate([AW, aw[:, None]], axis=1)
 
        else: # X_n ~ p(X_n | AW_1, X_1, ... AW_(n-1), X_(n-1), AW_n)

            # pad another zero to match the dimensionality of AW
            Xpad = jnp.concatenate([X, jnp.zeros((batchsize, 1, dim))], axis=1)
            outputs = inference(transformer, params, atom_types, G, Xpad, AW)[:, -4:-2]
            outputs = outputs.reshape(batchsize, 2, aw_types)
            hXL = outputs[:, 1, :] # (batchsize, aw_types)

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

            x = sample_von_mises(key_x, loc, kappa*jnp.sqrt(temperature), (batchsize, dim)) # [-pi, pi]
            x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into [0, 1]

            # randomly project to a wyckoff position according to G and w
            w = jnp.where(aw==0, jnp.zeros_like(aw), (aw-1)//(atom_types-1)+1) # (batchsize, )
            idx = jax.random.randint(key_op, (batchsize,), 0, symops.shape[2]) # (batchsize, ) 
            x = map_x(G, w, x, idx)

            X = jnp.concatenate([X, x[:, None, :]], axis=1)

            L = jnp.concatenate([L, lattice_params[:, None, :]], axis=1)
    
    A, W = jax.vmap(to_A_W, (0, None))(AW, atom_types)
    M = mult_table[G-1, W]
    num_sites = jnp.sum(A!=0, axis=1)
    num_atoms = jnp.sum(M, axis=1)
    
    l_logit, mu, sigma = jnp.split(L[jnp.arange(batchsize), num_sites, :], [Kl, Kl+6*Kl], axis=-1)

    key, key_k, key_l = jax.random.split(key, 3)
    # k is (batchsize, ) integer array whose value in [0, Kl) 
    k = jax.random.categorical(key_k, l_logit/temperature, axis=1)  # x_logit.shape : (batchsize, Kl)

    mu = mu.reshape(batchsize, Kl, 6)
    mu = mu[jnp.arange(batchsize), k]
    sigma = sigma.reshape(batchsize, Kl, 6)
    sigma = sigma[jnp.arange(batchsize), k]
    L = jax.random.normal(key_l, (batchsize, 6)) * sigma/temperature + mu # (batchsize, 6)
    
    #scale length according to atom number since we did reverse of that when loading data
    length, angle = jnp.split(L, 2, axis=-1)
    length = length*num_atoms[:, None]**(1/3)
    L = jnp.concatenate([length, angle], axis=-1)

    #impose space group constraint to lattice paraws
    L = jax.vmap(symmetrize_lattice, (None, 0))(G, L)  

    return X, A, W, M, L, AW
