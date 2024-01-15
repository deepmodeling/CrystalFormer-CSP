import jax
import jax.numpy as jnp
from functools import partial

from von_mises import sample_von_mises

@partial(jax.vmap, in_axes=(None, None, None, None, 0, 0, 0, 0), out_axes=(0, 0, 0, 0, 0))
def inference(model, params, K, G, L, X, A, M):
    dim = X.shape[-1]
    atom_types = A.shape[-1]
    outputs = model(params, G, L, X, A, M)
    x_logit, mu, kappa, atom_logit, mult_logit = jnp.split(outputs[-1], [K, K+K*dim, K+2*K*dim,
                                                                         K+2*K*dim+atom_types], axis=-1)
    return x_logit, mu, kappa, atom_logit, mult_logit

@partial(jax.jit, static_argnums=(1, 2, 4, 5, 6, 7, 8, 9))
def sample_crystal(key, lattice_mlp, transformer, params, n_max, dim, batchsize, atom_types, mult_types, K, G):
    
    mlp_params, transformer_params = params
    
    # L ~ p(L|G)
    G = jax.nn.one_hot(G, 230)
    mu, sigma = lattice_mlp(mlp_params, G) # (6, )

    key, key_l = jax.random.split(key)

    #TODO: now this is only for cubic, do this generally for any crystal system
    l = jax.random.normal(key_l, (batchsize, 1)) * sigma[0] + mu[0] # (batchsize, 1)
    L = jnp.concatenate([l, l, l, 
                        jnp.full((batchsize, 3), 90.0) # put in angles by hand for cubic system
                        ], axis=1) # (batchsize, 6)

    X = jnp.zeros((batchsize, 0, dim))
    A = jnp.zeros((batchsize, 0, atom_types))
    M = jnp.zeros((batchsize, 0, mult_types))

    #TODO replace this with a lax.scan
    for i in range(n_max):
        x_logit, mu, kappa, atom_logit, mult_logit = inference(transformer, transformer_params, K, G, L, X, A, M)
        key, key_k, key_x, key_a, key_m = jax.random.split(key, 5)
        
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

        a = jax.random.categorical(key_a, atom_logit, axis=1)  # atom_logit.shape : (batchsize, atom_types)
        a = jax.nn.one_hot(a, atom_types) # (batchsize, atom_types)
            
        #print (mult_logit)
        m = jax.random.categorical(key_m, mult_logit, axis=1)  # mult_logit.shape: (batchsize, mult_types)
        m = jax.nn.one_hot(m, mult_types) # (batchsize, mult_types)

        X = jnp.concatenate([X, x[:, None, :]], axis=1)
        A = jnp.concatenate([A, a[:, None, :]], axis=1)
        M = jnp.concatenate([M, m[:, None, :]], axis=1)

    return L, X, A, M
