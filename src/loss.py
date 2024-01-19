import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from von_mises import von_mises_logpdf
from lattice import make_spacegroup_mask
from utils import to_A_M, mult_list

def make_loss_fn(n_max, atom_types, mult_types, K, transformer):

    mult_table = jnp.array(mult_list[:mult_types])

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    def logp_fn(params, G, L, X, AM):
        '''
        G: (230, )
        L: [a, b, c, alpha, beta, gamma] 
        X: (n_max, dim)
        AM: (n_max, am_types)
        '''
        
        dim = X.shape[-1]
        am_types = AM.shape[-1]

        A, M = to_A_M(AM, atom_types)
        num_sites = jnp.sum(A!=0)

        outputs = transformer(params, G, X, AM)

        x_logit, loc, kappa, am_logit, _ = jnp.split(outputs[:-1], [K, 
                                                                   K+K*dim, 
                                                                   K+2*K*dim, 
                                                                   K+2*K*dim+am_types, 
                                                                   ], axis=-1) 

        loc = loc.reshape(n_max, K, dim)
        kappa = kappa.reshape(n_max, K, dim)

        logp_x = jax.vmap(von_mises_logpdf, (None, 1, 1), 1)(X*2*jnp.pi, loc, kappa) # (n_max, K, dim)
        logp_x = jax.scipy.special.logsumexp(x_logit[..., None] + logp_x, axis=1) # (n_max, dim)
    
        AM_flat = jnp.argmax(AM, axis=-1) #(n_max, ) 
        logp_x = jnp.sum(jnp.where((AM_flat>0)[:, None], logp_x, jnp.zeros_like(logp_x)))

        logp_am = jnp.sum(am_logit[jnp.arange(n_max), AM_flat.astype(int)])  

        # first convert one-hot to integer, then look for mask
        spacegroup_mask = make_spacegroup_mask(jnp.argmax(G, axis=-1)+1) 
        l_logit, mu, sigma = jnp.split(outputs[num_sites, K+2*K*dim+am_types:], [K, K+K*6], axis=-1)
        mu = mu.reshape(K, 6)
        sigma = sigma.reshape(K, 6)
        logp_l = jax.vmap(jax.scipy.stats.norm.logpdf, (None, 0, 0))(L,mu,sigma) # (6, ) #(K, 6)
        logp_l = jax.scipy.special.logsumexp(l_logit[:, None] + logp_l, axis=0) # (6,)
        logp_l = jnp.sum(jnp.where((spacegroup_mask>0), logp_l, jnp.zeros_like(logp_l)))

        return logp_x + logp_am + logp_l

    def loss_fn(params, G, L, X, AM):
        logp = logp_fn(params, G, L, X, AM)
        return -jnp.mean(logp)
        
    return loss_fn

if __name__=='__main__':
    from utils import GLXAM_from_file
    from transformer import make_transformer
    atom_types = 119
    mult_types = 4
    Nf = 5
    n_max = 5
    K = 8 
    dim = 3

    csv_file = '../data/mini.csv'
    G, L, X, AM = GLXAM_from_file(csv_file, atom_types, mult_types, n_max, dim)

    am_types = AM.shape[-1]

    key = jax.random.PRNGKey(42)

    params, transformer = make_transformer(key, Nf, K, n_max, dim, 128, 4, 4, 8, 16,atom_types, mult_types) 

    outputs = jax.vmap(transformer, in_axes=(None, 0, 0, 0), out_axes=0)(params, G, X, AM)
    x_logit, loc, kappa, am_logit, _ = jnp.split(outputs[:, :-1], [K, 
                                                                  K+K*dim, 
                                                                  K+2*K*dim, 
                                                                  K+2*K*dim+am_types, 
                                                                  ], axis=-1) 

    print ('x_logit.shape', x_logit.shape)
    print ('am_logit.shape', am_logit.shape)
    print ('AM_flat.shape', jnp.argmax(AM, axis=-1).shape)


    loss_fn = make_loss_fn(n_max, atom_types, mult_types, K, transformer)
    
    value, grad = jax.value_and_grad(loss_fn)(params, G[:1], L[:1], X[:1], AM[:1])
    print (value)

    value, grad = jax.value_and_grad(loss_fn)(params, G[:1], L[:1], X[:1]-1.0, AM[:1])
    print (value)
