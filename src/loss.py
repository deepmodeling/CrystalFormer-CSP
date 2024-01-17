import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from von_mises import von_mises_logpdf

def make_loss_fn(n_max, K, lattice_mlp, transformer):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    def logp_fn(params, G, L, X, AM):
        '''
        G: (230, )
        L: [a, b, c, alpha, beta, gamma] 
        X: (n, dim)
        AM: (n, am_types)
        '''
        
        mlp_params, transformer_params = params

        mu, sigma = lattice_mlp(mlp_params, G)
        #TODO only for these independent variables 
        logp_l = jnp.sum(jax.scipy.stats.norm.logpdf(L[:1],loc=mu[:1],scale=sigma[:1]))

        dim = X.shape[-1]
        am_types = AM.shape[-1]

        outputs = transformer(transformer_params, G, L, X, AM)
        x_logit, mu, kappa, am_logit = jnp.split(outputs[:-1], [K, K+K*dim, K+2*K*dim], axis=-1) 

        mu = mu.reshape(n_max, K, dim)
        kappa = kappa.reshape(n_max, K, dim)

        logp_x = jax.vmap(von_mises_logpdf, (None, 1, 1), 1)(X*2*jnp.pi, mu, kappa) # (n_max, K, dim)
        logp_x = jax.scipy.special.logsumexp(x_logit[..., None] + logp_x, axis=1) # (n_max, dim)
    
        AM_flat = jnp.argmax(AM, axis=-1) #(n_max, ) 
        logp_x = jnp.sum(jnp.where((AM_flat>0)[:, None], logp_x, jnp.zeros_like(logp_x)))

        logp_am = jnp.sum(am_logit[jnp.arange(n_max), AM_flat.astype(int)])  

        return logp_l + logp_x + logp_am

    def loss_fn(params, G, L, X, AM):
        logp = logp_fn(params, G, L, X, AM)
        return -jnp.mean(logp)
        
    return loss_fn

if __name__=='__main__':
    from utils import GLXAM_from_file
    from mlp import make_lattice_mlp
    from transformer import make_transformer
    atom_types = 118 
    mult_types = 5
    n_max = 5
    K = 8 
    dim = 3

    csv_file = './mini.csv'
    G, L, X, AM = GLXAM_from_file(csv_file, atom_types, mult_types, n_max, dim)

    key = jax.random.PRNGKey(42)
    
    mlp_params, lattice_mlp = make_lattice_mlp(key, 6, 32)
    transformer_params, transformer = make_transformer(key, K, 128, 4, 4, 8, 16, atom_types, mult_types) 

    params = mlp_params, transformer_params 

    outputs = jax.vmap(transformer, in_axes=(None, 0, 0, 0, 0), out_axes=0)(transformer_params, G, L, X, AM)
    x_logit, mu, kappa, am_logit = jnp.split(outputs[:, :-1], [K, K+K*dim, K+2*K*dim], axis=-1) 

    print ('x_logit.shape', x_logit.shape)
    print ('am_logit.shape', am_logit.shape)
    print ('AM_flat.shape', jnp.argmax(AM, axis=-1).shape)
    
    loss_fn = make_loss_fn(n_max, K, lattice_mlp, transformer)
    
    value, grad = jax.value_and_grad(loss_fn)(params, G[:1], L[:1], X[:1], AM[:1])
    print (value)

    value, grad = jax.value_and_grad(loss_fn)(params, G[:1], L[:1], X[:1]-1.0, AM[:1])
    print (value)
