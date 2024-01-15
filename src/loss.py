import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from von_mises import von_mises_logpdf

def make_loss_fn(n_max, transformer):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    def logp_fn(params, L, X, A, M):
        '''
        L: [spacegroup 1-hot, a, b, c, alpha, beta, gamma] 
        X: (n, dim)
        A: (n, n_max, atom_types)
        M: (n, n_max, mult_types)
        '''
        
        atom_types = A.shape[-1]
        mult_types = M.shape[-1]
        dim = X.shape[-1]

        outputs = transformer(params, L, X, A, M)
        mu, kappa, atom_logit, mult_logit = jnp.split(outputs, [dim, 2*dim, 2*dim+atom_types], axis=-1) 

        logp_x = von_mises_logpdf(X*2*jnp.pi, mu, kappa) # (n, dim)

        A_flat = jnp.argmax(A, axis=-1) #(n, n_max)
        M_flat = jnp.argmax(M, axis=-1) #(n, n_max)
        logp_x = jnp.sum(jnp.where((A_flat>0)[:, None], logp_x, jnp.zeros_like(logp_x)))

        logp_a = jnp.sum(atom_logit[jnp.arange(n_max), A_flat.astype(int)])  
        logp_m = jnp.sum(mult_logit[jnp.arange(n_max), M_flat.astype(int)])
        
        return logp_x + logp_a + logp_m

    def loss_fn(params, L, X, A, M):
        logp = logp_fn(params, L, X, A, M)
        return -jnp.mean(logp)
        
    return loss_fn

if __name__=='__main__':
    from utils import LXAM_from_file
    from transformer import make_transformer
    atom_types = 118 
    mult_types = 5
    n_max = 5
    dim = 3

    csv_file = './mini.csv'
    L, X, A, M = LXAM_from_file(csv_file, atom_types, mult_types, n_max, dim)

    key = jax.random.PRNGKey(42)

    params, transformer = make_transformer(key, 4, 4, 8, 16, atom_types, mult_types) 

    outputs = jax.vmap(transformer, in_axes=(None, 0, 0, 0, 0), out_axes=0)(params, L[:1], X[:1], A[:1], M[:1])
    mu, kappa, atom_logit, mult_logit = jnp.split(outputs, [dim, 2*dim, 2*dim+atom_types], axis=-1) 

    print (atom_logit)
    print (mult_logit)
    print (jnp.argmax(A, axis=-1)[:1])
    
    loss_fn = make_loss_fn(n_max, transformer)
    
    value, grad = jax.value_and_grad(loss_fn)(params, L[:1], X[:1], A[:1], M[:1])
    print (value)

    value, grad = jax.value_and_grad(loss_fn)(params, L[:1], X[:1]-1.0, A[:1], M[:1])
    print (value)
    

