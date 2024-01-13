import jax
import jax.numpy as jnp
from functools import partial

from von_mises import von_mises_logpdf

def make_loss_fn(atom_types, model):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def logp_fn(params, L, X, A):
        dim = X.shape[-1]
        outputs = model(params, L, X, A)
        mu, kappa, logit = jnp.split(outputs, [dim, 2*dim], axis=-1) 

        #logp_x = jax.scipy.stats.norm.logpdf(X[1:, :], 
        #                                    loc=mu[:-1], 
        #                                    scale=kappa[:-1])

        logp_x = von_mises_logpdf(X[1:]*2*jnp.pi, mu[:-1], kappa[:-1])

        logp_x = jnp.sum(jnp.where((A[1:]>0)[:, None], logp_x, jnp.zeros_like(logp_x)))
        logp_a = jnp.sum(logit[A.astype(int)[:, None], jnp.arange(atom_types)])

        return logp_x + logp_a

    def loss_fn(params, L, X, A):
        logp = logp_fn(params, L, X, A)
        return -jnp.mean(logp)
        
    return loss_fn

if __name__=='__main__':
    from utils import LXA_from_file
    from model import make_transformer
    atom_types = 2 
    n_max = 24 
    dim = 3

    csv_file = '../data/mini.csv'
    L, X, A = LXA_from_file(csv_file, atom_types, n_max, dim)

    key = jax.random.PRNGKey(42)
    params, model = make_transformer(key, 4, 4, 8, 16, atom_types) 

    outputs = jax.vmap(model, in_axes=(None, 0, 0, 0), out_axes=0)(params, L[:1], X[:1], A[:1])
    mu, kappa, logit = jnp.split(outputs, [dim, 2*dim], axis=-1) 

    print (kappa)
    print (A[:1])
    print (X[:1])

    loss_fn = make_loss_fn(atom_types, model)
    value, grad = jax.value_and_grad(loss_fn)(params, L[:100], X[:100], A[:100])
    
    print (value)
    print (grad)

