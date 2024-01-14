import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from von_mises import von_mises_logpdf

def make_loss_fn(n_max, flow, transformer):

    @partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    def logp_fn(params, L, X, A):
        flow_params, transformer_params = params 
        logp_l = flow(flow_params, L)

        dim = X.shape[-1]
        outputs = transformer(transformer_params, L, X, A)
        mu, kappa, logit = jnp.split(outputs, [dim, 2*dim], axis=-1) 

        logp_x = von_mises_logpdf(X[1:]*2*jnp.pi, mu[:-1], kappa[:-1]) # (n-1, dim)

        logp_x = jnp.sum(jnp.where((A[1:]>0)[:, None], logp_x, jnp.zeros_like(logp_x)))
        logp_a = jnp.sum(logit[:-1][jnp.arange(n_max-1), A[1:].astype(int)])  #TODO: currently the first atom always has type 1 so we do not consider it 
        
        return logp_l + logp_x + logp_a

    def loss_fn(params, L, X, A):
        logp = logp_fn(params, L, X, A)
        return -jnp.mean(logp)
        
    return loss_fn

if __name__=='__main__':
    from utils import LXA_from_file
    from realnvp import make_flow
    from transformer import make_transformer
    atom_types = 2 
    n_max = 24 
    dim = 3

    csv_file = '../data/mini.csv'
    L, X, A = LXA_from_file(csv_file, atom_types, n_max, dim)

    key = jax.random.PRNGKey(42)

    flow_params, flow_logprob_fn, flow_sample_fn = make_flow(key, 6, 32, 2)

    transformer_params, transformer = make_transformer(key, 4, 4, 8, 16, atom_types) 

    params = flow_params, transformer_params 

    outputs = jax.vmap(transformer, in_axes=(None, 0, 0, 0), out_axes=0)(transformer_params, L[:1], X[:1], A[:1])
    mu, kappa, logit = jnp.split(outputs, [dim, 2*dim], axis=-1) 

    print (logit)
    print (A[:1])
    
    loss_fn = make_loss_fn(n_max, flow_logprob_fn, transformer)
    
    value, grad = jax.value_and_grad(loss_fn)(params, L[:1], X[:1], A[:1])
    print (value)

    value, grad = jax.value_and_grad(loss_fn)(params, L[:1], X[:1]-1.0, A[:1])
    print (value)
    

