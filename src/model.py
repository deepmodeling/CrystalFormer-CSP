'''
https://github.com/google-deepmind/dm-haiku/blob/main/examples/transformer/model.py
'''
import jax
import jax.numpy as jnp
import haiku as hk

def make_transformer(key, num_layers, num_heads, key_size, model_size, atom_types, widening_factor=4):

    @hk.without_apply_rng
    @hk.transform
    def network(L, X, A):
        '''
        L: [a, b, c, alpha, beta, gamma] 
        X: (n, dim)
        A: (n, )
        '''

        n, dim = X.shape[0], X.shape[1]
        mask = jnp.tril(jnp.ones((1, n, n))) # mask for the attention matrix

        initializer = hk.initializers.TruncatedNormal(0.01)

        h = jnp.concatenate([L.reshape([1, 6]).repeat(n, axis=0), 
                             X.reshape([n, dim]),
                             A.reshape([n, 1]), 
                             ], 
                             axis=1) # (n, 10)
       
        h = hk.Linear(model_size, w_init=initializer)(h)
        
        for _ in range(num_layers):
            attn_block = hk.MultiHeadAttention(num_heads=num_heads,
                                               key_size=key_size,
                                               model_size=model_size,
                                               w_init =initializer
                                              )
            h_norm = _layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm, mask=mask)
            h = h + h_attn

            dense_block = hk.Sequential([hk.Linear(widening_factor * model_size, w_init=initializer),
                                         jax.nn.gelu,
                                         hk.Linear(model_size, w_init=initializer)]
                                         )
            h_norm = _layer_norm(h)
            h_dense = dense_block(h_norm)
            h = h + h_dense

        h = _layer_norm(h)

        h = hk.Linear(2*dim+atom_types, w_init=initializer)(h)

        mu, kappa, logit = jnp.split(h, [dim, 2*dim], axis=-1)
        kappa = jax.nn.softplus(kappa) # to ensure positivity

        mask = jnp.concatenate(
                [ jnp.where(A==0, jnp.ones((n)), jnp.zeros((n))).reshape(n, 1), 
                  jnp.zeros((n, atom_types-1))
                ], axis = 1 )  # (n, atom_types) mask = 1 for those locations to place pad atoms of type 0
        logit = logit + jnp.where(mask, 1e10, 0.0) # enhance the probability of pad atoms

        logit -= jax.scipy.special.logsumexp(logit, axis=1)[:, None] # normalization
 
        return jnp.concatenate([mu, kappa, logit], axis=-1) 

    n, dim = 24, 3
    L = jax.random.uniform(key, (6,))
    X = jax.random.uniform(key, (n, dim))
    A = jax.random.uniform(key, (n, )) 
    params = network.init(key, L, X, A)
    return params, network.apply

def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)
