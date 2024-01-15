'''
https://github.com/google-deepmind/dm-haiku/blob/main/examples/transformer/model.py
'''
import jax
import jax.numpy as jnp
import haiku as hk

def make_transformer(key, num_layers, num_heads, key_size, model_size, atom_types, mult_types, widening_factor=4):

    @hk.without_apply_rng
    @hk.transform
    def network(L, X, A, M):
        '''
        L: [spacegroup 1-hot, a, b, c, alpha, beta, gamma] 
        X: (n, dim)
        A: (n, n_max, atom_types)
        M: (n, n_max, mult_types)
        '''

        n, dim = X.shape[0], X.shape[1]
        output_size = 2*dim+atom_types+mult_types

        mask = jnp.tril(jnp.ones((1, n, n))) # mask for the attention matrix

        initializer = hk.initializers.TruncatedNormal(0.01)

        h = jnp.concatenate([L.reshape([1, 236]).repeat(n, axis=0), 
                             jnp.cos(2*jnp.pi*X).reshape([n, dim]),
                             jnp.sin(2*jnp.pi*X).reshape([n, dim]),
                             A, 
                             M
                             ], 
                             axis=1) # (n, 236+3+3+atom_types+mult_types)
       
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
        h = hk.Linear(output_size, w_init=initializer)(h)

        h0 = hk.Sequential([hk.Linear(widening_factor * model_size, w_init=initializer),
                                     jax.nn.gelu,
                                     hk.Linear(output_size, w_init=initializer)]
                                     )(L)
        h = jnp.concatenate( [h0.reshape(1, output_size), 
                              h[:-1]], axis = 0) # (n, output_size)

        mu, kappa, atom_logit, mult_logit = jnp.split(h, [dim, 2*dim, 2*dim+atom_types], axis=-1)
        kappa = jax.nn.softplus(kappa) # to ensure positivity
        
        A_flat = jnp.argmax(A, axis=-1) #(n, n_max)
        mask = jnp.concatenate(
                [ jnp.where(A_flat==0, jnp.ones((n)), jnp.zeros((n))).reshape(n, 1), 
                  jnp.zeros((n, atom_types-1))
                ], axis = 1 )  # (n, atom_types) mask = 1 for those locations to place pad atoms of type 0
        atom_logit = atom_logit + jnp.where(mask, 1e10, 0.0) # enhance the probability of pad atoms
        atom_logit -= jax.scipy.special.logsumexp(atom_logit, axis=1)[:, None] # normalization

        mask = jnp.concatenate(
                [ jnp.where(A_flat==0, jnp.ones((n)), jnp.zeros((n))).reshape(n, 1), 
                  jnp.zeros((n, mult_types-1))
                ], axis = 1 )  # (n, mult_types) mask = 1 for those locations to place pad atoms of type 0
        mult_logit = mult_logit + jnp.where(mask, 1e10, 0.0) # enhance the probability of pad atoms
        mult_logit -= jax.scipy.special.logsumexp(mult_logit, axis=1)[:, None] # normalization

        return jnp.concatenate([mu, kappa, atom_logit, mult_logit], axis=-1) 

    n, dim = 24, 3
    L = jax.random.uniform(key, (236,))
    X = jax.random.uniform(key, (n, dim))
    A = jax.random.uniform(key, (n, atom_types)) 
    M = jax.random.uniform(key, (n, mult_types))
    params = network.init(key, L, X, A, M)
    return params, network.apply

def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)
