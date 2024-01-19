'''
https://github.com/google-deepmind/dm-haiku/blob/main/examples/transformer/model.py
'''
import jax
import jax.numpy as jnp
import haiku as hk

def make_transformer(key, Nf, K, n_max, dim, h0_size, num_layers, num_heads, key_size, model_size, atom_types, mult_types, widening_factor=4):

    @hk.without_apply_rng
    @hk.transform
    def network(G, X, AM):
        '''
        G: (230, )
        X: (n, dim)
        AM: (n, am_types)
        '''

        n = X.shape[0]
        assert (X.shape[1] == dim)
        am_types = AM.shape[-1]
        output_size = K+2*K*dim+am_types+K+2*6*K

        initializer = hk.initializers.TruncatedNormal(0.01)

        # the first atom
        h0 = hk.Sequential([hk.Linear(h0_size, w_init=initializer),
                            jax.nn.gelu,
                            hk.Linear(output_size, w_init=initializer)]
                            )(G)
        #h0 = hk.get_parameter("h0", [output_size,], init=initializer)

        x_logit, loc, kappa, am_logit, l_logit, mu, sigma = jnp.split(h0, [K, 
                                                                  K+K*dim, 
                                                                  K+2*K*dim, 
                                                                  K+2*K*dim+am_types, 
                                                                  K+2*K*dim+am_types+K, 
                                                                  K+2*K*dim+am_types+K+K*6, 
                                                                  ])
        # ensure positivity
        kappa = jax.nn.softplus(kappa) 
        sigma = jax.nn.softplus(sigma)
        # normalization
        x_logit -= jax.scipy.special.logsumexp(x_logit)
        am_logit -= jax.scipy.special.logsumexp(am_logit)
        l_logit -= jax.scipy.special.logsumexp(l_logit) 
        h0 = jnp.concatenate([x_logit, loc, kappa, am_logit, l_logit, mu, sigma])  # (output_size,)
        if n==0: return h0.reshape(1, output_size)

        mask = jnp.tril(jnp.ones((1, n, n))) # mask for the attention matrix

        hG = hk.Sequential([hk.Linear(h0_size, w_init=initializer),
                            jax.nn.gelu,
                            hk.Linear(model_size, w_init=initializer)]
                            )(G) # (model_size,)

        hAM = hk.Sequential([hk.Linear(h0_size, w_init=initializer),
                            jax.nn.gelu,
                            hk.Linear(model_size, w_init=initializer)]
                            )(AM) # (n, model_size) 
        
        h = [hG.reshape([1, model_size]).repeat(n, axis=0), hAM] 
        for f in range(1, Nf+1):
            h += [jnp.cos(2*jnp.pi*X*f),
                  jnp.sin(2*jnp.pi*X*f)]
        h = jnp.concatenate(h,axis=1) # (n, 2*model_size+3*Nf+3*Nf)
        h = hk.Linear(model_size, w_init=initializer)(h)  # (n, model_size)

        positional_embeddings = hk.get_parameter(
                                'positional_embeddings', [n_max, model_size], init=initializer)
        h = h + positional_embeddings[:n, :]

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
        x_logit, loc, kappa, am_logit, l_logit, mu, sigma = jnp.split(h, [K, 
                                                                 K+K*dim, 
                                                                 K+2*K*dim, 
                                                                 K+2*K*dim+am_types,
                                                                 K+2*K*dim+am_types+K, 
                                                                 K+2*K*dim+am_types+K+K*6, 
                                                                 ], axis=-1)
        # ensure positivity
        kappa = jax.nn.softplus(kappa) 
        sigma = jax.nn.softplus(sigma)

        # normalization
        x_logit -= jax.scipy.special.logsumexp(x_logit, axis=1)[:, None] 
        l_logit -= jax.scipy.special.logsumexp(l_logit, axis=1)[:, None] 
        
        AM_flat = jnp.argmax(AM, axis=-1) #(n,)
        mask = jnp.concatenate(
                [ jnp.where(AM_flat==0, jnp.ones((n)), jnp.zeros((n))).reshape(n, 1), 
                  jnp.zeros((n, am_types-1))
                ], axis = 1 )  # (n, am_types) mask = 1 for those locations to place pad atoms of type 0
        am_logit = am_logit + jnp.where(mask, 1e10, 0.0) # enhance the probability of pad atoms
        am_logit -= jax.scipy.special.logsumexp(am_logit, axis=1)[:, None] # normalization

        h = jnp.concatenate([x_logit, loc, kappa, am_logit, l_logit, mu, sigma], axis=-1) # (n, output_size)

        h = jnp.concatenate( [h0.reshape(1, output_size), 
                              h
                           ], axis = 0) # (n+1, output_size)
        return h
 

    G = jax.random.uniform(key, (230, ))
    X = jax.random.uniform(key, (n_max, dim))
    am_types = (atom_types -1)*(mult_types -1) + 1
    AM = jax.random.uniform(key, (n_max, am_types)) 
    params = network.init(key, G, X, AM)
    return params, network.apply

def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)
