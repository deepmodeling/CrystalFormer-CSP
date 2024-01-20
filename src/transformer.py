'''
https://github.com/google-deepmind/dm-haiku/blob/main/examples/transformer/model.py
'''
import jax
import jax.numpy as jnp
import haiku as hk

from utils import to_A_M, mult_list

def make_transformer(key, Nf, K, n_max, dim, h0_size, num_layers, num_heads, key_size, model_size, atom_types, mult_types, widening_factor=4):

    mult_table = jnp.array(mult_list[:mult_types])

    @hk.without_apply_rng
    @hk.transform
    def network(G, X, AM):
        '''
        G: (230, )
        X: (n, dim)
        AM: (n, am_types)
        '''
    
        assert (X.shape[0] == AM.shape[0])
        n = X.shape[0]
        assert (X.shape[1] == dim)
        am_types = AM.shape[-1]
        xl_types = K+2*K*dim+K+2*6*K
        assert (am_types > xl_types)

        initializer = hk.initializers.TruncatedNormal(0.01)

        # first atom
        am_logit = hk.Sequential([hk.Linear(h0_size, w_init=initializer),
                                  jax.nn.gelu,
                                  hk.Linear(am_types, w_init=initializer)]
                                  )(G)
        # normalization
        am_logit -= jax.scipy.special.logsumexp(am_logit) # (am_types, )
        
        if n > 0: 
            '''
            here we unpack the am code so that the transformer get more physical 
            information about A the one hot code for atom type, and M the multplicities
            '''
            A, M = to_A_M(AM, atom_types) 
            A = jax.nn.one_hot(A, am_types) # (n, atom_types)
            M = mult_table[M].reshape(n, 1) # (n, 1)

            hXL = hk.Sequential([hk.Linear(h0_size, w_init=initializer),
                                jax.nn.gelu,
                                hk.Linear(xl_types, w_init=initializer)]
                                )(jnp.concatenate([G, A[0,:], M[0,:]], axis=0)) # (xl_types, )
     
            x_logit, loc, kappa, l_logit, mu, sigma = jnp.split(hXL, [K, 
                                                                     K+K*dim, 
                                                                     K+2*K*dim, 
                                                                     K+2*K*dim+K, 
                                                                     K+2*K*dim+K+K*6, 
                                                                      ])
            # ensure positivity
            kappa = jax.nn.softplus(kappa) 
            sigma = jax.nn.softplus(sigma)
            # normalization
            x_logit -= jax.scipy.special.logsumexp(x_logit)
            l_logit -= jax.scipy.special.logsumexp(l_logit) 
            # make it up to am_types
            hXL = jnp.concatenate([x_logit, loc, kappa, 
                                   l_logit, mu, sigma, 
                                   jnp.zeros(am_types-xl_types)])  # (am_size,)
        else:
            hXL = jnp.zeros((am_types,))
        h0 = jnp.concatenate([am_logit[None, :], hXL[None, :]], axis=0)
        if n == 0: return h0

        mask = jnp.tril(jnp.ones((1, 2*n, 2*n))) # mask for the attention matrix

        hAM = jnp.concatenate([jnp.arange(n).reshape(n, 1),   
               G.reshape([1, 230]).repeat(n, axis=0), 
               A, # (n, atom_types)
               M, # (n, 1)
               ], axis=1) # (n, ...)
        hAM = hk.Linear(model_size, w_init=initializer)(hAM)  # (n, model_size)

        hX = [jnp.arange(n).reshape(n, 1),
              G.reshape([1, 230]).repeat(n, axis=0)
              ]      
        for f in range(1, Nf+1):
            hX += [jnp.cos(2*jnp.pi*X*f),
                   jnp.sin(2*jnp.pi*X*f)]
        hX = jnp.concatenate(hX, axis=1) # (n, ...) 
        hX = hk.Linear(model_size, w_init=initializer)(hX)  # (n, model_size)

        # interleave the two matrix
        h = jnp.concatenate([hAM[:, None, :], hX[:, None, :]], axis=1) # (n, 2, model_size)
        h = h.reshape(2*n, -1)                                        # (2*n, model_size)

        del hAM 
        del hX

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
        h = hk.Linear(am_types, w_init=initializer)(h) # (2*n, am_types)
        
        h = h.reshape(n, 2, -1)
        am_logit, hXL = h[:, 0, :], h[:, 1, :]
        
        AM_flat = jnp.argmax(AM, axis=-1) #(n,)
        am_mask = jnp.concatenate(
                [ jnp.where(AM_flat==0, jnp.ones((n)), jnp.zeros((n))).reshape(n, 1), 
                  jnp.zeros((n, am_types-1))
                ], axis = 1 )  # (n, am_types) mask = 1 for those locations to place pad atoms of type 0
        am_logit = am_logit + jnp.where(am_mask, 1e10, 0.0) # enhance the probability of pad atoms
        am_logit -= jax.scipy.special.logsumexp(am_logit, axis=1)[:, None] # normalization

        x_logit, loc, kappa, l_logit, mu, sigma = jnp.split(hXL[:, :xl_types], [K, 
                                                                  K+K*dim, 
                                                                  K+2*K*dim, 
                                                                  K+2*K*dim+K, 
                                                                  K+2*K*dim+K+K*6, 
                                                                  ], axis=-1)
        # ensure positivity
        kappa = jax.nn.softplus(kappa) 
        sigma = jax.nn.softplus(sigma)

        # normalization
        x_logit -= jax.scipy.special.logsumexp(x_logit, axis=1)[:, None] 
        l_logit -= jax.scipy.special.logsumexp(l_logit, axis=1)[:, None] 

        hXL = jnp.concatenate([x_logit, loc, kappa, 
                               l_logit, mu, sigma,
                               jnp.zeros((n, am_types - xl_types))
                               ], axis=-1) # (n, am_types)
        
        h = jnp.concatenate([am_logit[:, None, :], hXL[:, None, :]], axis=1) # (n, 2, am_types)
        h = h.reshape(2*n, am_types) # (2*n, am_types)

        h = jnp.concatenate( [h0, h], axis = 0) # (2*n+2, am_types)

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
