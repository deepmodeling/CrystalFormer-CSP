'''
https://github.com/google-deepmind/dm-haiku/blob/main/examples/transformer/model.py
'''
import jax
import jax.numpy as jnp
import haiku as hk

from wyckoff import wmax_table

def make_transformer(key, Nf, Kx, Kl, n_max, dim, h0_size, num_layers, num_heads, key_size, model_size, atom_types, wyck_types, dropout_rate, widening_factor=4):

    @hk.transform
    def network(G, X, A, W, M, is_train):
        '''
        G: scalar integer for space group id 1-230
        X: (n, dim)
        A: (n, )  element type 
        W: (n, )  wyckoff position index
        M: (n, )  multiplicities
        '''
        
        assert (X.ndim == 2 )
        assert (X.shape[0] == A.shape[0])
        assert (X.shape[1] == dim)
        n = X.shape[0]
        
        aw_types = (atom_types -1)*(wyck_types-1) + 1
        xl_types = Kx+2*Kx*dim+Kl+2*6*Kl
        assert (aw_types > xl_types)
        aw_max = wmax_table[G-1]*(atom_types-1) #  (wmax-1) * (atom_types-1)+ (atom_types-1-1) +1  

        initializer = hk.initializers.TruncatedNormal(0.01)

        # aw_logit of the first atom is simply a table
        aw_params = hk.get_parameter('aw_params', [230, aw_types], init=initializer)
        aw_logit = aw_params[G-1]
        # mask out unavaiable position for the given spacegroup
        aw_logit = jnp.where(jnp.arange(aw_types)<=aw_max, aw_logit, aw_logit-1e10)
        # normalization
        aw_logit -= jax.scipy.special.logsumexp(aw_logit) # (aw_types, )
               
        if n > 0: 
            G_one_hot = jax.nn.one_hot(G-1, 230)
            hXL = hk.Sequential([hk.Linear(h0_size, w_init=initializer),
                                jax.nn.gelu,
                                hk.Linear(xl_types, w_init=initializer)]
                                )(jnp.concatenate([G_one_hot,              
                                                   jax.nn.one_hot(A[0], atom_types), 
                                                   jax.nn.one_hot(W[0], wyck_types), 
                                                   M[0, None]
                                                   ],axis=0)) #(230+atom_types+wyck_types+1, ) -> (xl_types, )
     
            x_logit, loc, kappa, l_logit, mu, sigma = jnp.split(hXL, [Kx, 
                                                                      Kx+Kx*dim, 
                                                                      Kx+2*Kx*dim, 
                                                                      Kx+2*Kx*dim+Kl, 
                                                                      Kx+2*Kx*dim+Kl+Kl*6, 
                                                                      ])
            # ensure positivity
            kappa = jax.nn.softplus(kappa) 
            sigma = jax.nn.softplus(sigma)
            # normalization
            x_logit -= jax.scipy.special.logsumexp(x_logit)
            l_logit -= jax.scipy.special.logsumexp(l_logit) 
            # make it up to aw_types
            hXL = jnp.concatenate([x_logit, loc, kappa, 
                                   l_logit, mu, sigma, 
                                   jnp.zeros(aw_types-xl_types)])  # (aw_types,)
        else:
            hXL = jnp.zeros((aw_types,))
        h0 = jnp.concatenate([aw_logit[None, :], hXL[None, :]], axis=0) # (2, aw_types)
        if n == 0: return h0

        mask = jnp.tril(jnp.ones((1, 2*n, 2*n))) # mask for the attention matrix

        hAM = jnp.concatenate([G_one_hot.reshape(1, 230).repeat(n, axis=0),  # (n, 230)
                               jax.nn.one_hot(A, atom_types), # (n, atom_types)
                               jax.nn.one_hot(W, wyck_types), # (n, wyck_types)
                               M.reshape(n, 1), # (n, 1)
                              ], axis=1) # (n, ...)
        hAM = hk.Linear(model_size, w_init=initializer)(hAM)  # (n, model_size)

        hX = [G_one_hot.reshape(1, 230).repeat(n, axis=0)]      
        for f in range(1, Nf+1):
            hX += [jnp.cos(2*jnp.pi*X*f),
                   jnp.sin(2*jnp.pi*X*f)]
        hX = jnp.concatenate(hX, axis=1) # (n, ...) 
        hX = hk.Linear(model_size, w_init=initializer)(hX)  # (n, model_size)

        # interleave the two matrix
        h = jnp.concatenate([hAM[:, None, :], hX[:, None, :]], axis=1) # (n, 2, model_size)
        h = h.reshape(2*n, -1)                                         # (2*n, model_size)

        positional_embeddings = hk.get_parameter(
                        'positional_embeddings', [2*n_max, model_size], init=initializer)
        h = h + positional_embeddings[:2*n, :]

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
            if is_train: 
                h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn

            dense_block = hk.Sequential([hk.Linear(widening_factor * model_size, w_init=initializer),
                                         jax.nn.gelu,
                                         hk.Linear(model_size, w_init=initializer)]
                                         )
            h_norm = _layer_norm(h)
            h_dense = dense_block(h_norm)
            if is_train:
                h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense

        h = _layer_norm(h)
        h = hk.Linear(aw_types, w_init=initializer)(h) # (2*n, aw_types)
        
        h = h.reshape(n, 2, -1)
        aw_logit, hXL = h[:, 0, :], h[:, 1, :]
        
        # (1) impose W_0 <= W_1 <= W_2 ...
        aw_mask = jnp.arange(1, aw_types).reshape(1, aw_types-1) < (W[:,None]-1)*(atom_types-1)+1 # (n, aw_types-1)
        aw_mask = jnp.concatenate([jnp.zeros((n, 1)), aw_mask], axis=1) # (n, aw_types)
        aw_logit = aw_logit - jnp.where(aw_mask, 1e10, 0.0)
        aw_logit -= jax.scipy.special.logsumexp(aw_logit, axis=1)[:, None] # normalization

        # (2) # enhance the probability of pad atoms if there is already a type 0 atom 
        aw_mask = jnp.concatenate(
                [ jnp.where(A==0, jnp.ones((n)), jnp.zeros((n))).reshape(n, 1), 
                  jnp.zeros((n, aw_types-1))
                ], axis = 1 )  # (n, aw_types) mask = 1 for those locations to place pad atoms of type 0
        aw_logit = aw_logit + jnp.where(aw_mask, 1e10, 0.0)
        aw_logit -= jax.scipy.special.logsumexp(aw_logit, axis=1)[:, None] # normalization

        # (3) mask out unavaiable position after aw_max for the given spacegroup
        aw_logit = jnp.where(jnp.arange(aw_types)<=aw_max, aw_logit,aw_logit-1e10)
        aw_logit -= jax.scipy.special.logsumexp(aw_logit, axis=1)[:, None] # normalization

        x_logit, loc, kappa, l_logit, mu, sigma = jnp.split(hXL[:, :xl_types], [Kx, 
                                                                  Kx+Kx*dim, 
                                                                  Kx+2*Kx*dim, 
                                                                  Kx+2*Kx*dim+Kl, 
                                                                  Kx+2*Kx*dim+Kl+Kl*6, 
                                                                  ], axis=-1)
        # ensure positivity
        kappa = jax.nn.softplus(kappa) 
        sigma = jax.nn.softplus(sigma)

        # normalization
        x_logit -= jax.scipy.special.logsumexp(x_logit, axis=1)[:, None] 
        l_logit -= jax.scipy.special.logsumexp(l_logit, axis=1)[:, None] 

        hXL = jnp.concatenate([x_logit, loc, kappa, 
                               l_logit, mu, sigma,
                               jnp.zeros((n, aw_types - xl_types))
                               ], axis=-1) # (n, aw_types)
        
        h = jnp.concatenate([aw_logit[:, None, :], hXL[:, None, :]], axis=1) # (n, 2, aw_types)
        h = h.reshape(2*n, aw_types) # (2*n, aw_types)

        h = jnp.concatenate( [h0, h], axis = 0) # (2*n+2, aw_types)

        return h
 

    G = jnp.array(123)
    X = jax.random.uniform(key, (n_max, dim))
    A = jax.random.uniform(key, (n_max, )) 
    W = jax.random.uniform(key, (n_max, )) 
    M = jax.random.uniform(key, (n_max, )) 

    params = network.init(key, G, X, A, W, M, 0.0)
    return params, network.apply

def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)
