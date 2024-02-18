'''
https://github.com/google-deepmind/dm-haiku/blob/main/examples/transformer/model.py
'''
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from attention import MultiHeadAttention
from wyckoff import wmax_table, dof0_table

def make_transformer(key, Nf, Kx, Kl, n_max, dim, h0_size, num_layers, num_heads, key_size, model_size, atom_types, wyck_types, dropout_rate, widening_factor=4, sigmamin=1e-3):

    xl_types = Kx+2*Kx*dim+Kl+2*6*Kl
    output_size = np.max(np.array([xl_types, atom_types, wyck_types]))

    @hk.transform
    def network(G, X, A, W, M, is_train):
        '''
        Args:
            G: scalar integer for space group id 1-230
            X: (n, dim)
            A: (n, )  element type 
            W: (n, )  wyckoff position index
            M: (n, )  multiplicities
            is_train: bool 
        Returns: 
            h: (3n+1, output_types) it contains params [w_1, xl_1, a_1, w_2, xl_2, ..., a_n, w_{n+1}]
        '''
        
        assert (X.ndim == 2 )
        assert (X.shape[0] == A.shape[0])
        assert (X.shape[1] == dim)
        n = X.shape[0]

        w_max = wmax_table[G-1]
        initializer = hk.initializers.TruncatedNormal(0.01)
        
        G_one_hot = jax.nn.one_hot(G-1, 230) # extend this if there are property conditions

        if h0_size >0:
            # compute w_logits depending on G_one_hot
            w_logit = hk.Sequential([hk.Linear(h0_size, w_init=initializer),
                                      jax.nn.gelu,
                                      hk.Linear(wyck_types, w_init=initializer)]
                                     )(G_one_hot)
        else:
            # w_logit of the first atom is simply a table
            w_params = hk.get_parameter('w_params', [230, wyck_types], init=initializer)
            w_logit = w_params[G-1]

        # (1) the first atom should not be the pad atom
        # (2) mask out unavaiable position for the given spacegroup
        w_mask = jnp.logical_and(jnp.arange(wyck_types)>0, jnp.arange(wyck_types)<=w_max)
        w_logit = jnp.where(w_mask, w_logit, w_logit-1e10)
        # normalization
        w_logit -= jax.scipy.special.logsumexp(w_logit) # (wyck_types, )
        
        h0 = jnp.concatenate([w_logit[None, :], 
                             jnp.zeros((1, output_size-wyck_types))], axis=-1)  # (1, output_size)
        if n == 0: return h0

        mask = jnp.tril(jnp.ones((1, 3*n, 3*n))) # mask for the attention matrix

        hW = jnp.concatenate([G_one_hot[None, :].repeat(n, axis=0),  # (n, 230)
                               jax.nn.one_hot(W, wyck_types), # (n, wyck_types)
                               M.reshape(n, 1), # (n, 1)
                              ], axis=1) # (n, ...)
        hW = hk.Linear(model_size, w_init=initializer)(hW)  # (n, model_size)

        hX = [G_one_hot[None, :].repeat(n, axis=0)]      
        for f in range(1, Nf+1):
            hX += [jnp.cos(2*jnp.pi*X*f),
                   jnp.sin(2*jnp.pi*X*f)]
        hX = jnp.concatenate(hX, axis=1) # (n, ...) 
        hX = hk.Linear(model_size, w_init=initializer)(hX)  # (n, model_size)

        hA = jnp.concatenate([G_one_hot[None, :].repeat(n, axis=0),  # (n, 230)
                              jax.nn.one_hot(A, atom_types), # (n, atom_types)
                             ], axis=1) # (n, ...)
        hA = hk.Linear(model_size, w_init=initializer)(hA)  # (n, model_size)

        # interleave the three matrices
        h = jnp.concatenate([hW[:, None, :], 
                             hA[:, None, :],
                             hX[:, None, :]
                             ], axis=1) # (n, 3, model_size)
        h = h.reshape(3*n, -1)                                         # (3*n, model_size)

        positional_embeddings = hk.get_parameter(
                        'positional_embeddings', [3*n_max, model_size], init=initializer)
        h = h + positional_embeddings[:3*n, :]

        del hW
        del hX
        del hA

        for _ in range(num_layers):
            attn_block = MultiHeadAttention(num_heads=num_heads,
                                               key_size=key_size,
                                               model_size=model_size,
                                               w_init =initializer, 
                                               dropout_rate =dropout_rate
                                              )
            h_norm = _layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm, 
                                mask=mask, is_train=is_train)
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
        h = hk.Linear(output_size, w_init=initializer)(h) # (3*n, output_size)
        
        h = h.reshape(n, 3, -1)
        a_logit, hXL, w_logit = h[:, 0, :], h[:, 1, :], h[:, 2, :]

        a_logit = a_logit[:, :atom_types]
        w_logit = w_logit[:, :wyck_types]
        
        # (1) impose the constrain that W_0 <= W_1 <= W_2 
        # while for Wyckoff points with zero dof it is even stronger W_0 < W_1 
        w_mask_less_equal = jnp.arange(1, wyck_types).reshape(1, wyck_types-1) < W[:, None]
        w_mask_less = jnp.arange(1, wyck_types).reshape(1, wyck_types-1) <= W[:, None]
        w_mask = jnp.where((dof0_table[G-1, W])[:, None], w_mask_less, w_mask_less_equal) # (n, wyck_types-1)

        w_mask = jnp.concatenate([jnp.zeros((n, 1)), w_mask], axis=1) # (n, wyck_types)
        w_logit = w_logit - jnp.where(w_mask, 1e10, 0.0)
        w_logit -= jax.scipy.special.logsumexp(w_logit, axis=1)[:, None] # normalization

        # (2) # enhance the probability of pad atoms if there is already a type 0 atom 
        w_mask = jnp.concatenate(
                [ jnp.where(W==0, jnp.ones((n)), jnp.zeros((n))).reshape(n, 1), 
                  jnp.zeros((n, wyck_types-1))
                ], axis = 1 )  # (n, wyck_types) mask = 1 for those locations to place pad atoms of type 0
        w_logit = w_logit + jnp.where(w_mask, 1e10, 0.0)
        w_logit -= jax.scipy.special.logsumexp(w_logit, axis=1)[:, None] # normalization

        # (3) mask out unavaiable position after w_max for the given spacegroup
        w_logit = jnp.where(jnp.arange(wyck_types)<=w_max, w_logit, w_logit-1e10)
        w_logit -= jax.scipy.special.logsumexp(w_logit, axis=1)[:, None] # normalization

        # (4) if w !=0 the mask out the pad atom, otherwise mask out true atoms
        a_mask = jnp.concatenate(
                 [(W>0).reshape(n, 1), 
                 (W==0).reshape(n, 1).repeat(atom_types-1, axis=1) 
                 ], axis = 1 )  # (n, atom_types) mask = 1 for those locations to be masked out
        a_logit = a_logit + jnp.where(a_mask, -1e10, 0.0)
        a_logit -= jax.scipy.special.logsumexp(a_logit, axis=1)[:, None] # normalization
            
        a_logit = jnp.concatenate([a_logit, 
                                   jnp.zeros((n, output_size - atom_types))
                                   ], axis = -1) 

        w_logit = jnp.concatenate([w_logit, 
                                   jnp.zeros((n, output_size - wyck_types))
                                   ], axis = -1) 

        x_logit, loc, kappa, l_logit, mu, sigma = jnp.split(hXL[:, :xl_types], [Kx, 
                                                                  Kx+Kx*dim, 
                                                                  Kx+2*Kx*dim, 
                                                                  Kx+2*Kx*dim+Kl, 
                                                                  Kx+2*Kx*dim+Kl+Kl*6, 
                                                                  ], axis=-1)
        # ensure positivity
        kappa = jax.nn.softplus(kappa) 
        sigma = jax.nn.softplus(sigma) + sigmamin

        # normalization
        x_logit -= jax.scipy.special.logsumexp(x_logit, axis=1)[:, None] 
        l_logit -= jax.scipy.special.logsumexp(l_logit, axis=1)[:, None] 

        hXL = jnp.concatenate([x_logit, loc, kappa, 
                               l_logit, mu, sigma, 
                               jnp.zeros((n, output_size - xl_types))
                               ], axis=-1) # (n, output_size)
        
        h = jnp.concatenate([a_logit[:, None, :], 
                             hXL[:, None, :], 
                             w_logit[:, None, :]
                             ], axis=1) # (n, 3, output_size)
        h = h.reshape(3*n, output_size) # (3*n, output_size)

        h = jnp.concatenate( [h0, h], axis = 0) # (3*n+1, output_size)

        return h
 

    G = jnp.array(123)
    X = jax.random.uniform(key, (n_max, dim))
    A = jnp.zeros((n_max, ), dtype=int) 
    W = jnp.zeros((n_max, ), dtype=int) 
    M = jnp.zeros((n_max, ), dtype=int) 

    params = network.init(key, G, X, A, W, M, True)
    return params, network.apply

def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)
