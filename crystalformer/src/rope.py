# https://github.com/VachanVY/Rotary-Embeddings/blob/main/rope.py
import jax
from jax import (
    Array, 
    numpy as jnp
)
import haiku as hk


class PositionalEmbedding:
    """```
    Sinusoidal Fixed Positional Embeddings
    Args:
        maxlen:int
        dim:int
    sinusoidal_embeddings: 
        pos_emb: (maxlen, dim)
    get_freqs:
        get_freqs: sin_freqs(maxlen, 1, dim), cos_freqs(maxlen, 1, dim)
    ```"""
    def __init__(self, maxlen:int, dim:int):
        p, i = jnp.meshgrid(jnp.arange(float(maxlen)), jnp.arange(dim/2)*2)
        theta = (p/1e4**(i/dim)).T

        self.pos_emb = jnp.stack([jnp.sin(theta), jnp.cos(theta)], axis=-1)
        self.pos_emb = self.pos_emb.reshape((maxlen, dim)) # (maxlen, dim)

    def sinusoidal_embeddings(self):
        return self.pos_emb # (maxlen, dim)
    
    def get_freqs(self):
        sin_freqs = jnp.repeat(self.pos_emb[..., None, ::2], repeats=2, axis=-1)
        cos_freqs = jnp.repeat(self.pos_emb[..., None, 1::2], repeats=2, axis=-1)
        return sin_freqs, cos_freqs # (maxlen, 1, dim), (maxlen, 1, dim)
    

def apply_rotary_embeddings(q:Array, k:Array, sin_freqs:Array, cos_freqs:Array):
    """
    q.shape=[seq_len, num_heads, hidden_size]
    k.shape=[seq_len, num_heads, hidden_size]

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.

    """
    T = q.shape[0]

    minus_swap_alternate = lambda x: jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)

    q = q*cos_freqs[:T, :, :] + minus_swap_alternate(q)*sin_freqs[:T, :, :] # (T, H, D)*(T, 1, D) + (T, H, D)*(T, 1, D)
    k = k*cos_freqs[:T, :, :] + minus_swap_alternate(k)*sin_freqs[:T, :, :] # (T, H, D)*(T, 1, D) + (T, H, D)*(T, 1, D)
    return q, k # (T, H, D), (T, H, D)


class RelativePosition(hk.Module):
    """
    Relative Positional Embeddings
    
    e_ij = (x_i * W^Q) * (x_j * W^K)^T / sqrt(d) + d_ij
    d_ij is the relative position embedding
    """
    def __init__(self, max_relative_position):
        """
        max_relative_position: maximum relative position
        """
        
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embeddings_table = hk.get_parameter(
            "embeddings_table",
            shape=(max_relative_position * 2 + 1, ),
            init=hk.initializers.TruncatedNormal(0.01)
        )

    def __call__(self, length_q, length_k):
        range_vec_q = jnp.arange(length_q)
        range_vec_k = jnp.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = jax.lax.clamp(-self.max_relative_position, distance_mat, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.astype(int)
        embeddings = self.embeddings_table[final_mat]

        return embeddings
