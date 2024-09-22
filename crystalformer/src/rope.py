# https://github.com/VachanVY/Rotary-Embeddings/blob/main/rope.py
from jax import (
    Array, 
    numpy as jnp
)

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
