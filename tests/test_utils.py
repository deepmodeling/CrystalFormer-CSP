from config import * 

from src.utils import to_A_W, to_AW 

def test_utils():

    atom_types = 118
    mult_types = 10
    n_max = 10
    dim = 3
    csv_file = '../data/mini.csv'
    mult_table = jnp.array(mult_list[:mult_types])

    G, L, X, AM = GLXAM_from_file(csv_file, atom_types, mult_types, n_max, dim)
    
    assert G.ndim == 2 
    assert G.shape[-1] == 230
    assert L.ndim == 2 
    assert L.shape[-1] == 6

    AM_flat = jnp.argmax(AM, axis=-1)

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    A, M = jax.vmap(to_A_M, (0, None))(AM, atom_types)
    N = mult_table[M].sum(axis=1)

    assert jnp.all(N==5)

def test_A_W():
    A = jnp.array([1, 2, 3, 0])
    W = jnp.array([4, 5, 6, 0])
    atom_types = 119

    AW = to_AW(A[None, :], W[None, :], atom_types)

    A_back, W_back = to_A_W(AW[None, :], atom_types)
    
    assert jnp.allclose(A, A_back)
    assert jnp.allclose(W, W_back)

test_A_W()

