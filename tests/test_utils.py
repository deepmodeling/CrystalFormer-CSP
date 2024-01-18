from config import * 

from src.utils import to_A_M, GLXAM_from_file, mult_list

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

test_utils()

