from config import *

from src.utils import to_A_W, to_AW, GLXAW_from_file, LXA_to_csv


def test_utils():

    return

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

def test_io():

    atom_types = 119
    wyck_types = 30
    n_max = 24
    dim = 3
    num_test = 5

    csv_file = '../data/mini.csv'
    out_file = 'temp_out.csv'

    G, L, X, AW = GLXAW_from_file(csv_file, atom_types, wyck_types, n_max, dim)
    A, W = to_A_W(AW, atom_types)

    LXA_to_csv(L[:num_test], X[:num_test], A[:num_test], num_worker=2, filename=out_file)
    G_io, L_io, X_io, AW_io = GLXAW_from_file(out_file, atom_types, wyck_types, n_max, dim)
    A_io, W_io = to_A_W(AW_io, atom_types)
    if True:
        G, L, X, AW = GLXAW_from_file(out_file, atom_types, wyck_types, n_max, dim)
        A, W = to_A_W(AW, atom_types)
    os.remove(out_file)

    assert jnp.allclose(A[:num_test], A_io)
    assert jnp.allclose(W[:num_test], W_io)
    assert jnp.allclose(G[:num_test], G_io)
    assert jnp.allclose(X[:num_test], X_io)
    assert jnp.allclose(L[:num_test], L_io)


if __name__ == '__main__':

    #test_utils()
    test_A_W()
    test_io()

