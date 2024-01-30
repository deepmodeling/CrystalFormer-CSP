from config import *

from utils import to_A_W, to_AW, GLXAW_from_file, GLXA_to_csv
from wyckoff import mult_table


def test_utils():

    atom_types = 119
    mult_types = 10
    n_max = 10
    dim = 3
    csv_file = '../data/mini.csv'

    G, L, X, AW = GLXAW_from_file(csv_file, atom_types, mult_types, n_max, dim)
    
    assert G.ndim == 1
    assert L.ndim == 2 
    assert L.shape[-1] == 6

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    A, W = jax.vmap(to_A_W, (0, None))(AW, atom_types)
    print ("A:\n", A)
    @jax.vmap
    def lookup(G, W):
        return mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    N = M.sum(axis=-1)

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

    GLXA_to_csv(G[:num_test], L[:num_test], X[:num_test], A[:num_test], num_worker=2, filename=out_file)
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

    test_utils()
    test_A_W()
    test_io()

