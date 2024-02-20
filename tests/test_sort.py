from config import *
from utils import sort_atoms
from utils import map_to_mesh

def test_sort_atoms():
    W = jnp.array([1, 1, 1,  1, 2, 2, 2, 0, 0])
    A = jnp.array([11, 11, 13, 14, 21, 22, 23, 0, 0])
    X = jnp.array([
                  [0., 0., 0.5555],
                  [0., 0., 0.4444],
                  [0., 0., 0.6666],
                  [0., 0., 0.1111], 
                  [0., 0., 0.1111], 
                  [0., 0., 0.3333],
                  [0., 0., 0.2222],
                  [0., 0., 0.],
                  [0., 0., 0.]])

    A, X = sort_atoms(W[None, ...], A[None, ...], X[None, ...])

    print (A)
    print (X)

def test_map_to_mesh():
    x = jnp.array([0.111, 0.3333, 0.499, 0.501, 0.666, 0.99, 0.999])
    x = map_to_mesh(x, 100)
    print (x)

test_sort_atoms()
test_map_to_mesh()
