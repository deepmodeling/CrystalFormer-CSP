from config import *
from augmentation import sort_atoms

def test_sort_atoms():
    W = jnp.array([1, 1, 1, 2, 2, 2, 0, 0])
    A = jnp.array([11, 12, 13, 21, 22, 23, 0, 0])
    X = jnp.array([
                  [0., 0., 0.5555],
                  [0., 0., 0.4444],
                  [0., 0., 0.6666],
                  [0., 0., 0.1111], 
                  [0., 0., 0.3333],
                  [0., 0., 0.2222],
                  [0., 0., 0.],
                  [0., 0., 0.]])

    A, X = sort_atoms(W, A, X)

    print (A)
    print (X)

test_sort_atoms()
