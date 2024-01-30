import jax
import jax.numpy as jnp
from functools import partial

from utils import to_AW
from wyckoff import symops

def perm_augmentation(key, atom_types, X, A, W, M):
    '''
    randomly permute atoms with the same wyckoff symbol e.g. 2a, 2a, 2a, ...  
    X: (n, dim)
    A: (n, )
    W: (n, )
    M: (n, )
    '''
    n = A.shape[0]
    temp = jnp.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort
    idx_perm = jax.random.permutation(key, jnp.arange(n))
    temp = temp[idx_perm]
    idx_sort = jnp.argsort(temp)
    idx = idx_perm[idx_sort]

    X = X[idx]
    A = A[idx]
    W = W[idx]
    M = M[idx]

    AW = to_AW(A, W, atom_types)

    return X, A, W, M, AW

@partial(jax.vmap, in_axes=(None, None, 0, 0), out_axes=0) # n 
def map_augmentation(key, G, X, W):
    '''
    randomly map atoms under spacegroup symmetry operation 
    G: scalar int
    X: (dim, )
    W: scalar int 
    '''
    
    def map_fn(X):
        idx = jax.random.randint(key, (1,), 0, symops.shape[2]) # randomly sample an operation
        op = symops[G-1, W-1, idx].reshape(3, 4)

        affine_point = jnp.array([*X, 1]) # (4, )
        return jnp.dot(op, affine_point)  # (3, )

    return jnp.where(W==0, X, map_fn(X))
