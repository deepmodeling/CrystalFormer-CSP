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

    assert (W == W[idx])

    X = X[idx]
    A = A[idx]
    M = M[idx]

    AW = to_AW(A, W, atom_types)

    return X, A, W, M, AW

@partial(jax.vmap, in_axes=(None, None, 0, 0), out_axes=0) # n 
def map_augmentation(key, G, W, X):
    '''
    randomly map atoms under spacegroup symmetry operation 
    Args:
        G: scalar int
        W: scalar int 
        X: (dim, )
    Return:
        X: (dim, ) transformed coordinate 
    '''
    
    idx = jax.random.randint(key, (), 0, symops.shape[2]) # randomly sample an operation
    return project_x(G, W, X, idx)

def project_x(g, w, x, idx):
    '''
    apply the randomly sampled Wyckoff symmetry op to sampled fc, which 
    should be (or close to) the first WP
    '''
    op = symops[g-1, w, idx].reshape(3, 4)
    affine_point = jnp.array([*x, 1]) # (4, )
    x = jnp.dot(op, affine_point)  # (3, )
    x -= jnp.floor(x)
    return x 
