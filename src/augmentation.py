import jax
import jax.numpy as jnp
from functools import partial

from wyckoff import symops


def perm_augmentation(key, A, W, X):
    '''
    randomly permute fractional coordinate of atoms with the same wyckoff letter
    A: (n, )
    W: (n, )
    X: (n, dim)
    '''
    n = W.shape[0]
    temp = jnp.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort
    idx_perm = jax.random.permutation(key, jnp.arange(n))
    temp = temp[idx_perm]
    idx_sort = jnp.argsort(temp)
    idx = idx_perm[idx_sort]

    # one should still jnp.allclose(W, W[idx])
    A = A[idx]
    X = X[idx]

    return A, X

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
    return project_xyz(G, W, X, idx)

def project_xyz(g, w, x, idx):
    '''
    apply the randomly sampled Wyckoff symmetry op to sampled fc, which 
    should be (or close to) the first WP
    '''
    op = symops[g-1, w, idx].reshape(3, 4)
    affine_point = jnp.array([*x, 1]) # (4, )
    x = jnp.dot(op, affine_point)  # (3, )
    x -= jnp.floor(x)
    return x 
