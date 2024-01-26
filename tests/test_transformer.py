from config import * 

from utils import GLXAM_from_file
from transformer import make_transformer

def test_autoregressive():
    atom_types = 119
    mult_types = 4
    Nf = 5
    n_max = 5
    K = 8 
    dim = 3

    csv_file = '../data/mini.csv'
    G, L, X, AM = GLXAM_from_file(csv_file, atom_types, mult_types, n_max, dim)

    am_types = AM.shape[-1]

    key = jax.random.PRNGKey(42)
    params, transformer = make_transformer(key, Nf, K, n_max, dim, 128, 4, 4, 8, 16, atom_types, mult_types) 

    def test_fn(G, AM, X):
        output = transformer(params, G, X, AM) # (2*n+2, am_types)
        output = output.reshape(n_max+1, 2, am_types)
        return (output[:, 0, :]).sum(axis=-1), (output[:, 1, :]).sum(axis=-1)
    
    print (X.shape, AM.shape)

    jac_am_am = jax.jacfwd(lambda G, AM, X: test_fn(G, AM, X)[0], argnums = 1)(G[0], AM[0], X[0])
    jac_am_x = jax.jacfwd(lambda G, AM, X: test_fn(G, AM, X)[0], argnums = 2)(G[0], AM[0], X[0])
    jac_x_am = jax.jacfwd(lambda G, AM, X: test_fn(G, AM, X)[1], argnums = 1)(G[0], AM[0], X[0])
    jac_x_x = jax.jacfwd(lambda G, AM, X: test_fn(G, AM, X)[1], argnums = 2)(G[0], AM[0], X[0])

    def print_dependencey(jac):
        dependencey = jnp.linalg.norm(jac, axis=-1)
        for row in (dependencey != 0.).astype(int):
            print(" ".join(str(val) for val in row))

    print ('jac_am_am')
    print_dependencey(jac_am_am)
    print ('jac_am_x')
    print_dependencey(jac_am_x)
    print ('jac_x_am')
    print_dependencey(jac_x_am)
    print ('jac_x_x')
    print_dependencey(jac_x_x)

def test_perm():

    key = jax.random.PRNGKey(42)

    W = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0])
    n = len(W)
    key = jax.random.PRNGKey(42)

    temp = jnp.where(W>0, W, 9999)
    idx_perm = jax.random.permutation(key, jnp.arange(n))
    temp = temp[idx_perm]
    idx_sort = jnp.argsort(temp)
    idx = idx_perm[idx_sort]

    print (W)
    print (idx)
    print (W[idx])
    assert jnp.allclose(W, W[idx])

test_perm()
test_autoregressive()
