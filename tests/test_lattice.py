from config import *

from src.lattice import make_spacegroup_lattice

def test_make_spacegroup_lattice():
    key = jax.random.PRNGKey(42)

    G = jnp.arange(230) + 1
    L = jax.random.uniform(key, (6,))
    L = L.reshape([1, 6]).repeat(230, axis=0)

    lattice = jax.jit(jax.vmap(make_spacegroup_lattice))(G, L)
    print (lattice)    
    
    a, b, c, alpha, beta, gamma = lattice[99-1] 
    assert (alpha==beta==gamma==90)
    assert (a==b)

test_make_spacegroup_lattice()

