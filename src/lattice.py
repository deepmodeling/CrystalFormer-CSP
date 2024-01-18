import jax
import jax.numpy as jnp 
import haiku as hk 

def make_spacegroup_mask(spacegroup):
    '''
    return mask for independent lattice params 
    '''

    mask = jnp.array([1, 1, 1, 1, 1, 1])

    mask = jnp.where(spacegroup <= 2,   mask, jnp.array([1, 1, 1, 0, 1, 0]))
    mask = jnp.where(spacegroup <= 15,  mask, jnp.array([1, 1, 1, 0, 0, 0]))
    mask = jnp.where(spacegroup <= 74,  mask, jnp.array([1, 0, 1, 0, 0, 0]))
    mask = jnp.where(spacegroup <= 142, mask, jnp.array([1, 0, 1, 0, 0, 0]))
    mask = jnp.where(spacegroup <= 194, mask, jnp.array([1, 0, 0, 0, 0, 0]))
    
    return mask

def make_spacegroup_lattice(spacegroup, lattice):
    '''
    place lattice params into lattice according to the space group 
    '''

    a, b, c, alpha, beta, gamma = lattice

    L = lattice
    L = jnp.where(spacegroup <= 2,   L, jnp.array([a, b, c, 90., beta, 90.]))
    L = jnp.where(spacegroup <= 15,  L, jnp.array([a, b, c, 90., 90., 90.]))
    L = jnp.where(spacegroup <= 74,  L, jnp.array([a, a, c, 90., 90., 90.]))
    L = jnp.where(spacegroup <= 142, L, jnp.array([a, a, c, 90., 90., 120.]))
    L = jnp.where(spacegroup <= 194, L, jnp.array([a, a, a, 90., 90., 90.]))

    return L

if __name__ == '__main__':
    
    G = jnp.array([25, 99, 221])
    mask = jax.vmap(make_spacegroup_mask)(G)
    print (mask)

    key = jax.random.PRNGKey(42)
    lattice = jax.random.normal(key, (6,))

    L = jax.vmap(make_spacegroup_lattice, (0, None))(G, lattice)

    print (L)

