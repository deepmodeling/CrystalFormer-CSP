import jax
import jax.numpy as jnp 
import haiku as hk 

def make_spacegroup_mask(spacegroup):
    '''
    return mask for those independent lattice params
    '''

    mask = jnp.array([1, 1, 1, 1, 1, 1])

    mask = jnp.where(spacegroup <= 2,   mask, jnp.array([1, 1, 1, 0, 1, 0]))
    mask = jnp.where(spacegroup <= 15,  mask, jnp.array([1, 1, 1, 0, 0, 0]))
    mask = jnp.where(spacegroup <= 74,  mask, jnp.array([1, 0, 1, 0, 0, 0]))
    mask = jnp.where(spacegroup <= 142, mask, jnp.array([1, 0, 1, 0, 0, 0]))
    mask = jnp.where(spacegroup <= 194, mask, jnp.array([1, 0, 0, 0, 0, 0]))
    
    return mask

def make_spacegroup_lattice(spacegroup, lattice):

    a, b, c, alpha, beta, gamma = lattice

    L = lattice
    L = jnp.where(spacegroup <= 2,   L, jnp.array([a, b, c, 90., beta, 90.]))
    L = jnp.where(spacegroup <= 15,  L, jnp.array([a, b, c, 90., 90., 90.]))
    L = jnp.where(spacegroup <= 74,  L, jnp.array([a, a, c, 90., 90., 90.]))
    L = jnp.where(spacegroup <= 142, L, jnp.array([a, a, c, 90., 90., 120.]))
    L = jnp.where(spacegroup <= 194, L, jnp.array([a, a, a, 90., 90., 90.]))

    return L

class LatticeMLP(hk.Module):
    def __init__(self, dim, hdim, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.hdim = hdim

    def __call__(self, G):
        '''
        G: (230, ) a one hot encoder of the space group number, conditioned on it we will compute the lattice parameters 
        '''
        mlp = hk.Sequential([
            hk.Linear(self.hdim), jax.nn.elu,
            hk.Linear(self.hdim), jax.nn.elu,
            hk.Linear(self.dim*2), jax.nn.softplus # lattice should be positive
        ])
        output = mlp(G) # note that we any way predict 6 outputs 
        mu, sigma = jnp.split(output, 2)
        return mu, sigma

def make_lattice_mlp(rng, dim, hdim):

    @hk.without_apply_rng
    @hk.transform
    def forward_fn(x):
        return LatticeMLP(dim, hdim)(x)
    params = forward_fn.init(rng, jnp.zeros((230, )))
    return params, forward_fn.apply

if __name__ == '__main__':
    
    dim = 6 
    hdim = 32 

    rng = jax.random.PRNGKey(42)
    params, lattice_mlp = make_lattice_mlp(rng, dim, hdim)
    
    G = jnp.array([99])
    G = jax.nn.one_hot(G-1, 230).reshape(230,)
    print (G.shape)

    print (lattice_mlp(params, G))

