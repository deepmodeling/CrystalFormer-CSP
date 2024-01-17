import jax
import jax.numpy as jnp 
import haiku as hk 

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
        output = mlp(G) #TODO conditioned on G output truely independent ones 
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
    G = jax.nn.one_hot(G, 230).reshape(230,)
    print (G.shape)

    print (lattice_mlp(params, G))

