import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk
from jax import random
import numpy as np 

class NVPCouplingLayer(hk.Module):
    def __init__(self, map_s, map_t, b, name=None):
        super().__init__(name=name)
        self.map_s = map_s
        self.map_t = map_t
        self.b = b 

    def __call__(self, x, is_forward):
        s, t = self.map_s(self.b * x), self.map_t(self.b * x)
        if is_forward:
            y = self.b * x + (1 - self.b) * (jnp.exp(s) * x + t)
            logjac = ((1 - self.b) * s).sum()
        else:
            y = self.b * x + (1 - self.b) * (jnp.exp(-s) * (x - t))
            logjac = -((1 - self.b) * s).sum()
        return y, logjac

class NVPNet(hk.Module):
    def __init__(self, dim, hdim, depth, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.depth = depth
        self.hdim = hdim

    def __call__(self, x, is_forward):
        logjac = 0.0
        layers = []

        mask = jnp.where(jnp.arange(self.dim)< self.dim//2, 0, 1)
        for d in range(self.depth):

            map_s = hk.Sequential([
                hk.Linear(self.hdim), jax.nn.elu,
                hk.Linear(self.hdim), jax.nn.elu,
                hk.Linear(self.dim, 
                          w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                          b_init=jnp.zeros), 
                jnp.tanh
            ])

            map_t = hk.Sequential([
                hk.Linear(self.hdim), jax.nn.elu,
                hk.Linear(self.hdim), jax.nn.elu,
                hk.Linear(self.dim, 
                          w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                          b_init=jnp.zeros)
            ])

            layer = NVPCouplingLayer(map_s, map_t, mask)
            layers.append(layer)
            mask = jnp.logical_not(mask)

        for layer in (layers if is_forward else reversed(layers)):
            x, ljac = layer(x, is_forward)
            logjac += ljac

        return x, logjac

def make_flow(rng, dim, hdim, depth):

    @hk.without_apply_rng
    @hk.transform
    def net_fn(x, is_forward):
        return NVPNet(dim, hdim, depth)(x, is_forward)
    params = net_fn.init(rng, jnp.zeros((dim, )), True)

    def logprob(params, x):
        z, logjac = net_fn.apply(params, x, False)
        return -0.5 * jnp.sum(z**2) + logjac

    def sample(rng, params, batch_size):
        z = random.normal(rng, (batch_size, dim))
        x, logp = jax.vmap(net_fn.apply, (None, 0, None))(params, z, True)
        return x, -0.5 * jnp.sum(z**2, axis=1) - logp

    return params, logprob, sample

# Example usage
def main():
    dim = 6
    hdim = 32
    depth = 8

    rng = jax.random.PRNGKey(42)

    @hk.without_apply_rng
    @hk.transform
    def net_fn(x, is_forward):
        return NVPNet(dim, hdim, depth)(x, is_forward)
    params = net_fn.init(rng, jnp.zeros((dim, )), True)

    z = random.normal(rng, (dim, ))
    x, logjac = net_fn.apply(params, z, True)
    z_infer, logjac_infer = net_fn.apply(params, x, False)

    # Testing
    np.testing.assert_array_almost_equal(z_infer, z, decimal=5)
    np.testing.assert_array_almost_equal(logjac, -logjac_infer, decimal=5)

if __name__ == '__main__':
    main()
    
    dim = 6 
    hdim = 32 
    depth = 8

    rng = jax.random.PRNGKey(42)
    params, logprob_fn, sample_fn = make_flow(rng, dim, hdim, depth)
    
    x = random.normal(rng, (dim, ))
    print (logprob_fn(params, x))

    x, logp_x = sample_fn(rng, params, 10)
    
    print (logp_x)
    print (jax.vmap(logprob_fn, (None, 0))(params, x))

