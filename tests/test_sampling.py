from config import *
from wyckoff import symops

def test_symops():
    from sample import project_x
    from wyckoff import wmax_table, mult_table

    def map_fn(g, w, x, idx):
        w_max = wmax_table[g-1].item()
        m_max = mult_table[g-1, w_max].item()
        x = project_x(g, w_max, m_max, w[None, ...], x[None, ...], idx[None, ...])
        return x[0]
    
    # these two tests shows that depending on the z coordinate (which is supposed to be rationals)
    # the WP can be recoginized differently, resulting different x
    # this motivate that we either predict idx in [1, m], or we predict all fc once there is a continuous dof
    g = 167 
    w = jnp.array(5)
    idx = jnp.array(5)
    x = jnp.array([0.123, 0.123, 0.75])
    y = map_fn(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([0.123, 0.123, 0.75]))

    x = jnp.array([0.123, 0.123, 0.25])
    y = map_fn(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([0.877, 0.877, 0.75]))

    g = 225
    w = jnp.array(5)
    x = jnp.array([0., 0., 0.7334])

    idx = jnp.array(0)
    y = map_fn(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([0.7334, 0., 0.]))

    idx = jnp.array(3)
    y = map_fn(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([0., 1.0-0.7334, 0.]))
    
    g = 166 
    w = jnp.array(8)
    x = jnp.array([0.1, 0.2, 0.3])

    idx = jnp.array(5)
    y = map_fn(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([1-0.1, 1-0.2, 1-0.3]))

test_symops()
