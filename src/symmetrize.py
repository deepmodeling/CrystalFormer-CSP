#see https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-wp-list?gnum=1
import jax
import jax.numpy as jnp 
from functools import partial

@partial(jax.jit, static_argnums=0)
def apply_wyckoff_condition(g, m, xyz):

    f_25 = [lambda x,y,z : jnp.array([0.0, 0.0, z]), 
            lambda x,y,z : jnp.array([0.0, 0.5, z]), 
            lambda x,y,z : jnp.array([0.5, 0.0, z]), 
            lambda x,y,z : jnp.array([0.5, 0.5, z]), 
            lambda x,y,z : jnp.array([x, 0.0 , z]), 
            lambda x,y,z : jnp.array([x, 0.5, z]), 
            lambda x,y,z : jnp.array([0.5, y, z])
            ] 

    f_47 = [lambda x,y,z : jnp.array([0.0, 0.0, 0.0]), 
            lambda x,y,z : jnp.array([0.5, 0.0, 0.0]), 
            lambda x,y,z : jnp.array([0.5, 0.0, 0.5]), 
            lambda x,y,z : jnp.array([0.5, 0.0, 0.5]), 
            lambda x,y,z : jnp.array([0.0, 0.5, 0.0]), 
            lambda x,y,z : jnp.array([0.5, 0.5, 0.0]), 
            lambda x,y,z : jnp.array([0.0, 0.5, 0.5]), 
            lambda x,y,z : jnp.array([0.5, 0.5, 0.5]), 
            lambda x,y,z : jnp.array([x, 0.0, 0.0]), 
            lambda x,y,z : jnp.array([x, 0.0, 0.5]), 
            lambda x,y,z : jnp.array([x, 0.5, 0.0]), 
            ]

    f_99 = [lambda x,y,z : jnp.array([0.0, 0.0, z]), 
            lambda x,y,z : jnp.array([0.5, 0.5, z]), 
            lambda x,y,z : jnp.array([0.5, 0.0, z]), 
            lambda x,y,z : jnp.array([x, x, z]), 
            ]

    f_123 = [lambda x,y,z : jnp.array([0.0, 0.0, 0.0]),
             lambda x,y,z : jnp.array([0.0, 0.0, 0.5]),
             lambda x,y,z : jnp.array([0.5, 0.5, 0.0]),
             lambda x,y,z : jnp.array([0.5, 0.5, 0.5]),
             lambda x,y,z : jnp.array([0.0, 0.5, 0.5]),
             lambda x,y,z : jnp.array([0.0, 0.5, 0.0]),
             lambda x,y,z : jnp.array([0.0, 0.0, z]),
             lambda x,y,z : jnp.array([0.5, 0.5, z]),
            ]

    f_221 = [lambda x,y,z : jnp.array([0.0, 0.0, 0.0]), 
             lambda x,y,z : jnp.array([0.5, 0.5, 0.5]), 
             lambda x,y,z : jnp.array([0.0, 0.5, 0.5]), 
             lambda x,y,z : jnp.array([0.5, 0.0, 0.0]), 
             lambda x,y,z : jnp.array([x, 0.0, 0.0]), 
             lambda x,y,z : jnp.array([x, 0.5, 0.5])
            ]

    fn_dict = {
        25: f_25, 
        47: f_47, 
        99: f_99, 
        123: f_123, 
        221: f_221, 
    }

    x, y, z = xyz[0], xyz[1], xyz[2]
    xyz = jax.lax.switch(m.sum(), fn_dict[g], x,y,z) # sum to get scalar

    return xyz

if __name__=='__main__':
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    xyz = np.array([0.12, 0.23, 0.45])
    g = 25 
    m = jnp.array([0, 1, 2])
    xyz = jax.vmap(apply_wyckoff_condition, (None, 0, None))(g, m, xyz)
    print (xyz)


