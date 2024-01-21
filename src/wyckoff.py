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
    xyz = jax.lax.switch(m, fn_dict[g], x,y,z)

    return xyz


def get_wyckoff_table(g):

    if g == 25:
        fn_list = ["1a", "1b", "1c", "1d", 
                   "2e", "2f", "2g"]
    elif g == 47:
        fn_list = ["1a", "1b", "1c", "1d", "1e", "1f", "1g", "1h", 
                   "2i", "2j", "2k", "2l", "2m", "2n", "2o", "2p", "2q", "2r", "2s", "2t"
                  ]

    elif g == 99:
        fn_list = ["1a", "1b", 
                   "2c", 
                   "4d"
                   ]

    elif g == 123:
        fn_list = ["1a", "1b", "1c", "1d", 
                   "2e", "2f", "2g", "2h", 
                   ]

    elif g == 221:
        fn_list = ["1a", "1b",
                   "3c", "3d", 
                   "6e", "6f"
                   ]
    else:
        raise NotImplementedError
    
    fn_list = ["0"] + fn_list
    fn_dict = {value: index for index, value in enumerate(fn_list)}
    return fn_dict # a table that maps Wyckoff symbol to an integer index 


if __name__=='__main__':

    import numpy as np
    
    xyz = np.array([0.12, 0.23, 0.45])
    g = 25 
    m = jnp.array([0, 1, 2])
    xyz = jax.vmap(apply_wyckoff_condition, (None, 0, None))(g, m, xyz)

    print (xyz)
