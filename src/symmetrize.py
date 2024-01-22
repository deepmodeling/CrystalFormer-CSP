#see https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-wp-list?gnum=
#see https://github.com/qzhu2017/PyXtal/blob/master/pyxtal/database/wyckoff_list.csv
import jax
import jax.numpy as jnp 
from functools import partial

from f_n import *

@partial(jax.jit, static_argnums=0)
def apply_wyckoff_condition(g, m, xyz):
    
    #TODO can try nest lax.switch to support tractable g
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


