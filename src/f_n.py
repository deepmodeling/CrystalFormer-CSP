import jax.numpy as jnp
f_1 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_2 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_3 = [
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_4 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_5 = [
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_6 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_7 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_8 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_9 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_10 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_11 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_12 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_13 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_14 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_15 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_16 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_17 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_18 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_19 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_20 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_21 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_22 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_23 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_24 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_25 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_26 = [
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_27 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_28 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_29 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_30 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_31 = [
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_32 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_33 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_34 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_35 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_36 = [
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_37 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_38 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_39 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_40 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_41 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_42 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_43 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_44 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_45 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_46 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_47 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_48 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_49 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_50 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_51 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_52 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_53 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_54 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_55 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_56 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_57 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_58 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_59 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, z]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_60 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_61 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_62 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_63 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_64 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_65 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_66 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_67 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_68 = [
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_69 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_70 = [
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 5/8]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([1/8, y, 1/8]),
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_71 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_72 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_73 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_74 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_75 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_76 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_77 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_78 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_79 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_80 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_81 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_82 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_83 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_84 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_85 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_86 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_87 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_88 = [
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/8]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 5/8]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_89 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_90 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_91 = [
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 3/8]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_92 = [
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_93 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_94 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_95 = [
          lambda x,y,z : 1.0*jnp.array([0, y, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 5/8]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_96 = [
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_97 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_98 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/8]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_99 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_100 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_101 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_102 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_103 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_104 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_105 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_106 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_107 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_108 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_109 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_110 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_111 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_112 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_113 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_114 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_115 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_116 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_117 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_118 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x+1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_119 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_120 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_121 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_122 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/8]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_123 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_124 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_125 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_126 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 3/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_127 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_128 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_129 = [
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_130 = [
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_131 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_132 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_133 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_134 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_135 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_136 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_137 = [
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_138 = [
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 0]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_139 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_140 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_141 = [
          lambda x,y,z : 1.0*jnp.array([0, 3/4, 1/8]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 3/8]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/4, 7/8]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_142 = [
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 3/8]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/8]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x+1/4, 1/8]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_143 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_144 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_145 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_146 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_147 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_148 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_149 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 0]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_150 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_151 = [
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/3]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 5/6]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_152 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/3]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 5/6]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_153 = [
          lambda x,y,z : 1.0*jnp.array([x, -x, 2/3]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/6]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_154 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, 2/3]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/6]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_155 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_156 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_157 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_158 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_159 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_160 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_161 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_162 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_163 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_164 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_165 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_166 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_167 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_168 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_169 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_170 = [
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_171 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_172 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_173 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_174 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 0]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_175 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_176 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_177 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_178 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_179 = [
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_180 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_181 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_182 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_183 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_184 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_185 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_186 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_187 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 0]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, -x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_188 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 0]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, -x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_189 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_190 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([2/3, 1/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_191 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_192 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_193 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_194 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, z]),
          lambda x,y,z : 1.0*jnp.array([1/3, 2/3, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 2*x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_195 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_196 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 3/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_197 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_198 = [
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_199 = [
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_200 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_201 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 3/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_202 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_203 = [
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([5/8, 5/8, 5/8]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_204 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_205 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_206 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_207 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, y]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, y]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_208 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 3/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, -y+1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, y+1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_209 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([0, y, y]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, y]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_210 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([5/8, 5/8, 5/8]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/8, y, -y+1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_211 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, y]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, -y+1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_212 = [
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([5/8, 5/8, 5/8]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([1/8, y, -y+1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_213 = [
          lambda x,y,z : 1.0*jnp.array([3/8, 3/8, 3/8]),
          lambda x,y,z : 1.0*jnp.array([7/8, 7/8, 7/8]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([1/8, y, y+1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_214 = [
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([7/8, 7/8, 7/8]),
          lambda x,y,z : 1.0*jnp.array([1/8, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([5/8, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/8, y, y+1/4]),
          lambda x,y,z : 1.0*jnp.array([1/8, y, -y+1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_215 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_216 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 3/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_217 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_218 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_219 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_220 = [
          lambda x,y,z : 1.0*jnp.array([3/8, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([7/8, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_221 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([0, y, y]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, y]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_222 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 3/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 3/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, y]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_223 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 1/2, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, y+1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_224 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 3/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 3/4]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, y+1/2]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, -y]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_225 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, y, y]),
          lambda x,y,z : 1.0*jnp.array([1/2, y, y]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_226 = [
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, y]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_227 = [
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([3/8, 3/8, 3/8]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/2, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([0, y, -y]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_228 = [
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([7/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, -y]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_229 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([0, 1/2, 1/2]),
          lambda x,y,z : 1.0*jnp.array([1/4, 1/4, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/4, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, y]),
          lambda x,y,z : 1.0*jnp.array([1/4, y, -y+1/2]),
          lambda x,y,z : 1.0*jnp.array([0, y, z]),
          lambda x,y,z : 1.0*jnp.array([x, x, z]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]

f_230 = [
          lambda x,y,z : 1.0*jnp.array([0, 0, 0]),
          lambda x,y,z : 1.0*jnp.array([1/8, 1/8, 1/8]),
          lambda x,y,z : 1.0*jnp.array([1/8, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([3/8, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([x, x, x]),
          lambda x,y,z : 1.0*jnp.array([x, 0, 1/4]),
          lambda x,y,z : 1.0*jnp.array([1/8, y, -y+1/4]),
          lambda x,y,z : 1.0*jnp.array([x, y, z]),
       ]


fn_dict = {1 : f_1,
2 : f_2,
3 : f_3,
4 : f_4,
5 : f_5,
6 : f_6,
7 : f_7,
8 : f_8,
9 : f_9,
10 : f_10,
11 : f_11,
12 : f_12,
13 : f_13,
14 : f_14,
15 : f_15,
16 : f_16,
17 : f_17,
18 : f_18,
19 : f_19,
20 : f_20,
21 : f_21,
22 : f_22,
23 : f_23,
24 : f_24,
25 : f_25,
26 : f_26,
27 : f_27,
28 : f_28,
29 : f_29,
30 : f_30,
31 : f_31,
32 : f_32,
33 : f_33,
34 : f_34,
35 : f_35,
36 : f_36,
37 : f_37,
38 : f_38,
39 : f_39,
40 : f_40,
41 : f_41,
42 : f_42,
43 : f_43,
44 : f_44,
45 : f_45,
46 : f_46,
47 : f_47,
48 : f_48,
49 : f_49,
50 : f_50,
51 : f_51,
52 : f_52,
53 : f_53,
54 : f_54,
55 : f_55,
56 : f_56,
57 : f_57,
58 : f_58,
59 : f_59,
60 : f_60,
61 : f_61,
62 : f_62,
63 : f_63,
64 : f_64,
65 : f_65,
66 : f_66,
67 : f_67,
68 : f_68,
69 : f_69,
70 : f_70,
71 : f_71,
72 : f_72,
73 : f_73,
74 : f_74,
75 : f_75,
76 : f_76,
77 : f_77,
78 : f_78,
79 : f_79,
80 : f_80,
81 : f_81,
82 : f_82,
83 : f_83,
84 : f_84,
85 : f_85,
86 : f_86,
87 : f_87,
88 : f_88,
89 : f_89,
90 : f_90,
91 : f_91,
92 : f_92,
93 : f_93,
94 : f_94,
95 : f_95,
96 : f_96,
97 : f_97,
98 : f_98,
99 : f_99,
100 : f_100,
101 : f_101,
102 : f_102,
103 : f_103,
104 : f_104,
105 : f_105,
106 : f_106,
107 : f_107,
108 : f_108,
109 : f_109,
110 : f_110,
111 : f_111,
112 : f_112,
113 : f_113,
114 : f_114,
115 : f_115,
116 : f_116,
117 : f_117,
118 : f_118,
119 : f_119,
120 : f_120,
121 : f_121,
122 : f_122,
123 : f_123,
124 : f_124,
125 : f_125,
126 : f_126,
127 : f_127,
128 : f_128,
129 : f_129,
130 : f_130,
131 : f_131,
132 : f_132,
133 : f_133,
134 : f_134,
135 : f_135,
136 : f_136,
137 : f_137,
138 : f_138,
139 : f_139,
140 : f_140,
141 : f_141,
142 : f_142,
143 : f_143,
144 : f_144,
145 : f_145,
146 : f_146,
147 : f_147,
148 : f_148,
149 : f_149,
150 : f_150,
151 : f_151,
152 : f_152,
153 : f_153,
154 : f_154,
155 : f_155,
156 : f_156,
157 : f_157,
158 : f_158,
159 : f_159,
160 : f_160,
161 : f_161,
162 : f_162,
163 : f_163,
164 : f_164,
165 : f_165,
166 : f_166,
167 : f_167,
168 : f_168,
169 : f_169,
170 : f_170,
171 : f_171,
172 : f_172,
173 : f_173,
174 : f_174,
175 : f_175,
176 : f_176,
177 : f_177,
178 : f_178,
179 : f_179,
180 : f_180,
181 : f_181,
182 : f_182,
183 : f_183,
184 : f_184,
185 : f_185,
186 : f_186,
187 : f_187,
188 : f_188,
189 : f_189,
190 : f_190,
191 : f_191,
192 : f_192,
193 : f_193,
194 : f_194,
195 : f_195,
196 : f_196,
197 : f_197,
198 : f_198,
199 : f_199,
200 : f_200,
201 : f_201,
202 : f_202,
203 : f_203,
204 : f_204,
205 : f_205,
206 : f_206,
207 : f_207,
208 : f_208,
209 : f_209,
210 : f_210,
211 : f_211,
212 : f_212,
213 : f_213,
214 : f_214,
215 : f_215,
216 : f_216,
217 : f_217,
218 : f_218,
219 : f_219,
220 : f_220,
221 : f_221,
222 : f_222,
223 : f_223,
224 : f_224,
225 : f_225,
226 : f_226,
227 : f_227,
228 : f_228,
229 : f_229,
230 : f_230,
}


