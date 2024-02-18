import jax.numpy as jnp

def test_mask():
    W = jnp.array([1, 2, 3, 0, 0, 0])
    n = W.shape[0]
    atom_types = 8
    mask = jnp.concatenate(
                [(W>0).reshape(n, 1), 
                 jnp.concatenate([jnp.array([0]), (W==0)[:-1]]).reshape(n, 1).repeat(atom_types-1, axis=1) 
                ], axis = 1 )  
    
    assert mask.shape == (n, atom_types)
    print (mask)
    '''
    [[1 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 1 1 1 1 1 1 1]
     [0 1 1 1 1 1 1 1]]
    '''

test_mask()  
