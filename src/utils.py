import jax
import jax.numpy as jnp
import pandas as pd
from pymatgen.core import Structure

from functools import partial

@partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0,0))
def random_permute_atoms(key, X, A):
    '''
    randomly permute atoms 
    '''
    idx = jax.random.permutation(key, jnp.arange(len(A)))
    A = A[idx]
    X = X[idx]
    
    #move nonzero elements to the front 
    non_zero_mask = (A!= 0)
    idx = jnp.argsort(non_zero_mask)[::-1]
    X, A = X[idx], A[idx]

    X -= X[0, None] # shift the first atom to 000
    return X, A

def shuffle(key, data):
    '''
    shuffle data along batch dimension
    '''
    L, X, A = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return L[idx], X[idx], A[idx]

def LXA_from_structures(structures, atom_types, n_max, dim):
    L = [] # abc alpha beta gamma
    X = [] # 
    A = [] # 1: C atom 0: placeholder
    for i, structure in enumerate(structures):
        L.append (structure.lattice.abc+ structure.lattice.angles)
        frac_coords = jnp.array([site.frac_coords for site in structure]).reshape(structure.num_sites, dim)
        frac_coords = jnp.concatenate([frac_coords, 
                                       jnp.full((n_max - structure.num_sites, dim), 1e10)], 
                                  axis = 0)
        X.append (frac_coords)  
        A.append ([1.]*structure.num_sites + [0.] * (n_max - structure.num_sites))

    L = jnp.array(L).reshape(-1, 6)
    X = jnp.array(X).reshape(-1, n_max, dim)
    A = jnp.array(A).reshape(-1, n_max)
    
    print ("shift the first atom to 000")
    X -= X[:, 0, None] # shift the first atom to 000
    return L, X, A
    
def LXA_from_file(csv_file, atom_types, n_max, dim):
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']
    structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]
    L, X, A = LXA_from_structures(structures, atom_types, n_max, dim)
    return L, X, A

if __name__=='__main__':
    atom_types = 2 
    n_max = 24 
    dim = 3

    csv_file = '/home/wanglei/cdvae/data/carbon_24/train.csv'
    L, X, A = LXA_from_file(csv_file, atom_types, n_max, dim)

    print (L.shape)
    print (X.shape)
    print (A.shape)
    print (L[:5])
    print (X[:5])
    print (A[:5])

    
    key = jax.random.PRNGKey(42)
    X, A = random_permute_atoms(key, X[:5], A[:5])

    print (X)
    print (A)
