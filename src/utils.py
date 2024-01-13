import jax
import jax.numpy as jnp
import pandas as pd
from pymatgen.core import Structure

def shuffle(key, data):
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



