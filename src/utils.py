import jax
import jax.numpy as jnp
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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
    L, X, A, M = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return L[idx], X[idx], A[idx], M[idx]

def LXAM_from_structures(structures, atom_types, mult_types, n_max, dim):
    G = [] 
    L = [] # abc alpha beta gamma
    X = [] # 
    A = [] # atom 0 for placeholder
    M = []
    for i, structure in enumerate(structures):
        analyzer = SpacegroupAnalyzer(structure)
        symmetrized_structure = analyzer.get_symmetrized_structure()
        #print (analyzer.get_space_group_number(), symmetrized_structure)

        G.append ([analyzer.get_space_group_number()])
        L.append (structure.lattice.abc+ structure.lattice.angles)
        num_sites = len(symmetrized_structure.equivalent_sites)
        frac_coords = jnp.array([site[0].frac_coords for site in 
                                symmetrized_structure.equivalent_sites]).reshape(num_sites, dim)
        frac_coords = jnp.concatenate([frac_coords, 
                                       jnp.full((n_max - num_sites, dim), 1e10)], 
                                       axis = 0)
        X.append (frac_coords)  

        A.append ([site[0].specie.number for site in symmetrized_structure.equivalent_sites] 
                 + [0] * (n_max - num_sites))
        M.append ([len(site) for site in symmetrized_structure.equivalent_sites] 
                  +[0] * (n_max - num_sites))
    
    G = jnp.array(G)
    G = jax.nn.one_hot(G, 230).reshape(-1, 230)
    L = jnp.array(L).reshape(-1, 6)
    L = jnp.concatenate([G, L], axis=-1) # (-1, 236)

    X = jnp.array(X).reshape(-1, n_max, dim)
    A = jnp.array(A).reshape(-1, n_max)
    assert (atom_types > jnp.max(A))
    A = jax.nn.one_hot(A, atom_types) # (-1, n_max, atom_types)
    M = jnp.array(M).reshape(-1, n_max)
    assert (mult_types > jnp.max(M))
    M = jax.nn.one_hot(M, mult_types) # (-1, n_max, mult_types)

    return L, X, A, M
    
def LXAM_from_file(csv_file, atom_types, mult_types, n_max, dim):
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']
    structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]
    L, X, A, M = LXAM_from_structures(structures, atom_types, mult_types, n_max, dim)
    return L, X, A, M

if __name__=='__main__':
    atom_types = 118
    mult_types = 5
    n_max = 5
    dim = 3

    csv_file = '/home/wanglei/cdvae/data/perov_5/train.csv'
    #csv_file = 'mini.csv'
    L, X, A, M = LXAM_from_file(csv_file, atom_types, mult_types, n_max, dim)

    print (L.shape)
    print (X.shape)
    print (A.shape)
    print (M.shape)

    print (L)
    print (X)
    print (jnp.argmax(A, axis=2))

    print (jnp.count_nonzero(jnp.argmax(A, axis=2), axis=1)) # number of inequavlent atoms 
    print (jnp.argmax(M, axis=2)) # total number of atoms

