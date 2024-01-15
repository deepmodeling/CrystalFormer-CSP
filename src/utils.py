import jax
import jax.numpy as jnp
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from functools import partial

def shuffle(key, data):
    '''
    shuffle data along batch dimension
    '''
    G, L, X, A, M = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], X[idx], A[idx], M[idx]

def GLXAM_from_structures(structures, atom_types, mult_types, n_max, dim):
    G = [] # space group
    L = [] # abc alpha beta gamma
    X = [] # fractional coordinate 
    A = [] # atom type 0 for placeholder
    M = [] # multiplicity
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

        #print (analyzer.get_space_group_number(), [site[0].specie.number for site in symmetrized_structure.equivalent_sites])
        A.append ([site[0].specie.number for site in symmetrized_structure.equivalent_sites] 
                 + [0] * (n_max - num_sites))
        M.append ([len(site) for site in symmetrized_structure.equivalent_sites] 
                  +[0] * (n_max - num_sites))
    
    G = jnp.array(G)
    G = jax.nn.one_hot(G, 230).reshape(-1, 230)
    L = jnp.array(L).reshape(-1, 6)

    X = jnp.array(X).reshape(-1, n_max, dim)
    A = jnp.array(A).reshape(-1, n_max)
    assert (atom_types > jnp.max(A))
    A = jax.nn.one_hot(A, atom_types) # (-1, n_max, atom_types)
    M = jnp.array(M).reshape(-1, n_max)
    assert (mult_types > jnp.max(M))
    M = jax.nn.one_hot(M, mult_types) # (-1, n_max, mult_types)

    return G, L, X, A, M
    
def GLXAM_from_file(csv_file, atom_types, mult_types, n_max, dim):
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']
    structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]
    G, L, X, A, M = GLXAM_from_structures(structures, atom_types, mult_types, n_max, dim)
    return G, L, X, A, M

if __name__=='__main__':
    atom_types = 118
    mult_types = 5
    n_max = 5
    dim = 3

    #csv_file = '/home/wanglei/cdvae/data/perov_5/train.csv'
    csv_file = 'mini.csv'
    G, L, X, A, M = GLXAM_from_file(csv_file, atom_types, mult_types, n_max, dim)

    print (G.shape)
    print (L.shape)
    print (X.shape)
    print (A.shape)
    print (M.shape)
    
    print (jnp.argmax(G, axis=1))
    print (jnp.argmax(M, axis=2)) 
    print (L)

    print (X)
    print (jnp.argmax(A, axis=2))

    print (jnp.count_nonzero(jnp.argmax(A, axis=2), axis=1)) # number of inequavlent atoms 
    print (jnp.argmax(M, axis=2)) # multplicities  
    print (jnp.sum(jnp.argmax(M, axis=2), axis=1))  # total number of atoms

