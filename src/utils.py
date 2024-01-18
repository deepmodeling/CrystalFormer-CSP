import jax
import jax.numpy as jnp
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from functools import partial

mult_list = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192] # possible multiplicites 
mult_dict = {value: index for index, value in enumerate(mult_list)}

@partial(jax.vmap, in_axes=(0, None), out_axes=0) # n 
def to_A_M(AM, atom_types):
    AM = jnp.argmax(AM, axis=-1)
    A = jnp.where(AM==0, jnp.zeros_like(AM), (AM-1)%(atom_types-1)+1)
    M = jnp.where(AM==0, jnp.zeros_like(AM), (AM-1)//(atom_types-1)+1)
    return A, M    

def shuffle(key, data):
    '''
    shuffle data along batch dimension
    '''
    G, L, X, AM = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], X[idx], AM[idx]

def GLXAM_from_structures(structures, atom_types, mult_types, n_max, dim):
    G = [] # space group
    L = [] # abc alpha beta gamma
    X = [] # fractional coordinate 
    AM = [] # atom type and multiplicity; 0 for placeholder
    for i, structure in enumerate(structures):
        analyzer = SpacegroupAnalyzer(structure)
        symmetrized_structure = analyzer.get_symmetrized_structure()
        #print (analyzer.get_space_group_number(), symmetrized_structure)
        
        #if analyzer.get_space_group_number() in Ga:
        #    Ga[analyzer.get_space_group_number()].append(structure.lattice.abc[0])
        #else:
        #    Ga[analyzer.get_space_group_number()] = [structure.lattice.abc[0]]

        #Ga.append(symmetrized_structure.equivalent_sites[0][0].specie.number)

        #print (structure.lattice.abc)
        G.append ([analyzer.get_space_group_number()])
        L.append (structure.lattice.abc + structure.lattice.angles)
        num_sites = len(symmetrized_structure.equivalent_sites)
        assert (n_max >= num_sites)
        frac_coords = jnp.array([site[0].frac_coords for site in 
                                symmetrized_structure.equivalent_sites]).reshape(num_sites, dim)
        frac_coords = jnp.concatenate([frac_coords, 
                                       jnp.full((n_max - num_sites, dim), 1e10)], 
                                       axis = 0)
        X.append (frac_coords)  
    
        #print (analyzer.get_space_group_number(), [site[0].specie.number for site in symmetrized_structure.equivalent_sites])
        
        am = []
        for site in symmetrized_structure.equivalent_sites:
            a = site[0].specie.number # element number 
            m = len(site)             # multiplicity
            assert (a < atom_types)
            assert (mult_dict[m] < mult_types)
            #print ('xxx', a, m)
            am.append( (mult_dict[m]-1) * (atom_types-1)+ (a-1) +1 )
        AM.append( am + [0] * (n_max - num_sites) )
   
    G = jnp.array(G)
    G = jax.nn.one_hot(G-1, 230).reshape(-1, 230) # G-1 to shift 1-230 to 0-229
    L = jnp.array(L).reshape(-1, 6)

    X = jnp.array(X).reshape(-1, n_max, dim)
    
    AM = jnp.array(AM).reshape(-1, n_max)
    am_types = (atom_types -1)*(mult_types -1) + 1
    AM = jax.nn.one_hot(AM, am_types) # (-1, n_max, am_types)
    return G, L, X, AM
    
def GLXAM_from_file(csv_file, atom_types, mult_types, n_max, dim):
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']
    structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]
    G, L, X, AM = GLXAM_from_structures(structures, atom_types, mult_types, n_max, dim)
    return G, L, X, AM

if __name__=='__main__':
    atom_types = 118
    mult_types = 6
    n_max = 5
    dim = 3

    mult_table = jnp.array(mult_list[:mult_types])
    #csv_file = '/home/wanglei/cdvae/data/perov_5/val.csv'
    csv_file = '../data/mini.csv'
    #csv_file = '/home/wanglei/cdvae/data/mp_20/train.csv'
    G, L, X, AM = GLXAM_from_file(csv_file, atom_types, mult_types, n_max, dim)
    
    print (G.shape)
    print (L.shape)
    print (X.shape)
    print (AM.shape)
    
    print (jnp.argmax(G, axis=1))
    print (L)
    print (X)

    AM_flat = jnp.argmax(AM, axis=-1)
    print (AM_flat)

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    A, M = jax.vmap(to_A_M, (0, None))(AM, atom_types)
    print (A)
    print (M) 

    print (M)
    print (mult_table[M]) # the actual degeneracy
    N = mult_table[M].sum(axis=1)
    print (N)

