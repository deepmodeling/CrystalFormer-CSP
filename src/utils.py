import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import partial

from wyckoff import wyckoff_dict, wyckoff_table

@partial(jax.vmap, in_axes=(0, None), out_axes=0) # n 
def to_A_W(AW, atom_types):
    A = jnp.where(AW==0, jnp.zeros_like(AW), (AW-1)%(atom_types-1)+1)
    W = jnp.where(AW==0, jnp.zeros_like(AW), (AW-1)//(atom_types-1)+1)
    return A, W

def shuffle(key, data):
    '''
    shuffle data along batch dimension
    '''
    G, L, X, AW = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], X[idx], AW[idx]

def GLXAW_from_structures(structures, atom_types, wyck_types, n_max, dim):
    G = [] # space group
    L = [] # abc alpha beta gamma
    X = [] # fractional coordinate 
    AW = [] # atom type and wyck type; 0 for placeholder
    for i, structure in enumerate(structures):
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)   # a looser tolerance of 0.1 (the value used in Materials Project) is often needed.
        refined_structure = analyzer.get_refined_structure()
        analyzer = SpacegroupAnalyzer(refined_structure)
        symmetrized_structure = analyzer.get_symmetrized_structure()
        
        
        g = analyzer.get_space_group_number()
        G.append (g)
        print (i, g, structure.num_sites, symmetrized_structure.num_sites)
        abc = tuple([l/symmetrized_structure.num_sites**(1./3.) for l in symmetrized_structure.lattice.abc])
        L.append (abc + symmetrized_structure.lattice.angles) # scale length with number of total atoms
        num_sites = len(symmetrized_structure.equivalent_sites)
        assert (n_max >= num_sites)
   
        aw = []
        ws = []
        fc = []
        for i, site in enumerate(symmetrized_structure.equivalent_sites):
            a = site[0].specie.number # element number 
            x = site[0].frac_coords
            m = len(site)             # multiplicity
            wyckoff_symbol = symmetrized_structure.wyckoff_symbols[i]
            w = wyckoff_dict[g-1][wyckoff_symbol]
            assert (a < atom_types)
            assert (w < wyck_types)
            aw.append( (w-1) * (atom_types-1)+ (a-1) +1 )
            ws.append( wyckoff_symbol)
            fc.append( x)
            print ('g, a, w, m, symbol', g, a, w, m, wyckoff_symbol, x)
        #sort atoms according to lexicographic order of wyckoff symbol
        idx = np.argsort(ws)
        ws = np.array(ws)[idx]
        aw = jnp.array(aw)[idx]
        fc = jnp.array(fc)[idx].reshape(num_sites, dim)
        print (ws) 
        
        aw = jnp.concatenate([aw, 
                              jnp.full((n_max - num_sites, ), 0)], 
                              axis = 0)
        fc = jnp.concatenate([fc, 
                             jnp.full((n_max - num_sites, dim), 1e10)], 
                             axis = 0)

        AW.append( aw )
        X.append ( fc )  
 
        print ('===================================')

   
    G = jnp.array(G) 
    L = jnp.array(L).reshape(-1, 6)
    X = jnp.array(X).reshape(-1, n_max, dim)
    AW = jnp.array(AW).reshape(-1, n_max)
    return G, L, X, AW
    
def GLXAW_from_file(csv_file, atom_types, wyck_types, n_max, dim):
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']
    structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]
    G, L, X, AW = GLXAW_from_structures(structures, atom_types, wyck_types, n_max, dim)
    return G, L, X, AW

if __name__=='__main__':
    atom_types = 119
    wyck_types = 30
    n_max = 20
    dim = 3

    #csv_file = '/home/wanglei/cdvae/data/perov_5/val.csv'
    #csv_file = '../data/mini.csv'
    #csv_file = '/home/wanglei/cdvae/data/mp_20/train.csv'
    csv_file = './mp_problem.csv'
    G, L, X, AW = GLXAW_from_file(csv_file, atom_types, wyck_types, n_max, dim)
    
    print (G.shape)
    print (L.shape)
    print (X.shape)
    print (AW.shape)
    
    print (L)
    print (X)

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    A, W = to_A_W(AW, atom_types)
    print (A)
    print (W)
    
    @jax.vmap
    def lookup(G, W):
        return wyckoff_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    print (M.sum(axis=-1))
