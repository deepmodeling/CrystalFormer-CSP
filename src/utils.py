import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import partial
from ast import literal_eval
import multiprocessing
import itertools
import os

from wyckoff import wyckoff_dict, mult_table

@partial(jax.vmap, in_axes=(0, None), out_axes=0) # n 
def to_A_W(AW, atom_types):
    A = jnp.where(AW==0, jnp.zeros_like(AW), (AW-1)%(atom_types-1)+1)
    W = jnp.where(AW==0, jnp.zeros_like(AW), (AW-1)//(atom_types-1)+1)
    return A, W

@partial(jax.vmap, in_axes=(0, 0, None), out_axes=0) # n 
def to_AW(A, W, atom_types):
    return jnp.where(A==0, jnp.zeros_like(A), (W-1)*(atom_types-1) + (A-1) +1)

def shuffle(key, data):
    '''
    shuffle data along batch dimension
    '''
    G, L, X, AW = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], X[idx], AW[idx]

def process_one(structure, atom_types, wyck_types, n_max, dim):
    analyzer = SpacegroupAnalyzer(structure, symprec=0.01) 
    # refined_structure = analyzer.get_refined_structure()
    # analyzer = SpacegroupAnalyzer(refined_structure)
    symmetrized_structure = analyzer.get_symmetrized_structure()

    sg = analyzer.get_space_group_symbol()
    g = analyzer.get_space_group_number()
    print (g, sg, structure.num_sites, symmetrized_structure.num_sites)
    abc = tuple([l/symmetrized_structure.num_sites**(1./3.) for l in symmetrized_structure.lattice.abc])
    l = abc + symmetrized_structure.lattice.angles # scale length with number of total atoms
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
    #sort atoms according to wyckoff symbol a-z,A
    char_list = [''.join(filter(str.isalpha, s)) for s in ws]
    idx, _ = zip(*sorted(enumerate(char_list), key=lambda x: (x[1].isupper(), x[1].lower())))
    idx = np.array(idx)
    ws = np.array(ws)[idx]
    aw = np.array(aw)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, dim)
    print (ws) 
    
    aw = np.concatenate([aw,
                         np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, dim), 1e10)],
                        axis=0)
    print ('===================================')

    return g, l, fc, aw
    
def process_structure(cif):
    try :
        structure = Structure.from_dict(literal_eval(cif))
    except:
        structure = Structure.from_str(cif, fmt='cif')
    return structure

def GLXAW_from_file(csv_file, atom_types, wyck_types, n_max, dim, num_workers=1):
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']

    p = multiprocessing.Pool(num_workers)
    structures = p.map_async(process_structure, cif_strings).get()
    partial_process_one = partial(process_one, atom_types=atom_types, wyck_types=wyck_types, n_max=n_max, dim=dim)
    results = p.map_async(partial_process_one, structures).get()
    p.close()
    p.join()

    G, L, X, AW = zip(*results)
    G = jnp.array(G) 
    L = jnp.array(L).reshape(-1, 6)
    X = jnp.array(X).reshape(-1, n_max, dim)
    AW = jnp.array(AW).reshape(-1, n_max)
    return G, L, X, AW

def LXA_to_structure_single(L, X, A):

    lattice = Lattice.from_parameters(*L)
    # filter out zero
    zero_idx = np.where(A == 0)[0]
    if zero_idx is not None:
        idx = zero_idx[0]
        A = A[:idx]
        X = X[:idx]
    structure = Structure(lattice=lattice, species=A, coords=X, coords_are_cartesian=False).as_dict()

    return structure

def LXA_to_csv(L, X, A, num_worker=1, filename='out_structure.csv'):

    L = np.array(L)
    X = np.array(X)
    A = np.array(A)
    p = multiprocessing.Pool(num_worker)
    structures = p.starmap_async(LXA_to_structure_single, zip(L, X, A)).get()
    p.close()
    p.join()

    data = pd.DataFrame()
    data['cif'] = structures
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)


if __name__=='__main__':
    atom_types = 119
    wyck_types = 30
    n_max = 24
    dim = 3

    #csv_file = '/home/wanglei/cdvae/data/carbon_24/val.csv'
    #csv_file = '/home/wanglei/cdvae/data/perov_5/val.csv'
    #csv_file = '../data/mini.csv'
    #csv_file = '/home/wanglei/cdvae/data/mp_20/train.csv'
    csv_file = './mini.csv'
    #csv_file = '../data/symm_data/train.csv'

    G, L, X, AW = GLXAW_from_file(csv_file, atom_types, wyck_types, n_max, dim)
    
    print (G.shape)
    print (L.shape)
    print (X.shape)
    print (AW.shape)
    
    print ('L:\n',L)
    print ('X:\n',X)

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    A, W = to_A_W(AW, atom_types)
    print ('A:\n', A)
    print ('W:\n', W)
    print(A.shape)
    
    @jax.vmap
    def lookup(G, W):
        return mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    print ('N:\n', M.sum(axis=-1))
