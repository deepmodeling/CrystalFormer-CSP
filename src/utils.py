import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pyxtal import pyxtal
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from functools import partial
import multiprocessing
import itertools
import os

from wyckoff import mult_table
from elements import element_list

def letter_to_number(letter):
    '''
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    '''
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

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
    
def process_one(cif, atom_types, wyck_types, n_max, dim, tol=0.01):
    # taken from https://anonymous.4open.science/r/DiffCSP-PP-8F0D/diffcsp/common/data_utils.py
    crystal = Structure.from_str(cif, fmt='cif')
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    
    g = c.group.number
    num_sites = len(c.atom_sites)
    assert (n_max > num_sites) # we will need at least one empty site for output of L params

    print (g, c.group.symbol, num_sites)
    natoms = 0
    aw = []
    ws = []
    fc = []
    for site in c.atom_sites:
        a = element_list.index(site.specie) 
        x = site.position
        m = site.wp.multiplicity
        w = letter_to_number(site.wp.letter)
        symbol = str(m) + site.wp.letter
        natoms += site.wp.multiplicity
        assert (a < atom_types)
        assert (w < wyck_types)
        assert (np.allclose(x, site.wp[0].operate(x)))
        aw.append( (w-1) * (atom_types-1)+ (a-1) +1 )
        ws.append( symbol )
        fc.append( x )  # the generator of the orbit
        print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    #sort atoms according to wyckoff symbol a-z,A
    char_list = [''.join(filter(str.isalpha, s)) for s in ws]
    idx, _ = zip(*sorted(enumerate(char_list), key=lambda x: (x[1].isupper(), x[1].lower())))
    idx = np.array(idx)
    ws = np.array(ws)[idx]
    aw = np.array(aw)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, dim)
    print (ws, natoms) 

    aw = np.concatenate([aw,
                         np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, dim), 1e10)],
                        axis=0)
    
    deg = 180.0 / np.pi 
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])*deg
    l = np.concatenate([abc, angles])
    
    print ('===================================')

    return g, l, fc, aw

def GLXAW_from_file(csv_file, atom_types, wyck_types, n_max, dim, num_workers=1):
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']

    p = multiprocessing.Pool(num_workers)
    partial_process_one = partial(process_one, atom_types=atom_types, wyck_types=wyck_types, n_max=n_max, dim=dim)
    results = p.map_async(partial_process_one, cif_strings).get()
    p.close()
    p.join()

    G, L, X, AW = zip(*results)
    G = jnp.array(G) 
    L = jnp.array(L).reshape(-1, 6)
    X = jnp.array(X).reshape(-1, n_max, dim)
    AW = jnp.array(AW).reshape(-1, n_max)
    return G, L, X, AW

def GLXA_to_structure_single(G, L, X, A):

    lattice = Lattice.from_parameters(*L)
    # filter out zero
    zero_idx = np.where(A == 0)[0]
    if zero_idx is not None and len(zero_idx) > 0:
        idx = zero_idx[0]
        A = A[:idx]
        X = X[:idx]
    structure = Structure.from_spacegroup(sg=G, lattice=lattice, species=A, coords=X).as_dict()

    return structure

def GLXA_to_csv(G, L, X, A, num_worker=1, filename='out_structure.csv'):

    L = np.array(L)
    X = np.array(X)
    A = np.array(A)
    p = multiprocessing.Pool(num_worker)
    if isinstance(G, int):
        G = np.array([G] * len(L))
    structures = p.starmap_async(GLXA_to_structure_single, zip(G, L, X, A)).get()
    p.close()
    p.join()

    data = pd.DataFrame()
    data['cif'] = structures
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)


if __name__=='__main__':
    atom_types = 119
    wyck_types = 28
    n_max = 24
    dim = 3
    
    #csv_file = './mini.csv'
    #csv_file = '/home/wanglei/cdvae/data/carbon_24/val.csv'
    #csv_file = '/home/wanglei/cdvae/data/perov_5/val.csv'
    csv_file = '/home/wanglei/cdvae/data/mp_20/train.csv'

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
