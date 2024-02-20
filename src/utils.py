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

@jax.vmap
def sort_atoms(W, A, X):
    '''
    lex sort atoms according W, A, X, Y, Z

    W: (n, )
    A: (n, )
    X: (n, dim)
    '''

    W_temp = jnp.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort
    A_temp = jnp.where(A>0, A, 9999)

    X -= jnp.floor(X) # wrap back to 0-1 
    idx = jnp.lexsort((X[:,2], X[:,1], X[:,0], A_temp, W_temp))

    #assert jnp.allclose(W, W[idx])
    A = A[idx]
    X = X[idx]
    return A, X

def map_to_mesh(array, coord_types):
    array -= jnp.floor(array)
    return jnp.minimum((array * coord_types).astype(int), coord_types - 1)

def letter_to_number(letter):
    '''
    'a' to 1 , 'b' to 2 , 'z' to 26, and 'A' to 27 
    '''
    return ord(letter) - ord('a') + 1 if 'a' <= letter <= 'z' else 27 if letter == 'A' else None

def shuffle(key, data):
    '''
    shuffle data along batch dimension
    '''
    G, L, XYZ, A, W = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], XYZ[idx], A[idx], W[idx]
    
def process_one(cif, atom_types, wyck_types, coord_types, n_max, tol=0.01):
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
    ww = []
    aa = []
    fc = []
    ws = []
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
        aa.append( a )
        ww.append( w )
        fc.append( x )  # the generator of the orbit
        ws.append( symbol )
        print ('g, a, w, m, symbol, x:', g, a, w, m, symbol, x)
    idx = np.argsort(ww)
    ww = np.array(ww)[idx]
    aa = np.array(aa)[idx]
    fc = np.array(fc)[idx].reshape(num_sites, 3)
    ws = np.array(ws)[idx]
    print (ws, aa, ww, natoms) 

    _, count = np.unique(np.array(ww), return_counts=True)
    if count.max() > 4:
        print (g, ww, fc)

    aa = np.concatenate([aa,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)

    ww = np.concatenate([ww,
                        np.full((n_max - num_sites, ), 0)],
                        axis=0)
    fc = np.concatenate([fc, 
                         np.full((n_max - num_sites, 3), 1e10)],
                        axis=0)
    
    abc = np.array([c.lattice.a, c.lattice.b, c.lattice.c])/natoms**(1./3.)
    angles = np.array([c.lattice.alpha, c.lattice.beta, c.lattice.gamma])
    l = np.concatenate([abc, angles])
    
    print ('===================================')

    return g, l, fc, aa, ww 

def GLXYZAW_from_file(csv_file, atom_types, wyck_types, coord_types, n_max, num_workers=1):
    data = pd.read_csv(csv_file)
    cif_strings = data['cif']

    p = multiprocessing.Pool(num_workers)
    partial_process_one = partial(process_one, atom_types=atom_types, wyck_types=wyck_types, coord_types=coord_types, n_max=n_max)
    results = p.map_async(partial_process_one, cif_strings).get()
    p.close()
    p.join()

    G, L, XYZ, A, W = zip(*results)

    G = jnp.array(G) 
    A = jnp.array(A).reshape(-1, n_max)
    W = jnp.array(W).reshape(-1, n_max)
    XYZ = jnp.array(XYZ).reshape(-1, n_max, 3)
    L = jnp.array(L).reshape(-1, 6)

    A, XYZ = sort_atoms(W, A, XYZ)
    
    XYZ = map_to_mesh(XYZ, coord_types)
    return G, L, XYZ, A, W

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
    coord_types = 128
    n_max = 24

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    #csv_file = '../data/mini.csv'
    #csv_file = '/home/wanglei/cdvae/data/carbon_24/val.csv'
    #csv_file = '/home/wanglei/cdvae/data/perov_5/val.csv'
    csv_file = '/home/wanglei/cdvae/data/mp_20/train.csv'

    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, coord_types, n_max)
    
    print (G.shape)
    print (L.shape)
    print (XYZ.shape)
    print (A.shape)
    print (W.shape)
    
    print ('L:\n',L)
    print ('XYZ:\n',XYZ)


    @jax.vmap
    def lookup(G, W):
        return mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    print ('N:\n', M.sum(axis=-1))
