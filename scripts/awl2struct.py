import sys
sys.path.append('./src/')

import pandas as pd
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
from ast import literal_eval
import multiprocessing
import itertools
import argparse

from pymatgen.core import Structure, Lattice
from wyckoff import symmetrize_atoms


def get_struct_from_lawx(G, L, A, W, X):
    A = np.array(A)
    X = np.array(X)
    L = np.array(L)
    W = np.array(W)

    A = A[np.nonzero(A)]
    X = X[np.nonzero(A)]
    W = W[np.nonzero(A)]

    lattice = Lattice.from_parameters(*L)
    # xs_list = [symmetrize_atoms(G, jnp.array(w), jnp.array(x)) for w, x in zip(W, X)]
    xs_list = []
    for w, x in zip(W, X):
        xs = symmetrize_atoms(jnp.array(G), jnp.array(w), jnp.array(x))
        xs_list.append(np.array(xs))
    as_list = [[A[idx] for _ in range(len(xs))] for idx, xs in enumerate(xs_list)]
    A_list = list(itertools.chain.from_iterable(as_list))
    X_list = list(itertools.chain.from_iterable(xs_list))
    struct = Structure(lattice, A_list, X_list)
    return struct, xs_list


def main(args):
    input_path = args.output_path + f'output_{args.label}_awxl.csv'
    origin_data = pd.read_csv(input_path)
    L,X,A,W = origin_data['L'],origin_data['X'],origin_data['A'],origin_data['W']
    L = L.apply(lambda x: literal_eval(x))
    X = X.apply(lambda x: literal_eval(x))
    A = A.apply(lambda x: literal_eval(x))
    W = W.apply(lambda x: literal_eval(x))
    # M = M.apply(lambda x: literal_eval(x))

    # p = multiprocessing.Pool(args.num_io_process)
    # G = np.array([int(args.label) for _ in range(len(L))])
    # structures = p.starmap_async(get_struct_from_lawx, zip(G, L, A, W, X)).get()
    # p.close()
    # p.join()

    structures = []
    G = np.array([int(args.label) for _ in range(len(L))])
    print(G)
    for idx, (g, l, a, w, x) in enumerate(zip(G, L, A, W, X)):
        struct, _ = get_struct_from_lawx(g, l, a, w, x)
        structures.append(struct.as_dict())

    output_path = args.output_path + f'output_{args.label}_struct.csv'

    data = pd.DataFrame()
    data['cif'] = structures
    data.to_csv(output_path, mode='a', index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_path', default='./', help='filepath of the output and input file')
    parser.add_argument('--label', default='194', help='output file label')
    parser.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
    args = parser.parse_args()
    main(args)
