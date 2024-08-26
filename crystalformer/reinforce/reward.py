import jax
import jax.numpy as jnp
import numpy as np
import itertools
from pymatgen.core import Structure, Lattice

from crystalformer.src.wyckoff import symmetrize_atoms


def get_atoms_from_GLXYZAW(G, L, XYZ, A, W):
    A = np.array(A)
    X = np.array(XYZ)
    W = np.array(W)

    A = A[np.nonzero(A)]
    X = X[np.nonzero(A)]
    W = W[np.nonzero(A)]

    lattice = Lattice.from_parameters(*L)
    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    as_list = [[A[idx] for _ in range(len(xs))] for idx, xs in enumerate(xs_list)]
    A_list = list(itertools.chain.from_iterable(as_list))
    X_list = list(itertools.chain.from_iterable(xs_list))
    struct = Structure(lattice, A_list, X_list).to_ase_atoms()
    return struct


def make_force_reward_fn(calculator):
    """
    ase calculator object
    """
    def reward_fn(x):
        G, L, XYZ, A, W = x
        atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
        atoms.calc = calculator
        forces = jnp.array(atoms.get_forces())
        forces = jnp.linalg.norm(forces, axis=-1)
        fmax = jnp.max(forces) # same definition as fmax in ase

        return fmax

    def batch_reward_fn(x):
        output = map(reward_fn, zip(*x))
        return jnp.array(list(output))

    return reward_fn, batch_reward_fn


def make_distance_reward_fn():

    def reward_fn(x):
        G, L, XYZ, A, W = x
        atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
        dis_matrix = jnp.array(atoms.get_all_distances(mic=True, vector=False))
        min_dis = jnp.min(jnp.where(dis_matrix == 0, jnp.inf, dis_matrix))  # avoid 0 distance

        return jnp.array(1/min_dis)

    def batch_reward_fn(x):
        output = map(reward_fn, zip(*x))
        return jnp.array(list(output))

    return reward_fn, batch_reward_fn
