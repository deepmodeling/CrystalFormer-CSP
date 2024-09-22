import jax
import jax.numpy as jnp
import numpy as np
from pymatgen.core import Structure, Lattice

from crystalformer.src.wyckoff import wmax_table, mult_table, symops

symops = np.array(symops)
mult_table = np.array(mult_table)
wmax_table = np.array(wmax_table)


def symmetrize_atoms(g, w, x):
    '''
    symmetrize atoms via, apply all sg symmetry op, finding the generator, and lastly apply symops 
    we need to do that because the sampled atom might not be at the first WP
    Args:
       g: int 
       w: int
       x: (3,)
    Returns:
       xs: (m, 3) symmetrize atom positions
    '''

    # (1) apply all space group symmetry op to the x 
    w_max = wmax_table[g-1].item()
    m_max = mult_table[g-1, w_max].item()
    ops = symops[g-1, w_max, :m_max] # (m_max, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    coords = ops@affine_point # (m_max, 3) 
    coords -= np.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
    # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs  
    #https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
    def dist_to_op0x(coord):
        diff = np.dot(symops[g-1, w, 0], np.array([*coord, 1])) - coord
        diff -= np.rint(diff)
        return np.sum(diff**2) 
   #  loc = np.argmin(jax.vmap(dist_to_op0x)(coords))
    loc = np.argmin([dist_to_op0x(coord) for coord in coords])
    x = coords[loc].reshape(3,)

    # (3) lastly, apply the given symmetry op to x
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]   # (m, 3, 4)
    affine_point = np.array([*x, 1]) # (4, )
    xs = ops@affine_point # (m, 3)
    xs -= np.floor(xs) # wrap back to 0-1 
    return xs


def get_atoms_from_GLXYZAW(G, L, XYZ, A, W):

    A = A[np.nonzero(A)]
    X = XYZ[np.nonzero(A)]
    W = W[np.nonzero(A)]

    lattice = Lattice.from_parameters(*L)
    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    A_list = np.repeat(A, [len(xs) for xs in xs_list])
    X_list = np.concatenate(xs_list)
    struct = Structure(lattice, A_list, X_list).to_ase_atoms()
    return struct


def make_force_reward_fn(calculator, weight=1.0):
    """
    Args:
        calculator: ase calculator object
        weight: weight for stress, total reward = log(forces + weight*stress)

    Returns:
        reward_fn: single reward function
        batch_reward_fn: batch reward function
    """
    def reward_fn(x):
        G, L, XYZ, A, W = x
        try: 
            atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
            atoms.calc = calculator
            forces = atoms.get_forces()
            stress = atoms.get_stress()
        except: 
            forces = np.ones((1, 3))*np.inf # avoid nan
            stress = np.ones((6,))*np.inf
        forces = np.linalg.norm(forces, axis=-1)
        forces = np.clip(forces, 1e-2, 1e2)  # avoid too large or too small forces
        forces = np.mean(forces)
        stress = np.clip(np.abs(stress), 1e-2, 1e2)
        stress = np.mean(stress)
        
        return np.log(forces + weight*stress)

    def batch_reward_fn(x):
        x = jax.tree_map(lambda _x: jax.device_put(_x, jax.devices('cpu')[0]), x)
        G, L, XYZ, A, W = x
        G, L, XYZ, A, W = np.array(G), np.array(L), np.array(XYZ), np.array(A), np.array(W)
        x = (G, L, XYZ, A, W)
        output = map(reward_fn, zip(*x))
        output = np.array(list(output))
        output = jax.device_put(output, jax.devices('gpu')[0]).block_until_ready()

        return output

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
