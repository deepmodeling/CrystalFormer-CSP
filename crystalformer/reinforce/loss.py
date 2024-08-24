import jax
import jax.numpy as jnp
import numpy as np
import itertools
from pymatgen.core import Structure, Lattice

from crystalformer.src.wyckoff import symmetrize_atoms


def make_reward_fn(calculator):
    """
    ase calculator object
    """

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

    def reward_fn(x):
        G, L, XYZ, A, W = x
        atoms = get_atoms_from_GLXYZAW(G, L, XYZ, A, W)
        atoms.calc = calculator
        forces = atoms.get_forces()
        fmax = np.max(np.abs(forces)) # same definition as fmax in ase

        return jnp.array(fmax)

    def batch_reward_fn(x):
        batch_reward = jnp.zeros(x[0].shape[0])
        for i in range(x[0].shape[0]):
            _x = jax.tree_map(lambda x: x[i], x)
            reward = reward_fn(_x)
            batch_reward = batch_reward.at[i].set(reward)
    
        return batch_reward

    return reward_fn, batch_reward_fn


def make_reinforce_loss(batch_logp, batch_reward_fn):

    def loss(params, key, x, is_train):
        
        # TODO: now only support for crystalformer logp
        logp_w, logp_xyz, logp_a, logp_l = jax.jit(batch_logp, static_argnums=7)(params, key, *x, is_train)
        entropy = logp_w + logp_xyz + logp_a + logp_l

        f = batch_reward_fn(x)
        f = jax.lax.stop_gradient(f)

        f_mean = jnp.mean(f)

        f_std = jnp.std(f)/jnp.sqrt(f.shape[0])

        return jnp.mean((f - f_mean) * entropy), (f_mean, f_std)

    return loss