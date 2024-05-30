import jax
import jax.numpy as jnp
from functools import partial

from wyckoff import fc_mask_table

get_fc_mask = lambda g, w: jnp.logical_and((w>0)[:, None], fc_mask_table[g-1, w])


@partial(jax.jit, static_argnums=0)
def mcmc(logp_fn, params, x_init, key, mc_steps, mc_width):
    """
        Markov Chain Monte Carlo sampling algorithm.

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (batch, n, dim).
        x_init: initial value of x, with shape (batch, n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal.

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """
    def step(i, state):
        x, logp, key, num_accepts = state
        G, L, XYZ, A, W = x
        key, key_proposal_A, key_proposal_XYZ, key_accept, key_logp = jax.random.split(key, 5)
        
        _A = jax.random.choice(key_proposal_A, jnp.arange(1, 118), shape=A.shape)
        A_proposal = jnp.where(A == 0, A, _A)

        fc_mask = jax.vmap(get_fc_mask, in_axes=(0, 0))(G, W)
        _XYZ = XYZ + mc_width * jax.random.normal(key_proposal_XYZ, XYZ.shape) # TODO: von-mises 
        XYZ_proposal = jnp.where(fc_mask, _XYZ, XYZ)
        x_proposal = (G, L, XYZ_proposal, A_proposal, W)

        logp_w, logp_xyz, logp_a, _ = logp_fn(params, key_logp, *x_proposal, False)
        logp_proposal = logp_w + logp_xyz + logp_a

        ratio = jnp.exp((logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        A_new = jnp.where(accept[:, None], A_proposal, A)  # update atom types
        XYZ_new = jnp.where(accept[:, None, None], XYZ_proposal, XYZ)  # update atom positions
        x_new = (G, L, XYZ_new, A_new, W)
        logp_new = jnp.where(accept, logp_proposal, logp)
        num_accepts += accept.sum()
        return x_new, logp_new, key, num_accepts
    
    key, subkey = jax.random.split(key)
    logp_w, logp_xyz, logp_a, _ = logp_fn(params, subkey, *x_init, False)
    logp_init = logp_w + logp_xyz + logp_a
    # print("logp_init", logp_init)
    
    x, logp, key, num_accepts = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key, 0.))
    # print("logp", logp)
    accept_rate = num_accepts / (mc_steps * x[0].shape[0])
    return x, accept_rate


if __name__  == "__main__":
    from utils import GLXYZAW_from_file
    from loss import make_loss_fn
    from transformer import make_transformer
    atom_types = 119
    n_max = 21
    wyck_types = 28
    Nf = 5
    Kx = 16
    Kl  = 4
    dropout_rate = 0.3

    csv_file = '../data/mini.csv'
    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)

    key = jax.random.PRNGKey(42)

    params, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 128, 4, 4, 8, 16, 16, atom_types, wyck_types, dropout_rate) 
 
    loss_fn, logp_fn = make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, transformer)

    # MCMC sampling test
    mc_steps = 200
    mc_width = 0.1
    x_init = (G[:5], L[:5], XYZ[:5], A[:5], W[:5])

    value = jax.jit(logp_fn, static_argnums=7)(params, key, *x_init, False)

    for i in range(5):
        key, subkey = jax.random.split(key)
        x, acc = mcmc(logp_fn, params, x_init=x_init, key=subkey, mc_steps=mc_steps, mc_width=mc_width)
        print(i, acc)
