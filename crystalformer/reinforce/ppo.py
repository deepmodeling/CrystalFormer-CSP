import jax
import jax.numpy as jnp
import os
import optax

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.lattice import norm_lattice


def make_ppo_loss_fn(logp_fn, eps_clip, beta=0.01):

    def ppo_loss_fn(params, key, x, old_logp, advantages):

        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, key, *x, False)
        logp = logp_w + logp_xyz + logp_a + logp_l
        entropy = - jnp.mean(jnp.exp(logp) * logp)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = jnp.exp(logp - old_logp)
        # jax.debug.print("ratios mean {x}", x=jnp.mean(ratios))

        # Finding Surrogate Loss  
        surr1 = ratios * advantages
        surr2 = jax.lax.clamp(1-eps_clip, ratios, 1+eps_clip) * advantages

        # Final loss of clipped objective PPO
        ppo_loss = jnp.mean(jnp.minimum(surr1, surr2)) + beta * entropy

        return ppo_loss
    
    return ppo_loss_fn


def train(key, optimizer, opt_state, logp_fn, batch_reward_fn, ppo_loss_fn, sample_crystal, params, epoch_finished, epochs, ppo_epochs, batchsize, path):

    @jax.jit
    def step(params, key, opt_state, x, old_logp, advantages):
        value, grad = jax.value_and_grad(ppo_loss_fn)(params, key, x, old_logp, advantages)
        grad = jax.tree_util.tree_map(lambda g_: g_ * -1.0, grad)  # invert gradient for maximization
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch f_mean f_err\n")
 
    spacegroup = 1

    for epoch in range(epoch_finished+1, epochs):

        key, sample_key, loss_key = jax.random.split(key, 3)
        XYZ, A, W, M, L = sample_crystal(sample_key, params=params, g=spacegroup, batchsize=batchsize) 
        G = spacegroup * jnp.ones((batchsize), dtype=int)

        x = (G, L, XYZ, A, W)
        rewards = - batch_reward_fn(x)  # inverse reward
        f_mean = jnp.mean(rewards)
        f_err = jnp.std(rewards) / jnp.sqrt(batchsize)
        advantages = rewards - f_mean

        f.write( ("%6d" + 2*"  %.6f" + "\n") % (epoch, f_mean, f_err))

        G, L, XYZ, A, W = x
        L = norm_lattice(G, W, L)
        x = (G, L, XYZ, A, W)
        logp_w, logp_xyz, logp_a, logp_l = jax.jit(logp_fn, static_argnums=7)(params, loss_key, *x, False)
        old_logp = logp_w + logp_xyz + logp_a + logp_l
        
        for _ in range(ppo_epochs):
            key, subkey = jax.random.split(key)
            params, opt_state, value = step(params, subkey, opt_state, x, old_logp, advantages)
            print("epoch %d, loss %.6f" % (epoch, value))
        
        if epoch % 5 == 0:
            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()

    return params, opt_state
