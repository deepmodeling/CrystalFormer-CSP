import jax
import jax.numpy as jnp
import os
import optax
import math
from functools import partial

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.formula import find_composition_vector
from crystalformer.src.lattice import norm_lattice


def make_ppo_loss_fn(logp_fn, eps_clip, beta=0.1, alpha=0.0, lamb_g=1.0, lamb_xyz=1.0,lamb_a=1.0, lamb_w=1.0, lamb_l=1.0):

    """
    PPO clipped objective function with KL divergence regularization
    PPO_loss = PPO-clip + beta  * KL(P || P_pretrain)

    Note that we only consider the logp_xyz and logp_l in the logp_fn
    """

    def ppo_loss_fn(params, key, x, old_logp, pretrain_logp, advantages):

        logp_g, logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, key, *x, False)
        logp = lamb_g*logp_g + lamb_w*logp_w + lamb_xyz *logp_xyz + lamb_a*logp_a + lamb_l*logp_l
            
        kl_loss = logp - pretrain_logp  
        advantages = advantages - beta * kl_loss - alpha * logp

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = jnp.exp(logp - old_logp)

        # Finding Surrogate Loss  
        surr1 = ratios * advantages
        surr2 = jax.lax.clamp(1-eps_clip, ratios, 1+eps_clip) * advantages

        # Final loss of clipped objective PPO
        ppo_loss = jnp.mean(jnp.minimum(surr1, surr2))

        return ppo_loss, (jnp.mean(kl_loss), -jnp.mean(logp_g), -jnp.mean(logp_w), -jnp.mean(logp_a), -jnp.mean(logp_xyz), -jnp.mean(-logp_l))
    
    return ppo_loss_fn


def train(key, optimizer, opt_state, loss_fn, logp_fn, batch_reward_fn, ppo_loss_fn, sample_crystal, composition, params, epoch_finished, epochs, ppo_epochs, batchsize, path, clip_value):

    num_devices = jax.local_device_count()
    batch_per_device = batchsize // num_devices
    shape_prefix = (num_devices, batch_per_device)
    print("num_devices: ", num_devices)
    print("batch_per_device: ", batch_per_device)
    print("shape_prefix: ", shape_prefix)

    @partial(jax.pmap, axis_name="p", in_axes=(None, None, None, 0, 0, 0, 0), out_axes=(None, None, 0),)
    def step(params, key, opt_state, x, old_logp, pretrain_logp, advantages):
        value, grad = jax.value_and_grad(ppo_loss_fn, has_aux=True)(params, key, x, old_logp, pretrain_logp, advantages)
        grad = jax.lax.pmean(grad, axis_name="p")
        value = jax.lax.pmean(value, axis_name="p")
        grad = jax.tree_util.tree_map(lambda g_: g_ * -1.0, grad)  # invert gradient for maximization
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch f_mean f_err f_min f_max formula_match_fraction unique_space_groups unique_wyckoff_sequences unique_atom_sequences unique_WA_combinations kl g w a xyz l\n")

    pretrain_params = params
    logp_fn = jax.jit(logp_fn, static_argnums=7)
    loss_fn = jax.jit(loss_fn, static_argnums=7)
    
    for epoch in range(epoch_finished+1, epoch_finished+epochs+1):

        # Keep sampling until at least one crystal matches the formula
        formula_match = jnp.array([False])
        max_attempts = 100
        attempt = 0
        while not jnp.any(formula_match) and attempt < max_attempts:
            key, subkey = jax.random.split(key)
            G, XYZ, A, W, M, L = sample_crystal(subkey, params, batchsize, composition)

            actual_compositions = jax.vmap(find_composition_vector)(A, M)
            formula_match = jnp.all(actual_compositions == composition, axis=1)
            attempt += 1
        
        if attempt >= max_attempts:
            raise RuntimeError(f"Epoch {epoch} - Could not generate formula-matching crystals after {max_attempts} attempts. Stopping training.")
        
        # Compute fraction of formula matches
        formula_match_fraction = jnp.mean(formula_match.astype(jnp.float32))
 
        # Compute unique number of space groups 
        unique_space_groups = jnp.unique(G[formula_match], return_counts=False).shape[0]
        # Number of unique wyckoff sequences for those matched formula
        unique_wyckoff_sequences = jnp.unique(W[formula_match], axis=0, return_counts=False).shape[0]
        unique_atom_sequences = jnp.unique(A[formula_match], axis=0, return_counts=False).shape[0]
        # Number of unique (W, A) combinations
        WA_combined = jnp.concatenate([W[formula_match], A[formula_match]], axis=1)
        unique_WA_combinations = jnp.unique(WA_combined, axis=0, return_counts=False).shape[0]

        x = (G, L, XYZ, A, W)

        x_matched = jax.tree_util.tree_map(lambda arr: arr[formula_match], x)
        rewards_matched = -batch_reward_fn(x_matched)
        
        rewards = jnp.full((batchsize,), -clip_value) # default reward for unmatched structure
        rewards = rewards.at[formula_match].set(rewards_matched)

        f_mean = jnp.mean(rewards)
        f_err = jnp.std(rewards) / jnp.sqrt(batchsize)
        f_min = jnp.min(rewards)
        f_max = jnp.max(rewards)

        advantages = rewards - f_mean
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 

        G, L, XYZ, A, W = x
        L = norm_lattice(G, W, L)
        x = (G, L, XYZ, A, W)

        key, subkey1, subkey2 = jax.random.split(key, 3)
        logp_g, logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, subkey1, *x, False)
        old_logp = logp_g + logp_w + logp_xyz + logp_a + logp_l

        logp_g, logp_w, logp_xyz, logp_a, logp_l = logp_fn(pretrain_params, subkey2, *x, False)
        pretrain_logp = logp_g + logp_w + logp_xyz + logp_a + logp_l

        x = jax.tree_util.tree_map(lambda _x: _x.reshape(shape_prefix + _x.shape[1:]), x)
        old_logp = old_logp.reshape(shape_prefix + old_logp.shape[1:])
        pretrain_logp = pretrain_logp.reshape(shape_prefix + pretrain_logp.shape[1:])
        advantages = advantages.reshape(shape_prefix + advantages.shape[1:])

        for _ in range(ppo_epochs):
            key, subkey = jax.random.split(key)
            params, opt_state, value = step(params, subkey, opt_state, x, old_logp, pretrain_logp, advantages)
            ppo_loss, (kl_loss, entropy_g, entropy_w, entropy_a, entropy_xyz, entropy_l) = value
        
        f.write( ("%6d" + 5*"  %.6f" + 4*"  %3d" + 6*"  %.6f"+ "\n") % (epoch, f_mean, f_err, f_min, f_max, formula_match_fraction, unique_space_groups, unique_wyckoff_sequences, unique_atom_sequences, unique_WA_combinations, 
            kl_loss[0], entropy_g[0], entropy_w[0], entropy_a[0], entropy_xyz[0], entropy_l[0]))

        if epoch % 10 == 0:
            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()

    return params, opt_state
