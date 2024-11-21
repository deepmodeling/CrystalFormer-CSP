import jax
import jax.numpy as jnp
import optax
import os
import math

from crystalformer.src.utils import shuffle
import crystalformer.src.checkpoint as checkpoint


def make_dpo_loss(logp_fn, beta, label_smoothing=0.0, gamma=0.0):
    
    def dpo_logp_fn(policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps):
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        # label_smoothing=0 gives original DPO 
        losses = -jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing) - jax.nn.log_sigmoid(-beta * logits) * label_smoothing
        return jnp.mean(losses)
    
    def loss_fn(params, key, x_w, x_l, ref_chosen_logps, ref_rejected_logps):
        key, subkey = jax.random.split(key)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, subkey, *x_w, False)
        policy_chosen_logps = logp_w + logp_xyz + logp_a + logp_l

        key, subkey = jax.random.split(key)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, subkey, *x_l, False)
        policy_rejected_logps = logp_w + logp_xyz + logp_a + logp_l

        dpo_loss = dpo_logp_fn(policy_chosen_logps,
                               policy_rejected_logps,
                               ref_chosen_logps,
                               ref_rejected_logps)
        loss = dpo_loss - gamma * jnp.mean(policy_chosen_logps)

        return loss, (dpo_loss, jnp.mean(policy_chosen_logps))

    return loss_fn


def train(key, optimizer, opt_state, dpo_loss_fn, logp_fn, params, epoch_finished, epochs, batchsize, chosen_data, rejected_data, path):

    @jax.jit
    def step(params, key, opt_state, x_w, x_l, ref_chosen_logps, ref_rejected_logps):
        value, grad = jax.value_and_grad(dpo_loss_fn, has_aux=True)(params, key, x_w, x_l, ref_chosen_logps, ref_rejected_logps)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch loss dpo_loss chosen_logp\n")
    ref_params = params
    logp_fn = jax.jit(logp_fn, static_argnums=7)

    ref_chosen_logps = jnp.array([])
    ref_rejected_logps = jnp.array([])
    _, chosen_L, _, _, _ = chosen_data
    num_samples = len(chosen_L)
    num_batches = math.ceil(num_samples / batchsize)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batchsize
        end_idx = min(start_idx + batchsize, num_samples)
        key, subkey1, subkey2 = jax.random.split(key, 3)

        data = jax.tree_map(lambda x: x[start_idx:end_idx], chosen_data)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(ref_params, subkey1, *data, False)
        logp = logp_w + logp_xyz + logp_a + logp_l
        ref_chosen_logps = jnp.append(ref_chosen_logps, logp, axis=0)

        data = jax.tree_map(lambda x: x[start_idx:end_idx], rejected_data)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(ref_params, subkey2, *data, False)
        logp = logp_w + logp_xyz + logp_a + logp_l
        ref_rejected_logps = jnp.append(ref_rejected_logps, logp, axis=0)

    print(ref_chosen_logps.shape, ref_rejected_logps.shape)
    print("Finished calculating reference logp")
    
    for epoch in range(epoch_finished+1, epochs+1):
        key, subkey = jax.random.split(key)
        chosen_data = shuffle(subkey, chosen_data)
        rejected_data = shuffle(subkey, rejected_data)  

        idx = jax.random.permutation(subkey, jnp.arange(len(ref_chosen_logps)))
        ref_chosen_logps = ref_chosen_logps[idx]
        ref_rejected_logps = ref_rejected_logps[idx]

        _, chosen_L, _, _, _ = chosen_data
        num_samples = chosen_L.shape[0]
        if num_samples % batchsize == 0:
            num_batches = math.ceil(num_samples / batchsize)
        else:
            num_batches = math.ceil(num_samples / batchsize) - 1
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            x_w = jax.tree_map(lambda x: x[start_idx:end_idx], chosen_data)
            x_l = jax.tree_map(lambda x: x[start_idx:end_idx], rejected_data)
            ref_chosen_logps_batch = ref_chosen_logps[start_idx:end_idx]
            ref_rejected_logps_batch = ref_rejected_logps[start_idx:end_idx]

            key, subkey = jax.random.split(key)
            params, opt_state, value = step(params, subkey, opt_state, x_w, x_l, ref_chosen_logps_batch, ref_rejected_logps_batch)
            loss, (dpo_loss, policy_chosen_logps) = value
        
            f.write( ("%6d" + 3*"  %.6f" + "\n") % (epoch, loss, dpo_loss, policy_chosen_logps))

            if batch_idx % 10 == 0:
                ckpt = {"params": params,
                        "opt_state" : opt_state
                    }
                ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %((epoch-1)*num_batches+batch_idx))
                checkpoint.save_data(ckpt, ckpt_filename)
                print("Save checkpoint file: %s" % ckpt_filename)

    f.close()

    return params, opt_state
