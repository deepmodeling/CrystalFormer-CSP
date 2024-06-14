import os
import jax
import optax
import math

import checkpoint


def train(key, optimizer, opt_state, loss_fn, params, epoch_finished, epochs, batchsize, train_data, valid_data, path):
           
    @jax.jit
    def update(params, key, opt_state, data):
        g, l, w, inputs, targets = data
        value, grad = jax.value_and_grad(loss_fn)(params, key, g, l, w, inputs, targets, True)
        # jnp.set_printoptions(threshold=jnp.inf)
        # jax.debug.print("grad {x}", x=grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch t_loss v_loss\n")
 
    for epoch in range(epoch_finished+1, epochs):
        key, subkey = jax.random.split(key)
        train_data = jax.tree_map(lambda x: jax.random.permutation(subkey, x), train_data)

        train_g, train_l, train_w, train_inputs, train_targets = train_data 

        train_loss = 0.0 
        num_samples = len(train_targets)
        num_batches = math.ceil(num_samples / batchsize)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            data = train_g[start_idx:end_idx], \
                   train_l[start_idx:end_idx], \
                   train_w[start_idx:end_idx], \
                   train_inputs[start_idx:end_idx], \
                   train_targets[start_idx:end_idx]
                  
            key, subkey = jax.random.split(key)
            params, opt_state, loss = update(params, subkey, opt_state, data)
            train_loss = train_loss + loss

        train_loss = train_loss / num_batches

        if epoch % 10 == 0:
            valid_g, valid_l, valid_w, valid_inputs, valid_targets = valid_data 
            valid_loss = 0.0 
            num_samples = len(valid_targets)
            num_batches = math.ceil(num_samples / batchsize)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batchsize
                end_idx = min(start_idx + batchsize, num_samples)
                g, l, w, inputs, targets = valid_g[start_idx:end_idx], \
                                           valid_l[start_idx:end_idx], \
                                           valid_w[start_idx:end_idx], \
                                           valid_inputs[start_idx:end_idx], \
                                           valid_targets[start_idx:end_idx]

                key, subkey = jax.random.split(key)
                loss = loss_fn(params, subkey, g, l, w, inputs, targets, False)
                valid_loss = valid_loss + loss

            valid_loss = valid_loss / num_batches

            f.write( ("%6d" + 2*"  %.6f" + "\n") % (epoch, 
                                                    train_loss,   valid_loss
                                                    ))
            
            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return params, opt_state
