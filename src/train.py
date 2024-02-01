import jax
import jax.numpy as jnp
from functools import partial
import os
import optax
import math

from utils import shuffle
import checkpoint

def train(key, optimizer, opt_state, loss_fn, params, epoch_finished, epochs, batchsize, train_data, valid_data, path):
           
    @jax.jit
    def update(params, key, opt_state, data):
        G, L, X, AW = data
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(params, key, G, L, X, AW, True)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch t_loss v_loss t_loss_x v_loss_x t_loss_aw v_loss_aw t_loss_l v_loss_l\n")
 
    for epoch in range(epoch_finished+1, epochs):
        key, subkey = jax.random.split(key)
        train_data = shuffle(subkey, train_data)

        train_G, train_L, train_X, train_AW = train_data 

        train_loss = 0.0 
        train_aux = 0.0, 0.0, 0.0
        num_samples = len(train_L)
        num_batches = math.ceil(num_samples / batchsize)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            data = train_G[start_idx:end_idx], \
                   train_L[start_idx:end_idx], \
                   train_X[start_idx:end_idx], \
                   train_AW[start_idx:end_idx]
            
            key, subkey = jax.random.split(key)
            params, opt_state, (loss, aux) = update(params, subkey, opt_state, data)
            train_loss, train_aux = jax.tree_map(
                        lambda acc, i: acc + i,
                        (train_loss, train_aux), 
                        (loss, aux)
                        )

        train_loss, train_aux = jax.tree_map(
                        lambda x: x/num_batches, 
                        (train_loss, train_aux)
                        ) 

        if epoch % 100 == 0:
            valid_G, valid_L, valid_X, valid_AW = valid_data 
            valid_loss = 0.0 
            valid_aux = 0.0, 0.0, 0.0
            num_samples = len(valid_L)
            num_batches = math.ceil(num_samples / batchsize)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batchsize
                end_idx = min(start_idx + batchsize, num_samples)
                G, L, X, AW = valid_G[start_idx:end_idx], \
                              valid_L[start_idx:end_idx], \
                              valid_X[start_idx:end_idx], \
                              valid_AW[start_idx:end_idx]

                key, subkey = jax.random.split(key)
                loss, aux = loss_fn(params, subkey, G, L, X, AW, False)
                valid_loss, valid_aux = jax.tree_map(
                        lambda acc, i: acc + i,
                        (valid_loss, valid_aux), 
                        (loss, aux)
                        )

            valid_loss, valid_aux = jax.tree_map(
                        lambda x: x/num_batches, 
                        (valid_loss, valid_aux)
                        ) 

            train_loss_x, train_loss_aw, train_loss_l = train_aux
            valid_loss_x, valid_loss_aw, valid_loss_l = valid_aux

            f.write( ("%6d" + 8*"  %.6f" + "\n") % (epoch, 
                                                    train_loss, valid_loss,
                                                    train_loss_x, valid_loss_x, 
                                                    train_loss_aw, valid_loss_aw, 
                                                    train_loss_l, valid_loss_l
                                                    ))

            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return params, opt_state
