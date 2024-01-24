import jax 
import jax.numpy as jnp
from functools import partial
import os
import optax
import math

from utils import shuffle
import checkpoint

def train(key, optimizer, opt_state, loss_fn, params, epoch_finished, epochs, batchsize, train_data, valid_data, path, dropout_rate):
           
    @jax.jit
    def update(params, key, opt_state, data):
        G, L, X, AW = data
        value, grad = jax.value_and_grad(loss_fn)(params, key, G, L, X, AW, dropout_rate)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    for epoch in range(epoch_finished+1, epochs):
        key, subkey = jax.random.split(key)
        train_data = shuffle(subkey, train_data)

        train_G, train_L, train_X, train_AW = train_data 

        train_loss = 0.0 
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
            params, opt_state, loss = update(params, subkey, opt_state, data)
            train_loss += loss 
        train_loss = train_loss/num_batches

        if epoch % 100 == 0:
            valid_G, valid_L, valid_X, valid_AW = valid_data 
            valid_loss = 0.0 
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
                loss = loss_fn(params, subkey, G, L, X, AW, 0.0) # dropout =0 in validation
                valid_loss += loss 
            valid_loss = valid_loss/num_batches

            f.write( ("%6d" + 2*"  %.6f" + "\n") % (epoch, train_loss, valid_loss) )

            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return params, opt_state
