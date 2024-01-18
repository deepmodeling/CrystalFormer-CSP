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
    def update(params, opt_state, data):
        G, L, X, AM = data
        value, grad = jax.value_and_grad(loss_fn)(params, G, L, X, AM)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    for epoch in range(epoch_finished+1, epochs):
        key, subkey = jax.random.split(key)
        train_data = shuffle(subkey, train_data)

        train_G, train_L, train_X, train_AM = train_data 

        train_loss = 0.0 
        num_samples = len(train_L)
        num_batches = math.ceil(num_samples / batchsize)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            data = train_G[start_idx:end_idx], \
                   train_L[start_idx:end_idx], \
                   train_X[start_idx:end_idx], \
                   train_AM[start_idx:end_idx]

            params, opt_state, loss = update(params, opt_state, data)
            train_loss += loss 
        train_loss = train_loss/num_batches

        if epoch % 100 == 0:
            valid_G, valid_L, valid_X, valid_AM = valid_data 
            valid_loss = 0.0 
            num_samples = len(valid_L)
            num_batches = math.ceil(num_samples / batchsize)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batchsize
                end_idx = min(start_idx + batchsize, num_samples)
                G, L, X, AM = valid_G[start_idx:end_idx], \
                              valid_L[start_idx:end_idx], \
                              valid_X[start_idx:end_idx], \
                              valid_AM[start_idx:end_idx]
                loss = loss_fn(params, G, L, X, AM)
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
