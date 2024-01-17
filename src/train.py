import jax 
import jax.numpy as jnp
from functools import partial
import os
import optax

from utils import shuffle
import checkpoint

def train(key, optimizer, loss_fn, params, epoch_finished, epochs, batchsize, train_data, valid_data, path):
           
    opt_state = optimizer.init(params)

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
        counter = 0 
        for batch_index in range(0, len(train_L), batchsize):
            data = train_G[batch_index:batch_index+batchsize], \
                   train_L[batch_index:batch_index+batchsize], \
                   train_X[batch_index:batch_index+batchsize], \
                   train_AM[batch_index:batch_index+batchsize]

            params, opt_state, loss = update(params, opt_state, data)
            train_loss += loss 
            counter += 1
        train_loss = train_loss/counter

        if epoch % 100 == 0:
            valid_G, valid_L, valid_X, valid_AM = valid_data 
            valid_loss = 0.0 
            counter = 0 
            for batch_index in range(0, len(valid_L), batchsize):
                G, L, X, AM = valid_G[batch_index:batch_index+batchsize], \
                              valid_L[batch_index:batch_index+batchsize], \
                              valid_X[batch_index:batch_index+batchsize], \
                              valid_AM[batch_index:batch_index+batchsize]
                loss = loss_fn(params, G, L, X, AM)
                valid_loss += loss 
                counter += 1
            valid_loss = valid_loss/counter

            f.write( ("%6d" + 2*"  %.6f" + "\n") % (epoch, train_loss, valid_loss) )

            ckpt = {"params": params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return params
