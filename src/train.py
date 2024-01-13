import jax 
import jax.numpy as jnp
import optax
from functools import partial
import os

from utils import shuffle
import checkpoint

def train(key, loss_fn, params, epoch_finished, epochs, lr, batchsize, train_data, path):

    L, X, A = train_data
    assert len(L)%batchsize==0

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, data):
        L, X, A = data
        value, grad = jax.value_and_grad(loss_fn)(params, L, X, A)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    for epoch in range(epoch_finished+1, epochs):
        key, subkey = jax.random.split(key)
        L, X, A = shuffle(subkey, L, X, A)
        total_loss = 0.0 
        counter = 0 
        for batch_index in range(0, len(L), batchsize):
            data = L[batch_index:batch_index+batchsize], \
                   X[batch_index:batch_index+batchsize], \
                   A[batch_index:batch_index+batchsize]

            params, opt_state, loss = update(params, opt_state, data)

            total_loss += loss 
            counter += 1

        f.write( ("%6d" + "  %.6f" + "\n") % (epoch, total_loss/counter) )
        if epoch % 1000 == 0:
            ckpt = {"params": params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return params
