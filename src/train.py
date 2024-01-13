import jax 
import jax.numpy as jnp
import optax
from functools import partial

from utils import shuffle

def train(key, loss_fn, params, epochs, lr, batchsize, train_data):

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

    for epoch in range(epochs):
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

        print(epoch, total_loss/counter) 
    return params
