import jax
import jax.numpy as jnp
import haiku as hk

import pandas as pd
from functools import partial
import os
import optax
import math

from config import *
import checkpoint


def make_classifier(key,
                    n_max = 21,
                    embed_size=32,
                    sequence_length=105,
                    outputs_size=64,
                    hidden_sizes=[128, 128],
                    num_classes=2,
                    dropout_rate=0.3):

    @hk.transform
    def network(g, l, w, h, is_training):
        """
        sequence_length = n_max * 5
        l : (6, )
        w : (n_max,)
        h : (sequence_length, ouputs_size)
        """
        mask = jnp.where(w > 0, 1, 0)
        mask = jnp.repeat(mask, 5, axis=-1)
        # mask = hk.Reshape((sequence_length, ))(mask)
        h = h * mask[:, None]

        w = jnp.mean(h[0::5, :], axis=-2)
        a = jnp.mean(h[1::5, :], axis=-2)
        xyz = jnp.mean(h[2::5, :], axis=-2) + jnp.mean(h[3::5, :], axis=-2) + jnp.mean(h[4::5, :], axis=-2)

        h = jnp.concatenate([w, a, xyz], axis=0) 
        h = hk.Flatten()(h)

        h = jnp.concatenate([g, h, l], axis=0)

        h = jax.nn.relu(hk.Linear(hidden_sizes[0])(h))
        h = hk.dropout(hk.next_rng_key(), dropout_rate, h) if is_training else h  # Dropout after the first ReLU

        for hidden_size in hidden_sizes[1: -1]:
            h_dense = jax.nn.relu(hk.Linear(hidden_size)(h))
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense) if is_training else h_dense
            h = h + h_dense

        h = hk.Linear(hidden_sizes[-1])(h)
        h = jax.nn.relu(h)
        h = hk.Linear(num_classes)(h)
    
        return h
        
    g = jnp.ones(embed_size)
    w = jnp.ones(n_max)
    l = jnp.ones(6)
    h = jnp.zeros((sequence_length, outputs_size))

    params = network.init(key, g, l, w, h, True)
    return params, network.apply


def make_classifier_loss(classifier):

    @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, None))
    def mae_loss(params, key, g, l, w, h, labels, is_training):
        y = classifier(params, key, g, l, w, h, is_training)
        return jnp.abs(y - labels)
    
    def loss_fn(params, key, g, l, w, h, labels, is_training):
        loss = jnp.mean(mae_loss(params, key, g, l, w, h, labels, is_training))
        return loss
    
    return loss_fn


def get_labels(csv_file, label_col):
    data = pd.read_csv(csv_file)
    labels = data[label_col].values
    labels = jnp.array(labels, dtype=float)
    return labels


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


if __name__  == "__main__":
    from jax.flatten_util import ravel_pytree

    from utils import GLXYZAW_from_file
    from transformer import make_transformer  
    from wyckoff import mult_table

    def get_inputs(key, batchsize, train_data, params, state, transformer):

        train_G, train_L, train_X, train_A, train_W = train_data 
        num_samples = len(train_L)
        num_batches = math.ceil(num_samples / batchsize)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)

            G = train_G[start_idx:end_idx]
            L = train_L[start_idx:end_idx]
            XYZ = train_X[start_idx:end_idx]
            A = train_A[start_idx:end_idx]
            W = train_W[start_idx:end_idx]

            M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) # (batchsize, n_max)

            key, subkey = jax.random.split(key)

            h, state = jax.vmap(transformer, in_axes=(None, None, None, 0, 0, 0, 0, 0, None))(params, state, subkey, G, XYZ, A, W, M, False)

            last_hidden_state = state['~']['last_hidden_state']
            _g_embeddings = state['~']['_g_embeddings']
            if batch_idx == 0:
                train_inputs = last_hidden_state
                g_embeddings = _g_embeddings
            else:
                train_inputs = jnp.concatenate([train_inputs, last_hidden_state], axis=0)
                g_embeddings = jnp.concatenate([g_embeddings, _g_embeddings], axis=0)

        # save the train_inputs
        print(train_inputs.shape)
        return g_embeddings, train_inputs


    key = jax.random.PRNGKey(42)
    mode = "train"

    if mode == "data":

        train_path = "/data/zdcao/crystal_gpt/dataset/mp_20/train.csv"
        valid_path = "/data/zdcao/crystal_gpt/dataset/mp_20/val.csv"
        test_path = "/data/zdcao/crystal_gpt/dataset/mp_20/test.csv"
        atom_types = 119
        wyck_types = 28
        n_max = 21
        num_io_process = 40
        Nf = 5
        Kx = 16
        Kl = 4
        h0_size = 256
        transformer_layers = 16
        num_heads = 16
        key_size = 64
        model_size = 64
        embed_size = 32
        dropout_rate = 0.5
        restore_path = "/home/zdcao/pipeline_crystalgpt/crystal_gpt/experimental/"

        batchsize = 2000

        train_data = GLXYZAW_from_file(train_path, atom_types, wyck_types, n_max, num_io_process)
        valid_data = GLXYZAW_from_file(valid_path, atom_types, wyck_types, n_max, num_io_process)
        test_data = GLXYZAW_from_file(test_path, atom_types, wyck_types, n_max, num_io_process)


        params, state, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 
                                                      h0_size, 
                                                      transformer_layers, num_heads, 
                                                      key_size, model_size, embed_size, 
                                                      atom_types, wyck_types,
                                                      dropout_rate)

        print("\n========== Load checkpoint==========")
        ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path) 
        if ckpt_filename is not None:
            print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
            ckpt = checkpoint.load_data(ckpt_filename)
            params = ckpt["params"]
        else:
            print("No checkpoint file found. Start from scratch.")

        
        key, subkey = jax.random.split(key)
        train_g_embeddings, train_inputs = get_inputs(subkey, batchsize, train_data, params, state, transformer)
        train_targets = get_labels(train_path, 'band_gap')
        _, train_L, _, _, train_W = train_data
        jnp.savez("train_data.npz", g=train_g_embeddings,
                                    l=train_L, w=train_W,
                                    inputs=train_inputs, targets=train_targets)

        key, subkey = jax.random.split(key)
        valid_g_embeddings, valid_inputs = get_inputs(subkey, batchsize, valid_data, params, state, transformer)
        valid_targets = get_labels(valid_path, 'band_gap')
        _, valid_L, _, _, valid_W = valid_data
        jnp.savez("valid_data.npz", g=valid_g_embeddings,
                                    l=valid_L, w=valid_W, 
                                    inputs=valid_inputs, targets=valid_targets)

        key, subkey = jax.random.split(key)
        test_g_embeddings, test_inputs = get_inputs(subkey, batchsize, test_data, params, state, transformer)
        test_targets = get_labels(test_path, 'band_gap')
        _, test_L, _, _, test_W = test_data
        jnp.savez("test_data.npz", g=test_g_embeddings,
                                   l=test_L, w=test_W,
                                   inputs=test_inputs, targets=test_targets)

    elif mode == "train":
        
        train_path = '/data/zdcao/crystal_gpt/dataset/mp_20/classifier/bandgap/train_data.npz'
        valid_path = '/data/zdcao/crystal_gpt/dataset/mp_20/classifier/bandgap/valid_data.npz'
        n_max = 21
        sequence_length = 105
        outputs_size = 64
        hidden_sizes = [128, 128, 64]
        num_classes = 1
        restore_path = "/data/zdcao/crystal_gpt/classifier/"
        lr = 1e-4
        lr_decay = 0
        epochs = 1000
        batchsize = 256

        data = jnp.load(train_path)
        train_data = data['g'], data['l'], data['w'], data['inputs'], data['targets']

        data = jnp.load(valid_path)
        valid_data = data['g'], data['l'], data['w'], data['inputs'], data['targets']

        print(train_data[0].shape, train_data[1].shape)
        print(valid_data[0].shape, valid_data[1].shape)

        key, subkey = jax.random.split(key)
        params, classifier = make_classifier(subkey,
                                             n_max=n_max,
                                             sequence_length=sequence_length,
                                             outputs_size=outputs_size,
                                             hidden_sizes=hidden_sizes,
                                             num_classes=num_classes)

        print ("# of classifier params", ravel_pytree(params)[0].size) 

        loss_fn = make_classifier_loss(classifier)

        print("\n========== Prepare logs ==========")

        output_path = os.path.dirname(restore_path)
        print("Will output samples to: %s" % output_path)

        print("\n========== Load checkpoint==========")
        ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path or output_path) 
        if ckpt_filename is not None:
            print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
            ckpt = checkpoint.load_data(ckpt_filename)
            params = ckpt["params"]
        else:
            print("No checkpoint file found. Start from scratch.")

        schedule = lambda t: lr/(1+lr_decay*t)
        # optimizer = optax.sgd(learning_rate=schedule, momentum=0.9)
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(params)

        print("\n========== Start training ==========")
        key, subkey = jax.random.split(key)
        params, opt_state = train(subkey, optimizer, opt_state, loss_fn, params, epoch_finished, epochs, batchsize, train_data, valid_data, output_path)

    elif mode == "test":
        raise NotImplementedError

