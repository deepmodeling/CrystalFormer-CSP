import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import pandas as pd
import numpy as np
from ast import literal_eval

from config import *
from _loss import make_classifier_loss, make_cond_logp
from _model import make_classifier
from _transformer import make_transformer as make_transformer_with_state
from _mcmc import make_mcmc_step

import checkpoint
from wyckoff import mult_table
from loss import make_loss_fn
from transformer import make_transformer


if __name__  == "__main__":
    key = jax.random.PRNGKey(42)

    atom_types = 119
    wyck_types = 28
    n_max = 21
    num_io_process = 40
    Nf = 5
    Kx = 16
    Kl = 4
    h0_size = 256
    transformer_layers = 4
    num_heads = 8
    key_size = 32
    model_size = 64
    embed_size = 32
    dropout_rate = 0.3

    sequence_length = 105
    outputs_size = 64
    hidden_sizes = [128, 128, 64]
    num_classes = 1
    restore_path = "/data/zdcao/crystal_gpt/classifier/"
    lr = 1e-4
    epochs = 1000
    batchsize = 256

    ################### Load Classifier Model #############################
    transformer_params, state, cond_transformer = make_transformer_with_state(key, Nf, Kx, Kl, n_max, 
                                                                              h0_size, 
                                                                              transformer_layers, num_heads, 
                                                                              key_size, model_size, embed_size, 
                                                                              atom_types, wyck_types,
                                                                              dropout_rate)
    print ("# of transformer params", ravel_pytree(transformer_params)[0].size) 
    
    key, subkey = jax.random.split(key)
    classifier_params, classifier = make_classifier(subkey,
                                                    n_max=n_max,
                                                    embed_size=embed_size,
                                                    sequence_length=sequence_length,
                                                    outputs_size=outputs_size,
                                                    hidden_sizes=hidden_sizes,
                                                    num_classes=num_classes)

    print ("# of classifier params", ravel_pytree(classifier_params)[0].size) 

    cond_params = (transformer_params, classifier_params)

    print("\n========== Prepare logs ==========")
    output_path = os.path.dirname(restore_path)
    print("Will output samples to: %s" % output_path)

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        cond_params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")
    
    _, forward_fn = make_classifier_loss(cond_transformer, classifier)

    ################### Load BASE Model #############################
    transformer_layers = 16
    num_heads = 16
    key_size = 64
    dropout_rate = 0.5
    restore_path = "/home/zdcao/pipeline_crystalgpt/crystal_gpt/experimental/"

    base_params, base_transformer = make_transformer(key, Nf, Kx, Kl, n_max, 
                                                     h0_size, 
                                                     transformer_layers, num_heads, 
                                                     key_size, model_size, embed_size, 
                                                     atom_types, wyck_types,
                                                     dropout_rate)
    print ("# of transformer params", ravel_pytree(base_params)[0].size) 

    _, logp_fn = make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, base_transformer)

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        base_params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    ################### Conditional Generation ############################
    forward = jax.vmap(forward_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, None))
    cond_logp_fn = make_cond_logp(logp_fn, forward, 
                                  target=jnp.array(-3),
                                  alpha=10)

    print("\n========== Load sampled data ==========")
    spg = 225
    csv_file = f"/home/zdcao/pipeline_crystalgpt/crystal_gpt/experimental/output_{spg}.csv"
    origin_data = pd.read_csv(csv_file)
    L, XYZ, A, W = origin_data['L'], origin_data['X'], origin_data['A'], origin_data['W']
    L = L.apply(lambda x: literal_eval(x))
    XYZ = XYZ.apply(lambda x: literal_eval(x))
    A = A.apply(lambda x: literal_eval(x))
    W = W.apply(lambda x: literal_eval(x))

    # convert array of list to numpy ndarray
    G = jnp.array([spg]*len(L))
    L = jnp.array(L.tolist())
    XYZ = jnp.array(XYZ.tolist())
    A = jnp.array(A.tolist())
    W = jnp.array(W.tolist())

    M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) # (batchsize, n_max)
    num_atoms = jnp.sum(M, axis=1)
    length, angle = jnp.split(L, 2, axis=-1)
    length = length/num_atoms[:, None]**(1/3)
    angle = angle * (jnp.pi / 180) # to rad
    L = jnp.concatenate([length, angle], axis=-1)

    print(G.shape, L.shape, XYZ.shape, A.shape, W.shape)

    print("\n========== Start MCMC ==========")
    mcmc = make_mcmc_step(base_params, cond_params, state, n_max=n_max, atom_types=atom_types)

    mc_steps = 2300
    mc_width = 0.1
    x = (G, L, XYZ, A, W)

    print("====== before mcmc =====")
    print ("XYZ:\n", XYZ)  # fractional coordinate 
    print ("A:\n", A)  # element type
    print ("W:\n", W)  # Wyckoff positions
    print ("L:\n", L)  # lattice

    key, subkey = jax.random.split(key)
    x, acc = mcmc(cond_logp_fn, x_init=x, key=subkey, mc_steps=mc_steps, mc_width=mc_width)
    print("acc", acc)

    G, L, XYZ, A, W = x

    key, subkey = jax.random.split(key)
    logp_new = jax.jit(cond_logp_fn, static_argnums=9)(base_params, cond_params, state, subkey, G, L, XYZ, A, W, False)

    print("====== after mcmc =====")
    M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) 
    num_atoms = jnp.sum(M, axis=1)

    #scale length according to atom number since we did reverse of that when loading data
    length, angle = jnp.split(L, 2, axis=-1)
    length = length*num_atoms[:, None]**(1/3)
    angle = angle * (180.0 / jnp.pi) # to deg
    L = jnp.concatenate([length, angle], axis=-1)

    print ("XYZ:\n", XYZ)  # fractional coordinate 
    print ("A:\n", A)  # element type
    print ("W:\n", W)  # Wyckoff positions
    print ("L:\n", L)  # lattice

    data = pd.DataFrame()
    data['L'] = np.array(L).tolist()
    data['X'] = np.array(XYZ).tolist()
    data['A'] = np.array(A).tolist()
    data['W'] = np.array(W).tolist()
    data['M'] = np.array(M).tolist()
    data['logp'] = np.array(logp_new).tolist()

    filename = f'./cond_ouput_{spg}.csv'
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)

    print ("Wrote samples to %s"%filename)
