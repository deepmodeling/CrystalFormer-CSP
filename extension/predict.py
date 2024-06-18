import os
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from ast import literal_eval

from config import *
from _model import make_classifier
from _transformer import make_transformer  
from classifier import get_labels
from _loss import make_classifier_loss

from utils import GLXYZAW_from_file
import checkpoint
from wyckoff import mult_table

from jax.flatten_util import ravel_pytree

import warnings
warnings.filterwarnings("ignore")


key = jax.random.PRNGKey(42)
spg = 225 
test_path = f"/home/zdcao/pipeline_crystalgpt/crystal_gpt/cond_ouput_{spg}.csv"
# test_path = f"/home/zdcao/pipeline_crystalgpt/crystal_gpt/experimental/output_{spg}.csv"

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

################### Data #############################
### read from cif files
# test_data = GLXYZAW_from_file(test_path, atom_types, wyck_types, n_max, num_io_process)
# test_labels = get_labels(test_path, "formation_energy_per_atom")  # band_gap or formation_energy_per_atom
# test_data = (*test_data, test_labels)


### read from generated data
test_data = pd.read_csv(test_path)
L, XYZ, A, W = test_data['L'], test_data['X'], test_data['A'], test_data['W']
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
test_data = (G, L, XYZ, A, W)


################### Model #############################
transformer_params, state, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 
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

params = (transformer_params, classifier_params)

print("\n========== Prepare logs ==========")
output_path = os.path.dirname(restore_path)
print("Will output samples to: %s" % output_path)

print("\n========== Load checkpoint==========")
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path) 
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    _params = ckpt["params"]
else:
    print("No checkpoint file found. Start from scratch.")

if len(_params) == len(params):
    params = _params
else:
    params = (_params, params[1])  # only restore transformer params
    print("only restore transformer params")

_, forward_fn = make_classifier_loss(transformer, classifier)

if len(test_data) == 6:
    G, L, XYZ, A, W, labels = test_data
else:
    G, L, XYZ, A, W = test_data
    labels = None

y = jax.vmap(forward_fn,
             in_axes=(None, None, None, 0, 0, 0, 0, 0, None)
             )(params, state, key, G, L, XYZ, A, W, False)

jnp.save("./predict.npy", y)
