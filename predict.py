import os
import pandas as pd
import jax
import jax.numpy as jnp
from ast import literal_eval

from crystalformer.extension.model import make_classifier
from crystalformer.extension.transformer import make_transformer  
# from classifier import get_labels
from crystalformer.extension.loss import make_classifier_loss

from crystalformer.src.utils import GLXYZAW_from_file
import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.wyckoff import mult_table

from jax.flatten_util import ravel_pytree

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('dataset')
group.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
group.add_argument('--spacegroup', type=int, help='The space group number')
group.add_argument('--input_path', type=str, help='The path to the data need to be predicted')
group.add_argument('--output_path', type=str, default='./predict.npy', help='The path to save the prediction result')

group = parser.add_argument_group('physics parameters')
group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

group = parser.add_argument_group('transformer parameters')
group.add_argument('--Nf', type=int, default=5, help='number of frequencies for fc')
group.add_argument('--Kx', type=int, default=16, help='number of modes in x')
group.add_argument('--Kl', type=int, default=4, help='number of modes in lattice')
group.add_argument('--h0_size', type=int, default=256, help='hidden layer dimension for the first atom, 0 means we simply use a table for first aw_logit')
group.add_argument('--transformer_layers', type=int, default=4, help='The number of layers in transformer')
group.add_argument('--num_heads', type=int, default=8, help='The number of heads')
group.add_argument('--key_size', type=int, default=32, help='The key size')
group.add_argument('--model_size', type=int, default=64, help='The model size')
group.add_argument('--embed_size', type=int, default=32, help='The enbedding size')
group.add_argument('--dropout_rate', type=float, default=0.3, help='The dropout rate')


group = parser.add_argument_group('classifier parameters')
group.add_argument('--sequence_length', type=int, default=105, help='The sequence length')
group.add_argument('--outputs_size', type=int, default=64, help='The outputs size')
group.add_argument('--hidden_sizes', type=str, default='128,128,64' , help='The hidden sizes')
group.add_argument('--num_classes', type=int, default=1, help='The number of classes')
group.add_argument('--restore_path', type=str, default="/data/zdcao/crystal_gpt/classifier/", help='The restore path')

args = parser.parse_args()
key = jax.random.PRNGKey(42)

################### Data #############################
### read from cif files
# test_data = GLXYZAW_from_file(test_path, atom_types, wyck_types, n_max, num_io_process)
# test_labels = get_labels(test_path, "formation_energy_per_atom")  # band_gap or formation_energy_per_atom
# test_data = (*test_data, test_labels)

### read from generated data
test_data = pd.read_csv(args.input_path)
L, XYZ, A, W = test_data['L'], test_data['X'], test_data['A'], test_data['W']
L = L.apply(lambda x: literal_eval(x))
XYZ = XYZ.apply(lambda x: literal_eval(x))
A = A.apply(lambda x: literal_eval(x))
W = W.apply(lambda x: literal_eval(x))

# convert array of list to numpy ndarray
G = jnp.array([int(args.spacegroup)]*len(L))
L = jnp.array(L.tolist())
XYZ = jnp.array(XYZ.tolist())
A = jnp.array(A.tolist())
W = jnp.array(W.tolist())

# rescale the length and angle
M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) # (batchsize, n_max)
num_atoms = jnp.sum(M, axis=1)
length, angle = jnp.split(L, 2, axis=-1)
length = length/num_atoms[:, None]**(1/3)
angle = angle * (jnp.pi / 180) # to rad
L = jnp.concatenate([length, angle], axis=-1)
test_data = (G, L, XYZ, A, W)


################### Model #############################
transformer_params, state, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                                          args.h0_size, 
                                                          args.transformer_layers, args.num_heads, 
                                                          args.key_size, args.model_size, args.embed_size, 
                                                          args.atom_types, args.wyck_types,
                                                          args.dropout_rate)
print ("# of transformer params", ravel_pytree(transformer_params)[0].size) 

key, subkey = jax.random.split(key)
classifier_params, classifier = make_classifier(subkey,
                                                n_max=args.n_max,
                                                embed_size=args.embed_size,
                                                sequence_length=args.sequence_length,
                                                outputs_size=args.outputs_size,
                                                hidden_sizes=[int(x) for x in args.hidden_sizes.split(',')],
                                                num_classes=args.num_classes)

print ("# of classifier params", ravel_pytree(classifier_params)[0].size) 

params = (transformer_params, classifier_params)

print("\n========== Prepare logs ==========")
output_path = os.path.dirname(args.restore_path)
print("Will output samples to: %s" % output_path)

print("\n========== Load checkpoint==========")
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path) 
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

jnp.save(args.output_path, y)
