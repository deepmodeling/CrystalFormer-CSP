import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree

from utils import LXA_from_file
from model import make_transformer  
from train import train
from loss import make_loss_fn

import argparse
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('training parameters')
group.add_argument('--epochs', type=int, default=10000, help='')
group.add_argument('--batchsize', type=int, default=100, help='')
group.add_argument('--lr', type=float, default=1e-3, help='learning rate')

group.add_argument("--folder", default="../data/", help="the folder to save data")
group.add_argument("--restore_path", default=None, help="checkpoint path or file")

group = parser.add_argument_group('dataset')
group.add_argument('--train_path', default='/home/wanglei/cdvae/data/carbon_24/train.csv', help='')

group = parser.add_argument_group('network parameters')
group.add_argument('--num_layers', type=int, default=4, help='The number of layers')
group.add_argument('--num_heads', type=int, default=8, help='The number of heads')
group.add_argument('--key_size', type=int, default=16, help='The key size')
group.add_argument('--model_size', type=int, default=32, help='The model size')

group = parser.add_argument_group('physics parameters')
group.add_argument('--n_max', type=int, default=24, help='The maximum number of atoms in the cell')
group.add_argument('--atom_types', type=int, default=2, help='Atom types including the padded atoms')
group.add_argument('--dim', type=int, default=3, help='The spatial dimension')

args = parser.parse_args()

key = jax.random.PRNGKey(42)


################### Data #############################
train_data = LXA_from_file(args.train_path, args.atom_types, args.n_max, args.dim)


################### Model #############################

params, model = make_transformer(key, args.num_layers, args.num_heads, 
                                      args.key_size, args.model_size, 
                                      args.atom_types)
print ("# of params", ravel_pytree(params)[0].size) # number of parameters in the model

################### Train #############################

loss_fn = make_loss_fn(args.atom_types, model)

train_data = jax.tree_map(lambda x : x[:6000], train_data)
params = train(key, loss_fn, params, args.epochs, args.lr, args.batchsize, train_data)
