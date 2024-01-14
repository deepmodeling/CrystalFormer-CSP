import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree
import optax
import os

from utils import LXA_from_file
from realnvp import make_flow
from transformer import make_transformer  
from train import train
from sample import sample_crystal
from loss import make_loss_fn
import checkpoint

import argparse
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('training parameters')
group.add_argument('--epochs', type=int, default=100000, help='')
group.add_argument('--batchsize', type=int, default=100, help='')
group.add_argument('--lr', type=float, default=1e-3, help='learning rate')
group.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument("--optimizer", type=str, default="adamw", choices=["none", "adam", "adamw"], help="optimizer type")

group.add_argument("--folder", default="../data/", help="the folder to save data")
group.add_argument("--restore_path", default=None, help="checkpoint path or file")

group = parser.add_argument_group('dataset')
group.add_argument('--train_path', default='/home/wanglei/cdvae/data/carbon_24/train.csv', help='')
group.add_argument('--valid_path', default='/home/wanglei/cdvae/data/carbon_24/val.csv', help='')

group = parser.add_argument_group('flow parameters')
group.add_argument('--flow_layers', type=int, default=4, help='The number of layers in flow')
group.add_argument('--hidden_size', type=int, default=64, help='The number of hidden size')

group = parser.add_argument_group('transformer parameters')
group.add_argument('--transformer_layers', type=int, default=4, help='The number of layers in transformer')
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
valid_data = LXA_from_file(args.valid_path, args.atom_types, args.n_max, args.dim)

################### Model #############################
flow_params, flow, flow_sample_fn = make_flow(key, 6, args.hidden_size, args.flow_layers)

flow_name = 'f_%d_h_%d'%(args.flow_layers, args.hidden_size)
print ("# of flow params", ravel_pytree(flow_params)[0].size) 

transformer_params, transformer = make_transformer(key, args.transformer_layers, args.num_heads, 
                                      args.key_size, args.model_size, 
                                      args.atom_types)
transformer_name = 'l_%d_h_%d_k_%d_m_%d'%(args.transformer_layers, args.num_heads, args.key_size, args.model_size)

print ("# of transformer params", ravel_pytree(transformer_params)[0].size) 

params = flow_params, transformer_params
################### Train #############################

loss_fn = make_loss_fn(args.n_max, flow, transformer)

train_data = jax.tree_map(lambda x : x[:6000], train_data)
valid_data = jax.tree_map(lambda x : x[:2000], valid_data)

print("\n========== Prepare logs ==========")
path = args.folder + args.optimizer+"_bs_%d_lr_%g" % (args.batchsize, args.lr) \
                   + ("_wd_%g"%(args.weight_decay) if args.optimizer == "adamw" else "") \
                   + "_" + flow_name + "_" + transformer_name
os.makedirs(path, exist_ok=True)
print("Create directory: %s" % path)

print("\n========== Load checkpoint==========")
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path or path) 
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    params = ckpt["params"]
else:
    print("No checkpoint file found. Start from scratch.")

if args.optimizer != "none":

    if args.optimizer == 'adam':
        optimizer = optax.adam(args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
 
    print("\n========== Start training ==========")
    params = train(key, optimizer, loss_fn, params, epoch_finished, args.epochs, args.batchsize, train_data, valid_data, path)

else:
    L, X, A = valid_data
    flow_params, transformer_params = params
    outputs = jax.vmap(transformer, (None, 0, 0, 0), 0)(transformer_params, L[:5], X[:5], A[:5])
    mu, kappa, logit = jnp.split(outputs, [args.dim, 2*args.dim], axis=-1) 
    print (A[:5])
    print (jnp.exp(logit))

    print("\n========== Start sampling ==========")
    L, X, A = sample_crystal(key, flow_sample_fn, transformer, params, args.n_max, args.dim, args.batchsize)
    print (L)
    print (A)
    print (X)
