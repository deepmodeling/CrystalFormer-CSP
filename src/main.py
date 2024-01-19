import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree
import optax
import os

from utils import GLXAM_from_file
from elements import element_dict, element_list
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
group.add_argument('--lr', type=float, default=1e-4, help='learning rate')
group.add_argument('--lr_decay', type=float, default=1e-5, help='lr decay')
group.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
group.add_argument('--clip_grad', type=float, default=1.0, help='clip gradient')
parser.add_argument("--optimizer", type=str, default="adamw", choices=["none", "adam", "adamw"], help="optimizer type")

group.add_argument("--folder", default="../data/", help="the folder to save data")
group.add_argument("--restore_path", default=None, help="checkpoint path or file")

group = parser.add_argument_group('dataset')
group.add_argument('--train_path', default='/home/wanglei/cdvae/data/perov_5/train.csv', help='')
group.add_argument('--valid_path', default='/home/wanglei/cdvae/data/perov_5/val.csv', help='')
group.add_argument('--test_path', default='/home/wanglei/cdvae/data/perov_5/test.csv', help='')

group = parser.add_argument_group('transformer parameters')
group.add_argument('--K', type=int, default=8, help='number of modes in von-mises')
group.add_argument('--h0_size', type=int, default=512, help='hidden layer dimension for the first atom')
group.add_argument('--transformer_layers', type=int, default=4, help='The number of layers in transformer')
group.add_argument('--num_heads', type=int, default=8, help='The number of heads')
group.add_argument('--key_size', type=int, default=32, help='The key size')
group.add_argument('--model_size', type=int, default=8, help='The model size')

group = parser.add_argument_group('physics parameters')
group.add_argument('--n_max', type=int, default=5, help='The maximum number of atoms in the cell')
group.add_argument('--atom_types', type=int, default=118, help='Atom types including the padded atoms')
group.add_argument('--mult_types', type=int, default=15, help='Number of possible multiplicites including 0')
group.add_argument('--dim', type=int, default=3, help='The spatial dimension')
group.add_argument('--spacegroup', type=int, nargs='+', help='The space group id to be sampled (1-230), e.g., 25, 99, 221')
group.add_argument('--elements', type=str, default=None, nargs='+', help='name of the chemical elemenets, e.g. Bi, Ti, O')

args = parser.parse_args()

key = jax.random.PRNGKey(42)


################### Data #############################
if args.optimizer != "none":
    train_data = GLXAM_from_file(args.train_path, args.atom_types, args.mult_types, args.n_max, args.dim)
    valid_data = GLXAM_from_file(args.valid_path, args.atom_types, args.mult_types, args.n_max, args.dim)
else:
    test_data = GLXAM_from_file(args.test_path, args.atom_types, args.mult_types, args.n_max, args.dim)
    
    am_types = (args.atom_types -1)*(args.mult_types -1) + 1
    if args.elements is not None:
        idx = [element_dict[e] for e in args.elements]
        am_mask = [1] + [1 if ((i-1)%(args.atom_types-1)+1 in idx) else 0 for i in range(1, am_types)]
        am_mask = jnp.array(am_mask)
    else:
        am_mask = jnp.zeros((am_types), dtype=int)
    print (args.elements)
    print (am_mask)

################### Model #############################
params, transformer = make_transformer(key, args.K, args.h0_size, 
                                      args.transformer_layers, args.num_heads, 
                                      args.key_size, args.model_size, 
                                      args.atom_types, args.mult_types)
transformer_name = 'K_%d_h0_%d_l_%d_H_%d_k_%d_m_%d'%(args.K, args.h0_size, args.transformer_layers, args.num_heads, args.key_size, args.model_size)

print ("# of transformer params", ravel_pytree(params)[0].size) 

################### Train #############################

loss_fn = make_loss_fn(args.n_max, args.atom_types, args.mult_types, args.K, transformer)

print("\n========== Prepare logs ==========")
path = args.folder + args.optimizer+"_bs_%d_lr_%g_decay_%g_clip_%g" % (args.batchsize, args.lr, args.lr_decay, args.clip_grad) \
                   + '_A_%g_M_%g_N_%g'%(args.atom_types, args.mult_types, args.n_max) \
                   + ("_wd_%g"%(args.weight_decay) if args.optimizer == "adamw" else "") \
                   +  "_" + transformer_name 
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

    schedule = lambda t: args.lr/(1+args.lr_decay*t)

    if args.optimizer == "adam":
        optimizer = optax.chain(optax.clip_by_global_norm(args.clip_grad), 
                                optax.scale_by_adam(), 
                                optax.scale_by_schedule(schedule), 
                                optax.scale(-1.))
    elif args.optimizer == 'adamw':
        optimizer = optax.chain(optax.clip(args.clip_grad),
                                optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
                               )

    opt_state = optimizer.init(params)
    try:
        opt_state.update(ckpt["opt_state"])
    except: 
        print ("failed to update opt_state from checkpoint")
        pass 
 
    print("\n========== Start training ==========")
    params, opt_state = train(key, optimizer, opt_state, loss_fn, params, epoch_finished, args.epochs, args.batchsize, train_data, valid_data, path)

else:
    print("\n========== Inference on test data ==========")
    G, L, X, AM = test_data

    from lattice import make_spacegroup_mask
    spacegroup_mask = jax.vmap(make_spacegroup_mask)(jnp.argmax(G, axis=-1)+1) # first convert one-hot to integer rep, then look for mask
    
    from utils import to_A_M, mult_list
    mult_table = jnp.array(mult_list[:args.mult_types])

    A, M = jax.vmap(to_A_M, (0, None))(AM, args.atom_types)
    num_sites, num_atoms = jnp.sum(A!=0, axis=1), jnp.sum(mult_table[M], axis=1)
    print (num_sites)
    print (num_atoms)

    
    batchsize = 10
    for a in A[:batchsize]: 
       print([element_list[i] for i in a])
    print (M[:batchsize])
    outputs = jax.vmap(transformer, (None, 0, 0, 0), (0))(params, G[:batchsize], X[:batchsize], AM[:batchsize])
    print (outputs.shape)
    mu, sigma = jnp.split(outputs[jnp.arange(batchsize), num_sites[:batchsize], args.K+2*args.K*args.dim+am_types:], 2, axis=-1)
    print (spacegroup_mask[:batchsize])
    print (jnp.argmax(G, axis=-1)[:batchsize]+1)
    print (L[:batchsize])
    print (mu)
    print (sigma)

    print("\n========== Start sampling ==========")
    G = jnp.array(args.spacegroup)
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(subkey, jnp.arange(len(G)), shape=(args.batchsize,))
    G = G[idx]
    print (G) 
    spacegroup_mask = jax.vmap(make_spacegroup_mask)(G) 
    X, A, M, L = sample_crystal(key, transformer, params, args.n_max, args.dim, args.batchsize, args.atom_types, args.mult_types, args.K, G, am_mask)
    print (X)
    print (A)  # atom type
    print (mult_table[M])  # mutiplicities 

    num_sites, num_atoms = jnp.sum(A!=0, axis=1), jnp.sum(mult_table[M], axis=1)
    print (num_sites)
    print (num_atoms)
    
    print (L)  # sampled lattice

    for a in A: 
       print([element_list[i] for i in a])
