import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree
import optax
import os

from utils import GLXAW_from_file
from elements import element_dict, element_list
from transformer import make_transformer  
from train import train
from sample import sample_crystal
from loss import make_loss_fn
import checkpoint
from wyckoff import mult_table

import argparse
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('training parameters')
group.add_argument('--epochs', type=int, default=1000000, help='')
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
group.add_argument('--Nf', type=int, default=5, help='number of frequencies')
group.add_argument('--Kx', type=int, default=8, help='number of modes in von-mises')
group.add_argument('--Kl', type=int, default=1, help='number of modes in lattice')
group.add_argument('--h0_size', type=int, default=512, help='hidden layer dimension for the first atom')
group.add_argument('--transformer_layers', type=int, default=4, help='The number of layers in transformer')
group.add_argument('--num_heads', type=int, default=8, help='The number of heads')
group.add_argument('--key_size', type=int, default=32, help='The key size')
group.add_argument('--model_size', type=int, default=8, help='The model size')
group.add_argument('--dropout_rate', type=float, default=0.1, help='The dropout rate')

group = parser.add_argument_group('physics parameters')
group.add_argument('--n_max', type=int, default=5, help='The maximum number of atoms in the cell')
group.add_argument('--atom_types', type=int, default=118, help='Atom types including the padded atoms')
group.add_argument('--wyck_types', type=int, default=15, help='Number of possible multiplicites including 0')
group.add_argument('--dim', type=int, default=3, help='The spatial dimension')

group = parser.add_argument_group('sampling parameters')
group.add_argument('--spacegroup', type=int, help='The space group id to be sampled (1-230)')
group.add_argument('--elements', type=str, default=None, nargs='+', help='name of the chemical elemenets, e.g. Bi, Ti, O')
group.add_argument('--temperature', type=float, default=1.0, help='temperature used for sampling')

args = parser.parse_args()

key = jax.random.PRNGKey(42)


################### Data #############################
if args.optimizer != "none":
    train_data = GLXAW_from_file(args.train_path, args.atom_types, args.wyck_types, args.n_max, args.dim)
    valid_data = GLXAW_from_file(args.valid_path, args.atom_types, args.wyck_types, args.n_max, args.dim)
else:
    assert (args.spacegroup is not None) # for inference we need to specify space group
    test_data = GLXAW_from_file(args.test_path, args.atom_types, args.wyck_types, args.n_max, args.dim)
    
    aw_types = (args.atom_types -1)*(args.wyck_types -1) + 1
    if args.elements is not None:
        idx = [element_dict[e] for e in args.elements]
        aw_mask = [1] + [1 if ((aw-1)%(args.atom_types-1)+1 in idx) else 0 for aw in range(1, aw_types)]
        aw_mask = jnp.array(aw_mask)
        print ('sampling strucrure formed by these elements:', args.elements)
        print (aw_mask)
    else:
        aw_mask = jnp.zeros((aw_types), dtype=int) # we will do nothing to aw_logit in sampling

################### Model #############################
params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, args.dim, 
                                      args.h0_size, 
                                      args.transformer_layers, args.num_heads, 
                                      args.key_size, args.model_size, 
                                      args.atom_types, args.wyck_types, 
                                      args.dropout_rate)
transformer_name = 'Nf_%d_K_%d_%d_h0_%d_l_%d_H_%d_k_%d_m_%d_drop_%g'%(args.Nf, args.Kx, args.Kl, args.h0_size, args.transformer_layers, args.num_heads, args.key_size, args.model_size, args.dropout_rate)

print ("# of transformer params", ravel_pytree(params)[0].size) 

################### Train #############################

loss_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer)

print("\n========== Prepare logs ==========")
path = args.folder + args.optimizer+"_bs_%d_lr_%g_decay_%g_clip_%g" % (args.batchsize, args.lr, args.lr_decay, args.clip_grad) \
                   + '_A_%g_W_%g_N_%g'%(args.atom_types, args.wyck_types, args.n_max) \
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
    G, L, X, AW = test_data
    print (G.shape, L.shape, X.shape, AW.shape)
    
    from utils import to_A_W
    A, W = jax.vmap(to_A_W, (0, None))(AW, args.atom_types)
    num_sites = jnp.sum(A!=0, axis=1)
    print ("num_sites:", num_sites)
    @jax.vmap
    def lookup(G, W):
        return mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    num_atoms = M.sum(axis=-1)
    print ("num_atoms:", num_atoms)

    batchsize = args.batchsize
    print ("G:", G[:batchsize])
    print ("A\n", A[:batchsize])
    for a in A[:batchsize]: 
       print([element_list[i] for i in a])
    print ("W\n",W[:batchsize])
    print ("X\n",X[:batchsize])

    aw_types = (args.atom_types -1)*(args.wyck_types -1) + 1
    xl_types = args.Kx+2*args.Kx*args.dim+args.Kl+2*6*args.Kl
    print ("aw_types, xl_types:", aw_types, xl_types)

    outputs = jax.vmap(transformer, (None, None, 0, 0, 0, 0, 0, None), (0))(params, key, G[:batchsize], X[:batchsize], A[:batchsize], W[:batchsize], M[:batchsize], False)
    print ("outputs.shape", outputs.shape)

    outputs = outputs.reshape(args.batchsize, args.n_max+1, 2, aw_types)
    aw_logit = outputs[:, :, 0, :] # (batchsize, n_max+1, aw_types)
    print ("aw_logit.shape", aw_logit.shape)

    # sample given ground truth
    key, key_aw = jax.random.split(key)
    AW_sample = jax.random.categorical(key_aw, aw_logit, axis=-1) # (batchsize, n_max+1, )
    A_sample, W_sample = jax.vmap(to_A_W, (0, None))(AW_sample, args.atom_types)
    print ("A_sample\n", A_sample)
    print ("W_sample\n", W_sample)

    outputs = outputs.reshape(batchsize, args.n_max+1, 2, aw_types)[:, :, 1, :]
    offset = args.Kx+2*args.Kx*args.dim 
    l_logit, mu, sigma = jnp.split(outputs[jnp.arange(batchsize), num_sites[:batchsize], 
                                                      offset:offset+args.Kl+2*6*args.Kl], 
                                                      [args.Kl, args.Kl+6*args.Kl], axis=-1)
    print (L[:batchsize])
    print (jnp.exp(l_logit))
    print (mu.reshape(batchsize, args.Kl, 6))
    print (sigma.reshape(batchsize, args.Kl, 6))
 
    print("\n========== Start sampling ==========")
    X, A, W, M, L, AW = sample_crystal(key, transformer, params, args.n_max, args.dim, args.batchsize, args.atom_types, args.wyck_types, args.Kx, args.Kl, args.spacegroup, aw_mask, args.temperature)
    print ("X:\n", X)
    print ("A:\n", A)  # atom type
    print ("W:\n", W)  # Wyckoff positions
    print ("M:\n", M)
    print ("N:\n", M.sum(axis=-1))
    print ("L:\n", L)  # sampled lattice
    for a in A: 
       print([element_list[i] for i in a])

    print ("AW:\n", AW)
