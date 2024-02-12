import jax 
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree
import optax
import os
import multiprocessing
import math

from utils import GLXAW_from_file, GLXA_to_csv
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
group.add_argument("--optimizer", type=str, default="adamw", choices=["none", "adam", "adamw"], help="optimizer type")

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
group.add_argument('--h0_size', type=int, default=0, help='hidden layer dimension for the first atom, 0 means we simply use a table for first aw_logit')
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
group.add_argument('--num_io_process', type=int, default=10, help='number of process used in multiprocessing io')
group.add_argument('--num_samples', type=int, default=1, help='number of test samples')
group.add_argument('--output_filename', type=str, default='output.csv', help='outfile to save sampled structures')

args = parser.parse_args()

key = jax.random.PRNGKey(42)

num_cpu = multiprocessing.cpu_count()
print('number of available cpu: ', num_cpu)
if args.num_io_process > num_cpu:
    print('num_io_process should not exceed number of available cpu, reset to ', num_cpu)
    args.num_io_process = num_cpu


################### Data #############################
if args.optimizer != "none":
    train_data = GLXAW_from_file(args.train_path, args.atom_types, args.wyck_types, args.n_max, args.dim, args.num_io_process)
    valid_data = GLXAW_from_file(args.valid_path, args.atom_types, args.wyck_types, args.n_max, args.dim, args.num_io_process)
else:
    assert (args.spacegroup is not None) # for inference we need to specify space group
    test_data = GLXAW_from_file(args.test_path, args.atom_types, args.wyck_types, args.n_max, args.dim, args.num_io_process)
    
    aw_types = (args.atom_types -1)*(args.wyck_types -1) + 1
    if args.elements is not None:
        idx = [element_dict[e] for e in args.elements]
        aw_mask = [1] + [1 if ((aw-1)%(args.atom_types-1)+1 in idx) else 0 for aw in range(1, aw_types)]
        aw_mask = jnp.array(aw_mask)
        print ('sampling structure formed by these elements:', args.elements)
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
if args.optimizer != "none" or args.restore_path is None:
    output_path = args.folder + args.optimizer+"_bs_%d_lr_%g_decay_%g_clip_%g" % (args.batchsize, args.lr, args.lr_decay, args.clip_grad) \
                   + '_A_%g_W_%g_N_%g'%(args.atom_types, args.wyck_types, args.n_max) \
                   + ("_wd_%g"%(args.weight_decay) if args.optimizer == "adamw" else "") \
                   +  "_" + transformer_name 

    os.makedirs(output_path, exist_ok=True)
    print("Create directory for output: %s" % output_path)
else:
    output_path = os.path.dirname(args.restore_path)
    print("Will output samples to: %s" % output_path)


print("\n========== Load checkpoint==========")
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path or output_path) 
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
    params, opt_state = train(key, optimizer, opt_state, loss_fn, params, epoch_finished, args.epochs, args.batchsize, train_data, valid_data, output_path)

else:

    print("\n========== Print out some test data ==========")
    import numpy as np 
    np.set_printoptions(threshold=np.inf)

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

    print("\n========== Start sampling ==========")
    jax.config.update("jax_enable_x64", True) # to get off compilation warning, and to prevent sample nan lattice 
    '''
    FYI, the error was [Compiling module extracted] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
    '''

    num_batches = math.ceil(args.num_samples / args.batchsize)
    name, extension = args.output_filename.rsplit('.', 1)
    filename = os.path.join(output_path, 
                            f"{name}_{args.spacegroup}.{extension}")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batchsize
        end_idx = min(start_idx + args.batchsize, args.num_samples)
        n_sample = end_idx - start_idx
        key, subkey = jax.random.split(key)
        X, A, W, M, L, AW = sample_crystal(subkey, transformer, params, args.n_max, args.dim, n_sample, args.atom_types, args.wyck_types, args.Kx, args.Kl, args.spacegroup, aw_mask, args.temperature)
        print ("X:\n", X)
        print ("A:\n", A)  # atom type
        print ("W:\n", W)  # Wyckoff positions
        print ("M:\n", M)
        print ("N:\n", M.sum(axis=-1))
        print ("L:\n", L)  # sampled lattice
        for a in A:
           print([element_list[i] for i in a])
        #print ("AW:\n", AW)

        GLXA_to_csv(args.spacegroup, L, X, A, num_worker=args.num_io_process, filename=filename)
        print ("Wrote samples to %s"%filename)
