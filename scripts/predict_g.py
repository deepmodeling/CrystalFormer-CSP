import math
import jax 
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from functools import partial

from crystalformer.src.loss import topk_recall
from crystalformer.src.utils import GLXYZAW_from_file
from crystalformer.src.formula import find_composition_vector 
from crystalformer.src.wyckoff import mult_table
from crystalformer.src.transformer import make_transformer
import crystalformer.src.checkpoint as checkpoint

def main(args):

    num_devices = jax.local_device_count()
    batch_per_device = args.batchsize // num_devices
    shape_prefix = (num_devices, batch_per_device)
    print("num_devices: ", num_devices)
    print("batch_per_device: ", batch_per_device)
    print("shape_prefix: ", shape_prefix)

    key = jax.random.PRNGKey(42)

    params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                      args.h0_size, 
                                      args.transformer_layers, args.num_heads, 
                                      args.key_size, args.model_size, args.embed_size, 
                                      args.atom_types, args.wyck_types,
                                      args.dropout_rate, args.attn_dropout)

    print ("# of transformer params", ravel_pytree(params)[0].size) 

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    valid_data = GLXYZAW_from_file(args.valid_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
    G, L, XYZ, A, W = valid_data

    @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, None), out_axes=0) # batch 
    def g_logit_fn(params, key, G, L, XYZ, A, W, is_train):
        '''
        G: scalar 
        L: (6,) [a, b, c, alpha, beta, gamma] 
        XYZ: (n_max, 3)
        A: (n_max,)
        W: (n_max,)
        '''
        num_sites = jnp.sum(A!=0)
        M = mult_table[G-1, W]  # (n_max,) multplicities
        composition = find_composition_vector(A, M) # (atom_types, )
        g_logit, h = transformer(params, key, composition, G, XYZ, A, W, M, is_train) # (5*n_max+1, ...)
        return g_logit

    ks = [1,10,30,40,50]
    def compute_recall(data):
        G, L, XYZ, A, W = data
        g_logit = g_logit_fn(params, key, G, L, XYZ, A, W, False)
        return topk_recall(g_logit, G-1, ks)
    
    total_recall = {k: 0.0 for k in ks}

    num_samples = G.shape[0]
    if num_samples % args.batchsize == 0:
        num_batches = math.ceil(num_samples / args.batchsize)
    else:
        num_batches = math.ceil(num_samples / args.batchsize) - 1

    for batch_idx in range(num_batches):
        
        start_idx = batch_idx * args.batchsize
        end_idx = min(start_idx + args.batchsize, num_samples)
        data = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], valid_data)
        data = jax.tree_util.tree_map(lambda x: x.reshape(shape_prefix + x.shape[1:]), data)

        recall = jax.pmap(compute_recall)(data)
        
        total_recall = jax.tree_util.tree_map(
                 lambda acc, i: acc + jnp.mean(i),
                 total_recall, 
                 recall
                 )

    total_recall = jax.tree_util.tree_map(
                 lambda x: x/num_batches, 
                 total_recall
                 ) 
    print ('recall on validation data', total_recall)

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--restore_path", default="/data/wanglei/crystalgpt/mp-mpsort-xyz-embed/w-a-x-y-z-periodic-fixed-size-embed-eb630/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_8_H_8_k_32_m_64_e_32_drop_0.3/", help="")
    parser.add_argument("--batchsize", type=int, default=100, help="batch size")
    parser.add_argument("--valid_path", default='/opt/data/bcmdata/ZONES/data/PROJECTS/datafile/PRIVATE/zdcao/crystal_gpt/dataset/alex/PBE_20241204/val.lmdb')
    parser.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
    group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
    group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

    group = parser.add_argument_group('transformer parameters')
    group.add_argument('--Nf', type=int, default=5, help='number of frequencies for fc')
    group.add_argument('--Kx', type=int, default=16, help='number of modes in x')
    group.add_argument('--Kl', type=int, default=4, help='number of modes in lattice')
    group.add_argument('--h0_size', type=int, default=256, help='hidden layer dimension for the g and w of first atom')
    group.add_argument('--transformer_layers', type=int, default=16, help='The number of layers in transformer')
    group.add_argument('--num_heads', type=int, default=16, help='The number of heads')
    group.add_argument('--key_size', type=int, default=64, help='The key size')
    group.add_argument('--model_size', type=int, default=64, help='The model size')
    group.add_argument('--embed_size', type=int, default=32, help='The enbedding size')
    group.add_argument('--dropout_rate', type=float, default=0.1, help='The dropout rate for MLP')
    group.add_argument('--attn_dropout', type=float, default=0.1, help='The dropout rate for attention')
    args = parser.parse_args()

    main(args)




