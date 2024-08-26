import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import yaml
import os
import optax
from functools import partial

from mace.calculators import mace_mp

from crystalformer.src.loss import make_loss_fn
from crystalformer.src.transformer import make_transformer
from crystalformer.src.sample import sample_crystal
import crystalformer.src.checkpoint as checkpoint

from crystalformer.reinforce.train import train
from crystalformer.reinforce.loss import make_reinforce_loss
from crystalformer.reinforce.reward import make_force_reward_fn

class config:
    def __init__(self, d=None):
        for key, value in d.items():
            setattr(self, key, value)

    def __print__(self):
        for key, value in self.__dict__.items():
            print(f"{key} : {value}")

################## Read config file ###################
# load config file
with open("./model/config.yaml") as stream:
    args = yaml.safe_load(stream)
args = config(args)

args.restore_path = "./data/"
args.folder = "./experimental/"
args.epochs = 10
args.optimizer = "adam"
args.transformer_layers = 4
args.num_heads = 8
args.key_size = 32
args.dropout_rate = 0.3

print("\n========== Config ==========")
args.__print__()

################### Model #############################
key = jax.random.PRNGKey(42)
params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                      args.h0_size, 
                                      args.transformer_layers, args.num_heads, 
                                      args.key_size, args.model_size, args.embed_size, 
                                      args.atom_types, args.wyck_types,
                                      args.dropout_rate)

transformer_name = 'Nf_%d_Kx_%d_Kl_%d_h0_%d_l_%d_H_%d_k_%d_m_%d_e_%d_drop_%g'%(args.Nf, args.Kx, args.Kl, args.h0_size, args.transformer_layers, args.num_heads, args.key_size, args.model_size, args.embed_size, args.dropout_rate)

print ("# of transformer params", ravel_pytree(params)[0].size) 

################### Train #############################

loss_fn, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer)

print("\n========== Prepare logs ==========")
if args.optimizer != "none" or args.restore_path is None:
    output_path = args.folder + args.optimizer+"_bs_%d_lr_%g_decay_%g_clip_%g" % (args.batchsize, args.lr, args.lr_decay, args.clip_grad) \
                   + '_A_%g_W_%g_N_%g'%(args.atom_types, args.wyck_types, args.n_max) \
                   + ("_wd_%g"%(args.weight_decay) if args.optimizer == "adamw" else "") \
                   + ('_a_%g_w_%g_l_%g'%(args.lamb_a, args.lamb_w, args.lamb_l)) \
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

    print("\n========== Load calculator and rl loss ==========")
    # calc = mace_mp(model="./data/2023-12-03-mace-128-L1_epoch-199.model",
    #                dispersion=False,
    #                default_dtype="float32",
    #                device='cuda')
    from ase.calculators.lj import LennardJones
    calc = LennardJones()
    reward_fn, batch_reward_fn = make_force_reward_fn(calc)
    rl_loss_fn = make_reinforce_loss(logp_fn, batch_reward_fn)

    print("\n========== Load partial sample function ==========")
    w_mask = None
    atom_mask = jnp.zeros((args.atom_types), dtype=int) # we will do nothing to a_logit in sampling
    atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
    constraints = jnp.arange(0, args.n_max, 1)

    partial_sample_crystal = partial(sample_crystal, transformer=transformer,
                                     n_max=args.n_max, atom_types=args.atom_types,
                                     wyck_types=args.wyck_types, Kx=args.Kx, Kl=args.Kl,
                                     w_mask=None, atom_mask=atom_mask,
                                     top_p=args.top_p, temperature=args.temperature,
                                     T1=args.temperature, constraints=constraints)

    print("\n========== Start RL training ==========")
    params, opt_state = train(key, optimizer, opt_state, rl_loss_fn, partial_sample_crystal,
                              params, epoch_finished, args.epochs, args.batchsize, output_path)

else:
    raise NotImplementedError("No optimizer specified. Please specify an optimizer in the config file.")