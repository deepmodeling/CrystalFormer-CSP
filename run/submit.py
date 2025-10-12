#!/usr/bin/env python
import sys , os 

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action='store_true', help="Run or not")
    parser.add_argument('--mode', type=str, default=None, choices=['pretrain', 'finetune'], help='')
    parser.add_argument("--waitfor", type=int, help="wait for this job for finish")
    input = parser.parse_args()


    #this import might overwrite the above default parameters 
    #########################################################
    import socket, getpass
    machinename = socket.gethostname()
    username = getpass.getuser()
    print ('\n', username, 'on', machinename, '\n')
    if 'ip-10-0-0-26' in machinename:
        from config.aws import * 
    elif 'ln01' in machinename:
        from config.ln01 import * 
    elif 'bright90' in machinename:
        from config.bright90 import * 
    elif 'slurm-client' in machinename:
        from config.pmax import *
    else:
        print ('where am I ?', machinename)
        sys.exit(1)
    #########################################################

    from pygit2 import Repository
    head = Repository('.').head
    branch = head.shorthand + "-" + head.target.raw.hex()[:5]

    resfolder = os.path.join(resfolder, branch) + '/'

    jobdir='../jobs/' + nickname + '/'
    jobdir = os.path.join(jobdir, branch) + '/'

    cmd = ['mkdir', '-p', jobdir]
    subprocess.check_call(cmd)

    cmd = ['mkdir', '-p', resfolder]
    subprocess.check_call(cmd)

    # Base arguments common to all modes
    args = {'n_max':n_max,
                    'atom_types': atom_types,
                    'wyck_types': wyck_types,
                    'folder':resfolder,
                    'Nf':Nf,
                    'Kx':Kx,
                    'Kl':Kl,
                    'h0_size': h0_size,
                    'transformer_layers':transformer_layers,
                    'num_heads':num_heads,
                    'key_size':key_size,
                    'model_size':model_size,
                    'embed_size':embed_size
                    }
    
    if input.mode == 'finetune':
        print ("rl finetune  mode")

        prog = 'train_ppo'

        # Extend args with finetune-specific parameters
        args.update({
                        'epochs':finetune_epochs, 
                        'batchsize':finetune_batchsize, 
                        'lr':finetune_lr, 
                        'reward': reward, 
                        'formula': formula, 
                        'mlff_model': mlff_model, 
                        'restore_path': restore_path, 
                        'convex_path': convex_path, 
                        'mlff_path': mlff_path, 
                        'beta':beta, 
                        'K':K,
                        'dropout_rate' : finetune_dropout_rate
                        })

        if spacegroup is not None:
            args.update({'spacegroup': spacegroup})

    elif input.mode == 'pretrain':
        print ("pretrain mode")
                
        prog = 'python main.py'

        # Extend args with pretrain-specific parameters
        args.update({   'lr':pretrain_lr,
                        'lr_decay': lr_decay,
                        'epochs': pretrain_epochs, 
                        'weight_decay': weight_decay,
                        'clip_grad': clip_grad,
                        'batchsize': pretrain_batchsize,
                        'optimizer': optimizer,
                        'train_path' : train_path,
                        'valid_path' : valid_path,
                        'test_path' : test_path,
                        'dropout_rate' : pretrain_dropout_rate,
                        'num_io_process' : num_io_process,
                        'lamb_a': lamb_a,
                        'lamb_w': lamb_w,
                        'lamb_l': lamb_l,
                        })

        if restore_path is not None:
            args.update({'restore_path': restore_path})

    else:
        raise ValueError(f"Invalid mode '{input.mode}'. Must be one of: {['pretrain', 'finetune']}")

    logname = jobdir 
    for arg, value in args.items():
        if isinstance(value, bool):
            logname += ("%s_" % arg if value else "")
        elif not ('_path' in arg or 'folder' in arg):
            if '_' in arg:
                arg = "".join([s[0] for s in arg.split('_')])
            elif arg == 'elements':
                arg = ''
                value = elements_str 
            logname += "%s%s_" % (arg, value)
    logname = logname[:-1] + '.log'

    jobname = os.path.basename(os.path.dirname(logname))

    jobid = submitJob(prog,args,jobname,logname,run=input.run, wait=input.waitfor)
