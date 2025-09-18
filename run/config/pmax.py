import subprocess 
import numpy as np 
import time 

nickname = 'csp'

###############################
atom_types = 119

Nf = 5
Kx, Kl = 16, 4
h0_size = 256
transformer_layers = 4
num_heads = 8
key_size = 32
model_size = 64
embed_size = 32
dropout_rate = 0.3

optimizer = 'adam'
weight_decay = 0.0 
lr = 1e-4
lr_decay = 0.0
clip_grad = 1.0
batchsize = 100
epochs = 10000

lamb_a, lamb_w, lamb_l = 1.0, 1.0, 1.0

num_io_process = 20

n_max = 21
wyck_types = 28

mp20_folder = '/home/user_wanglei/private/homefile/cdvae/data/mp_20/'
train_path = mp20_folder+'/train.csv'
valid_path = mp20_folder+'/val.csv'
test_path = mp20_folder+'/test.csv'

###############################
num_io_process = 20

reward='ehull'
mlff_model='orb'
beta = 0.0

#restore_path='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20'
restore_path=None
convex_path='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20/convex_hull_pbe_2023.12.29.json.bz2'
mlff_path='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20/orb-v2-20241011.ckpt'

###############################
prog = 'train_ppo'
resfolder = '/home/user_wanglei/private/datafile/crystalgpt/' + nickname  + '/' 

def submitJob(bin,args,jobname,logname,run=False,wait=None):

    #prepare the job file 
    job='''#!/bin/bash -l
#SBATCH --partition=home
#SBATCH --nodes=1
#SBATCH --cpus-per-task=%g
#SBATCH --mem=64G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=24:00:00
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --error=%s'''%(num_io_process,jobname,logname,logname)

    if wait is not None:
        dependency ='''
#SBATCH --dependency=afterany:%d\n'''%(wait)
        job += dependency 

    job += '''
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "The current job ID is $SLURM_JOB_ID"
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo "Running on $SLURM_JOB_NUM_NODES nodes:"
echo $SLURM_JOB_NODELIST
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"\n

echo Job started at `date`\n'''
    job += str(bin) + ' '
    for key, val in args.items():
        if isinstance(val, bool) or val is None:
            job += (" --%s" % key if val else "")
        else:
            job += " --%s %s" % (key, val)
    job += '''
echo Job finished at `date`\n'''

    #print job
    jobfile = open("jobfile", "w")
    jobfile.write("%s"%job)
    jobfile.close()

    #submit the job 
    if run:
        cmd = ['sbatch', 'jobfile']
        time.sleep(0.1)
    else:
        cmd = ['cat','jobfile']

    subprocess.check_call(cmd)
    return None

