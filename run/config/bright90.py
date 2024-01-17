import subprocess 
import numpy as np 
import time 

#nickname = 'perov-mixture-spacegroup'
nickname = 'mp'

###############################
atom_types = 118

K = 8
mlp_size = 16
h0_size = 256
transformer_layers = 4
num_heads = 8
key_size = 16
model_size = 8

dataset = 'mp'

if dataset == 'perov':
    n_max = 5 
    train_path = '/home/wanglei/cdvae/data/perov/train.csv'
    valid_path = '/home/wanglei/cdvae/data/perov/val.csv'
    test_path = '/home/wanglei/cdvae/data/perov/test.csv'

elif dataset == 'mp':
    n_max = 20
    train_path = '/home/wanglei/cdvae/data/mp_20/train.csv'
    valid_path = '/home/wanglei/cdvae/data/mp_20/val.csv'
    test_path = '/home/wanglei/cdvae/data/mp_20/test.csv'
else:
    print (dataset)

lr = 1e-4
weight_decay = 1e-3 
batchsize = 100
epochs = 100000

optimizer = 'adamw'

###############################
prog = '../src/main.py'
resfolder = '/data/wanglei/crystalgpt/' + nickname  + '/' 

def submitJob(bin,args,jobname,logname,run=False,wait=None):

    #prepare the job file 
    job='''#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --error=%s'''%(jobname,logname,logname)

    if wait is not None:
        dependency ='''
#SBATCH --dependency=afterany:%d\n'''%(wait)
        job += dependency 

    job += '''
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes:"
echo $SLURM_JOB_NODELIST
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"\n
echo Job started at `date`\n'''

    job +='python '+ str(bin) + ' '
    for key, val in args.items():
        job += '--'+str(key) + ' '+ str(val) + ' '
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

