import subprocess 
import numpy as np 
import time 

dataset = 'alex20s'
nickname = 'csp'
resfolder = '/home/user_wanglei/private/datafile/crystalgpt/' + nickname  + '/' + dataset + '/'

###############################
n_max = 21
wyck_types = 28
atom_types = 119

Nf = 5
Kx, Kl = 16, 4
h0_size = 256
transformer_layers = 16
num_heads = 8
key_size = 16
model_size = 128
embed_size = 128

pretrain_dropout_rate = 0.1
finetune_dropout_rate = 0.0

optimizer = 'adam'
weight_decay = 0.0 
pretrain_lr = 1e-4
finetune_lr = 1e-5
lr_decay = 0.0
clip_grad = 1.0

pretrain_batchsize = 8000
finetune_batchsize = 1000

pretrain_epochs = 10000
finetune_epochs = 5000 

lamb_a, lamb_w, lamb_l = 1.0, 1.0, 1.0

num_io_process = 20

mp20_folder = '/home/user_wanglei/private/homefile/cdvae/data/mp_20/'
alex20_folder = '/opt/data/bcmdata/ZONES/data/PROJECTS/datafile/PRIVATE/zdcao/crystal_gpt/dataset/alex/PBE/alex20/'
alex20s_folder = '/opt/data/bcmdata/ZONES/data/PROJECTS/datafile/PRIVATE/zdcao/crystal_gpt/dataset/alex/PBE_20241204/'

if dataset == 'mp20':
    train_path = mp20_folder+'/train.csv'
    valid_path = mp20_folder+'/val.csv'
    test_path = mp20_folder+'/test.csv'
elif dataset == 'alex20':
    train_path = alex20_folder+'/train.lmdb'
    valid_path = alex20_folder+'/val.lmdb'
    test_path = alex20_folder+'/test.lmdb'
elif dataset == 'alex20s':
    train_path = alex20s_folder+'/train.lmdb'
    valid_path = alex20s_folder+'/val.lmdb'
    test_path = alex20s_folder+'/test.lmdb'
else:
    raise ValueError(f"Invalid mode '{dataset}'. Must be one of: {['mp20', 'alex20', 'alex20s']}")

###############################

reward='ehull'
beta = 0.1
formula = 'Ti13Al9Co8'
spacegroup = 160
K = 0

#restore_path='/home/user_wanglei/private/datafile/crystalgpt/csp/alex20/csp-6000f/adam_bs_8000_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_16_H_16_k_64_m_64_e_32_drop_0.1_0.1/'
#restore_path='/home/user_wanglei/private/datafile/crystalgpt/csp/alex20s/csp-07d3f/adam_bs_8000_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_16_H_16_k_64_m_64_e_32_drop_0.1_0.1/'
#restore_path='/home/user_wanglei/private/datafile/crystalgpt/csp/alex20s/csp-a20de/adam_bs_8000_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_16_H_16_k_64_m_64_e_32_drop_0.1_0.1/'
restore_path='/home/user_wanglei/private/datafile/crystalgpt/csp/alex20s/csp-0d128/adam_bs_8000_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_16_H_8_k_16_m_128_e_128_drop_0.1_0.1/'
#restore_path = None

#convex_path='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20/convex_hull_pbe_2023.12.29.json.bz2'
convex_path='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20/convex_hull_pbe.json.bz2'

mlff_model='orb-v2'
mlff_path='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20/orb-v2-20241011.ckpt'

#mlff_model='orb-v3-conservative-inf-mpa'
#mlff_path='/home/user_wanglei/private/datafile/crystalgpt/checkpoint/alex20/orb-v3-conservative-inf-mpa-20250404.ckpt'

###############################

def submitJob(bin,args,jobname,logname,run=False,wait=None):

    #prepare the job file 
    job='''#!/bin/bash -l
#SBATCH --partition=home
#SBATCH --nodes=1
#SBATCH --cpus-per-task=%g
#SBATCH --mem=32G
#SBATCH --gres=gpu:A800:1
#SBATCH --time=168:00:00
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --error=%s'''%(num_io_process,jobname,logname,logname)

    if wait is not None:
        dependency ='''
#SBATCH --dependency=afterany:%d\n'''%(wait)
        job += dependency 

    job += '''
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/user_wanglei/private/homefile/crystal_gpt/
echo Current working directory is `pwd`
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

