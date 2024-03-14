<div align="center">
<img align="middle" src="data/crystalformer.png" width="400" alt="logo"/>
</div>

# CrystalFormer: Space Group-Controlled Crystal Generation with Transformer

<p align="center">
  <img src="data/output.gif" width="400">
</p>

## Contents

- [CrystalFormer: Space Group-Controlled Crystal Generation with Transformer](#crystalformer-space-group-controlled-crystal-generation-with-transformer)
  - [Contents](#contents)
  - [Model card](#model-card)
  - [Installation](#installation)
    - [install required packages](#install-required-packages)
    - [CUDA (GPU) installation](#cuda-gpu-installation)
  - [Available Weights](#available-weights)
  - [How to run](#how-to-run)
    - [train](#train)
    - [sample](#sample)
    - [evaluate](#evaluate)
  - [How to cite](#how-to-cite)

## Model card

The model is an autoregressive transformer for the space group conditioned crystal probability distribution `P(C|g) = P (W_1 | ... ) P ( A_1 | ... ) P(X_1| ...) P(W_2|...) ... P(L| ...)`, where

`g`: space group number 1-230

`W`: Wyckoff letter ('a', 'b',...,'A')  

`A`: atom type ('H', 'He', ..., 'Og')

`X`: factional coordinates

`L`: lattice vector [a,b,c, $\alpha$, $\beta$, $\gamma$]

`P(W_i| ...)` and `P(A_i| ...)`  are categorical distributuions. 

`P(X_i| ...)` is the mixture of von Mises distribution. 

`P(L| ...)`  is the mixture of Gaussian distribution. 

We only consider symmetry inequivalent atoms. The remaining atoms are restored based on the space group and Wyckoff letter information. Note that there is a natural alphabetical ordering for the Wyckoff letters, starting with 'a' for a position with the site-symmetry group of maximal order and ending with the highest letter for the general position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). Only for the cases where discrete Wyckoff letters can not fully determine the structure, one needs to further consider factional coordinates in the loss or sampling. 

## Installation

### install required packages

```bash
pip install -r requirements.txt
```

### CUDA (GPU) installation

If you intend to use CUDA (GPU) to speed up the training, it is important to install the appropriate version of `JAX` and `jaxlib`. It is recommended to check the [JAX docs](https://github.com/google/jax?tab=readme-ov-file#installation) for the installation guide. The basic installation command is given below:

```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Available Weights

We release the weights of the model trained on the MP-20 dataset. More details can be seen in the [model](./model/README.md) folder.



## How to run

### train

```bash 
python ../src/main.py  --n_max 21 --atom_types 119 --wyck_types 28 --folder /data/wanglei/crystalgpt/mp-mpsort-xyz-embed/w-a-x-y-z-periodic-fixed-size-embed-eb630/ --Nf 5 --Kx 16 --Kl 4 --h0_size 256 --transformer_layers 8 --num_heads 8 --key_size 32 --model_size 64 --embed_size 32 --lr 0.0001 --lr_decay 0.0 --weight_decay 0.0 --clip_grad 1.0 --batchsize 100 --epochs 10000 --optimizer adam --train_path /home/wanglei/cdvae/data/mp_20/train.csv --valid_path /home/wanglei/cdvae/data/mp_20/val.csv --test_path /home/wanglei/cdvae/data/mp_20/test.csv --dropout_rate 0.3 --num_io_process 40 --lamb_a 1.0 --lamb_w 1.0 --lamb_l 1.0
```

### sample

```bash 
python ../src/main.py --Nf 5 --n_max 21 --atom_types 119 --wyck_types 28 --folder /data/wanglei/crystalgpt/mp-mp-wyckoff-debug-sortx-sortw-fc_mask-dropout-permloss-mult-aw_max-aw_params-pyxtal/mp-8b827/ --Kx 16 --Kl 4 --h0_size 256 --transformer_layers 4 --num_heads 8 --key_size 32 --model_size 64 --embed_size 32 --lr 0.0001 --lr_decay 0.0 --weight_decay 0.0 --clip_grad 1.0 --batchsize 10000 --epochs 50000 --optimizer none --train_path /home/wanglei/cdvae/data/mp_20/train.csv --valid_path /home/wanglei/cdvae/data/mp_20/val.csv --test_path /home/wanglei/cdvae/data/mp_20/test.csv --dropout_rate 0.3 --num_io_process 40 --restore_path /data/wanglei/crystalgpt/mp-mpsort-xyz/w-a-x-y-z-periodic-7ea88/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_a_1_w_1_l_1_Nf_5_Kx_16_Kl_4_h0_256_l_4_H_8_k_32_m_64_drop_0.3/ --spacegroup 160 --num_samples 100
```

### evaluate

```bash
python ../scripts/compute_metrics.py --root_path /data/zdcao/crystal_gpt/dataset/mp_20/symm_data/ --filename out_structure.csv --output_path ./ --num_io_process 40
```

## How to cite


```bibtex
@article{crystalformer2024,
  title = {Crystalformer},
  author = {Zhendong Cao and Lei Wang},
}
```
