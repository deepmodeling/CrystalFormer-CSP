<div align="center">
<img align="middle" src="data/crystalformer.png" width="400" alt="logo"/>
</div>


## model card

The model is an autoregressive transformer for the space group conditioned crystal probability distributuion `P(C|g) = P (W_1 | ... ) P ( A_1 | ... ) P(X_1| ...) P(W_2|...) ... P(L| ...)`, where

`g`: space group nubmer 1-230

`W`: wyckoff letter (0=empty, 1=a, 2=b, 3=c, ..., 27=A)  

`A`: atom type (0 stands for empty, 1=H, ..., 118=Og)

`X`: factional coordiate

`L`: lattice vector [a,b,c,alpha,beta,gamma]

 `P(W_i| ...)`are `P(A_i| ...)`  are categorical distributuions. 

`P(X_i| ...)`  is mixture of von Mises distributuion. 

`P(L| ...)`  is mixture of Gaussian distributuion. 

Note that there is a natural alphabetical ordering for the Wyckoff letters, starting with 'a' for a position with site-symmetry group of maximal order and ending with the highest letter for the general position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). Since the Wyckoff symbols are discrete objects, they can be used to gauge the numerical precesion issue when sampling (such as 0.5001). 


## how to run

### train

```bash 
python ../src/main.py  --n_max 21 --atom_types 119 --wyck_types 28 --folder /data/wanglei/crystalgpt/mp-mpsort_aw/fix-symops-pyxtal-5d111/ --Kx 16 --Kl 16 --h0_size 256 --transformer_layers 4 --num_heads 8 --key_size 32 --model_size 64 --lr 0.0001 --lr_decay 0.0 --weight_decay 0.0 --clip_grad 1.0 --batchsize 100 --epochs 1000 --optimizer adam --train_path /home/wanglei/cdvae/data/mp_20/train.csv --valid_path /home/wanglei/cdvae/data/mp_20/val.csv --test_path /home/wanglei/cdvae/data/mp_20/test.csv --dropout_rate 0.1 --num_io_process 40 --Nf 5 --perm_aug --map_aug 
```

### sample

```bash 
python ../src/main.py --n_max 21 --atom_types 119 --wyck_types 28 --Nf 5 --folder /data/wanglei/crystalgpt/mp-mp-wyckoff-debug-sortx-sortw-fc_mask-dropout-permloss-mult-aw_max-aw_params-pyxtal/mp-8b827/ --Kx 16 --Kl 16 --h0_size 256 --transformer_layers 4 --num_heads 8 --key_size 32 --model_size 64 --lr 0.0001 --lr_decay 0.0 --weight_decay 0.0 --clip_grad 1.0 --batchsize 100 --epochs 50000 --optimizer none --train_path /home/wanglei/cdvae/data/mp_20/train.csv --valid_path /home/wanglei/cdvae/data/mp_20/val.csv --test_path /home/wanglei/cdvae/data/mp_20/test.csv --dropout_rate 0.1 --num_io_process 40 --restore_path /data/wanglei/crystalgpt/mp-mpsort_aw/fix-symops-pyxtal-5d111/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_21_aw_1_l_1_perm_map_Nf_5_K_16_16_h0_256_l_4_H_8_k_32_m_64_drop_0.1/  --spacegroup 167 --num_samples 1000 --map_aug
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