# crystal_gpt

let's code a crystal gpt together, with the help of GPT! 

# todo 

~~sketch
see colab [notebook](https://colab.research.google.com/drive/17iAaHocQ8KSnheKz3JgKRFKUaOgQfFk6?usp=sharing)~~

- [X] for data start with carbon_24, write data parse to get LXA
- [X] adapt trainning code at https://github.com/wangleiphy/ml4p/blob/main/projects/alanine_dipeptide.ipynb for the present case 
- [X] move code from notebook to script 
- [ ] implement flow model for `L` based on the gaussian p(L|G)
- [X] train the model and get some samples 
- [ ] evaluate the model, may follow https://github.com/txie-93/cdvae or https://github.com/jiaor17/DiffCSP 
- [X] write samples back to CIF file

enhancement
- [X] extend the code to multiple atom species
- [X] only treat inequavalent atoms
- [X] consider perov_5 dataset
- [X] consider space group as a pre-condition 
- [ ] experiment with training with condition y, and conditional generation. 
~~[X] make a multiplicity table such as [1, 2, 3, 4, 8, 48, ...] to store possible multiplicities~~
- [X] fix the primitive versus conventional cell issue when loading the mp_20 dataset
- [ ] train for MP20 and evaluate the model again
- [ ] consider condition everying on the number of atoms 
- [X] specify possible elements at sampling time
- [X] implement more wyckoff symbols in `wyckoff.py`
- [X] implement more symmetrize function in `symmetrize.py`, or consider call an external library when sampling
- [X] build up (230, 28, 3) fc mask
- [ ] `pmap` the code for multi-gpu training
- [X] use pyxtal to load csv file (avoid build our own dataset)
- [X] impose the order constraint W_0 <= W_1 <= W_2  ... <= W_n

always welcome
- [ ] write more tests 
- [ ] polishing the code

side project
- [ ] symmetry constrainted flow-matching [this](https://github.com/wangleiphy/jax-flow-matching/) and spacegroup utils maybe useful

production
- [ ] high pressure dataset 


## data 

as a first step, let's make use of the data in the cdvae repo
https://github.com/txie-93/cdvae/tree/main/data


## model 

We will build an autoregressive model P(G, L, X, AW) = P (X_1, AW_1| G) P ( X_2, AW_2 | G, X_1, AW_1 ) ... P(L| ...)
The autoregressive model will be a causal transformer (what else ?).

`G`: space group 1-230

`L`: lattice vector [a,b,c,alpha,beta,gamma]

`X`: factional coordiate

`A`: atom type (0 stands for empty, 1=H, ..., 118=Og)

`W`: wyckoff symbol (0=empty, 1=a, 2=b, 3=c, ..., 27=A)  

there is an associated data `M` stands for multiplicity, which can be read out by looking at the table `M=mult_table[G-1, W]`. 

Sec. A2 of [MatterGen paper](https://arxiv.org/abs/2312.03687) contains a discussion of the relevant symmetries between them. 

For X we consider to use a distributuion with periodic variables (e.g., wrapped Gaussuan, wrapped Cauchy, von Mises, ...). Here are some useful codes. In particular, we use mixture of von Mises distribution as the atom position is multi-modal. 

https://code.itp.ac.cn/wanglei/hydrogen/-/blob/van/src/sampler.py
https://code.itp.ac.cn/wanglei/hydrogen/-/blob/van/src/von_mises.py

For space group other than P1, we sample the Wyckoff positions. Note that there is a natural alphabetical order, starting with 'a' for a position with site-symmetry group of maximal order and ending with the highest letter for the general position. In this way, we actually sample the occupied atom type and fractional coordinate for each Wyckoff position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). To ensure that certain special coordiates are sampled with accurate precision, we will model the Wyckoff symbol (1a, 2b, ...)  along with the coordiate. Since the Wyckoff symbols are discrete objects, they can be used to gauge the numerical precesion issue when sampling (such as 0.5001). 

In practice, the space group label `G` plays three effects to the code: 1) it acts as the one-hot condition in the transformer, so everything sampled (`X`, `AW`, `L`) depends on it; 2) it determines the lattice_mask such as [111000] that will be placed on the lattice regression loss, so we only score those free params that was not fixed by the space group. In sampling, we do similar thing to impose lattice according to the spacegroup with `symmetrize_lattice`  function.  3) G and W together constraints on the factional coordinate that is currenly used in training (via fc_mask) and  sampling (via apply_wyckoff_condition)

Since the number of atoms may vary, we will pad the atoms to the same length up to `n_max`. The paded atoms have type 0. 
Note that we have designed an encoding scheme for atom type and wyckoff symbol into an integer. In the transformer, that encoding is handelled as a one-hot vector. In this way, we avoid predicting factorized atom type and multiplicity P(A, W| ...) = P(A| ... ) * P(W | ...)

Note that we have to the lattice `L` to the very end of the sampling. That was due to the consideration that generating lattice out of vacumm is much harder than if we already have the lattice. 


## optimization 

SGD 

## objective 

MLE 


# how to run

train
```bash 
python ../src/main.py --n_max 20 --atom_types 119 --wyck_types 28 --folder /data/wanglei/crystalgpt/mp-mp-wyckoff-debug-sortx-sortw-fc_mask-dropout-permloss-mult-aw_max-aw_params-pyxtal/mp-8b827/ --Kx 16 --Kl 16 --h0_size 256 --transformer_layers 4 --num_heads 8 --key_size 32 --model_size 512 --lr 0.0001 --lr_decay 0.0 --weight_decay 0.0 --clip_grad 1.0 --batchsize 100 --epochs 50000 --optimizer adam --train_path /home/wanglei/cdvae/data/mp_20/train.csv --valid_path /home/wanglei/cdvae/data/mp_20/val.csv --test_path /home/wanglei/cdvae/data/mp_20/test.csv --dropout_rate 0.5 --num_io_process 40 
```

sample
```bash 
python ../src/main.py --n_max 20 --atom_types 119 --wyck_types 28 --folder /data/wanglei/crystalgpt/mp-mp-wyckoff-debug-sortx-sortw-fc_mask-dropout-permloss-mult-aw_max-aw_params-pyxtal/mp-8b827/ --Kx 16 --Kl 16 --h0_size 256 --transformer_layers 4 --num_heads 8 --key_size 32 --model_size 512 --lr 0.0001 --lr_decay 0.0 --weight_decay 0.0 --clip_grad 1.0 --batchsize 100 --epochs 50000 --optimizer none --train_path /home/wanglei/cdvae/data/mp_20/train.csv --valid_path /home/wanglei/cdvae/data/mp_20/val.csv --test_path /home/wanglei/cdvae/data/mp_20/test.csv --dropout_rate 0.5 --num_io_process 40  --restore_path /data/wanglei/crystalgpt/mp-mp-wyckoff-debug-sortx-sortw-fc_mask-dropout-permloss-mult-aw_max-aw_params-pyxtal/mp-8b827/adam_bs_100_lr_0.0001_decay_0_clip_1_A_119_W_28_N_20_Nf_5_K_16_16_h0_256_l_4_H_8_k_32_m_512_drop_0.5/ --spacegroup 123 --num_samples 123 
```

evaluate
```bash
python ../scripts/compute_metrics.py --root_path /data/zdcao/crystal_gpt/dataset/mp_20/symm_data/ --filename out_structure.csv --output_path ./ --num_io_process 40
```
