# crystal_gpt

let's code a crystal gpt together, with the help of GPT! 

# todo 

~~sketch
see colab [notebook](https://colab.research.google.com/drive/17iAaHocQ8KSnheKz3JgKRFKUaOgQfFk6?usp=sharing)~~

- [X] for data start with carbon_24, write data parse to get LXA
- [X] adapt trainning code at https://github.com/wangleiphy/ml4p/blob/main/projects/alanine_dipeptide.ipynb for the present case 
- [X] move code from notebook to script 
- [ ] write more tests 
- [ ] implement flow model for `L` based on the gaussian p(L|G)
- [X] train the model and get some samples 
- [ ] find out a way to evaluate the model, see whether this is indeed promising.
- [ ] write samples back to CIF file

enhancement
- [X] extend the code to multiple atom species
- [X] only treat inequavalent atoms
- [X] consider perov_5 dataset
- [X] consider space group as a pre-condition 
- [ ] experiment with training with condition y, and conditional generation. 
- [X] make a multiplicity table such as [1, 2, 3, 4, 8, 48, ...] to store possible multiplicities
- [ ] train for MP20 and evaluate the model again
- [ ] consider condition everying on the number of atoms 

production
- [ ] high pressure dataset 


## data 

as a first step, let's make use of the data in the cdvae repo
https://github.com/txie-93/cdvae/tree/main/data


## model 

We will build an autoregressive model P(G, L, X, AM) = P (X_1, AM_1| G) P ( X_2, AM_2 | G, X_1, AM_1 ) ... P(L| ...)
The autoregressive model will be a causal transformer (what else ?).

G: space group
L: lattice vector
X: factional coordiate
AM: atom type and multiplicity

Sec. A2 of [MatterGen paper](https://arxiv.org/abs/2312.03687) contains a discussion of the relevant symmetries between them. 

For X we consider to use a distributuion with periodic variables (e.g., wrapped Gaussuan, wrapped Cauchy, von Mises, ...). Here are some useful codes. In particular, we use mixture of von Mises distribution as the atom position is multi-modal. 

https://code.itp.ac.cn/wanglei/hydrogen/-/blob/van/src/sampler.py
https://code.itp.ac.cn/wanglei/hydrogen/-/blob/van/src/von_mises.py

For space group other than P1, we sample the Wyckoff positions. Note that there is a natural alphabetical order, starting with 'a' for a position with site-symmetry group of maximal order and ending with the highest letter for the general position. In this way, we actually sample the occupied atom type and fractional coordinate for each Wyckoff position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). To ensure that certain special coordiates are sampled with accurate precision, we will model the Wyckoff symbol (1a, 2b, ...)  along with the coordiate. Since the Wyckoff symbols are discrete objects, they can be used to gauge the numerical precesion issue when sampling (such as 0.5001). 

Since the number of atoms can vary, we will pad the atoms to the same length. The paded atoms have a special type 0 element. 
Note that we have designed an encoding scheme for atom type and multipliciy into an integer. In the transformer, that encoding is handelled as a one-hot vector. In this way, we avoid predicting factorized atom type and multiplicity P(A, M| ...) = P(A| ... ) * P(M | ...)


## optimization 

SGD 

## objective 

MLE 


# how to run

train
```bash 
python ../src/main.py --n_max 5 --atom_types 118 --mult_types 5 --folder /data/wanglei/crystalgpt/perov-mixture-noangle-am/ --K 8 --mlp_size 16 --h0_size 256 --transformer_layers 4 --num_heads 8 --key_size 16 --model_size 8 --lr 0.0001 --weight_decay 0.001 --batchsize 100 --epochs 100000 --optimizer adamw 
```

sample
```bash 
python ../src/main.py --n_max 5 --atom_types 118 --mult_types 5 --folder /data/wanglei/crystalgpt/perov-mixture-noangle-am/ --K 8 --mlp_size 16 --h0_size 256 --transformer_layers 4 --num_heads 8 --key_size 16 --model_size 8 --lr 0.0001 --weight_decay 0.001 --batchsize 100 --epochs 100000 --optimizer none --restore_path /data/wanglei/crystalgpt/perov-mixture-noangle-am/adamw_bs_100_lr_0.0001_a_118_m_5_wd_0.001_mlp_16_K_8_h0_256_l_4_H_8_k_16_m_8/
```
