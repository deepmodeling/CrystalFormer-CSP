# crystal_gpt

let's code a crystal gpt together, with the help of GPT! 

# todo 

~~sketch
see colab [notebook](https://colab.research.google.com/drive/17iAaHocQ8KSnheKz3JgKRFKUaOgQfFk6?usp=sharing)~~

- [X] for data start with carbon_24, write data parse to get LXA
- [X] adapt trainning code at https://github.com/wangleiphy/ml4p/blob/main/projects/alanine_dipeptide.ipynb for the present case 
- [X] move code from notebook to script 
- [ ] write tests 
- [X] implement flow model for `L`
- [ ] train the model and get some samples 
- [ ] find out a way to evaluate the model, see whether this is indeed promising.
- [ ] write samples back to CIF file

enhancement
- [ ] extend the code to multiple atom species
- [ ] only treat inequavalent atoms
- [ ] consider perov_5 dataset
- [ ] consider space group as a pre-condition 
- [ ] experiment with training with condition y, and conditional generation. 
- [ ] train for MP20 and evaluate the model again
- [ ] consider condition everying on the number of atoms 

production
- [ ] high pressure dataset 


## data 

as a first step, let's make use of the data in the cdvae repo
https://github.com/txie-93/cdvae/tree/main/data


## model 

The basic objects to model are `L`, `A`, and `X`, 
where `L` stands for the lattice vector, `A` atom type, and `X` the fractional coordinate. 

Sec. A2 of [MatterGen paper](https://arxiv.org/abs/2312.03687) contains a discussion of the relevant symmetries between them. 

We will build an autoregressive model P(L, A, X) = P(L) P (A_1, X_1 | L) P ( A_2, X_2 | L , A_1, X_1 ) ... 
The autoregressive model will be a causal transformer (what else ?).
 Note that X is the factional coordiate in a crystal. For that we consider to use a distributuion with periodic variables (e.g., wrapped Gaussuan, wrapped Cauchy, von Mises, ...). Here are some useful codes. 

https://code.itp.ac.cn/wanglei/hydrogen/-/blob/van/src/sampler.py
https://code.itp.ac.cn/wanglei/hydrogen/-/blob/van/src/von_mises.py

For P(L) we will use a 6-dimensional flow model. 

For space group other than P1, we will sample the Wyckoff positions. Note that there is a natural alphabetical order, starting with 'a' for a position with site-symmetry group of maximal order and ending with the highest letter for the general position. In this way, we actually sample the occupied atom type and fractional coordinate for each Wyckoff position. The sampling procedure starts from higher symmetry sites (with smaller multiplicities) and then goes on to lower symmetry ones (with larger multiplicities). To ensure that certain special coordiates are sampled with accurate precision, we will model the Wyckoff symbol (1a, 2b, ...)  along with the coordiate. Since the Wyckoff symbols are discrete objects, they can be used to gauge the numerical precesion issue when sampling (such as 0.5001). 

Since the number of atoms can vary, we will pad the atoms to the same length. The paded atoms have a special type `X`.


## optimization 

SGD 

## objective 

MLE 


# how to run

train
```bash 
python ../src/main.py --n_max 24 --atom_types 2 --folder /data/wanglei/crystalgpt/perov/ --transformer_layers 4 --num_heads 8 --key_size 16 --model_size 32 --lr 0.0001 --weight_decay 0.001 --batchsize 100 --epochs 100000 --optimizer adamw 
```

sample
```bash 
python main.py --optimizer none --restore_path /data/wanglei/crystalgpt/perov/adamw_bs_100_lr_0.0001_wd_0.001_l_4_h_8_k_16_m_32/  --batchsize 10  
```
