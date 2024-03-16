<div align="center">
	<img align="middle" src="imgs/crystalformer.png" width="400" alt="logo"/>
  <h2> Crystal Generation with Space Group Informed Transformer</h2> 
</div>

<div align="center">
  <img align="middle" src="imgs/output.gif" width="400">
  <h3> Generating Cs<sub>2</sub>ZnFe(CN)<sub>6</sub> Crystal (<a href=https://next-gen.materialsproject.org/materials/mp-570545>mp-570545</a>) </h3>
</div>



## Contents

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
python ../src/main.py --folder ./ --train_path /home/wanglei/cdvae/data/mp_20/train.csv --valid_path /home/wanglei/cdvae/data/mp_20/val.csv --test_path /home/wanglei/cdvae/data/mp_20/test.csv      
```
`folder`: the folder to save the model and logs  
`train_path`: the path to the training dataset  
`valid_path`: the path to the validation dataset  
`test_path`: the path to the test dataset

### sample

```bash 
python ../src/main.py --optimizer none --train_path /home/wanglei/cdvae/data/mp_20/train.csv --valid_path /home/wanglei/cdvae/data/mp_20/val.csv --test_path /home/wanglei/cdvae/data/mp_20/test.csv --restore_path YOUR_MODEL_PATH --spacegroup 160 --num_samples 100  --batchsize 10000 --temperature 1.0 --use_foriloop
```

`optimizer`: the optimizer to use, `none` means no training, only sampling  
`restore_path`: the path to the model weights  
`spacegroup`: the space group number to sample  
`num_samples`: the number of samples to generate  
`batchsize`: the batch size for sampling  
`temperature`: the temperature for sampling  
`use_foriloop`: use `lax.fori_loop` to speed up the sampling

### evaluate

Transform the generated `G, L, W, A, X` to the `cif` format:
```bash
python ../scripts/awl2struct.py --output_path YOUR_PATH  --num_io_process 40
```

Calculate the structure and composition validity of the generated structures:
```bash
python ../scripts/compute_metrics.py --root_path /data/zdcao/crystal_gpt/dataset/mp_20/symm_data/ --filename out_structure.csv --output_path ./ --num_io_process 40
```
More details about the post-processing can be seen in the [scripts](./scripts/README.md) folder.

## How to cite

```bibtex
@article{crystalformer2024,
  title = {Crystalformer},
  author = {Zhendong Cao and Lei Wang},
}
```
