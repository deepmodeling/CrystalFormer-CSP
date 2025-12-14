# Model Card

## Alex-20s

The pre-trained model is available on [Google Drive](https://drive.google.com/file/d/13Dj8EXYOjAZoKrhMRQHRpXoU6N-E0Cto/view?usp=sharing) and [Hugging Face Model Hub](https://huggingface.co/zdcao/CrystalFormer-CSP). 

### Model Parameters

```python
params, transformer = make_transformer(
        key=jax.random.PRNGKey(42),
        Nf=5,
        Kx=16,
        Kl=4,
        n_max=21,
        h0_size=256,
        num_layers=16,
        num_heads=8,
        key_size=32,
        model_size=256,
        embed_size=256,
        atom_types=119,
        wyck_types=28,
        dropout_rate=0.1,
        attn_dropout=0.1,
        widening_factor=4,
        sigmamin=1e-3
)
```

### Training dataset

Alex-20s: contains ~1.7M general inorganic materials curated from the [Alexandria database](https://alexandria.icams.rub.de/), with $E_{hull} < 0.1$ eV/atom and no more than 20 Wyckoff sites in conventional cell. The dataset can be found in the [Hugging Face Datasets](https://huggingface.co/datasets/zdcao/alex-20s).


### Speeds, Sizes, Times
- _CrystalFormer-CSP_ contains ~13.8 M parameters
- It takes 1058 seconds to generate a batch size 29,000 crystal samples on a single A100 GPU, which translates to a generation speed of 37 milliseconds per sample.