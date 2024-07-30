import jax
import jax.numpy as jnp
import pandas as pd
import os
import optax

import crystalformer.src.checkpoint as checkpoint


def get_labels(csv_file, label_col):
    data = pd.read_csv(csv_file)
    labels = data[label_col].values
    labels = jnp.array(labels, dtype=float)
    return labels


if __name__  == "__main__":
    from jax.flatten_util import ravel_pytree

    from utils import GLXYZAW_from_file

    from _model import make_classifier
    from _transformer import make_transformer  
    from _train import train
    from _loss import make_classifier_loss


    key = jax.random.PRNGKey(42)
   
    train_path = "/data/zdcao/crystal_gpt/dataset/mp_20/train.csv"
    valid_path = "/data/zdcao/crystal_gpt/dataset/mp_20/val.csv"
    test_path = "/data/zdcao/crystal_gpt/dataset/mp_20/test.csv"
    atom_types = 119
    wyck_types = 28
    n_max = 21
    num_io_process = 40
    Nf = 5
    Kx = 16
    Kl = 4
    h0_size = 256
    transformer_layers = 4
    num_heads = 8
    key_size = 32
    model_size = 64
    embed_size = 32
    dropout_rate = 0.3

    sequence_length = 105
    outputs_size = 64
    hidden_sizes = [128, 128, 64]
    num_classes = 1
    restore_path = "/data/zdcao/crystal_gpt/classifier/"
    lr = 1e-4
    epochs = 1000
    batchsize = 256

    train_data = GLXYZAW_from_file(train_path, atom_types, wyck_types, n_max, num_io_process)
    valid_data = GLXYZAW_from_file(valid_path, atom_types, wyck_types, n_max, num_io_process)
    # test_data = GLXYZAW_from_file(test_path, atom_types, wyck_types, n_max, num_io_process)

    train_labels = get_labels(train_path, "band_gap")
    valid_labels = get_labels(valid_path, "band_gap")

    train_data = (*train_data, train_labels)
    valid_data = (*valid_data, valid_labels)

    ################### Model #############################
    transformer_params, state, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 
                                                              h0_size, 
                                                              transformer_layers, num_heads, 
                                                              key_size, model_size, embed_size, 
                                                              atom_types, wyck_types,
                                                              dropout_rate)
    print ("# of transformer params", ravel_pytree(transformer_params)[0].size) 
    
    key, subkey = jax.random.split(key)
    classifier_params, classifier = make_classifier(subkey,
                                                    n_max=n_max,
                                                    embed_size=embed_size,
                                                    sequence_length=sequence_length,
                                                    outputs_size=outputs_size,
                                                    hidden_sizes=hidden_sizes,
                                                    num_classes=num_classes)

    print ("# of classifier params", ravel_pytree(classifier_params)[0].size) 

    params = (transformer_params, classifier_params)

    print("\n========== Prepare logs ==========")
    output_path = os.path.dirname(restore_path)
    print("Will output samples to: %s" % output_path)

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        _params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    if len(_params) == len(params):
        params = _params
    else:
        params = (_params, params[1])  # only restore transformer params
        print("only restore transformer params")
    
    loss_fn, _ = make_classifier_loss(transformer, classifier)

    param_labels = ('transformer', 'classifier')
    optimizer = optax.multi_transform({'transformer': optax.adam(lr*0.1), 'classifier': optax.adam(lr)},
                                      param_labels)
    opt_state = optimizer.init(params)

    print("\n========== Start training ==========")
    key, subkey = jax.random.split(key)
    params, opt_state = train(subkey, optimizer, opt_state, loss_fn, params, state, epoch_finished, epochs, batchsize, train_data, valid_data, output_path)
