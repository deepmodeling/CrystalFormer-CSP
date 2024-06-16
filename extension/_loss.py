import jax
import jax.numpy as jnp
from functools import partial

# from config import *   
from wyckoff import mult_table


def make_classifier_loss(transformer, classifier):
    
    def forward_fn(params, state, key, G, L, XYZ, A, W, is_train):
        M = mult_table[G-1, W]  # (n_max,) multplicities
        transformer_params, classifier_params = params
        _, state = transformer(transformer_params, state, key, G, XYZ, A, W, M, is_train)

        h = state['~']['last_hidden_state']
        g = state['~']['_g_embeddings']

        key, subkey = jax.random.split(key)
        y = classifier(classifier_params, subkey, g, L, W, h, is_train)
        return y

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, None))
    def mae_loss(params, state, key, G, L, XYZ, A, W, labels, is_training):
        y = forward_fn(params, state, key, G, L, XYZ, A, W, is_training)
        return jnp.abs(y - labels)
    
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, None))
    def mse_loss(params, state, key, G, L, XYZ, A, W, labels, is_training):
        y = forward_fn(params, state, key, G, L, XYZ, A, W, is_training)
        return jnp.square(y - labels)
    
    def loss_fn(params, state, key, G, L, XYZ, A, W, labels, is_training):
        loss = jnp.mean(mae_loss(params, state, key, G, L, XYZ, A, W, labels, is_training))
        return loss
    
    return loss_fn, forward_fn


if __name__ == "__main__":
    from utils import GLXYZAW_from_file

    from _model import make_classifier
    from _transformer import make_transformer

    atom_types = 119
    n_max = 21
    wyck_types = 28
    Nf = 5
    Kx = 16
    Kl  = 4
    dropout_rate = 0.1 

    csv_file = '../data/mini.csv'
    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)

    key = jax.random.PRNGKey(42)

    transformer_params, state, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 128, 4, 4, 8, 16, 16, atom_types, wyck_types, dropout_rate) 
    classifier_params, classifier = make_classifier(key,
                                                    n_max=n_max,
                                                    embed_size=16,
                                                    sequence_length=105,
                                                    outputs_size=16,
                                                    hidden_sizes=[16, 16],
                                                    num_classes=1)

    params = (transformer_params, classifier_params)
    loss_fn, _ = make_classifier_loss(transformer, classifier)

    # M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) # (batchsize, n_max)
    # value, state = jax.vmap(transformer,
    #                         in_axes=(None, None, None, 0, 0, 0, 0, 0, None)
    #                         )(transformer_params, state, key,
    #                           G, XYZ, A, W, M, False)
    # print(state['~']['_g_embeddings'].shape)
    # print(state['~']['last_hidden_state'].shape)

    # g = state['~']['_g_embeddings']
    # h = state['~']['last_hidden_state']

    # y = jax.vmap(classifier, in_axes=(None, None, 0, 0, 0, 0, None))(classifier_params, key, g, L, W, h, False)
    # print(y.shape)
    # labels = jnp.ones(G.shape)
    # print(jnp.abs(y-labels).shape)

    # test = jax.vmap(lambda x, y: x-y, in_axes=(0, 0))(y, labels)
    # print("test shape:", test.shape)

    labels = jnp.ones(G.shape)
    value = jax.jit(loss_fn, static_argnums=9)(params, state, key, G[:1], L[:1], XYZ[:1], A[:1], W[:1], labels[:1], True)
    print (value)

    value = jax.jit(loss_fn, static_argnums=9)(params, state, key, G[:1], L[:1], XYZ[:1]+1.0, A[:1], W[:1], labels[:1], True)
    print (value)
