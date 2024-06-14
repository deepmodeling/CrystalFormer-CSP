import jax
import jax.numpy as jnp
from functools import partial


def make_classifier_loss(classifier):

    @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, None))
    def mae_loss(params, key, g, l, w, h, labels, is_training):
        y = classifier(params, key, g, l, w, h, is_training)
        return jnp.abs(y - labels)
    
    def loss_fn(params, key, g, l, w, h, labels, is_training):
        loss = jnp.mean(mae_loss(params, key, g, l, w, h, labels, is_training))
        return loss
    
    return loss_fn