import jax
import jax.numpy as jnp


def make_cond_logp(logp_fn, process_fn, forward_fn, target, alpha):
    '''
    logp_fn: function to calculate log p(x)
    forward_fn: function to calculate y = f(x)
    process_fn: function to process G, L, XYZ, A, W to forward_fn input
    target: target label
    alpha: hyperparameter to control the trade-off between log p(x) and log p(y|x)
    NOTE that the logp_fn and forward_fn should be vmapped before passing to this function
    '''

    def forward(G, L, XYZ, A, W):
        # TODO: need to be wrapped using callback
        x = process_fn(G, L, XYZ, A, W)
        y = forward_fn(x)
        return y

    def cond_logp_fn(params, key, G, L, XYZ, A, W, is_training):
        '''
        base_params: base model parameters
        cond_params: conditional model parameters
        '''
        # calculate log p(x)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, key, G, L, XYZ, A, W, is_training)
        logp_base = logp_xyz + logp_w + logp_a + logp_l

        # calculate p(y|x)
        y = forward(G, L, XYZ, A, W) # f(x)
        logp_cond = jnp.abs(target - y) # |y - f(x)|

        # trade-off between log p(x) and p(y|x)
        logp = logp_base - alpha * logp_cond.squeeze()
        return logp
    
    return cond_logp_fn
