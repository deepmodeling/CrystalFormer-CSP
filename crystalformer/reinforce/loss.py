import jax
import jax.numpy as jnp


def make_reinforce_loss(batch_logp, batch_reward_fn):

    def loss(params, key, x, is_train):
        
        # TODO: now only support for crystalformer logp
        logp_w, logp_xyz, logp_a, logp_l = jax.jit(batch_logp, static_argnums=7)(params, key, *x, is_train)
        entropy = logp_w + logp_xyz + logp_a + logp_l

        f = batch_reward_fn(x)
        f = jax.lax.stop_gradient(f)

        f_mean = jnp.mean(f)

        f_std = jnp.std(f)/jnp.sqrt(f.shape[0])

        return jnp.mean((f - f_mean) * entropy), (f_mean, f_std)

    return loss