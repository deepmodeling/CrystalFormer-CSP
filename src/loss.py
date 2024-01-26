import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from von_mises import von_mises_logpdf
from lattice import make_lattice_mask
from utils import to_A_W, to_AW
from wyckoff import mult_table
from fc_mask import fc_mask_table

def make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, transformer):

    lattice_mask = make_lattice_mask()
    assert mult_table.shape[:2] == fc_mask_table.shape[:2] == (230, 28)

    @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, None), out_axes=0) # batch 
    def logp_fn(params, key, G, L, X, AW, is_train):
        '''
        G: scalar 
        L: (6,) [a, b, c, alpha, beta, gamma] 
        X: (n_max, dim)
        AW: (n_max,)
        '''
        
        dim = X.shape[-1]

        A, W = to_A_W(AW, atom_types) # (n_max,) (n_max,)
        num_sites = jnp.sum(A!=0)
        M = mult_table[G-1, W]  # (n_max,) multplicities
        #num_atoms = jnp.sum(M)

        if is_train:
            #randomly permute atoms with the same wyckoff symbol for data augmentation
            temp = jnp.where(W>0, W, 9999) # change 0 to 9999 so they remain in the end after sort
            key, subkey = jax.random.split(key)
            idx_perm = jax.random.permutation(subkey, jnp.arange(n_max))
            temp = temp[idx_perm]
            idx_sort = jnp.argsort(temp)
            idx = idx_perm[idx_sort]

            X = X[idx]
            A = A[idx]
            W = W[idx]
            M = M[idx]

            AW = to_AW(A, W, atom_types)

        h = transformer(params, key, G, X, A, W, M, is_train)
        h = h.reshape(n_max+1, 2, -1)
        hAW, hXL = h[:, 0, :], h[:, 1, :]

        aw_logit = hAW[:-1] # (n_max, am_types)
        x_logit, loc, kappa, _ = jnp.split(hXL[:-1], [Kx, 
                                                      Kx+Kx*dim, 
                                                      Kx+2*Kx*dim, 
                                                      ], axis=-1) 

        loc = loc.reshape(n_max, Kx, dim)
        kappa = kappa.reshape(n_max, Kx, dim)

        logp_x = jax.vmap(von_mises_logpdf, (None, 1, 1), 1)(X*2*jnp.pi, loc, kappa) # (n_max, Kx, dim)
        logp_x = jax.scipy.special.logsumexp(x_logit[..., None] + logp_x, axis=1) # (n_max, dim)

        fc_mask = jnp.logical_and((AW>0)[:, None], (fc_mask_table[G-1, W]>0)[None, :])
        logp_x = jnp.sum(jnp.where(fc_mask, logp_x, jnp.zeros_like(logp_x)))

        logp_aw = jnp.sum(aw_logit[jnp.arange(n_max), AW.astype(int)])  

        # first convert one-hot to integer, then look for mask
        l_logit, mu, sigma = jnp.split(hXL[num_sites, 
                                           Kx+2*Kx*dim:Kx+2*Kx*dim+Kl+2*6*Kl], [Kl, Kl+Kl*6], axis=-1)
        mu = mu.reshape(Kl, 6)
        sigma = sigma.reshape(Kl, 6)
        logp_l = jax.vmap(jax.scipy.stats.norm.logpdf, (None, 0, 0))(L,mu,sigma) #(Kl, 6)
        logp_l = jax.scipy.special.logsumexp(l_logit[:, None] + logp_l, axis=0) # (6,)
        logp_l = jnp.sum(jnp.where((lattice_mask[G-1]>0), logp_l, jnp.zeros_like(logp_l)))
        
        return logp_x + logp_aw + logp_l

    def loss_fn(params, key, G, L, X, AW, is_train):
        logp = logp_fn(params, key, G, L, X, AW, is_train)
        return -jnp.mean(logp)
        
    return loss_fn

if __name__=='__main__':
    from utils import GLXAW_from_file
    from transformer import make_transformer
    atom_types = 119
    Nf = 5
    n_max = 20
    wyck_types = 20
    Kx, Kl  = 8, 1
    dim = 3
    dropout_rate = 0.1 

    csv_file = '../data/mini.csv'
    G, L, X, AW = GLXAW_from_file(csv_file, atom_types, wyck_types, n_max, dim)

    key = jax.random.PRNGKey(42)

    params, transformer = make_transformer(key, Nf, Kx, Kl, n_max, dim, 128, 4, 4, 8, 16,atom_types, wyck_types, dropout_rate) 

    loss_fn = make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, transformer)
    
    value, grad = jax.value_and_grad(loss_fn)(params, key, G[:1], L[:1], X[:1], AW[:1], True)
    print (value)

    value, grad = jax.value_and_grad(loss_fn)(params, key, G[:1], L[:1], X[:1]-1.0, AW[:1], True)
    print (value)
