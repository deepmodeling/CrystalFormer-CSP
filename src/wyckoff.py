import pandas as pd
import os
import numpy as np 
import re
import jax.numpy as jnp

def from_xyz_str(xyz_str: str):
    """
    Args:
        xyz_str: string of the form 'x, y, z', '-x, -y, z', '-2y+1/2, 3x+1/2, z-y+1/2', etc.
    Returns:
        affine operator as a 3x4 array
    """
    rot_matrix = np.zeros((3, 3))
    trans = np.zeros(3)
    tokens = xyz_str.strip().replace(" ", "").lower().split(",")
    re_rot = re.compile(r"([+-]?)([\d\.]*)/?([\d\.]*)([x-z])")
    re_trans = re.compile(r"([+-]?)([\d\.]+)/?([\d\.]*)(?![x-z])")
    for i, tok in enumerate(tokens):
        # build the rotation matrix
        for m in re_rot.finditer(tok):
            factor = -1.0 if m.group(1) == "-" else 1.0
            if m.group(2) != "":
                factor *= float(m.group(2)) / float(m.group(3)) if m.group(3) != "" else float(m.group(2))
            j = ord(m.group(4)) - 120
            rot_matrix[i, j] = factor
        # build the translation vector
        for m in re_trans.finditer(tok):
            factor = -1 if m.group(1) == "-" else 1
            num = float(m.group(2)) / float(m.group(3)) if m.group(3) != "" else float(m.group(2))
            trans[i] = num * factor
    return np.concatenate( [rot_matrix, trans[:, None]], axis=1) # (3, 4)


df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/wyckoff_list.csv'))
df['Wyckoff Positions'] = df['Wyckoff Positions'].apply(eval)  # convert string to list
wyckoff_positions = df['Wyckoff Positions'].tolist()

symops = np.zeros((230, 28, 576, 3, 4)) # 576 is the least common multiple for all possible mult
mult_table = np.zeros((230, 28), dtype=int) # mult_table[g-1, w] = multiplicity , 28 because we had pad 0 
wmax_table = np.zeros((230,), dtype=int)    # wmax_table[g-1] = number of possible wyckoff letters for g 
dof0_table = np.ones((230, 28), dtype=bool)  # dof0_table[g-1, w] = True for those wyckoff points with dof = 0 (no continuous dof)

for g in range(230):
    wyckoffs = []
    for x in wyckoff_positions[g]:
        wyckoffs.append([])
        for y in x:
            wyckoffs[-1].append(from_xyz_str(y))
    wyckoffs = wyckoffs[::-1] # a-z,A

    mult = [len(w) for w in wyckoffs]
    mult_table[g, 1:len(mult)+1] = mult
    wmax_table[g] = len(mult)

    print (g+1, [len(w) for w in wyckoffs])
    for w, wyckoff in enumerate(wyckoffs):
        wyckoff = np.array(wyckoff)
        repeats = symops.shape[2] // wyckoff.shape[0]
        symops[g, w+1, :, :, :] = np.tile(wyckoff, (repeats, 1, 1))
        dof0_table[g, w+1] = np.linalg.matrix_rank(wyckoff[0, :3, :3]) == 0

symops = jnp.array(symops)
mult_table = jnp.array(mult_table)
wmax_table = jnp.array(wmax_table)
dof0_table = jnp.array(dof0_table)

def symmetrize_atoms(g, w, x):
    '''
    Args:
       g: int 
       w: int
       x: (3,)
    Returns:
       xs: (M, 3)  symmetrize atom positions
    '''
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]
    affine_point = jnp.array([*x, 1]) # (4, )
    xs = []
    for op in ops:
        xs.append( jnp.dot(op, affine_point)[None, :])
    xs = jnp.concatenate(xs, axis=0) # (M, 3)
    xs -= jnp.floor(xs) # wrap back to 0-1 
    return xs

if __name__=='__main__':
    print (symops.shape)
    print (symops.size*symops.dtype.itemsize//(1024*1024))

    import numpy as np 
    np.set_printoptions(threshold=np.inf)

    print (symops[166-1,3, :6])
    op = symops[166-1, 3, 0]
    print (op)
    print ((jnp.abs(op[:3, :3]).sum(axis=1)!=0)) # fc_mask

    print (mult_table[166-1])

    sys.exit(0)
    
    print ('mult_table')
    print (mult_table[25-1]) # space group id -> multiplicity table
    print (mult_table[42-1])
    print (mult_table[47-1])
    print (mult_table[99-1])
    print (mult_table[123-1])
    print (mult_table[221-1])
    print (mult_table[166-1])

    print ('dof0_table')
    print (dof0_table[25-1])
    print (dof0_table[42-1])
    print (dof0_table[47-1])
    print (dof0_table[225-1])
    print (dof0_table[166-1])
    
    print ('wmax_table')
    print (wmax_table[47-1])
    print (wmax_table[123-1])
    print (wmax_table[166-1])

    print ('wmax_table', wmax_table)
    
    atom_types = 119 
    aw_max = wmax_table*(atom_types-1)    # the maximum value of aw
    print ( (aw_max-1)%(atom_types-1)+1 ) # = 118 
    print ( (aw_max-1)//(atom_types-1)+1 ) # = wmax
