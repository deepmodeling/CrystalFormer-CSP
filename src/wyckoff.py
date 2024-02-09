import pandas as pd
import os
import numpy as np 
import re
import jax
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

#https://github.com/materialsproject/pymatgen/blob/1e347c42c01a4e926e15b910cca8964c1a0cc826/pymatgen/symmetry/groups.py#L547
def in_array_list(array_list: list[np.ndarray], arr: np.ndarray, tol: float = 1e-5) -> bool:
    """Extremely efficient nd-array comparison using numpy's broadcasting. This
    function checks if a particular array a, is present in a list of arrays.
    It works for arrays of any size, e.g., even matrix searches.

    Args:
        array_list ([array]): A list of arrays to compare to.
        arr (array): The test array for comparison.
        tol (float): The tolerance. Defaults to 1e-5. If 0, an exact match is done.

    Returns:
        (bool)
    """
    if len(array_list) == 0:
        return False
    axes = tuple(range(1, arr.ndim + 1))
    if not tol:
        return any(np.all(array_list == arr[None, :], axes))
    return any(np.sum(np.abs(array_list - arr[None, :]), axes) < tol)

def symmetrize_atoms_deduplication(g, w, x):
    '''
    symmetrize atoms via deduplication
    this implements the same method as pmg get_orbit function, see
    #https://github.com/materialsproject/pymatgen/blob/1e347c42c01a4e926e15b910cca8964c1a0cc826/pymatgen/symmetry/groups.py#L328
    Args:
       g: int 
       w: int
       x: (3,)
    Returns:
       xs: (m, 3)  symmetrized atom positions
    '''
    # (1) apply all space group symmetry ops to x 
    w_max = wmax_table[g-1].item()
    m_max = mult_table[g-1, w_max].item()
    ops = symops[g-1, w_max, :m_max] # (m_max, 3, 4)
    affine_point = jnp.array([*x, 1]) # (4, )
    coords = ops@affine_point # (m_max, 3) 
    
    # (2) deduplication to select the orbit 
    orbit: list[np.ndarray] = []
    for pp in coords:
        pp = np.mod(np.round(pp, decimals=10), 1) # round and mod to avoid duplication
        if not in_array_list(orbit, pp):
            orbit.append(pp)
    orbit -= np.floor(orbit)   # wrap back to 0-1 
    assert (orbit.shape[0] == mult_table[g-1, w]) # double check that the orbit has the right length
    return orbit

def symmetrize_atoms(g, w, x):
    '''
    symmetrize atoms via, apply all sg symmetry op, finding the generator, and lastly apply symops 
    Args:
       g: int 
       w: int
       x: (3,)
    Returns:
       xs: (m, 3) symmetrize atom positions
    '''

    # (1) apply all space group symmetry op to the x 
    w_max = wmax_table[g-1].item()
    m_max = mult_table[g-1, w_max].item()
    ops = symops[g-1, w_max, :m_max] # (m_max, 3, 4)
    affine_point = jnp.array([*x, 1]) # (4, )
    coords = ops@affine_point # (m_max, 3) 
    coords -= jnp.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
    # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs  
    #https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
    def dist_to_op0x(coord):
        diff = jnp.dot(symops[g-1, w, 0], jnp.array([*coord, 1])) - coord
        diff -= jnp.floor(diff)
        return jnp.sum(diff**2) 
    loc = jnp.argmin(jax.vmap(dist_to_op0x)(coords))
    x = coords[loc].reshape(3,)

    # (3) lastly, apply the given symmetry op to x
    m = mult_table[g-1, w] 
    ops = symops[g-1, w, :m]   # (m, 3, 4)
    affine_point = jnp.array([*x, 1]) # (4, )
    xs = ops@affine_point # (m, 3)
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
    
    w_max = wmax_table[225-1]
    m_max = mult_table[225-1, w_max]
    print ('w_max, m_max', w_max, m_max)
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
