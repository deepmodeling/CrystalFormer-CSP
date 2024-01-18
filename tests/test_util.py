import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from pymatgen.core.structure import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def GLXAM_from_structures(structures, atom_types, mult_types, n_max, dim):
    G = [] # space group
    L = [] # abc alpha beta gamma
    X = [] # fractional coordinate 
    AM = [] # atom type and multiplicity; 0 for placeholder
    for i, structure in enumerate(structures):
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
        refined_structure = analyzer.get_refined_structure()
        analyzer = SpacegroupAnalyzer(refined_structure)
        symmetrized_structure = analyzer.get_symmetrized_structure()

        G.append ([analyzer.get_space_group_number()])
        L.append (symmetrized_structure.lattice.abc+ symmetrized_structure.lattice.angles)
        num_sites = len(symmetrized_structure.equivalent_sites)
        frac_coords = jnp.array([site[0].frac_coords for site in 
                                symmetrized_structure.equivalent_sites]).reshape(num_sites, dim)
        frac_coords = jnp.concatenate([frac_coords, 
                                       jnp.full((n_max - num_sites, dim), 1e10)], 
                                       axis = 0)
        X.append (frac_coords)  
    
        #print (analyzer.get_space_group_number(), [site[0].specie.number for site in symmetrized_structure.equivalent_sites])
        
        am = []
        for site in symmetrized_structure.equivalent_sites:
            a = site[0].specie.number # element number 
            assert (a < atom_types)
            m = len(site)             # multiplicity
            # print (f'sturct{i} {a}  {m}')
            am.append( (m-1) * (atom_types-1)+ (a-1) )
        AM.append( am + [0] * (n_max - num_sites) )
   
    G = jnp.array(G)
    # G = jax.nn.one_hot(G-1, 230).reshape(-1, 230) # G-1 to shift 1-230 to 0-229  # for test
    L = jnp.array(L).reshape(-1, 6)

    X = jnp.array(X).reshape(-1, n_max, dim)
    
    AM = jnp.array(AM).reshape(-1, n_max)
    am_types = (atom_types -1)*(mult_types -1) + 1
    AM = jax.nn.one_hot(AM, am_types) # (-1, n_max, am_types)
    return G, L, X, AM


def main():
    data = pd.read_csv('/data/zdcao/crystal_gpt/dataset/mp_20/train.csv')
    cif_strings = data['cif']
    structures = [Structure.from_str(cif_string, fmt='cif') for cif_string in cif_strings]

    G, L, X, AM = GLXAM_from_structures(structures, atom_types=118, mult_types=6, n_max=20, dim=3)

    # Iterate over G and L arrays
    for idx, (space_group, lattice_params) in enumerate(zip(G, L)):

        a, b, c, alpha, beta, gamma = lattice_params

        angles_epsilon = 0.5  # numerical error tolerance
        abc_epsilon = 1e-3  # numerical error tolerance

        # Apply constraints based on space group number (g)
        if space_group < 3:
            continue  # no constraints
        elif space_group < 16:
            assert np.allclose([alpha, gamma], [90, 90], atol=angles_epsilon)
        elif space_group < 75:
            assert np.allclose([alpha, beta, gamma], [90, 90, 90], atol=angles_epsilon)
        elif space_group < 143:
            assert np.allclose([alpha, beta, gamma], [90, 90, 90], atol=angles_epsilon)
            assert np.allclose(a, b, atol=abc_epsilon)
        elif space_group < 195:
            assert np.allclose([alpha, beta, gamma], [90, 90, 120], atol=angles_epsilon)
            assert np.allclose(a, b, atol=abc_epsilon)
        else:
            assert np.allclose([alpha, beta, gamma], [90, 90, 90], atol=angles_epsilon)
            assert np.allclose([a, b, c], [a, b, c], atol=abc_epsilon)


if __name__ == "__main__":
    main()
