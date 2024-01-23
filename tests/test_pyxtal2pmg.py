from pyxtal import pyxtal
import pandas as pd

from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# from pymatgen.io.cif import CifWriter

import multiprocessing
import numpy as np


def process_one(cif_string, tol=0.01):

    # Take from https://anonymous.4open.science/r/DiffCSP-PP-8F0D/diffcsp/common/data_utils.py
    # get_symmetry_info
    crystal = Structure.from_str(cif_string, fmt='cif')
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    # print(crystal)
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)

    # space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)

    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    # print("reduced composition:", crystal.composition)

    # cif = CifWriter(crystal, symprec=0.1)
    # symm_cif_string = cif.__str__()
    # print(symm_cif_string)

    return crystal.as_dict()


csv_file = './data/mp_20/test.csv'

data = pd.read_csv(csv_file)
cif_strings = data['cif']

num_workers = 40
p = multiprocessing.Pool(num_workers)
symm_cif_dicts = p.map_async(process_one, cif_strings).get()
p.close()
p.join()

# 用 symm_cif_dicts 替换 cif_strings
data['cif'] = symm_cif_dicts
data.to_csv('./symm_data/test.csv', index=False)
