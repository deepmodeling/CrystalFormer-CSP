from pyxtal import pyxtal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def find_spg(structure, tol=0.01):
    spga = SpacegroupAnalyzer(structure, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    c.from_seed(crystal, tol=tol)
    return c.group.number
