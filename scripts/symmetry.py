from pyxtal import pyxtal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def find_spg(structure, tol=0.01):
    try:
        spga = SpacegroupAnalyzer(structure, symprec=tol)
        crystal = spga.get_refined_structure()
        c = pyxtal()
        c.from_seed(crystal, tol=tol)
        spacegroup = c.group.number 
    except:
        spacegroup = None
    return spacegroup
