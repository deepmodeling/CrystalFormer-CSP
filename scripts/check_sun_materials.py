import os
import pandas as pd

from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher


def make_compare_structures(StructureMatcher):
    
    def compare_structures(s1, s2):
        if s1.composition.reduced_composition != s2.composition.reduced_composition:
            return False
        else:
            return StructureMatcher.fit(s1, s2)

    return compare_structures


def make_search_duplicate(ref_data, StructureMatcher):

    def search_duplicate(s):
        # pick all structures with the same composition
        sub_data = ref_data[ref_data['composition'] == s.composition.reduced_composition]

        duplicate = False
        # compare the structure with all structures with the same composition
        for s2 in sub_data['structure']:
            s2 = Structure.from_dict(eval(s2))
            if StructureMatcher.fit(s, s2):
                duplicate = True
                break

        return duplicate
    
    return search_duplicate


def main(args):

    # print all the parameters
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    data = pd.read_csv(os.path.join(args.restore_path, args.filename))
    ref_data = pd.read_csv(args.ref_path)

    sm = StructureMatcher()
    compare_structures = make_compare_structures(sm)

    # remove unstable structures
    data = data[data['relaxed_ehull'] <= 0.1]
    structures = [Structure.from_dict(eval(crys_dict)) for crys_dict in data['relaxed_cif']]
    print(f"Number of stable structures: {len(structures)}")

    # remove duplicates (Uniqueness)
    idx_list = []
    unique_structures = []
    for idx, s in enumerate(structures):
        if not any([compare_structures(s, us) for us in unique_structures]):
            unique_structures.append(s)
            idx_list.append(idx)

    data = data.iloc[idx_list]
    print(f"Number of stable and unique structures: {len(unique_structures)}")

    # remove structures that are already in the reference data (Novelty)
    comp_list = []
    for idx, formula in enumerate(ref_data['formula']):
        try:
            comp = Composition(formula)
            comp_list.append(comp)
        except Exception as e:
            # Can't parse formula when formula is NaN
            print(e)
            print(f"Error with formula {formula}")
            if ref_data.iloc[idx]['elements'] == "['Na', 'N']":
                comp_list.append(Composition("NaN"))

    print(len(comp_list))
    comp_list = [comp.reduced_composition for comp in comp_list]
    ref_data['composition'] = comp_list

    search_duplicate = make_search_duplicate(ref_data, sm)
    duplicate_list = list(map(lambda s: search_duplicate(s), unique_structures))

    # pick the idx of False in duplicate_list
    idx_list = [idx for idx, duplicate in enumerate(duplicate_list) if not duplicate]
    data = data.iloc[idx_list]
    print(f"Number of stable, unique and novel structures: {data.shape[0]}")
    data.to_csv(os.path.join(args.restore_path, "stable_unique_novel_structures.csv"), index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Check the stable, Unique and Novelty structures")
    parser.add_argument("--restore_path", type=str, default=None, help="Path to the restored data")
    parser.add_argument("--filename", type=str, default="relaxed_structures_ehull.csv", help="Filename of the restored data")
    parser.add_argument("--ref_path", type=str, default="/data/zdcao/crystal_gpt/dataset/alex/PBE/alex20/alex20.csv", help="Path to the reference data")
    args = parser.parse_args()
    main(args)
