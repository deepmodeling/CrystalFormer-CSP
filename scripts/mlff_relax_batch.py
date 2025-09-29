import warnings
# To suppress warnings for clearer output
warnings.simplefilter("ignore")

import pandas as pd
import os
from time import time
from ast import literal_eval
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from BatchRelaxer import BatchRelaxer

from symmetry import find_spg

def make_orb_calc(model_name, model_path, device="cuda"):
    from orb_models.forcefield import pretrained

    if model_name in pretrained.ORB_PRETRAINED_MODELS:
        # Load the ORB forcefield model
        orbff = pretrained.ORB_PRETRAINED_MODELS[model_name](model_path, device=device)

    else:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(pretrained.ORB_PRETRAINED_MODELS.keys())}")

    return orbff


def relax_structures(relaxer, structures):
    """
    Args:
        relaxer: BatchRelaxer object
        structures: List of pymatgen Structure objects

    Returns:
        initial_energies: List of initial energies
        final_energies: List of final energies
        relaxed_cif_dicts: List of relaxed structures in CIF format
        formula_list: List of formulas of the structures

    if relaxation is False, the final energies will be the same as the initial energies
    """

    ase_adaptor = AseAtomsAdaptor()
    atoms_list = [ase_adaptor.get_atoms(struct) for struct in structures]

    relaxation_trajectories = relaxer.relax(atoms_list)

    # Extract the initial structures and energies
    initial_structures = [traj[0] for traj in relaxation_trajectories.values()]
    initial_energies = [structure.info['total_energy'] for structure in initial_structures]

    # Extract the relaxed structures and corresponding energies
    final_structures = [traj[-1] for traj in relaxation_trajectories.values()]
    final_energies = [structure.info['total_energy'] for structure in final_structures]

    formula_list = [struct.composition.formula for struct in structures]

    final_structures = [ase_adaptor.get_structure(struct) for struct in final_structures]
    relaxed_cif_dicts = []
    for structure in final_structures:
        structure = structure.remove_site_property("forces")
        structure.properties = None
        relaxed_cif_dicts.append(structure.as_dict())

    return initial_energies, final_energies, relaxed_cif_dicts, formula_list


def main(args):
    csv_file = os.path.join(args.restore_path, args.filename)

    data = pd.read_csv(csv_file)
    cif_strings = data['cif']

    try: structures = [Structure.from_dict(literal_eval(cif)) for cif in cif_strings]
    except: structures = [Structure.from_str(cif, fmt="cif") for cif in cif_strings]

    if args.primitive:
        print("Converting structures to primitive form...")
        structures = [struct.get_primitive_structure() for struct in structures]

    print("Relaxing structures...")
    if args.relaxation:
        print("Relaxation is enabled. This may take a while.")
    else:
        print("Relaxation is disabled. Only initial energies will be calculated.")

    print(f"Using {args.model} model at {args.model_path}")
    potential = make_orb_calc(args.model, args.model_path, args.device)

    print("Initializing BatchRelaxer...")
    if args.relaxation:
        print(f"Relaxation settings: fmax={args.fmax}, max steps={args.steps}")
    else:
        args.steps = 0  # No relaxation steps
        print(f"Setting max steps to {args.steps} to skip relaxation.")

    relaxer = BatchRelaxer(potential,
                           device=args.device,
                           fmax=args.fmax,
                           max_n_steps=args.steps,
                           max_natoms_per_batch=args.max_natoms_per_batch,
                           filter="FRECHETCELLFILTER",
                           optimizer="FIRE")

    print("Calculating energies...")
    start_time = time()
    results  = relax_structures(relaxer, structures)
    end_time = time()
    print(f"Relaxation took {end_time - start_time:.2f} seconds")

    initial_energies, final_energies, relaxed_cif_dicts, formula_list = results
    output_data = pd.DataFrame()
    output_data['initial_energy'] = initial_energies
    output_data['final_energy'] = final_energies
    output_data['relaxed_cif'] = relaxed_cif_dicts
    output_data['formula'] = formula_list

    print ("spacegroup of unrelaxed structures")
    print ([find_spg(structure, args.tol) for structure in structures])

    if args.relaxation:
        print ("spacegroup of relaxed structures")
        print ([find_spg(Structure.from_dict(structure, args.tol)) for structure in relaxed_cif_dicts])

    if args.label:
        output_data.to_csv(os.path.join(args.restore_path, f"relaxed_structures_{args.label}.csv"),
                           index=False)
    else:
        output_data.to_csv(os.path.join(args.restore_path, "relaxed_structures.csv"),
                           index=False)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="orb-v2", help="choose the specific orb model to use")
    parser.add_argument("--device", type=str, default="cuda", help="choose the device to run the model on")
    parser.add_argument("--model_path", type=str, default="./data/orb-v2-20241011.ckpt", help="path to the model checkpoint")
    parser.add_argument("--restore_path", type=str, default="./experimental/", help="")
    parser.add_argument('--filename', default='output_struct.csv', help='filename of the csv file containing the structures')
    parser.add_argument('--relaxation', action='store_true', help='whether to relax the structures')
    parser.add_argument('--fmax', type=float, default=0.1, help='maximum force tolerance for relaxation')
    parser.add_argument('--tol', type=float, default=0.01, help='tolerance for symmetry analysis')
    parser.add_argument('--steps', type=int, default=500, help='max number of steps for relaxation')
    parser.add_argument('--max_natoms_per_batch', type=int, default=10000, help='maximum number of atoms per batch')
    parser.add_argument('--label', default=None, help='label for the output file')
    parser.add_argument('--primitive', action='store_true', help='convert structures to primitive form')

    args = parser.parse_args()
    main(args)
