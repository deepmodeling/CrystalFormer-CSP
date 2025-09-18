import jax.numpy as jnp
import jax
from jax import lax
from math import gcd
from functools import reduce

from crystalformer.src.elements import element_list, element_dict

def find_composition_vector(atoms, multiplicities):
    """Sum counts per atomic number and divide by the GCD for simplest formula.
    
    Args:
        atoms: JAX array of atomic numbers (Z values)
        multiplicities: JAX array of multiplicities
        
    Returns:
        JAX array of length 119 with counts at index Z (H=1, He=2, ...)
    """
    # Create a composition vector of length 119 (index 0 unused, H=1, He=2, ...)
    composition = jnp.zeros(119, dtype=jnp.int32)
    
    # Use JAX operations to sum multiplicities for each atomic number
    def update_composition(i, composition):
        z = atoms[i]
        m = multiplicities[i]
        # Only update if both z and m are non-zero
        mask = (z > 0) & (m > 0)
        # Use atomic number directly as index (H=1, He=2, ...)
        composition = jnp.where(
            mask & (z >= 1) & (z < 119),
            composition.at[z].add(m),
            composition
        )
        return composition
    
    # Apply the update function across all atoms
    composition = lax.fori_loop(0, len(atoms), update_composition, composition)
    
    # Calculate GCD and normalize
    
    # Calculate GCD using JAX operations
    def gcd_reduce(a, b):
        return jnp.gcd(a, b)

    # Get the first non-zero value as initial GCD
    nonzero_mask = composition > 0
    first_nonzero_idx = jnp.argmax(nonzero_mask)
    initial_gcd = composition[first_nonzero_idx]
    
    # Compute GCD with all other non-zero values
    def gcd_scan(carry, x):
        current_gcd = carry
        new_gcd = jnp.gcd(current_gcd, x)
        return new_gcd, new_gcd
    
    # Apply scan to compute final GCD
    final_gcd, _ = lax.scan(gcd_scan, initial_gcd, composition)
    
    # Normalize by dividing by GCD
    normalized_composition = jnp.where(
        final_gcd > 1,
        composition // final_gcd,
        composition
    )
    
    return normalized_composition

def formula_string(composition_vector):
    """Make a formula string; put oxygen (Z=8) at the end (oxide style), others by increasing Z.
    
    Args:
        composition_vector: JAX array of length 119 with counts at index Z (H=1, He=2, ...)
        
    Returns:
        String representation of the chemical formula
    """
    # Convert JAX array to numpy for easier processing
    composition = jnp.asarray(composition_vector)
    
    # Find non-zero elements and their atomic numbers
    nonzero_indices = jnp.where(composition > 0)[0]
    atomic_numbers = nonzero_indices  # Atomic numbers are directly the indices
    counts = composition[nonzero_indices]
    
    # Sort elements (oxygen last)
    oxygen_idx = jnp.where(atomic_numbers == 8)[0]
    other_indices = jnp.where(atomic_numbers != 8)[0]
    
    # Sort other elements by atomic number
    if len(other_indices) > 0:
        other_atomic_nums = atomic_numbers[other_indices]
        other_counts = counts[other_indices]
        sorted_other_indices = jnp.argsort(other_atomic_nums)
        sorted_other_nums = other_atomic_nums[sorted_other_indices]
        sorted_other_counts = other_counts[sorted_other_indices]
    else:
        sorted_other_nums = jnp.array([])
        sorted_other_counts = jnp.array([])
    
    # Combine with oxygen at the end
    if len(oxygen_idx) > 0:
        final_atomic_nums = jnp.concatenate([sorted_other_nums, jnp.array([8])])
        final_counts = jnp.concatenate([sorted_other_counts, counts[oxygen_idx]])
    else:
        final_atomic_nums = sorted_other_nums
        final_counts = sorted_other_counts
    
    # Build formula string
    parts = []
    for z, n in zip(final_atomic_nums, final_counts):
        sym = element_list[int(z)]  # element_list is a list, index corresponds to atomic number
        n_int = int(n)
        parts.append(sym if n_int == 1 else f"{sym}{n_int}")
    
    return "".join(parts)

def formula_string_to_composition_vector(formula_string):
    """Convert a chemical formula string to a 119-dimensional composition vector.
    
    Args:
        formula_string: String representation of chemical formula (e.g., "H2O", "Na3MnCoNiO6")
        
    Returns:
        JAX array of length 119 with counts at index Z (H=1, He=2, ...)
    """
    import re
    
    # Create composition vector
    composition = jnp.zeros(119, dtype=jnp.int32)
    
    # Regular expression to match element symbols and their counts
    # Matches: element symbol (1-2 letters) followed by optional count
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula_string)
    
    for symbol, count_str in matches:
        if symbol in element_dict:
            z = element_dict[symbol]
            count = int(count_str) if count_str else 1  # Default count is 1
            
            # Add to composition vector
            composition = composition.at[z].add(count)
    
    return composition

if __name__=='__main__':

    atoms = jnp.array([11, 8, 27, 8, 11, 8, 25, 8, 11, 8, 28, 8] + [0]*12)
    multiplicities = jnp.array([2]*12 + [0]*12)

    composition = find_composition_vector(atoms, multiplicities)
    formula = formula_string(composition)

    print("Formula:", formula)        # -> Na3MnCoNiO6
    
    # Show the full 119-vector
    print("Full 119-vector:", composition)
    
    # Test formula string to composition vector conversion
    print("\n=== Testing formula string to composition vector ===")
    
    test_formulas = ["H2O", "CO2", "Na3MnCoNiO6", "Fe2O3", "CaCO3"]
    
    for formula in test_formulas:
        comp_vec = formula_string_to_composition_vector(formula)
        # Show non-zero entries
        nonzero_indices = jnp.where(comp_vec > 0)[0]
        atomic_numbers = nonzero_indices
        counts = comp_vec[nonzero_indices]
        
        print(f"Formula: {formula}")
        print("  Nonzero entries:")
        for z, n in zip(atomic_numbers, counts):
            element_symbol = element_list[int(z)]
            print(f"    {element_symbol} (Z={int(z)}): {int(n)}")
        print()
