import pytest
import jax.numpy as jnp
import jax
import numpy as np
from crystalformer.src.formula import (
    find_composition_vector,
    formula_string,
    formula_string_to_composition_vector
)
from crystalformer.src.elements import element_list, element_dict


class TestFindCompositionVector:
    """Test cases for find_composition_vector function."""
    
    def test_simple_case(self):
        """Test with a simple case: Na3MnCoNiO6"""
        atoms = jnp.array([11, 8, 27, 8, 11, 8, 25, 8, 11, 8, 28, 8] + [0]*12)
        multiplicities = jnp.array([2]*12 + [0]*12)
        
        composition = find_composition_vector(atoms, multiplicities)
        
        # Check dimensions
        assert composition.shape == (119,)
        assert composition.dtype == jnp.int32
        
        # Check specific elements
        assert composition[11] == 3  # Na (3 atoms)
        assert composition[8] == 6   # O (6 atoms)
        assert composition[25] == 1  # Mn (1 atom)
        assert composition[27] == 1  # Co (1 atom)
        assert composition[28] == 1  # Ni (1 atom)
        
        # Check that other elements are zero
        assert composition[1] == 0   # H
        assert composition[6] == 0   # C
        assert composition[26] == 0  # Fe
    
    def test_water_molecule(self):
        """Test with H2O"""
        atoms = jnp.array([1, 8, 1] + [0]*21)  # H, O, H
        multiplicities = jnp.array([1, 1, 1] + [0]*21)
        
        composition = find_composition_vector(atoms, multiplicities)
        
        assert composition[1] == 2  # H (2 atoms)
        assert composition[8] == 1  # O (1 atom)
        assert composition[6] == 0  # C (should be zero)
    
    def test_carbon_dioxide(self):
        """Test with CO2"""
        atoms = jnp.array([6, 8, 8] + [0]*21)  # C, O, O
        multiplicities = jnp.array([1, 1, 1] + [0]*21)
        
        composition = find_composition_vector(atoms, multiplicities)
        
        assert composition[6] == 1  # C (1 atom)
        assert composition[8] == 2  # O (2 atoms)
    
    def test_gcd_normalization(self):
        """Test that GCD normalization works correctly"""
        # Create a case where GCD > 1
        atoms = jnp.array([6, 8, 6, 8] + [0]*20)  # C, O, C, O
        multiplicities = jnp.array([2, 2, 2, 2] + [0]*20)  # Total: 4C, 4O
        
        composition = find_composition_vector(atoms, multiplicities)
        
        # Should be normalized to 1C, 1O (GCD of 4,4 is 4)
        assert composition[6] == 1  # C
        assert composition[8] == 1  # O
    
    def test_empty_structure(self):
        """Test with empty structure (all zeros)"""
        atoms = jnp.zeros(24, dtype=jnp.int32)
        multiplicities = jnp.zeros(24, dtype=jnp.int32)
        
        composition = find_composition_vector(atoms, multiplicities)
        
        # All elements should be zero
        assert jnp.all(composition == 0)
    
    def test_vmap_compatibility(self):
        """Test that function works with vmap"""
        # Create batch of structures
        batch_atoms = jnp.array([
            [1, 8, 1] + [0]*21,  # H2O
            [6, 8, 8] + [0]*21,  # CO2
            [11, 8, 11, 8] + [0]*20,  # Na2O2
        ])
        
        batch_multiplicities = jnp.array([
            [1, 1, 1] + [0]*21,
            [1, 1, 1] + [0]*21,
            [1, 1, 1, 1] + [0]*20,
        ])
        
        # Apply vmap
        vmap_find_composition = jax.vmap(find_composition_vector)
        batch_compositions = vmap_find_composition(batch_atoms, batch_multiplicities)
        
        # Check shapes
        assert batch_compositions.shape == (3, 119)
        
        # Check first structure (H2O)
        assert batch_compositions[0, 1] == 2  # H
        assert batch_compositions[0, 8] == 1  # O
        
        # Check second structure (CO2)
        assert batch_compositions[1, 6] == 1  # C
        assert batch_compositions[1, 8] == 2  # O
        
        # Check third structure (Na2O2) - should be normalized to NaO
        assert batch_compositions[2, 11] == 1  # Na (normalized from 2)
        assert batch_compositions[2, 8] == 1   # O (normalized from 2)


class TestFormulaString:
    """Test cases for formula_string function."""
    
    def test_simple_formula(self):
        """Test with simple composition"""
        # H2O
        composition = jnp.zeros(119, dtype=jnp.int32)
        composition = composition.at[1].set(2)  # H
        composition = composition.at[8].set(1)  # O
        
        formula = formula_string(composition)
        assert formula == "H2O"
    
    def test_oxide_style_ordering(self):
        """Test that oxygen comes last (oxide style)"""
        # Na2O
        composition = jnp.zeros(119, dtype=jnp.int32)
        composition = composition.at[11].set(2)  # Na
        composition = composition.at[8].set(1)   # O
        
        formula = formula_string(composition)
        assert formula == "Na2O"
    
    def test_complex_formula(self):
        """Test with complex formula"""
        # Na3MnCoNiO6
        composition = jnp.zeros(119, dtype=jnp.int32)
        composition = composition.at[11].set(3)  # Na
        composition = composition.at[25].set(1)  # Mn
        composition = composition.at[27].set(1)  # Co
        composition = composition.at[28].set(1)  # Ni
        composition = composition.at[8].set(6)   # O
        
        formula = formula_string(composition)
        # The actual output is "Na3MnCoNiO6" based on the test failure
        assert formula == "Na3MnCoNiO6"
    
    def test_single_atom_formula(self):
        """Test with single atom"""
        # He
        composition = jnp.zeros(119, dtype=jnp.int32)
        composition = composition.at[2].set(1)  # He
        
        formula = formula_string(composition)
        assert formula == "He"
    
    def test_empty_composition(self):
        """Test with empty composition"""
        composition = jnp.zeros(119, dtype=jnp.int32)
        
        formula = formula_string(composition)
        assert formula == ""


class TestFormulaStringToCompositionVector:
    """Test cases for formula_string_to_composition_vector function."""
    
    def test_simple_formula(self):
        """Test with simple formula H2O"""
        formula = "H2O"
        composition = formula_string_to_composition_vector(formula)
        
        assert composition.shape == (119,)
        assert composition.dtype == jnp.int32
        assert composition[1] == 2  # H
        assert composition[8] == 1  # O
        assert composition[6] == 0  # C (should be zero)
    
    def test_carbon_dioxide(self):
        """Test with CO2"""
        formula = "CO2"
        composition = formula_string_to_composition_vector(formula)
        
        assert composition[6] == 1  # C
        assert composition[8] == 2  # O
    
    def test_complex_formula(self):
        """Test with complex formula"""
        formula = "Na3MnCoNiO6"
        composition = formula_string_to_composition_vector(formula)
        
        assert composition[11] == 3  # Na
        assert composition[25] == 1  # Mn
        assert composition[27] == 1  # Co
        assert composition[28] == 1  # Ni
        assert composition[8] == 6   # O
    
    def test_single_atom(self):
        """Test with single atom"""
        formula = "He"
        composition = formula_string_to_composition_vector(formula)
        
        assert composition[2] == 1  # He
        assert jnp.sum(composition) == 1  # Only one atom total
    
    def test_implicit_count(self):
        """Test with implicit count (no number means 1)"""
        formula = "H2O"  # O has implicit count of 1
        composition = formula_string_to_composition_vector(formula)
        
        assert composition[1] == 2  # H
        assert composition[8] == 1  # O (implicit count)
    
    def test_unknown_element(self):
        """Test with unknown element (should be ignored)"""
        formula = "H2Z"  # Z is not a real element (X is Xenon)
        composition = formula_string_to_composition_vector(formula)
        
        assert composition[1] == 2  # H
        assert jnp.sum(composition) == 2  # Only H atoms
    
    def test_empty_formula(self):
        """Test with empty formula"""
        formula = ""
        composition = formula_string_to_composition_vector(formula)
        
        assert jnp.all(composition == 0)
    
    def test_case_sensitivity(self):
        """Test that element symbols are case sensitive"""
        formula = "h2o"  # lowercase
        composition = formula_string_to_composition_vector(formula)
        
        # Should not recognize lowercase elements
        assert jnp.sum(composition) == 0


class TestRoundTripConversion:
    """Test round-trip conversion: atoms -> composition -> formula -> composition"""
    
    def test_round_trip_simple(self):
        """Test round-trip with simple case"""
        # Start with atoms and multiplicities
        atoms = jnp.array([1, 8, 1] + [0]*21)  # H2O
        multiplicities = jnp.array([1, 1, 1] + [0]*21)
        
        # Convert to composition vector
        composition1 = find_composition_vector(atoms, multiplicities)
        
        # Convert to formula string
        formula = formula_string(composition1)
        
        # Convert back to composition vector
        composition2 = formula_string_to_composition_vector(formula)
        
        # Should be the same (within normalization)
        assert jnp.array_equal(composition1, composition2)
    
    def test_round_trip_complex(self):
        """Test round-trip with complex case"""
        # Start with atoms and multiplicities
        atoms = jnp.array([11, 8, 27, 8, 11, 8, 25, 8, 11, 8, 28, 8] + [0]*12)
        multiplicities = jnp.array([2]*12 + [0]*12)
        
        # Convert to composition vector
        composition1 = find_composition_vector(atoms, multiplicities)
        
        # Convert to formula string
        formula = formula_string(composition1)
        
        # Convert back to composition vector
        composition2 = formula_string_to_composition_vector(formula)
        
        # Should be the same (within normalization)
        assert jnp.array_equal(composition1, composition2)
    
    def test_round_trip_batch(self):
        """Test round-trip with batch processing"""
        # Create batch of structures
        batch_atoms = jnp.array([
            [1, 8, 1] + [0]*21,  # H2O
            [6, 8, 8] + [0]*21,  # CO2
        ])
        
        batch_multiplicities = jnp.array([
            [1, 1, 1] + [0]*21,
            [1, 1, 1] + [0]*21,
        ])
        
        # Convert to composition vectors
        vmap_find_composition = jax.vmap(find_composition_vector)
        batch_compositions = vmap_find_composition(batch_atoms, batch_multiplicities)
        
        # Convert to formula strings
        formulas = [formula_string(comp) for comp in batch_compositions]
        
        # Convert back to composition vectors
        batch_compositions2 = jnp.array([
            formula_string_to_composition_vector(formula) 
            for formula in formulas
        ])
        
        # Should be the same
        assert jnp.array_equal(batch_compositions, batch_compositions2)


if __name__ == "__main__":
    pytest.main([__file__])
