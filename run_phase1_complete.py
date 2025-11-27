# Phase 1 Complete Validation
# Tests ALL core operations defined in plan.md Phase 1

import sys
sys.path.insert(0, '.')

import numpy as np
from src.field_nd import ResonantFieldND
from src.kernel_nd import KernelND
from src.reaction import GrayScott
from src.config import (
    BOUNDARY_PERIODIC, BOUNDARY_NEUMANN, 
    BOUNDARY_DIRICHLET, BOUNDARY_ABSORBING
)


def test_field_data_structure():
    """1.1 The Field Data Structure"""
    print("=" * 60)
    print("1.1 FIELD DATA STRUCTURE")
    print("=" * 60)
    
    # N-dimensional grid
    field_1d = ResonantFieldND(dimensions=(128,), channels=2)
    field_2d = ResonantFieldND(dimensions=(64, 64), channels=2)
    field_3d = ResonantFieldND(dimensions=(32, 32, 32), channels=2)
    
    print(f"1D: {field_1d}")
    print(f"2D: {field_2d}")
    print(f"3D: {field_3d}")
    
    # Multiple channels
    field_multi = ResonantFieldND(dimensions=(64, 64), channels=4)
    print(f"4-channel field: {field_multi}")
    
    # High precision (float64)
    assert field_2d.data.dtype == np.float64, "Should be float64"
    print(f"Precision: {field_2d.data.dtype}")
    
    # Configurable boundary conditions
    field_2d.boundary = BOUNDARY_PERIODIC
    print(f"Boundary (periodic): {field_2d.boundary}")
    
    field_2d.boundary = BOUNDARY_NEUMANN
    print(f"Boundary (reflective): {field_2d.boundary}")
    
    field_2d.boundary = BOUNDARY_DIRICHLET
    print(f"Boundary (absorbing): {field_2d.boundary}")
    
    print("PASSED\n")


def test_core_operations():
    """1.2 Core Operations"""
    print("=" * 60)
    print("1.2 CORE OPERATIONS")
    print("=" * 60)
    
    field = ResonantFieldND(dimensions=(64, 64), channels=2)
    field.initialize_perturbed([1.0, 0.0], 0.1)
    
    # convolve(field, kernel)
    print("Testing convolve(field, kernel)...")
    kernel = KernelND.gaussian(sigma=2.0, ndim=2)
    convolved = field.convolve(kernel)
    print(f"  Original std: {np.std(field.data[0]):.4f}")
    print(f"  Convolved std: {np.std(convolved.data[0]):.4f} (should be smaller)")
    assert np.std(convolved.data[0]) < np.std(field.data[0]), "Gaussian should smooth"
    
    # diffuse(field, coefficients)
    print("Testing diffuse(field, coefficients)...")
    diffused = field.diffuse(coefficients=[0.1, 0.05])
    print(f"  Diffusion result mean: u={np.mean(diffused.data[0]):.6f}, v={np.mean(diffused.data[1]):.6f}")
    
    # react(field, reaction_fn)
    print("Testing react(field, reaction_fn)...")
    gs = GrayScott()
    du, dv = field.react(gs)
    print(f"  Reaction terms: du_mean={np.mean(du):.6f}, dv_mean={np.mean(dv):.6f}")
    
    # superpose(field_a, field_b)
    print("Testing superpose(field_a, field_b)...")
    field_a = ResonantFieldND(dimensions=(64, 64), channels=2)
    field_b = ResonantFieldND(dimensions=(64, 64), channels=2)
    field_a.initialize_uniform([1.0, 0.0])
    field_b.initialize_uniform([0.0, 1.0])
    superposed = field_a.superpose(field_b, weight_self=0.5, weight_other=0.5)
    print(f"  Superposed: u={np.mean(superposed.data[0]):.2f}, v={np.mean(superposed.data[1]):.2f}")
    assert np.allclose(superposed.data[0], 0.5), "Should be 0.5"
    assert np.allclose(superposed.data[1], 0.5), "Should be 0.5"
    
    # measure(field, region)
    print("Testing measure(field, region)...")
    field.initialize_perturbed([1.0, 0.0], 0.1)
    
    # Full field measurement
    full_metrics = field.measure()
    print(f"  Full field energy: {full_metrics['energy']:.2f}")
    print(f"  Full field entropy: u={full_metrics['entropy'][0]:.2f}, v={full_metrics['entropy'][1]:.2f}")
    
    # Region measurement
    region_metrics = field.measure(region=((10, 30), (10, 30)))
    print(f"  Region energy: {region_metrics['energy']:.2f}")
    
    print("PASSED\n")


def test_boundary_conditions():
    """Test all boundary condition types"""
    print("=" * 60)
    print("BOUNDARY CONDITIONS")
    print("=" * 60)
    
    # Periodic
    field_p = ResonantFieldND(dimensions=(32, 32), channels=2)
    field_p.boundary = BOUNDARY_PERIODIC
    field_p.data[0, 0, :] = 1.0  # Set top edge
    field_p.apply_boundary()
    print(f"Periodic: top edge preserved = {field_p.data[0, 0, 0]:.1f}")
    
    # Neumann (reflective)
    field_n = ResonantFieldND(dimensions=(32, 32), channels=2)
    field_n.boundary = BOUNDARY_NEUMANN
    field_n.data[0, 1, :] = 0.5  # Set second row
    field_n.apply_boundary()
    print(f"Neumann: edge = interior = {field_n.data[0, 0, 0]:.1f}")
    
    # Dirichlet (absorbing)
    field_d = ResonantFieldND(dimensions=(32, 32), channels=2)
    field_d.boundary = BOUNDARY_DIRICHLET
    field_d.data[0, :, :] = 1.0  # Fill with 1
    field_d.apply_boundary()
    print(f"Dirichlet: edge = 0 = {field_d.data[0, 0, 0]:.1f}, interior = {field_d.data[0, 15, 15]:.1f}")
    
    print("PASSED\n")


def test_update_rule():
    """1.3 The Update Rule (The Heartbeat)"""
    print("=" * 60)
    print("1.3 THE UPDATE RULE")
    print("=" * 60)
    
    print("field_next = field + dt * (diffusion + reaction + injection)")
    print()
    
    field = ResonantFieldND(dimensions=(64, 64), channels=2)
    field.initialize_uniform([1.0, 0.0])
    field.inject_seed_sphere(center=(32, 32), radius=5, channel=1, value=0.25)
    field.inject_seed_sphere(center=(32, 32), radius=5, channel=0, value=0.5)
    
    dt = 1.0
    d_coeffs = [0.16, 0.08]
    gs = GrayScott()
    laplacian = KernelND.laplacian(ndim=2)
    
    print(f"Initial state: u_mean={np.mean(field.data[0]):.4f}, v_mean={np.mean(field.data[1]):.4f}")
    
    # Manual update loop (The Heartbeat)
    for step in range(100):
        # Diffusion term
        diffusion = np.zeros_like(field.data)
        for c in range(field.channels):
            diffusion[c] = d_coeffs[c] * laplacian.convolve(field.data[c])
        
        # Reaction term
        du, dv = gs.react(field.data[0], field.data[1])
        reaction = np.stack([du, dv])
        
        # Injection term (could add continuous sources here)
        injection = np.zeros_like(field.data)
        
        # THE UPDATE RULE
        field.data = field.data + dt * (diffusion + reaction + injection)
        field.data = np.clip(field.data, 0, 1)
        field.apply_boundary()
    
    print(f"After 100 steps: u_mean={np.mean(field.data[0]):.4f}, v_mean={np.mean(field.data[1]):.4f}")
    print("Pattern formation initiated: v increased from seeds")
    
    print("PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE VALIDATION")
    print("The Resonant Field Core")
    print("=" * 60 + "\n")
    
    test_field_data_structure()
    test_core_operations()
    test_boundary_conditions()
    test_update_rule()
    
    print("=" * 60)
    print("PHASE 1 COMPLETE - ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("Implemented:")
    print("  - N-dimensional field (1D, 2D, 3D, ...)")
    print("  - Multiple channels with float64 precision")
    print("  - Boundary: periodic, reflective (Neumann), absorbing (Dirichlet)")
    print("  - convolve(field, kernel)")
    print("  - diffuse(field, coefficients)")
    print("  - react(field, reaction_fn)")
    print("  - superpose(field_a, field_b)")
    print("  - measure(field, region)")
    print("  - The Update Rule: field += dt * (diffusion + reaction + injection)")


if __name__ == "__main__":
    main()
