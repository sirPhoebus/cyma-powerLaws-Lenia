# Chladni Plate Test - Phase 2.3 Validation
# Tests biharmonic operator, modal frequencies, and pattern formation

import numpy as np
from src.chladni_plate import (
    PlateState, BiharmonicOperator, ChladniPlate, ChladniSimulation
)
from src.config import BOUNDARY_DIRICHLET, BOUNDARY_NEUMANN

# Test parameters
GRID_SIZE = 64
NUM_STEPS = 2000
DT = 0.005


def test_biharmonic_operator():
    """Test the biharmonic operator on known functions."""
    print("=" * 60)
    print("Test 1: Biharmonic Operator")
    print("=" * 60)
    
    # For f(x,y) = sin(kx*x)*sin(ky*y), nabla^4(f) = (kx^2 + ky^2)^2 * f
    size = 64
    kx, ky = 2 * np.pi / size, 2 * np.pi / size
    
    x = np.arange(size)
    y = np.arange(size)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    f = np.sin(kx * X) * np.sin(ky * Y)
    
    # Theoretical biharmonic
    theoretical = (kx**2 + ky**2)**2 * f
    
    # Numerical biharmonic
    biharmonic = BiharmonicOperator(ndim=2, dx=1.0, boundary=BOUNDARY_DIRICHLET)
    numerical = biharmonic.compute(f)
    
    # Compare in interior (avoid boundary effects)
    interior = slice(10, -10)
    error = np.max(np.abs(numerical[interior, interior] - theoretical[interior, interior]))
    rel_error = error / np.max(np.abs(theoretical))
    
    print(f"Max absolute error (interior): {error:.6f}")
    print(f"Relative error: {rel_error*100:.4f}%")
    print("PASS" if rel_error < 0.1 else "FAIL")


def test_modal_frequencies():
    """Test modal frequency computation."""
    print("\n" + "=" * 60)
    print("Test 2: Modal Frequencies")
    print("=" * 60)
    
    state = PlateState(dimensions=(GRID_SIZE, GRID_SIZE))
    state.thickness = 0.01
    state.youngs_modulus = 1.0
    state.poisson_ratio = 0.3
    state.density = 1.0
    
    print(f"Plate properties:")
    print(f"  Thickness h = {state.thickness}")
    print(f"  Young's modulus E = {state.youngs_modulus}")
    print(f"  Poisson ratio nu = {state.poisson_ratio}")
    print(f"  Flexural rigidity D = {state.flexural_rigidity:.6e}")
    
    modes = state.get_resonant_frequencies(max_mode=3)
    print("\nResonant frequencies (first 9 modes):")
    for m, n, f in modes:
        print(f"  Mode ({m},{n}): f = {f:.6f}")
    
    # Check that frequencies increase with mode number
    freqs = [f for _, _, f in modes]
    is_sorted = all(freqs[i] <= freqs[i+1] for i in range(len(freqs)-1))
    print(f"\nFrequencies properly ordered: {is_sorted}")
    print("PASS" if is_sorted else "FAIL")


def test_plate_vibration():
    """Test plate vibration with point driver."""
    print("\n" + "=" * 60)
    print("Test 3: Plate Vibration")
    print("=" * 60)
    
    sim = ChladniSimulation(
        size=GRID_SIZE,
        thickness=0.01,
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        density=1.0,
        damping=0.001,
        dt=DT
    )
    
    # Get first few resonant frequencies
    modes = sim.get_resonant_frequencies(max_mode=3)
    print("Available modes:")
    for m, n, f in modes[:5]:
        print(f"  ({m},{n}): f = {f:.6f}")
    
    # Drive at first mode frequency
    target_mode = modes[0]
    m, n, freq = target_mode
    print(f"\nDriving at mode ({m},{n}) frequency: {freq:.6f}")
    
    sim.drive_at_frequency(freq, amplitude=0.1)
    
    # Run simulation
    print(f"Running {NUM_STEPS} steps...")
    rms_history = []
    for i in range(NUM_STEPS):
        sim.step()
        if (i + 1) % 500 == 0:
            rms = np.sqrt(np.mean(sim.get_amplitude()**2))
            rms_history.append(rms)
            print(f"  Step {i+1}: RMS = {rms:.6f}")
    
    # Check that amplitude built up
    final_rms = rms_history[-1]
    print(f"\nFinal RMS amplitude: {final_rms:.6f}")
    print("PASS (vibration established)" if final_rms > 1e-6 else "FAIL")
    
    return sim


def test_nodal_pattern():
    """Test nodal pattern detection."""
    print("\n" + "=" * 60)
    print("Test 4: Nodal Pattern Detection")
    print("=" * 60)
    
    # Balanced parameters for stable simulation with reasonable frequencies
    sim = ChladniSimulation(
        size=GRID_SIZE,
        thickness=0.5,
        youngs_modulus=100.0,  # Moderate stiffness
        poisson_ratio=0.3,
        density=1.0,
        damping=0.01,
        dt=0.01
    )
    
    # Get modes and print frequencies
    modes = sim.get_resonant_frequencies(max_mode=3)
    print("Available modes (scaled parameters):")
    for m, n, f in modes[:5]:
        print(f"  ({m},{n}): f = {f:.4f}")
    
    # Drive at (2,1) or (1,2) mode for asymmetric pattern
    target = modes[1]  # Second lowest frequency
    m, n, freq = target
    print(f"\nDriving at mode ({m},{n}) frequency: {freq:.4f}")
    
    sim.drive_at_frequency(freq, amplitude=1.0)
    
    # Run simulation
    total_steps = 5000
    print(f"Running {total_steps} steps...")
    
    for i in range(total_steps):
        sim.step()
        if (i + 1) % 1000 == 0:
            rms = np.sqrt(np.mean(sim.get_amplitude()**2))
            print(f"  Step {i+1}: RMS = {rms:.6f}")
    
    # Get nodal pattern - analyze interior only (exclude clamped boundary)
    amp = sim.get_amplitude()
    max_amp = np.max(np.abs(amp))
    print(f"Max amplitude: {max_amp:.6f}")
    
    if max_amp > 1e-10:
        # Analyze interior region only (exclude boundary effects)
        interior = slice(10, -10)
        amp_interior = amp[interior, interior]
        max_interior = np.max(np.abs(amp_interior))
        
        # Check for spatial structure (mode pattern)
        # A proper mode should have clear nodes in interior
        normalized = np.abs(amp_interior) / max_interior
        nodes_interior = normalized < 0.15
        node_fraction = np.mean(nodes_interior)
        
        # Check variance - a real pattern has spatial structure
        variance = np.var(normalized)
        
        print(f"Interior nodal fraction: {node_fraction*100:.1f}%")
        print(f"Pattern variance: {variance:.4f}")
        
        # Pass if we have spatial structure and some nodes
        passed = variance > 0.01 and node_fraction > 0.05
    else:
        print("Amplitude too low to detect pattern")
        passed = False
    
    print("PASS" if passed else "FAIL")
    return sim


def test_energy_behavior():
    """Test energy behavior with damping."""
    print("\n" + "=" * 60)
    print("Test 5: Energy Behavior")
    print("=" * 60)
    
    sim = ChladniSimulation(
        size=GRID_SIZE,
        thickness=0.01,
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        density=1.0,
        damping=0.01,  # Significant damping
        dt=DT
    )
    
    # Initial impulse
    center = GRID_SIZE // 2
    sim.state.amplitude[0, center-3:center+3, center-3:center+3] = 1.0
    
    e0 = sim.state.compute_total_energy()
    print(f"Initial energy: {e0:.6f}")
    
    # Run with damping, no driving
    sim.plate._source_fn = None
    
    for i in range(NUM_STEPS):
        sim.step()
    
    e_final = sim.state.compute_total_energy()
    energy_ratio = e_final / e0
    
    print(f"Final energy: {e_final:.6f}")
    print(f"Energy ratio: {energy_ratio:.4f}")
    print("PASS (energy dissipated)" if e_final < e0 else "FAIL")


def test_mode_overlap():
    """Test mode shape overlap computation."""
    print("\n" + "=" * 60)
    print("Test 6: Mode Shape Overlap")
    print("=" * 60)
    
    sim = ChladniSimulation(
        size=GRID_SIZE,
        thickness=0.01,
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        density=1.0,
        damping=0.0005,
        dt=DT
    )
    
    # Drive at (1,1) mode
    freq = sim.drive_at_mode(1, 1, amplitude=0.1)
    print(f"Driving at mode (1,1) frequency: {freq:.6f}")
    
    # Run to build pattern
    for i in range(NUM_STEPS * 3):
        sim.step()
    
    # Check overlap with (1,1) mode
    overlap_11 = sim.plate.compute_mode_overlap(1, 1)
    overlap_12 = sim.plate.compute_mode_overlap(1, 2)
    overlap_22 = sim.plate.compute_mode_overlap(2, 2)
    
    print(f"Overlap with (1,1): {overlap_11:.4f}")
    print(f"Overlap with (1,2): {overlap_12:.4f}")
    print(f"Overlap with (2,2): {overlap_22:.4f}")
    
    # (1,1) overlap should be highest
    is_best = overlap_11 > overlap_12 and overlap_11 > overlap_22
    print("PASS (correct mode excited)" if is_best else "FAIL")


if __name__ == "__main__":
    print("Chladni Plate Dynamics - Phase 2.3 Validation")
    print("=" * 60)
    
    test_biharmonic_operator()
    test_modal_frequencies()
    test_plate_vibration()
    test_nodal_pattern()
    test_energy_behavior()
    test_mode_overlap()
    
    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
