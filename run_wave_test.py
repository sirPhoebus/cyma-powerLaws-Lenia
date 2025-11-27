# Wave Equation Test - Phase 2.2 Validation
# Tests wave propagation, energy conservation, and damping

import numpy as np
from src.wave_state import create_wave_state_2d
from src.wave_equation import WaveEquation, WaveSimulation
from src.config import WAVE_SPEED, WAVE_DAMPING

# Test parameters
GRID_SIZE = 128
CENTER = (GRID_SIZE // 2, GRID_SIZE // 2)
PULSE_SIGMA = 5.0
PULSE_AMPLITUDE = 1.0
NUM_STEPS = 500
DT = 0.05


def test_wave_propagation():
    """Test basic wave propagation from a central pulse."""
    print("=" * 60)
    print("Test 1: Wave Propagation")
    print("=" * 60)
    
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.0  # Undamped for energy conservation test
    )
    
    # Check CFL stability
    stable, cfl_limit = sim.equation.check_stability()
    print(f"CFL limit: {cfl_limit:.4f}, dt={DT}, stable={stable}")
    
    # Inject central pulse
    sim.inject_pulse(CENTER, sigma=PULSE_SIGMA, amplitude=PULSE_AMPLITUDE)
    
    # Measure initial state
    e0 = sim.state.compute_total_energy()
    print(f"Initial energy: {e0:.6f}")
    print(f"Initial RMS amplitude: {sim.state.compute_amplitude_rms():.6f}")
    
    # Run simulation
    print(f"\nRunning {NUM_STEPS} steps...")
    energy_samples = []
    for i in range(NUM_STEPS):
        sim.step(method="leapfrog")
        if (i + 1) % 100 == 0:
            e = sim.state.compute_total_energy()
            energy_samples.append(e)
            print(f"  Step {i+1}: energy={e:.6f}, RMS={sim.state.compute_amplitude_rms():.6f}")
    
    # Check energy conservation
    e_final = energy_samples[-1]
    energy_drift = abs(e_final - e0) / e0
    print(f"\nEnergy conservation: drift = {energy_drift*100:.4f}%")
    print("PASS" if energy_drift < 0.01 else "FAIL")
    
    return sim


def test_damping():
    """Test energy dissipation with damping."""
    print("\n" + "=" * 60)
    print("Test 2: Damping")
    print("=" * 60)
    
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.05  # Significant damping
    )
    
    sim.inject_pulse(CENTER, sigma=PULSE_SIGMA, amplitude=PULSE_AMPLITUDE)
    
    e0 = sim.state.compute_total_energy()
    print(f"Initial energy: {e0:.6f}")
    
    # Run with damping
    for i in range(NUM_STEPS):
        sim.step(method="leapfrog")
    
    e_final = sim.state.compute_total_energy()
    energy_ratio = e_final / e0
    print(f"Final energy: {e_final:.6f}")
    print(f"Energy ratio (final/initial): {energy_ratio:.4f}")
    print("PASS (energy decreased)" if e_final < e0 else "FAIL")
    
    return sim


def test_boundary_reflection():
    """Test wave reflection at boundaries."""
    print("\n" + "=" * 60)
    print("Test 3: Boundary Reflection (Neumann)")
    print("=" * 60)
    
    from src.config import BOUNDARY_NEUMANN
    
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.0
    )
    sim.state.boundary = BOUNDARY_NEUMANN
    
    # Off-center pulse to hit boundary
    off_center = (GRID_SIZE // 4, GRID_SIZE // 2)
    sim.inject_pulse(off_center, sigma=PULSE_SIGMA, amplitude=PULSE_AMPLITUDE)
    
    e0 = sim.state.compute_total_energy()
    
    # Run long enough to hit boundaries and reflect
    for i in range(NUM_STEPS * 2):
        sim.step(method="leapfrog")
    
    e_final = sim.state.compute_total_energy()
    energy_drift = abs(e_final - e0) / e0
    print(f"Energy drift after reflections: {energy_drift*100:.4f}%")
    print("PASS" if energy_drift < 0.05 else "FAIL")
    
    return sim


def test_continuous_source():
    """Test continuous wave source (oscillating point)."""
    print("\n" + "=" * 60)
    print("Test 4: Continuous Source")
    print("=" * 60)
    
    FREQUENCY = 0.5
    SOURCE_AMPLITUDE = 1.0  # Stronger source to overcome damping
    
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.005  # Reduced damping
    )
    
    # Define oscillating point source
    def source_fn(state, t):
        source = np.zeros(state.shape, dtype=np.float64)
        cx, cy = CENTER
        source[0, cx-2:cx+2, cy-2:cy+2] = SOURCE_AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * t)
        return source
    
    sim.set_source(source_fn)
    
    # Run to build up standing wave pattern
    print(f"Running with continuous source at f={FREQUENCY} Hz...")
    for i in range(NUM_STEPS * 2):
        sim.step(method="leapfrog")
        if (i + 1) % 200 == 0:
            rms = sim.state.compute_amplitude_rms()
            print(f"  Step {i+1}: RMS amplitude = {rms:.6f}")
    
    final_rms = sim.state.compute_amplitude_rms()
    print(f"Final RMS: {final_rms:.6f}")
    print("PASS (waves established)" if final_rms > 0.01 else "FAIL")
    
    return sim


def test_3d_wave():
    """Test wave propagation in 3D."""
    print("\n" + "=" * 60)
    print("Test 5: 3D Wave Propagation")
    print("=" * 60)
    
    SIZE_3D = 32
    CENTER_3D = (SIZE_3D // 2, SIZE_3D // 2, SIZE_3D // 2)
    
    sim = WaveSimulation(
        dimensions=(SIZE_3D, SIZE_3D, SIZE_3D),
        dt=DT,
        wave_speed=1.0,
        damping=0.0
    )
    
    sim.inject_pulse(CENTER_3D, sigma=3.0, amplitude=PULSE_AMPLITUDE)
    
    e0 = sim.state.compute_total_energy()
    print(f"Initial energy (3D): {e0:.6f}")
    
    # Run 3D simulation
    for i in range(100):
        sim.step(method="leapfrog")
    
    e_final = sim.state.compute_total_energy()
    energy_drift = abs(e_final - e0) / e0
    print(f"Energy drift (3D): {energy_drift*100:.4f}%")
    print("PASS" if energy_drift < 0.05 else "FAIL")
    
    return sim


if __name__ == "__main__":
    print("Wave Equation Core - Phase 2.2 Validation")
    print("=" * 60)
    
    test_wave_propagation()
    test_damping()
    test_boundary_reflection()
    test_continuous_source()
    test_3d_wave()
    
    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
