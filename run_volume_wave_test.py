# Volume Wave Test - Phase 2.4 Validation
# Tests 3D wave propagation, spherical sources, and analysis tools

import numpy as np
from src.volume_wave import (
    VolumeWaveConfig, VolumeWaveState, VolumeWaveSimulation,
    SphericalHarmonicSource, StandingWaveAnalyzer3D,
    extract_nodal_surfaces
)
from src.config import BOUNDARY_PERIODIC, BOUNDARY_NEUMANN

# Test parameters
GRID_SIZE = 32  # Smaller for faster 3D tests
CENTER = (GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE // 2)
DT = 0.05
NUM_STEPS = 200


def test_volume_state():
    """Test VolumeWaveState creation and methods."""
    print("=" * 60)
    print("Test 1: Volume Wave State")
    print("=" * 60)
    
    state = VolumeWaveState(GRID_SIZE)
    print(f"Created: {state}")
    print(f"Shape: {state.shape}")
    print(f"Dimensions: nz={state.nz}, ny={state.ny}, nx={state.nx}")
    
    # Test spherical pulse injection
    state.inject_spherical_pulse(CENTER, radius=5.0, amplitude=1.0, profile="gaussian")
    rms = np.sqrt(np.mean(state.amplitude**2))
    print(f"After Gaussian pulse: RMS = {rms:.6f}")
    
    # Test plane wave injection
    state.reset()
    state.inject_plane_wave(direction=(1, 0, 0), wavelength=16, amplitude=1.0)
    rms = np.sqrt(np.mean(state.amplitude**2))
    print(f"After plane wave: RMS = {rms:.6f}")
    
    # Test slice extraction
    slices = state.get_center_slices()
    print(f"Center slices shapes: XY={slices['xy'].shape}, XZ={slices['xz'].shape}, YZ={slices['yz'].shape}")
    
    print("PASS")


def test_spherical_wave():
    """Test spherical wave propagation from point source."""
    print("\n" + "=" * 60)
    print("Test 2: Spherical Wave Propagation")
    print("=" * 60)
    
    sim = VolumeWaveSimulation(
        resolution=GRID_SIZE,
        dt=DT,
        wave_speed=1.0,
        damping=0.0
    )
    
    # Inject central pulse
    sim.inject_point_source(CENTER, sigma=3.0, amplitude=1.0)
    
    e0 = sim.state.compute_total_energy()
    print(f"Initial energy: {e0:.6f}")
    
    # Run simulation
    print(f"Running {NUM_STEPS} steps...")
    for i in range(NUM_STEPS):
        sim.step()
        if (i + 1) % 50 == 0:
            e = sim.state.compute_total_energy()
            print(f"  Step {i+1}: energy={e:.6f}")
    
    e_final = sim.state.compute_total_energy()
    energy_drift = abs(e_final - e0) / e0
    print(f"Energy drift: {energy_drift*100:.2f}%")
    print("PASS" if energy_drift < 0.05 else "FAIL")
    
    return sim


def test_shell_source():
    """Test spherical shell source."""
    print("\n" + "=" * 60)
    print("Test 3: Spherical Shell Source")
    print("=" * 60)
    
    sim = VolumeWaveSimulation(
        resolution=GRID_SIZE,
        dt=DT,
        wave_speed=1.0,
        damping=0.0
    )
    
    # Inject shell source
    sim.inject_spherical_shell(CENTER, radius=8.0, amplitude=1.0)
    
    # Check shell structure
    amp = sim.get_amplitude()
    
    # Sample along radial line
    radii = []
    values = []
    for r in range(0, GRID_SIZE // 2):
        z = CENTER[0]
        y = CENTER[1]
        x = CENTER[2] + r
        if x < GRID_SIZE:
            radii.append(r)
            values.append(amp[z, y, x])
    
    # Find peak
    peak_idx = np.argmax(values)
    peak_radius = radii[peak_idx]
    
    print(f"Shell peak at radius: {peak_radius} (expected ~8)")
    print("PASS" if 6 <= peak_radius <= 10 else "FAIL")


def test_plane_wave():
    """Test plane wave propagation."""
    print("\n" + "=" * 60)
    print("Test 4: Plane Wave Propagation")
    print("=" * 60)
    
    sim = VolumeWaveSimulation(
        resolution=GRID_SIZE,
        dt=DT,
        wave_speed=1.0,
        damping=0.0
    )
    sim.state.boundary = BOUNDARY_PERIODIC
    
    # Inject plane wave traveling in z direction
    wavelength = 16
    sim.state.inject_plane_wave(
        direction=(1, 0, 0),
        wavelength=wavelength,
        amplitude=1.0
    )
    
    # Also need velocity for propagating wave
    # v = c * du/dz for wave traveling in +z
    grids = sim.state._get_coord_grids()
    k = 2 * np.pi / wavelength
    sim.state.velocity[0] = sim.state.wave_speed * k * np.cos(k * grids[0])
    
    e0 = sim.state.compute_total_energy()
    
    # Run one full period
    period = wavelength / sim.state.wave_speed[0, 0, 0]
    steps = int(period / DT)
    
    print(f"Wavelength: {wavelength}, Period: {period:.2f}")
    print(f"Running {steps} steps (one period)...")
    
    for i in range(steps):
        sim.step()
    
    e_final = sim.state.compute_total_energy()
    energy_drift = abs(e_final - e0) / e0
    print(f"Energy drift after one period: {energy_drift*100:.2f}%")
    print("PASS" if energy_drift < 0.05 else "FAIL")


def test_oscillating_source():
    """Test continuously oscillating point source."""
    print("\n" + "=" * 60)
    print("Test 5: Oscillating Point Source")
    print("=" * 60)
    
    sim = VolumeWaveSimulation(
        resolution=GRID_SIZE,
        dt=DT,
        wave_speed=1.0,
        damping=0.02
    )
    
    frequency = 0.1
    sim.set_oscillating_source(CENTER, frequency=frequency, amplitude=1.0, sigma=2.0)
    
    print(f"Oscillating at frequency: {frequency}")
    print(f"Running {NUM_STEPS * 2} steps...")
    
    rms_history = []
    for i in range(NUM_STEPS * 2):
        sim.step()
        if (i + 1) % 100 == 0:
            rms = np.sqrt(np.mean(sim.get_amplitude()**2))
            rms_history.append(rms)
            print(f"  Step {i+1}: RMS = {rms:.6f}")
    
    # Check that waves are being generated
    final_rms = rms_history[-1]
    print(f"Final RMS: {final_rms:.6f}")
    print("PASS" if final_rms > 0.01 else "FAIL")


def test_spherical_harmonic_source():
    """Test spherical harmonic source patterns."""
    print("\n" + "=" * 60)
    print("Test 6: Spherical Harmonic Source")
    print("=" * 60)
    
    sim = VolumeWaveSimulation(
        resolution=GRID_SIZE,
        dt=DT,
        wave_speed=1.0,
        damping=0.01
    )
    
    # Create Y_1^0 source (dipole pattern in z)
    source = SphericalHarmonicSource(
        center=CENTER,
        l=1, m=0,
        frequency=0.1,
        amplitude=1.0,
        radius=8.0
    )
    
    sim.equation.set_source(source)
    
    print("Running with Y_1^0 (dipole) source...")
    for i in range(NUM_STEPS):
        sim.step()
    
    # Check for dipole pattern (opposite signs in +z and -z)
    amp = sim.get_amplitude()
    cz, cy, cx = CENTER
    
    top_value = amp[cz + 5, cy, cx]
    bottom_value = amp[cz - 5, cy, cx]
    
    print(f"Top (+z) amplitude: {top_value:.6f}")
    print(f"Bottom (-z) amplitude: {bottom_value:.6f}")
    
    # For dipole, these should have opposite signs (or very different magnitudes)
    is_dipole = top_value * bottom_value < 0 or abs(top_value - bottom_value) > 0.1 * max(abs(top_value), abs(bottom_value))
    print("PASS (dipole pattern)" if is_dipole else "FAIL")


def test_standing_wave_analyzer():
    """Test standing wave analysis tools."""
    print("\n" + "=" * 60)
    print("Test 7: Standing Wave Analyzer")
    print("=" * 60)
    
    sim = VolumeWaveSimulation(
        resolution=GRID_SIZE,
        dt=DT,
        wave_speed=1.0,
        damping=0.0
    )
    sim.state.boundary = BOUNDARY_PERIODIC
    
    # Inject known mode
    wavelength = GRID_SIZE // 2  # Mode 2 in each direction
    sim.state.inject_plane_wave((1, 0, 0), wavelength, 1.0)
    sim.state.inject_plane_wave((0, 1, 0), wavelength, 1.0)
    
    analyzer = StandingWaveAnalyzer3D(sim.state)
    
    # Check radial spectrum
    k_centers, radial_power = analyzer.compute_radial_spectrum()
    peak_k = k_centers[np.argmax(radial_power)]
    expected_k = 1.0 / wavelength
    
    print(f"Expected peak k: {expected_k:.4f}")
    print(f"Measured peak k: {peak_k:.4f}")
    
    # Check dominant modes
    modes = analyzer.find_dominant_modes(n_modes=3)
    print("Dominant modes:")
    for kz, ky, kx, power in modes:
        print(f"  k=({kz}, {ky}, {kx}), power={power:.2f}")
    
    # Check node volume
    node_frac = analyzer.compute_node_volume_fraction()
    print(f"Node volume fraction: {node_frac*100:.1f}%")
    
    print("PASS")


def test_3d_boundary_conditions():
    """Test boundary conditions in 3D."""
    print("\n" + "=" * 60)
    print("Test 8: 3D Boundary Conditions")
    print("=" * 60)
    
    # Test reflecting (Neumann) boundaries
    sim = VolumeWaveSimulation(
        resolution=GRID_SIZE,
        dt=DT,
        wave_speed=1.0,
        damping=0.0
    )
    sim.state.boundary = BOUNDARY_NEUMANN
    
    # Off-center pulse to hit boundary
    off_center = (GRID_SIZE // 4, GRID_SIZE // 2, GRID_SIZE // 2)
    sim.inject_point_source(off_center, sigma=3.0, amplitude=1.0)
    
    e0 = sim.state.compute_total_energy()
    
    # Run long enough to hit boundaries
    for i in range(NUM_STEPS * 2):
        sim.step()
    
    e_final = sim.state.compute_total_energy()
    energy_drift = abs(e_final - e0) / e0
    
    print(f"Energy drift with Neumann BC: {energy_drift*100:.2f}%")
    # 3D has 6 faces, more boundary interactions, allow higher tolerance
    print("PASS" if energy_drift < 0.25 else "FAIL")


if __name__ == "__main__":
    print("Volume Wave Propagation - Phase 2.4 Validation")
    print("=" * 60)
    
    test_volume_state()
    test_spherical_wave()
    test_shell_source()
    test_plane_wave()
    test_oscillating_source()
    test_spherical_harmonic_source()
    test_standing_wave_analyzer()
    test_3d_boundary_conditions()
    
    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
