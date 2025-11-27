# Interference Mechanics Test - Phase 2.5 Validation
# Tests phase coherence, interference patterns, and standing wave detection

import numpy as np
from src.interference import (
    InterferenceAnalyzer, StandingWaveDetector, TwoSourceInterference,
    compute_beat_frequency, compute_group_velocity
)
from src.wave_equation import WaveSimulation

# Test parameters
GRID_SIZE = 128
DT = 0.05


def test_phase_extraction():
    """Test phase extraction from wave fields."""
    print("=" * 60)
    print("Test 1: Phase Extraction")
    print("=" * 60)
    
    analyzer = InterferenceAnalyzer((GRID_SIZE, GRID_SIZE))
    
    # Create simple wave with known phase
    x = np.arange(GRID_SIZE)
    y = np.arange(GRID_SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    wavelength = 32
    k = 2 * np.pi / wavelength
    wave = np.sin(k * X)  # Wave traveling in x
    
    phase = analyzer.compute_phase(wave)
    
    # Check that phase wraps correctly over one wavelength
    # Look at one row and count zero crossings of phase
    row = phase[GRID_SIZE // 2, :]
    
    # Count sign changes in cos(phase) to detect wrapping
    cos_phase = np.cos(row)
    sign_changes = np.sum(np.abs(np.diff(np.sign(cos_phase))) > 0)
    expected_changes = 2 * (GRID_SIZE // wavelength)  # ~8 for 128/32 = 4 periods
    
    print(f"Wavelength: {wavelength} pixels")
    print(f"Phase sign changes: {sign_changes} (expected ~{expected_changes})")
    print(f"Phase range: [{np.min(phase):.2f}, {np.max(phase):.2f}]")
    
    # Phase should have some structure (not constant)
    phase_std = np.std(phase)
    print(f"Phase std: {phase_std:.4f}")
    
    passed = phase_std > 0.1 and abs(sign_changes - expected_changes) < expected_changes
    print("PASS" if passed else "FAIL")


def test_coherence():
    """Test phase coherence computation."""
    print("\n" + "=" * 60)
    print("Test 2: Phase Coherence")
    print("=" * 60)
    
    analyzer = InterferenceAnalyzer((GRID_SIZE, GRID_SIZE))
    
    x = np.arange(GRID_SIZE)
    y = np.arange(GRID_SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    k = 2 * np.pi / 32
    
    # Two identical waves = perfect coherence
    wave1 = np.sin(k * X)
    wave2 = np.sin(k * X)
    coherence_same = analyzer.compute_coherence(wave1, wave2)
    print(f"Identical waves coherence: {coherence_same:.4f} (expected ~1.0)")
    
    # Two waves with pi phase difference
    wave3 = np.sin(k * X + np.pi)
    coherence_opposite = analyzer.compute_coherence(wave1, wave3)
    print(f"Opposite phase coherence: {coherence_opposite:.4f} (expected ~1.0)")
    
    # Wave and noise = low coherence
    noise = np.random.randn(GRID_SIZE, GRID_SIZE)
    coherence_noise = analyzer.compute_coherence(wave1, noise)
    print(f"Wave vs noise coherence: {coherence_noise:.4f} (expected ~0)")
    
    # Check expected behaviors
    passed = coherence_same > 0.8 and coherence_noise < 0.3
    print("PASS" if passed else "FAIL")


def test_interference_detection():
    """Test constructive/destructive interference detection."""
    print("\n" + "=" * 60)
    print("Test 3: Interference Type Detection")
    print("=" * 60)
    
    analyzer = InterferenceAnalyzer((GRID_SIZE, GRID_SIZE))
    
    x = np.arange(GRID_SIZE)
    y = np.arange(GRID_SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    k = 2 * np.pi / 32
    
    # Two in-phase waves
    wave1 = np.sin(k * X)
    wave2 = np.sin(k * X)
    
    result = analyzer.detect_interference_type(wave1, wave2)
    
    constructive_frac = np.mean(result['constructive'])
    destructive_frac = np.mean(result['destructive'])
    
    print(f"In-phase waves:")
    print(f"  Constructive fraction: {constructive_frac*100:.1f}%")
    print(f"  Destructive fraction: {destructive_frac*100:.1f}%")
    
    # Two out-of-phase waves
    wave3 = np.sin(k * X + np.pi)
    result2 = analyzer.detect_interference_type(wave1, wave3)
    
    constructive_frac2 = np.mean(result2['constructive'])
    destructive_frac2 = np.mean(result2['destructive'])
    
    print(f"Out-of-phase waves:")
    print(f"  Constructive fraction: {constructive_frac2*100:.1f}%")
    print(f"  Destructive fraction: {destructive_frac2*100:.1f}%")
    
    passed = constructive_frac > 0.8 and destructive_frac2 > 0.8
    print("PASS" if passed else "FAIL")


def test_two_source_interference():
    """Test classic two-source interference pattern."""
    print("\n" + "=" * 60)
    print("Test 4: Two-Source Interference")
    print("=" * 60)
    
    # Two sources separated horizontally
    source1 = (GRID_SIZE // 2, GRID_SIZE // 3)
    source2 = (GRID_SIZE // 2, 2 * GRID_SIZE // 3)
    
    wavelength = 16
    frequency = 1.0 / wavelength
    
    interference = TwoSourceInterference(
        dimensions=(GRID_SIZE, GRID_SIZE),
        source1_pos=source1,
        source2_pos=source2,
        frequency=frequency
    )
    
    # Compute intensity pattern
    intensity = interference.compute_intensity_pattern()
    
    print(f"Source separation: {source2[1] - source1[1]} pixels")
    print(f"Wavelength: {wavelength} pixels")
    print(f"Intensity range: [{np.min(intensity):.4f}, {np.max(intensity):.4f}]")
    
    # Find fringes
    constructive = interference.find_constructive_fringes()
    destructive = interference.find_destructive_fringes()
    
    print(f"Constructive fringe fraction: {np.mean(constructive)*100:.1f}%")
    print(f"Destructive fringe fraction: {np.mean(destructive)*100:.1f}%")
    
    # Fringe spacing at observation distance
    obs_dist = GRID_SIZE // 4
    fringe_spacing = interference.compute_fringe_spacing(obs_dist)
    print(f"Theoretical fringe spacing at distance {obs_dist}: {fringe_spacing:.2f} pixels")
    
    # Check that we have both constructive and destructive regions
    passed = np.mean(constructive) > 0.1 and np.mean(destructive) > 0.1
    print("PASS" if passed else "FAIL")
    
    return interference


def test_standing_wave_detector():
    """Test standing wave node/antinode detection."""
    print("\n" + "=" * 60)
    print("Test 5: Standing Wave Detector")
    print("=" * 60)
    
    detector = StandingWaveDetector((GRID_SIZE, GRID_SIZE))
    
    x = np.arange(GRID_SIZE)
    y = np.arange(GRID_SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    # Simulate standing wave with known mode (2,3)
    kx = 2 * np.pi * 2 / GRID_SIZE
    ky = 2 * np.pi * 3 / GRID_SIZE
    
    # Record multiple frames at different phases
    for t in range(50):
        phase = 2 * np.pi * t / 20
        amplitude = np.sin(kx * X) * np.sin(ky * Y) * np.cos(phase)
        detector.record_frame(amplitude)
    
    # Detect nodes and antinodes
    nodes = detector.detect_nodes()
    antinodes = detector.detect_antinodes()
    
    print(f"Node fraction: {np.mean(nodes)*100:.1f}%")
    print(f"Antinode fraction: {np.mean(antinodes)*100:.1f}%")
    
    # Estimate oscillation amplitude
    osc_amp = detector.compute_oscillation_amplitude()
    print(f"Max oscillation amplitude: {np.max(osc_amp):.4f}")
    
    # Estimate frequency
    freq = detector.estimate_standing_wave_frequency()
    print(f"Estimated frequency: {freq:.4f} (expected ~0.05)")
    
    # Estimate mode numbers
    modes = detector.compute_mode_number_estimate()
    print(f"Estimated mode numbers: {modes} (expected ~(2,3))")
    
    # Check that we detect structure
    passed = nodes is not None and np.mean(nodes) > 0.05
    print("PASS" if passed else "FAIL")


def test_superposition():
    """Test wave superposition."""
    print("\n" + "=" * 60)
    print("Test 6: Wave Superposition")
    print("=" * 60)
    
    analyzer = InterferenceAnalyzer((GRID_SIZE, GRID_SIZE))
    
    x = np.arange(GRID_SIZE)
    y = np.arange(GRID_SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    k = 2 * np.pi / 32
    
    wave1 = np.sin(k * X)
    wave2 = np.sin(k * Y)
    wave3 = np.sin(k * (X + Y) / np.sqrt(2))
    
    # Equal superposition
    result = analyzer.superpose(wave1, wave2, wave3)
    
    # Check RMS is reasonable
    rms = np.sqrt(np.mean(result**2))
    print(f"Superposed RMS: {rms:.4f}")
    
    # Weighted superposition
    result_weighted = analyzer.superpose(wave1, wave2, wave3, weights=[1.0, 0.5, 0.25])
    rms_weighted = np.sqrt(np.mean(result_weighted**2))
    print(f"Weighted superposed RMS: {rms_weighted:.4f}")
    
    # Check that weighting makes a difference
    print("PASS" if abs(rms - rms_weighted) > 0.01 else "FAIL")


def test_beat_frequency():
    """Test beat frequency computation."""
    print("\n" + "=" * 60)
    print("Test 7: Beat Frequency")
    print("=" * 60)
    
    freq1 = 440  # A4
    freq2 = 442  # Slightly sharp
    
    beat = compute_beat_frequency(freq1, freq2)
    print(f"Frequencies: {freq1} Hz and {freq2} Hz")
    print(f"Beat frequency: {beat} Hz")
    
    expected = abs(freq1 - freq2)
    print("PASS" if beat == expected else "FAIL")


def test_interference_from_simulation():
    """Test interference using actual wave simulation."""
    print("\n" + "=" * 60)
    print("Test 8: Interference from Wave Simulation")
    print("=" * 60)
    
    # Create simulation with two oscillating sources
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.002  # Lower damping to let pattern build
    )
    
    # Two point sources
    source1 = (GRID_SIZE // 2, GRID_SIZE // 3)
    source2 = (GRID_SIZE // 2, 2 * GRID_SIZE // 3)
    frequency = 0.1  # Higher frequency for more cycles
    
    def two_source_fn(state, t):
        source = np.zeros(state.shape, dtype=np.float64)
        omega = 2 * np.pi * frequency
        
        # Source 1 - stronger amplitude
        y1, x1 = source1
        source[0, y1-2:y1+3, x1-2:x1+3] = 2.0 * np.sin(omega * t)
        
        # Source 2 (in phase)
        y2, x2 = source2
        source[0, y2-2:y2+3, x2-2:x2+3] = 2.0 * np.sin(omega * t)
        
        return source
    
    sim.equation.set_source(two_source_fn)
    
    # Run simulation longer to build pattern
    print("Running two-source simulation...")
    detector = StandingWaveDetector((GRID_SIZE, GRID_SIZE))
    
    for i in range(600):
        sim.step()
        if i >= 300:  # Record after transient
            detector.record_frame(sim.state.amplitude[0])
    
    # Analyze interference pattern
    osc_amp = detector.compute_oscillation_amplitude()
    
    print(f"Final RMS: {np.sqrt(np.mean(sim.state.amplitude**2)):.4f}")
    print(f"Max oscillation amplitude: {np.max(osc_amp):.4f}")
    
    # Check for interference structure via spatial variance
    # A good interference pattern has high variance (some regions oscillate more)
    osc_variance = np.var(osc_amp)
    osc_mean = np.mean(osc_amp)
    coefficient_of_variation = np.sqrt(osc_variance) / (osc_mean + 1e-10)
    
    print(f"Oscillation mean: {osc_mean:.4f}")
    print(f"Oscillation std: {np.sqrt(osc_variance):.4f}")
    print(f"Coefficient of variation: {coefficient_of_variation:.4f}")
    
    # High CV means clear spatial structure (interference pattern)
    has_structure = coefficient_of_variation > 0.3 and osc_mean > 0.1
    print("PASS" if has_structure else "FAIL")


if __name__ == "__main__":
    print("Interference Mechanics - Phase 2.5 Validation")
    print("=" * 60)
    
    test_phase_extraction()
    test_coherence()
    test_interference_detection()
    test_two_source_interference()
    test_standing_wave_detector()
    test_superposition()
    test_beat_frequency()
    test_interference_from_simulation()
    
    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
