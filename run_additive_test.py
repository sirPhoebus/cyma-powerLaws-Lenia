# Additive Synthesis Test - Phase 2.6 Validation
# Tests harmonic series, ADSR envelopes, spatial modes, and frequency bundles

import numpy as np
from src.additive_synthesis import (
    ADSREnvelope, Harmonic, HarmonicSeries, SpatialMode,
    AdditiveSynthesizer, FrequencyBundle
)

# Test parameters
GRID_SIZE = 64


def test_adsr_envelope():
    """Test ADSR envelope generation."""
    print("=" * 60)
    print("Test 1: ADSR Envelope")
    print("=" * 60)
    
    env = ADSREnvelope(attack=0.1, decay=0.2, sustain=0.6, release=0.3)
    
    # Generate envelope curve
    duration = 1.0
    gate_off = 0.5
    t, curve = env.generate_curve(duration, sample_rate=100, gate_off_time=gate_off)
    
    # Check key points
    # At t=0, should be 0
    print(f"At t=0: {curve[0]:.4f} (expected 0)")
    
    # At t=attack, should be ~1
    attack_idx = int(0.1 * 100)
    print(f"At t=attack: {curve[attack_idx]:.4f} (expected ~1)")
    
    # At t=attack+decay, should be ~sustain
    sustain_idx = int(0.3 * 100)
    print(f"At t=attack+decay: {curve[sustain_idx]:.4f} (expected ~0.6)")
    
    # At end of release, should be ~0
    print(f"At t=end: {curve[-1]:.4f} (expected ~0)")
    
    passed = (curve[0] < 0.1 and 
              curve[attack_idx] > 0.8 and 
              0.5 < curve[sustain_idx] < 0.7 and
              curve[-1] < 0.1)
    print("PASS" if passed else "FAIL")


def test_harmonic_series():
    """Test harmonic series creation."""
    print("\n" + "=" * 60)
    print("Test 2: Harmonic Series")
    print("=" * 60)
    
    # Test sawtooth
    saw = HarmonicSeries.sawtooth(440, n_harmonics=8)
    freqs = saw.get_frequencies()
    print(f"Sawtooth harmonics: {len(saw.harmonics)}")
    print(f"Frequencies: {freqs[:4]}...")
    
    # Test square (odd harmonics only)
    square = HarmonicSeries.square(440, n_harmonics=8)
    print(f"Square harmonics: {len(square.harmonics)}")
    
    # Test triangle
    tri = HarmonicSeries.triangle(440, n_harmonics=8)
    print(f"Triangle harmonics: {len(tri.harmonics)}")
    
    # Check spectrum
    spectrum = saw.get_spectrum()
    print(f"First 3 spectrum entries: {spectrum[:3]}")
    
    # Verify 1/n amplitude decay for sawtooth
    amp_ratios = [s[1] for s in spectrum[:4]]
    expected_ratios = [1.0, 0.5, 0.333, 0.25]
    errors = [abs(a - e) for a, e in zip(amp_ratios, expected_ratios)]
    
    print(f"Amplitude errors: {[f'{e:.3f}' for e in errors]}")
    passed = all(e < 0.01 for e in errors)
    print("PASS" if passed else "FAIL")


def test_spatial_modes():
    """Test spatial mode computation."""
    print("\n" + "=" * 60)
    print("Test 3: Spatial Modes")
    print("=" * 60)
    
    modes = SpatialMode((GRID_SIZE, GRID_SIZE))
    
    # Test Cartesian mode (2,3)
    cart_mode = modes.compute_cartesian_mode((2, 3))
    print(f"Cartesian mode (2,3) shape: {cart_mode.shape}")
    print(f"Cartesian mode range: [{np.min(cart_mode):.4f}, {np.max(cart_mode):.4f}]")
    
    # Count zero crossings to verify mode number
    center_row = cart_mode[GRID_SIZE // 2, :]
    zero_crossings_x = np.sum(np.abs(np.diff(np.sign(center_row))) > 0)
    center_col = cart_mode[:, GRID_SIZE // 2]
    zero_crossings_y = np.sum(np.abs(np.diff(np.sign(center_col))) > 0)
    
    # For sin(n*pi*x/L), there are n-1 interior zero crossings
    print(f"Zero crossings in x: {zero_crossings_x} (expected ~3 for mode 3)")
    print(f"Zero crossings in y: {zero_crossings_y} (expected ~2 for mode 2)")
    
    # Test radial mode
    radial_mode = modes.compute_radial_mode(n=3, m=2)
    print(f"Radial mode (3,2) shape: {radial_mode.shape}")
    
    # Test angular mode
    angular_mode = modes.compute_angular_mode(m=4)
    print(f"Angular mode m=4 shape: {angular_mode.shape}")
    
    # sin(n*pi*x/L) has approximately n zero crossings for full grid
    passed = (cart_mode.shape == (GRID_SIZE, GRID_SIZE) and
              2 <= zero_crossings_x <= 4 and
              1 <= zero_crossings_y <= 3)
    print("PASS" if passed else "FAIL")


def test_additive_synthesizer():
    """Test additive synthesis."""
    print("\n" + "=" * 60)
    print("Test 4: Additive Synthesizer")
    print("=" * 60)
    
    synth = AdditiveSynthesizer((GRID_SIZE, GRID_SIZE), fundamental=0.1)
    
    # Add harmonics with envelopes
    for n in range(1, 5):
        env = ADSREnvelope(attack=0.1, decay=0.1, sustain=0.8, release=0.2)
        synth.harmonic_series.add_harmonic(n, amplitude=1.0/n, phase=0, envelope=env)
    
    # Assign spatial modes
    synth.assign_modes_cartesian()
    
    print(f"Harmonics: {len(synth.harmonic_series.harmonics)}")
    print(f"Mode assignments: {len(synth._mode_assignments)}")
    
    # Trigger and synthesize
    synth.trigger(0.0)
    
    # Generate at different times
    field_t0 = synth.synthesize(0.0)
    field_t1 = synth.synthesize(0.5)
    field_t2 = synth.synthesize(1.0)
    
    rms_t0 = np.sqrt(np.mean(field_t0**2))
    rms_t1 = np.sqrt(np.mean(field_t1**2))
    rms_t2 = np.sqrt(np.mean(field_t2**2))
    
    print(f"RMS at t=0.0: {rms_t0:.4f}")
    print(f"RMS at t=0.5: {rms_t1:.4f}")
    print(f"RMS at t=1.0: {rms_t2:.4f}")
    
    # RMS should vary with envelope and oscillation
    passed = rms_t1 > 0.01  # Should have significant amplitude
    print("PASS" if passed else "FAIL")
    
    return synth


def test_synthesis_sequence():
    """Test time sequence generation."""
    print("\n" + "=" * 60)
    print("Test 5: Synthesis Sequence")
    print("=" * 60)
    
    synth = AdditiveSynthesizer((GRID_SIZE, GRID_SIZE), fundamental=0.5)
    
    # Simple harmonic with envelope
    env = ADSREnvelope(attack=0.2, decay=0.1, sustain=0.7, release=0.3)
    synth.harmonic_series.add_harmonic(1, 1.0, 0, env)
    synth.assign_mode(0, {'type': 'cartesian', 'n': (2, 2)})
    
    # Generate sequence
    frames = []
    times = []
    for t, field in synth.synthesize_sequence(duration=1.0, dt=0.05, gate_off_time=0.6):
        frames.append(np.sqrt(np.mean(field**2)))
        times.append(t)
    
    frames = np.array(frames)
    times = np.array(times)
    
    print(f"Generated {len(frames)} frames")
    print(f"RMS range: [{np.min(frames):.4f}, {np.max(frames):.4f}]")
    
    # Check envelope shape
    peak_idx = np.argmax(frames)
    peak_time = times[peak_idx]
    print(f"Peak at t={peak_time:.2f}")
    
    # Peak should be around attack time
    passed = len(frames) == 20 and peak_time < 0.5
    print("PASS" if passed else "FAIL")


def test_frequency_bundle():
    """Test frequency bundle creation and serialization."""
    print("\n" + "=" * 60)
    print("Test 6: Frequency Bundle")
    print("=" * 60)
    
    # Create organ pipe bundle
    bundle = FrequencyBundle.create_organ_pipe(fundamental=0.1, n_harmonics=4)
    print(f"Organ bundle harmonics: {len(bundle.harmonics)}")
    
    # Create synthesizer from bundle
    synth = bundle.create_synthesizer((GRID_SIZE, GRID_SIZE))
    synth.trigger(0.0)
    
    field = synth.synthesize(0.5)
    rms = np.sqrt(np.mean(field**2))
    print(f"Synthesized RMS: {rms:.4f}")
    
    # Test serialization
    data = bundle.to_dict()
    print(f"Serialized keys: {list(data.keys())}")
    
    # Deserialize
    bundle2 = FrequencyBundle.from_dict(data)
    print(f"Deserialized harmonics: {len(bundle2.harmonics)}")
    
    passed = len(bundle.harmonics) == len(bundle2.harmonics) and rms > 0.01
    print("PASS" if passed else "FAIL")


def test_bell_timbre():
    """Test inharmonic bell bundle."""
    print("\n" + "=" * 60)
    print("Test 7: Bell Timbre (Inharmonic)")
    print("=" * 60)
    
    bundle = FrequencyBundle.create_bell(fundamental=0.2)
    print(f"Bell harmonics: {len(bundle.harmonics)}")
    
    # Check for inharmonic ratios
    ratios = [h[0] for h in bundle.harmonics]
    print(f"Frequency ratios: {ratios}")
    
    # Bell should have non-integer ratios
    has_inharmonic = any(r != int(r) for r in ratios)
    print(f"Has inharmonic partials: {has_inharmonic}")
    
    # Create and run
    synth = bundle.create_synthesizer((GRID_SIZE, GRID_SIZE))
    synth.trigger(0.0)
    
    # Bell should decay (envelopes have sustain=0)
    field_start = synth.synthesize(0.1)
    field_end = synth.synthesize(3.0)
    
    rms_start = np.sqrt(np.mean(field_start**2))
    rms_end = np.sqrt(np.mean(field_end**2))
    
    print(f"RMS at t=0.1: {rms_start:.4f}")
    print(f"RMS at t=3.0: {rms_end:.4f}")
    
    passed = has_inharmonic and rms_end < rms_start
    print("PASS" if passed else "FAIL")


def test_cymatics_mode_bundle():
    """Test Chladni-like mode bundle."""
    print("\n" + "=" * 60)
    print("Test 8: Cymatics Mode Bundle")
    print("=" * 60)
    
    # Create mode (3, 2)
    bundle = FrequencyBundle.create_cymatics_mode(fundamental=0.1, mode_numbers=(3, 2))
    
    synth = bundle.create_synthesizer((GRID_SIZE, GRID_SIZE))
    synth.trigger(0.0)
    
    # Let attack phase complete
    field = synth.synthesize(1.0)
    
    print(f"Field shape: {field.shape}")
    print(f"Field range: [{np.min(field):.4f}, {np.max(field):.4f}]")
    
    # Check mode structure by counting zero crossings
    center_row = field[GRID_SIZE // 2, :]
    zero_crossings = np.sum(np.abs(np.diff(np.sign(center_row))) > 0)
    
    print(f"Zero crossings in center row: {zero_crossings} (expected ~4 for mode 2)")
    
    # Should have clear spatial structure
    variance = np.var(field)
    print(f"Field variance: {variance:.6f}")
    
    passed = zero_crossings >= 2 and variance > 0.001
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    print("Additive Synthesis Engine - Phase 2.6 Validation")
    print("=" * 60)
    
    test_adsr_envelope()
    test_harmonic_series()
    test_spatial_modes()
    test_additive_synthesizer()
    test_synthesis_sequence()
    test_frequency_bundle()
    test_bell_timbre()
    test_cymatics_mode_bundle()
    
    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)
