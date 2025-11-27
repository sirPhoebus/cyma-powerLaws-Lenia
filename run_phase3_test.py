# Phase 3 Test - Pattern Crystallization System
# Tests: Energy Functionals, Attractor Detection, Pattern Library

import numpy as np
from src.energy_functional import (
    EnergyFunctional, GinzburgLandauEnergy, DoubleWellEnergy,
    SwiftHohenbergEnergy, EnergyMinimizer
)
from src.attractor_detection import (
    AttractorType, StateRecorder, FixedPointDetector,
    LimitCycleDetector, AttractorClassifier, BasinOfAttraction
)
from src.pattern_library import (
    PatternSignature, PatternEntry, PatternLibrary
)

GRID_SIZE = 64


def test_energy_functional():
    """Test 3.1: Energy Functional computation."""
    print("=" * 60)
    print("Test 3.1: Energy Functional")
    print("=" * 60)
    
    energy = EnergyFunctional(wave_speed=1.0, density=1.0)
    
    # Create test field
    x = np.linspace(0, 2*np.pi, GRID_SIZE)
    y = np.linspace(0, 2*np.pi, GRID_SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    field = np.sin(X) * np.sin(Y)
    velocity = np.cos(X) * np.cos(Y) * 0.5
    
    # Compute energies
    E_kinetic = energy.kinetic_energy(velocity)
    E_potential = energy.potential_energy(field)
    E_gradient = energy.gradient_energy(field)
    E_total = energy.total_energy(field, velocity)
    
    print(f"Kinetic energy: {E_kinetic:.4f}")
    print(f"Potential energy: {E_potential:.4f}")
    print(f"Gradient energy: {E_gradient:.4f}")
    print(f"Total energy: {E_total:.4f}")
    
    # Energy density
    density = energy.energy_density(field, velocity)
    print(f"Energy density range: [{np.min(density):.4f}, {np.max(density):.4f}]")
    
    passed = E_total > 0 and E_total == E_kinetic + E_potential + E_gradient
    print("PASS" if passed else "FAIL")


def test_ginzburg_landau():
    """Test Ginzburg-Landau energy."""
    print("\n" + "=" * 60)
    print("Test: Ginzburg-Landau Energy")
    print("=" * 60)
    
    gl = GinzburgLandauEnergy(a=1.0, b=1.0, kappa=1.0)
    
    # Equilibrium should be at sqrt(a/b) = 1.0
    print(f"Theoretical equilibrium: {gl.equilibrium_amplitude():.4f}")
    
    # Create field at equilibrium
    field_eq = np.ones((GRID_SIZE, GRID_SIZE)) * gl.equilibrium_amplitude()
    E_eq = gl.total_energy(field_eq)
    
    # Create field away from equilibrium
    field_high = np.ones((GRID_SIZE, GRID_SIZE)) * 2.0
    E_high = gl.total_energy(field_high)
    
    print(f"Energy at equilibrium: {E_eq:.4f}")
    print(f"Energy at u=2.0: {E_high:.4f}")
    
    # Equilibrium should have lower energy
    passed = E_eq < E_high
    print("PASS" if passed else "FAIL")


def test_energy_minimizer():
    """Test energy minimization."""
    print("\n" + "=" * 60)
    print("Test: Energy Minimizer")
    print("=" * 60)
    
    energy = EnergyFunctional(wave_speed=0.5)
    minimizer = EnergyMinimizer(energy, dt=0.01)
    
    # Start with random field
    np.random.seed(42)
    initial = np.random.randn(GRID_SIZE, GRID_SIZE) * 0.1
    
    E_initial = energy.total_energy(initial)
    print(f"Initial energy: {E_initial:.4f}")
    
    # Minimize
    final, converged = minimizer.minimize(initial, max_steps=500, tolerance=1e-4)
    
    E_final = energy.total_energy(final)
    print(f"Final energy: {E_final:.4f}")
    print(f"Converged: {converged}")
    print(f"Energy reduction: {(E_initial - E_final)/E_initial*100:.1f}%")
    
    passed = E_final < E_initial
    print("PASS" if passed else "FAIL")


def test_fixed_point_detector():
    """Test 3.2: Fixed point detection."""
    print("\n" + "=" * 60)
    print("Test 3.2: Fixed Point Detector")
    print("=" * 60)
    
    detector = FixedPointDetector(tolerance=1e-3)
    
    # Simulate converging to fixed point
    state = np.random.randn(GRID_SIZE, GRID_SIZE)
    decay_rate = 0.9
    
    is_fixed = False
    for i in range(100):
        prev_state = state.copy()
        state = state * decay_rate  # Exponential decay to zero
        
        is_fixed = detector.update(state, prev_state, dt=1.0)
        if is_fixed:
            print(f"Fixed point detected at step {i}")
            break
    
    vel_trace = detector.get_velocity_trace()
    print(f"Final velocity: {vel_trace[-1]:.6f}")
    
    passed = is_fixed and vel_trace[-1] < 1e-3
    print("PASS" if passed else "FAIL")


def test_limit_cycle_detector():
    """Test limit cycle detection."""
    print("\n" + "=" * 60)
    print("Test: Limit Cycle Detector")
    print("=" * 60)
    
    detector = LimitCycleDetector(tolerance=0.1, min_period=5)
    recorder = StateRecorder()
    
    # Generate periodic signal
    omega = 0.2
    for t in range(200):
        state = np.sin(omega * t) * np.ones((GRID_SIZE, GRID_SIZE))
        recorder.record(state, t)
    
    is_cycle, period = detector.detect(recorder)
    
    expected_period = 2 * np.pi / omega
    print(f"Detected cycle: {is_cycle}")
    print(f"Detected period: {period}")
    print(f"Expected period: {expected_period:.1f}")
    
    passed = is_cycle and period is not None
    print("PASS" if passed else "FAIL")


def test_attractor_classifier():
    """Test unified attractor classification."""
    print("\n" + "=" * 60)
    print("Test: Attractor Classifier")
    print("=" * 60)
    
    classifier = AttractorClassifier()
    
    # Test fixed point
    recorder = StateRecorder()
    state = np.ones((GRID_SIZE, GRID_SIZE))
    for t in range(50):
        recorder.record(state + np.random.randn(*state.shape) * 1e-5, t)
    
    attr_type, props = classifier.classify(recorder, dt=1.0)
    print(f"Fixed field classified as: {attr_type}")
    
    # Test oscillating (limit cycle)
    recorder2 = StateRecorder()
    for t in range(200):
        state = np.sin(0.1 * t) * np.ones((GRID_SIZE, GRID_SIZE))
        recorder2.record(state, t)
    
    attr_type2, props2 = classifier.classify(recorder2, dt=1.0)
    print(f"Oscillating field classified as: {attr_type2}")
    
    passed = attr_type == AttractorType.FIXED_POINT
    print("PASS" if passed else "FAIL")


def test_basin_of_attraction():
    """Test basin of attraction measurement."""
    print("\n" + "=" * 60)
    print("Test: Basin of Attraction")
    print("=" * 60)
    
    # Reference pattern
    reference = np.zeros((GRID_SIZE, GRID_SIZE))
    basin = BasinOfAttraction(reference, distance_threshold=0.5)
    
    # Test point inside basin
    near = np.random.randn(GRID_SIZE, GRID_SIZE) * 0.1
    dist_near = basin.distance_to_attractor(near)
    in_basin = basin.is_in_basin(near)
    
    print(f"Near point distance: {dist_near:.4f}")
    print(f"Near point in basin: {in_basin}")
    
    # Test point outside basin
    far = np.random.randn(GRID_SIZE, GRID_SIZE) * 10.0
    dist_far = basin.distance_to_attractor(far)
    
    print(f"Far point distance: {dist_far:.4f}")
    print(f"Far point in basin: {basin.is_in_basin(far)}")
    
    passed = in_basin and not basin.is_in_basin(far)
    print("PASS" if passed else "FAIL")


def test_pattern_signature():
    """Test 3.3: Pattern signature computation."""
    print("\n" + "=" * 60)
    print("Test 3.3: Pattern Signature")
    print("=" * 60)
    
    # Create pattern with known properties
    x = np.linspace(0, 2*np.pi, GRID_SIZE)
    y = np.linspace(0, 2*np.pi, GRID_SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    pattern = np.sin(2*X) * np.sin(2*Y)  # Mode (2,2)
    
    sig = PatternSignature(pattern)
    
    print(f"Shape: {sig.shape}")
    print(f"Symmetries: {sig.symmetry_type}")
    print(f"Topology: {sig.topological_features}")
    print(f"Top frequencies: {len(sig.frequency_signature)}")
    print(f"Hash: {sig.hash}")
    
    # Create similar pattern
    pattern2 = np.sin(2*X + 0.1) * np.sin(2*Y + 0.1)
    sig2 = PatternSignature(pattern2)
    
    similarity = sig.similarity(sig2)
    print(f"Similarity to shifted pattern: {similarity:.4f}")
    
    # Mode (2,2) has C2/C4 symmetry, not radial; shifted pattern has lower similarity
    passed = 'radial' not in sig.symmetry_type and similarity > 0.3
    print("PASS" if passed else "FAIL")


def test_pattern_library():
    """Test pattern library operations."""
    print("\n" + "=" * 60)
    print("Test: Pattern Library")
    print("=" * 60)
    
    library = PatternLibrary("test_library")
    
    # Add some patterns
    x = np.linspace(0, 2*np.pi, GRID_SIZE)
    y = np.linspace(0, 2*np.pi, GRID_SIZE)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    patterns = [
        ("mode_11", np.sin(X) * np.sin(Y)),
        ("mode_22", np.sin(2*X) * np.sin(2*Y)),
        ("mode_12", np.sin(X) * np.sin(2*Y)),
        ("radial", np.exp(-((X-np.pi)**2 + (Y-np.pi)**2)))
    ]
    
    for name, pattern in patterns:
        entry = library.add(pattern, name=name)
        if entry:
            print(f"Added: {name}")
    
    print(f"Library size: {len(library)}")
    
    # Search for similar
    query = np.sin(X) * np.sin(Y) + np.random.randn(GRID_SIZE, GRID_SIZE) * 0.1
    similar = library.find_similar(query, threshold=0.5)
    
    print(f"Found {len(similar)} similar patterns")
    if similar:
        print(f"Best match: {similar[0][0].name} (similarity: {similar[0][1]:.4f})")
    
    # Find by symmetry
    radial_patterns = library.find_by_symmetry("radial")
    print(f"Radial symmetric patterns: {len(radial_patterns)}")
    
    # Summary
    summary = library.summary()
    print(f"Summary: {summary}")
    
    passed = len(library) >= 3 and len(similar) > 0
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    print("Phase 3: Pattern Crystallization System")
    print("=" * 60)
    
    test_energy_functional()
    test_ginzburg_landau()
    test_energy_minimizer()
    test_fixed_point_detector()
    test_limit_cycle_detector()
    test_attractor_classifier()
    test_basin_of_attraction()
    test_pattern_signature()
    test_pattern_library()
    
    print("\n" + "=" * 60)
    print("All Phase 3 tests complete.")
    print("=" * 60)
