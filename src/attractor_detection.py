# Attractor Detection - Phase 3.2
#
# Systems for identifying stable configurations:
#   - Fixed points (stationary patterns)
#   - Limit cycles (periodic oscillations)
#   - Strange attractors (chaotic dynamics)
#   - Basin of attraction measurement
#   - Bifurcation detection

import numpy as np
from scipy import signal
from src.config import DTYPE


class AttractorType:
    """Enumeration of attractor types."""
    FIXED_POINT = "fixed_point"
    LIMIT_CYCLE = "limit_cycle"
    STRANGE = "strange_attractor"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class StateRecorder:
    """
    Record system states for attractor analysis.
    """
    
    def __init__(self, max_history=1000):
        self.states = []
        self.times = []
        self.max_history = max_history
    
    def record(self, state, time):
        """Record a state snapshot."""
        self.states.append(state.copy())
        self.times.append(time)
        
        if len(self.states) > self.max_history:
            self.states.pop(0)
            self.times.pop(0)
    
    def clear(self):
        """Clear recorded states."""
        self.states.clear()
        self.times.clear()
    
    def get_trajectory(self):
        """Return recorded trajectory as array."""
        return np.array(self.states)
    
    def get_times(self):
        """Return recorded times."""
        return np.array(self.times)


class FixedPointDetector:
    """
    Detect fixed points (stationary patterns).
    
    A fixed point satisfies: df/dt = 0
    """
    
    def __init__(self, tolerance=1e-4, window_size=20):
        """
        Args:
            tolerance: Maximum velocity for fixed point detection
            window_size: Number of frames to average for stability check
        """
        self.tolerance = tolerance
        self.window_size = window_size
        self._velocity_history = []
    
    def update(self, state, prev_state, dt):
        """
        Update with new state, check for fixed point.
        
        Args:
            state: Current state
            prev_state: Previous state
            dt: Time step
            
        Returns:
            True if fixed point detected
        """
        velocity = (state - prev_state) / dt
        velocity_norm = np.sqrt(np.mean(velocity**2))
        
        self._velocity_history.append(velocity_norm)
        if len(self._velocity_history) > self.window_size:
            self._velocity_history.pop(0)
        
        if len(self._velocity_history) >= self.window_size:
            avg_velocity = np.mean(self._velocity_history)
            return avg_velocity < self.tolerance
        
        return False
    
    def get_velocity_trace(self):
        """Return velocity history."""
        return np.array(self._velocity_history)
    
    def clear(self):
        """Reset detector state."""
        self._velocity_history.clear()


class LimitCycleDetector:
    """
    Detect limit cycles (periodic oscillations).
    
    A limit cycle satisfies: state(t + T) = state(t) for some period T
    """
    
    def __init__(self, tolerance=0.1, min_period=10, max_period=500):
        """
        Args:
            tolerance: Maximum state difference for recurrence
            min_period: Minimum period to detect
            max_period: Maximum period to search
        """
        self.tolerance = tolerance
        self.min_period = min_period
        self.max_period = max_period
    
    def detect(self, recorder):
        """
        Detect limit cycle from recorded trajectory.
        
        Args:
            recorder: StateRecorder with trajectory
            
        Returns:
            (detected, period) or (False, None)
        """
        trajectory = recorder.get_trajectory()
        times = recorder.get_times()
        
        if len(trajectory) < self.min_period * 2:
            return False, None
        
        current_state = trajectory[-1]
        
        # Search for recurrence
        for i in range(len(trajectory) - self.min_period - 1, -1, -1):
            past_state = trajectory[i]
            
            # Compute distance
            diff = np.sqrt(np.mean((current_state - past_state)**2))
            
            if diff < self.tolerance:
                period_steps = len(trajectory) - 1 - i
                if period_steps >= self.min_period and period_steps <= self.max_period:
                    period_time = times[-1] - times[i]
                    return True, period_time
        
        return False, None
    
    def compute_period_spectrum(self, recorder, point_index=None):
        """
        Compute power spectrum to find dominant period.
        
        Args:
            recorder: StateRecorder
            point_index: Spatial point to analyze (None = use center)
            
        Returns:
            periods, power arrays
        """
        trajectory = recorder.get_trajectory()
        times = recorder.get_times()
        
        if len(trajectory) < 10:
            return None, None
        
        # Extract time series at a point
        if point_index is None:
            # Use center point
            center = tuple(d // 2 for d in trajectory[0].shape)
            point_index = center
        
        time_series = np.array([state[point_index] for state in trajectory])
        
        # FFT
        spectrum = np.abs(np.fft.rfft(time_series - np.mean(time_series)))
        freqs = np.fft.rfftfreq(len(time_series), times[1] - times[0])
        
        # Convert to periods
        with np.errstate(divide='ignore'):
            periods = 1.0 / (freqs + 1e-10)
        
        return periods[1:], spectrum[1:]  # Skip DC


class StrangeAttractorDetector:
    """
    Detect strange attractors (chaotic dynamics).
    
    Characteristics:
    - Sensitive dependence on initial conditions (positive Lyapunov exponent)
    - Fractal structure in phase space
    - Non-periodic but bounded dynamics
    """
    
    def __init__(self, perturbation_size=1e-6):
        """
        Args:
            perturbation_size: Size of perturbation for Lyapunov estimation
        """
        self.perturbation_size = perturbation_size
    
    def estimate_lyapunov_exponent(self, recorder):
        """
        Estimate largest Lyapunov exponent from trajectory.
        
        Lambda > 0 indicates chaos.
        """
        trajectory = recorder.get_trajectory()
        times = recorder.get_times()
        
        if len(trajectory) < 20:
            return None
        
        # Use finite-time Lyapunov method
        dt = times[1] - times[0]
        
        # Compute local divergence rates
        divergences = []
        for i in range(1, len(trajectory)):
            diff = trajectory[i] - trajectory[i-1]
            divergence = np.log(np.sqrt(np.mean(diff**2)) + 1e-10) / dt
            divergences.append(divergence)
        
        # Average divergence rate
        return np.mean(divergences)
    
    def compute_correlation_dimension(self, recorder, embedding_dim=3, 
                                       max_points=1000):
        """
        Estimate correlation dimension (fractal dimension estimate).
        
        Uses Grassberger-Procaccia algorithm.
        
        D2 ~ 2 for limit cycle
        D2 non-integer for strange attractor
        """
        trajectory = recorder.get_trajectory()
        
        if len(trajectory) < 100:
            return None
        
        # Flatten spatial dimensions
        n_points = min(len(trajectory), max_points)
        points = trajectory[-n_points:].reshape(n_points, -1)
        
        # Use only first few dimensions for speed
        if points.shape[1] > 100:
            indices = np.random.choice(points.shape[1], 100, replace=False)
            points = points[:, indices]
        
        # Compute pairwise distances
        n = len(points)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d > 0:
                    distances.append(d)
        
        if len(distances) < 100:
            return None
        
        distances = np.array(distances)
        
        # Correlation sum at different scales
        epsilons = np.logspace(np.log10(np.percentile(distances, 5)),
                              np.log10(np.percentile(distances, 95)), 20)
        
        C = []
        for eps in epsilons:
            c = np.sum(distances < eps) / len(distances)
            if c > 0:
                C.append(c)
            else:
                C.append(1e-10)
        
        C = np.array(C)
        
        # Fit slope of log(C) vs log(eps)
        log_eps = np.log(epsilons[:len(C)])
        log_C = np.log(C)
        
        valid = np.isfinite(log_C)
        if np.sum(valid) < 5:
            return None
        
        coeffs = np.polyfit(log_eps[valid], log_C[valid], 1)
        return coeffs[0]  # Slope is correlation dimension
    
    def is_chaotic(self, recorder):
        """
        Determine if dynamics appear chaotic.
        
        Returns True if Lyapunov > 0 and dimension is non-integer.
        """
        lyap = self.estimate_lyapunov_exponent(recorder)
        dim = self.compute_correlation_dimension(recorder)
        
        if lyap is None or dim is None:
            return False
        
        return lyap > 0 and dim % 1 > 0.1


class AttractorClassifier:
    """
    Unified attractor classification system.
    """
    
    def __init__(self, fixed_point_tol=1e-4, cycle_tol=0.1):
        """
        Args:
            fixed_point_tol: Tolerance for fixed point detection
            cycle_tol: Tolerance for limit cycle detection
        """
        self.fixed_point_detector = FixedPointDetector(fixed_point_tol)
        self.cycle_detector = LimitCycleDetector(cycle_tol)
        self.strange_detector = StrangeAttractorDetector()
    
    def classify(self, recorder, dt=1.0):
        """
        Classify the attractor type from recorded trajectory.
        
        Args:
            recorder: StateRecorder with trajectory
            dt: Time step
            
        Returns:
            AttractorType and additional info dict
        """
        trajectory = recorder.get_trajectory()
        
        if len(trajectory) < 20:
            return AttractorType.TRANSIENT, {}
        
        # Check for fixed point
        velocities = []
        for i in range(1, len(trajectory)):
            v = np.sqrt(np.mean((trajectory[i] - trajectory[i-1])**2)) / dt
            velocities.append(v)
        
        avg_velocity = np.mean(velocities[-min(20, len(velocities)):])
        
        if avg_velocity < self.fixed_point_detector.tolerance:
            return AttractorType.FIXED_POINT, {'velocity': avg_velocity}
        
        # Check for limit cycle
        is_cycle, period = self.cycle_detector.detect(recorder)
        if is_cycle:
            return AttractorType.LIMIT_CYCLE, {'period': period}
        
        # Check for strange attractor
        if self.strange_detector.is_chaotic(recorder):
            lyap = self.strange_detector.estimate_lyapunov_exponent(recorder)
            dim = self.strange_detector.compute_correlation_dimension(recorder)
            return AttractorType.STRANGE, {
                'lyapunov_exponent': lyap,
                'correlation_dimension': dim
            }
        
        return AttractorType.UNKNOWN, {}


class BasinOfAttraction:
    """
    Measure the basin of attraction for a stable pattern.
    
    Basin = set of initial conditions that converge to the attractor.
    """
    
    def __init__(self, reference_pattern, distance_threshold=0.1):
        """
        Args:
            reference_pattern: The attractor pattern
            distance_threshold: Maximum distance to consider "converged"
        """
        self.reference = reference_pattern.copy()
        self.threshold = distance_threshold
    
    def distance_to_attractor(self, state):
        """Compute distance from state to attractor."""
        return np.sqrt(np.mean((state - self.reference)**2))
    
    def is_in_basin(self, final_state):
        """Check if final state converged to attractor."""
        return self.distance_to_attractor(final_state) < self.threshold
    
    def measure_basin_volume(self, sample_fn, n_samples=100, 
                             simulation_fn=None, steps=1000):
        """
        Monte Carlo estimation of basin volume.
        
        Args:
            sample_fn: Function() -> random initial condition
            simulation_fn: Function(initial) -> final state
            n_samples: Number of random samples
            steps: Simulation steps per sample
            
        Returns:
            Fraction of samples that converged (basin measure)
        """
        if simulation_fn is None:
            return None
        
        converged = 0
        for _ in range(n_samples):
            initial = sample_fn()
            final = simulation_fn(initial)
            if self.is_in_basin(final):
                converged += 1
        
        return converged / n_samples
    
    def find_basin_boundary(self, direction, max_distance=10.0, 
                            simulation_fn=None, steps=1000, resolution=100):
        """
        Find basin boundary in a specific direction.
        
        Args:
            direction: Unit vector direction to search
            max_distance: Maximum distance to search
            simulation_fn: Function(initial) -> final state
            steps: Simulation steps
            resolution: Number of distance samples
            
        Returns:
            Distance to basin boundary in this direction
        """
        if simulation_fn is None:
            return None
        
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        distances = np.linspace(0, max_distance, resolution)
        
        for d in distances:
            initial = self.reference + d * direction
            final = simulation_fn(initial)
            
            if not self.is_in_basin(final):
                return d
        
        return max_distance  # Basin extends beyond search range


class BifurcationDetector:
    """
    Detect bifurcations (qualitative changes in dynamics).
    
    Types:
    - Saddle-node: Fixed point appears/disappears
    - Pitchfork: Symmetry breaking
    - Hopf: Fixed point becomes limit cycle
    - Period-doubling: Limit cycle period doubles
    """
    
    def __init__(self):
        self.parameter_values = []
        self.attractor_types = []
        self.properties = []
    
    def scan_parameter(self, parameter_values, simulation_fn, 
                       classifier, n_steps=500, transient=200):
        """
        Scan a parameter range and classify attractors.
        
        Args:
            parameter_values: Array of parameter values
            simulation_fn: Function(param) -> recorder after simulation
            classifier: AttractorClassifier
            n_steps: Total simulation steps
            transient: Steps to discard
            
        Returns:
            List of (parameter, type, properties) tuples
        """
        results = []
        
        for param in parameter_values:
            recorder = simulation_fn(param)
            attr_type, props = classifier.classify(recorder)
            results.append((param, attr_type, props))
            
            self.parameter_values.append(param)
            self.attractor_types.append(attr_type)
            self.properties.append(props)
        
        return results
    
    def find_bifurcations(self):
        """
        Identify bifurcation points from scan results.
        
        Returns:
            List of (parameter, old_type, new_type) at bifurcations
        """
        bifurcations = []
        
        for i in range(1, len(self.attractor_types)):
            if self.attractor_types[i] != self.attractor_types[i-1]:
                bifurcations.append((
                    self.parameter_values[i],
                    self.attractor_types[i-1],
                    self.attractor_types[i]
                ))
        
        return bifurcations
    
    def clear(self):
        """Reset scan results."""
        self.parameter_values.clear()
        self.attractor_types.clear()
        self.properties.clear()
