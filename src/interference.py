# Interference Mechanics - Phase 2.5
#
# Wave interference analysis:
#   - Constructive interference: phases align, amplitudes add
#   - Destructive interference: phases oppose, amplitudes cancel
#   - Standing waves: trapped energy forms stable patterns
#
# Phase coherence metric:
#   coherence(w1, w2) = |<exp(i * (phase(w1) - phase(w2)))>|

import numpy as np
from src.config import DTYPE


class InterferenceAnalyzer:
    """
    Analyze interference between wave fields.
    
    Interference is the fundamental mechanism by which waves interact.
    Constructive interference creates antinodes (high amplitude),
    destructive interference creates nodes (zero amplitude).
    """
    
    def __init__(self, dimensions):
        """
        Args:
            dimensions: Spatial dimensions of the field
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
    
    def compute_phase(self, field, use_hilbert=True):
        """
        Extract instantaneous phase from a real-valued field.
        
        Uses Hilbert transform to get analytic signal, then extracts phase.
        
        Args:
            field: Real-valued wave field
            use_hilbert: If True, use Hilbert transform. If False, use gradient.
            
        Returns:
            Phase field in radians [-pi, pi]
        """
        if use_hilbert:
            return self._phase_from_hilbert(field)
        else:
            return self._phase_from_gradient(field)
    
    def _phase_from_hilbert(self, field):
        """Extract phase using N-dimensional Hilbert transform."""
        # For N-D, apply Hilbert along each axis and combine
        analytic = np.zeros_like(field, dtype=np.complex128)
        
        for axis in range(self.ndim):
            # 1D Hilbert transform along this axis
            spectrum = np.fft.fft(field, axis=axis)
            n = field.shape[axis]
            h = np.zeros(n)
            if n % 2 == 0:
                h[0] = h[n // 2] = 1
                h[1:n // 2] = 2
            else:
                h[0] = 1
                h[1:(n + 1) // 2] = 2
            
            # Reshape h for broadcasting
            shape = [1] * self.ndim
            shape[axis] = n
            h = h.reshape(shape)
            
            analytic += np.fft.ifft(spectrum * h, axis=axis)
        
        analytic /= self.ndim
        return np.angle(analytic)
    
    def _phase_from_gradient(self, field):
        """
        Estimate phase from spatial gradients.
        
        Less accurate but faster for large fields.
        """
        # Use atan2 of gradients as rough phase estimate
        grad_sum = np.zeros_like(field)
        for axis in range(self.ndim):
            grad_sum += np.gradient(field, axis=axis)**2
        
        # Phase is approximately arctan(field / gradient_magnitude)
        grad_mag = np.sqrt(grad_sum) + 1e-10
        return np.arctan2(field, grad_mag)
    
    def compute_coherence(self, field1, field2, window_size=None):
        """
        Compute phase coherence between two wave fields.
        
        coherence = |<exp(i * (phase1 - phase2))>|
        
        Coherence = 1 means perfectly phase-locked
        Coherence = 0 means random phase relationship
        
        Args:
            field1, field2: Wave fields
            window_size: If set, compute local coherence in windows
            
        Returns:
            Scalar coherence (global) or coherence field (local)
        """
        phase1 = self.compute_phase(field1)
        phase2 = self.compute_phase(field2)
        
        phase_diff = phase1 - phase2
        coherence_complex = np.exp(1j * phase_diff)
        
        if window_size is None:
            # Global coherence
            return np.abs(np.mean(coherence_complex))
        else:
            # Local coherence using convolution
            from scipy import ndimage
            kernel_shape = tuple([window_size] * self.ndim)
            kernel = np.ones(kernel_shape) / np.prod(kernel_shape)
            
            real_part = ndimage.convolve(np.real(coherence_complex), kernel, mode='wrap')
            imag_part = ndimage.convolve(np.imag(coherence_complex), kernel, mode='wrap')
            
            return np.sqrt(real_part**2 + imag_part**2)
    
    def superpose(self, *fields, weights=None):
        """
        Superpose multiple wave fields.
        
        result = sum_i(weight_i * field_i)
        
        Args:
            fields: Variable number of wave fields
            weights: Optional weights (default: equal weights)
            
        Returns:
            Superposed field
        """
        if weights is None:
            weights = [1.0] * len(fields)
        
        result = np.zeros_like(fields[0])
        for field, weight in zip(fields, weights):
            result += weight * field
        
        return result
    
    def detect_interference_type(self, field1, field2, threshold=0.5):
        """
        Classify interference at each point as constructive or destructive.
        
        Args:
            field1, field2: Wave fields
            threshold: Coherence threshold for classification
            
        Returns:
            Dictionary with:
              - 'constructive': Boolean mask where interference is constructive
              - 'destructive': Boolean mask where interference is destructive
              - 'phase_diff': Phase difference field
        """
        phase1 = self.compute_phase(field1)
        phase2 = self.compute_phase(field2)
        
        phase_diff = phase1 - phase2
        # Wrap to [-pi, pi]
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
        
        # Constructive: phase difference near 0 or 2*pi
        constructive = np.abs(phase_diff) < np.pi / 2
        
        # Destructive: phase difference near pi
        destructive = np.abs(np.abs(phase_diff) - np.pi) < np.pi / 2
        
        return {
            'constructive': constructive,
            'destructive': destructive,
            'phase_diff': phase_diff
        }
    
    def compute_interference_pattern(self, sources, positions, frequencies, 
                                      amplitudes, time, wave_speed=1.0):
        """
        Compute interference pattern from multiple point sources.
        
        Args:
            sources: Not used (positions define sources)
            positions: List of source positions [(x,y,...), ...]
            frequencies: List of frequencies for each source
            amplitudes: List of amplitudes for each source
            time: Current time
            wave_speed: Wave propagation speed
            
        Returns:
            Superposed field from all sources
        """
        # Create coordinate grids
        grids = np.ogrid[tuple(slice(0, d) for d in self.dimensions)]
        
        result = np.zeros(self.dimensions, dtype=DTYPE)
        
        for pos, freq, amp in zip(positions, frequencies, amplitudes):
            # Distance from this source
            dist_squared = sum((g - p)**2 for g, p in zip(grids, pos))
            dist = np.sqrt(dist_squared) + 1e-10
            
            # Wave from this source: A * sin(k*r - omega*t) / r
            omega = 2 * np.pi * freq
            k = omega / wave_speed
            
            # Spherical wave with 1/r falloff (or 1/sqrt(r) in 2D)
            if self.ndim == 2:
                falloff = 1.0 / np.sqrt(dist)
            else:
                falloff = 1.0 / dist
            
            wave = amp * falloff * np.sin(k * dist - omega * time)
            result += wave
        
        return result


class StandingWaveDetector:
    """
    Detect and analyze standing wave patterns.
    
    Standing waves form when waves are trapped by boundaries
    or when counter-propagating waves interfere.
    """
    
    def __init__(self, dimensions):
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        
        # History for temporal analysis
        self._amplitude_history = []
        self._max_history = 100
    
    def record_frame(self, amplitude):
        """Record amplitude snapshot for temporal analysis."""
        self._amplitude_history.append(amplitude.copy())
        if len(self._amplitude_history) > self._max_history:
            self._amplitude_history.pop(0)
    
    def clear_history(self):
        """Clear recorded history."""
        self._amplitude_history.clear()
    
    def detect_nodes(self, threshold_fraction=0.1):
        """
        Detect nodal regions from temporal history.
        
        Nodes are regions where amplitude remains consistently low.
        
        Returns:
            Boolean mask of nodal regions
        """
        if len(self._amplitude_history) < 10:
            return None
        
        # Compute temporal max at each point
        history_array = np.array(self._amplitude_history)
        temporal_max = np.max(np.abs(history_array), axis=0)
        
        global_max = np.max(temporal_max)
        if global_max < 1e-10:
            return np.ones(self.dimensions, dtype=bool)
        
        threshold = threshold_fraction * global_max
        return temporal_max < threshold
    
    def detect_antinodes(self, threshold_fraction=0.5):
        """
        Detect antinode regions from temporal history.
        
        Antinodes are regions of maximum oscillation.
        
        Returns:
            Boolean mask of antinode regions
        """
        if len(self._amplitude_history) < 10:
            return None
        
        history_array = np.array(self._amplitude_history)
        temporal_max = np.max(np.abs(history_array), axis=0)
        
        global_max = np.max(temporal_max)
        if global_max < 1e-10:
            return np.zeros(self.dimensions, dtype=bool)
        
        threshold = threshold_fraction * global_max
        return temporal_max > threshold
    
    def compute_oscillation_amplitude(self):
        """
        Compute peak-to-peak oscillation amplitude at each point.
        
        Returns:
            Field of oscillation amplitudes
        """
        if len(self._amplitude_history) < 10:
            return None
        
        history_array = np.array(self._amplitude_history)
        return np.max(history_array, axis=0) - np.min(history_array, axis=0)
    
    def estimate_standing_wave_frequency(self):
        """
        Estimate dominant frequency of standing wave oscillation.
        
        Uses FFT of temporal history at high-amplitude points.
        
        Returns:
            Estimated frequency (normalized)
        """
        if len(self._amplitude_history) < 20:
            return None
        
        history_array = np.array(self._amplitude_history)
        
        # Find point with maximum oscillation
        osc_amp = self.compute_oscillation_amplitude()
        max_idx = np.unravel_index(np.argmax(osc_amp), osc_amp.shape)
        
        # Get time series at this point
        time_series = history_array[(slice(None),) + max_idx]
        
        # FFT to find frequency
        spectrum = np.abs(np.fft.rfft(time_series))
        freqs = np.fft.rfftfreq(len(time_series))
        
        # Find peak (excluding DC)
        spectrum[0] = 0
        peak_idx = np.argmax(spectrum)
        
        return freqs[peak_idx]
    
    def compute_mode_number_estimate(self):
        """
        Estimate mode numbers from nodal pattern.
        
        Counts zero-crossings along each axis.
        
        Returns:
            Tuple of estimated mode numbers for each dimension
        """
        nodes = self.detect_nodes()
        if nodes is None:
            return None
        
        # Convert to float for analysis
        nodes_float = nodes.astype(float)
        
        mode_numbers = []
        for axis in range(self.ndim):
            # Sum along all other axes
            projection = np.sum(nodes_float, axis=tuple(i for i in range(self.ndim) if i != axis))
            
            # Count peaks (nodal lines/planes)
            from scipy import signal
            peaks, _ = signal.find_peaks(projection)
            mode_numbers.append(len(peaks))
        
        return tuple(mode_numbers)


class TwoSourceInterference:
    """
    Classic two-source interference pattern analysis.
    
    Creates the familiar interference fringes from two coherent sources.
    """
    
    def __init__(self, dimensions, source1_pos, source2_pos, frequency, wave_speed=1.0):
        """
        Args:
            dimensions: Field dimensions
            source1_pos, source2_pos: Source positions
            frequency: Common frequency (coherent sources)
            wave_speed: Wave propagation speed
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.source1 = tuple(source1_pos)
        self.source2 = tuple(source2_pos)
        self.frequency = frequency
        self.wave_speed = wave_speed
        
        self.omega = 2 * np.pi * frequency
        self.wavelength = wave_speed / frequency
        self.k = 2 * np.pi / self.wavelength
        
        # Precompute distance fields
        grids = np.ogrid[tuple(slice(0, d) for d in self.dimensions)]
        
        dist1_sq = sum((g - p)**2 for g, p in zip(grids, self.source1))
        dist2_sq = sum((g - p)**2 for g, p in zip(grids, self.source2))
        
        self.dist1 = np.sqrt(dist1_sq)
        self.dist2 = np.sqrt(dist2_sq)
        self.path_difference = self.dist1 - self.dist2
    
    def compute_pattern(self, time, amp1=1.0, amp2=1.0, phase_diff=0.0):
        """
        Compute interference pattern at given time.
        
        Args:
            time: Current time
            amp1, amp2: Source amplitudes
            phase_diff: Phase difference between sources
            
        Returns:
            Interference field
        """
        # Wave from source 1
        wave1 = amp1 * np.sin(self.k * self.dist1 - self.omega * time) / (self.dist1 + 1e-10)
        
        # Wave from source 2 with phase difference
        wave2 = amp2 * np.sin(self.k * self.dist2 - self.omega * time + phase_diff) / (self.dist2 + 1e-10)
        
        return wave1 + wave2
    
    def compute_intensity_pattern(self, amp1=1.0, amp2=1.0):
        """
        Compute time-averaged intensity pattern.
        
        I = I1 + I2 + 2*sqrt(I1*I2)*cos(k*delta_path)
        
        This is the classic fringe pattern.
        """
        I1 = amp1**2 / (self.dist1 + 1e-10)**2
        I2 = amp2**2 / (self.dist2 + 1e-10)**2
        
        # Interference term
        interference = 2 * np.sqrt(I1 * I2) * np.cos(self.k * self.path_difference)
        
        return I1 + I2 + interference
    
    def find_constructive_fringes(self):
        """
        Find locations of constructive interference.
        
        Constructive when path difference = n * wavelength
        """
        n_fringes = self.path_difference / self.wavelength
        # Constructive at integer multiples
        return np.abs(n_fringes - np.round(n_fringes)) < 0.25
    
    def find_destructive_fringes(self):
        """
        Find locations of destructive interference.
        
        Destructive when path difference = (n + 0.5) * wavelength
        """
        n_fringes = self.path_difference / self.wavelength
        # Destructive at half-integer multiples
        return np.abs(n_fringes - np.round(n_fringes) - 0.5) < 0.25
    
    def compute_fringe_spacing(self, observation_distance):
        """
        Compute fringe spacing at given distance from sources.
        
        For far-field: delta_y = lambda * D / d
        where D = observation distance, d = source separation
        """
        source_separation = np.sqrt(sum((s1 - s2)**2 
                                        for s1, s2 in zip(self.source1, self.source2)))
        return self.wavelength * observation_distance / source_separation


def compute_beat_frequency(freq1, freq2):
    """
    Compute beat frequency from two sources.
    
    f_beat = |f1 - f2|
    """
    return abs(freq1 - freq2)


def compute_group_velocity(frequencies, wave_speeds):
    """
    Compute group velocity from dispersion relation.
    
    v_g = dw/dk
    
    For non-dispersive medium, v_g = v_phase
    """
    if len(frequencies) < 2:
        return wave_speeds[0] if wave_speeds else 1.0
    
    # Numerical derivative
    omega = 2 * np.pi * np.array(frequencies)
    k = omega / np.array(wave_speeds)
    
    return np.gradient(omega) / np.gradient(k)
