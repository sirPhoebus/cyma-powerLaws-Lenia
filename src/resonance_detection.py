# Resonance Detection - Phase 2.7
#
# Analyzer pipeline for identifying resonant patterns:
#   - temporal_fft: Extract frequency content over time window
#   - spatial_fft: Extract spatial wavenumber content
#   - peak_detect: Find resonant frequencies
#   - mode_extract: Isolate spatial pattern at frequency
#   - stability_measure: Track amplitude evolution

import numpy as np
from scipy import signal
from scipy import ndimage
from src.config import DTYPE


class ResonanceAnalyzer:
    """
    Resonance detection and analysis pipeline.
    
    Identifies nodes, antinodes, modal patterns, and stability.
    """
    
    def __init__(self, dimensions, sample_rate=1.0):
        """
        Args:
            dimensions: Spatial dimensions of field
            sample_rate: Temporal sampling rate
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.sample_rate = sample_rate
        
        # Temporal buffer for frequency analysis
        self._time_buffer = []
        self._max_buffer_size = 1024
    
    def record_sample(self, field):
        """Add field snapshot to temporal buffer."""
        self._time_buffer.append(field.copy())
        if len(self._time_buffer) > self._max_buffer_size:
            self._time_buffer.pop(0)
    
    def clear_buffer(self):
        """Clear temporal buffer."""
        self._time_buffer.clear()
    
    def temporal_fft(self, window=None):
        """
        Extract frequency content over time window.
        
        Args:
            window: Number of samples to use (None = all)
            
        Returns:
            frequencies: Array of frequency values
            spectrum: Power at each frequency (spatially averaged)
            full_spectrum: Full 3D/4D spectrum (freq, spatial...)
        """
        if len(self._time_buffer) < 2:
            return None, None, None
        
        if window is None:
            window = len(self._time_buffer)
        
        data = np.array(self._time_buffer[-window:])
        n_samples = data.shape[0]
        
        # FFT along time axis
        spectrum = np.fft.rfft(data, axis=0)
        power = np.abs(spectrum)**2
        
        # Frequencies
        frequencies = np.fft.rfftfreq(n_samples, d=1.0/self.sample_rate)
        
        # Spatially averaged power
        spatial_axes = tuple(range(1, power.ndim))
        avg_power = np.mean(power, axis=spatial_axes)
        
        return frequencies, avg_power, power
    
    def spatial_fft(self, field):
        """
        Extract spatial wavenumber content.
        
        Args:
            field: Spatial field to analyze
            
        Returns:
            wavenumbers: List of wavenumber arrays for each axis
            power: Power spectrum
        """
        spectrum = np.fft.fftn(field)
        power = np.abs(spectrum)**2
        power_shifted = np.fft.fftshift(power)
        
        wavenumbers = [np.fft.fftshift(np.fft.fftfreq(d)) for d in self.dimensions]
        
        return wavenumbers, power_shifted
    
    def peak_detect(self, spectrum, threshold=0.1, min_distance=3):
        """
        Find peaks in spectrum (resonant frequencies).
        
        Args:
            spectrum: 1D power spectrum
            threshold: Minimum peak height as fraction of max
            min_distance: Minimum distance between peaks
            
        Returns:
            peak_indices: Indices of peaks
            peak_heights: Heights of peaks
        """
        if spectrum is None or len(spectrum) < 3:
            return np.array([]), np.array([])
        
        height_threshold = threshold * np.max(spectrum)
        peaks, properties = signal.find_peaks(
            spectrum, 
            height=height_threshold,
            distance=min_distance
        )
        
        return peaks, spectrum[peaks]
    
    def mode_extract(self, frequency_idx):
        """
        Extract spatial pattern at specific frequency.
        
        Args:
            frequency_idx: Index into frequency array
            
        Returns:
            Complex mode amplitude at each spatial point
        """
        if len(self._time_buffer) < 2:
            return None
        
        data = np.array(self._time_buffer)
        spectrum = np.fft.rfft(data, axis=0)
        
        if frequency_idx >= spectrum.shape[0]:
            return None
        
        return spectrum[frequency_idx]
    
    def stability_measure(self, n_samples=None):
        """
        Measure amplitude stability over time.
        
        Returns:
            mean_amplitude: Time-averaged amplitude at each point
            amplitude_std: Standard deviation of amplitude
            stability_index: mean / std (high = stable)
        """
        if len(self._time_buffer) < 2:
            return None, None, None
        
        if n_samples is None:
            n_samples = len(self._time_buffer)
        
        data = np.array(self._time_buffer[-n_samples:])
        
        mean_amp = np.mean(np.abs(data), axis=0)
        std_amp = np.std(np.abs(data), axis=0)
        
        # Stability = mean / std (avoid division by zero)
        stability = mean_amp / (std_amp + 1e-10)
        
        return mean_amp, std_amp, stability
    
    def find_nodes(self, threshold=0.1):
        """
        Find nodal regions (minimal oscillation).
        
        Returns boolean mask of node locations.
        """
        if len(self._time_buffer) < 10:
            return None
        
        data = np.array(self._time_buffer)
        temporal_max = np.max(np.abs(data), axis=0)
        
        global_max = np.max(temporal_max)
        if global_max < 1e-10:
            return np.ones(self.dimensions, dtype=bool)
        
        return temporal_max < threshold * global_max
    
    def find_antinodes(self, threshold=0.5):
        """
        Find antinode regions (maximal oscillation).
        
        Returns boolean mask of antinode locations.
        """
        if len(self._time_buffer) < 10:
            return None
        
        data = np.array(self._time_buffer)
        temporal_max = np.max(np.abs(data), axis=0)
        
        global_max = np.max(temporal_max)
        if global_max < 1e-10:
            return np.zeros(self.dimensions, dtype=bool)
        
        return temporal_max > threshold * global_max
    
    def compute_modal_decomposition(self, n_modes=5):
        """
        Decompose field into dominant spatial modes.
        
        Uses SVD on temporal data to extract principal modes.
        
        Returns:
            modes: Spatial mode patterns
            amplitudes: Time series of mode amplitudes
            singular_values: Importance of each mode
        """
        if len(self._time_buffer) < n_modes:
            return None, None, None
        
        data = np.array(self._time_buffer)
        n_times = data.shape[0]
        n_spatial = np.prod(self.dimensions)
        
        # Reshape to (time, space)
        data_2d = data.reshape(n_times, n_spatial)
        
        # SVD
        U, S, Vt = np.linalg.svd(data_2d, full_matrices=False)
        
        # Extract top n_modes
        modes = Vt[:n_modes].reshape((n_modes,) + self.dimensions)
        amplitudes = U[:, :n_modes] * S[:n_modes]
        singular_values = S[:n_modes]
        
        return modes, amplitudes, singular_values
    
    def estimate_q_factor(self, frequency_idx):
        """
        Estimate Q factor (quality factor) of resonance.
        
        Q = f_center / bandwidth
        High Q = sharp resonance, long decay
        
        Args:
            frequency_idx: Index of resonant peak
            
        Returns:
            Q factor estimate
        """
        freqs, spectrum, _ = self.temporal_fft()
        if spectrum is None:
            return None
        
        if frequency_idx >= len(spectrum):
            return None
        
        peak_power = spectrum[frequency_idx]
        half_power = peak_power / 2
        
        # Find bandwidth at half power
        left = frequency_idx
        while left > 0 and spectrum[left] > half_power:
            left -= 1
        
        right = frequency_idx
        while right < len(spectrum) - 1 and spectrum[right] > half_power:
            right += 1
        
        bandwidth = freqs[right] - freqs[left] if right > left else freqs[1]
        center_freq = freqs[frequency_idx]
        
        if bandwidth < 1e-10:
            return float('inf')
        
        return center_freq / bandwidth


class ResonancePipeline:
    """
    Complete resonance analysis pipeline.
    
    Combines all analysis stages for automated resonance detection.
    """
    
    def __init__(self, dimensions, sample_rate=1.0):
        self.analyzer = ResonanceAnalyzer(dimensions, sample_rate)
        self.dimensions = tuple(dimensions)
        
        # Results storage
        self.detected_frequencies = []
        self.detected_modes = []
        self.stability_map = None
    
    def process_frame(self, field):
        """Add frame to analysis buffer."""
        self.analyzer.record_sample(field)
    
    def run_analysis(self, threshold=0.1, n_modes=5):
        """
        Run complete analysis pipeline.
        
        Returns:
            Dictionary of analysis results
        """
        results = {}
        
        # Temporal frequency analysis
        freqs, power, full_spectrum = self.analyzer.temporal_fft()
        if freqs is not None:
            peaks, heights = self.analyzer.peak_detect(power, threshold)
            results['frequencies'] = freqs[peaks] if len(peaks) > 0 else []
            results['peak_powers'] = heights
            
            # Extract modes at peak frequencies
            self.detected_modes = []
            for peak_idx in peaks[:n_modes]:
                mode = self.analyzer.mode_extract(peak_idx)
                if mode is not None:
                    self.detected_modes.append(np.abs(mode))
            results['modes'] = self.detected_modes
        
        # Stability analysis
        mean_amp, std_amp, stability = self.analyzer.stability_measure()
        if stability is not None:
            self.stability_map = stability
            results['mean_amplitude'] = mean_amp
            results['stability'] = stability
        
        # Node/antinode detection
        nodes = self.analyzer.find_nodes()
        antinodes = self.analyzer.find_antinodes()
        if nodes is not None:
            results['nodes'] = nodes
            results['antinodes'] = antinodes
            results['node_fraction'] = np.mean(nodes)
            results['antinode_fraction'] = np.mean(antinodes)
        
        # Modal decomposition
        modes, amplitudes, singular_values = self.analyzer.compute_modal_decomposition(n_modes)
        if modes is not None:
            results['svd_modes'] = modes
            results['svd_amplitudes'] = amplitudes
            results['singular_values'] = singular_values
            results['energy_captured'] = np.sum(singular_values[:n_modes]**2) / np.sum(singular_values**2)
        
        return results
