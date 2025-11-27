# ResonantFieldND - N-Dimensional Substrate Layer
# True N-dimensional field supporting 1D, 2D, 3D, and higher dimensional spaces

import numpy as np
from scipy import ndimage
from src.config import (
    FIELD_DIMENSIONS, FIELD_CHANNELS,
    DTYPE, DEFAULT_BOUNDARY,
    BOUNDARY_PERIODIC, BOUNDARY_NEUMANN, BOUNDARY_DIRICHLET, 
    BOUNDARY_ABSORBING, DIRICHLET_VALUE
)


class ResonantFieldND:
    """
    N-Dimensional Resonant Field - The universal substrate.
    
    Supports arbitrary spatial dimensions (1D, 2D, 3D, ...).
    Each point in space has multiple channels (field components).
    
    Shape: (channels, dim_0, dim_1, ..., dim_n)
    """
    
    def __init__(self, dimensions=FIELD_DIMENSIONS, channels=FIELD_CHANNELS):
        """
        Initialize an N-dimensional field.
        
        Args:
            dimensions: Tuple of spatial dimensions, e.g., (256,) for 1D,
                       (256, 256) for 2D, (64, 64, 64) for 3D
            channels: Number of field components (e.g., 2 for activator/inhibitor)
        """
        if isinstance(dimensions, int):
            dimensions = (dimensions,)
        
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.channels = channels
        self.shape = (channels,) + self.dimensions
        self.spatial_shape = self.dimensions
        self.boundary = DEFAULT_BOUNDARY
        
        # Primary field state
        self.data = np.zeros(self.shape, dtype=DTYPE)
        
        # Energy tracking
        self.total_energy = 0.0
        self.energy_history = []
        
        # Precompute coordinate grids for distance calculations
        self._coord_grids = None
    
    @property
    def size(self):
        """Total number of spatial points."""
        return np.prod(self.dimensions)
    
    @property
    def total_size(self):
        """Total number of values (channels * spatial points)."""
        return np.prod(self.shape)
    
    def _get_coord_grids(self):
        """Lazily compute coordinate grids for distance calculations."""
        if self._coord_grids is None:
            self._coord_grids = np.ogrid[tuple(slice(0, d) for d in self.dimensions)]
        return self._coord_grids
    
    def initialize_uniform(self, values):
        """Set uniform values across the field for each channel."""
        for c in range(self.channels):
            self.data[c] = values[c]
    
    def initialize_random(self, low=0.0, high=1.0):
        """Initialize with uniform random noise."""
        self.data = np.random.uniform(low, high, self.shape).astype(DTYPE)
    
    def initialize_perturbed(self, base_values, perturbation_scale=0.01):
        """Initialize with base values plus small perturbations."""
        for c in range(self.channels):
            self.data[c] = base_values[c] + perturbation_scale * np.random.randn(*self.dimensions)
    
    def inject_seed_sphere(self, center, radius, channel, value):
        """
        Inject a hyperspherical seed at specified location.
        
        Args:
            center: Tuple of center coordinates (length = ndim)
            radius: Radius of the hypersphere
            channel: Which channel to modify
            value: Value to set inside the sphere
        """
        if len(center) != self.ndim:
            raise ValueError(f"Center must have {self.ndim} coordinates, got {len(center)}")
        
        grids = self._get_coord_grids()
        
        # Compute squared distance from center
        dist_squared = sum((g - c)**2 for g, c in zip(grids, center))
        mask = dist_squared <= radius**2
        self.data[channel][mask] = value
    
    def inject_seed_box(self, corner_min, corner_max, channel, value):
        """
        Inject a hyperbox seed.
        
        Args:
            corner_min: Tuple of minimum coordinates
            corner_max: Tuple of maximum coordinates
            channel: Which channel to modify
            value: Value to set inside the box
        """
        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(corner_min, corner_max))
        self.data[(channel,) + slices] = value
    
    def inject_frequency_bundle(self, wavenumbers, amplitudes, phases):
        """
        Inject a frequency bundle (genome) via inverse N-dimensional FFT.
        
        Args:
            wavenumbers: List of wavenumber tuples, each of length ndim
            amplitudes: List of amplitude arrays, one per channel
            phases: List of phase arrays, one per channel
        """
        for c in range(self.channels):
            spectrum = np.zeros(self.dimensions, dtype=np.complex128)
            for k_vec, amp, phase in zip(wavenumbers, amplitudes[c], phases[c]):
                # Wrap wavenumbers to valid indices
                idx = tuple(k % d for k, d in zip(k_vec, self.dimensions))
                spectrum[idx] = amp * np.exp(1j * phase)
            self.data[c] += np.real(np.fft.ifftn(spectrum))
    
    def inject_sinusoidal_mode(self, wavenumber, amplitude, phase, channel):
        """
        Inject a single sinusoidal mode.
        
        Args:
            wavenumber: Tuple of wavenumbers (k0, k1, ..., kn)
            amplitude: Amplitude of the mode
            phase: Phase of the mode
            channel: Which channel to modify
        """
        grids = self._get_coord_grids()
        
        # Compute k dot x
        k_dot_x = sum(k * (2 * np.pi * g / d) 
                      for k, g, d in zip(wavenumber, grids, self.dimensions))
        
        self.data[channel] += amplitude * np.cos(k_dot_x + phase)
    
    def apply_boundary(self):
        """Apply boundary conditions to all spatial boundaries."""
        if self.boundary == BOUNDARY_PERIODIC:
            # Periodic handled implicitly by FFT convolutions
            pass
        elif self.boundary == BOUNDARY_NEUMANN:
            # Zero-flux (reflective): each face equals adjacent interior
            for c in range(self.channels):
                for axis in range(self.ndim):
                    slice_first = [slice(None)] * (self.ndim + 1)
                    slice_second = [slice(None)] * (self.ndim + 1)
                    slice_first[axis + 1] = 0
                    slice_second[axis + 1] = 1
                    self.data[tuple(slice_first)] = self.data[tuple(slice_second)]
                    
                    slice_first[axis + 1] = -1
                    slice_second[axis + 1] = -2
                    self.data[tuple(slice_first)] = self.data[tuple(slice_second)]
        elif self.boundary in (BOUNDARY_DIRICHLET, BOUNDARY_ABSORBING):
            # Fixed value (absorbing): boundaries set to constant
            for c in range(self.channels):
                for axis in range(self.ndim):
                    slice_first = [slice(None)] * (self.ndim + 1)
                    slice_last = [slice(None)] * (self.ndim + 1)
                    slice_first[axis + 1] = 0
                    slice_last[axis + 1] = -1
                    self.data[tuple(slice_first)] = DIRICHLET_VALUE
                    self.data[tuple(slice_last)] = DIRICHLET_VALUE
    
    def get_channel(self, channel):
        """Return a specific channel of the field."""
        return self.data[channel]
    
    def set_channel(self, channel, values):
        """Set a specific channel of the field."""
        self.data[channel] = values
    
    def compute_energy(self):
        """
        Compute total field energy.
        
        Energy = sum over channels of:
            - Potential energy: integral of field^2
            - Gradient energy: integral of |grad(field)|^2
        """
        energy = 0.0
        for c in range(self.channels):
            # Potential energy
            energy += np.sum(self.data[c]**2)
            
            # Gradient energy (along each spatial axis)
            for axis in range(self.ndim):
                grad = np.diff(self.data[c], axis=axis)
                energy += np.sum(grad**2)
        
        self.total_energy = energy
        self.energy_history.append(energy)
        return energy
    
    def compute_gradient_magnitude(self, channel):
        """Compute the magnitude of the gradient for a channel."""
        grad_squared = np.zeros(self.dimensions, dtype=DTYPE)
        
        for axis in range(self.ndim):
            # Central difference gradient
            grad = np.zeros_like(self.data[channel])
            grad[tuple(slice(1, -1) if i == axis else slice(None) 
                      for i in range(self.ndim))] = (
                np.diff(self.data[channel], n=1, axis=axis)[
                    tuple(slice(1, None) if i == axis else slice(None) 
                          for i in range(self.ndim))] +
                np.diff(self.data[channel], n=1, axis=axis)[
                    tuple(slice(None, -1) if i == axis else slice(None) 
                          for i in range(self.ndim))]
            ) / 2
            grad_squared += grad**2
        
        return np.sqrt(grad_squared)
    
    def get_spectrum(self, channel):
        """Return the N-dimensional Fourier spectrum of a channel."""
        return np.fft.fftn(self.data[channel])
    
    def get_power_spectrum(self, channel):
        """Return the power spectrum (magnitude squared) of a channel."""
        spectrum = self.get_spectrum(channel)
        return np.abs(spectrum)**2
    
    def get_radial_power_spectrum(self, channel):
        """
        Compute radially averaged power spectrum.
        Useful for detecting characteristic wavelengths in isotropic patterns.
        """
        power = self.get_power_spectrum(channel)
        power_shifted = np.fft.fftshift(power)
        
        # Create radial coordinate
        center = tuple(d // 2 for d in self.dimensions)
        grids = self._get_coord_grids()
        r = np.sqrt(sum((g - c)**2 for g, c in zip(grids, center)))
        
        # Bin by radius
        r_int = r.astype(int).flatten()
        power_flat = power_shifted.flatten()
        
        max_r = int(np.max(r_int)) + 1
        radial_sum = np.bincount(r_int, weights=power_flat, minlength=max_r)
        radial_count = np.bincount(r_int, minlength=max_r)
        
        radial_mean = np.zeros(max_r)
        nonzero = radial_count > 0
        radial_mean[nonzero] = radial_sum[nonzero] / radial_count[nonzero]
        
        return radial_mean
    
    def get_slice(self, axis, index):
        """
        Get a slice of the field perpendicular to the specified axis.
        Useful for visualizing 3D+ fields.
        """
        slices = [slice(None)] * (self.ndim + 1)  # +1 for channel dimension
        slices[axis + 1] = index
        return self.data[tuple(slices)]
    
    def copy(self):
        """Create a deep copy of the field."""
        new_field = ResonantFieldND(self.dimensions, self.channels)
        new_field.data = self.data.copy()
        new_field.boundary = self.boundary
        new_field.total_energy = self.total_energy
        new_field.energy_history = self.energy_history.copy()
        return new_field
    
    # =========================================================================
    # CORE OPERATIONS (Phase 1.2)
    # =========================================================================
    
    def superpose(self, other, weight_self=1.0, weight_other=1.0):
        """
        Wave superposition: combine two fields linearly.
        
        result = weight_self * self + weight_other * other
        
        Args:
            other: Another ResonantFieldND with same dimensions
            weight_self: Weight for this field
            weight_other: Weight for other field
            
        Returns:
            New ResonantFieldND with superposed values
        """
        if self.dimensions != other.dimensions:
            raise ValueError(f"Dimension mismatch: {self.dimensions} vs {other.dimensions}")
        if self.channels != other.channels:
            raise ValueError(f"Channel mismatch: {self.channels} vs {other.channels}")
        
        result = self.copy()
        result.data = weight_self * self.data + weight_other * other.data
        return result
    
    def superpose_inplace(self, other, weight_other=1.0):
        """
        Add another field to this one in-place.
        
        self = self + weight_other * other
        """
        if self.dimensions != other.dimensions:
            raise ValueError(f"Dimension mismatch: {self.dimensions} vs {other.dimensions}")
        self.data += weight_other * other.data
    
    def diffuse(self, coefficients, kernel=None):
        """
        Apply diffusion operator: D * nabla^2(field)
        
        Args:
            coefficients: List of diffusion coefficients per channel [D_u, D_v, ...]
            kernel: Optional custom Laplacian kernel (uses default if None)
            
        Returns:
            New field with diffusion applied
        """
        from src.kernel_nd import KernelND
        
        if kernel is None:
            kernel = KernelND.laplacian(ndim=self.ndim)
        
        result = self.copy()
        for c in range(self.channels):
            laplacian = kernel.convolve(self.data[c], mode='wrap')
            result.data[c] = coefficients[c] * laplacian
        return result
    
    def measure(self, region=None, metrics=None):
        """
        Extract observables from the field (energy, entropy, statistics).
        
        Args:
            region: Optional tuple of slices to measure a subregion.
                    None means entire field.
                    Example: ((10, 50), (10, 50)) for 2D subregion
            metrics: List of metrics to compute. Default: all.
                     Options: 'energy', 'entropy', 'mean', 'std', 'min', 'max', 'sum'
        
        Returns:
            Dictionary of measurement results
        """
        # Extract region
        if region is not None:
            slices = (slice(None),) + tuple(slice(r[0], r[1]) for r in region)
            data = self.data[slices]
        else:
            data = self.data
        
        if metrics is None:
            metrics = ['energy', 'entropy', 'mean', 'std', 'min', 'max', 'sum']
        
        results = {}
        
        for metric in metrics:
            if metric == 'mean':
                results['mean'] = [np.mean(data[c]) for c in range(self.channels)]
            elif metric == 'std':
                results['std'] = [np.std(data[c]) for c in range(self.channels)]
            elif metric == 'min':
                results['min'] = [np.min(data[c]) for c in range(self.channels)]
            elif metric == 'max':
                results['max'] = [np.max(data[c]) for c in range(self.channels)]
            elif metric == 'sum':
                results['sum'] = [np.sum(data[c]) for c in range(self.channels)]
            elif metric == 'energy':
                energy = 0.0
                for c in range(self.channels):
                    energy += np.sum(data[c]**2)
                    for axis in range(data.ndim - 1):  # -1 for channel dim
                        grad = np.diff(data[c], axis=axis)
                        energy += np.sum(grad**2)
                results['energy'] = energy
            elif metric == 'entropy':
                # Shannon entropy (treating values as probabilities)
                entropy_list = []
                for c in range(self.channels):
                    p = np.abs(data[c]) + 1e-10
                    p = p / np.sum(p)
                    entropy_list.append(-np.sum(p * np.log(p)))
                results['entropy'] = entropy_list
        
        return results
    
    def convolve(self, kernel, channel=None):
        """
        Apply a convolution kernel to the field.
        
        Args:
            kernel: KernelND instance
            channel: Specific channel to convolve (None = all channels)
            
        Returns:
            New field with convolution applied
        """
        result = self.copy()
        
        if channel is not None:
            result.data[channel] = kernel.convolve(self.data[channel])
        else:
            for c in range(self.channels):
                result.data[c] = kernel.convolve(self.data[c])
        
        return result
    
    def react(self, reaction_system):
        """
        Apply a reaction system to compute local dynamics.
        
        Args:
            reaction_system: ReactionSystem instance (e.g., GrayScott)
            
        Returns:
            Tuple of reaction terms (du, dv, ...)
        """
        if self.channels < 2:
            raise ValueError("Reaction requires at least 2 channels (u, v)")
        
        return reaction_system.react(self.data[0], self.data[1])
    
    def __repr__(self):
        return (f"ResonantFieldND(dimensions={self.dimensions}, "
                f"channels={self.channels}, ndim={self.ndim})")


# Backward compatibility: alias for 2D case
def ResonantField2D(width, height, channels=FIELD_CHANNELS):
    """Create a 2D ResonantField (convenience wrapper)."""
    return ResonantFieldND(dimensions=(height, width), channels=channels)


def ResonantField3D(depth, height, width, channels=FIELD_CHANNELS):
    """Create a 3D ResonantField (convenience wrapper)."""
    return ResonantFieldND(dimensions=(depth, height, width), channels=channels)
