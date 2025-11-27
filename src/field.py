# ResonantField - The substrate layer (computational ether)
# All patterns exist as configurations within this field

import numpy as np
from src.config import (
    FIELD_WIDTH, FIELD_HEIGHT, FIELD_CHANNELS,
    DTYPE, DEFAULT_BOUNDARY,
    BOUNDARY_PERIODIC, BOUNDARY_NEUMANN
)


class ResonantField:
    """
    The fundamental substrate upon which all patterns emerge.
    
    A ResonantField is an N-dimensional grid with multiple channels,
    representing coupled field values (e.g., activator/inhibitor).
    """
    
    def __init__(self, width=FIELD_WIDTH, height=FIELD_HEIGHT, channels=FIELD_CHANNELS):
        self.width = width
        self.height = height
        self.channels = channels
        self.shape = (channels, height, width)
        self.boundary = DEFAULT_BOUNDARY
        
        # Primary field state: shape = (channels, height, width)
        self.data = np.zeros(self.shape, dtype=DTYPE)
        
        # Energy tracking
        self.total_energy = 0.0
        self.energy_history = []
    
    def initialize_uniform(self, values):
        """Set uniform values across the field for each channel."""
        for c in range(self.channels):
            self.data[c, :, :] = values[c]
    
    def initialize_random(self, low=0.0, high=1.0):
        """Initialize with uniform random noise."""
        self.data = np.random.uniform(low, high, self.shape).astype(DTYPE)
    
    def initialize_perturbed(self, base_values, perturbation_scale=0.01):
        """Initialize with base values plus small perturbations."""
        for c in range(self.channels):
            self.data[c, :, :] = base_values[c] + perturbation_scale * np.random.randn(self.height, self.width)
    
    def inject_seed(self, center_x, center_y, radius, channel, value):
        """Inject a circular seed at specified location."""
        y, x = np.ogrid[:self.height, :self.width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = distance <= radius
        self.data[channel, mask] = value
    
    def inject_frequency_bundle(self, wavenumbers, amplitudes, phases):
        """
        Inject a frequency bundle (genome) via inverse Fourier.
        
        wavenumbers: list of (kx, ky) tuples
        amplitudes: corresponding amplitudes
        phases: corresponding phases
        """
        for c in range(self.channels):
            spectrum = np.zeros((self.height, self.width), dtype=np.complex128)
            for (kx, ky), amp, phase in zip(wavenumbers, amplitudes[c], phases[c]):
                spectrum[ky % self.height, kx % self.width] = amp * np.exp(1j * phase)
            self.data[c, :, :] += np.real(np.fft.ifft2(spectrum))
    
    def apply_boundary(self):
        """Apply boundary conditions to the field."""
        if self.boundary == BOUNDARY_PERIODIC:
            # Periodic boundaries are handled implicitly by numpy roll/FFT
            pass
        elif self.boundary == BOUNDARY_NEUMANN:
            # Zero-flux: edges equal adjacent interior
            for c in range(self.channels):
                self.data[c, 0, :] = self.data[c, 1, :]
                self.data[c, -1, :] = self.data[c, -2, :]
                self.data[c, :, 0] = self.data[c, :, 1]
                self.data[c, :, -1] = self.data[c, :, -2]
    
    def get_channel(self, channel):
        """Return a specific channel of the field."""
        return self.data[channel]
    
    def set_channel(self, channel, values):
        """Set a specific channel of the field."""
        self.data[channel, :, :] = values
    
    def compute_energy(self):
        """Compute total field energy (sum of squared values + gradients)."""
        energy = 0.0
        for c in range(self.channels):
            # Potential energy (field magnitude)
            energy += np.sum(self.data[c]**2)
            # Gradient energy (spatial variation)
            grad_y = np.diff(self.data[c], axis=0)
            grad_x = np.diff(self.data[c], axis=1)
            energy += np.sum(grad_y**2) + np.sum(grad_x**2)
        self.total_energy = energy
        self.energy_history.append(energy)
        return energy
    
    def get_spectrum(self, channel):
        """Return the Fourier spectrum of a channel."""
        return np.fft.fft2(self.data[channel])
    
    def get_power_spectrum(self, channel):
        """Return the power spectrum (magnitude squared) of a channel."""
        spectrum = self.get_spectrum(channel)
        return np.abs(spectrum)**2
    
    def copy(self):
        """Create a deep copy of the field."""
        new_field = ResonantField(self.width, self.height, self.channels)
        new_field.data = self.data.copy()
        new_field.boundary = self.boundary
        return new_field
