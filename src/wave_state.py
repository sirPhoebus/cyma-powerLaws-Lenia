# Wave State - Phase 2.1 Wave Representation
# 
# Core Philosophy:
#   - Waves are NOT objects - they are disturbances in the field
#   - A wave is defined by its SOURCE FUNCTION (where/when/how energy is injected)
#   - Propagation is EMERGENT from the field equation, not coded separately
#
# The WaveState extends ResonantFieldND to include velocity (du/dt) required
# for second-order wave equations. The "wave" exists implicitly in the
# amplitude-velocity pair, not as a separate entity.

import numpy as np
from src.field_nd import ResonantFieldND
from src.config import (
    FIELD_DIMENSIONS, FIELD_CHANNELS, DTYPE,
    WAVE_SPEED, WAVE_DAMPING, WAVE_DENSITY,
    DEFAULT_BOUNDARY, BOUNDARY_PERIODIC
)


class WaveState:
    """
    Wave State - The substrate for wave dynamics.
    
    A wave is not an object but a disturbance pattern in the field.
    This class pairs:
      - amplitude field u(x,t) : the displacement from equilibrium
      - velocity field v(x,t) = du/dt : the rate of change
    
    Together they form the complete state for second-order wave equations:
        d2u/dt2 = c^2 * Laplacian(u) - gamma * du/dt + F(x,t)
    
    Rewritten as first-order system:
        du/dt = v
        dv/dt = c^2 * Laplacian(u) - gamma * v + F(x,t)
    """
    
    def __init__(self, dimensions=FIELD_DIMENSIONS, channels=FIELD_CHANNELS):
        """
        Initialize wave state.
        
        Args:
            dimensions: Spatial dimensions tuple (e.g., (256, 256) for 2D)
            channels: Number of field components
        """
        if isinstance(dimensions, int):
            dimensions = (dimensions,)
        
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.channels = channels
        self.shape = (channels,) + self.dimensions
        
        # Amplitude field: u(x,t) - the displacement
        self.amplitude = np.zeros(self.shape, dtype=DTYPE)
        
        # Velocity field: v(x,t) = du/dt - the rate of change
        self.velocity = np.zeros(self.shape, dtype=DTYPE)
        
        # Material properties (can be spatially varying)
        self._wave_speed = np.full(self.dimensions, WAVE_SPEED, dtype=DTYPE)
        self._damping = np.full(self.dimensions, WAVE_DAMPING, dtype=DTYPE)
        self._density = np.full(self.dimensions, WAVE_DENSITY, dtype=DTYPE)
        
        # Boundary condition
        self.boundary = DEFAULT_BOUNDARY
        
        # Time tracking
        self.time = 0.0
        
        # Energy history
        self.energy_history = []
        
        # Precompute coordinate grids
        self._coord_grids = None
    
    @property
    def wave_speed(self):
        """Wave speed field c(x)."""
        return self._wave_speed
    
    @wave_speed.setter
    def wave_speed(self, value):
        """Set wave speed (scalar or array)."""
        if np.isscalar(value):
            self._wave_speed.fill(value)
        else:
            self._wave_speed = np.asarray(value, dtype=DTYPE)
    
    @property
    def damping(self):
        """Damping coefficient field gamma(x)."""
        return self._damping
    
    @damping.setter
    def damping(self, value):
        """Set damping coefficient (scalar or array)."""
        if np.isscalar(value):
            self._damping.fill(value)
        else:
            self._damping = np.asarray(value, dtype=DTYPE)
    
    @property
    def density(self):
        """Density field rho(x)."""
        return self._density
    
    @density.setter
    def density(self, value):
        """Set density (scalar or array)."""
        if np.isscalar(value):
            self._density.fill(value)
        else:
            self._density = np.asarray(value, dtype=DTYPE)
    
    def _get_coord_grids(self):
        """Lazily compute coordinate grids."""
        if self._coord_grids is None:
            self._coord_grids = np.ogrid[tuple(slice(0, d) for d in self.dimensions)]
        return self._coord_grids
    
    def reset(self):
        """Reset state to equilibrium (zero amplitude and velocity)."""
        self.amplitude.fill(0.0)
        self.velocity.fill(0.0)
        self.time = 0.0
        self.energy_history.clear()
    
    def set_amplitude(self, channel, values):
        """Set amplitude for a specific channel."""
        self.amplitude[channel] = values
    
    def set_velocity(self, channel, values):
        """Set velocity for a specific channel."""
        self.velocity[channel] = values
    
    def get_amplitude(self, channel=0):
        """Get amplitude for a specific channel."""
        return self.amplitude[channel]
    
    def get_velocity(self, channel=0):
        """Get velocity for a specific channel."""
        return self.velocity[channel]
    
    def inject_gaussian_pulse(self, center, sigma, amplitude, channel=0):
        """
        Inject a Gaussian pulse at specified location.
        
        This is a SOURCE operation - creates initial displacement
        that will propagate according to the wave equation.
        
        Args:
            center: Tuple of center coordinates
            sigma: Width of Gaussian
            amplitude: Peak amplitude
            channel: Target channel
        """
        grids = self._get_coord_grids()
        dist_squared = sum((g - c)**2 for g, c in zip(grids, center))
        pulse = amplitude * np.exp(-dist_squared / (2 * sigma**2))
        self.amplitude[channel] += pulse
    
    def inject_velocity_pulse(self, center, sigma, velocity_amplitude, channel=0):
        """
        Inject a velocity pulse (impulse) at specified location.
        
        This creates outgoing waves from the point without initial displacement.
        
        Args:
            center: Tuple of center coordinates
            sigma: Width of Gaussian
            velocity_amplitude: Peak velocity
            channel: Target channel
        """
        grids = self._get_coord_grids()
        dist_squared = sum((g - c)**2 for g, c in zip(grids, center))
        pulse = velocity_amplitude * np.exp(-dist_squared / (2 * sigma**2))
        self.velocity[channel] += pulse
    
    def compute_kinetic_energy(self, channel=None):
        """
        Compute kinetic energy: E_k = 0.5 * integral(rho * v^2).
        
        Args:
            channel: Specific channel (None = sum over all)
        """
        if channel is not None:
            return 0.5 * np.sum(self._density * self.velocity[channel]**2)
        
        total = 0.0
        for c in range(self.channels):
            total += 0.5 * np.sum(self._density * self.velocity[c]**2)
        return total
    
    def compute_potential_energy(self, channel=None):
        """
        Compute potential energy: E_p = 0.5 * integral(c^2 * |grad(u)|^2).
        
        Args:
            channel: Specific channel (None = sum over all)
        """
        def channel_potential(c):
            grad_squared = np.zeros(self.dimensions, dtype=DTYPE)
            for axis in range(self.ndim):
                grad = np.gradient(self.amplitude[c], axis=axis)
                grad_squared += grad**2
            return 0.5 * np.sum(self._wave_speed**2 * grad_squared)
        
        if channel is not None:
            return channel_potential(channel)
        
        return sum(channel_potential(c) for c in range(self.channels))
    
    def compute_total_energy(self, channel=None):
        """
        Compute total energy: E = E_k + E_p.
        
        In undamped systems, this should be conserved.
        With damping, energy decreases over time.
        """
        kinetic = self.compute_kinetic_energy(channel)
        potential = self.compute_potential_energy(channel)
        total = kinetic + potential
        self.energy_history.append(total)
        return total
    
    def compute_amplitude_rms(self, channel=0):
        """Root-mean-square amplitude."""
        return np.sqrt(np.mean(self.amplitude[channel]**2))
    
    def compute_velocity_rms(self, channel=0):
        """Root-mean-square velocity."""
        return np.sqrt(np.mean(self.velocity[channel]**2))
    
    def get_phase_space_point(self, channel=0):
        """
        Return (amplitude, velocity) pair for phase space analysis.
        
        The full state of a wave system is the point in phase space
        defined by (u, du/dt) at every spatial location.
        """
        return self.amplitude[channel].copy(), self.velocity[channel].copy()
    
    def to_resonant_field(self):
        """
        Convert wave state to ResonantFieldND for compatibility.
        
        Maps: channel 0 -> amplitude, channel 1 -> velocity
        """
        field = ResonantFieldND(self.dimensions, channels=2 * self.channels)
        for c in range(self.channels):
            field.data[2*c] = self.amplitude[c]
            field.data[2*c + 1] = self.velocity[c]
        field.boundary = self.boundary
        return field
    
    @classmethod
    def from_resonant_field(cls, field):
        """
        Create WaveState from ResonantFieldND.
        
        Assumes even channels are amplitude, odd channels are velocity.
        """
        channels = field.channels // 2
        state = cls(field.dimensions, channels)
        for c in range(channels):
            state.amplitude[c] = field.data[2*c]
            state.velocity[c] = field.data[2*c + 1]
        state.boundary = field.boundary
        return state
    
    def copy(self):
        """Create deep copy of wave state."""
        new_state = WaveState(self.dimensions, self.channels)
        new_state.amplitude = self.amplitude.copy()
        new_state.velocity = self.velocity.copy()
        new_state._wave_speed = self._wave_speed.copy()
        new_state._damping = self._damping.copy()
        new_state._density = self._density.copy()
        new_state.boundary = self.boundary
        new_state.time = self.time
        new_state.energy_history = self.energy_history.copy()
        return new_state
    
    def __repr__(self):
        return (f"WaveState(dimensions={self.dimensions}, channels={self.channels}, "
                f"time={self.time:.4f})")


def create_wave_state_1d(size, channels=1):
    """Create 1D wave state."""
    return WaveState(dimensions=(size,), channels=channels)


def create_wave_state_2d(height, width, channels=1):
    """Create 2D wave state."""
    return WaveState(dimensions=(height, width), channels=channels)


def create_wave_state_3d(depth, height, width, channels=1):
    """Create 3D wave state."""
    return WaveState(dimensions=(depth, height, width), channels=channels)
