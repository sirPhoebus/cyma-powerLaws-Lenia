# Wave Source Primitives - Phase 2.10
#
# Source types for energy injection:
#   - point_oscillator: Single oscillating point
#   - line_driver: Uniform oscillation along line
#   - ring_source: Circular wave emitter
#   - boundary_bow: Edge excitation (Chladni bow)
#   - volume_pulse: Impulse within 3D region

import numpy as np
from src.config import DTYPE


class WaveSource:
    """Base class for wave sources."""
    
    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.phase = phase
        self.active = True
    
    def get_signal(self, t):
        """Get temporal signal at time t."""
        if not self.active:
            return 0.0
        return self.amplitude * np.sin(self.omega * t + self.phase)
    
    def generate(self, dimensions, t):
        """Generate source field. Override in subclasses."""
        raise NotImplementedError


class PointOscillator(WaveSource):
    """
    Single oscillating point source.
    
    A * sin(2*pi*f*t) * delta(x - x0)
    """
    
    def __init__(self, position, amplitude=1.0, frequency=1.0, phase=0.0, sigma=2.0):
        """
        Args:
            position: Center position tuple
            amplitude: Oscillation amplitude
            frequency: Oscillation frequency
            phase: Initial phase
            sigma: Source width (Gaussian spread)
        """
        super().__init__(amplitude, frequency, phase)
        self.position = tuple(position)
        self.sigma = sigma
        self._mask = None
        self._dimensions = None
    
    def _compute_mask(self, dimensions):
        """Compute Gaussian mask centered at position."""
        if self._mask is not None and self._dimensions == dimensions:
            return self._mask
        
        grids = np.ogrid[tuple(slice(0, d) for d in dimensions)]
        dist_sq = sum((g - p)**2 for g, p in zip(grids, self.position))
        
        self._mask = np.exp(-dist_sq / (2 * self.sigma**2))
        self._dimensions = dimensions
        return self._mask
    
    def generate(self, dimensions, t):
        """Generate source field at time t."""
        mask = self._compute_mask(dimensions)
        signal = self.get_signal(t)
        return signal * mask


class LineDriver(WaveSource):
    """
    Uniform oscillation along line segment.
    
    For 2D: line from (y1, x1) to (y2, x2)
    """
    
    def __init__(self, start, end, amplitude=1.0, frequency=1.0, phase=0.0, width=2.0):
        """
        Args:
            start: Start position
            end: End position
            amplitude: Oscillation amplitude
            frequency: Oscillation frequency
            phase: Initial phase
            width: Line width
        """
        super().__init__(amplitude, frequency, phase)
        self.start = np.array(start, dtype=DTYPE)
        self.end = np.array(end, dtype=DTYPE)
        self.width = width
        self._mask = None
        self._dimensions = None
    
    def _compute_mask(self, dimensions):
        """Compute line mask."""
        if self._mask is not None and self._dimensions == dimensions:
            return self._mask
        
        ndim = len(dimensions)
        grids = np.ogrid[tuple(slice(0, d) for d in dimensions)]
        coords = np.stack([g * np.ones(dimensions) for g in grids], axis=-1)
        
        # Vector from start to end
        line_vec = self.end - self.start
        line_length = np.linalg.norm(line_vec)
        if line_length < 1e-10:
            line_length = 1.0
        line_unit = line_vec / line_length
        
        # Vector from start to each point
        to_point = coords - self.start
        
        # Project onto line
        proj_length = np.sum(to_point * line_unit, axis=-1)
        proj_length = np.clip(proj_length, 0, line_length)
        
        # Closest point on line
        closest = self.start + proj_length[..., np.newaxis] * line_unit
        
        # Distance from line
        dist = np.linalg.norm(coords - closest, axis=-1)
        
        self._mask = np.exp(-dist**2 / (2 * self.width**2))
        self._dimensions = dimensions
        return self._mask
    
    def generate(self, dimensions, t):
        """Generate source field at time t."""
        mask = self._compute_mask(dimensions)
        signal = self.get_signal(t)
        return signal * mask


class RingSource(WaveSource):
    """
    Circular/spherical ring source.
    
    Emits from a ring (2D) or spherical shell (3D).
    """
    
    def __init__(self, center, radius, amplitude=1.0, frequency=1.0, 
                 phase=0.0, ring_width=2.0):
        """
        Args:
            center: Center position
            radius: Ring radius
            amplitude: Oscillation amplitude
            frequency: Oscillation frequency
            phase: Initial phase
            ring_width: Ring thickness
        """
        super().__init__(amplitude, frequency, phase)
        self.center = tuple(center)
        self.radius = radius
        self.ring_width = ring_width
        self._mask = None
        self._dimensions = None
    
    def _compute_mask(self, dimensions):
        """Compute ring mask."""
        if self._mask is not None and self._dimensions == dimensions:
            return self._mask
        
        grids = np.ogrid[tuple(slice(0, d) for d in dimensions)]
        dist_sq = sum((g - c)**2 for g, c in zip(grids, self.center))
        dist = np.sqrt(dist_sq)
        
        # Gaussian ring profile
        self._mask = np.exp(-((dist - self.radius)**2) / (2 * self.ring_width**2))
        self._dimensions = dimensions
        return self._mask
    
    def generate(self, dimensions, t):
        """Generate source field at time t."""
        mask = self._compute_mask(dimensions)
        signal = self.get_signal(t)
        return signal * mask


class BoundaryBow(WaveSource):
    """
    Edge excitation source (like bowing a Chladni plate).
    
    Applies force along an edge of the domain.
    """
    
    def __init__(self, edge, amplitude=1.0, frequency=1.0, phase=0.0, 
                 depth=3, profile="uniform"):
        """
        Args:
            edge: Which edge ('top', 'bottom', 'left', 'right', 
                  'front', 'back' for 3D)
            amplitude: Excitation amplitude
            frequency: Excitation frequency
            phase: Initial phase
            depth: Penetration depth into domain
            profile: 'uniform', 'gaussian', or 'sine'
        """
        super().__init__(amplitude, frequency, phase)
        self.edge = edge
        self.depth = depth
        self.profile = profile
        self._mask = None
        self._dimensions = None
    
    def _compute_mask(self, dimensions):
        """Compute edge mask."""
        if self._mask is not None and self._dimensions == dimensions:
            return self._mask
        
        ndim = len(dimensions)
        mask = np.zeros(dimensions, dtype=DTYPE)
        
        # Determine axis and side
        edge_map = {
            'top': (0, 0),
            'bottom': (0, -1),
            'left': (1, 0),
            'right': (1, -1),
            'front': (2, 0) if ndim == 3 else None,
            'back': (2, -1) if ndim == 3 else None
        }
        
        axis_info = edge_map.get(self.edge)
        if axis_info is None:
            return mask
        
        axis, side = axis_info
        
        # Create edge region
        for d in range(self.depth):
            slices = [slice(None)] * ndim
            if side == 0:
                slices[axis] = d
            else:
                slices[axis] = -(d + 1)
            
            if self.profile == "uniform":
                value = 1.0
            elif self.profile == "gaussian":
                value = np.exp(-d**2 / (self.depth**2))
            elif self.profile == "sine":
                value = np.cos(0.5 * np.pi * d / self.depth)
            else:
                value = 1.0
            
            mask[tuple(slices)] = value
        
        self._mask = mask
        self._dimensions = dimensions
        return self._mask
    
    def generate(self, dimensions, t):
        """Generate source field at time t."""
        mask = self._compute_mask(dimensions)
        signal = self.get_signal(t)
        return signal * mask


class VolumePulse(WaveSource):
    """
    Impulse within 3D region.
    
    Single pulse or repeated pulses in a volume.
    """
    
    def __init__(self, center, size, pulse_times=None, amplitude=1.0, 
                 decay_time=0.1):
        """
        Args:
            center: Center of volume
            size: Size of volume (scalar or tuple)
            pulse_times: List of times to pulse (None = t=0 only)
            amplitude: Pulse amplitude
            decay_time: Pulse decay time constant
        """
        super().__init__(amplitude, frequency=0.0, phase=0.0)
        self.center = tuple(center)
        
        if isinstance(size, (int, float)):
            self.size = tuple([size] * len(center))
        else:
            self.size = tuple(size)
        
        self.pulse_times = pulse_times if pulse_times else [0.0]
        self.decay_time = decay_time
        self._mask = None
        self._dimensions = None
    
    def _compute_mask(self, dimensions):
        """Compute volume mask."""
        if self._mask is not None and self._dimensions == dimensions:
            return self._mask
        
        mask = np.zeros(dimensions, dtype=DTYPE)
        
        slices = []
        for c, s, d in zip(self.center, self.size, dimensions):
            start = max(0, int(c - s/2))
            end = min(d, int(c + s/2))
            slices.append(slice(start, end))
        
        mask[tuple(slices)] = 1.0
        
        self._mask = mask
        self._dimensions = dimensions
        return self._mask
    
    def generate(self, dimensions, t):
        """Generate source field at time t."""
        mask = self._compute_mask(dimensions)
        
        # Sum contributions from all pulses
        signal = 0.0
        for pulse_time in self.pulse_times:
            dt = t - pulse_time
            if dt >= 0:
                signal += self.amplitude * np.exp(-dt / self.decay_time)
        
        return signal * mask


class SourceManager:
    """
    Manages multiple wave sources.
    """
    
    def __init__(self, dimensions):
        """
        Args:
            dimensions: Field dimensions
        """
        self.dimensions = tuple(dimensions)
        self.sources = []
    
    def add_source(self, source):
        """Add a source to the manager."""
        self.sources.append(source)
        return len(self.sources) - 1
    
    def remove_source(self, index):
        """Remove source by index."""
        if 0 <= index < len(self.sources):
            self.sources.pop(index)
    
    def clear(self):
        """Remove all sources."""
        self.sources.clear()
    
    def generate_combined(self, t):
        """Generate combined source field from all active sources."""
        total = np.zeros(self.dimensions, dtype=DTYPE)
        
        for source in self.sources:
            if source.active:
                total += source.generate(self.dimensions, t)
        
        return total
    
    def create_source_function(self):
        """
        Create a source function for use with WaveEquation.
        
        Returns callable(state, t) -> source_field
        """
        def source_fn(state, t):
            field = self.generate_combined(t)
            # Reshape to match state shape (channels, spatial...)
            return field.reshape((1,) + field.shape)
        
        return source_fn
