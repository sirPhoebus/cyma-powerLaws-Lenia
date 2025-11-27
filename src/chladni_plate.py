# Chladni Plate Dynamics - Phase 2.3
#
# Implements the biharmonic plate equation for vibrating plates:
#   rho * h * d2w/dt2 = -D * Laplacian(Laplacian(w)) - gamma * dw/dt + F(x,t)
#
# Where:
#   w = vertical displacement
#   D = flexural rigidity = E * h^3 / (12 * (1 - nu^2))
#   E = Young's modulus
#   h = plate thickness
#   nu = Poisson ratio
#   rho = density
#   gamma = damping
#
# Modal frequency (generalized Chladni's Law):
#   f_mn = (pi / 2) * sqrt(D / (rho * h)) * ((m/L_x)^2 + (n/L_y)^2)

import numpy as np
from src.wave_state import WaveState
from src.config import (
    DTYPE, WAVE_DT,
    PLATE_THICKNESS, YOUNGS_MODULUS, POISSON_RATIO, FLEXURAL_RIGIDITY,
    WAVE_DENSITY, WAVE_DAMPING,
    BOUNDARY_PERIODIC, BOUNDARY_NEUMANN, BOUNDARY_DIRICHLET
)


class PlateState(WaveState):
    """
    Plate State - Extension of WaveState for plate dynamics.
    
    Adds plate-specific material properties.
    """
    
    def __init__(self, dimensions, channels=1):
        super().__init__(dimensions, channels)
        
        # Plate material properties
        self._thickness = PLATE_THICKNESS
        self._youngs_modulus = YOUNGS_MODULUS
        self._poisson_ratio = POISSON_RATIO
        self._flexural_rigidity = None  # Computed from other properties
        
        self._update_flexural_rigidity()
    
    def _update_flexural_rigidity(self):
        """Compute flexural rigidity D = E * h^3 / (12 * (1 - nu^2))."""
        E = self._youngs_modulus
        h = self._thickness
        nu = self._poisson_ratio
        self._flexural_rigidity = E * h**3 / (12 * (1 - nu**2))
    
    @property
    def thickness(self):
        return self._thickness
    
    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        self._update_flexural_rigidity()
    
    @property
    def youngs_modulus(self):
        return self._youngs_modulus
    
    @youngs_modulus.setter
    def youngs_modulus(self, value):
        self._youngs_modulus = value
        self._update_flexural_rigidity()
    
    @property
    def poisson_ratio(self):
        return self._poisson_ratio
    
    @poisson_ratio.setter
    def poisson_ratio(self, value):
        self._poisson_ratio = value
        self._update_flexural_rigidity()
    
    @property
    def flexural_rigidity(self):
        return self._flexural_rigidity
    
    def compute_modal_frequency(self, m, n, L_x=None, L_y=None):
        """
        Compute modal frequency using generalized Chladni's Law.
        
        f_mn = (pi / 2) * sqrt(D / (rho * h)) * ((m/L_x)^2 + (n/L_y)^2)
        
        Args:
            m, n: Mode numbers (integers >= 1)
            L_x, L_y: Plate dimensions (defaults to grid dimensions)
            
        Returns:
            Frequency in Hz
        """
        if L_x is None:
            L_x = self.dimensions[-1]
        if L_y is None:
            L_y = self.dimensions[-2] if self.ndim >= 2 else L_x
        
        D = self._flexural_rigidity
        rho = np.mean(self._density)
        h = self._thickness
        
        coeff = (np.pi / 2) * np.sqrt(D / (rho * h))
        spatial_term = (m / L_x)**2 + (n / L_y)**2
        
        return coeff * spatial_term
    
    def get_resonant_frequencies(self, max_mode=5, L_x=None, L_y=None):
        """
        Get list of resonant frequencies for modes up to max_mode.
        
        Returns:
            List of (m, n, frequency) tuples sorted by frequency
        """
        modes = []
        for m in range(1, max_mode + 1):
            for n in range(1, max_mode + 1):
                f = self.compute_modal_frequency(m, n, L_x, L_y)
                modes.append((m, n, f))
        
        modes.sort(key=lambda x: x[2])
        return modes


class BiharmonicOperator:
    """
    Biharmonic operator: Laplacian(Laplacian(u)) = nabla^4(u)
    
    For 2D: d4u/dx4 + 2*d4u/dx2dy2 + d4u/dy4
    """
    
    def __init__(self, ndim, dx=1.0, boundary=BOUNDARY_PERIODIC):
        self.ndim = ndim
        self.dx = dx
        self.boundary = boundary
        
        # 1D fourth derivative stencil (second-order accurate)
        # d4u/dx4 ~ (u[i-2] - 4*u[i-1] + 6*u[i] - 4*u[i+1] + u[i+2]) / dx^4
        self.stencil_4th = np.array([1, -4, 6, -4, 1])
        
        # 1D second derivative stencil
        self.stencil_2nd = np.array([1, -2, 1])
    
    def _apply_1d_stencil(self, u, stencil, axis):
        """Apply 1D stencil along specified axis with boundary handling."""
        half_width = len(stencil) // 2
        pad_width = [(0, 0)] * u.ndim
        pad_width[axis] = (half_width, half_width)
        
        if self.boundary == BOUNDARY_PERIODIC:
            u_padded = np.pad(u, pad_width, mode='wrap')
        elif self.boundary == BOUNDARY_NEUMANN:
            u_padded = np.pad(u, pad_width, mode='reflect')
        else:
            u_padded = np.pad(u, pad_width, mode='constant', constant_values=0)
        
        result = np.zeros_like(u_padded)
        for i, coeff in enumerate(stencil):
            offset = i - half_width
            if offset == 0:
                result += coeff * u_padded
            else:
                result += coeff * np.roll(u_padded, -offset, axis=axis)
        
        slices = [slice(None)] * u.ndim
        slices[axis] = slice(half_width, -half_width)
        return result[tuple(slices)]
    
    def compute(self, u):
        """
        Compute biharmonic: nabla^4(u) = Laplacian(Laplacian(u))
        
        For 2D: d4u/dx4 + 2*d4u/dx2dy2 + d4u/dy4
        """
        dx4 = self.dx ** 4
        
        if self.ndim == 1:
            # 1D: just d4u/dx4
            return self._apply_1d_stencil(u, self.stencil_4th, 0) / dx4
        
        elif self.ndim == 2:
            # 2D biharmonic: d4u/dx4 + 2*d4u/dx2dy2 + d4u/dy4
            
            # d4u/dx4 (along axis 1)
            d4_dx4 = self._apply_1d_stencil(u, self.stencil_4th, 1) / dx4
            
            # d4u/dy4 (along axis 0)
            d4_dy4 = self._apply_1d_stencil(u, self.stencil_4th, 0) / dx4
            
            # d4u/dx2dy2 = d2/dx2(d2u/dy2)
            d2_dy2 = self._apply_1d_stencil(u, self.stencil_2nd, 0) / (self.dx ** 2)
            d4_dx2dy2 = self._apply_1d_stencil(d2_dy2, self.stencil_2nd, 1) / (self.dx ** 2)
            
            return d4_dx4 + 2 * d4_dx2dy2 + d4_dy4
        
        else:
            # N-D: sum of all fourth derivatives and mixed terms
            # For simplicity, compute as Laplacian(Laplacian(u))
            laplacian_u = self._compute_laplacian(u)
            return self._compute_laplacian(laplacian_u)
    
    def _compute_laplacian(self, u):
        """Compute Laplacian for N-D case."""
        result = np.zeros_like(u)
        dx2 = self.dx ** 2
        for axis in range(self.ndim):
            result += self._apply_1d_stencil(u, self.stencil_2nd, axis) / dx2
        return result


class ChladniPlate:
    """
    Chladni Plate Simulator.
    
    Solves: rho * h * d2w/dt2 = -D * nabla^4(w) - gamma * dw/dt + F(x,t)
    
    Supports:
      - Eigenmode computation
      - Frequency-driven excitation
      - Visualization of nodal patterns
    """
    
    def __init__(self, state, dt=WAVE_DT):
        """
        Initialize Chladni plate simulator.
        
        Args:
            state: PlateState instance
            dt: Time step
        """
        self.state = state
        self.dt = dt
        
        # Biharmonic operator
        self.biharmonic = BiharmonicOperator(
            ndim=state.ndim,
            dx=1.0,
            boundary=state.boundary
        )
        
        # Source function
        self._source_fn = None
        
        # Leapfrog velocity half-step
        self._velocity_half = None
    
    def set_source(self, source_fn):
        """Set driving force function F(x,t)."""
        self._source_fn = source_fn
    
    def set_point_driver(self, position, frequency, amplitude):
        """
        Set up a point oscillator at specified position.
        
        Args:
            position: (y, x) tuple for 2D
            frequency: Driving frequency in Hz
            amplitude: Oscillation amplitude
        """
        omega = 2 * np.pi * frequency
        
        def source_fn(state, t):
            source = np.zeros(state.shape, dtype=DTYPE)
            # Create small driving region around position
            slices = []
            for i, p in enumerate(position):
                slices.append(slice(max(0, p-1), min(state.dimensions[i], p+2)))
            source[(0,) + tuple(slices)] = amplitude * np.sin(omega * t)
            return source
        
        self._source_fn = source_fn
    
    def set_edge_driver(self, edge, frequency, amplitude):
        """
        Set up edge excitation (like bowing a plate edge).
        
        Args:
            edge: 'top', 'bottom', 'left', 'right'
            frequency: Driving frequency
            amplitude: Oscillation amplitude
        """
        omega = 2 * np.pi * frequency
        
        def source_fn(state, t):
            source = np.zeros(state.shape, dtype=DTYPE)
            signal = amplitude * np.sin(omega * t)
            
            if edge == 'top':
                source[0, 0:2, :] = signal
            elif edge == 'bottom':
                source[0, -2:, :] = signal
            elif edge == 'left':
                source[0, :, 0:2] = signal
            elif edge == 'right':
                source[0, :, -2:] = signal
            
            return source
        
        self._source_fn = source_fn
    
    def compute_acceleration(self, amplitude, velocity, t):
        """
        Compute d2w/dt2 = -(D / (rho*h)) * nabla^4(w) - (gamma/(rho*h)) * dw/dt + F/(rho*h)
        """
        acceleration = np.zeros_like(amplitude)
        
        rho = np.mean(self.state.density)
        h = self.state.thickness
        D = self.state.flexural_rigidity
        gamma = np.mean(self.state.damping)
        
        mass_density = rho * h  # Mass per unit area
        
        for c in range(self.state.channels):
            # Biharmonic term: -D * nabla^4(w)
            biharmonic = self.biharmonic.compute(amplitude[c])
            plate_term = -(D / mass_density) * biharmonic
            
            # Damping term: -gamma * dw/dt
            damping_term = -(gamma / mass_density) * velocity[c]
            
            acceleration[c] = plate_term + damping_term
        
        # Add source term
        if self._source_fn is not None:
            source = self._source_fn(self.state, t)
            acceleration += source / mass_density
        
        return acceleration
    
    def step_leapfrog(self):
        """Leapfrog integration for plate equation."""
        t = self.state.time
        
        if self._velocity_half is None:
            accel = self.compute_acceleration(
                self.state.amplitude,
                self.state.velocity,
                t
            )
            self._velocity_half = self.state.velocity + 0.5 * self.dt * accel
        
        self.state.amplitude += self.dt * self._velocity_half
        
        accel = self.compute_acceleration(
            self.state.amplitude,
            self._velocity_half,
            t + self.dt
        )
        
        self._velocity_half += self.dt * accel
        self.state.velocity = self._velocity_half - 0.5 * self.dt * accel
        
        self.state.time += self.dt
    
    def step(self):
        """Advance one time step."""
        self.step_leapfrog()
        self._apply_boundary()
    
    def _apply_boundary(self):
        """Apply plate boundary conditions."""
        boundary = self.state.boundary
        
        if boundary == BOUNDARY_DIRICHLET:
            # Clamped edges: w = 0 and dw/dn = 0
            for c in range(self.state.channels):
                for axis in range(self.state.ndim):
                    for edge in [0, -1]:
                        idx = [slice(None)] * self.state.ndim
                        idx[axis] = edge
                        self.state.amplitude[c][tuple(idx)] = 0.0
                        self.state.velocity[c][tuple(idx)] = 0.0
        
        elif boundary == BOUNDARY_NEUMANN:
            # Free edges: handled by reflect padding in biharmonic
            pass
    
    def run(self, steps, callback=None):
        """Run simulation for multiple steps."""
        for i in range(steps):
            self.step()
            if callback is not None:
                callback(self.state, i)
    
    def find_nodes(self, threshold=0.1, channel=0):
        """
        Find nodal lines (regions where amplitude ~ 0).
        
        Args:
            threshold: Amplitude threshold for node detection
            channel: Which channel to analyze
            
        Returns:
            Boolean mask where True indicates nodal regions
        """
        amp = np.abs(self.state.amplitude[channel])
        max_amp = np.max(amp)
        if max_amp > 0:
            normalized = amp / max_amp
            return normalized < threshold
        return np.ones_like(amp, dtype=bool)
    
    def compute_mode_overlap(self, m, n, channel=0):
        """
        Compute overlap with theoretical mode shape.
        
        For rectangular plate with simply supported edges:
        phi_mn(x,y) = sin(m*pi*x/L_x) * sin(n*pi*y/L_y)
        
        Returns correlation coefficient.
        """
        if self.state.ndim != 2:
            raise ValueError("Mode overlap only for 2D plates")
        
        Ly, Lx = self.state.dimensions
        y = np.arange(Ly) / Ly
        x = np.arange(Lx) / Lx
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        mode = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
        
        amp = self.state.amplitude[channel]
        
        # Normalize both
        amp_norm = amp / (np.linalg.norm(amp) + 1e-10)
        mode_norm = mode / np.linalg.norm(mode)
        
        # Correlation (absolute value since phase can flip)
        return abs(np.sum(amp_norm * mode_norm))


class ChladniSimulation:
    """
    High-level Chladni plate simulation.
    
    Convenience wrapper for running Chladni experiments.
    """
    
    def __init__(self, size, thickness=PLATE_THICKNESS, 
                 youngs_modulus=YOUNGS_MODULUS, poisson_ratio=POISSON_RATIO,
                 density=WAVE_DENSITY, damping=0.001, dt=0.01):
        """
        Initialize Chladni simulation.
        
        Args:
            size: Grid size (square plate)
            thickness: Plate thickness h
            youngs_modulus: Young's modulus E
            poisson_ratio: Poisson ratio nu
            density: Material density rho
            damping: Damping coefficient
            dt: Time step
        """
        self.state = PlateState(dimensions=(size, size), channels=1)
        self.state.thickness = thickness
        self.state.youngs_modulus = youngs_modulus
        self.state.poisson_ratio = poisson_ratio
        self.state.density = density
        self.state.damping = damping
        self.state.boundary = BOUNDARY_DIRICHLET  # Clamped edges
        
        self.plate = ChladniPlate(self.state, dt=dt)
        self.step_count = 0
    
    def drive_at_frequency(self, frequency, amplitude=1.0, position=None):
        """
        Set up frequency-driven excitation.
        
        Args:
            frequency: Driving frequency
            amplitude: Driving amplitude
            position: Driver position (default: center)
        """
        if position is None:
            size = self.state.dimensions[0]
            position = (size // 2, size // 2)
        
        self.plate.set_point_driver(position, frequency, amplitude)
    
    def drive_at_mode(self, m, n, amplitude=1.0, position=None):
        """
        Drive plate at modal frequency (m, n).
        
        This should excite the corresponding Chladni pattern.
        """
        freq = self.state.compute_modal_frequency(m, n)
        self.drive_at_frequency(freq, amplitude, position)
        return freq
    
    def step(self):
        """Advance one time step."""
        self.plate.step()
        self.step_count += 1
    
    def run(self, steps):
        """Run simulation for multiple steps."""
        for _ in range(steps):
            self.step()
    
    def get_amplitude(self):
        """Get current amplitude field."""
        return self.state.amplitude[0]
    
    def get_nodes(self, threshold=0.1):
        """Get nodal pattern (Chladni figure)."""
        return self.plate.find_nodes(threshold)
    
    def get_resonant_frequencies(self, max_mode=5):
        """Get list of resonant frequencies."""
        return self.state.get_resonant_frequencies(max_mode)
