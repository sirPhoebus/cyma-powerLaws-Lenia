# Phase Space Tracking - Phase 2.11
#
# Track system state in phase space for dynamical analysis:
#   - field_snapshot: u(x)
#   - velocity_snapshot: du/dt(x)
#   - total_energy: E = 0.5 * integral(rho * (du/dt)^2 + c^2 * |grad(u)|^2)
#   - modal_amplitudes: Projection onto eigenmodes

import numpy as np
from src.config import DTYPE


class PhaseSpaceState:
    """
    Complete phase space state of the wave system.
    
    Stores (u, du/dt) pair representing the full dynamical state.
    """
    
    def __init__(self, field_snapshot, velocity_snapshot, time=0.0,
                 wave_speed=1.0, density=1.0):
        """
        Args:
            field_snapshot: Amplitude field u(x)
            velocity_snapshot: Velocity field du/dt(x)
            time: Time of snapshot
            wave_speed: Wave speed for energy calculation
            density: Medium density
        """
        self.field = np.array(field_snapshot, dtype=DTYPE)
        self.velocity = np.array(velocity_snapshot, dtype=DTYPE)
        self.time = time
        self.wave_speed = wave_speed
        self.density = density
        
        self.dimensions = self.field.shape
        self.ndim = len(self.dimensions)
    
    def compute_kinetic_energy(self):
        """E_k = 0.5 * integral(rho * v^2)"""
        return 0.5 * self.density * np.sum(self.velocity**2)
    
    def compute_potential_energy(self):
        """E_p = 0.5 * integral(c^2 * |grad(u)|^2)"""
        grad_squared = np.zeros_like(self.field)
        for axis in range(self.ndim):
            grad = np.gradient(self.field, axis=axis)
            grad_squared += grad**2
        return 0.5 * self.wave_speed**2 * np.sum(grad_squared)
    
    def compute_total_energy(self):
        """E = E_k + E_p"""
        return self.compute_kinetic_energy() + self.compute_potential_energy()
    
    def compute_modal_amplitudes(self, modes):
        """
        Project state onto given modes.
        
        Args:
            modes: List of spatial mode patterns
            
        Returns:
            List of (amplitude_u, amplitude_v) for each mode
        """
        amplitudes = []
        for mode in modes:
            # Normalize mode
            mode_norm = mode / (np.linalg.norm(mode) + 1e-10)
            
            # Project field and velocity
            amp_u = np.sum(self.field * mode_norm)
            amp_v = np.sum(self.velocity * mode_norm)
            
            amplitudes.append((amp_u, amp_v))
        
        return amplitudes
    
    def distance_to(self, other):
        """
        Compute phase space distance to another state.
        
        Uses L2 norm: sqrt(||u1-u2||^2 + ||v1-v2||^2)
        """
        du = self.field - other.field
        dv = self.velocity - other.velocity
        return np.sqrt(np.sum(du**2) + np.sum(dv**2))
    
    def copy(self):
        """Create copy of state."""
        return PhaseSpaceState(
            self.field.copy(),
            self.velocity.copy(),
            self.time,
            self.wave_speed,
            self.density
        )


class PhaseSpaceTracker:
    """
    Track system evolution through phase space.
    
    Records trajectory and computes dynamical properties.
    """
    
    def __init__(self, wave_speed=1.0, density=1.0, max_history=1000):
        """
        Args:
            wave_speed: Wave speed for energy calculations
            density: Medium density
            max_history: Maximum trajectory length
        """
        self.wave_speed = wave_speed
        self.density = density
        self.max_history = max_history
        
        self.trajectory = []
        self.energy_history = []
        self.time_history = []
    
    def record(self, field, velocity, time):
        """
        Record a phase space point.
        
        Args:
            field: Current amplitude field
            velocity: Current velocity field
            time: Current time
        """
        state = PhaseSpaceState(field, velocity, time, 
                                self.wave_speed, self.density)
        
        self.trajectory.append(state)
        self.energy_history.append(state.compute_total_energy())
        self.time_history.append(time)
        
        # Limit history size
        if len(self.trajectory) > self.max_history:
            self.trajectory.pop(0)
            self.energy_history.pop(0)
            self.time_history.pop(0)
    
    def clear(self):
        """Clear recorded trajectory."""
        self.trajectory.clear()
        self.energy_history.clear()
        self.time_history.clear()
    
    def get_energy_trace(self):
        """Return energy over time."""
        return np.array(self.time_history), np.array(self.energy_history)
    
    def compute_energy_conservation(self):
        """
        Measure energy conservation quality.
        
        Returns relative standard deviation of energy.
        """
        if len(self.energy_history) < 2:
            return 0.0
        
        energies = np.array(self.energy_history)
        mean_e = np.mean(energies)
        if mean_e < 1e-10:
            return 0.0
        
        return np.std(energies) / mean_e
    
    def compute_lyapunov_estimate(self, perturbation_size=0.01):
        """
        Estimate Lyapunov exponent from trajectory divergence.
        
        Positive = chaotic, Negative = stable, Zero = periodic
        """
        if len(self.trajectory) < 10:
            return None
        
        # Use distance between consecutive states
        distances = []
        for i in range(1, len(self.trajectory)):
            d = self.trajectory[i].distance_to(self.trajectory[i-1])
            if d > 1e-10:
                distances.append(np.log(d))
        
        if len(distances) < 2:
            return 0.0
        
        # Linear fit to log(distance) gives Lyapunov exponent
        times = np.arange(len(distances))
        coeffs = np.polyfit(times, distances, 1)
        
        return coeffs[0]  # Slope
    
    def find_periodic_orbit(self, tolerance=0.1):
        """
        Detect if trajectory is periodic.
        
        Returns period if periodic, None otherwise.
        """
        if len(self.trajectory) < 20:
            return None
        
        current = self.trajectory[-1]
        
        # Search for recurrence
        for i in range(len(self.trajectory) - 10):
            dist = current.distance_to(self.trajectory[i])
            if dist < tolerance:
                period = self.time_history[-1] - self.time_history[i]
                return period
        
        return None
    
    def compute_phase_portrait_2d(self, point_index):
        """
        Extract 2D phase portrait at a single spatial point.
        
        Args:
            point_index: Tuple index into spatial field
            
        Returns:
            u_values, v_values arrays for plotting
        """
        u_vals = [state.field[point_index] for state in self.trajectory]
        v_vals = [state.velocity[point_index] for state in self.trajectory]
        return np.array(u_vals), np.array(v_vals)
    
    def compute_poincare_section(self, crossing_value=0.0, channel=None):
        """
        Compute Poincare section (crossings of a hyperplane).
        
        Args:
            crossing_value: Value to detect crossings through
            channel: Which spatial point to monitor (center if None)
            
        Returns:
            List of (velocity, time) at crossings
        """
        if len(self.trajectory) < 2:
            return []
        
        if channel is None:
            # Use center point
            center = tuple(d // 2 for d in self.trajectory[0].dimensions)
            channel = center
        
        crossings = []
        for i in range(1, len(self.trajectory)):
            u_prev = self.trajectory[i-1].field[channel]
            u_curr = self.trajectory[i].field[channel]
            
            # Check for upward crossing
            if u_prev < crossing_value <= u_curr:
                v = self.trajectory[i].velocity[channel]
                t = self.time_history[i]
                crossings.append((v, t))
        
        return crossings


class ModalProjector:
    """
    Project wave state onto eigenmode basis.
    
    Useful for analyzing which modes are excited.
    """
    
    def __init__(self, dimensions):
        """
        Args:
            dimensions: Spatial dimensions
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.modes = {}
        self._computed_modes = {}
    
    def compute_cartesian_mode(self, mode_numbers):
        """
        Compute Cartesian eigenmode.
        
        phi_n(x) = prod_i sin(n_i * pi * x_i / L_i)
        """
        key = ('cartesian', mode_numbers)
        if key in self._computed_modes:
            return self._computed_modes[key]
        
        grids = np.ogrid[tuple(slice(0, d) for d in self.dimensions)]
        mode = np.ones(self.dimensions, dtype=DTYPE)
        
        for n, g, L in zip(mode_numbers, grids, self.dimensions):
            mode *= np.sin(n * np.pi * g / L)
        
        # Normalize
        mode /= np.linalg.norm(mode)
        
        self._computed_modes[key] = mode
        return mode
    
    def add_mode(self, name, mode_numbers):
        """Add a mode to track by name."""
        self.modes[name] = mode_numbers
    
    def add_modes_up_to(self, max_mode):
        """Add all Cartesian modes up to max_mode in each dimension."""
        for nx in range(1, max_mode + 1):
            for ny in range(1, max_mode + 1):
                if self.ndim == 2:
                    name = f"mode_{nx}_{ny}"
                    self.modes[name] = (ny, nx)
                elif self.ndim == 3:
                    for nz in range(1, max_mode + 1):
                        name = f"mode_{nx}_{ny}_{nz}"
                        self.modes[name] = (nz, ny, nx)
    
    def project(self, field):
        """
        Project field onto all registered modes.
        
        Returns dict of mode_name -> amplitude
        """
        amplitudes = {}
        for name, mode_numbers in self.modes.items():
            mode = self.compute_cartesian_mode(mode_numbers)
            amplitudes[name] = np.sum(field * mode)
        return amplitudes
    
    def reconstruct(self, amplitudes):
        """
        Reconstruct field from modal amplitudes.
        
        Args:
            amplitudes: Dict of mode_name -> amplitude
            
        Returns:
            Reconstructed field
        """
        field = np.zeros(self.dimensions, dtype=DTYPE)
        for name, amp in amplitudes.items():
            if name in self.modes:
                mode = self.compute_cartesian_mode(self.modes[name])
                field += amp * mode
        return field
    
    def compute_modal_energy(self, field, velocity):
        """
        Compute energy in each mode.
        
        E_n = 0.5 * (a_n^2 + b_n^2) where a_n, b_n are field/velocity projections
        """
        field_proj = self.project(field)
        velocity_proj = self.project(velocity)
        
        energies = {}
        for name in self.modes:
            a = field_proj.get(name, 0.0)
            b = velocity_proj.get(name, 0.0)
            energies[name] = 0.5 * (a**2 + b**2)
        
        return energies
