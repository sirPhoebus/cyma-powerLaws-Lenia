# Wave Equation Core - Phase 2.2
#
# Implements the damped wave equation:
#   d2u/dt2 = c^2 * Laplacian(u) - gamma * du/dt + F(x,t)
#
# Rewritten as first-order system:
#   du/dt = v
#   dv/dt = c^2 * Laplacian(u) - gamma * v + F(x,t)
#
# Propagation is EMERGENT - we solve the equation, waves appear naturally.

import numpy as np
from scipy import ndimage
from src.wave_state import WaveState
from src.config import (
    DTYPE, WAVE_DT, WAVE_SPEED, WAVE_DAMPING,
    DEFAULT_STENCIL_ORDER, STENCIL_ORDER_2, STENCIL_ORDER_4, STENCIL_ORDER_6,
    BOUNDARY_PERIODIC, BOUNDARY_NEUMANN, BOUNDARY_DIRICHLET, BOUNDARY_ABSORBING
)


# Finite difference stencil coefficients for Laplacian
LAPLACIAN_STENCILS = {
    STENCIL_ORDER_2: {
        'coeffs': np.array([1.0, -2.0, 1.0]),
        'offsets': [-1, 0, 1]
    },
    STENCIL_ORDER_4: {
        'coeffs': np.array([-1/12, 4/3, -5/2, 4/3, -1/12]),
        'offsets': [-2, -1, 0, 1, 2]
    },
    STENCIL_ORDER_6: {
        'coeffs': np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]),
        'offsets': [-3, -2, -1, 0, 1, 2, 3]
    }
}


class WaveEquation:
    """
    Wave Equation Solver.
    
    Solves: d2u/dt2 = c^2 * Laplacian(u) - gamma * du/dt + F(x,t)
    
    Supports multiple time integration schemes:
      - leapfrog: Symplectic, energy-preserving for undamped waves
      - rk4: Fourth-order Runge-Kutta, accurate but more expensive
      - euler: Simple forward Euler (for testing only)
    """
    
    def __init__(self, state, dt=WAVE_DT, stencil_order=DEFAULT_STENCIL_ORDER):
        """
        Initialize wave equation solver.
        
        Args:
            state: WaveState instance to evolve
            dt: Time step size
            stencil_order: Finite difference accuracy (2, 4, or 6)
        """
        self.state = state
        self.dt = dt
        self.stencil_order = stencil_order
        
        # Get stencil coefficients
        stencil_info = LAPLACIAN_STENCILS[stencil_order]
        self.stencil_coeffs = stencil_info['coeffs']
        self.stencil_offsets = stencil_info['offsets']
        
        # Source function F(x,t) - can be set externally
        self._source_fn = None
        
        # Precompute grid spacing (assume uniform)
        self.dx = 1.0  # Can be modified for non-unit spacing
        
        # For leapfrog: store velocity at half-steps
        self._velocity_half = None
    
    def set_source(self, source_fn):
        """
        Set the driving force function F(x,t).
        
        Args:
            source_fn: Callable(state, t) -> array of shape (channels,) + dimensions
                       Returns the forcing term at each point
        """
        self._source_fn = source_fn
    
    def compute_laplacian(self, field, channel=0):
        """
        Compute the Laplacian using finite differences.
        
        For N dimensions: Laplacian(u) = sum_i d2u/dx_i^2
        
        Args:
            field: The field array (shape = (channels,) + dimensions)
            channel: Which channel to compute Laplacian for
            
        Returns:
            Laplacian array of shape = dimensions
        """
        u = field[channel]
        laplacian = np.zeros_like(u)
        
        # Sum second derivatives along each spatial axis
        for axis in range(self.state.ndim):
            laplacian += self._compute_1d_laplacian(u, axis)
        
        return laplacian / (self.dx ** 2)
    
    def _compute_1d_laplacian(self, u, axis):
        """Compute second derivative along one axis using stencil."""
        boundary = self.state.boundary
        
        if boundary == BOUNDARY_PERIODIC:
            # Use np.roll for periodic boundaries
            result = np.zeros_like(u)
            for coeff, offset in zip(self.stencil_coeffs, self.stencil_offsets):
                if offset == 0:
                    result += coeff * u
                else:
                    result += coeff * np.roll(u, -offset, axis=axis)
            return result
        
        # Non-periodic: use np.pad for boundary handling
        pad_width = [(0, 0)] * u.ndim
        max_offset = max(abs(o) for o in self.stencil_offsets)
        pad_width[axis] = (max_offset, max_offset)
        
        if boundary == BOUNDARY_NEUMANN:
            # Reflect for zero-flux boundaries
            u_padded = np.pad(u, pad_width, mode='reflect')
        elif boundary in (BOUNDARY_DIRICHLET, BOUNDARY_ABSORBING):
            # Zero padding for fixed boundaries
            u_padded = np.pad(u, pad_width, mode='constant', constant_values=0)
        else:
            # Default to periodic
            u_padded = np.pad(u, pad_width, mode='wrap')
        
        result = np.zeros_like(u_padded)
        for coeff, offset in zip(self.stencil_coeffs, self.stencil_offsets):
            if offset == 0:
                result += coeff * u_padded
            else:
                result += coeff * np.roll(u_padded, -offset, axis=axis)
        
        # Extract the valid region
        slices = [slice(None)] * u.ndim
        slices[axis] = slice(max_offset, -max_offset)
        return result[tuple(slices)]
    
    def compute_acceleration(self, amplitude, velocity, t):
        """
        Compute dv/dt = c^2 * Laplacian(u) - gamma * v + F(x,t).
        
        This is the core of the wave equation.
        
        Args:
            amplitude: Current amplitude field
            velocity: Current velocity field
            t: Current time
            
        Returns:
            Acceleration field (dv/dt)
        """
        acceleration = np.zeros_like(amplitude)
        
        for c in range(self.state.channels):
            # Wave propagation term: c^2 * Laplacian(u)
            laplacian = self.compute_laplacian(amplitude.reshape(self.state.shape), c)
            wave_term = self.state.wave_speed**2 * laplacian
            
            # Damping term: -gamma * v
            damping_term = -self.state.damping * velocity[c]
            
            acceleration[c] = wave_term + damping_term
        
        # Add source term F(x,t) if defined
        if self._source_fn is not None:
            source = self._source_fn(self.state, t)
            acceleration += source
        
        return acceleration
    
    def step_euler(self):
        """
        Forward Euler integration (first-order, for testing).
        
        u_new = u + dt * v
        v_new = v + dt * a
        """
        t = self.state.time
        
        # Compute acceleration
        accel = self.compute_acceleration(
            self.state.amplitude, 
            self.state.velocity, 
            t
        )
        
        # Update velocity first, then amplitude
        self.state.velocity += self.dt * accel
        self.state.amplitude += self.dt * self.state.velocity
        
        self.state.time += self.dt
    
    def step_leapfrog(self):
        """
        Leapfrog (Verlet) integration - symplectic, energy-conserving.
        
        v(t + dt/2) = v(t - dt/2) + dt * a(t)
        u(t + dt) = u(t) + dt * v(t + dt/2)
        
        This scheme preserves energy for undamped systems.
        """
        t = self.state.time
        
        # Initialize half-step velocity if first call
        if self._velocity_half is None:
            accel = self.compute_acceleration(
                self.state.amplitude,
                self.state.velocity,
                t
            )
            self._velocity_half = self.state.velocity + 0.5 * self.dt * accel
        
        # Update amplitude using half-step velocity
        self.state.amplitude += self.dt * self._velocity_half
        
        # Compute acceleration at new position
        accel = self.compute_acceleration(
            self.state.amplitude,
            self._velocity_half,
            t + self.dt
        )
        
        # Update half-step velocity
        self._velocity_half += self.dt * accel
        
        # Sync displayed velocity (average of half-steps)
        self.state.velocity = self._velocity_half - 0.5 * self.dt * accel
        
        self.state.time += self.dt
    
    def step_rk4(self):
        """
        Fourth-order Runge-Kutta integration - high accuracy.
        
        More expensive but very accurate for smooth solutions.
        """
        t = self.state.time
        dt = self.dt
        
        u0 = self.state.amplitude.copy()
        v0 = self.state.velocity.copy()
        
        # k1
        a1 = self.compute_acceleration(u0, v0, t)
        k1_u = v0
        k1_v = a1
        
        # k2
        u2 = u0 + 0.5 * dt * k1_u
        v2 = v0 + 0.5 * dt * k1_v
        a2 = self.compute_acceleration(u2, v2, t + 0.5 * dt)
        k2_u = v2
        k2_v = a2
        
        # k3
        u3 = u0 + 0.5 * dt * k2_u
        v3 = v0 + 0.5 * dt * k2_v
        a3 = self.compute_acceleration(u3, v3, t + 0.5 * dt)
        k3_u = v3
        k3_v = a3
        
        # k4
        u4 = u0 + dt * k3_u
        v4 = v0 + dt * k3_v
        a4 = self.compute_acceleration(u4, v4, t + dt)
        k4_u = v4
        k4_v = a4
        
        # Combine
        self.state.amplitude = u0 + (dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        self.state.velocity = v0 + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        self.state.time += dt
    
    def step(self, method="leapfrog"):
        """
        Advance the wave state by one time step.
        
        Args:
            method: Integration method ("euler", "leapfrog", "rk4")
        """
        if method == "euler":
            self.step_euler()
        elif method == "leapfrog":
            self.step_leapfrog()
        elif method == "rk4":
            self.step_rk4()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply boundary conditions
        self._apply_boundary()
    
    def _apply_boundary(self):
        """Apply boundary conditions to amplitude and velocity."""
        boundary = self.state.boundary
        
        if boundary == BOUNDARY_PERIODIC:
            # Handled implicitly by np.roll in Laplacian
            pass
        
        elif boundary == BOUNDARY_NEUMANN:
            # Zero-flux: gradient = 0 at boundaries
            for c in range(self.state.channels):
                for axis in range(self.state.ndim):
                    # First boundary
                    idx_first = [slice(None)] * self.state.ndim
                    idx_second = [slice(None)] * self.state.ndim
                    idx_first[axis] = 0
                    idx_second[axis] = 1
                    self.state.amplitude[c][tuple(idx_first)] = \
                        self.state.amplitude[c][tuple(idx_second)]
                    self.state.velocity[c][tuple(idx_first)] = \
                        self.state.velocity[c][tuple(idx_second)]
                    
                    # Last boundary
                    idx_first[axis] = -1
                    idx_second[axis] = -2
                    self.state.amplitude[c][tuple(idx_first)] = \
                        self.state.amplitude[c][tuple(idx_second)]
                    self.state.velocity[c][tuple(idx_first)] = \
                        self.state.velocity[c][tuple(idx_second)]
        
        elif boundary in (BOUNDARY_DIRICHLET, BOUNDARY_ABSORBING):
            # Fixed value at boundaries (absorbing = Dirichlet with zero)
            for c in range(self.state.channels):
                for axis in range(self.state.ndim):
                    idx_first = [slice(None)] * self.state.ndim
                    idx_last = [slice(None)] * self.state.ndim
                    idx_first[axis] = 0
                    idx_last[axis] = -1
                    self.state.amplitude[c][tuple(idx_first)] = 0.0
                    self.state.amplitude[c][tuple(idx_last)] = 0.0
                    self.state.velocity[c][tuple(idx_first)] = 0.0
                    self.state.velocity[c][tuple(idx_last)] = 0.0
    
    def run(self, steps, method="leapfrog", callback=None):
        """
        Run simulation for multiple steps.
        
        Args:
            steps: Number of time steps
            method: Integration method
            callback: Optional callable(state, step) for monitoring
        """
        for i in range(steps):
            self.step(method)
            if callback is not None:
                callback(self.state, i)
    
    def compute_cfl_limit(self):
        """
        Compute CFL stability limit for explicit integration.
        
        For wave equation: dt <= dx / (c * sqrt(ndim))
        
        Returns the maximum stable time step.
        """
        c_max = np.max(self.state.wave_speed)
        return self.dx / (c_max * np.sqrt(self.state.ndim))
    
    def check_stability(self):
        """Check if current dt satisfies CFL condition."""
        cfl_limit = self.compute_cfl_limit()
        return self.dt <= cfl_limit, cfl_limit


class WaveSimulation:
    """
    High-level wave simulation runner.
    
    Combines WaveState and WaveEquation with energy tracking
    and optional visualization hooks.
    """
    
    def __init__(self, dimensions, channels=1, dt=WAVE_DT, 
                 wave_speed=WAVE_SPEED, damping=WAVE_DAMPING):
        """
        Initialize wave simulation.
        
        Args:
            dimensions: Spatial dimensions tuple
            channels: Number of wave channels
            dt: Time step
            wave_speed: Wave propagation speed
            damping: Damping coefficient
        """
        self.state = WaveState(dimensions, channels)
        self.state.wave_speed = wave_speed
        self.state.damping = damping
        
        self.equation = WaveEquation(self.state, dt)
        
        # Check stability
        stable, cfl_limit = self.equation.check_stability()
        if not stable:
            print(f"Warning: dt={dt} exceeds CFL limit={cfl_limit:.6f}")
        
        # History tracking
        self.step_count = 0
        self.amplitude_history = []
    
    def inject_pulse(self, center, sigma=5.0, amplitude=1.0, pulse_type="displacement"):
        """
        Inject initial pulse.
        
        Args:
            center: Center coordinates
            sigma: Gaussian width
            amplitude: Peak amplitude
            pulse_type: "displacement" or "velocity"
        """
        if pulse_type == "displacement":
            self.state.inject_gaussian_pulse(center, sigma, amplitude)
        else:
            self.state.inject_velocity_pulse(center, sigma, amplitude)
    
    def set_source(self, source_fn):
        """Set continuous driving source."""
        self.equation.set_source(source_fn)
    
    def step(self, method="leapfrog"):
        """Advance one time step."""
        self.equation.step(method)
        self.step_count += 1
    
    def run(self, steps, method="leapfrog", record_interval=None):
        """
        Run simulation.
        
        Args:
            steps: Number of steps
            method: Integration method
            record_interval: If set, record amplitude every N steps
        """
        for i in range(steps):
            self.step(method)
            
            if record_interval and (i + 1) % record_interval == 0:
                self.amplitude_history.append(self.state.amplitude.copy())
    
    def get_energy_trace(self):
        """Return energy over time."""
        return np.array(self.state.energy_history)
    
    def get_amplitude_snapshot(self, channel=0):
        """Return current amplitude field."""
        return self.state.get_amplitude(channel)
