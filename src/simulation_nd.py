# SimulationND - N-Dimensional Reaction-Diffusion Simulation
# Unified simulation engine for fields of any dimensionality

import numpy as np
from src.config import DT, D_U, D_V, DTYPE
from src.field_nd import ResonantFieldND
from src.kernel_nd import KernelND
from src.reaction import ReactionSystem, GrayScott


class SimulationND:
    """
    N-Dimensional simulation engine for reaction-diffusion dynamics.
    
    Combines:
    - ResonantFieldND (N-dimensional substrate)
    - KernelND (N-dimensional diffusion operator)
    - ReactionSystem (local nonlinear dynamics)
    """
    
    def __init__(self, field, reaction_system=None, dt=DT):
        self.field = field
        self.reaction = reaction_system if reaction_system else GrayScott()
        self.dt = dt
        self.step_count = 0
        self.time = 0.0
        
        # Diffusion coefficients per channel
        self.diffusion_coefficients = [D_U, D_V]
        
        # N-dimensional Laplacian kernel
        self.laplacian = KernelND.laplacian(ndim=field.ndim)
        
        # Use FFT convolution for efficiency
        self.use_fft = True
        
        # History tracking
        self.history = []
        self.record_interval = 100
    
    def set_diffusion_coefficients(self, coefficients):
        """Set diffusion rates for each channel."""
        self.diffusion_coefficients = coefficients
    
    def compute_diffusion(self):
        """Compute the diffusion term: D * nabla^2(field)"""
        diffusion = np.zeros_like(self.field.data)
        
        for c in range(self.field.channels):
            if self.use_fft:
                laplacian_result = self.laplacian.convolve_fft(self.field.data[c])
            else:
                laplacian_result = self.laplacian.convolve(self.field.data[c])
            diffusion[c] = self.diffusion_coefficients[c] * laplacian_result
        
        return diffusion
    
    def compute_reaction(self):
        """Compute the reaction term: f(u,v), g(u,v)"""
        u = self.field.data[0]
        v = self.field.data[1]
        du, dv = self.reaction.react(u, v)
        
        reaction = np.zeros_like(self.field.data)
        reaction[0] = du
        reaction[1] = dv
        return reaction
    
    def step_euler(self):
        """Forward Euler integration step."""
        diffusion = self.compute_diffusion()
        reaction = self.compute_reaction()
        
        self.field.data += self.dt * (diffusion + reaction)
        self.field.apply_boundary()
        self.field.data = np.clip(self.field.data, 0.0, 1.0)
    
    def step_rk4(self):
        """4th-order Runge-Kutta integration step."""
        y0 = self.field.data.copy()
        
        k1 = self._compute_derivatives(y0)
        k2 = self._compute_derivatives(y0 + 0.5 * self.dt * k1)
        k3 = self._compute_derivatives(y0 + 0.5 * self.dt * k2)
        k4 = self._compute_derivatives(y0 + self.dt * k3)
        
        self.field.data = y0 + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.field.apply_boundary()
        self.field.data = np.clip(self.field.data, 0.0, 1.0)
    
    def _compute_derivatives(self, field_data):
        """Compute total derivatives for given field state."""
        original = self.field.data
        self.field.data = field_data
        
        result = self.compute_diffusion() + self.compute_reaction()
        
        self.field.data = original
        return result
    
    def step(self, method='euler'):
        """Advance simulation by one time step."""
        if method == 'rk4':
            self.step_rk4()
        else:
            self.step_euler()
        
        self.step_count += 1
        self.time += self.dt
        
        if self.step_count % self.record_interval == 0:
            self._record_history()
    
    def _record_history(self):
        """Record current state statistics."""
        self.history.append({
            'step': self.step_count,
            'time': self.time,
            'energy': self.field.compute_energy(),
            'u_mean': np.mean(self.field.data[0]),
            'v_mean': np.mean(self.field.data[1]),
            'u_std': np.std(self.field.data[0]),
            'v_std': np.std(self.field.data[1]),
            'u_max': np.max(self.field.data[0]),
            'v_max': np.max(self.field.data[1])
        })
    
    def run(self, num_steps, method='euler'):
        """Run simulation for specified number of steps."""
        for _ in range(num_steps):
            self.step(method)
    
    def get_state(self):
        """Return current field state."""
        return self.field.data.copy()
    
    def get_statistics(self):
        """Return current field statistics."""
        return {
            'step': self.step_count,
            'time': self.time,
            'ndim': self.field.ndim,
            'dimensions': self.field.dimensions,
            'energy': self.field.compute_energy(),
            'u_mean': np.mean(self.field.data[0]),
            'v_mean': np.mean(self.field.data[1]),
            'u_std': np.std(self.field.data[0]),
            'v_std': np.std(self.field.data[1]),
            'u_max': np.max(self.field.data[0]),
            'v_max': np.max(self.field.data[1]),
            'u_min': np.min(self.field.data[0]),
            'v_min': np.min(self.field.data[1])
        }
