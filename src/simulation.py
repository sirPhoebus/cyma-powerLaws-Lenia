# Simulation Engine - Time integration of the reaction-diffusion system
# Implements the core update loop: field_next = field + dt * (diffusion + reaction + injection)

import numpy as np
from src.config import DT, D_U, D_V, DTYPE
from src.field import ResonantField
from src.kernel import Kernel
from src.reaction import ReactionSystem, GrayScott


class Simulation:
    """
    Core simulation engine for reaction-diffusion dynamics.
    
    Combines:
    - Field (the substrate)
    - Kernel (diffusion operator)
    - Reaction system (local nonlinear dynamics)
    
    Into a unified time evolution.
    """
    
    def __init__(self, field, reaction_system=None, dt=DT):
        self.field = field
        self.reaction = reaction_system if reaction_system else GrayScott()
        self.dt = dt
        self.step_count = 0
        self.time = 0.0
        
        # Diffusion coefficients per channel
        self.diffusion_coefficients = [D_U, D_V]
        
        # Laplacian kernel for diffusion
        self.laplacian = Kernel.laplacian_2d()
        
        # History tracking
        self.history = []
        self.record_interval = 100
    
    def set_diffusion_coefficients(self, coefficients):
        """Set diffusion rates for each channel."""
        self.diffusion_coefficients = coefficients
    
    def compute_diffusion(self):
        """
        Compute the diffusion term: D * nabla^2(field)
        """
        diffusion = np.zeros_like(self.field.data)
        for c in range(self.field.channels):
            laplacian_result = self.laplacian.convolve(self.field.data[c])
            diffusion[c] = self.diffusion_coefficients[c] * laplacian_result
        return diffusion
    
    def compute_reaction(self):
        """
        Compute the reaction term: f(u,v), g(u,v)
        """
        u = self.field.data[0]
        v = self.field.data[1]
        du, dv = self.reaction.react(u, v)
        
        reaction = np.zeros_like(self.field.data)
        reaction[0] = du
        reaction[1] = dv
        return reaction
    
    def step(self):
        """
        Advance simulation by one time step.
        
        Uses forward Euler integration:
        field_next = field + dt * (diffusion + reaction)
        """
        diffusion = self.compute_diffusion()
        reaction = self.compute_reaction()
        
        # Forward Euler update
        self.field.data += self.dt * (diffusion + reaction)
        
        # Apply boundary conditions
        self.field.apply_boundary()
        
        # Clamp values to valid range (stability)
        self.field.data = np.clip(self.field.data, 0.0, 1.0)
        
        self.step_count += 1
        self.time += self.dt
        
        # Record history periodically
        if self.step_count % self.record_interval == 0:
            self.history.append({
                'step': self.step_count,
                'time': self.time,
                'energy': self.field.compute_energy(),
                'u_mean': np.mean(self.field.data[0]),
                'v_mean': np.mean(self.field.data[1]),
                'u_std': np.std(self.field.data[0]),
                'v_std': np.std(self.field.data[1])
            })
    
    def run(self, num_steps):
        """Run simulation for specified number of steps."""
        for _ in range(num_steps):
            self.step()
    
    def get_state(self):
        """Return current field state."""
        return self.field.data.copy()
    
    def get_statistics(self):
        """Return current field statistics."""
        return {
            'step': self.step_count,
            'time': self.time,
            'energy': self.field.compute_energy(),
            'u_mean': np.mean(self.field.data[0]),
            'v_mean': np.mean(self.field.data[1]),
            'u_max': np.max(self.field.data[0]),
            'v_max': np.max(self.field.data[1]),
            'u_min': np.min(self.field.data[0]),
            'v_min': np.min(self.field.data[1])
        }


class RungeKuttaSimulation(Simulation):
    """
    4th-order Runge-Kutta integration for higher accuracy.
    
    More stable for stiff systems and larger time steps.
    """
    
    def compute_derivatives(self, field_data):
        """Compute total derivatives for given field state."""
        # Temporarily set field data
        original_data = self.field.data.copy()
        self.field.data = field_data
        
        diffusion = self.compute_diffusion()
        reaction = self.compute_reaction()
        
        # Restore original data
        self.field.data = original_data
        
        return diffusion + reaction
    
    def step(self):
        """
        Advance simulation using RK4 integration.
        """
        y = self.field.data.copy()
        
        k1 = self.compute_derivatives(y)
        k2 = self.compute_derivatives(y + 0.5 * self.dt * k1)
        k3 = self.compute_derivatives(y + 0.5 * self.dt * k2)
        k4 = self.compute_derivatives(y + self.dt * k3)
        
        self.field.data = y + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Apply boundary conditions
        self.field.apply_boundary()
        
        # Clamp values
        self.field.data = np.clip(self.field.data, 0.0, 1.0)
        
        self.step_count += 1
        self.time += self.dt
        
        if self.step_count % self.record_interval == 0:
            self.history.append({
                'step': self.step_count,
                'time': self.time,
                'energy': self.field.compute_energy(),
                'u_mean': np.mean(self.field.data[0]),
                'v_mean': np.mean(self.field.data[1]),
                'u_std': np.std(self.field.data[0]),
                'v_std': np.std(self.field.data[1])
            })
