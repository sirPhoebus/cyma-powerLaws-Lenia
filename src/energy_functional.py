# Energy Functional - Phase 3.1
#
# Computable energy measures for fields:
#   E(field) = kinetic + potential + gradient
#
# Patterns are local minima of the energy functional.

import numpy as np
from src.config import DTYPE


class EnergyFunctional:
    """
    Compute total energy of a field configuration.
    
    E = E_kinetic + E_potential + E_gradient + E_interaction
    
    Local minima of E correspond to stable patterns.
    """
    
    def __init__(self, wave_speed=1.0, density=1.0, 
                 potential_fn=None, interaction_fn=None):
        """
        Args:
            wave_speed: Wave propagation speed
            density: Medium density
            potential_fn: Custom potential V(u) function (default: quadratic)
            interaction_fn: Custom interaction between channels
        """
        self.wave_speed = wave_speed
        self.density = density
        self.potential_fn = potential_fn
        self.interaction_fn = interaction_fn
    
    def kinetic_energy(self, velocity):
        """
        E_kinetic = 0.5 * rho * integral(v^2)
        
        Energy from motion.
        """
        return 0.5 * self.density * np.sum(velocity**2)
    
    def potential_energy(self, field):
        """
        E_potential = integral(V(u))
        
        Energy from field values themselves.
        Default: V(u) = 0.5 * u^2 (harmonic potential)
        """
        if self.potential_fn is not None:
            return np.sum(self.potential_fn(field))
        
        # Default harmonic potential
        return 0.5 * np.sum(field**2)
    
    def gradient_energy(self, field):
        """
        E_gradient = 0.5 * c^2 * integral(|grad(u)|^2)
        
        Energy from spatial variations (elasticity/stiffness).
        """
        grad_squared = np.zeros_like(field)
        
        for axis in range(field.ndim):
            grad = np.gradient(field, axis=axis)
            grad_squared += grad**2
        
        return 0.5 * self.wave_speed**2 * np.sum(grad_squared)
    
    def interaction_energy(self, fields):
        """
        E_interaction = integral(W(u1, u2, ...))
        
        Energy from interaction between multiple fields/channels.
        """
        if self.interaction_fn is None or len(fields) < 2:
            return 0.0
        
        return np.sum(self.interaction_fn(*fields))
    
    def total_energy(self, field, velocity=None, fields=None):
        """
        Compute total energy.
        
        E = E_k + E_p + E_g + E_int
        
        Args:
            field: Primary field (amplitude)
            velocity: Optional velocity field
            fields: Optional list of additional fields for interaction
            
        Returns:
            Total energy (scalar)
        """
        E_kinetic = 0.0
        if velocity is not None:
            E_kinetic = self.kinetic_energy(velocity)
        
        E_potential = self.potential_energy(field)
        E_gradient = self.gradient_energy(field)
        
        E_interaction = 0.0
        if fields is not None:
            E_interaction = self.interaction_energy(fields)
        
        return E_kinetic + E_potential + E_gradient + E_interaction
    
    def energy_density(self, field, velocity=None):
        """
        Compute local energy density at each point.
        
        Returns field of same shape as input.
        """
        density = np.zeros_like(field)
        
        # Kinetic density
        if velocity is not None:
            density += 0.5 * self.density * velocity**2
        
        # Potential density
        if self.potential_fn is not None:
            density += self.potential_fn(field)
        else:
            density += 0.5 * field**2
        
        # Gradient density
        grad_squared = np.zeros_like(field)
        for axis in range(field.ndim):
            grad = np.gradient(field, axis=axis)
            grad_squared += grad**2
        density += 0.5 * self.wave_speed**2 * grad_squared
        
        return density
    
    def compute_gradient_force(self, field):
        """
        Compute force from energy gradient: F = -dE/du
        
        For harmonic system: F = c^2 * Laplacian(u) - u
        """
        # Laplacian term (from gradient energy)
        laplacian = np.zeros_like(field)
        for axis in range(field.ndim):
            laplacian += np.gradient(np.gradient(field, axis=axis), axis=axis)
        
        force = self.wave_speed**2 * laplacian
        
        # Potential term
        if self.potential_fn is None:
            force -= field  # dV/du for V = 0.5 * u^2
        
        return force


class GinzburgLandauEnergy(EnergyFunctional):
    """
    Ginzburg-Landau free energy functional.
    
    F = integral( 0.5 * |grad(psi)|^2 + V(psi) )
    
    where V(psi) = -0.5 * a * |psi|^2 + 0.25 * b * |psi|^4
    
    Has phase transition at a = 0.
    """
    
    def __init__(self, a=1.0, b=1.0, kappa=1.0):
        """
        Args:
            a: Linear coefficient (controls phase transition)
            b: Quartic coefficient (nonlinearity strength)
            kappa: Gradient coefficient
        """
        self.a = a
        self.b = b
        self.kappa = kappa
        
        def gl_potential(psi):
            psi_sq = psi**2
            return -0.5 * a * psi_sq + 0.25 * b * psi_sq**2
        
        super().__init__(wave_speed=np.sqrt(kappa), potential_fn=gl_potential)
    
    def order_parameter(self, field):
        """Compute order parameter (average |psi|)."""
        return np.mean(np.abs(field))
    
    def equilibrium_amplitude(self):
        """Theoretical equilibrium amplitude for a > 0."""
        if self.a > 0:
            return np.sqrt(self.a / self.b)
        return 0.0


class DoubleWellEnergy(EnergyFunctional):
    """
    Double-well potential energy.
    
    V(u) = 0.25 * (u^2 - 1)^2
    
    Has two stable minima at u = +1 and u = -1.
    Creates domain walls between phases.
    """
    
    def __init__(self, kappa=1.0, barrier_height=1.0):
        """
        Args:
            kappa: Gradient coefficient
            barrier_height: Height of barrier between wells
        """
        self.barrier_height = barrier_height
        
        def double_well(u):
            return barrier_height * 0.25 * (u**2 - 1)**2
        
        super().__init__(wave_speed=np.sqrt(kappa), potential_fn=double_well)
    
    def domain_wall_width(self):
        """Theoretical width of domain wall."""
        return np.sqrt(2.0 * self.wave_speed**2 / self.barrier_height)


class SwiftHohenbergEnergy(EnergyFunctional):
    """
    Swift-Hohenberg energy functional for pattern formation.
    
    F = integral( 0.5 * u * (r + (nabla^2 + q0^2)^2) * u + 0.25 * u^4 )
    
    Creates periodic stripe/spot patterns.
    """
    
    def __init__(self, r=-0.1, q0=1.0):
        """
        Args:
            r: Control parameter (patterns for r < 0)
            q0: Preferred wavenumber
        """
        self.r = r
        self.q0 = q0
        
        super().__init__()
    
    def total_energy(self, field, velocity=None, fields=None):
        """Compute Swift-Hohenberg energy."""
        # Quadratic term: u * (r + (nabla^2 + q0^2)^2) * u
        laplacian = np.zeros_like(field)
        for axis in range(field.ndim):
            laplacian += np.gradient(np.gradient(field, axis=axis), axis=axis)
        
        # (nabla^2 + q0^2)
        shifted_laplacian = laplacian + self.q0**2 * field
        
        # Apply Laplacian again for (nabla^2 + q0^2)^2
        bi_laplacian = np.zeros_like(shifted_laplacian)
        for axis in range(field.ndim):
            bi_laplacian += np.gradient(np.gradient(shifted_laplacian, axis=axis), axis=axis)
        
        linear_term = self.r * field + bi_laplacian + self.q0**2 * shifted_laplacian
        
        E_quadratic = 0.5 * np.sum(field * linear_term)
        E_quartic = 0.25 * np.sum(field**4)
        
        return E_quadratic + E_quartic
    
    def preferred_wavelength(self):
        """Return preferred pattern wavelength."""
        return 2 * np.pi / self.q0


class EnergyMinimizer:
    """
    Gradient descent energy minimization.
    
    Finds local minima of energy functional.
    """
    
    def __init__(self, energy_fn, dt=0.01, damping=0.1):
        """
        Args:
            energy_fn: EnergyFunctional instance
            dt: Time step
            damping: Damping coefficient for overdamped dynamics
        """
        self.energy_fn = energy_fn
        self.dt = dt
        self.damping = damping
        
        self.energy_history = []
    
    def gradient_descent_step(self, field):
        """
        Single gradient descent step.
        
        du/dt = -dE/du (overdamped)
        """
        force = self.energy_fn.compute_gradient_force(field)
        return field + self.dt * force
    
    def minimize(self, field, max_steps=1000, tolerance=1e-6):
        """
        Run energy minimization until convergence.
        
        Args:
            field: Initial field configuration
            max_steps: Maximum iteration steps
            tolerance: Convergence criterion (relative energy change)
            
        Returns:
            Minimized field, converged boolean
        """
        self.energy_history = []
        current_field = field.copy()
        
        E_prev = self.energy_fn.total_energy(current_field)
        self.energy_history.append(E_prev)
        
        for step in range(max_steps):
            current_field = self.gradient_descent_step(current_field)
            
            E_current = self.energy_fn.total_energy(current_field)
            self.energy_history.append(E_current)
            
            # Check convergence
            if E_prev > 0:
                rel_change = abs(E_current - E_prev) / E_prev
                if rel_change < tolerance:
                    return current_field, True
            
            E_prev = E_current
        
        return current_field, False
    
    def get_energy_trace(self):
        """Return energy history."""
        return np.array(self.energy_history)
