# Hybrid System Integration - Phase 2.12
#
# Couples wave dynamics with reaction-diffusion (Phase 1):
#
# field_next = field + dt * (
#     diffusion_term(field) +
#     reaction_term(field) +
#     wave_acceleration_term(field, velocity) +
#     injection_term(sources)
# )
#
# velocity_next = velocity + dt * wave_acceleration_term(field, velocity)
#
# This enables cymatics-RD hybrid patterns where:
#   - Waves create local excitations
#   - RD dynamics process those excitations into patterns
#   - Patterns feedback into wave propagation

import numpy as np
from src.field_nd import ResonantFieldND
from src.wave_state import WaveState
from src.wave_equation import WaveEquation
from src.kernel_nd import KernelND
from src.config import (
    DTYPE, DT, WAVE_DT,
    D_U, D_V,
    WAVE_SPEED, WAVE_DAMPING,
    FIELD_CHANNELS
)


class HybridState:
    """
    Combined state for wave + reaction-diffusion system.
    
    Contains:
      - amplitude: Wave displacement field
      - velocity: Wave velocity field  
      - concentrations: RD chemical concentrations (u, v, ...)
    """
    
    def __init__(self, dimensions, rd_channels=2, wave_channels=1):
        """
        Args:
            dimensions: Spatial dimensions
            rd_channels: Number of RD channels (e.g., 2 for Gray-Scott)
            wave_channels: Number of wave channels
        """
        if isinstance(dimensions, int):
            dimensions = (dimensions, dimensions)
        
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.rd_channels = rd_channels
        self.wave_channels = wave_channels
        
        # Wave fields
        self.amplitude = np.zeros((wave_channels,) + self.dimensions, dtype=DTYPE)
        self.velocity = np.zeros((wave_channels,) + self.dimensions, dtype=DTYPE)
        
        # RD concentration fields
        self.concentrations = np.zeros((rd_channels,) + self.dimensions, dtype=DTYPE)
        
        # Material properties (can be modulated by RD)
        self._wave_speed = np.full(self.dimensions, WAVE_SPEED, dtype=DTYPE)
        self._damping = np.full(self.dimensions, WAVE_DAMPING, dtype=DTYPE)
        
        # Time tracking
        self.time = 0.0
    
    @property
    def u(self):
        """RD activator/substrate concentration."""
        return self.concentrations[0]
    
    @u.setter
    def u(self, value):
        self.concentrations[0] = value
    
    @property
    def v(self):
        """RD inhibitor/autocatalyst concentration."""
        return self.concentrations[1] if self.rd_channels > 1 else None
    
    @v.setter
    def v(self, value):
        if self.rd_channels > 1:
            self.concentrations[1] = value
    
    @property
    def wave_speed(self):
        return self._wave_speed
    
    @wave_speed.setter
    def wave_speed(self, value):
        if np.isscalar(value):
            self._wave_speed.fill(value)
        else:
            self._wave_speed = np.asarray(value, dtype=DTYPE)
    
    @property
    def damping(self):
        return self._damping
    
    @damping.setter
    def damping(self, value):
        if np.isscalar(value):
            self._damping.fill(value)
        else:
            self._damping = np.asarray(value, dtype=DTYPE)
    
    def initialize_rd_uniform(self, u_value, v_value):
        """Initialize RD fields uniformly."""
        self.u = np.full(self.dimensions, u_value, dtype=DTYPE)
        if self.v is not None:
            self.v = np.full(self.dimensions, v_value, dtype=DTYPE)
    
    def initialize_rd_perturbed(self, u_base, v_base, perturbation=0.01):
        """Initialize RD fields with small random perturbations."""
        self.u = u_base + perturbation * np.random.randn(*self.dimensions)
        if self.v is not None:
            self.v = v_base + perturbation * np.random.randn(*self.dimensions)
    
    def inject_wave_pulse(self, center, sigma, amplitude, channel=0):
        """Inject Gaussian pulse into wave field."""
        grids = np.ogrid[tuple(slice(0, d) for d in self.dimensions)]
        dist_sq = sum((g - c)**2 for g, c in zip(grids, center))
        pulse = amplitude * np.exp(-dist_sq / (2 * sigma**2))
        self.amplitude[channel] += pulse
    
    def inject_rd_seed(self, center, radius, u_value, v_value, channel_u=0, channel_v=1):
        """Inject circular seed in RD field."""
        grids = np.ogrid[tuple(slice(0, d) for d in self.dimensions)]
        dist_sq = sum((g - c)**2 for g, c in zip(grids, center))
        mask = dist_sq <= radius**2
        
        self.concentrations[channel_u][mask] = u_value
        if self.rd_channels > 1:
            self.concentrations[channel_v][mask] = v_value
    
    def compute_total_energy(self):
        """Compute total energy (wave + RD potential)."""
        # Wave kinetic energy
        wave_kinetic = 0.5 * np.sum(self.velocity**2)
        
        # Wave potential energy
        wave_potential = 0.0
        for c in range(self.wave_channels):
            for axis in range(self.ndim):
                grad = np.gradient(self.amplitude[c], axis=axis)
                wave_potential += 0.5 * np.sum(self._wave_speed**2 * grad**2)
        
        # RD "energy" (sum of concentrations squared)
        rd_energy = 0.5 * np.sum(self.concentrations**2)
        
        return wave_kinetic + wave_potential + rd_energy
    
    def copy(self):
        """Create deep copy."""
        new_state = HybridState(self.dimensions, self.rd_channels, self.wave_channels)
        new_state.amplitude = self.amplitude.copy()
        new_state.velocity = self.velocity.copy()
        new_state.concentrations = self.concentrations.copy()
        new_state._wave_speed = self._wave_speed.copy()
        new_state._damping = self._damping.copy()
        new_state.time = self.time
        return new_state


class CouplingFunction:
    """
    Defines how wave and RD systems couple.
    """
    
    def __init__(self, wave_to_rd_strength=0.1, rd_to_wave_strength=0.1,
                 coupling_type="additive"):
        """
        Args:
            wave_to_rd_strength: How much waves affect RD
            rd_to_wave_strength: How much RD affects waves
            coupling_type: "additive", "multiplicative", or "modulation"
        """
        self.wave_to_rd = wave_to_rd_strength
        self.rd_to_wave = rd_to_wave_strength
        self.coupling_type = coupling_type
    
    def wave_to_rd_term(self, amplitude, velocity):
        """
        Compute wave contribution to RD dynamics.
        
        Wave excitations can trigger RD reactions.
        """
        if self.coupling_type == "additive":
            # Wave amplitude adds to activator
            return self.wave_to_rd * np.mean(amplitude, axis=0)
        elif self.coupling_type == "multiplicative":
            # Wave amplitude modulates reaction rate
            return self.wave_to_rd * np.abs(np.mean(amplitude, axis=0))
        else:
            return np.zeros_like(amplitude[0])
    
    def rd_to_wave_speed(self, concentrations):
        """
        Compute RD modulation of wave speed.
        
        RD patterns can create regions of different wave velocity.
        """
        if self.coupling_type == "modulation":
            # Higher concentration = higher wave speed
            u = concentrations[0]
            return 1.0 + self.rd_to_wave * (u - np.mean(u))
        else:
            return np.ones_like(concentrations[0])
    
    def rd_to_wave_damping(self, concentrations):
        """
        Compute RD modulation of wave damping.
        
        RD patterns can create absorbing regions.
        """
        if self.rd_to_wave > 0 and len(concentrations) > 1:
            v = concentrations[1]
            return self.rd_to_wave * v
        return np.zeros_like(concentrations[0])


class HybridSimulation:
    """
    Unified simulation of coupled wave-RD system.
    """
    
    def __init__(self, dimensions, dt_wave=WAVE_DT, dt_rd=DT,
                 diffusion_coeffs=(D_U, D_V), wave_speed=WAVE_SPEED,
                 coupling=None):
        """
        Args:
            dimensions: Spatial dimensions
            dt_wave: Wave equation time step
            dt_rd: RD equation time step
            diffusion_coeffs: (D_u, D_v) diffusion coefficients
            wave_speed: Base wave speed
            coupling: CouplingFunction (None = no coupling)
        """
        self.dimensions = tuple(dimensions) if not isinstance(dimensions, tuple) else dimensions
        self.dt_wave = dt_wave
        self.dt_rd = dt_rd
        self.diffusion_coeffs = diffusion_coeffs
        
        # Create hybrid state
        self.state = HybridState(self.dimensions)
        self.state.wave_speed = wave_speed
        
        # Coupling function
        self.coupling = coupling if coupling else CouplingFunction(0, 0)
        
        # Laplacian kernel for RD diffusion
        self._laplacian_kernel = KernelND.laplacian(ndim=len(self.dimensions))
        
        # Wave equation components
        self._velocity_half = None  # For leapfrog
        
        # Source function
        self._source_fn = None
        
        # Reaction function
        self._reaction_fn = self._gray_scott_reaction
        
        # Step counter
        self.step_count = 0
    
    def set_source(self, source_fn):
        """Set wave source function."""
        self._source_fn = source_fn
    
    def set_reaction(self, reaction_fn):
        """
        Set RD reaction function.
        
        Args:
            reaction_fn: Callable(u, v) -> (du, dv)
        """
        self._reaction_fn = reaction_fn
    
    def _gray_scott_reaction(self, u, v, F=0.055, k=0.062):
        """Default Gray-Scott reaction."""
        uvv = u * v * v
        du = -uvv + F * (1 - u)
        dv = uvv - (F + k) * v
        return du, dv
    
    def _compute_diffusion(self):
        """Compute diffusion terms for RD."""
        diffusion = np.zeros_like(self.state.concentrations)
        
        for c in range(self.state.rd_channels):
            laplacian = self._laplacian_kernel.convolve(
                self.state.concentrations[c], mode='wrap'
            )
            diffusion[c] = self.diffusion_coeffs[c] * laplacian
        
        return diffusion
    
    def _compute_wave_laplacian(self, amplitude):
        """Compute Laplacian of wave amplitude."""
        laplacian = np.zeros_like(amplitude)
        for c in range(self.state.wave_channels):
            laplacian[c] = self._laplacian_kernel.convolve(amplitude[c], mode='wrap')
        return laplacian
    
    def _compute_wave_acceleration(self):
        """
        Compute wave acceleration: c^2 * Laplacian(u) - gamma * v
        """
        laplacian = self._compute_wave_laplacian(self.state.amplitude)
        
        # Apply RD-modulated wave speed
        if self.coupling.rd_to_wave > 0:
            speed_mod = self.coupling.rd_to_wave_speed(self.state.concentrations)
            effective_speed = self.state.wave_speed * speed_mod
        else:
            effective_speed = self.state.wave_speed
        
        # Apply RD-modulated damping
        if self.coupling.rd_to_wave > 0:
            damping_mod = self.coupling.rd_to_wave_damping(self.state.concentrations)
            effective_damping = self.state.damping + damping_mod
        else:
            effective_damping = self.state.damping
        
        acceleration = np.zeros_like(self.state.amplitude)
        for c in range(self.state.wave_channels):
            acceleration[c] = (effective_speed**2 * laplacian[c] - 
                             effective_damping * self.state.velocity[c])
        
        return acceleration
    
    def step_wave(self, method="leapfrog"):
        """Advance wave equation by one time step."""
        dt = self.dt_wave
        
        if method == "leapfrog":
            if self._velocity_half is None:
                accel = self._compute_wave_acceleration()
                self._velocity_half = self.state.velocity + 0.5 * dt * accel
            
            self.state.amplitude += dt * self._velocity_half
            
            # Add source
            if self._source_fn is not None:
                source = self._source_fn(self.state, self.state.time)
                self.state.amplitude += dt * source
            
            accel = self._compute_wave_acceleration()
            self._velocity_half += dt * accel
            self.state.velocity = self._velocity_half - 0.5 * dt * accel
        else:
            # Simple Euler
            accel = self._compute_wave_acceleration()
            self.state.velocity += dt * accel
            self.state.amplitude += dt * self.state.velocity
    
    def step_rd(self):
        """Advance RD equation by one time step."""
        dt = self.dt_rd
        
        # Diffusion
        diffusion = self._compute_diffusion()
        
        # Reaction
        if self.state.rd_channels >= 2:
            du, dv = self._reaction_fn(self.state.u, self.state.v)
            reaction = np.stack([du, dv], axis=0)
        else:
            reaction = np.zeros_like(self.state.concentrations)
        
        # Wave coupling
        if self.coupling.wave_to_rd > 0:
            wave_term = self.coupling.wave_to_rd_term(
                self.state.amplitude, self.state.velocity
            )
            # Add to activator
            reaction[0] += wave_term
        
        # Update
        self.state.concentrations += dt * (diffusion + reaction)
        
        # Clamp to valid range
        self.state.concentrations = np.clip(self.state.concentrations, 0, 1)
    
    def step(self, wave_steps=1, rd_steps=1):
        """
        Advance hybrid system.
        
        Args:
            wave_steps: Number of wave sub-steps
            rd_steps: Number of RD sub-steps
        """
        for _ in range(wave_steps):
            self.step_wave()
        
        for _ in range(rd_steps):
            self.step_rd()
        
        self.state.time += max(self.dt_wave * wave_steps, self.dt_rd * rd_steps)
        self.step_count += 1
    
    def run(self, total_steps, wave_ratio=10, callback=None):
        """
        Run hybrid simulation.
        
        Args:
            total_steps: Number of macro steps
            wave_ratio: Wave steps per RD step (waves are faster)
            callback: Optional callback(state, step)
        """
        for i in range(total_steps):
            self.step(wave_steps=wave_ratio, rd_steps=1)
            
            if callback is not None:
                callback(self.state, i)
    
    def get_wave_field(self, channel=0):
        """Get wave amplitude field."""
        return self.state.amplitude[channel]
    
    def get_rd_field(self, channel=0):
        """Get RD concentration field."""
        return self.state.concentrations[channel]
    
    def get_combined_visualization(self, wave_weight=0.5):
        """
        Get combined wave+RD field for visualization.
        
        Args:
            wave_weight: Blending weight (0=RD only, 1=wave only)
        """
        wave = self.state.amplitude[0]
        rd = self.state.u
        
        # Normalize each
        wave_norm = wave / (np.max(np.abs(wave)) + 1e-10)
        rd_norm = rd / (np.max(np.abs(rd)) + 1e-10)
        
        return wave_weight * wave_norm + (1 - wave_weight) * rd_norm


def create_cymatics_rd_hybrid(size, rd_type="gray_scott"):
    """
    Create a cymatics + RD hybrid simulation.
    
    Wave patterns trigger RD dynamics, creating emergent structures.
    """
    coupling = CouplingFunction(
        wave_to_rd_strength=0.05,
        rd_to_wave_strength=0.1,
        coupling_type="additive"
    )
    
    sim = HybridSimulation(
        dimensions=(size, size),
        dt_wave=0.1,
        dt_rd=1.0,
        coupling=coupling
    )
    
    # Initialize RD to Gray-Scott steady state
    sim.state.initialize_rd_uniform(1.0, 0.0)
    
    return sim
