# Boundary Condition Framework - Phase 2.8
#
# Comprehensive boundary handling:
#   - DIRICHLET: Fixed value
#   - NEUMANN: Zero flux (reflective)
#   - ROBIN: Mixed condition
#   - ABSORBING_PML: Perfectly Matched Layer

import numpy as np
from src.config import (
    DTYPE,
    BOUNDARY_PERIODIC, BOUNDARY_NEUMANN, BOUNDARY_DIRICHLET, BOUNDARY_ABSORBING,
    BOUNDARY_PML, PML_THICKNESS, PML_SIGMA_MAX
)


class BoundaryConfig:
    """
    Configuration for boundary conditions.
    """
    
    def __init__(self, boundary_type=BOUNDARY_PERIODIC, value=0.0,
                 pml_thickness=PML_THICKNESS, pml_sigma_max=PML_SIGMA_MAX,
                 reflection_coeff=1.0):
        """
        Args:
            boundary_type: Type of boundary condition
            value: Value for Dirichlet boundaries
            pml_thickness: Number of cells for PML layer
            pml_sigma_max: Maximum absorption coefficient
            reflection_coeff: For partial reflection (0=absorb, 1=reflect)
        """
        self.boundary_type = boundary_type
        self.value = value
        self.pml_thickness = pml_thickness
        self.pml_sigma_max = pml_sigma_max
        self.reflection_coeff = reflection_coeff


class PerfectlyMatchedLayer:
    """
    Perfectly Matched Layer (PML) for absorbing boundaries.
    
    Gradually absorbs outgoing waves without reflection.
    
    sigma(x) = sigma_max * ((x - x_boundary) / pml_thickness)^n
    """
    
    def __init__(self, dimensions, thickness=PML_THICKNESS, 
                 sigma_max=PML_SIGMA_MAX, order=2):
        """
        Args:
            dimensions: Field dimensions
            thickness: PML layer thickness in cells
            sigma_max: Maximum absorption at boundary
            order: Polynomial order for sigma profile
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.thickness = thickness
        self.sigma_max = sigma_max
        self.order = order
        
        # Precompute absorption profiles
        self._sigma = self._compute_sigma_field()
    
    def _compute_sigma_field(self):
        """Compute spatially varying absorption coefficient."""
        sigma = np.zeros(self.dimensions, dtype=DTYPE)
        
        for axis in range(self.ndim):
            size = self.dimensions[axis]
            
            # Profile along this axis
            profile = np.zeros(size, dtype=DTYPE)
            
            # Left boundary
            for i in range(self.thickness):
                depth = self.thickness - i
                profile[i] = self.sigma_max * (depth / self.thickness) ** self.order
            
            # Right boundary
            for i in range(self.thickness):
                depth = i + 1
                profile[size - 1 - i] = self.sigma_max * (depth / self.thickness) ** self.order
            
            # Broadcast to full field
            shape = [1] * self.ndim
            shape[axis] = size
            profile = profile.reshape(shape)
            
            sigma = np.maximum(sigma, profile)
        
        return sigma
    
    def apply_damping(self, velocity, dt):
        """
        Apply PML damping to velocity field.
        
        dv/dt += -sigma * v
        
        Args:
            velocity: Velocity field (modified in place)
            dt: Time step
        """
        velocity *= np.exp(-self._sigma * dt)
    
    def get_absorption_mask(self):
        """Return absorption coefficient field."""
        return self._sigma.copy()
    
    def get_interior_mask(self, margin=0):
        """
        Get boolean mask of interior (non-PML) region.
        
        Args:
            margin: Additional margin inside PML boundary
        """
        mask = np.ones(self.dimensions, dtype=bool)
        boundary = self.thickness + margin
        
        for axis in range(self.ndim):
            slices = [slice(None)] * self.ndim
            slices[axis] = slice(0, boundary)
            mask[tuple(slices)] = False
            slices[axis] = slice(-boundary, None)
            mask[tuple(slices)] = False
        
        return mask


class BoundaryHandler:
    """
    Unified boundary condition handler.
    
    Applies boundary conditions to fields based on configuration.
    """
    
    def __init__(self, dimensions, config=None):
        """
        Args:
            dimensions: Field dimensions
            config: BoundaryConfig or boundary type string
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        
        if config is None:
            config = BoundaryConfig()
        elif isinstance(config, str):
            config = BoundaryConfig(boundary_type=config)
        
        self.config = config
        
        # Initialize PML if needed
        self._pml = None
        if config.boundary_type == BOUNDARY_PML:
            self._pml = PerfectlyMatchedLayer(
                dimensions,
                config.pml_thickness,
                config.pml_sigma_max
            )
    
    def apply_amplitude_bc(self, amplitude):
        """Apply boundary conditions to amplitude field."""
        bc = self.config.boundary_type
        
        if bc == BOUNDARY_PERIODIC:
            # Handled by FFT operations
            pass
        
        elif bc == BOUNDARY_DIRICHLET:
            self._apply_dirichlet(amplitude, self.config.value)
        
        elif bc == BOUNDARY_NEUMANN:
            self._apply_neumann(amplitude)
        
        elif bc == BOUNDARY_ABSORBING:
            self._apply_dirichlet(amplitude, 0.0)
        
        elif bc == BOUNDARY_PML:
            # PML doesn't modify amplitude directly
            pass
    
    def apply_velocity_bc(self, velocity, dt=1.0):
        """Apply boundary conditions to velocity field."""
        bc = self.config.boundary_type
        
        if bc == BOUNDARY_PERIODIC:
            pass
        
        elif bc in (BOUNDARY_DIRICHLET, BOUNDARY_ABSORBING):
            self._apply_dirichlet(velocity, 0.0)
        
        elif bc == BOUNDARY_NEUMANN:
            self._apply_neumann(velocity)
        
        elif bc == BOUNDARY_PML:
            if self._pml is not None:
                self._pml.apply_damping(velocity, dt)
    
    def _apply_dirichlet(self, field, value):
        """Set boundaries to fixed value."""
        for axis in range(self.ndim):
            slices_first = [slice(None)] * self.ndim
            slices_last = [slice(None)] * self.ndim
            slices_first[axis] = 0
            slices_last[axis] = -1
            field[tuple(slices_first)] = value
            field[tuple(slices_last)] = value
    
    def _apply_neumann(self, field):
        """Zero-flux (reflective) boundary."""
        for axis in range(self.ndim):
            slices_first = [slice(None)] * self.ndim
            slices_second = [slice(None)] * self.ndim
            slices_first[axis] = 0
            slices_second[axis] = 1
            field[tuple(slices_first)] = field[tuple(slices_second)]
            
            slices_first[axis] = -1
            slices_second[axis] = -2
            field[tuple(slices_first)] = field[tuple(slices_second)]
    
    def get_pml(self):
        """Get PML object if configured."""
        return self._pml


class MixedBoundaryHandler:
    """
    Handler for different boundary conditions on each face.
    
    Allows specifying different conditions for each boundary.
    """
    
    def __init__(self, dimensions):
        """
        Args:
            dimensions: Field dimensions
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        
        # Default: all Neumann
        # faces[axis][side] where side=0 is low, side=1 is high
        self.faces = {}
        for axis in range(self.ndim):
            self.faces[axis] = {
                0: BoundaryConfig(BOUNDARY_NEUMANN),
                1: BoundaryConfig(BOUNDARY_NEUMANN)
            }
    
    def set_face(self, axis, side, config):
        """
        Set boundary condition for specific face.
        
        Args:
            axis: Which axis (0=y/z, 1=x for 2D)
            side: 0 for low boundary, 1 for high boundary
            config: BoundaryConfig
        """
        if isinstance(config, str):
            config = BoundaryConfig(boundary_type=config)
        self.faces[axis][side] = config
    
    def apply(self, amplitude, velocity=None, dt=1.0):
        """Apply all boundary conditions."""
        for axis in range(self.ndim):
            for side in [0, 1]:
                config = self.faces[axis][side]
                self._apply_face(amplitude, axis, side, config)
                if velocity is not None:
                    self._apply_face(velocity, axis, side, config, is_velocity=True, dt=dt)
    
    def _apply_face(self, field, axis, side, config, is_velocity=False, dt=1.0):
        """Apply boundary condition to one face."""
        bc = config.boundary_type
        
        slices = [slice(None)] * self.ndim
        slices_neighbor = [slice(None)] * self.ndim
        
        if side == 0:
            slices[axis] = 0
            slices_neighbor[axis] = 1
        else:
            slices[axis] = -1
            slices_neighbor[axis] = -2
        
        if bc == BOUNDARY_DIRICHLET:
            value = 0.0 if is_velocity else config.value
            field[tuple(slices)] = value
        
        elif bc == BOUNDARY_NEUMANN:
            field[tuple(slices)] = field[tuple(slices_neighbor)]
        
        elif bc == BOUNDARY_ABSORBING:
            field[tuple(slices)] = 0.0
