# Volume Wave Propagation - Phase 2.4
#
# 3D wave propagation in volumetric domains.
# Extends WaveEquation with 3D-specific features:
#   - Spherical and shell wave sources
#   - Higher-order stencils optimized for 3D
#   - Slice extraction for visualization
#   - Isosurface extraction for nodal surfaces

import numpy as np
from src.wave_state import WaveState
from src.wave_equation import WaveEquation, LAPLACIAN_STENCILS
from src.config import (
    DTYPE, WAVE_DT, WAVE_SPEED, WAVE_DAMPING,
    DEFAULT_STENCIL_ORDER, STENCIL_ORDER_2, STENCIL_ORDER_4, STENCIL_ORDER_6,
    BOUNDARY_PERIODIC, BOUNDARY_NEUMANN, BOUNDARY_DIRICHLET, BOUNDARY_ABSORBING
)


class VolumeWaveConfig:
    """
    Configuration for 3D volume wave simulation.
    """
    
    def __init__(self, resolution=(64, 64, 64), 
                 boundary=BOUNDARY_PERIODIC,
                 stencil_order=STENCIL_ORDER_4,
                 timestepper="leapfrog"):
        """
        Args:
            resolution: (nz, ny, nx) voxel dimensions
            boundary: Boundary condition type
            stencil_order: Finite difference accuracy (2, 4, or 6)
            timestepper: Integration method
        """
        self.resolution = tuple(resolution)
        self.boundary = boundary
        self.stencil_order = stencil_order
        self.timestepper = timestepper
        
        # Derived properties
        self.ndim = 3
        self.nz, self.ny, self.nx = self.resolution
        self.total_voxels = np.prod(self.resolution)


class VolumeWaveState(WaveState):
    """
    3D Wave State with volume-specific methods.
    """
    
    def __init__(self, resolution, channels=1):
        """
        Args:
            resolution: (nz, ny, nx) or single int for cubic grid
        """
        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)
        
        super().__init__(dimensions=resolution, channels=channels)
    
    @property
    def nz(self):
        return self.dimensions[0]
    
    @property
    def ny(self):
        return self.dimensions[1]
    
    @property
    def nx(self):
        return self.dimensions[2]
    
    def inject_spherical_pulse(self, center, radius, amplitude, channel=0, profile="gaussian"):
        """
        Inject a spherical pulse.
        
        Args:
            center: (z, y, x) center coordinates
            radius: Characteristic radius
            amplitude: Peak amplitude
            channel: Target channel
            profile: "gaussian", "solid", or "shell"
        """
        grids = self._get_coord_grids()
        dist_squared = sum((g - c)**2 for g, c in zip(grids, center))
        dist = np.sqrt(dist_squared)
        
        if profile == "gaussian":
            pulse = amplitude * np.exp(-dist_squared / (2 * radius**2))
        elif profile == "solid":
            pulse = np.where(dist <= radius, amplitude, 0.0)
        elif profile == "shell":
            # Thin shell at radius
            shell_width = radius * 0.2
            pulse = amplitude * np.exp(-((dist - radius)**2) / (2 * shell_width**2))
        else:
            raise ValueError(f"Unknown profile: {profile}")
        
        self.amplitude[channel] += pulse
    
    def inject_plane_wave(self, direction, wavelength, amplitude, phase=0.0, channel=0):
        """
        Inject a plane wave.
        
        Args:
            direction: (kz, ky, kx) wave vector direction (will be normalized)
            wavelength: Wavelength in grid units
            amplitude: Wave amplitude
            phase: Initial phase
            channel: Target channel
        """
        # Normalize direction
        direction = np.array(direction, dtype=DTYPE)
        direction = direction / np.linalg.norm(direction)
        
        # Wave number
        k = 2 * np.pi / wavelength
        
        grids = self._get_coord_grids()
        k_dot_r = sum(k * d * g for d, g in zip(direction, grids))
        
        wave = amplitude * np.sin(k_dot_r + phase)
        self.amplitude[channel] += wave
    
    def get_slice_xy(self, z_index, channel=0):
        """Get XY slice at given Z."""
        return self.amplitude[channel, z_index, :, :]
    
    def get_slice_xz(self, y_index, channel=0):
        """Get XZ slice at given Y."""
        return self.amplitude[channel, :, y_index, :]
    
    def get_slice_yz(self, x_index, channel=0):
        """Get YZ slice at given X."""
        return self.amplitude[channel, :, :, x_index]
    
    def get_center_slices(self, channel=0):
        """Get all three center slices."""
        cz, cy, cx = self.nz // 2, self.ny // 2, self.nx // 2
        return {
            'xy': self.get_slice_xy(cz, channel),
            'xz': self.get_slice_xz(cy, channel),
            'yz': self.get_slice_yz(cx, channel)
        }
    
    def compute_3d_energy_density(self, channel=0):
        """
        Compute energy density field.
        
        Returns 3D array of energy at each voxel.
        """
        u = self.amplitude[channel]
        v = self.velocity[channel]
        
        # Kinetic energy density: 0.5 * rho * v^2
        kinetic = 0.5 * self._density * v**2
        
        # Potential energy density: 0.5 * c^2 * |grad(u)|^2
        grad_squared = np.zeros_like(u)
        for axis in range(3):
            grad = np.gradient(u, axis=axis)
            grad_squared += grad**2
        potential = 0.5 * self._wave_speed**2 * grad_squared
        
        return kinetic + potential


class VolumeWaveEquation(WaveEquation):
    """
    3D Wave Equation with optimizations for volumetric grids.
    """
    
    def __init__(self, state, dt=WAVE_DT, stencil_order=STENCIL_ORDER_4):
        super().__init__(state, dt, stencil_order)
        
        if state.ndim != 3:
            raise ValueError(f"VolumeWaveEquation requires 3D state, got {state.ndim}D")
    
    def compute_laplacian_optimized(self, u):
        """
        Optimized 3D Laplacian using separable stencils.
        
        Exploits separability: nabla^2 = d^2/dx^2 + d^2/dy^2 + d^2/dz^2
        """
        laplacian = np.zeros_like(u)
        
        for axis in range(3):
            laplacian += self._compute_1d_laplacian(u, axis)
        
        return laplacian / (self.dx ** 2)


class VolumeWaveSimulation:
    """
    High-level 3D wave simulation.
    """
    
    def __init__(self, resolution, dt=WAVE_DT, wave_speed=WAVE_SPEED, 
                 damping=WAVE_DAMPING, stencil_order=STENCIL_ORDER_4):
        """
        Args:
            resolution: Grid size (int for cubic, or (nz, ny, nx) tuple)
            dt: Time step
            wave_speed: Wave propagation speed
            damping: Damping coefficient
            stencil_order: FD accuracy
        """
        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)
        
        self.state = VolumeWaveState(resolution, channels=1)
        self.state.wave_speed = wave_speed
        self.state.damping = damping
        
        self.equation = VolumeWaveEquation(self.state, dt, stencil_order)
        
        # Check stability
        stable, cfl_limit = self.equation.check_stability()
        if not stable:
            print(f"Warning: dt={dt} exceeds CFL limit={cfl_limit:.6f}")
        
        self.step_count = 0
    
    def inject_point_source(self, center, sigma=3.0, amplitude=1.0):
        """Inject Gaussian point source."""
        self.state.inject_spherical_pulse(center, sigma, amplitude, profile="gaussian")
    
    def inject_spherical_shell(self, center, radius, amplitude=1.0):
        """Inject spherical shell source."""
        self.state.inject_spherical_pulse(center, radius, amplitude, profile="shell")
    
    def inject_plane_wave(self, direction, wavelength, amplitude=1.0):
        """Inject plane wave."""
        self.state.inject_plane_wave(direction, wavelength, amplitude)
    
    def set_oscillating_source(self, center, frequency, amplitude, sigma=3.0):
        """
        Set up a continuously oscillating point source.
        
        Args:
            center: (z, y, x) source position
            frequency: Oscillation frequency
            amplitude: Source amplitude
            sigma: Source width
        """
        omega = 2 * np.pi * frequency
        
        # Precompute source mask
        grids = self.state._get_coord_grids()
        dist_squared = sum((g - c)**2 for g, c in zip(grids, center))
        mask = np.exp(-dist_squared / (2 * sigma**2))
        
        def source_fn(state, t):
            source = np.zeros(state.shape, dtype=DTYPE)
            source[0] = amplitude * np.sin(omega * t) * mask
            return source
        
        self.equation.set_source(source_fn)
    
    def step(self, method="leapfrog"):
        """Advance one time step."""
        self.equation.step(method)
        self.step_count += 1
    
    def run(self, steps, method="leapfrog", callback=None):
        """Run simulation for multiple steps."""
        for i in range(steps):
            self.step(method)
            if callback is not None:
                callback(self.state, i)
    
    def get_amplitude(self, channel=0):
        """Get current amplitude field."""
        return self.state.amplitude[channel]
    
    def get_center_slices(self):
        """Get XY, XZ, YZ slices through center."""
        return self.state.get_center_slices()
    
    def get_energy_density(self):
        """Get 3D energy density field."""
        return self.state.compute_3d_energy_density()


def extract_isosurface(volume, threshold, spacing=(1.0, 1.0, 1.0)):
    """
    Extract isosurface from volume using marching cubes.
    
    Args:
        volume: 3D array
        threshold: Iso-value for surface extraction
        spacing: Voxel spacing (dz, dy, dx)
        
    Returns:
        vertices, faces tuple (or None if skimage not available)
    """
    try:
        from skimage import measure
        vertices, faces, _, _ = measure.marching_cubes(
            volume, 
            level=threshold,
            spacing=spacing
        )
        return vertices, faces
    except ImportError:
        print("Warning: skimage not available for isosurface extraction")
        return None


def extract_nodal_surfaces(state, threshold_fraction=0.1, channel=0):
    """
    Extract nodal surfaces (where amplitude ~ 0).
    
    Args:
        state: VolumeWaveState
        threshold_fraction: Fraction of max amplitude for node detection
        channel: Which channel
        
    Returns:
        vertices, faces for nodal surface
    """
    amp = np.abs(state.amplitude[channel])
    max_amp = np.max(amp)
    
    if max_amp < 1e-10:
        return None
    
    # Nodal surface is where |amplitude| crosses threshold
    threshold = threshold_fraction * max_amp
    return extract_isosurface(amp, threshold)


class SphericalHarmonicSource:
    """
    Source based on spherical harmonics Y_l^m.
    
    Creates angular patterns in 3D wave fields.
    """
    
    def __init__(self, center, l, m, frequency, amplitude, radius=10.0):
        """
        Args:
            center: (z, y, x) center position
            l: Angular momentum quantum number (l >= 0)
            m: Magnetic quantum number (-l <= m <= l)
            frequency: Oscillation frequency
            amplitude: Source amplitude
            radius: Characteristic radius
        """
        self.center = center
        self.l = l
        self.m = m
        self.omega = 2 * np.pi * frequency
        self.amplitude = amplitude
        self.radius = radius
        
        self._pattern = None
        self._initialized = False
    
    def _compute_pattern(self, state):
        """Compute spherical harmonic pattern (cached)."""
        if self._initialized:
            return self._pattern
        
        try:
            from scipy.special import sph_harm
        except ImportError:
            # Fallback to simple patterns
            return self._compute_simple_pattern(state)
        
        grids = state._get_coord_grids()
        z, y, x = [g - c for g, c in zip(grids, self.center)]
        
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        
        # Spherical harmonic (real part)
        Y_lm = np.real(sph_harm(self.m, self.l, phi, theta))
        
        # Radial envelope
        radial = np.exp(-r / self.radius)
        
        self._pattern = Y_lm * radial
        self._initialized = True
        return self._pattern
    
    def _compute_simple_pattern(self, state):
        """Simple fallback patterns without scipy."""
        grids = state._get_coord_grids()
        z, y, x = [g - c for g, c in zip(grids, self.center)]
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
        
        # Simple patterns based on l value
        if self.l == 0:
            angular = np.ones_like(r)
        elif self.l == 1:
            if self.m == 0:
                angular = z / r
            else:
                angular = x / r if self.m > 0 else y / r
        elif self.l == 2:
            if self.m == 0:
                angular = (3 * z**2 - r**2) / r**2
            else:
                angular = x * y / r**2
        else:
            angular = np.ones_like(r)
        
        radial = np.exp(-r / self.radius)
        self._pattern = angular * radial
        self._initialized = True
        return self._pattern
    
    def __call__(self, state, t):
        """Generate source term at time t."""
        pattern = self._compute_pattern(state)
        source = np.zeros(state.shape, dtype=DTYPE)
        source[0] = self.amplitude * np.sin(self.omega * t) * pattern
        return source


class StandingWaveAnalyzer3D:
    """
    Analyze 3D standing wave patterns.
    """
    
    def __init__(self, state):
        self.state = state
    
    def compute_spatial_spectrum(self, channel=0):
        """
        Compute 3D spatial frequency spectrum.
        
        Returns power spectrum in k-space.
        """
        u = self.state.amplitude[channel]
        spectrum = np.fft.fftn(u)
        power = np.abs(spectrum)**2
        return np.fft.fftshift(power)
    
    def compute_radial_spectrum(self, channel=0):
        """
        Compute radially averaged power spectrum.
        
        Returns 1D array of power vs |k|.
        """
        power = self.compute_spatial_spectrum(channel)
        
        # Create radial coordinate in k-space
        nz, ny, nx = self.state.dimensions
        kz = np.fft.fftshift(np.fft.fftfreq(nz))
        ky = np.fft.fftshift(np.fft.fftfreq(ny))
        kx = np.fft.fftshift(np.fft.fftfreq(nx))
        KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
        K = np.sqrt(KZ**2 + KY**2 + KX**2)
        
        # Bin by radial wavenumber
        k_bins = np.linspace(0, 0.5, 50)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        radial_power = np.zeros(len(k_centers))
        
        for i in range(len(k_centers)):
            mask = (K >= k_bins[i]) & (K < k_bins[i+1])
            if np.any(mask):
                radial_power[i] = np.mean(power[mask])
        
        return k_centers, radial_power
    
    def find_dominant_modes(self, channel=0, n_modes=5):
        """
        Find dominant spatial modes.
        
        Returns list of (kz, ky, kx, power) tuples.
        """
        power = self.compute_spatial_spectrum(channel)
        
        # Find peaks
        flat_indices = np.argsort(power.flatten())[::-1][:n_modes]
        
        modes = []
        for idx in flat_indices:
            kz, ky, kx = np.unravel_index(idx, power.shape)
            # Convert to actual wavenumbers
            nz, ny, nx = self.state.dimensions
            kz = kz - nz // 2
            ky = ky - ny // 2
            kx = kx - nx // 2
            modes.append((kz, ky, kx, power.flatten()[idx]))
        
        return modes
    
    def compute_node_volume_fraction(self, threshold=0.1, channel=0):
        """
        Compute fraction of volume that is nodal.
        """
        amp = np.abs(self.state.amplitude[channel])
        max_amp = np.max(amp)
        if max_amp < 1e-10:
            return 1.0
        
        normalized = amp / max_amp
        return np.mean(normalized < threshold)
