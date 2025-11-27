# Lenia - Continuous Cellular Automata for Artificial Life
#
# Core equation: A(t+dt) = clip(A + dt * G(K * A), 0, 1)
#
# Where:
#   K = convolution kernel (ring-shaped)
#   G = growth function (bell-shaped)
#
# This produces lifelike self-organizing creatures.

import numpy as np
from scipy import ndimage
from scipy import signal
from src.config import DTYPE


# Growth function types
GROWTH_GAUSSIAN = "gaussian"
GROWTH_POLYNOMIAL = "polynomial"
GROWTH_STEP = "step"


class LeniaKernel:
    """
    Ring-shaped convolution kernel for Lenia.
    
    K(r) = exp(-(r - peak_radius)^2 / (2 * width^2))
    
    Normalized so sum = 1.
    """
    
    def __init__(self, radius=13, peak=0.5, width=0.15):
        """
        Args:
            radius: Kernel radius in pixels
            peak: Peak position as fraction of radius (0-1)
            width: Width of ring as fraction
        """
        self.radius = radius
        self.peak = peak
        self.width = width
        
        self.kernel = self._build_kernel()
    
    def _build_kernel(self):
        """Build the kernel array."""
        size = self.radius * 2 + 1
        y, x = np.ogrid[-self.radius:self.radius+1, -self.radius:self.radius+1]
        r = np.sqrt(x**2 + y**2) / self.radius
        
        # Ring-shaped kernel
        kernel = np.exp(-((r - self.peak)**2) / (2 * self.width**2))
        
        # Zero outside radius
        kernel[r > 1] = 0
        
        # Normalize
        kernel = kernel / np.sum(kernel)
        
        return kernel.astype(DTYPE)
    
    def convolve(self, field):
        """Apply kernel convolution."""
        return signal.fftconvolve(field, self.kernel, mode='same')


class GrowthFunction:
    """
    Growth function G that determines cell fate.
    
    G(u) is bell-shaped: positive near target, negative elsewhere.
    
    Cells grow when neighborhood matches target, shrink otherwise.
    """
    
    def __init__(self, mu=0.15, sigma=0.015, growth_type=GROWTH_GAUSSIAN):
        """
        Args:
            mu: Target neighborhood value (center of bell)
            sigma: Width of growth region
            growth_type: Type of growth function
        """
        self.mu = mu
        self.sigma = sigma
        self.growth_type = growth_type
    
    def __call__(self, u):
        """Compute growth value."""
        if self.growth_type == GROWTH_GAUSSIAN:
            return 2 * np.exp(-((u - self.mu)**2) / (2 * self.sigma**2)) - 1
        
        elif self.growth_type == GROWTH_POLYNOMIAL:
            # Smoother polynomial version
            x = (u - self.mu) / (3 * self.sigma)
            return np.clip(1 - x**2, -1, 1) * 2 - 1
        
        elif self.growth_type == GROWTH_STEP:
            # Step function (classic CA style)
            return np.where(np.abs(u - self.mu) < self.sigma, 1.0, -1.0)
        
        else:
            return 2 * np.exp(-((u - self.mu)**2) / (2 * self.sigma**2)) - 1


class LeniaWorld:
    """
    Complete Lenia simulation world.
    
    Supports multiple channels and kernel/growth configurations.
    """
    
    def __init__(self, size=256, dt=0.1, channels=1):
        """
        Args:
            size: Grid size (square)
            dt: Time step
            channels: Number of channels
        """
        if isinstance(size, int):
            size = (size, size)
        
        self.size = size
        self.dt = dt
        self.channels = channels
        
        # Field state
        self.field = np.zeros((channels,) + size, dtype=DTYPE)
        
        # Per-channel kernels and growth functions
        self.kernels = [LeniaKernel() for _ in range(channels)]
        self.growth_fns = [GrowthFunction() for _ in range(channels)]
        
        # Time tracking
        self.time = 0.0
        self.step_count = 0
    
    def set_kernel(self, channel, radius=13, peak=0.5, width=0.15):
        """Set kernel for a channel."""
        self.kernels[channel] = LeniaKernel(radius, peak, width)
    
    def set_growth(self, channel, mu=0.15, sigma=0.015, growth_type=GROWTH_GAUSSIAN):
        """Set growth function for a channel."""
        self.growth_fns[channel] = GrowthFunction(mu, sigma, growth_type)
    
    def step(self):
        """Advance simulation by one time step."""
        for c in range(self.channels):
            # Convolve to get neighborhood values
            U = self.kernels[c].convolve(self.field[c])
            
            # Apply growth function
            G = self.growth_fns[c](U)
            
            # Update field
            self.field[c] = np.clip(self.field[c] + self.dt * G, 0, 1)
        
        self.time += self.dt
        self.step_count += 1
    
    def add_creature(self, center, pattern, channel=0):
        """
        Add a creature pattern to the field.
        
        Args:
            center: (y, x) center position
            pattern: 2D array to place
            channel: Which channel
        """
        cy, cx = center
        h, w = pattern.shape
        
        y_start = max(0, cy - h // 2)
        y_end = min(self.size[0], cy + h // 2 + h % 2)
        x_start = max(0, cx - w // 2)
        x_end = min(self.size[1], cx + w // 2 + w % 2)
        
        py_start = max(0, h // 2 - cy)
        py_end = h - max(0, (cy + h // 2 + h % 2) - self.size[0])
        px_start = max(0, w // 2 - cx)
        px_end = w - max(0, (cx + w // 2 + w % 2) - self.size[1])
        
        self.field[channel, y_start:y_end, x_start:x_end] = \
            pattern[py_start:py_end, px_start:px_end]
    
    def add_random_blob(self, center, radius=15, channel=0):
        """Add a random circular blob."""
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        r = np.sqrt(x**2 + y**2)
        blob = np.exp(-r**2 / (radius/2)**2) * np.random.rand(2*radius+1, 2*radius+1)
        self.add_creature(center, blob, channel)
    
    def clear(self):
        """Clear the field."""
        self.field.fill(0)
        self.time = 0.0
        self.step_count = 0
    
    def compute_mass(self, channel=0):
        """Compute total mass (sum of field)."""
        return np.sum(self.field[channel])
    
    def compute_center_of_mass(self, channel=0):
        """Compute center of mass."""
        f = self.field[channel]
        total = np.sum(f) + 1e-10
        
        y_coords = np.arange(self.size[0])
        x_coords = np.arange(self.size[1])
        
        cy = np.sum(f * y_coords[:, np.newaxis]) / total
        cx = np.sum(f * x_coords[np.newaxis, :]) / total
        
        return cy, cx


# Creature presets - known configurations that produce lifelike behavior
CREATURE_PRESETS = {
    "orbium": {
        "kernel": {"radius": 13, "peak": 0.5, "width": 0.15},
        "growth": {"mu": 0.15, "sigma": 0.015},
        "dt": 0.1,
        "description": "Classic Lenia creature - smooth glider"
    },
    "geminium": {
        "kernel": {"radius": 10, "peak": 0.5, "width": 0.23},
        "growth": {"mu": 0.14, "sigma": 0.014},
        "dt": 0.1,
        "description": "Gemini-like splitting creature"
    },
    "scutium": {
        "kernel": {"radius": 12, "peak": 0.65, "width": 0.12},
        "growth": {"mu": 0.19, "sigma": 0.022},
        "dt": 0.15,
        "description": "Shield-shaped crawler"
    },
    "wave_life": {
        "kernel": {"radius": 15, "peak": 0.4, "width": 0.2},
        "growth": {"mu": 0.12, "sigma": 0.02},
        "dt": 0.05,
        "description": "Wave-like oscillating creature"
    },
    "pulsing": {
        "kernel": {"radius": 8, "peak": 0.5, "width": 0.3},
        "growth": {"mu": 0.25, "sigma": 0.03},
        "dt": 0.2,
        "description": "Pulsating stationary life"
    }
}


def create_orbium_seed(radius=15):
    """
    Create seed pattern for Orbium-like creature.
    
    Returns 2D array.
    """
    size = radius * 2 + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    r = np.sqrt(x**2 + y**2)
    
    # Ring-like initial pattern
    inner = radius * 0.3
    outer = radius * 0.7
    
    pattern = np.zeros((size, size), dtype=DTYPE)
    mask = (r >= inner) & (r <= outer)
    pattern[mask] = 0.5 + 0.5 * np.cos(np.pi * (r[mask] - inner) / (outer - inner))
    
    # Add some asymmetry for motion
    pattern[:radius, :] *= 0.8
    
    return pattern


def create_blob_seed(radius=10, roughness=0.3):
    """Create a rough blob seed."""
    size = radius * 2 + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    r = np.sqrt(x**2 + y**2)
    
    base = np.exp(-r**2 / (radius * 0.6)**2)
    noise = 1 + roughness * (np.random.rand(size, size) - 0.5)
    
    return np.clip(base * noise, 0, 1).astype(DTYPE)


class LeniaSimulation:
    """
    High-level Lenia simulation with presets and analysis.
    """
    
    def __init__(self, size=256, preset="orbium"):
        """
        Args:
            size: Grid size
            preset: Name of creature preset
        """
        self.size = size
        self.preset_name = preset
        
        # Get preset configuration
        if preset in CREATURE_PRESETS:
            config = CREATURE_PRESETS[preset]
        else:
            config = CREATURE_PRESETS["orbium"]
        
        # Create world
        self.world = LeniaWorld(size, dt=config["dt"])
        self.world.set_kernel(0, **config["kernel"])
        self.world.set_growth(0, **config["growth"])
        
        # Tracking
        self.mass_history = []
        self.com_history = []
    
    def seed_creature(self, position=None):
        """Add a creature seed."""
        if position is None:
            position = (self.size // 2, self.size // 2)
        
        seed = create_orbium_seed(radius=15)
        self.world.add_creature(position, seed)
    
    def seed_random(self, n_blobs=3):
        """Add random blobs."""
        for _ in range(n_blobs):
            pos = (
                np.random.randint(50, self.size - 50),
                np.random.randint(50, self.size - 50)
            )
            self.world.add_random_blob(pos, radius=np.random.randint(10, 20))
    
    def step(self):
        """Advance simulation."""
        self.world.step()
        
        # Track
        mass = self.world.compute_mass()
        com = self.world.compute_center_of_mass()
        
        self.mass_history.append(mass)
        self.com_history.append(com)
    
    def run(self, steps):
        """Run multiple steps."""
        for _ in range(steps):
            self.step()
    
    def get_field(self):
        """Get current field."""
        return self.world.field[0]
    
    def is_alive(self, min_mass=10, max_mass=None):
        """Check if creature is still alive."""
        mass = self.world.compute_mass()
        if mass < min_mass:
            return False
        if max_mass is not None and mass > max_mass:
            return False
        return True
    
    def get_velocity(self, window=10):
        """Estimate creature velocity from center of mass history."""
        if len(self.com_history) < window:
            return 0.0, 0.0
        
        recent = self.com_history[-window:]
        dy = recent[-1][0] - recent[0][0]
        dx = recent[-1][1] - recent[0][1]
        
        return dy / window, dx / window
