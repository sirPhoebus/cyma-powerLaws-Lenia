# Global Configuration - All constants and parameters defined here
# No hardcoded values in other modules - everything references this file

import numpy as np

# Field Dimensions (N-dimensional support)
FIELD_DIMENSIONS = (256, 256)  # 2D default, can be (64, 64, 64) for 3D, etc.
FIELD_CHANNELS = 2  # u (activator), v (inhibitor)

# Legacy 2D aliases (for backward compatibility)
FIELD_WIDTH = FIELD_DIMENSIONS[-1] if len(FIELD_DIMENSIONS) >= 1 else 256
FIELD_HEIGHT = FIELD_DIMENSIONS[-2] if len(FIELD_DIMENSIONS) >= 2 else 256

# Numerical Precision
DTYPE = np.float64

# Time Integration
DT = 1.0  # Time step
TOTAL_STEPS = 10000

# Diffusion Coefficients
# NOTE: Gray-Scott is a SUBSTRATE-AUTOCATALYST system, not activator-inhibitor
# u = substrate (diffuses FASTER) - gets consumed
# v = autocatalyst (diffuses SLOWER) - forms patterns
# This is CORRECT for Gray-Scott: D_U > D_V
# For classic Turing (activator-inhibitor), you'd want D_inhibitor > D_activator
D_U = 0.16  # Substrate diffusion (faster)
D_V = 0.08  # Autocatalyst diffusion (slower, pattern-forming)

# Gray-Scott Parameters
FEED_RATE = 0.055  # F - feed rate
KILL_RATE = 0.062  # k - kill rate

# FitzHugh-Nagumo Parameters
FHN_EPSILON = 0.01
FHN_GAMMA = 1.0
FHN_BETA = 0.5

# Boundary Conditions
BOUNDARY_PERIODIC = "periodic"
BOUNDARY_NEUMANN = "neumann"       # Zero-flux (reflective)
BOUNDARY_DIRICHLET = "dirichlet"   # Fixed value (absorbing)
BOUNDARY_ABSORBING = "absorbing"   # Alias for Dirichlet with zero
DEFAULT_BOUNDARY = BOUNDARY_PERIODIC

# Dirichlet boundary value (for absorbing boundaries)
DIRICHLET_VALUE = 0.0

# Kernel Parameters
KERNEL_RADIUS = 1  # For discrete Laplacian
GAUSSIAN_SIGMA = 1.0

# Visualization
COLORMAP = "viridis"
FPS_TARGET = 30
