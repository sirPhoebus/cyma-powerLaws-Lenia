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

# Diffusion Coefficients (Turing condition: D_V > D_U)
D_U = 0.16  # Activator diffusion
D_V = 0.08  # Inhibitor diffusion (note: for some models this is higher)

# Gray-Scott Parameters
FEED_RATE = 0.055  # F - feed rate
KILL_RATE = 0.062  # k - kill rate

# FitzHugh-Nagumo Parameters
FHN_EPSILON = 0.01
FHN_GAMMA = 1.0
FHN_BETA = 0.5

# Boundary Conditions
BOUNDARY_PERIODIC = "periodic"
BOUNDARY_NEUMANN = "neumann"
BOUNDARY_DIRICHLET = "dirichlet"
DEFAULT_BOUNDARY = BOUNDARY_PERIODIC

# Kernel Parameters
KERNEL_RADIUS = 1  # For discrete Laplacian
GAUSSIAN_SIGMA = 1.0

# Visualization
COLORMAP = "viridis"
FPS_TARGET = 30
