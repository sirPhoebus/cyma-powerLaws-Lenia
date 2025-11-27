# Universal Constructor - Resonant Field Framework
# Phase 0: Mathematical Foundation Validation
# Phase 1: N-Dimensional Substrate

from src.config import *

# Original 2D modules (backward compatibility)
from src.field import ResonantField
from src.kernel import Kernel
from src.reaction import ReactionSystem
from src.simulation import Simulation

# N-Dimensional modules (Phase 1)
from src.field_nd import ResonantFieldND, ResonantField2D, ResonantField3D
from src.kernel_nd import KernelND
from src.simulation_nd import (
    SimulationND, 
    compute_dimensional_scale, 
    scale_diffusion_coefficients
)
