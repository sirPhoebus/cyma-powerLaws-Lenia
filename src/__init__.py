# Universal Constructor - Resonant Field Framework
# Phase 0: Mathematical Foundation Validation
# Phase 1: N-Dimensional Substrate
# Phase 2: Wave Dynamics Engine

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

# Wave Dynamics modules (Phase 2)
from src.wave_state import (
    WaveState,
    create_wave_state_1d,
    create_wave_state_2d,
    create_wave_state_3d
)
from src.wave_equation import (
    WaveEquation,
    WaveSimulation,
    LAPLACIAN_STENCILS
)
from src.chladni_plate import (
    PlateState,
    BiharmonicOperator,
    ChladniPlate,
    ChladniSimulation
)
from src.volume_wave import (
    VolumeWaveConfig,
    VolumeWaveState,
    VolumeWaveEquation,
    VolumeWaveSimulation,
    SphericalHarmonicSource,
    StandingWaveAnalyzer3D,
    extract_isosurface,
    extract_nodal_surfaces
)
from src.interference import (
    InterferenceAnalyzer,
    StandingWaveDetector,
    TwoSourceInterference,
    compute_beat_frequency,
    compute_group_velocity
)
from src.additive_synthesis import (
    ADSREnvelope,
    Harmonic,
    HarmonicSeries,
    SpatialMode,
    AdditiveSynthesizer,
    FrequencyBundle
)
from src.resonance_detection import (
    ResonanceAnalyzer,
    ResonancePipeline
)
from src.boundary_conditions import (
    BoundaryConfig,
    PerfectlyMatchedLayer,
    BoundaryHandler,
    MixedBoundaryHandler
)
from src.wave_sources import (
    WaveSource,
    PointOscillator,
    LineDriver,
    RingSource,
    BoundaryBow,
    VolumePulse,
    SourceManager
)
from src.phase_space import (
    PhaseSpaceState,
    PhaseSpaceTracker,
    ModalProjector
)
from src.hybrid_system import (
    HybridState,
    CouplingFunction,
    HybridSimulation,
    create_cymatics_rd_hybrid
)
# Phase 3: Pattern Crystallization
from src.energy_functional import (
    EnergyFunctional,
    GinzburgLandauEnergy,
    DoubleWellEnergy,
    SwiftHohenbergEnergy,
    EnergyMinimizer
)
from src.attractor_detection import (
    AttractorType,
    StateRecorder,
    FixedPointDetector,
    LimitCycleDetector,
    StrangeAttractorDetector,
    AttractorClassifier,
    BasinOfAttraction,
    BifurcationDetector
)
from src.pattern_library import (
    PatternSignature,
    PatternEntry,
    PatternLibrary,
    discover_patterns
)
# Lenia - Continuous Cellular Automata
from src.lenia import (
    LeniaKernel,
    GrowthFunction,
    LeniaWorld,
    LeniaSimulation,
    CREATURE_PRESETS,
    create_orbium_seed,
    create_blob_seed
)
