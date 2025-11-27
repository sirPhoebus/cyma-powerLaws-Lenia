# Additive Synthesis Engine - Phase 2.6
#
# Progressive wave construction via harmonic superposition:
#   u(x, t) = sum_n [ A_n(t) * sin(2*pi*n*f0*t + phi_n) * mode_n(x) ]
#
# Components:
#   - Harmonic series with amplitude and phase control
#   - ADSR envelope for time-varying amplitude
#   - Spatial modes: radial, cartesian, angular
#   - Frequency bundles as "genomes" for pattern generation

import numpy as np
from src.config import DTYPE


class ADSREnvelope:
    """
    Attack-Decay-Sustain-Release envelope generator.
    
    Classic audio synthesis envelope adapted for wave field modulation.
    """
    
    def __init__(self, attack=0.1, decay=0.1, sustain=0.7, release=0.2, 
                 attack_curve=1.0, decay_curve=1.0, release_curve=1.0):
        """
        Args:
            attack: Attack time (0 to peak)
            decay: Decay time (peak to sustain level)
            sustain: Sustain level (0-1)
            release: Release time (sustain to 0)
            attack_curve: Curve exponent for attack (1=linear, <1=fast, >1=slow start)
            decay_curve: Curve exponent for decay
            release_curve: Curve exponent for release
        """
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.attack_curve = attack_curve
        self.decay_curve = decay_curve
        self.release_curve = release_curve
        
        # State
        self._gate_on = False
        self._gate_off_time = None
        self._amplitude_at_release = 0.0
    
    def gate_on(self):
        """Trigger the envelope (note on)."""
        self._gate_on = True
        self._gate_off_time = None
    
    def gate_off(self, current_time):
        """Release the envelope (note off)."""
        self._gate_on = False
        self._gate_off_time = current_time
        self._amplitude_at_release = self.get_amplitude(current_time)
    
    def get_amplitude(self, t, gate_on_time=0.0):
        """
        Get envelope amplitude at time t.
        
        Args:
            t: Current time
            gate_on_time: Time when gate was turned on
            
        Returns:
            Amplitude value [0, 1]
        """
        if not self._gate_on and self._gate_off_time is not None:
            # In release phase
            release_t = t - self._gate_off_time
            if release_t >= self.release:
                return 0.0
            progress = release_t / self.release
            return self._amplitude_at_release * (1 - progress ** self.release_curve)
        
        # ADS phases
        elapsed = t - gate_on_time
        
        if elapsed < self.attack:
            # Attack phase
            progress = elapsed / self.attack
            return progress ** self.attack_curve
        
        elapsed -= self.attack
        
        if elapsed < self.decay:
            # Decay phase
            progress = elapsed / self.decay
            return 1.0 - (1.0 - self.sustain) * (progress ** self.decay_curve)
        
        # Sustain phase
        return self.sustain
    
    def generate_curve(self, duration, sample_rate=100, gate_off_time=None):
        """
        Generate full envelope curve.
        
        Args:
            duration: Total duration
            sample_rate: Samples per time unit
            gate_off_time: When to release (None = hold sustain)
            
        Returns:
            Array of amplitude values
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        self.gate_on()
        if gate_off_time is not None:
            # Will trigger release at gate_off_time
            pass
        
        amplitudes = np.zeros(n_samples)
        for i, time in enumerate(t):
            if gate_off_time is not None and time >= gate_off_time and self._gate_on:
                self.gate_off(time)
            amplitudes[i] = self.get_amplitude(time)
        
        return t, amplitudes


class Harmonic:
    """
    Single harmonic component with amplitude, phase, and envelope.
    """
    
    def __init__(self, ratio, amplitude=1.0, phase=0.0, envelope=None):
        """
        Args:
            ratio: Frequency ratio to fundamental (1 = fundamental, 2 = octave, etc.)
            amplitude: Base amplitude
            phase: Initial phase in radians
            envelope: Optional ADSREnvelope (None = constant amplitude)
        """
        self.ratio = ratio
        self.amplitude = amplitude
        self.phase = phase
        self.envelope = envelope
    
    def get_amplitude(self, t, gate_on_time=0.0):
        """Get modulated amplitude at time t."""
        if self.envelope is None:
            return self.amplitude
        return self.amplitude * self.envelope.get_amplitude(t, gate_on_time)


class HarmonicSeries:
    """
    Collection of harmonics forming a timbre/spectrum.
    """
    
    def __init__(self, fundamental=1.0):
        """
        Args:
            fundamental: Base frequency f0
        """
        self.fundamental = fundamental
        self.harmonics = []
    
    def add_harmonic(self, ratio, amplitude=1.0, phase=0.0, envelope=None):
        """Add a harmonic to the series."""
        self.harmonics.append(Harmonic(ratio, amplitude, phase, envelope))
    
    def add_harmonic_series(self, n_harmonics, amplitude_decay=1.0, 
                            amplitude_fn=None, phase_fn=None):
        """
        Add multiple harmonics with systematic amplitude/phase.
        
        Args:
            n_harmonics: Number of harmonics to add
            amplitude_decay: Exponential decay rate for amplitude
            amplitude_fn: Custom function(n) -> amplitude (overrides decay)
            phase_fn: Custom function(n) -> phase
        """
        for n in range(1, n_harmonics + 1):
            if amplitude_fn is not None:
                amp = amplitude_fn(n)
            else:
                amp = 1.0 / (n ** amplitude_decay)
            
            phase = phase_fn(n) if phase_fn is not None else 0.0
            self.add_harmonic(n, amp, phase)
    
    @classmethod
    def sawtooth(cls, fundamental, n_harmonics=16):
        """Create sawtooth wave harmonic series."""
        series = cls(fundamental)
        series.add_harmonic_series(n_harmonics, amplitude_fn=lambda n: 1.0 / n)
        return series
    
    @classmethod
    def square(cls, fundamental, n_harmonics=16):
        """Create square wave harmonic series (odd harmonics only)."""
        series = cls(fundamental)
        for n in range(1, n_harmonics + 1, 2):  # Odd only
            series.add_harmonic(n, 1.0 / n)
        return series
    
    @classmethod
    def triangle(cls, fundamental, n_harmonics=16):
        """Create triangle wave harmonic series."""
        series = cls(fundamental)
        sign = 1
        for n in range(1, n_harmonics + 1, 2):  # Odd only
            series.add_harmonic(n, sign / (n ** 2))
            sign *= -1
        return series
    
    def get_frequencies(self):
        """Get list of all frequencies in the series."""
        return [self.fundamental * h.ratio for h in self.harmonics]
    
    def get_spectrum(self):
        """Get (frequency, amplitude) pairs."""
        return [(self.fundamental * h.ratio, h.amplitude) for h in self.harmonics]


class SpatialMode:
    """
    Spatial eigenfunction for wave field.
    
    Modes define the spatial pattern that oscillates at each frequency.
    """
    
    def __init__(self, dimensions, mode_type="cartesian"):
        """
        Args:
            dimensions: Field dimensions
            mode_type: "cartesian", "radial", or "angular"
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.mode_type = mode_type
        
        # Precompute coordinate grids
        self._grids = None
        self._center = tuple(d // 2 for d in self.dimensions)
    
    def _get_grids(self):
        """Get coordinate grids."""
        if self._grids is None:
            self._grids = np.ogrid[tuple(slice(0, d) for d in self.dimensions)]
        return self._grids
    
    def compute_cartesian_mode(self, mode_numbers):
        """
        Compute Cartesian mode: product of sines.
        
        phi(x) = prod_i sin(n_i * pi * x_i / L_i)
        
        Args:
            mode_numbers: Tuple of mode numbers for each dimension
        """
        grids = self._get_grids()
        mode = np.ones(self.dimensions, dtype=DTYPE)
        
        for i, (n, g, L) in enumerate(zip(mode_numbers, grids, self.dimensions)):
            mode *= np.sin(n * np.pi * g / L)
        
        return mode
    
    def compute_radial_mode(self, n, m=0):
        """
        Compute radial mode (Bessel-like).
        
        For 2D: J_m(k_n * r) * cos(m * theta)
        Simplified: sin(n * pi * r / R)
        
        Args:
            n: Radial mode number
            m: Angular mode number (0 for radially symmetric)
        """
        grids = self._get_grids()
        
        # Distance from center
        dist_sq = sum((g - c)**2 for g, c in zip(grids, self._center))
        r = np.sqrt(dist_sq)
        R = min(self.dimensions) / 2
        
        # Radial part
        radial = np.sin(n * np.pi * r / R)
        
        if m == 0 or self.ndim < 2:
            return radial
        
        # Angular part (2D only)
        theta = np.arctan2(grids[0] - self._center[0], grids[1] - self._center[1])
        angular = np.cos(m * theta)
        
        return radial * angular
    
    def compute_angular_mode(self, m, radial_envelope=None):
        """
        Compute purely angular mode.
        
        Args:
            m: Angular mode number
            radial_envelope: Optional radial envelope function
        """
        grids = self._get_grids()
        
        if self.ndim < 2:
            return np.ones(self.dimensions, dtype=DTYPE)
        
        theta = np.arctan2(grids[0] - self._center[0], grids[1] - self._center[1])
        mode = np.cos(m * theta)
        
        if radial_envelope is not None:
            dist_sq = sum((g - c)**2 for g, c in zip(grids, self._center))
            r = np.sqrt(dist_sq)
            mode *= radial_envelope(r)
        
        return mode
    
    def compute_mode(self, mode_spec):
        """
        Compute mode from specification.
        
        Args:
            mode_spec: Dict with 'type' and parameters
                       e.g., {'type': 'cartesian', 'n': (2, 3)}
                       or {'type': 'radial', 'n': 2, 'm': 1}
        """
        mode_type = mode_spec.get('type', self.mode_type)
        
        if mode_type == 'cartesian':
            return self.compute_cartesian_mode(mode_spec['n'])
        elif mode_type == 'radial':
            return self.compute_radial_mode(mode_spec['n'], mode_spec.get('m', 0))
        elif mode_type == 'angular':
            return self.compute_angular_mode(mode_spec['m'])
        else:
            raise ValueError(f"Unknown mode type: {mode_type}")


class AdditiveSynthesizer:
    """
    Additive synthesis engine for wave fields.
    
    Generates spatial wave patterns by summing harmonics:
    u(x, t) = sum_n [ A_n(t) * sin(2*pi*n*f0*t + phi_n) * mode_n(x) ]
    """
    
    def __init__(self, dimensions, fundamental=1.0):
        """
        Args:
            dimensions: Field dimensions
            fundamental: Base frequency
        """
        self.dimensions = tuple(dimensions)
        self.ndim = len(self.dimensions)
        self.fundamental = fundamental
        
        self.harmonic_series = HarmonicSeries(fundamental)
        self.spatial_modes = SpatialMode(dimensions)
        
        # Mode assignments: harmonic index -> spatial mode spec
        self._mode_assignments = {}
        
        # State
        self.gate_on_time = 0.0
    
    def set_harmonic_series(self, series):
        """Set the harmonic series."""
        self.harmonic_series = series
    
    def assign_mode(self, harmonic_index, mode_spec):
        """
        Assign a spatial mode to a harmonic.
        
        Args:
            harmonic_index: Which harmonic (0-indexed)
            mode_spec: Mode specification dict
        """
        self._mode_assignments[harmonic_index] = mode_spec
    
    def assign_modes_cartesian(self):
        """Assign Cartesian modes based on harmonic number."""
        for i, h in enumerate(self.harmonic_series.harmonics):
            # Map harmonic ratio to mode numbers
            n = int(h.ratio)
            if self.ndim == 1:
                mode_spec = {'type': 'cartesian', 'n': (n,)}
            elif self.ndim == 2:
                # Distribute across dimensions
                ny = (n - 1) // 3 + 1
                nx = (n - 1) % 3 + 1
                mode_spec = {'type': 'cartesian', 'n': (ny, nx)}
            else:
                mode_spec = {'type': 'cartesian', 'n': tuple([n] * self.ndim)}
            self._mode_assignments[i] = mode_spec
    
    def assign_modes_radial(self):
        """Assign radial modes based on harmonic number."""
        for i, h in enumerate(self.harmonic_series.harmonics):
            n = int(h.ratio)
            m = i % 4  # Cycle through angular modes
            self._mode_assignments[i] = {'type': 'radial', 'n': n, 'm': m}
    
    def trigger(self, time=0.0):
        """Trigger all envelopes (gate on)."""
        self.gate_on_time = time
        for h in self.harmonic_series.harmonics:
            if h.envelope is not None:
                h.envelope.gate_on()
    
    def release(self, time):
        """Release all envelopes (gate off)."""
        for h in self.harmonic_series.harmonics:
            if h.envelope is not None:
                h.envelope.gate_off(time)
    
    def synthesize(self, t):
        """
        Generate wave field at time t.
        
        u(x, t) = sum_n [ A_n(t) * sin(2*pi*f_n*t + phi_n) * mode_n(x) ]
        
        Args:
            t: Current time
            
        Returns:
            2D/3D wave field
        """
        field = np.zeros(self.dimensions, dtype=DTYPE)
        
        for i, harmonic in enumerate(self.harmonic_series.harmonics):
            freq = self.fundamental * harmonic.ratio
            omega = 2 * np.pi * freq
            
            # Time-varying amplitude
            amp = harmonic.get_amplitude(t, self.gate_on_time)
            
            # Temporal oscillation
            temporal = amp * np.sin(omega * t + harmonic.phase)
            
            # Spatial mode
            if i in self._mode_assignments:
                spatial = self.spatial_modes.compute_mode(self._mode_assignments[i])
            else:
                # Default: flat mode
                spatial = np.ones(self.dimensions, dtype=DTYPE)
            
            field += temporal * spatial
        
        return field
    
    def synthesize_sequence(self, duration, dt, gate_off_time=None):
        """
        Generate time sequence of wave fields.
        
        Args:
            duration: Total duration
            dt: Time step
            gate_off_time: When to release (None = hold)
            
        Yields:
            (time, field) pairs
        """
        self.trigger(0.0)
        
        t = 0.0
        while t < duration:
            if gate_off_time is not None and t >= gate_off_time:
                self.release(gate_off_time)
                gate_off_time = None  # Only release once
            
            yield t, self.synthesize(t)
            t += dt


class FrequencyBundle:
    """
    Frequency bundle - a "genome" for pattern generation.
    
    Encapsulates all information needed to recreate a wave pattern:
    - Harmonic series (frequencies, amplitudes, phases)
    - Spatial mode assignments
    - Envelope parameters
    """
    
    def __init__(self, fundamental=1.0):
        self.fundamental = fundamental
        self.harmonics = []  # List of (ratio, amplitude, phase, envelope_params)
        self.mode_assignments = {}  # harmonic_index -> mode_spec
        self.duration = None
        self.gate_off_time = None
    
    def add_harmonic(self, ratio, amplitude=1.0, phase=0.0, envelope_params=None):
        """Add harmonic to bundle."""
        self.harmonics.append((ratio, amplitude, phase, envelope_params))
    
    def set_mode(self, index, mode_spec):
        """Assign spatial mode to harmonic."""
        self.mode_assignments[index] = mode_spec
    
    def create_synthesizer(self, dimensions):
        """
        Instantiate a synthesizer from this bundle.
        
        Args:
            dimensions: Field dimensions
            
        Returns:
            Configured AdditiveSynthesizer
        """
        synth = AdditiveSynthesizer(dimensions, self.fundamental)
        
        for ratio, amp, phase, env_params in self.harmonics:
            envelope = None
            if env_params is not None:
                envelope = ADSREnvelope(**env_params)
            synth.harmonic_series.add_harmonic(ratio, amp, phase, envelope)
        
        for idx, mode_spec in self.mode_assignments.items():
            synth.assign_mode(idx, mode_spec)
        
        return synth
    
    def to_dict(self):
        """Serialize bundle to dictionary."""
        return {
            'fundamental': self.fundamental,
            'harmonics': self.harmonics,
            'mode_assignments': self.mode_assignments,
            'duration': self.duration,
            'gate_off_time': self.gate_off_time
        }
    
    @classmethod
    def from_dict(cls, data):
        """Deserialize bundle from dictionary."""
        bundle = cls(data['fundamental'])
        bundle.harmonics = data['harmonics']
        bundle.mode_assignments = data['mode_assignments']
        bundle.duration = data.get('duration')
        bundle.gate_off_time = data.get('gate_off_time')
        return bundle
    
    @classmethod
    def create_organ_pipe(cls, fundamental, n_harmonics=8):
        """Create organ pipe timbre bundle."""
        bundle = cls(fundamental)
        for n in range(1, n_harmonics + 1):
            amp = 1.0 / n
            env = {'attack': 0.05, 'decay': 0.1, 'sustain': 0.8, 'release': 0.3}
            bundle.add_harmonic(n, amp, 0.0, env)
        return bundle
    
    @classmethod
    def create_bell(cls, fundamental):
        """Create bell-like inharmonic bundle."""
        bundle = cls(fundamental)
        # Bells have inharmonic partials
        ratios = [1.0, 2.0, 2.4, 3.0, 4.0, 4.5, 5.2, 6.0]
        for i, ratio in enumerate(ratios):
            amp = 1.0 / (i + 1)
            # Bells have fast attack, long decay
            env = {'attack': 0.01, 'decay': 2.0, 'sustain': 0.0, 'release': 0.5}
            bundle.add_harmonic(ratio, amp, 0.0, env)
        return bundle
    
    @classmethod
    def create_cymatics_mode(cls, fundamental, mode_numbers):
        """
        Create bundle for specific Chladni-like mode.
        
        Args:
            fundamental: Base frequency
            mode_numbers: (ny, nx) mode numbers for 2D
        """
        bundle = cls(fundamental)
        bundle.add_harmonic(1.0, 1.0, 0.0, 
                          {'attack': 0.5, 'decay': 0.2, 'sustain': 1.0, 'release': 1.0})
        bundle.set_mode(0, {'type': 'cartesian', 'n': mode_numbers})
        return bundle
