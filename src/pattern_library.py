# Pattern Library - Phase 3.3
#
# Catalog of discovered stable patterns:
#   - Indexed by: Frequency signature, symmetry group, topological features
#   - Each pattern stores: The minimal frequency bundle to recreate it
#   - Purpose: Building vocabulary for the Constructor

import numpy as np
import json
import hashlib
from pathlib import Path
from src.config import DTYPE


class PatternSignature:
    """
    Unique signature for a pattern based on its properties.
    
    Used for indexing and comparing patterns.
    """
    
    def __init__(self, pattern):
        """
        Args:
            pattern: 2D or 3D field array
        """
        self.shape = pattern.shape
        self.ndim = len(self.shape)
        
        # Compute signatures
        self.frequency_signature = self._compute_frequency_signature(pattern)
        self.symmetry_type = self._detect_symmetry(pattern)
        self.topological_features = self._compute_topology(pattern)
        self.energy = np.sum(pattern**2)
        self.hash = self._compute_hash(pattern)
    
    def _compute_frequency_signature(self, pattern):
        """Extract dominant spatial frequencies."""
        spectrum = np.abs(np.fft.fftn(pattern))
        spectrum = np.fft.fftshift(spectrum)
        
        # Find peaks in spectrum
        flat = spectrum.flatten()
        threshold = np.percentile(flat, 95)
        peaks = flat > threshold
        
        # Get peak frequencies
        peak_indices = np.where(peaks)[0]
        peak_coords = np.unravel_index(peak_indices, spectrum.shape)
        
        # Convert to normalized frequencies
        center = tuple(s // 2 for s in self.shape)
        frequencies = []
        for coords in zip(*peak_coords):
            freq = tuple((c - center[i]) / self.shape[i] 
                        for i, c in enumerate(coords))
            power = spectrum[coords]
            frequencies.append((freq, float(power)))
        
        # Sort by power and keep top frequencies
        frequencies.sort(key=lambda x: -x[1])
        return frequencies[:10]
    
    def _detect_symmetry(self, pattern):
        """Detect symmetry type of pattern."""
        symmetries = []
        
        # Check reflection symmetry
        if self.ndim >= 2:
            # Horizontal
            if np.allclose(pattern, np.flip(pattern, axis=0), rtol=0.1):
                symmetries.append("horizontal_mirror")
            # Vertical
            if np.allclose(pattern, np.flip(pattern, axis=1), rtol=0.1):
                symmetries.append("vertical_mirror")
        
        # Check rotational symmetry (for 2D)
        if self.ndim == 2:
            for angle in [90, 180, 270]:
                rotated = np.rot90(pattern, k=angle // 90)
                if rotated.shape == pattern.shape:
                    if np.allclose(pattern, rotated, rtol=0.1):
                        symmetries.append(f"C{360 // angle}")
        
        # Check radial symmetry
        if self._check_radial_symmetry(pattern):
            symmetries.append("radial")
        
        if not symmetries:
            symmetries.append("asymmetric")
        
        return symmetries
    
    def _check_radial_symmetry(self, pattern, tolerance=0.15):
        """Check if pattern has radial symmetry."""
        if self.ndim != 2:
            return False
        
        center = (self.shape[0] // 2, self.shape[1] // 2)
        max_radius = min(self.shape) // 2
        
        # Sample at different angles for each radius
        for r in range(5, max_radius, 5):
            values = []
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                y = int(center[0] + r * np.cos(angle))
                x = int(center[1] + r * np.sin(angle))
                if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]:
                    values.append(pattern[y, x])
            
            if len(values) > 4:
                if np.std(values) / (np.mean(np.abs(values)) + 1e-10) > tolerance:
                    return False
        
        return True
    
    def _compute_topology(self, pattern):
        """Compute topological features (connected components, holes)."""
        # Threshold to binary
        threshold = (np.max(pattern) + np.min(pattern)) / 2
        binary = pattern > threshold
        
        features = {
            'n_high_regions': self._count_components(binary),
            'n_low_regions': self._count_components(~binary),
            'euler_characteristic': None  # Could compute Euler number
        }
        
        return features
    
    def _count_components(self, binary):
        """Count connected components in binary image."""
        try:
            from scipy import ndimage
            labeled, n_components = ndimage.label(binary)
            return n_components
        except:
            return -1
    
    def _compute_hash(self, pattern):
        """Compute hash for pattern identification."""
        # Downsample for stable hash
        if self.ndim == 2:
            small = pattern[::max(1, self.shape[0]//16), 
                          ::max(1, self.shape[1]//16)]
        else:
            small = pattern[::max(1, self.shape[0]//8), 
                          ::max(1, self.shape[1]//8),
                          ::max(1, self.shape[2]//8)]
        
        # Quantize and hash
        quantized = np.round(small * 100).astype(np.int32)
        return hashlib.md5(quantized.tobytes()).hexdigest()[:16]
    
    def similarity(self, other):
        """
        Compute similarity to another signature.
        
        Returns value in [0, 1] where 1 = identical.
        """
        if self.shape != other.shape:
            return 0.0
        
        # Compare frequency signatures
        freq_sim = self._compare_frequencies(other)
        
        # Compare symmetries
        sym_overlap = len(set(self.symmetry_type) & set(other.symmetry_type))
        sym_total = len(set(self.symmetry_type) | set(other.symmetry_type))
        sym_sim = sym_overlap / max(sym_total, 1)
        
        # Compare topology
        topo_sim = 1.0 if self.topological_features == other.topological_features else 0.5
        
        return 0.5 * freq_sim + 0.3 * sym_sim + 0.2 * topo_sim
    
    def _compare_frequencies(self, other):
        """Compare frequency signatures."""
        if not self.frequency_signature or not other.frequency_signature:
            return 0.5
        
        # Compare dominant frequencies
        self_freqs = set(f[0] for f in self.frequency_signature[:5])
        other_freqs = set(f[0] for f in other.frequency_signature[:5])
        
        overlap = len(self_freqs & other_freqs)
        total = len(self_freqs | other_freqs)
        
        return overlap / max(total, 1)
    
    def to_dict(self):
        """Serialize signature to dictionary."""
        return {
            'shape': self.shape,
            'frequency_signature': self.frequency_signature,
            'symmetry_type': self.symmetry_type,
            'topological_features': self.topological_features,
            'energy': float(self.energy),
            'hash': self.hash
        }
    
    @classmethod
    def from_dict(cls, data):
        """Deserialize signature from dictionary."""
        sig = object.__new__(cls)
        sig.shape = tuple(data['shape'])
        sig.ndim = len(sig.shape)
        sig.frequency_signature = data['frequency_signature']
        sig.symmetry_type = data['symmetry_type']
        sig.topological_features = data['topological_features']
        sig.energy = data['energy']
        sig.hash = data['hash']
        return sig


class PatternEntry:
    """
    Single entry in the pattern library.
    """
    
    def __init__(self, pattern, name=None, frequency_bundle=None, 
                 metadata=None):
        """
        Args:
            pattern: The pattern field
            name: Human-readable name
            frequency_bundle: FrequencyBundle to recreate pattern
            metadata: Additional metadata dict
        """
        self.pattern = np.array(pattern, dtype=DTYPE)
        self.name = name or f"pattern_{id(self)}"
        self.signature = PatternSignature(self.pattern)
        self.frequency_bundle = frequency_bundle
        self.metadata = metadata or {}
        
        # Add automatic metadata
        self.metadata['shape'] = self.pattern.shape
        self.metadata['energy'] = float(self.signature.energy)
        self.metadata['symmetry'] = self.signature.symmetry_type
    
    def to_dict(self):
        """Serialize entry to dictionary."""
        bundle_data = None
        if self.frequency_bundle is not None:
            bundle_data = self.frequency_bundle.to_dict()
        
        return {
            'name': self.name,
            'pattern_hash': self.signature.hash,
            'signature': self.signature.to_dict(),
            'frequency_bundle': bundle_data,
            'metadata': self.metadata
        }
    
    def matches(self, other_pattern, threshold=0.9):
        """Check if this entry matches another pattern."""
        other_sig = PatternSignature(other_pattern)
        return self.signature.similarity(other_sig) >= threshold


class PatternLibrary:
    """
    Catalog of discovered stable patterns.
    
    Supports:
    - Adding patterns with automatic signature computation
    - Searching by similarity
    - Indexing by symmetry, frequency, topology
    - Persistence to disk
    """
    
    def __init__(self, name="default"):
        """
        Args:
            name: Library name for persistence
        """
        self.name = name
        self.patterns = {}  # hash -> PatternEntry
        
        # Indices for fast lookup
        self._symmetry_index = {}  # symmetry_type -> set of hashes
        self._topology_index = {}  # topology_key -> set of hashes
    
    def add(self, pattern, name=None, frequency_bundle=None, 
            metadata=None, check_duplicate=True):
        """
        Add pattern to library.
        
        Args:
            pattern: Pattern field
            name: Optional name
            frequency_bundle: Optional bundle to recreate
            metadata: Optional metadata
            check_duplicate: If True, skip if similar pattern exists
            
        Returns:
            PatternEntry or None if duplicate
        """
        entry = PatternEntry(pattern, name, frequency_bundle, metadata)
        
        if check_duplicate:
            similar = self.find_similar(pattern, threshold=0.95)
            if similar:
                return None
        
        # Add to main storage
        self.patterns[entry.signature.hash] = entry
        
        # Update indices
        for sym in entry.signature.symmetry_type:
            if sym not in self._symmetry_index:
                self._symmetry_index[sym] = set()
            self._symmetry_index[sym].add(entry.signature.hash)
        
        topo_key = str(entry.signature.topological_features)
        if topo_key not in self._topology_index:
            self._topology_index[topo_key] = set()
        self._topology_index[topo_key].add(entry.signature.hash)
        
        return entry
    
    def get(self, pattern_hash):
        """Get pattern by hash."""
        return self.patterns.get(pattern_hash)
    
    def find_similar(self, pattern, threshold=0.8, max_results=10):
        """
        Find patterns similar to input.
        
        Args:
            pattern: Query pattern
            threshold: Minimum similarity
            max_results: Maximum number of results
            
        Returns:
            List of (PatternEntry, similarity) tuples
        """
        query_sig = PatternSignature(pattern)
        
        results = []
        for entry in self.patterns.values():
            sim = entry.signature.similarity(query_sig)
            if sim >= threshold:
                results.append((entry, sim))
        
        results.sort(key=lambda x: -x[1])
        return results[:max_results]
    
    def find_by_symmetry(self, symmetry_type):
        """Find all patterns with given symmetry."""
        if symmetry_type not in self._symmetry_index:
            return []
        
        return [self.patterns[h] for h in self._symmetry_index[symmetry_type]]
    
    def find_by_topology(self, n_high_regions=None, n_low_regions=None):
        """Find patterns matching topological constraints."""
        results = []
        for entry in self.patterns.values():
            topo = entry.signature.topological_features
            
            if n_high_regions is not None:
                if topo.get('n_high_regions') != n_high_regions:
                    continue
            
            if n_low_regions is not None:
                if topo.get('n_low_regions') != n_low_regions:
                    continue
            
            results.append(entry)
        
        return results
    
    def list_all(self):
        """List all patterns in library."""
        return list(self.patterns.values())
    
    def __len__(self):
        return len(self.patterns)
    
    def save(self, directory="pattern_library"):
        """
        Save library to disk.
        
        Args:
            directory: Directory to save to
        """
        path = Path(directory) / self.name
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index = {
            'name': self.name,
            'n_patterns': len(self.patterns),
            'patterns': {}
        }
        
        for hash_id, entry in self.patterns.items():
            # Save pattern data
            np.save(path / f"{hash_id}.npy", entry.pattern)
            
            # Add to index
            index['patterns'][hash_id] = entry.to_dict()
        
        with open(path / "index.json", 'w') as f:
            json.dump(index, f, indent=2)
    
    def load(self, directory="pattern_library"):
        """
        Load library from disk.
        
        Args:
            directory: Directory to load from
        """
        path = Path(directory) / self.name
        
        with open(path / "index.json", 'r') as f:
            index = json.load(f)
        
        self.patterns.clear()
        self._symmetry_index.clear()
        self._topology_index.clear()
        
        for hash_id, entry_data in index['patterns'].items():
            pattern = np.load(path / f"{hash_id}.npy")
            
            entry = PatternEntry(
                pattern,
                name=entry_data['name'],
                metadata=entry_data.get('metadata', {})
            )
            
            self.patterns[hash_id] = entry
            
            # Rebuild indices
            for sym in entry.signature.symmetry_type:
                if sym not in self._symmetry_index:
                    self._symmetry_index[sym] = set()
                self._symmetry_index[sym].add(hash_id)
            
            topo_key = str(entry.signature.topological_features)
            if topo_key not in self._topology_index:
                self._topology_index[topo_key] = set()
            self._topology_index[topo_key].add(hash_id)
    
    def summary(self):
        """Get library summary statistics."""
        symmetries = {}
        for sym, hashes in self._symmetry_index.items():
            symmetries[sym] = len(hashes)
        
        return {
            'name': self.name,
            'total_patterns': len(self.patterns),
            'symmetry_distribution': symmetries,
            'unique_topologies': len(self._topology_index)
        }


def discover_patterns(simulation_fn, n_trials=100, library=None,
                     initial_fn=None, steps=1000, 
                     stability_threshold=0.01):
    """
    Automated pattern discovery by running simulations.
    
    Args:
        simulation_fn: Function(initial) -> final state
        n_trials: Number of random trials
        library: PatternLibrary to add discoveries to
        initial_fn: Function() -> random initial condition
        steps: Simulation steps per trial
        stability_threshold: Velocity threshold for stable patterns
        
    Returns:
        List of discovered PatternEntry objects
    """
    if library is None:
        library = PatternLibrary("discoveries")
    
    discovered = []
    
    for i in range(n_trials):
        if initial_fn is not None:
            initial = initial_fn()
        else:
            continue
        
        final = simulation_fn(initial)
        
        # Check if stable (low velocity)
        # This assumes simulation_fn returns the final state
        # In practice, would need velocity check
        
        entry = library.add(
            final,
            name=f"discovery_{i}",
            metadata={'trial': i}
        )
        
        if entry is not None:
            discovered.append(entry)
    
    return discovered
