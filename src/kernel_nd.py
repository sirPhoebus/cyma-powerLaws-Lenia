# KernelND - N-Dimensional Convolution Operators
# Supports kernels in arbitrary spatial dimensions

import numpy as np
from scipy import ndimage
from scipy.fft import fftn, ifftn
from scipy.special import jv as bessel_j
from src.config import DTYPE


class KernelND:
    """
    N-Dimensional Kernel for spatial convolutions.
    
    Kernels define the interaction rules in the field.
    They can be composed, scaled, and applied via direct or FFT convolution.
    """
    
    def __init__(self, data):
        """Initialize kernel with explicit weight array."""
        self.data = np.asarray(data, dtype=DTYPE)
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        self._spectrum_cache = {}
    
    @classmethod
    def laplacian(cls, ndim=2):
        """
        N-dimensional discrete Laplacian operator.
        
        For each dimension, uses the standard [1, -2, 1] stencil.
        Total center weight = -2 * ndim
        """
        # Build the kernel by summing 1D Laplacians along each axis
        size = 3
        shape = (size,) * ndim
        kernel = np.zeros(shape, dtype=DTYPE)
        center = (1,) * ndim
        
        # Center value
        kernel[center] = -2.0 * ndim
        
        # Neighbors along each axis
        for axis in range(ndim):
            for offset in [-1, 1]:
                idx = list(center)
                idx[axis] = center[axis] + offset
                kernel[tuple(idx)] = 1.0
        
        return cls(kernel)
    
    @classmethod
    def gaussian(cls, sigma, ndim=2, radius=None):
        """
        N-dimensional Gaussian kernel.
        
        G(r) = (1 / (2*pi*sigma^2)^(n/2)) * exp(-r^2 / (2*sigma^2))
        """
        if radius is None:
            radius = int(3 * sigma) + 1
        
        size = 2 * radius + 1
        shape = (size,) * ndim
        
        # Create coordinate grids centered at origin
        coords = [np.arange(size) - radius for _ in range(ndim)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        # Squared distance from center
        r_squared = sum(g**2 for g in grids)
        
        # Gaussian
        kernel = np.exp(-r_squared / (2 * sigma**2))
        kernel /= kernel.sum()  # Normalize
        
        return cls(kernel)
    
    @classmethod
    def mexican_hat(cls, sigma, ndim=2, radius=None):
        """
        N-dimensional Mexican hat (Laplacian of Gaussian).
        
        Implements Local Activation, Long-range Inhibition (LALI).
        """
        if radius is None:
            radius = int(4 * sigma) + 1
        
        size = 2 * radius + 1
        shape = (size,) * ndim
        
        coords = [np.arange(size) - radius for _ in range(ndim)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        r_squared = sum(g**2 for g in grids)
        sigma_squared = sigma**2
        
        # Mexican hat: (1 - r^2/(2*sigma^2)) * exp(-r^2/(2*sigma^2))
        # Generalized for N dimensions
        kernel = (1 - r_squared / (ndim * sigma_squared)) * np.exp(-r_squared / (2 * sigma_squared))
        kernel -= kernel.mean()  # Zero-sum for stability
        
        return cls(kernel)
    
    @classmethod
    def ring(cls, r_inner, r_outer, ndim=2, radius=None):
        """
        N-dimensional ring (hyperspherical shell) kernel.
        
        Active between r_inner and r_outer, zero elsewhere.
        Lenia-style activation kernel.
        """
        if radius is None:
            radius = int(r_outer) + 1
        
        size = 2 * radius + 1
        
        coords = [np.arange(size) - radius for _ in range(ndim)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        r = np.sqrt(sum(g**2 for g in grids))
        
        kernel = ((r >= r_inner) & (r <= r_outer)).astype(DTYPE)
        
        if kernel.sum() > 0:
            kernel /= kernel.sum()
        
        return cls(kernel)
    
    @classmethod
    def power_law(cls, alpha, ndim=2, radius=None, cutoff=1e-6):
        """
        N-dimensional power-law kernel for long-range interactions.
        
        K(r) = 1 / r^alpha  (with cutoff at center)
        
        Power-law interactions are fundamental to:
        - Scale-free networks
        - Fractal patterns
        - Long-range correlations in critical systems
        
        Args:
            alpha: Power-law exponent (typically 1-3)
            ndim: Number of dimensions
            radius: Extent of kernel
            cutoff: Minimum value to avoid singularity at r=0
        """
        if radius is None:
            radius = 10
        
        size = 2 * radius + 1
        
        coords = [np.arange(size) - radius for _ in range(ndim)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        r = np.sqrt(sum(g**2 for g in grids))
        r = np.maximum(r, cutoff)  # Avoid division by zero
        
        kernel = 1.0 / (r ** alpha)
        kernel[tuple([radius] * ndim)] = 0  # Zero at center (or set to max neighbor)
        
        # Normalize
        if kernel.sum() > 0:
            kernel /= kernel.sum()
        
        return cls(kernel)
    
    @classmethod
    def exponential_decay(cls, decay_rate, ndim=2, radius=None):
        """
        N-dimensional exponential decay kernel.
        
        K(r) = exp(-decay_rate * r)
        """
        if radius is None:
            radius = int(5 / decay_rate) + 1
        
        size = 2 * radius + 1
        
        coords = [np.arange(size) - radius for _ in range(ndim)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        r = np.sqrt(sum(g**2 for g in grids))
        
        kernel = np.exp(-decay_rate * r)
        kernel /= kernel.sum()
        
        return cls(kernel)
    
    @classmethod
    def bessel(cls, order, scale, ndim=2, radius=None):
        """
        N-dimensional Bessel function kernel.
        
        K(r) = J_order(r / scale)
        """
        if radius is None:
            radius = int(4 * scale) + 1
        
        size = 2 * radius + 1
        
        coords = [np.arange(size) - radius for _ in range(ndim)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        r = np.sqrt(sum(g**2 for g in grids))
        r_scaled = r / scale
        r_scaled = np.maximum(r_scaled, 1e-10)
        
        kernel = bessel_j(order, r_scaled)
        
        return cls(kernel)
    
    @classmethod
    def from_radial_profile(cls, profile_func, ndim=2, radius=10):
        """
        Create kernel from a radial profile function.
        
        Args:
            profile_func: Function f(r) that returns kernel value at radius r
            ndim: Number of dimensions
            radius: Extent of the kernel
        """
        size = 2 * radius + 1
        
        coords = [np.arange(size) - radius for _ in range(ndim)]
        grids = np.meshgrid(*coords, indexing='ij')
        
        r = np.sqrt(sum(g**2 for g in grids))
        kernel = profile_func(r)
        
        return cls(kernel)
    
    def convolve(self, field_data, mode="wrap"):
        """
        Apply kernel via direct convolution.
        
        Args:
            field_data: N-dimensional array to convolve
            mode: 'wrap' for periodic, 'constant' for zero-padded, 'nearest', etc.
        """
        return ndimage.convolve(field_data, self.data, mode=mode)
    
    def convolve_fft(self, field_data):
        """
        Apply kernel via FFT (efficient for large kernels).
        Assumes periodic boundary conditions.
        """
        field_shape = field_data.shape
        cache_key = field_shape
        
        if cache_key not in self._spectrum_cache:
            # Pad kernel to field size and compute spectrum
            padded = np.zeros(field_shape, dtype=DTYPE)
            
            # Place kernel at center
            slices = tuple(slice(0, s) for s in self.shape)
            padded[slices] = self.data
            
            # Roll to put center at origin
            for axis in range(self.ndim):
                padded = np.roll(padded, -(self.shape[axis] // 2), axis=axis)
            
            self._spectrum_cache[cache_key] = fftn(padded)
        
        kernel_spectrum = self._spectrum_cache[cache_key]
        field_spectrum = fftn(field_data)
        
        return np.real(ifftn(field_spectrum * kernel_spectrum))
    
    def __add__(self, other):
        """Kernel superposition (sum)."""
        if not isinstance(other, KernelND):
            return NotImplemented
        
        if self.ndim != other.ndim:
            raise ValueError("Cannot add kernels with different dimensions")
        
        # Align to maximum size
        max_shape = tuple(max(s, o) for s, o in zip(self.shape, other.shape))
        result = np.zeros(max_shape, dtype=DTYPE)
        
        # Center both kernels in result
        def center_slice(kernel_shape, result_shape):
            return tuple(slice((r - k) // 2, (r - k) // 2 + k) 
                        for k, r in zip(kernel_shape, result_shape))
        
        result[center_slice(self.shape, max_shape)] += self.data
        result[center_slice(other.shape, max_shape)] += other.data
        
        return KernelND(result)
    
    def __mul__(self, scalar):
        """Kernel scaling."""
        return KernelND(self.data * scalar)
    
    def __rmul__(self, scalar):
        """Kernel scaling (reverse)."""
        return self.__mul__(scalar)
    
    def normalize(self):
        """Return normalized kernel (sum = 1)."""
        total = np.sum(self.data)
        if abs(total) > 1e-10:
            return KernelND(self.data / total)
        return KernelND(self.data.copy())
    
    def zero_center(self):
        """Return zero-centered kernel (mean = 0)."""
        return KernelND(self.data - self.data.mean())
    
    def get_radial_profile(self):
        """Extract the radial profile of the kernel."""
        center = tuple(s // 2 for s in self.shape)
        
        coords = [np.arange(s) - c for s, c in zip(self.shape, center)]
        grids = np.meshgrid(*coords, indexing='ij')
        r = np.sqrt(sum(g**2 for g in grids))
        
        r_int = r.astype(int).flatten()
        values = self.data.flatten()
        
        max_r = int(np.max(r_int)) + 1
        radial_sum = np.bincount(r_int, weights=values, minlength=max_r)
        radial_count = np.bincount(r_int, minlength=max_r)
        
        radial_mean = np.zeros(max_r)
        nonzero = radial_count > 0
        radial_mean[nonzero] = radial_sum[nonzero] / radial_count[nonzero]
        
        return np.arange(max_r), radial_mean
    
    def __repr__(self):
        return f"KernelND(shape={self.shape}, ndim={self.ndim})"
