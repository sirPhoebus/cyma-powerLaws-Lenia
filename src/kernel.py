# Kernel Algebra - Convolutional operators as first-class objects
# Kernels define the "forces" that propagate influence across space

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2
from src.config import DTYPE, KERNEL_RADIUS, GAUSSIAN_SIGMA


class Kernel:
    """
    A kernel represents a spatial interaction operator.
    
    Kernels can be composed, superposed, and applied to fields.
    They form a Banach algebra under convolution.
    """
    
    def __init__(self, data):
        """Initialize kernel with explicit weight matrix."""
        self.data = np.asarray(data, dtype=DTYPE)
        self.shape = self.data.shape
        self._spectrum = None  # Cached Fourier transform
    
    @classmethod
    def laplacian_2d(cls):
        """
        Standard 5-point discrete Laplacian operator.
        Approximates nabla^2 for diffusion.
        """
        data = np.array([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=DTYPE)
        return cls(data)
    
    @classmethod
    def laplacian_9point(cls):
        """
        9-point discrete Laplacian (more isotropic).
        """
        data = np.array([
            [0.25, 0.5, 0.25],
            [0.5, -3.0, 0.5],
            [0.25, 0.5, 0.25]
        ], dtype=DTYPE)
        return cls(data)
    
    @classmethod
    def gaussian(cls, sigma=GAUSSIAN_SIGMA, radius=None):
        """
        Gaussian kernel for smoothing/diffusion.
        
        K(x) = (1 / 2*pi*sigma^2) * exp(-|x|^2 / 2*sigma^2)
        """
        if radius is None:
            radius = int(3 * sigma) + 1
        size = 2 * radius + 1
        x = np.arange(size) - radius
        y = np.arange(size) - radius
        xx, yy = np.meshgrid(x, y)
        data = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        data /= data.sum()  # Normalize
        return cls(data)
    
    @classmethod
    def mexican_hat(cls, sigma=GAUSSIAN_SIGMA, radius=None):
        """
        Mexican hat (Laplacian of Gaussian) - classic LALI kernel.
        Implements local activation, long-range inhibition.
        """
        if radius is None:
            radius = int(4 * sigma) + 1
        size = 2 * radius + 1
        x = np.arange(size) - radius
        y = np.arange(size) - radius
        xx, yy = np.meshgrid(x, y)
        r2 = xx**2 + yy**2
        sigma2 = sigma**2
        data = (1 - r2 / (2 * sigma2)) * np.exp(-r2 / (2 * sigma2))
        data -= data.mean()  # Zero-sum for stability
        return cls(data)
    
    @classmethod
    def ring(cls, r_inner, r_outer, radius=None):
        """
        Ring (donut) kernel - Lenia-style activation.
        
        Active between r_inner and r_outer, zero elsewhere.
        """
        if radius is None:
            radius = int(r_outer) + 1
        size = 2 * radius + 1
        x = np.arange(size) - radius
        y = np.arange(size) - radius
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        data = ((r >= r_inner) & (r <= r_outer)).astype(DTYPE)
        if data.sum() > 0:
            data /= data.sum()
        return cls(data)
    
    @classmethod
    def bessel(cls, order, scale, radius=None):
        """
        Bessel function kernel for radial harmonics.
        """
        from scipy.special import jv
        if radius is None:
            radius = int(4 * scale) + 1
        size = 2 * radius + 1
        x = np.arange(size) - radius
        y = np.arange(size) - radius
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2) / scale
        r = np.maximum(r, 1e-10)  # Avoid division by zero
        data = jv(order, r)
        return cls(data)
    
    def convolve(self, field_channel, mode="wrap"):
        """
        Apply kernel to a 2D field via convolution.
        
        mode: 'wrap' for periodic, 'constant' for zero-padded
        """
        return ndimage.convolve(field_channel, self.data, mode=mode)
    
    def convolve_fft(self, field_channel):
        """
        Apply kernel via FFT (efficient for large kernels).
        Assumes periodic boundaries.
        """
        # Pad kernel to field size
        padded = np.zeros_like(field_channel)
        kh, kw = self.shape
        padded[:kh, :kw] = self.data
        # Roll to center
        padded = np.roll(padded, -(kh // 2), axis=0)
        padded = np.roll(padded, -(kw // 2), axis=1)
        # Convolution via FFT
        return np.real(ifft2(fft2(field_channel) * fft2(padded)))
    
    def get_spectrum(self, field_shape):
        """
        Get kernel spectrum at specified field size (for FFT convolution).
        """
        if self._spectrum is None or self._spectrum.shape != field_shape:
            padded = np.zeros(field_shape, dtype=DTYPE)
            kh, kw = self.shape
            fh, fw = field_shape
            # Center kernel in padded array
            start_h = (fh - kh) // 2
            start_w = (fw - kw) // 2
            padded[start_h:start_h + kh, start_w:start_w + kw] = self.data
            padded = np.roll(padded, -start_h - kh // 2, axis=0)
            padded = np.roll(padded, -start_w - kw // 2, axis=1)
            self._spectrum = fft2(padded)
        return self._spectrum
    
    def __add__(self, other):
        """Kernel superposition (sum)."""
        if isinstance(other, Kernel):
            # Align kernels to same size
            max_h = max(self.shape[0], other.shape[0])
            max_w = max(self.shape[1], other.shape[1])
            result = np.zeros((max_h, max_w), dtype=DTYPE)
            
            sh, sw = self.shape
            oh, ow = other.shape
            
            # Center both kernels
            result[(max_h - sh) // 2:(max_h - sh) // 2 + sh,
                   (max_w - sw) // 2:(max_w - sw) // 2 + sw] += self.data
            result[(max_h - oh) // 2:(max_h - oh) // 2 + oh,
                   (max_w - ow) // 2:(max_w - ow) // 2 + ow] += other.data
            
            return Kernel(result)
        return NotImplemented
    
    def __mul__(self, scalar):
        """Kernel scaling."""
        return Kernel(self.data * scalar)
    
    def __rmul__(self, scalar):
        """Kernel scaling (reverse)."""
        return self.__mul__(scalar)
    
    def rotate(self, angle_degrees):
        """Rotate kernel by specified angle."""
        from scipy.ndimage import rotate
        rotated = rotate(self.data, angle_degrees, reshape=False, mode='constant')
        return Kernel(rotated)
    
    def normalize(self):
        """Return normalized kernel (sum = 1)."""
        total = np.sum(self.data)
        if abs(total) > 1e-10:
            return Kernel(self.data / total)
        return Kernel(self.data.copy())
    
    def zero_center(self):
        """Return zero-centered kernel (mean = 0)."""
        return Kernel(self.data - self.data.mean())
