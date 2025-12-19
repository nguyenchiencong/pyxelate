"""Backend abstraction for NumPy/CuPy array operations.

This module provides a unified interface for array operations that can
run on either CPU (NumPy) or GPU (CuPy). The backend is selected at
runtime based on availability and user preference.

Usage:
    from pyxelate.backend import get_backend

    # Auto-detect (use GPU if available)
    backend = get_backend("auto")

    # Force CPU
    backend = get_backend("cpu")

    # Force GPU (raises ImportError if CuPy not installed)
    backend = get_backend("cuda")
"""

from typing import Literal, Union
import numpy as np
from numpy.typing import NDArray

# Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cupy_convolve
    from cupyx.scipy.ndimage import median_filter as cupy_median_filter

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cupy_convolve = None
    cupy_median_filter = None
    CUPY_AVAILABLE = False

# Backend type hint
BackendType = Literal["cpu", "cuda", "auto"]


class ArrayBackend:
    """Abstraction layer for NumPy/CuPy array operations.

    This class provides a unified interface for array operations that can
    run on either CPU (NumPy) or GPU (CuPy).

    Attributes:
        use_gpu: Whether GPU acceleration is enabled.
        xp: The array module (numpy or cupy).
        name: Human-readable backend name.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """Initialize the backend.

        Args:
            use_gpu: If True, use CuPy for GPU acceleration.
                    Requires CuPy to be installed.

        Raises:
            ImportError: If use_gpu=True but CuPy is not installed.
        """
        if use_gpu and not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is not installed. Install with:\n"
                "  pip install cupy-cuda11x  # For CUDA 11.x\n"
                "  pip install cupy-cuda12x  # For CUDA 12.x\n"
                "Or see: https://docs.cupy.dev/en/stable/install.html"
            )

        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.name = "cuda" if self.use_gpu else "cpu"

    def to_device(self, arr: NDArray) -> NDArray:
        """Move array to the compute device (GPU if using CuPy).

        Args:
            arr: NumPy array to move.

        Returns:
            Array on the target device.
        """
        if self.use_gpu:
            return cp.asarray(arr)
        return arr

    def to_host(self, arr: NDArray) -> np.ndarray:
        """Move array back to CPU (host) memory.

        Args:
            arr: Array to move (can be NumPy or CuPy).

        Returns:
            NumPy array on CPU.
        """
        if self.use_gpu and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    def zeros(self, shape, dtype=np.float32) -> NDArray:
        """Create a zero-filled array on the target device."""
        return self.xp.zeros(shape, dtype=dtype)

    def empty(self, shape, dtype=np.float32) -> NDArray:
        """Create an uninitialized array on the target device."""
        return self.xp.empty(shape, dtype=dtype)

    def clip(self, arr: NDArray, a_min, a_max) -> NDArray:
        """Clip array values to a range."""
        return self.xp.clip(arr, a_min, a_max)

    def stack(self, arrays, axis: int = 0) -> NDArray:
        """Stack arrays along a new axis."""
        return self.xp.stack(arrays, axis=axis)

    def convolve(self, arr: NDArray, kernel: NDArray, mode: str = "reflect") -> NDArray:
        """Apply convolution to an array.

        Args:
            arr: Input array (2D).
            kernel: Convolution kernel.
            mode: Border handling mode.

        Returns:
            Convolved array.
        """
        if self.use_gpu:
            return cupy_convolve(arr, kernel, mode=mode)
        else:
            from scipy.ndimage import convolve

            return convolve(arr, kernel, mode=mode)

    def median_filter(self, arr: NDArray, size: int) -> NDArray:
        """Apply median filter to an array.

        Args:
            arr: Input array (2D).
            size: Filter size.

        Returns:
            Filtered array.
        """
        if self.use_gpu:
            return cupy_median_filter(arr, size=size)
        else:
            from scipy.ndimage import median_filter

            return median_filter(arr, size=size)

    def resize(self, arr: NDArray, output_shape: tuple, **kwargs) -> NDArray:
        """Resize an image array.

        Note: For GPU, we transfer back to CPU for resize as cupy doesn't
        have a direct equivalent to skimage.transform.resize.

        Args:
            arr: Input image array.
            output_shape: Desired output shape (H, W) or (H, W, C).
            **kwargs: Additional arguments for skimage.transform.resize.

        Returns:
            Resized array.
        """
        from skimage.transform import resize as skimage_resize

        # Resize is complex, use CPU implementation
        host_arr = self.to_host(arr)
        result = skimage_resize(host_arr, output_shape, **kwargs)
        return self.to_device(result) if self.use_gpu else result

    def sobel(self, arr: NDArray) -> NDArray:
        """Apply Sobel edge detection filter.

        Args:
            arr: Input array (2D).

        Returns:
            Edge magnitude array.
        """
        if self.use_gpu:
            # Use CuPy's implementation via scipy.ndimage
            from cupyx.scipy.ndimage import sobel as cupy_sobel

            sx = cupy_sobel(arr, axis=0)
            sy = cupy_sobel(arr, axis=1)
            return self.xp.hypot(sx, sy)
        else:
            from skimage.filters import sobel as skimage_sobel

            return skimage_sobel(arr)

    def svd(self, arr: NDArray, n_components: int, **kwargs) -> tuple:
        """Compute truncated SVD.

        Args:
            arr: Input 2D array.
            n_components: Number of components to keep.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (U, s, V) matrices.
        """
        if self.use_gpu:
            # CuPy has built-in SVD
            U, s, V = self.xp.linalg.svd(arr, full_matrices=False)
            return U[:, :n_components], s[:n_components], V[:n_components, :]
        else:
            from sklearn.utils.extmath import randomized_svd

            return randomized_svd(arr, n_components=n_components, **kwargs)


def get_backend(backend: BackendType = "auto") -> ArrayBackend:
    """Get an ArrayBackend instance based on the specified type.

    Args:
        backend: One of:
            - "cpu": Force CPU (NumPy) backend
            - "cuda": Force GPU (CuPy) backend
            - "auto": Use GPU if available, otherwise CPU

    Returns:
        Configured ArrayBackend instance.

    Raises:
        ImportError: If backend="cuda" but CuPy is not installed.
        ValueError: If backend is not a valid option.
    """
    if backend == "cpu":
        return ArrayBackend(use_gpu=False)
    elif backend == "cuda":
        return ArrayBackend(use_gpu=True)
    elif backend == "auto":
        return ArrayBackend(use_gpu=CUPY_AVAILABLE)
    else:
        raise ValueError(
            f"Invalid backend: {backend}. Must be one of: 'cpu', 'cuda', 'auto'"
        )


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        True if CuPy is installed and a CUDA device is available.
    """
    if not CUPY_AVAILABLE:
        return False
    try:
        # Try to access a CUDA device
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False
