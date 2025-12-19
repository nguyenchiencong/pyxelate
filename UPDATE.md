# Pyxelate Improvement Plan

This document outlines the performance and quality improvements planned for pyxelate.

## Overview

After thorough analysis of the codebase, we identified **22 specific improvement opportunities** across performance bottlenecks, code quality issues, and algorithm improvements.

---

## Phase 1: Quick Wins (Low effort, high impact)

**Status:** Completed  
**Version:** 2.1.3

### 1.1 Fix Deprecation Warning - `skimage.morphology.square`

**Priority:** HIGH  
**Files:** `pyx.py`, `vid.py`

**Issue:** `skimage.morphology.square` is deprecated since scikit-image 0.25 and will be removed in 0.27.

**Changes:**
- Replace `from skimage.morphology import square as skimage_square` with `from skimage.morphology import footprint_rectangle`
- Update all usages: `skimage_square(n)` → `footprint_rectangle(n, n)`

**Locations:**
- `pyx.py` line 15 (import)
- `pyx.py` line 364 (`_dilate` method)
- `pyx.py` line 379 (`_median` method)
- `vid.py` line 2 (import)
- `vid.py` line 45 (erosion call)

### 1.2 Optimize BGM LAB Conversion

**Priority:** HIGH  
**File:** `pyx.py` lines 95-101

**Issue:** `rgb2lab(X)` is computed repeatedly for each palette color in the list comprehension.

**Change:** Pre-compute LAB conversion once before the loop.

**Expected Impact:** 2-3x speedup for palette-based fitting

### 1.3 Replace Assertions with Proper Exceptions

**Priority:** MEDIUM  
**File:** `pyx.py`

**Issue:** Assertions can be disabled with `python -O`, making runtime checks ineffective.

**Change:** Replace all `assert` statements with proper `ValueError` or `RuntimeError` exceptions.

**Locations:**
- Line 79: BGM init_params check
- Lines 172-176: height, width, factor validation
- Lines 183-184: sobel validation
- Line 191: upscale validation
- Line 206: depth validation
- Lines 255-256: palette color validation
- Line 419: is_fitted check in transform()
- Lines 424-429: image size validation

---

## Phase 2: Performance Optimizations (Medium effort, high impact)

**Status:** Completed  
**Version:** 2.2.0

### 2.1 Optimize Atkinson Dithering (CRITICAL)

**Priority:** CRITICAL  
**File:** `pyx.py` lines 520-535  
**Expected Impact:** 10-100x speedup

**Issue:** Nested loop calls `model.predict_proba()` for every single pixel. For a 256x256 image, this results in ~65,536 individual model predictions.

**Solution:** 
1. Batch the initial prediction to get all probabilities at once
2. Use numba `@njit` for the error diffusion loop
3. Pre-compute `self.model.means_` lookup outside the loop

```python
# Before: O(n) model calls where n = pixels
for y in range(final_h):
    for x in range(final_w):
        pred = self.model.predict_proba(pixel)  # SLOW

# After: O(1) model call + O(n) numba loop  
probs = self.model.predict_proba(all_pixels)  # Single batch call
result = _atkinson_njit(image, probs, means)   # Fast numba loop
```

### 2.2 Vectorize Naive Dithering

**Priority:** MEDIUM  
**File:** `pyx.py` lines 489-498  
**Expected Impact:** 2-5x speedup

**Issue:** Python loop iterating over pixels.

**Solution:** Replace with vectorized numpy operations using checkerboard mask:

```python
rows = np.arange(final_h)
cols = np.arange(final_w)
checkerboard = (rows[:, None] + cols[None, :]) % 2
flat_checker = checkerboard.ravel()
mask_v1 = (flat_checker == 1) & v1
mask_v2 = (flat_checker == 0) & v2
X_[mask_v1 | mask_v2] = self.colors[p2[mask_v1 | mask_v2]]
```

### 2.3 Optimize Bayer Dithering

**Priority:** MEDIUM  
**File:** `pyx.py` lines 499-512  
**Expected Impact:** 1.5x speedup

**Issue:** List comprehension with repeated reshaping and convolution operations.

**Solution:** Use numpy stacking and batch convolution.

### 2.4 Parallelize Video Frame Loading

**Priority:** MEDIUM  
**File:** `vid.py` lines 24-48  
**Expected Impact:** 1.5-2x speedup

**Issue:** Video frames are processed sequentially.

**Solution:** Use `concurrent.futures.ThreadPoolExecutor` for frame loading and pre-processing while keeping sequential keyframe detection.

---

## Phase 3: Quality Improvements

**Status:** Completed  
**Version:** 2.3.0

### 3.1 LAB Color Space for BGM Fitting

**Priority:** MEDIUM  
**File:** `pyx.py`  
**Expected Impact:** More perceptually uniform color palettes

**Issue:** BGM fitting happens in RGB space, but perceptual color differences are better measured in LAB space.

**Solution:** Convert to LAB before fitting BGM, convert means back to RGB after.

### 3.2 Add Type Hints to vid.py

**Priority:** MEDIUM  
**File:** `vid.py`

**Issue:** All methods lack type hints.

**Solution:** Add complete type annotations.

### 3.3 Add 8x8 Bayer Matrix Option

**Priority:** LOW  
**File:** `pyx.py`

**Issue:** Only 4x4 Bayer matrix available.

**Solution:** Add `DITHER_BAYER_MATRIX_8x8` for higher quality at larger scales.

### 3.4 Adaptive SVD Components

**Priority:** LOW  
**File:** `pyx.py` lines 142-143

**Issue:** Fixed number of SVD components (32) regardless of image size.

**Solution:** Adapt SVD components based on image dimensions:
```python
def _get_adaptive_svd_components(self, h: int, w: int) -> int:
    min_dim = min(h, w)
    return min(self.SVD_N_COMPONENTS, max(8, min_dim // 4))
```

---

## Phase 4: Advanced Optimizations & GPU Acceleration

**Status:** Phase 4a Completed  
**Version:** 2.4.0

### 4.1 Parallelize Channel Processing

**Priority:** HIGH  
**Files:** `pyx.py`  
**Expected Impact:** 2-3x speedup on multi-core CPUs

**Issue:** Many operations use `@adapt_rgb(each_channel)` decorator which processes R, G, B channels sequentially. This includes:
- `_pyxelate()` - Sobel filter and block operations
- `_svd()` - Randomized SVD per channel
- `_median()` - Median filter per channel
- `_dilate()` - Dilation per channel

**Solution:** Use `concurrent.futures.ThreadPoolExecutor` to process all 3 channels in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def _process_channels_parallel(self, X: np.ndarray, func) -> np.ndarray:
    """Process RGB channels in parallel."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(func, X[:, :, i]) for i in range(3)]
        results = [f.result() for f in futures]
    return np.stack(results, axis=-1)
```

### 4.2 Batch SVD Processing

**Priority:** HIGH  
**File:** `pyx.py`  
**Expected Impact:** 1.5-2x speedup

**Issue:** `randomized_svd` is called 3 times (once per channel) with separate memory allocations.

**Solution:** Process all channels in a single batched operation or use parallel execution:

```python
def _svd_batch(self, X: np.ndarray) -> np.ndarray:
    """Batch SVD across all channels."""
    h, w, c = X.shape
    n_components = self._get_adaptive_svd_components(h, w)
    
    results = np.zeros_like(X)
    for i in range(c):
        U, s, V = randomized_svd(X[:, :, i], n_components=n_components, ...)
        results[:, :, i] = np.clip(U @ np.diag(s) @ V / 255.0, 0.0, 1.0)
    return results
```

### 4.3 Optimize Sobel-based Downsampling

**Priority:** MEDIUM  
**File:** `pyx.py` `_pyxelate()` method  
**Expected Impact:** 1.3-1.5x speedup

**Issue:** Current implementation:
1. Computes Sobel filter per channel via decorator
2. Uses `view_as_blocks` which creates memory copies
3. Performs division per block

**Solution:** 
1. Use `scipy.ndimage.sobel` with pre-allocated output buffer
2. Replace `view_as_blocks` with strided views (no copy)
3. Vectorize block operations across all channels simultaneously

```python
def _pyxelate_optimized(self, X: np.ndarray) -> np.ndarray:
    from scipy.ndimage import sobel
    
    h, w, c = X.shape
    # Compute Sobel for all channels at once
    sobel_mag = np.zeros_like(X)
    for i in range(c):
        sobel_mag[:, :, i] = np.abs(sobel(X[:, :, i], axis=0)) + \
                             np.abs(sobel(X[:, :, i], axis=1))
    sobel_mag += 1e-8
    
    # Use stride tricks for zero-copy block views
    from numpy.lib.stride_tricks import as_strided
    # ... strided implementation
```

### 4.4 Depth Loop Optimization

**Priority:** MEDIUM  
**File:** `pyx.py` `transform()` method  
**Expected Impact:** 1.2-1.5x speedup per depth iteration

**Issue:** The depth loop runs median + pyxelate sequentially for each iteration:
```python
for _ in range(self.depth):
    if d == 3:
        X_ = self._median(X_)  # Sequential per channel
    X_ = self._pyxelate(X_)    # Sequential per channel
```

**Solution:** 
1. Fuse median and pyxelate into single pass where possible
2. Use parallel channel processing (see 4.1)
3. Consider adaptive depth based on image size

### 4.5 GPU Acceleration with CuPy (Optional)

**Priority:** HIGH  
**Files:** `pyx.py`, new `backend.py`  
**Expected Impact:** 5-20x speedup on CUDA-capable GPUs

**Issue:** All operations are CPU-bound. Large images (1000x1000+) are slow.

**Solution:** Add optional CuPy backend with automatic fallback to NumPy:

```python
# New backend.py
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class ArrayBackend:
    """Abstraction layer for NumPy/CuPy arrays."""
    
    def __init__(self, use_gpu: bool = False):
        if use_gpu and not CUPY_AVAILABLE:
            raise ImportError("CuPy not installed. Install with: pip install cupy-cuda12x")
        self.xp = cp if (use_gpu and CUPY_AVAILABLE) else np
        self.use_gpu = use_gpu and CUPY_AVAILABLE
    
    def to_device(self, arr: np.ndarray):
        """Move array to GPU if using CuPy."""
        return self.xp.asarray(arr) if self.use_gpu else arr
    
    def to_host(self, arr) -> np.ndarray:
        """Move array back to CPU."""
        return cp.asnumpy(arr) if self.use_gpu else arr
```

**Usage:**
```python
pyx = Pyx(factor=14, palette=7, backend="cuda")  # or "cpu" (default)
```

**GPU-accelerated operations:**
- Image resize (cupy + cupyx.scipy.ndimage)
- SVD (cupy.linalg.svd or cupy randomized_svd)
- Sobel filter (cupyx.scipy.ndimage.sobel)
- Convolutions for Bayer dithering (cupyx.scipy.ndimage.convolve)
- Color space conversions (custom CUDA kernels or cucim)

### 4.6 Numba CUDA Kernels for Dithering

**Priority:** MEDIUM  
**File:** `pyx.py`  
**Expected Impact:** 10-50x speedup for dithering on GPU

**Issue:** Atkinson and Floyd-Steinberg dithering have data dependencies that limit parallelization, but can still benefit from GPU.

**Solution:** Use Numba CUDA for error diffusion with wavefront parallelism:

```python
from numba import cuda

@cuda.jit
def atkinson_cuda_kernel(image, means, result, h, w):
    """CUDA kernel for Atkinson dithering with diagonal wavefront."""
    # Process anti-diagonals in parallel
    # Each thread handles one pixel on the current wavefront
    ...
```

**Note:** Error diffusion algorithms have sequential dependencies, so we use diagonal wavefront parallelism where pixels on the same anti-diagonal can be processed in parallel.

### 4.7 Memory Optimization

**Priority:** LOW  
**File:** `pyx.py`  
**Expected Impact:** 30-50% memory reduction for large images

**Issue:** Multiple intermediate arrays are created during processing.

**Solution:**
1. Use in-place operations where possible
2. Explicitly delete intermediate arrays
3. Use `np.empty` instead of `np.zeros` when array will be fully overwritten
4. Consider memory-mapped arrays for very large images

```python
def transform(self, X: np.ndarray, ...) -> np.ndarray:
    # Use in-place operations
    X_ = X.astype(np.float32, copy=True)  # Single copy
    np.divide(X_, 255.0, out=X_)          # In-place
    ...
```

---

## GPU Backend Comparison

| Library | Pros | Cons | Recommended For |
|---------|------|------|-----------------|
| **CuPy** | Drop-in NumPy API, easy migration, mature | New dependency (~500MB) | General GPU acceleration |
| **Numba CUDA** | Already a dependency, fine-grained control | Complex API, manual memory management | Custom kernels (dithering) |
| **PyTorch** | Excellent ecosystem, auto-differentiation | Heavy (~2GB), overkill for this use case | Not recommended |
| **cuCIM** | GPU-accelerated skimage equivalent | NVIDIA-only, less mature | Image filtering operations |

**Recommendation:** 
- Phase 4a: CPU optimizations (4.1-4.4) - no new dependencies
- Phase 4b: CuPy backend (4.5) - optional dependency for GPU users
- Phase 4c: Numba CUDA kernels (4.6) - for dithering hotspots

---

## Summary Table

| Phase | Priority | Issue | File | Impact |
|-------|----------|-------|------|--------|
| 1 | HIGH | Deprecated `square` function | pyx.py, vid.py | Future compatibility |
| 1 | HIGH | Redundant LAB conversions | pyx.py | 2-3x speedup |
| 1 | MEDIUM | Assertions for runtime checks | pyx.py | Code robustness |
| 2 | CRITICAL | Atkinson dithering per-pixel calls | pyx.py | 10-100x speedup |
| 2 | MEDIUM | Naive dithering Python loop | pyx.py | 2-5x speedup |
| 2 | MEDIUM | Bayer dithering optimization | pyx.py | 1.5x speedup |
| 2 | MEDIUM | Video frame parallelization | vid.py | 1.5-2x speedup |
| 3 | MEDIUM | LAB color space for fitting | pyx.py | Quality improvement |
| 3 | MEDIUM | Type hints for vid.py | vid.py | Maintainability |
| 3 | LOW | 8x8 Bayer matrix option | pyx.py | Quality option |
| 3 | LOW | Adaptive SVD components | pyx.py | Quality improvement |
| 4 | HIGH | Parallelize channel processing | pyx.py | 2-3x speedup |
| 4 | HIGH | GPU acceleration (CuPy) | pyx.py, backend.py | 5-20x speedup |
| 4 | MEDIUM | Batch SVD processing | pyx.py | 1.5-2x speedup |
| 4 | MEDIUM | Optimize Sobel downsampling | pyx.py | 1.3-1.5x speedup |
| 4 | MEDIUM | Numba CUDA dithering | pyx.py | 10-50x speedup |
| 4 | LOW | Memory optimization | pyx.py | 30-50% less memory |

---

## Version History

- **2.1.2** - NumPy 2.x support, modernized project config
- **2.1.3** - Phase 1 improvements (deprecation fixes, LAB optimization, exception handling)
- **2.2.0** - Phase 2 improvements (dithering optimizations)
- **2.3.0** - Phase 3 improvements (type hints, bayer8, adaptive SVD)
- **2.4.0** - Phase 4a improvements (CPU parallelization) [Planned]
- **2.5.0** - Phase 4b improvements (GPU acceleration with CuPy) [Planned]
