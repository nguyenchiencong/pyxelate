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

---

## Version History

- **2.1.2** - NumPy 2.x support, modernized project config
- **2.1.3** - Phase 1 improvements (deprecation fixes, LAB optimization, exception handling)
- **2.2.0** - Phase 2 improvements (dithering optimizations, video parallelization) [Planned]
- **2.3.0** - Phase 3 improvements (quality enhancements) [Planned]
