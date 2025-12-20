import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils.extmath import randomized_svd

from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, deltaE_ciede2000
from skimage.exposure import equalize_adapthist
from skimage.filters import sobel as skimage_sobel
from skimage.filters import median as skimage_median
from skimage.morphology import footprint_rectangle
from skimage.morphology import dilation as skimage_dilation
from skimage.transform import resize
from skimage.util import view_as_blocks

from scipy.ndimage import convolve, label, binary_dilation

from numba import njit

try:
    from .pal import BasePalette
except ImportError:
    from pal import BasePalette

from typing import Callable, Literal, Optional, Union, Tuple

try:
    from .backend import ArrayBackend, get_backend, BackendType, CUPY_AVAILABLE
except ImportError:
    from backend import ArrayBackend, get_backend, BackendType, CUPY_AVAILABLE


class PyxWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class BGM(BayesianGaussianMixture):
    """Wrapper for BayesianGaussianMixture"""

    MAX_ITER = 128
    RANDOM_STATE = 1234567

    def __init__(self, palette: Union[int, BasePalette], find_palette: bool) -> None:
        """Init BGM with different default parameters depending on use-case"""
        self.palette = palette
        self.find_palette = find_palette
        if self.find_palette:
            super().__init__(
                n_components=self.palette,
                max_iter=self.MAX_ITER,
                covariance_type="tied",
                weight_concentration_prior_type="dirichlet_distribution",
                weight_concentration_prior=1.0 / self.palette,
                mean_precision_prior=1.0 / 256.0,
                warm_start=False,
                random_state=self.RANDOM_STATE,
            )
        else:
            super().__init__(
                n_components=len(self.palette),
                max_iter=self.MAX_ITER,
                covariance_type="tied",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=1e-7,
                mean_precision_prior=1.0 / len(self.palette),
                warm_start=False,
                random_state=self.RANDOM_STATE,
            )
            # start centroid search from the palette's values
            self.mean_prior = np.mean([val[0] for val in self.palette], axis=0)

    def _initialize_parameters(
        self, X: np.ndarray, random_state: int, **kwargs
    ) -> None:
        """Changes init parameters from K-means to CIE LAB distance when palette is assigned"""
        if self.init_params != "kmeans":
            raise ValueError(
                "Initialization is overwritten, can only be set as 'kmeans'."
            )
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))
        if self.find_palette:
            # original centroids
            label = (
                KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
        else:
            # color distance based centroids
            # Pre-compute LAB conversion once for efficiency
            X_lab = rgb2lab(X.reshape(-1, 1, 3)).reshape(-1, 3)
            label = np.argmin(
                [
                    deltaE_ciede2000(
                        X_lab,
                        rgb2lab(np.array(p).reshape(1, 1, 3)).reshape(1, 3),
                        kH=3,
                        kL=2,
                    )
                    for p in self.palette
                ],
                axis=0,
            )
        resp[np.arange(n_samples), label] = 1
        self._initialize(X, resp)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BGM":
        """Fits BGM model but alters convergence warning"""
        converged = True
        with warnings.catch_warnings(record=True) as w:
            super().fit(X)
            if w and w[-1].category == ConvergenceWarning:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                converged = False
        if not converged:
            warnings.warn(
                "Pyxelate could not properly assign colors, try a different palette size for better results!",
                PyxWarning,
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = super().predict_proba(X)
        # adjust for monochrome
        if self.find_palette:
            if self.palette < 3:
                return np.sqrt(p)
        elif len(self.palette) < 3:
            return np.sqrt(p)
        return p


class Pyx(BaseEstimator, TransformerMixin):
    """Pyx extends scikit-learn transformers"""

    BGM_RESIZE = 256
    SCALE_RGB = 1.07
    HIST_BRIGHTNESS = 1.19
    COLOR_QUANT = 8
    DITHER_AUTO_SIZE_LIMIT_HI = 512
    DITHER_AUTO_SIZE_LIMIT_LO = 16
    DITHER_AUTO_COLOR_LIMIT = 8
    DITHER_NAIVE_BOOST = 1.33
    SVD_N_COMPONENTS = 32
    SVD_MAX_ITER = 16
    SVD_RANDOM_STATE = 1234
    # precalculated 4x4 Bayer Matrix / 16 - 0.5
    DITHER_BAYER_MATRIX = np.array(
        [
            [-0.5, 0.0, -0.375, 0.125],
            [0.25, -0.25, 0.375, -0.125],
            [-0.3125, 0.1875, -0.4375, 0.0625],
            [0.4375, -0.0625, 0.3125, -0.1875],
        ]
    )
    # precalculated 8x8 Bayer Matrix / 64 - 0.5 (higher quality for larger images)
    DITHER_BAYER_MATRIX_8x8 = np.array(
        [
            [-0.5, 0.0, -0.375, 0.125, -0.46875, 0.03125, -0.34375, 0.15625],
            [0.25, -0.25, 0.375, -0.125, 0.28125, -0.21875, 0.40625, -0.09375],
            [-0.3125, 0.1875, -0.4375, 0.0625, -0.28125, 0.21875, -0.40625, 0.09375],
            [0.4375, -0.0625, 0.3125, -0.1875, 0.46875, -0.03125, 0.34375, -0.15625],
            [
                -0.453125,
                0.046875,
                -0.328125,
                0.171875,
                -0.484375,
                0.015625,
                -0.359375,
                0.140625,
            ],
            [
                0.296875,
                -0.203125,
                0.421875,
                -0.078125,
                0.265625,
                -0.234375,
                0.390625,
                -0.109375,
            ],
            [
                -0.265625,
                0.234375,
                -0.390625,
                0.109375,
                -0.296875,
                0.203125,
                -0.421875,
                0.078125,
            ],
            [
                0.484375,
                -0.015625,
                0.359375,
                -0.140625,
                0.453125,
                -0.046875,
                0.328125,
                -0.171875,
            ],
        ]
    )

    def __init__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        factor: Optional[int] = None,
        upscale: Union[Tuple[int, int], int] = 1,
        depth: int = 1,
        palette: Union[int, BasePalette] = 8,
        dither: Optional[str] = "none",
        sobel: int = 3,
        svd: bool = True,
        alpha: float = 0.6,
        backend: BackendType = "cpu",
        postprocess: bool = True,
    ) -> None:
        if (width is not None or height is not None) and factor is not None:
            raise ValueError(
                "You can only set either height + width or the downscaling factor, but not both!"
            )
        if height is not None and height < 1:
            raise ValueError("Height must be a positive integer!")
        if width is not None and width < 1:
            raise ValueError("Width must be a positive integer!")
        if factor is not None and factor < 1:
            raise ValueError("Factor must be a positive integer!")
        if not isinstance(sobel, int) or sobel < 2:
            raise ValueError("Sobel must be an integer strictly greater than 1!")
        self.height = int(height) if height else None
        self.width = int(width) if width else None
        self.factor = int(factor) if factor else None
        self.sobel = sobel
        if isinstance(upscale, (list, tuple, set, np.ndarray)):
            if len(upscale) != 2:
                raise ValueError("Upscale must be len 2, with 2 positive integers!")
            if upscale[0] < 1 or upscale[1] < 1:
                raise ValueError("Upscale must have 2 positive values!")
            self.upscale = (upscale[0], upscale[1])
        else:
            if upscale < 1:
                raise ValueError("Upscale must be a positive integer!")
            self.upscale = (upscale, upscale)
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("Depth must be a positive integer!")
        if depth > 2:
            warnings.warn(
                "Depth too high, it will probably take really long to finish!",
                PyxWarning,
            )
        self.depth = depth
        self.palette = palette
        self.find_palette = isinstance(
            self.palette, (int, float)
        )  # palette is a number
        if self.find_palette and palette < 2:
            raise ValueError("The minimum number of colors in a palette is 2")
        elif not self.find_palette and len(palette) < 2:
            raise ValueError("The minimum number of colors in a palette is 2")
        if dither not in (
            None,
            "none",
            "naive",
            "bayer",
            "bayer8",
            "floyd",
            "atkinson",
        ):
            raise ValueError("Unknown dithering algorithm!")
        self.dither = dither
        self.svd = bool(svd)
        self.alpha = float(alpha)
        self.postprocess = bool(postprocess)
        # instantiate BGM model
        self.model = BGM(self.palette, self.find_palette)
        self.is_fitted = False
        self.palette_cache = None
        # setup compute backend (CPU or GPU)
        self._backend = get_backend(backend)

    def _get_size(self, original_height: int, original_width: int) -> Tuple[int, int]:
        """Calculate new size depending on settings"""
        if self.height is not None and self.width is not None:
            return self.height, self.width
        elif self.height is not None:
            return self.height, int(self.height / original_height * original_width)
        elif self.width is not None:
            return int(self.width / original_width * original_height), self.width
        elif self.factor is not None:
            return original_height // self.factor, original_width // self.factor
        else:
            return original_height, original_width

    def _perceptual_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """Compute perceptual luminance using ITU-R BT.601 weights.

        Args:
            rgb: Array of RGB values (uint8, 0-255)

        Returns:
            Luminance values (0-255 scale)
        """
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    def _fix_dark_speckles(
        self,
        X: np.ndarray,
        max_cluster_size: int = 10,
        dark_lum_threshold: int = 30,
        bright_lum_threshold: int = 60,
    ) -> np.ndarray:
        """Remove isolated dark pixel clusters that appear as artifacts in bright regions.

        This post-processing step identifies small clusters of dark palette colors
        that are surrounded by bright pixels and replaces them with the most common
        neighboring color. This fixes speckle artifacts caused by the HSV/SCALE_RGB
        processing pushing certain saturated colors to extreme values.

        Args:
            X: The pixelated image (H, W, 3 or 4) as uint8
            max_cluster_size: Maximum size of clusters to consider as artifacts (default 10)
            dark_lum_threshold: Maximum perceptual luminance for a color to be considered "dark" (default 30)
            bright_lum_threshold: Minimum perceptual luminance for a neighbor to be considered "bright" (default 100)

        Returns:
            The image with dark speckle artifacts removed
        """
        # Handle images with alpha channel
        has_alpha = X.shape[2] == 4
        if has_alpha:
            rgb = X[:, :, :3]
            alpha = X[:, :, 3]
        else:
            rgb = X

        # Find all unique colors in the palette
        colors = np.unique(rgb.reshape(-1, 3), axis=0)
        if len(colors) <= 1:
            return X  # Nothing to fix

        # Use perceptual luminance to find dark colors
        color_luminances = self._perceptual_luminance(colors)
        dark_colors = colors[color_luminances < dark_lum_threshold]

        if len(dark_colors) == 0:
            return X  # No dark colors to fix

        # Create mask for ALL dark color pixels
        dark_mask = np.zeros(rgb.shape[:2], dtype=bool)
        for dark_color in dark_colors:
            dark_mask |= np.all(rgb == dark_color, axis=2)

        # Label connected components of dark pixels
        labeled, num_features = label(dark_mask)
        if num_features == 0:
            return X  # No dark pixels to fix

        # Process each cluster
        fixed = rgb.copy()
        for i in range(1, num_features + 1):
            cluster_mask = labeled == i
            size = np.sum(cluster_mask)

            # Only process small clusters (likely artifacts)
            if size > max_cluster_size:
                continue

            # Get the boundary of this cluster (2 pixels out for better context)
            dilated = binary_dilation(binary_dilation(cluster_mask))
            boundary = dilated & ~cluster_mask

            # Get colors of neighboring pixels
            neighbor_colors = rgb[boundary]
            if len(neighbor_colors) == 0:
                continue

            # Check if neighbors are predominantly bright (using perceptual luminance)
            neighbor_lums = self._perceptual_luminance(neighbor_colors)
            bright_ratio = np.mean(neighbor_lums > bright_lum_threshold)

            # Only fix if most neighbors are bright (cluster is isolated in bright region)
            if bright_ratio > 0.7:
                # Replace with most common bright neighbor color
                bright_neighbors = neighbor_colors[neighbor_lums > bright_lum_threshold]
                if len(bright_neighbors) > 0:
                    unique_neighbors, counts = np.unique(
                        bright_neighbors, axis=0, return_counts=True
                    )
                    most_common = unique_neighbors[np.argmax(counts)]
                    fixed[cluster_mask] = most_common

        # Reconstruct with alpha if needed
        if has_alpha:
            return np.dstack((fixed, alpha))
        return fixed

    def _image_to_float(self, image: np.ndarray) -> np.ndarray:
        """Helper function that changes 0 - 255 color representation to 0. - 1.

        Uses float32 for better memory efficiency.
        """
        if np.issubdtype(image.dtype, np.integer):
            return np.clip(image.astype(np.float32) / 255.0, 0, 1)
        return image.astype(np.float32) if image.dtype == np.float64 else image

    def _image_to_int(self, image: np.ndarray) -> np.ndarray:
        """Helper function that changes 0. - 1. color representation to 0 - 255"""
        if isinstance(image, BasePalette):
            image = np.array(image.value, dtype=float)
        elif isinstance(image, (list, tuple)):
            is_int = np.all([isinstance(x, int) for x in image])
            if is_int:
                return np.clip(np.array(image, dtype=int), 0, 255)
            else:
                image = np.array(image, dtype=float)
        if image.dtype in (float, np.float32, np.float64):  # np.float is deprecated
            return np.clip(np.array(image, dtype=float) * 255.0, 0, 255).astype(int)
        return image

    def _process_channels_parallel(
        self, X: np.ndarray, func: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Process RGB channels in parallel using ThreadPoolExecutor.

        Args:
            X: Input image array with shape (H, W, C)
            func: Function to apply to each channel

        Returns:
            Processed image with same shape as input
        """
        n_channels = X.shape[2] if X.ndim == 3 else 1
        if n_channels == 1:
            return func(X)

        with ThreadPoolExecutor(max_workers=min(n_channels, 3)) as executor:
            futures = [executor.submit(func, X[:, :, i]) for i in range(n_channels)]
            results = [f.result() for f in futures]
        return np.stack(results, axis=-1)

    @property
    def colors(self) -> np.ndarray:
        """Get colors in palette (0 - 255 range)"""
        if self.palette_cache is None:
            if self.find_palette:
                if not self.is_fitted:
                    raise RuntimeError("Call 'fit(image_as_numpy)' first!")
                c = rgb2hsv(self.model.means_.reshape(-1, 1, 3))
                c[:, :, 1:] *= self.SCALE_RGB
                c = hsv2rgb(c)
                c = np.clip(
                    c * 255 // self.COLOR_QUANT * self.COLOR_QUANT, 0, 255
                ).astype(int)
                c[c < self.COLOR_QUANT * 2] = 0
                c[c > 255 - self.COLOR_QUANT * 2] = 255
                self.palette_cache = c
                if len(np.unique([f"{pc[0]}" for pc in self.palette_cache])) != len(c):
                    warnings.warn(
                        "Some colors are redundant, try a different palette size for better results!",
                        PyxWarning,
                    )
            else:
                self.palette_cache = self._image_to_int(self.palette)
        return self.palette_cache

    @property
    def _palette(self) -> np.ndarray:
        """Get colors in palette as a plottable palette format (0. - 1. in correct shape)"""
        return self._image_to_float(self.colors.reshape(-1, 3))

    @property
    def backend_name(self) -> str:
        """Get the name of the current compute backend ('cpu' or 'cuda')."""
        return self._backend.name

    @property
    def uses_gpu(self) -> bool:
        """Check if GPU acceleration is currently enabled."""
        return self._backend.use_gpu

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Pyx":
        """Fit palette and optionally calculate automatic dithering"""
        h, w, d = X.shape
        # create a smaller image for BGM without alpha channel
        if d > 3:
            # separate color and alpha channels
            X_ = self._dilate(X).reshape(-1, 4)
            alpha_mask = X_[:, 3]
            X_ = X_[alpha_mask >= self.alpha]
            X_ = X_.reshape(1, -1, 4)
            X_ = resize(
                X[:, :, :3],
                (1, min(h, self.BGM_RESIZE) * min(w, self.BGM_RESIZE)),
                anti_aliasing=False,
            )
        else:
            X_ = resize(
                X[:, :, :3],
                (min(h, self.BGM_RESIZE), min(w, self.BGM_RESIZE)),
                anti_aliasing=False,
            )
        X_ = self._image_to_float(X_).reshape(
            -1, 3
        )  # make sure colors have a float representation
        if self.find_palette:
            X_ = (
                (X_ - 0.5) * self.SCALE_RGB
            ) + 0.5  # move values away from grayish colors

        # fit BGM to generate palette
        self.model.fit(X_)
        self.is_fitted = True  # all done, user may call transform()
        return self

    def _pyxelate(self, X: np.ndarray) -> np.ndarray:
        """Downsample image based on the magnitude of its gradients in sobel-sided tiles.

        Uses parallel processing across RGB channels for better performance.
        Supports GPU acceleration via CuPy when backend="cuda".
        """
        sobel_size = self.sobel

        if self._backend.use_gpu:
            # GPU path: process all channels on GPU
            X_pad = self._pad(X, self.sobel)
            X_gpu = self._backend.to_device(X_pad.astype(np.float32))
            results = []
            for i in range(X_gpu.shape[2]):
                channel = X_gpu[:, :, i]
                sobel = self._backend.sobel(channel)
                sobel = sobel + 1e-8  # avoid division by zero
                # view_as_blocks equivalent on GPU
                h, w = channel.shape
                new_h, new_w = h // sobel_size, w // sobel_size
                channel_blocks = channel[
                    : new_h * sobel_size, : new_w * sobel_size
                ].reshape(new_h, sobel_size, new_w, sobel_size)
                sobel_blocks = sobel[
                    : new_h * sobel_size, : new_w * sobel_size
                ].reshape(new_h, sobel_size, new_w, sobel_size)
                sobel_norm = sobel_blocks.sum(axis=(1, 3))
                sum_prod = (sobel_blocks * channel_blocks).sum(axis=(1, 3))
                results.append(sum_prod / sobel_norm)
            result = self._backend.stack(results, axis=-1)
            return self._backend.to_host(result).copy()
        else:
            # CPU path: use parallel channel processing
            def _process_channel(channel: np.ndarray) -> np.ndarray:
                """Process a single channel with Sobel-weighted downsampling."""
                sobel = skimage_sobel(channel)
                sobel += 1e-8  # avoid division by zero
                sobel_norm = view_as_blocks(sobel, (sobel_size, sobel_size)).sum((2, 3))
                sum_prod = view_as_blocks(
                    (sobel * channel), (sobel_size, sobel_size)
                ).sum((2, 3))
                return sum_prod / sobel_norm

            X_pad = self._pad(X, self.sobel)
            return self._process_channels_parallel(X_pad, _process_channel).copy()

    def _pad(
        self,
        X: np.ndarray,
        pad_size: int,
        nh: Optional[int] = None,
        nw: Optional[int] = None,
    ) -> np.ndarray:
        """Pad image if it's not pad_size divisable or remove such padding"""
        if nh is None and nw is None:
            # pad edges so image is divisible by pad_size
            h, w, d = X.shape
            h1, h2 = (1 if h % pad_size > 0 else 0), (1 if h % pad_size == 1 else 0)
            w1, w2 = (1 if w % pad_size > 0 else 0), (1 if w % pad_size == 1 else 0)
            return np.pad(X, ((h1, h2), (w1, w2), (0, 0)), "edge")
        else:
            # remove previous padding
            return X[
                slice(
                    (1 if nh % pad_size > 0 else 0),
                    (-1 if nh % pad_size == 1 else None),
                ),
                slice(
                    (1 if nw % pad_size > 0 else 0),
                    (-1 if nw % pad_size == 1 else None),
                ),
                :,
            ]

    def _dilate(self, X: np.ndarray) -> np.ndarray:
        """Dilate semi-transparent edges to remove artifacts (for images with opacity).

        Uses parallel processing across RGB channels.
        """

        def _dilate_channel(channel: np.ndarray) -> np.ndarray:
            return skimage_dilation(channel, footprint=footprint_rectangle((3, 3)))

        h, w, d = X.shape
        X_ = self._pad(X, 3)
        mask = X_[:, :, 3]
        alter = self._process_channels_parallel(X_[:, :, :3], _dilate_channel)
        X_[:, :, :3][mask < self.alpha] = alter[mask < self.alpha]
        return self._pad(X_, 3, h, w)

    def _median(self, X: np.ndarray) -> np.ndarray:
        """Custom median filter on HSV channels using 3x3 squares.

        Uses parallel processing across HSV channels.
        Supports GPU acceleration via CuPy when backend="cuda".
        """
        h, w, d = X.shape
        X_ = self._pad(X, 3)  # add padding for median filter
        X_ = rgb2hsv(X_)  # change to HSV

        if self._backend.use_gpu:
            # GPU path: use CuPy's median filter
            X_gpu = self._backend.to_device(X_.astype(np.float32))
            results = []
            for i in range(X_gpu.shape[2]):
                filtered = self._backend.median_filter(X_gpu[:, :, i], size=3)
                results.append(filtered)
            X_ = self._backend.stack(results, axis=-1)
            X_ = self._backend.to_host(X_)
        else:
            # CPU path
            def _median_channel(channel: np.ndarray) -> np.ndarray:
                return skimage_median(channel, footprint_rectangle((3, 3)))

            X_ = self._process_channels_parallel(X_, _median_channel)

        X_ = hsv2rgb(X_)  # go back to RGB
        return self._pad(X_, 3, h, w)  # remove added padding

    def _warn_on_dither_with_alpha(self, d: int) -> None:
        if d > 3 and self.dither in ("bayer", "bayer8", "floyd", "atkinson"):
            warnings.warn(
                "Images with transparency can have unwanted artifacts around the edges with this dithering method. Use 'naive' instead.",
                PyxWarning,
            )

    def _get_adaptive_svd_components(self, h: int, w: int) -> int:
        """Calculate adaptive SVD components based on image dimensions.

        Smaller images need fewer components to avoid over-smoothing,
        while larger images can benefit from more components.

        Args:
            h: Image height
            w: Image width

        Returns:
            Number of SVD components to use (between 8 and SVD_N_COMPONENTS)
        """
        min_dim = min(h, w)
        return min(self.SVD_N_COMPONENTS, max(8, min_dim // 4))

    def _svd(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct image via truncated SVD on each RGB channel.

        Uses parallel processing across RGB channels for better performance.
        Supports GPU acceleration via CuPy when backend="cuda".
        """
        h, w = X.shape[:2]
        n_components = self._get_adaptive_svd_components(h, w)

        if n_components >= h - 1 and n_components >= w - 1:
            return X  # skip SVD

        if self._backend.use_gpu:
            # GPU path: use CuPy's SVD
            X_gpu = self._backend.to_device(X.astype(np.float32))
            results = []
            for i in range(X_gpu.shape[2]):
                channel = X_gpu[:, :, i]
                U, s, V = self._backend.svd(channel, n_components)
                # Reconstruct: U @ diag(s) @ V
                A = U @ (self._backend.xp.diag(s) @ V)
                A = self._backend.clip(A / 255.0, 0.0, 1.0)
                results.append(A)
            result = self._backend.stack(results, axis=-1)
            return self._backend.to_host(result)
        else:
            # CPU path: use parallel channel processing with randomized SVD
            def _svd_channel(channel: np.ndarray) -> np.ndarray:
                U, s, V = randomized_svd(
                    channel,
                    n_components=n_components,
                    n_iter=self.SVD_MAX_ITER,
                    random_state=self.SVD_RANDOM_STATE,
                )
                A = U @ np.diag(s) @ V
                return np.clip(A / 255.0, 0.0, 1.0)

            return self._process_channels_parallel(X, _svd_channel)

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform image to pyxelated version"""
        if not self.is_fitted:
            raise RuntimeError(
                "Call 'fit(image_as_numpy)' first before calling 'transform(image_as_numpy)'!"
            )
        h, w, d = X.shape
        if self.find_palette:
            if h * w <= self.palette:
                raise ValueError(
                    "Too many colors for such a small image! Use a larger image or a smaller palette."
                )
        else:
            if h * w <= len(self.palette):
                raise ValueError(
                    "Too many colors for such a small image! Use a larger image or a smaller palette."
                )

        new_h, new_w = self._get_size(h, w)  # get desired size depending on settings
        if d > 3:
            # image has alpha channel
            X_ = self._dilate(X)
            alpha_mask = resize(X_[:, :, 3], (new_h, new_w), anti_aliasing=True)
        else:
            # image has no alpha channel
            X_ = X
            alpha_mask = None
        if self.depth:
            # change size depending on the number of iterations
            new_h, new_w = (
                new_h * (self.sobel**self.depth),
                new_w * (self.sobel**self.depth),
            )
        X_ = resize(
            X_[:, :, :3], (new_h, new_w), anti_aliasing=True
        )  # colors are now 0. - 1.

        # optionally apply svd for a somewhat blockier low-pass look
        if self.svd:
            X_ = self._svd(X_)

        # adjust contrast
        X_ = rgb2hsv(equalize_adapthist(X_))  # to hsv after local contrast fix
        X_[:, :, 1:] *= self.HIST_BRIGHTNESS  # adjust v only
        X_ = hsv2rgb(np.clip(X_, 0.0, 1.0))  # back to rgb

        # pyxelate iteratively
        for _ in range(self.depth):
            if d == 3:
                # remove noise
                X_ = self._median(X_)
            X_ = self._pyxelate(X_)  # downsample in each iteration

        final_h, final_w, _ = X_.shape
        if self.find_palette:
            X_ = (
                (X_ - 0.5) * self.SCALE_RGB
            ) + 0.5  # values were already altered before in .fit()
        reshaped = np.reshape(X_, (final_h * final_w, 3))

        # add dithering, took a lot of ideas from https://surma.dev/things/ditherpunk/
        if self.dither is None or self.dither == "none":
            probs = self.model.predict(reshaped)
            X_ = self.colors[probs]
        elif self.dither == "naive":
            # pyxelate dithering based on BGM probability density only (vectorized)
            probs = self.model.predict_proba(reshaped)
            p = np.argmax(probs, axis=1)
            X_ = self.colors[p]
            probs[np.arange(len(p)), p] = 0
            p2 = np.argmax(probs, axis=1)  # second best
            max_probs = np.max(probs, axis=1)
            v1 = max_probs > (1.0 / (len(self.colors) + 1))
            v2 = max_probs > (1.0 / (len(self.colors) * self.DITHER_NAIVE_BOOST + 1))

            # Create checkerboard pattern using vectorized operations
            rows = np.arange(final_h)
            cols = np.arange(final_w)
            # row_parity[y, x] = y % 2 (0 for even rows, 1 for odd rows)
            row_parity = (rows[:, None] + np.zeros(final_w, dtype=int)).ravel() % 2
            # col_parity[i] = (i % final_w) % 2, but we want every other pixel
            pixel_idx = np.arange(final_h * final_w)
            col_idx = pixel_idx % final_w
            is_even_col = (col_idx % 2) == 0

            # For odd rows (row_parity == 1): apply v1 on even columns
            # For even rows (row_parity == 0): apply v2 on even columns
            odd_row_mask = (row_parity == 1) & is_even_col & v1
            even_row_mask = (row_parity == 0) & is_even_col & v2

            X_[odd_row_mask] = self.colors[p2[odd_row_mask]]
            X_[even_row_mask] = self.colors[p2[even_row_mask]]
        elif self.dither in ("bayer", "bayer8"):
            # Bayer-like dithering (optimized with batch convolution)
            # bayer = 4x4 matrix, bayer8 = 8x8 matrix for higher quality
            self._warn_on_dither_with_alpha(d)
            probs = self.model.predict_proba(reshaped)
            n_colors = len(self.colors)
            # Reshape all probabilities at once and stack for batch processing
            probs_reshaped = probs.T.reshape(n_colors, final_h, final_w)
            # Select appropriate Bayer matrix
            bayer_matrix = (
                self.DITHER_BAYER_MATRIX_8x8
                if self.dither == "bayer8"
                else self.DITHER_BAYER_MATRIX
            )
            # Apply convolution to all color channels at once
            if self._backend.use_gpu:
                # GPU path: use CuPy's convolve
                probs_gpu = self._backend.to_device(probs_reshaped.astype(np.float32))
                kernel_gpu = self._backend.to_device(bayer_matrix.astype(np.float32))
                probs_convolved = self._backend.xp.array(
                    [
                        self._backend.convolve(probs_gpu[i], kernel_gpu, mode="reflect")
                        for i in range(n_colors)
                    ]
                )
                probs = self._backend.xp.argmin(probs_convolved, axis=0)
                probs = self._backend.to_host(probs)
            else:
                # CPU path
                probs_convolved = np.array(
                    [
                        convolve(probs_reshaped[i], bayer_matrix, mode="reflect")
                        for i in range(n_colors)
                    ]
                )
                probs = np.argmin(probs_convolved, axis=0)
            X_ = self.colors[probs]
        elif self.dither == "floyd":
            # Floyd-Steinberg-like algorithm
            self._warn_on_dither_with_alpha(d)
            X_ = self._dither_floyd(reshaped, (final_h, final_w))
        elif self.dither == "atkinson":
            # Atkinson-like algorithm (optimized with numba)
            self._warn_on_dither_with_alpha(d)
            X_ = self._dither_atkinson(reshaped, (final_h, final_w))

        X_ = np.reshape(X_, (final_h, final_w, 3))  # reshape to actual image dimensions
        if alpha_mask is not None:
            # attach lost alpha layer
            alpha_mask[alpha_mask >= self.alpha] = 255
            alpha_mask[alpha_mask < self.alpha] = 0
            X_ = np.dstack((X_[:, :, :3], alpha_mask.astype(int)))

        # return upscaled image
        X_ = np.repeat(np.repeat(X_, self.upscale[0], axis=0), self.upscale[1], axis=1)
        X_ = X_.astype(np.uint8)

        # optionally fix dark speckle artifacts
        if self.postprocess:
            X_ = self._fix_dark_speckles(X_)

        return X_

    def _dither_floyd(
        self, reshaped: np.ndarray, final_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Floyd-Steinberg-like dithering (multiple steps are applied for speed up)"""

        @njit()
        def _wrapper(probs, final_h, final_w):
            # probs = 1. / np.where(probs == 1, 1., -np.log(probs))
            probs = np.power(probs, (1.0 / 6.0))
            res = np.zeros((final_h, final_w), dtype=np.int8)
            for y in range(final_h - 1):
                for x in range(1, final_w - 1):
                    quant_error = probs[:, y, x] / 16.0
                    res[y, x] = np.argmax(quant_error)
                    quant_error[res[y, x]] = 0.0
                    probs[:, y, x + 1] += quant_error * 7.0
                    probs[:, y + 1, x - 1] += quant_error * 3.0
                    probs[:, y + 1, x] += quant_error * 5.0
                    probs[:, y + 1, x + 1] += quant_error
            # fix edges
            x = final_w - 1
            for y in range(final_h):
                res[y, x] = np.argmax(probs[:, y, x])
                res[y, 0] = np.argmax(probs[:, y, 0])
            y = final_h - 1
            for x in range(1, final_w - 1):
                res[y, x] = np.argmax(probs[:, y, x])
            return res

        final_h, final_w = final_shape
        probs = self.model.predict_proba(reshaped)
        probs = np.array(
            [probs[:, i].reshape((final_h, final_w)) for i in range(len(self.colors))]
        )
        res = _wrapper(probs, final_h, final_w)
        return self.colors[res.reshape(final_h * final_w)]

    def _dither_atkinson(
        self, reshaped: np.ndarray, final_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Atkinson-like dithering optimized with numba"""

        @njit()
        def _atkinson_njit(
            image: np.ndarray,
            means: np.ndarray,
            final_h: int,
            final_w: int,
        ) -> np.ndarray:
            """
            Atkinson dithering with error diffusion.

            Unlike Floyd-Steinberg which diffuses 100% of error, Atkinson only
            diffuses 6/8 (75%) of the quantization error, giving a lighter,
            more open appearance typical of early Macintosh graphics.

            Error diffusion pattern:
                    [*] [1] [1]
                [1] [1] [1]
                    [1]
            Each [1] receives 1/8 of the error (total 6/8 = 75%)
            """
            res = np.zeros((final_h, final_w), dtype=np.int32)
            # Pad image for boundary handling: 2 rows below, 1 col left, 2 cols right
            padded = np.zeros((final_h + 2, final_w + 3, 3), dtype=np.float64)
            for y in range(final_h):
                for x in range(final_w):
                    for c in range(3):
                        padded[y, x + 1, c] = image[y * final_w + x, c]

            n_colors = means.shape[0]

            for y in range(final_h):
                for x in range(1, final_w + 1):
                    # Find closest color using squared Euclidean distance
                    pixel = padded[y, x]
                    min_dist = 1e30
                    best_idx = 0
                    for i in range(n_colors):
                        dist = 0.0
                        for c in range(3):
                            diff = pixel[c] - means[i, c]
                            dist += diff * diff
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = i

                    res[y, x - 1] = best_idx

                    # Compute quantization error (only 1/8 per neighbor, 6 neighbors = 75% total)
                    for c in range(3):
                        quant_error = (pixel[c] - means[best_idx, c]) / 8.0
                        # Diffuse to 6 neighbors
                        padded[y, x + 1, c] += quant_error  # right
                        padded[y, x + 2, c] += quant_error  # right+1
                        padded[y + 1, x - 1, c] += quant_error  # below-left
                        padded[y + 1, x, c] += quant_error  # below
                        padded[y + 1, x + 1, c] += quant_error  # below-right
                        padded[y + 2, x, c] += quant_error  # 2 below

            return res

        final_h, final_w = final_shape
        # Use model means directly (they're in 0-1 float space)
        means = self.model.means_.astype(np.float64)
        image = reshaped.astype(np.float64)

        res = _atkinson_njit(image, means, final_h, final_w)
        return self.colors[res.reshape(final_h * final_w)]
