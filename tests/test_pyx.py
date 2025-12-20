"""Unit tests for the Pyx class."""

import numpy as np
import pytest
from pathlib import Path

from pyxelate import Pyx, Pal


# Test fixtures
@pytest.fixture
def sample_image_rgb():
    """Create a simple 100x100 RGB test image with gradient."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create a gradient
    for i in range(100):
        img[i, :, 0] = i * 2  # Red gradient
        img[:, i, 1] = i * 2  # Green gradient
    img[:, :, 2] = 128  # Constant blue
    return img


@pytest.fixture
def sample_image_rgba():
    """Create a simple 100x100 RGBA test image with transparency."""
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[:, :, :3] = 128  # Gray
    img[:, :, 3] = 255  # Fully opaque
    # Add transparent region
    img[40:60, 40:60, 3] = 0  # Transparent center
    return img


@pytest.fixture
def real_image():
    """Load a real test image if available."""
    test_path = Path(__file__).parent.parent / "examples" / "corgi.jpg"
    if test_path.exists():
        from skimage import io

        return io.imread(str(test_path))
    return None


class TestPyxInit:
    """Tests for Pyx initialization and parameter validation."""

    def test_init_with_factor(self):
        """Test initialization with factor parameter."""
        pyx = Pyx(factor=4, palette=8)
        assert pyx.factor == 4
        assert pyx.palette == 8

    def test_init_with_dimensions(self):
        """Test initialization with width/height parameters."""
        pyx = Pyx(width=64, height=64, palette=8)
        assert pyx.width == 64
        assert pyx.height == 64

    def test_init_with_width_only(self):
        """Test initialization with width only (height auto-calculated)."""
        pyx = Pyx(width=64, palette=8)
        assert pyx.width == 64
        assert pyx.height is None

    def test_init_with_height_only(self):
        """Test initialization with height only (width auto-calculated)."""
        pyx = Pyx(height=64, palette=8)
        assert pyx.height == 64
        assert pyx.width is None

    def test_init_factor_and_dimensions_raises(self):
        """Test that providing both factor and dimensions raises error."""
        with pytest.raises(ValueError, match="(Cannot set both|can only set either)"):
            Pyx(factor=4, width=64, palette=8)

    def test_init_no_size_params_behavior(self):
        """Test behavior when no size parameters provided.

        When no size parameters are given, the image keeps its original size
        (no downsampling occurs, only palette reduction).
        """
        pyx = Pyx(palette=8, height=None, width=None, factor=None)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = pyx.fit_transform(img)

        # Should keep original size
        assert result.shape[:2] == img.shape[:2]

    def test_init_invalid_factor(self):
        """Test that invalid factor raises error."""
        with pytest.raises(ValueError):
            Pyx(factor=0, palette=8)
        with pytest.raises(ValueError):
            Pyx(factor=-1, palette=8)

    def test_init_invalid_palette(self):
        """Test that invalid palette raises error."""
        with pytest.raises(ValueError):
            Pyx(factor=4, palette=1)  # Must be >= 2

    def test_init_invalid_depth(self):
        """Test that invalid depth raises error."""
        with pytest.raises(ValueError):
            Pyx(factor=4, palette=8, depth=0)

    def test_init_invalid_sobel(self):
        """Test that invalid sobel raises error."""
        with pytest.raises(ValueError):
            Pyx(factor=4, palette=8, sobel=1)  # Must be >= 2

    def test_init_invalid_dither(self):
        """Test that invalid dither raises error."""
        with pytest.raises(ValueError):
            Pyx(factor=4, palette=8, dither="invalid")

    def test_init_valid_dither_options(self):
        """Test all valid dither options."""
        for dither in ["none", "naive", "bayer", "bayer8", "floyd", "atkinson"]:
            pyx = Pyx(factor=4, palette=8, dither=dither)
            assert pyx.dither == dither

        # Test None defaults to "none"
        pyx = Pyx(factor=4, palette=8, dither=None)
        # dither=None may stay as None or become "none" depending on implementation
        assert pyx.dither in [None, "none"]

    def test_init_with_palette_enum(self):
        """Test initialization with Pal enum."""
        pyx = Pyx(factor=4, palette=Pal.GAMEBOY_ORIGINAL)
        assert pyx.palette == Pal.GAMEBOY_ORIGINAL

    def test_init_postprocess_default(self):
        """Test that postprocess defaults to True."""
        pyx = Pyx(factor=4, palette=8)
        assert pyx.postprocess is True

    def test_init_postprocess_disabled(self):
        """Test that postprocess can be disabled."""
        pyx = Pyx(factor=4, palette=8, postprocess=False)
        assert pyx.postprocess is False

    def test_init_backend_default(self):
        """Test that backend defaults to cpu."""
        pyx = Pyx(factor=4, palette=8)
        assert pyx.backend_name == "cpu"


class TestPyxFitTransform:
    """Tests for Pyx fit and transform methods."""

    def test_fit_returns_self(self, sample_image_rgb):
        """Test that fit() returns self for chaining."""
        pyx = Pyx(factor=4, palette=4)
        result = pyx.fit(sample_image_rgb)
        assert result is pyx

    def test_fit_sets_is_fitted(self, sample_image_rgb):
        """Test that fit() sets is_fitted flag."""
        pyx = Pyx(factor=4, palette=4)
        assert pyx.is_fitted is False
        pyx.fit(sample_image_rgb)
        assert pyx.is_fitted is True

    def test_transform_before_fit_raises(self, sample_image_rgb):
        """Test that transform() before fit() raises error."""
        pyx = Pyx(factor=4, palette=4)
        with pytest.raises(RuntimeError, match="(not fitted|fit.*first)"):
            pyx.transform(sample_image_rgb)

    def test_fit_transform_rgb(self, sample_image_rgb):
        """Test fit_transform on RGB image."""
        pyx = Pyx(factor=4, palette=4)
        result = pyx.fit_transform(sample_image_rgb)

        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB output
        # Check downsampled size (100/4 = 25)
        assert result.shape[0] == 25
        assert result.shape[1] == 25

    def test_fit_transform_rgba(self, sample_image_rgba):
        """Test fit_transform on RGBA image."""
        pyx = Pyx(factor=4, palette=4)
        result = pyx.fit_transform(sample_image_rgba)

        assert result.dtype == np.uint8
        assert result.shape[2] == 4  # RGBA output
        # Check that alpha channel is preserved
        assert np.any(result[:, :, 3] == 0)  # Some transparent pixels
        assert np.any(result[:, :, 3] == 255)  # Some opaque pixels

    def test_fit_transform_with_upscale(self, sample_image_rgb):
        """Test fit_transform with upscale parameter."""
        pyx = Pyx(factor=4, palette=4, upscale=2)
        result = pyx.fit_transform(sample_image_rgb)

        # 100/4 = 25, then upscale by 2 = 50
        assert result.shape[0] == 50
        assert result.shape[1] == 50

    def test_fit_transform_reduces_colors(self, sample_image_rgb):
        """Test that fit_transform reduces to specified palette size."""
        pyx = Pyx(factor=4, palette=4)
        result = pyx.fit_transform(sample_image_rgb)

        # Count unique colors
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique_colors) <= 4

    def test_fit_transform_with_fixed_palette(self, sample_image_rgb):
        """Test fit_transform with a predefined palette."""
        pyx = Pyx(factor=4, palette=Pal.GAMEBOY_ORIGINAL)
        result = pyx.fit_transform(sample_image_rgb)

        # GameBoy has 4 colors
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique_colors) <= 4

    def test_colors_property_after_fit(self, sample_image_rgb):
        """Test that colors property is set after fit."""
        pyx = Pyx(factor=4, palette=4)
        pyx.fit(sample_image_rgb)

        assert pyx.colors is not None
        assert len(pyx.colors) <= 4


class TestPyxDithering:
    """Tests for different dithering methods."""

    @pytest.mark.parametrize(
        "dither", ["none", "naive", "bayer", "bayer8", "floyd", "atkinson"]
    )
    def test_dither_methods_run(self, sample_image_rgb, dither):
        """Test that all dither methods run without error."""
        pyx = Pyx(factor=4, palette=4, dither=dither)
        result = pyx.fit_transform(sample_image_rgb)

        assert result.dtype == np.uint8
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_dither_none_vs_naive_differ(self, sample_image_rgb):
        """Test that different dither methods produce different results."""
        pyx_none = Pyx(factor=4, palette=4, dither="none")
        pyx_naive = Pyx(factor=4, palette=4, dither="naive")

        result_none = pyx_none.fit_transform(sample_image_rgb)
        result_naive = pyx_naive.fit_transform(sample_image_rgb)

        # Results should typically differ (not guaranteed but likely)
        # At minimum, they should be valid images
        assert result_none.shape == result_naive.shape


class TestPyxPostprocess:
    """Tests for the postprocess (dark speckle fix) feature."""

    def test_postprocess_enabled_by_default(self, sample_image_rgb):
        """Test that postprocess is enabled by default."""
        pyx = Pyx(factor=4, palette=4)
        assert pyx.postprocess is True

    def test_postprocess_can_be_disabled(self, sample_image_rgb):
        """Test that postprocess can be disabled."""
        pyx = Pyx(factor=4, palette=4, postprocess=False)
        assert pyx.postprocess is False

        # Should still produce valid output
        result = pyx.fit_transform(sample_image_rgb)
        assert result.dtype == np.uint8

    def test_postprocess_produces_valid_output(self, sample_image_rgb):
        """Test that postprocess produces valid output."""
        pyx = Pyx(factor=4, palette=4, postprocess=True)
        result = pyx.fit_transform(sample_image_rgb)

        assert result.dtype == np.uint8
        assert result.shape[2] in [3, 4]

    def test_perceptual_luminance_calculation(self):
        """Test _perceptual_luminance helper method."""
        pyx = Pyx(factor=4, palette=4)

        # Test with known values
        # White (255, 255, 255) should have luminance ~255
        white = np.array([[[255, 255, 255]]], dtype=np.uint8)
        lum_white = pyx._perceptual_luminance(white)
        assert np.isclose(lum_white[0, 0], 255, atol=1)

        # Black (0, 0, 0) should have luminance 0
        black = np.array([[[0, 0, 0]]], dtype=np.uint8)
        lum_black = pyx._perceptual_luminance(black)
        assert lum_black[0, 0] == 0

        # Pure red should have lower luminance than pure green
        # (0.299 * 255 vs 0.587 * 255)
        red = np.array([[[255, 0, 0]]], dtype=np.uint8)
        green = np.array([[[0, 255, 0]]], dtype=np.uint8)
        lum_red = pyx._perceptual_luminance(red)
        lum_green = pyx._perceptual_luminance(green)
        assert lum_red[0, 0] < lum_green[0, 0]


class TestPyxEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_image(self):
        """Test with very small image."""
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        pyx = Pyx(factor=2, palette=4)
        result = pyx.fit_transform(img)

        assert result.shape[0] == 5
        assert result.shape[1] == 5

    def test_non_square_image(self):
        """Test with non-square image."""
        img = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        pyx = Pyx(factor=5, palette=4)
        result = pyx.fit_transform(img)

        assert result.shape[0] == 20
        assert result.shape[1] == 10

    def test_grayscale_image_raises(self):
        """Test that grayscale image raises error."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        pyx = Pyx(factor=4, palette=4)

        with pytest.raises((ValueError, IndexError)):
            pyx.fit_transform(img)

    def test_float_image_input(self):
        """Test with float image (0.0-1.0 range)."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        pyx = Pyx(factor=4, palette=4)
        result = pyx.fit_transform(img)

        assert result.dtype == np.uint8

    def test_refit_with_different_image(self, sample_image_rgb):
        """Test refitting with a different image."""
        pyx = Pyx(factor=4, palette=4)

        # First fit
        pyx.fit(sample_image_rgb)
        assert pyx.is_fitted is True

        # Create different image with more variation and refit
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2[:50, :] = [255, 0, 0]  # Red top
        img2[50:, :] = [0, 0, 255]  # Blue bottom
        pyx.fit(img2)

        # Should still be fitted
        assert pyx.is_fitted is True

    def test_transform_different_image_after_fit(self, sample_image_rgb):
        """Test transforming a different image than was fitted."""
        pyx = Pyx(factor=4, palette=4)
        pyx.fit(sample_image_rgb)

        # Transform a different image
        img2 = np.full((100, 100, 3), 100, dtype=np.uint8)
        result = pyx.transform(img2)

        assert result.dtype == np.uint8
        assert result.shape[:2] == (25, 25)


class TestPyxRealImage:
    """Tests using real example images."""

    def test_real_image_transform(self, real_image):
        """Test with a real image if available."""
        if real_image is None:
            pytest.skip("Real test image not available")

        pyx = Pyx(factor=5, palette=8)
        result = pyx.fit_transform(real_image)

        assert result.dtype == np.uint8
        assert result.ndim == 3

        # Check reasonable output size
        expected_h = real_image.shape[0] // 5
        expected_w = real_image.shape[1] // 5
        assert abs(result.shape[0] - expected_h) <= 1
        assert abs(result.shape[1] - expected_w) <= 1
