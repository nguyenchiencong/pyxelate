"""Unit tests for the Pal class (palette handling)."""

import numpy as np
import pytest

from pyxelate import Pal
from pyxelate.pal import BasePalette


class TestPalEnum:
    """Tests for Pal enum members."""

    def test_pal_has_members(self):
        """Test that Pal has palette members."""
        assert len(list(Pal)) > 0

    def test_common_palettes_exist(self):
        """Test that common palettes exist."""
        expected_palettes = [
            "GAMEBOY_ORIGINAL",
            "PICO_8",
            "COMMODORE_64",
            "CGA_MODE4_PAL1",
            "MICROSOFT_WINDOWS_PAINT",
            "MONO_BW",
        ]
        for name in expected_palettes:
            assert hasattr(Pal, name), f"Missing palette: {name}"

    def test_palette_values_are_lists(self):
        """Test that palette values are lists of colors."""
        for pal in Pal:
            assert isinstance(pal.value, list)
            assert len(pal.value) > 0

    def test_palette_colors_are_rgb(self):
        """Test that palette colors are RGB format (3 values per color)."""
        for pal in Pal:
            for color in pal.value:
                assert len(color) == 1  # Each color is wrapped in a list
                assert len(color[0]) == 3  # RGB values

    def test_palette_colors_normalized(self):
        """Test that palette colors are normalized (0.0-1.0 range)."""
        for pal in Pal:
            for color in pal.value:
                rgb = color[0]
                assert all(0.0 <= c <= 1.0 for c in rgb), (
                    f"Color out of range in {pal.name}: {rgb}"
                )


class TestPalLength:
    """Tests for palette length functionality."""

    def test_len_gameboy(self):
        """Test length of GameBoy palette (4 colors)."""
        assert len(Pal.GAMEBOY_ORIGINAL) == 4

    def test_len_pico8(self):
        """Test length of PICO-8 palette (16 colors)."""
        assert len(Pal.PICO_8) == 16

    def test_len_mono(self):
        """Test length of mono palettes (2 colors)."""
        assert len(Pal.MONO_BW) == 2

    def test_len_commodore64(self):
        """Test length of C64 palette (16 colors)."""
        assert len(Pal.COMMODORE_64) == 16


class TestPalIteration:
    """Tests for palette iteration."""

    def test_iteration(self):
        """Test that palettes can be iterated."""
        colors = list(Pal.GAMEBOY_ORIGINAL)
        assert len(colors) == 4

    def test_iteration_returns_colors(self):
        """Test that iteration returns color values."""
        for color in Pal.MONO_BW:
            assert len(color) == 1
            assert len(color[0]) == 3


class TestPalList:
    """Tests for Pal.list() method."""

    def test_list_returns_strings(self):
        """Test that list() returns list of string names."""
        names = Pal.list()
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)

    def test_list_contains_known_palettes(self):
        """Test that list() contains known palette names."""
        names = Pal.list()
        assert "GAMEBOY_ORIGINAL" in names
        assert "PICO_8" in names


class TestPalFromHex:
    """Tests for Pal.from_hex() static method."""

    def test_from_hex_simple(self):
        """Test from_hex with simple colors."""
        result = Pal.from_hex(["#000000", "#FFFFFF"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1, 3)

        # Black should be [0, 0, 0]
        np.testing.assert_array_almost_equal(result[0, 0], [0, 0, 0])
        # White should be [1, 1, 1]
        np.testing.assert_array_almost_equal(result[1, 0], [1, 1, 1])

    def test_from_hex_without_hash(self):
        """Test from_hex without # prefix."""
        result = Pal.from_hex(["FF0000", "00FF00", "0000FF"])

        assert result.shape == (3, 1, 3)
        np.testing.assert_array_almost_equal(result[0, 0], [1, 0, 0])  # Red
        np.testing.assert_array_almost_equal(result[1, 0], [0, 1, 0])  # Green
        np.testing.assert_array_almost_equal(result[2, 0], [0, 0, 1])  # Blue

    def test_from_hex_mixed(self):
        """Test from_hex with mixed # prefix."""
        result = Pal.from_hex(["#FF0000", "00FF00"])
        assert result.shape == (2, 1, 3)

    def test_from_hex_gameboy_colors(self):
        """Test from_hex with GameBoy-like colors."""
        result = Pal.from_hex(["#0f380f", "#306230", "#8bac0f", "#9bbc0f"])
        assert result.shape == (4, 1, 3)
        # All values should be in 0-1 range
        assert np.all(result >= 0) and np.all(result <= 1)


class TestPalFromRgb:
    """Tests for Pal.from_rgb() static method."""

    def test_from_rgb_simple(self):
        """Test from_rgb with simple colors."""
        result = Pal.from_rgb([[0, 0, 0], [255, 255, 255]])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1, 3)

        # Black should be [0, 0, 0]
        np.testing.assert_array_almost_equal(result[0, 0], [0, 0, 0])
        # White should be [1, 1, 1]
        np.testing.assert_array_almost_equal(result[1, 0], [1, 1, 1])

    def test_from_rgb_primary_colors(self):
        """Test from_rgb with primary colors."""
        result = Pal.from_rgb([[255, 0, 0], [0, 255, 0], [0, 0, 255]])

        assert result.shape == (3, 1, 3)
        np.testing.assert_array_almost_equal(result[0, 0], [1, 0, 0])  # Red
        np.testing.assert_array_almost_equal(result[1, 0], [0, 1, 0])  # Green
        np.testing.assert_array_almost_equal(result[2, 0], [0, 0, 1])  # Blue

    def test_from_rgb_normalized_output(self):
        """Test that from_rgb normalizes to 0-1 range."""
        result = Pal.from_rgb([[128, 128, 128]])

        # Should be approximately 0.5
        np.testing.assert_array_almost_equal(
            result[0, 0], [128 / 255, 128 / 255, 128 / 255]
        )


class TestPalWithPyx:
    """Integration tests for Pal with Pyx."""

    def test_pyx_with_gameboy_palette(self):
        """Test that Pyx works with GameBoy palette."""
        from pyxelate import Pyx

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pyx = Pyx(factor=4, palette=Pal.GAMEBOY_ORIGINAL)
        result = pyx.fit_transform(img)

        # Should have at most 4 colors (GameBoy palette)
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique_colors) <= 4

    def test_pyx_with_custom_hex_palette(self):
        """Test that Pyx works with custom hex palette."""
        from pyxelate import Pyx

        custom_palette = Pal.from_hex(["#000000", "#FFFFFF", "#FF0000", "#00FF00"])
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pyx = Pyx(factor=4, palette=custom_palette)
        result = pyx.fit_transform(img)

        assert result.dtype == np.uint8
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        assert len(unique_colors) <= 4

    def test_pyx_with_custom_rgb_palette(self):
        """Test that Pyx works with custom RGB palette."""
        from pyxelate import Pyx

        custom_palette = Pal.from_rgb([[0, 0, 0], [255, 255, 255], [128, 128, 128]])
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pyx = Pyx(factor=4, palette=custom_palette)
        result = pyx.fit_transform(img)

        assert result.dtype == np.uint8


class TestPalAliases:
    """Tests for palette aliases."""

    def test_bbc_micro_is_teletext(self):
        """Test that BBC_MICRO is alias for TELETEXT."""
        assert Pal.BBC_MICRO.value == Pal.TELETEXT.value

    def test_apple_ii_mono_aliases(self):
        """Test Apple II mono aliases."""
        assert Pal.APPLE_II_MONO.value == Pal.MONO_PHOSPHOR_APPLE.value
        assert Pal.APPLE_II_MONOC.value == Pal.MONO_PHOSPHOR_APPLEC.value
