# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.3] - 2025-06-19

### Changed
- Replaced `skimage.morphology.square` with `footprint_rectangle` (fixes deprecation warning)
- Optimized BGM LAB conversion by pre-computing `rgb2lab(X)` outside the loop in `_initialize_parameters()`
- Replaced all `assert` statements with proper exceptions (`ValueError`, `TypeError`, `RuntimeError`) for better error handling

## [2.1.2] - 2025-06-19

### Changed
- Migrated project configuration from `setup.py` and `requirements.txt` to `pyproject.toml`
- Updated to support NumPy 2.x (tested with NumPy 2.3.5)
- Updated to support latest scikit-learn (fixed `BGM._initialize_parameters` compatibility)
- Updated installation instructions to recommend [uv](https://docs.astral.sh/uv/)
- Repository moved to [github.com/nguyenchiencong/pyxelate](https://github.com/nguyenchiencong/pyxelate)

### Removed
- Removed legacy `setup.py` file
- Removed legacy `requirements.txt` file

### Fixed
- Fixed compatibility issue with scikit-learn's new array API support (`xp` parameter)

## [2.1.1] - Previous Release

### Features
- Super Pyxelate converts images to 8-bit pixel art
- Palette transfer support with predefined retro palettes (Apple II, CGA, Game Boy, etc.)
- Multiple dithering algorithms: none, naive, bayer, floyd, atkinson
- SVD-based noise reduction
- Alpha channel support for sprites
- Video/animation support via `Vid` class
- CLI tool for batch processing
