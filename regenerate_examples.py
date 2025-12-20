#!/usr/bin/env python
"""Regenerate all example images in the examples/ directory.

This script recreates all the p_*.png images used in the README and documentation.
It uses the same parameters as examples.ipynb but runs as a standalone script.

Usage:
    uv run python regenerate_examples.py

Note: The p_palms.png image takes longer to generate (~30-60s) as it demonstrates
all 5 dithering methods on a larger image.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from pyxelate import Pyx, Pal


def plot_and_save(subplots: list, save_as: str, fig_h: int = 9) -> None:
    """Create a subplot figure and save to examples/ directory.

    Args:
        subplots: List of images or dicts with 'title' and 'image' keys.
        save_as: Filename to save (will be saved to examples/).
        fig_h: Figure height in inches.
    """
    n_subplots = len(subplots)
    n_cols = min(3, n_subplots)
    n_rows = int(np.ceil(n_subplots / 3))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_h))

    # Handle single subplot case
    if n_subplots == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for i, subplot in enumerate(subplots):
        if isinstance(subplot, dict):
            axes[i].set_title(subplot["title"])
            # Use interpolation='nearest' to preserve crisp pixel art edges
            axes[i].imshow(subplot["image"], interpolation="nearest")
        else:
            # Use interpolation='nearest' to preserve crisp pixel art edges
            axes[i].imshow(subplot, interpolation="nearest")
        axes[i].axis("off")

    # Hide any unused subplots
    for i in range(n_subplots, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()

    save_path = Path("examples") / save_as
    plt.savefig(save_path, transparent=True, dpi=100)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_blazkowicz() -> None:
    """Generate p_blazkowicz.png - basic example."""
    print("Generating p_blazkowicz.png...")
    start = time.time()

    image = io.imread("examples/blazkowicz.jpg")
    # Increased palette from 7 to 8 to capture bright blue highlights
    # with newer sklearn/skimage versions
    pyx = Pyx(factor=14, palette=8)
    pyx.fit(image)
    new_image = pyx.transform(image)

    plot_and_save([image, new_image], "p_blazkowicz.png")
    print(f"  Done in {time.time() - start:.1f}s")


def generate_fit_transform() -> None:
    """Generate p_fit_transform.png - demonstrates fit/transform on different images."""
    print("Generating p_fit_transform.png...")
    start = time.time()

    car = io.imread("examples/f1.jpg")
    robocop = io.imread("examples/robocop.jpg")

    # Fit a model on each
    pyx_car = Pyx(factor=5, palette=8, dither="none").fit(car)
    # Increased palette from 7 to 8 to capture orange visor reflection
    pyx_robocop = Pyx(factor=6, palette=8, dither="naive", svd=True).fit(robocop)

    plot_and_save(
        [
            {"title": "fit(car)", "image": car},
            {"title": "transform(car)", "image": pyx_car.transform(car)},
            {"title": "transform(robocop)", "image": pyx_car.transform(robocop)},
            {"title": "fit(robocop)", "image": robocop},
            {"title": "transform(car)", "image": pyx_robocop.transform(car)},
            {"title": "transform(robocop)", "image": pyx_robocop.transform(robocop)},
        ],
        "p_fit_transform.png",
        fig_h=18,
    )
    print(f"  Done in {time.time() - start:.1f}s")


def generate_br() -> None:
    """Generate p_br.png - Blade Runner with naive dither."""
    print("Generating p_br.png...")
    start = time.time()

    br = io.imread("examples/br.jpg")
    br_p = Pyx(factor=6, palette=6, dither="naive").fit_transform(br)

    plot_and_save([br, br_p], "p_br.png", fig_h=6)
    print(f"  Done in {time.time() - start:.1f}s")


def generate_br2() -> None:
    """Generate p_br2.png - Blade Runner 2 with atkinson dither."""
    print("Generating p_br2.png...")
    start = time.time()

    br2 = io.imread("examples/br2.jpg")
    br2_p = Pyx(factor=3, palette=7, dither="atkinson").fit_transform(br2)

    plot_and_save([br2, br2_p], "p_br2.png", fig_h=6)
    print(f"  Done in {time.time() - start:.1f}s")


def generate_trex() -> None:
    """Generate p_trex.png - PNG with alpha channel."""
    print("Generating p_trex.png...")
    start = time.time()

    trex = io.imread("examples/trex.png")
    p_trex = Pyx(factor=9, palette=4, dither="naive", alpha=0.6).fit_transform(trex)

    plot_and_save(
        [
            {"title": "Converting PNG with alpha channel", "image": trex},
            {
                "title": "Pixels are either opaque/transparent above/below alpha threshold",
                "image": p_trex,
            },
        ],
        "p_trex.png",
    )
    print(f"  Done in {time.time() - start:.1f}s")


def generate_palms() -> None:
    """Generate p_palms.png - demonstrates all dithering methods.

    Note: This takes longer as it processes the image with 5 different dither methods.
    """
    print("Generating p_palms.png (this may take a while)...")
    start = time.time()

    palm = io.imread("examples/palms3.jpg")

    print("  Processing dither='none'...")
    palm_none = Pyx(factor=4, palette=6, dither="none").fit_transform(palm)

    print("  Processing dither='naive'...")
    palm_naive = Pyx(factor=4, palette=6, dither="naive").fit_transform(palm)

    print("  Processing dither='bayer'...")
    palm_bayer = Pyx(factor=4, palette=6, dither="bayer").fit_transform(palm)

    print("  Processing dither='floyd'...")
    palm_floyd = Pyx(factor=4, palette=6, dither="floyd").fit_transform(palm)

    print("  Processing dither='atkinson'...")
    palm_atkinson = Pyx(factor=4, palette=6, dither="atkinson").fit_transform(palm)

    plot_and_save(
        [
            {"title": "Original", "image": palm},
            {"title": 'Pyx(factor=5, palette=6, dither="none")', "image": palm_none},
            {"title": 'Pyx(factor=5, palette=6, dither="naive")', "image": palm_naive},
            {"title": 'Pyx(factor=5, palette=6, dither="bayer")', "image": palm_bayer},
            {"title": 'Pyx(factor=5, palette=6, dither="floyd")', "image": palm_floyd},
            {
                "title": 'Pyx(factor=5, palette=6, dither="atkinson")',
                "image": palm_atkinson,
            },
        ],
        "p_palms.png",
        fig_h=18,
    )
    print(f"  Done in {time.time() - start:.1f}s")


def generate_vangogh() -> None:
    """Generate p_vangogh.png - demonstrates retro palettes."""
    print("Generating p_vangogh.png...")
    start = time.time()

    vangogh = io.imread("examples/vangogh.jpg")

    vangogh_apple = Pyx(
        factor=12, palette=Pal.APPLE_II_HI, dither="atkinson"
    ).fit_transform(vangogh)

    vangogh_mspaint = Pyx(
        factor=8, palette=Pal.MICROSOFT_WINDOWS_PAINT, dither="none"
    ).fit_transform(vangogh)

    plot_and_save(
        [
            {"title": "Apple II", "image": vangogh_apple},
            {"title": "Windows Paint", "image": vangogh_mspaint},
        ],
        "p_vangogh.png",
    )
    print(f"  Done in {time.time() - start:.1f}s")


def generate_corgi() -> None:
    """Generate p_corgi.png - demonstrates various palettes."""
    print("Generating p_corgi.png...")
    start = time.time()

    corgi = io.imread("examples/corgi.jpg")

    # Find 5 colors automatically
    corgi_5 = Pyx(factor=8, palette=5, dither="none", svd=True).fit_transform(corgi)

    # Assign CGA palette
    corgi_cga = Pyx(
        factor=8, palette=Pal.CGA_MODE4_PAL1, dither="naive", svd=True
    ).fit_transform(corgi)

    # Assign ZX Spectrum palette
    corgi_bbc = Pyx(
        factor=8, palette=Pal.ZX_SPECTRUM, dither="naive", svd=False
    ).fit_transform(corgi)

    # Assign Game Boy palette
    corgi_gb = Pyx(
        factor=8, palette=Pal.GAMEBOY_ORIGINAL, dither="none", svd=True
    ).fit_transform(corgi)

    # Assign C64 palette
    corgi_c64 = Pyx(
        factor=8, palette=Pal.COMMODORE_64, dither="naive", svd=True
    ).fit_transform(corgi)

    plot_and_save(
        [
            {"title": "Original", "image": corgi},
            {"title": "5 color palette", "image": corgi_5},
            {"title": "CGA", "image": corgi_cga},
            {"title": "BBC Micro", "image": corgi_bbc},
            {"title": "Game Boy Original", "image": corgi_gb},
            {"title": "Commodore 64", "image": corgi_c64},
        ],
        "p_corgi.png",
        fig_h=9,
    )
    print(f"  Done in {time.time() - start:.1f}s")


def main() -> None:
    """Regenerate all example images."""
    import warnings

    # Suppress warnings during generation (some palette combinations trigger warnings)
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("Regenerating example images for pyxelate")
    print("=" * 60)
    print()

    total_start = time.time()

    # Generate all images
    generate_blazkowicz()
    generate_fit_transform()
    generate_br()
    generate_br2()
    generate_trex()
    generate_palms()
    generate_vangogh()
    generate_corgi()

    print()
    print("=" * 60)
    print(f"All images regenerated in {time.time() - total_start:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
