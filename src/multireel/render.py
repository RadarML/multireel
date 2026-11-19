"""Rendering helpers: turning arrays into 8-bit RGB frames.

Colormaps are given as a `(256, 3)` `uint8` lookup table, or by name, which
needs the `matplotlib` extra:

```sh
pip install multireel[matplotlib]
```
"""

from collections.abc import Sequence
from functools import cache

import numpy as np
from jaxtyping import Bool, Float, Integer, UInt8


def colormap(
    cmap: str | UInt8[np.ndarray, "256 3"],
) -> UInt8[np.ndarray, "256 3"]:
    """Return a colormap as a lookup table.

    Args:
        cmap: a matplotlib colormap name, such as `turbo`, or a lookup
            table, which is returned unchanged.

    Raises:
        ValueError: if a lookup table is not `(256, 3)` `uint8`.
        ImportError: if `cmap` is a name and matplotlib is not installed.
    """
    if isinstance(cmap, str):
        return _named_colormap(cmap)
    if cmap.shape != (256, 3) or cmap.dtype != np.uint8:
        raise ValueError(
            f"Expected a (256, 3) uint8 lookup table, got {cmap.dtype} "
            f"{cmap.shape}.")
    return cmap


@cache
def _named_colormap(name: str) -> UInt8[np.ndarray, "256 3"]:
    """Return a matplotlib colormap as a read-only lookup table."""
    try:
        from matplotlib import colormaps
    except ImportError as e:
        raise ImportError(
            "Colormaps by name need matplotlib; install "
            "multireel[matplotlib].") from e
    rgba = colormaps[name](np.arange(256))
    lut = np.round(rgba[:, :3] * 255).astype(np.uint8)
    lut.flags.writeable = False
    return lut


def colorize(
    values: Float[np.ndarray, "*shape"],
    cmap: str | UInt8[np.ndarray, "256 3"],
    vmin: float = 0.0,
    vmax: float = 1.0,
    valid: Bool[np.ndarray, "*shape"] | None = None,
) -> UInt8[np.ndarray, "*shape 3"]:
    """Color values through a colormap.

    Values from `vmin` to `vmax` span the colormap, and values outside are
    clipped to its ends. Non-finite values, and values where `valid` is
    false, are black.

    Args:
        values: values to color.
        cmap: a matplotlib colormap name, or a `(256, 3)` lookup table.
        vmin: value shown as the first color.
        vmax: value shown as the last color.
        valid: which values to color; defaults to every finite value.

    Returns:
        An RGB image.
    """
    lut = colormap(cmap)
    scaled = (values - vmin) / (vmax - vmin) * 255
    index = np.clip(
        np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0), 0, 255
    ).astype(np.uint8)
    rgb = lut[index]
    keep = np.isfinite(values)
    if valid is not None:
        keep &= valid
    rgb[~keep] = 0
    return rgb


def colorize_labels(
    labels: Integer[np.ndarray, "*shape"],
    colors: Sequence[tuple[int, int, int]] | UInt8[np.ndarray, "N 3"],
) -> UInt8[np.ndarray, "*shape 3"]:
    """Color integer labels, such as semantic classes.

    Args:
        labels: label of each pixel.
        colors: RGB color of each label, from label `0`. Labels without a
            color, including negative ones, are black.

    Returns:
        An RGB image.
    """
    palette = np.concatenate([
        np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
        np.zeros((1, 3), dtype=np.uint8)])
    n = len(palette) - 1
    return palette[np.where((labels >= 0) & (labels < n), labels, n)]


def resize_nearest(
    image: UInt8[np.ndarray, "h w *c"], height: int, width: int
) -> UInt8[np.ndarray, "height width *c"]:
    """Resize an image with nearest-neighbor sampling.

    Each axis is resized separately. Integer scale factors use `np.repeat`,
    and other factors gather whole rows or columns. Columns are resized
    while the image has the fewer rows, since that axis is resized pixel by
    pixel.
    """

    def resize_axis(x: np.ndarray, size: int, axis: int) -> np.ndarray:
        n = x.shape[axis]
        if size == n:
            return x
        if size % n == 0:
            return np.repeat(x, size // n, axis=axis)
        return np.take(x, np.arange(size) * n // size, axis=axis)

    if height > image.shape[0]:
        return resize_axis(resize_axis(image, width, 1), height, 0)
    return resize_axis(resize_axis(image, height, 0), width, 1)
