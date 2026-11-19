"""Tests for the rendering helpers."""

import numpy as np
import pytest

from multireel.render import (
    colorize,
    colorize_labels,
    colormap,
    resize_nearest,
)

GRAY = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)


def test_colorize_range_and_mask() -> None:
    """Values span the colormap; masked and non-finite values are black."""
    values = np.array([0.0, 0.5, 1.0, 2.0, -1.0, np.nan, np.inf, 0.5])
    valid = np.array([True] * 7 + [False])
    rgb = colorize(values, GRAY, valid=valid)
    assert rgb.shape == (8, 3) and rgb.dtype == np.uint8
    assert rgb[:, 0].tolist() == [0, 127, 255, 255, 0, 0, 0, 0]


def test_colormap_by_name() -> None:
    """Named colormaps come from matplotlib, as read-only lookup tables."""
    lut = colormap("gray")
    assert lut.shape == (256, 3) and lut.dtype == np.uint8
    assert lut[0].tolist() == [0, 0, 0] and lut[-1].tolist() == [255] * 3
    assert not lut.flags.writeable
    assert np.array_equal(colorize(np.array([1.0]), "gray"), [[255] * 3])


def test_colormap_rejects_bad_table() -> None:
    """Lookup tables must be (256, 3) uint8."""
    with pytest.raises(ValueError):
        colormap(np.zeros((255, 3), dtype=np.uint8))


def test_colorize_labels() -> None:
    """Labels index the colors; labels without a color are black."""
    labels = np.array([[0, 1], [2, -1]])
    rgb = colorize_labels(labels, [(10, 20, 30), (40, 50, 60)])
    assert rgb.tolist() == [
        [[10, 20, 30], [40, 50, 60]], [[0, 0, 0], [0, 0, 0]]]


@pytest.mark.parametrize("size", [(8, 12), (2, 3), (6, 4), (5, 7)])
def test_resize_nearest(size: tuple[int, int]) -> None:
    """Resizing matches nearest-neighbor indexing."""
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    height, width = size
    rows = np.arange(height) * 4 // height
    cols = np.arange(width) * 6 // width
    assert np.array_equal(
        resize_nearest(image, height, width), image[rows][:, cols])
