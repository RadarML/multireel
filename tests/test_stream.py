"""Tests for stream constructors: which samples are loaded and rendered."""

import threading
from collections.abc import Iterator

import numpy as np
import pytest
from jaxtyping import UInt8
from synthetic import SyntheticSensor

from multireel import Stream
from multireel._stream import _ordered_map


def tag(i: int) -> UInt8[np.ndarray, "1 1 3"]:
    """A 1x1 frame holding `i`."""
    return np.full((1, 1, 3), i, dtype=np.uint8)


def values(frames: Iterator[UInt8[np.ndarray, "H W C"]]) -> list[int]:
    """The value held by each frame."""
    return [int(f[0, 0, 0]) for f in frames]


@pytest.mark.parametrize("workers", [1, 4])
def test_random_access(workers: int) -> None:
    """Only the requested samples are loaded, and frames come in order."""
    loaded: list[int] = []
    lock = threading.Lock()

    def load(i: int) -> int:
        with lock:
            loaded.append(i)
        return i

    stream = Stream.random_access(
        np.arange(50.0), load, tag, workers=workers)
    indices = np.array([3, 4, 10, 30, 49])
    assert values(stream.frames(indices)) == [3, 4, 10, 30, 49]
    assert sorted(loaded) == [3, 4, 10, 30, 49]


@pytest.mark.parametrize("workers", [1, 4])
def test_sequential(workers: int) -> None:
    """Samples are read in order up to the last requested one."""
    read: list[int] = []

    def samples() -> Iterator[int]:
        for i in range(50):
            read.append(i)
            yield i

    rendered: list[int] = []
    lock = threading.Lock()

    def render(i: int) -> UInt8[np.ndarray, "1 1 3"]:
        with lock:
            rendered.append(i)
        return tag(i)

    stream = Stream.sequential(
        np.arange(50.0), samples, render, workers=workers)
    assert values(stream.frames(np.array([2, 5, 20]))) == [2, 5, 20]
    assert read == list(range(21))
    assert sorted(rendered) == [2, 5, 20]


@pytest.mark.parametrize("sequential", [False, True])
def test_from_sensor(sequential: bool) -> None:
    """Sensor streams use the sensor's timestamps and samples."""
    sensor = SyntheticSensor(np.arange(20.0))
    stream = Stream.from_sensor(
        sensor, lambda s: tag(int(s)), sequential=sequential, workers=2)
    assert np.array_equal(stream.timestamps, np.arange(20.0))
    assert values(stream.frames(np.array([0, 7, 19]))) == [0, 7, 19]


def test_ordered_map_stops_early() -> None:
    """Closing the iterator early stops submitting work."""
    calls: list[int] = []
    lock = threading.Lock()

    def fn(i: int) -> int:
        with lock:
            calls.append(i)
        return i

    results = _ordered_map(fn, range(1000), workers=4)
    assert next(results) == 0
    results.close()
    assert len(calls) <= 2 * 4 + 1


def test_ordered_map_raises() -> None:
    """An error in a worker is raised in the consumer."""

    def fn(i: int) -> int:
        if i == 3:
            raise RuntimeError("bad sample")
        return i

    with pytest.raises(RuntimeError, match="bad sample"):
        list(_ordered_map(fn, range(10), workers=2))
