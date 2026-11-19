"""Tests for stream validation: one crafted violation per rule."""

import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from synthetic import ORIGIN, sample_sensors, sample_streams, sample_sync

from multireel import BundleWriter, Manifest, validate_stream
from multireel._validate import StreamProbe, check_stream, probe


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A valid sample bundle, shared by every test in this module."""
    path = tmp_path_factory.mktemp("bundle")
    BundleWriter(
        path, sample_streams(sample_sensors()), sample_sync(),
        recording_id="drive_0423_a", timecode_origin=ORIGIN).write()
    return path


def _without_encoder(p: StreamProbe) -> dict[str, Any]:
    return {k: v for k, v in p.tags.items() if k != "multireel_encoder"}


STREAM_VIOLATIONS: dict[str, Callable[[StreamProbe], dict[str, Any]]] = {
    "profile": lambda p: {"pixel_format": "yuv420p"},
    "edit-rate": lambda p: {"frame_rate": "25/1"},
    "constant-frame-rate": lambda p: {
        "durations": np.r_[p.durations[:-1], p.durations[-1] * 2]},
    "frame-count": lambda p: {
        "durations": p.durations[:-1], "keyframes": p.keyframes[:-1]},
    "start-timecode": lambda p: {"timecode": "01:00:00:00"},
    "no-b-frames": lambda p: {"has_reordering": True},
    "keyframe-alignment": lambda p: {
        "keyframes": np.arange(len(p.keyframes)) == 0},
    "dimensions": lambda p: {"width": p.width + 1},
    "square-pixels": lambda p: {"sample_aspect_ratio": "4:3"},
    "progressive": lambda p: {"rotation": 90.0},
    "color": lambda p: {"color": {**p.color, "color_range": "pc"}},
    "tracks": lambda p: {"tracks": [*p.tracks, ("audio", "mp4a")]},
    "hvc1-tag": lambda p: {"tracks": [("video", "hev1"), ("data", "tmcd")]},
    "timecode-track": lambda p: {"timecode": "01;12;00;00"},
    "colr-tag": lambda p: {"colr": None},
    "reel-name": lambda p: {"reel_name": "drive_0423_a.radar"},
    "clip-name": lambda p: {"tags": {**p.tags, "title": "cam"}},
    "encoder-metadata": lambda p: {"tags": _without_encoder(p)},
}


@pytest.mark.parametrize("rule", list(STREAM_VIOLATIONS))
def test_stream_violation(bundle: Path, rule: str) -> None:
    """Each crafted stream property breaks exactly the expected rule."""
    manifest = Manifest.read(bundle)
    good = probe(Manifest.stream_path(bundle, "cam"))
    assert check_stream(good, "cam", manifest) == []
    bad = replace(good, **STREAM_VIOLATIONS[rule](good))
    assert any(m.startswith(f"{rule}: ")
               for m in check_stream(bad, "cam", manifest))


def test_decode_violation(bundle: Path, tmp_path: Path) -> None:
    """A stream file with corrupted frame data breaks the `decode` rule."""
    manifest = Manifest.read(bundle)
    file = tmp_path / "cam.mov"
    shutil.copy(Manifest.stream_path(bundle, "cam"), file)
    assert validate_stream(file, "cam", manifest) == []
    data = bytearray(file.read_bytes())
    start = data.find(b"mdat") + 64
    data[start:start + 2000] = np.random.default_rng(0).integers(
        0, 256, 2000, dtype=np.uint8).tobytes()
    file.write_bytes(bytes(data))
    assert any(m.startswith("decode: ")
               for m in validate_stream(file, "cam", manifest))



def test_prores_level_violation(bundle: Path) -> None:
    """A ProRes stream with another level's sample entry breaks `profile`."""
    manifest = Manifest.read(bundle)
    good = probe(Manifest.stream_path(bundle, "lidar"))
    assert check_stream(good, "lidar", manifest) == []
    tracks = [("video", "apcs"), ("data", "tmcd")]
    assert any(m.startswith("profile: ") for m in check_stream(
        replace(good, tracks=tracks), "lidar", manifest))
