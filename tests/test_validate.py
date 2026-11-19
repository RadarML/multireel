"""Tests for the validator: one crafted violation per rule."""

import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest
import yaml
from synthetic import ORIGIN, sample_sensors, sample_streams, sample_sync

from multireel import read_manifest, validate_bundle, write_bundle
from multireel.bundle.manifest import stream_path
from multireel.validate import Rule
from multireel.validate.checks import check_stream
from multireel.validate.cli import main
from multireel.validate.probe import StreamProbe, probe


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A valid sample bundle, shared by every test in this module."""
    path = tmp_path_factory.mktemp("bundle")
    write_bundle(
        path, sample_streams(sample_sensors()), sample_sync(),
        recording_id="drive_0423_a", timecode_origin=ORIGIN)
    return path


@pytest.fixture
def copy(bundle: Path, tmp_path: Path) -> Path:
    """A copy of the sample bundle that a test may break."""
    shutil.copytree(bundle, tmp_path / "bundle")
    return tmp_path / "bundle"


def test_valid(bundle: Path) -> None:
    """The sample bundle is valid."""
    assert validate_bundle(bundle) == []


def _without_encoder(p: StreamProbe) -> dict[str, Any]:
    return {k: v for k, v in p.tags.items() if k != "multireel_encoder"}


STREAM_VIOLATIONS: dict[Rule, Callable[[StreamProbe], dict[str, Any]]] = {
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
def test_stream_violation(bundle: Path, rule: Rule) -> None:
    """Each crafted stream property breaks exactly the expected rule."""
    manifest = read_manifest(bundle)
    good = probe(stream_path(bundle, "cam"))
    assert check_stream(good, "cam", manifest) == []
    bad = replace(good, **STREAM_VIOLATIONS[rule](good))
    assert rule in {v.rule for v in check_stream(bad, "cam", manifest)}


def _edit_manifest(path: Path, edit: Callable[[dict], None]) -> None:
    file = path / "manifest.yaml"
    data = yaml.safe_load(file.read_text())
    edit(data)
    file.write_text(yaml.safe_dump(data, sort_keys=False))


def _corrupt(file: Path) -> None:
    """Overwrite part of the frame data with noise."""
    data = bytearray(file.read_bytes())
    start = data.find(b"mdat") + 64
    data[start:start + 2000] = np.random.default_rng(0).integers(
        0, 256, 2000, dtype=np.uint8).tobytes()
    file.write_bytes(bytes(data))


BUNDLE_VIOLATIONS: dict[Rule, Callable[[Path], object]] = {
    "manifest-present": lambda b: (b / "manifest.yaml").unlink(),
    "manifest-yaml": lambda b: (b / "manifest.yaml").write_text(
        (b / "manifest.yaml").read_text() + "extra: &a 1\nmore: *a\n"),
    "schema-version": lambda b: _edit_manifest(
        b, lambda d: d.update(schema_version="2.0")),
    "manifest-fields": lambda b: _edit_manifest(
        b, lambda d: d.pop("frame_count")),
    "stream-id": lambda b: stream_path(b, "radar").unlink(),
    "timecode-origin": lambda b: _edit_manifest(
        b, lambda d: d.update(timecode_origin="2026-11-18T16:00:00.5Z")),
    "timecode-range": lambda b: _edit_manifest(
        b, lambda d: d.update(frame_count=30 * 3600 * 23)),
    "overlay-link": lambda b: _edit_manifest(
        b, lambda d: d["streams"]["boxes"].update(overlay_of="nope")),
    "overlay-match": lambda b: _edit_manifest(
        b, lambda d: d["streams"]["boxes"].update(overlay_of="radar")),
    "decode": lambda b: _corrupt(stream_path(b, "cam")),
}


@pytest.mark.parametrize("rule", list(BUNDLE_VIOLATIONS))
def test_bundle_violation(copy: Path, rule: Rule) -> None:
    """Each crafted bundle breaks the expected rule."""
    BUNDLE_VIOLATIONS[rule](copy)
    assert rule in {v.rule for v in validate_bundle(copy)}


def test_every_rule_is_tested() -> None:
    """Every rule has a crafted violation."""
    assert set(STREAM_VIOLATIONS) | set(BUNDLE_VIOLATIONS) == set(
        get_args(Rule))


def test_cli(bundle: Path, tmp_path: Path) -> None:
    """The command exits with 0 for a valid bundle, and 1 otherwise."""
    assert main([str(bundle)]) == 0
    assert main([str(tmp_path)]) == 1
