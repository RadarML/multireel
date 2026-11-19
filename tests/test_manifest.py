"""Tests for reading and writing manifests."""

from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml
from synthetic import ORIGIN

from multireel.bundle.manifest import (
    Manifest,
    StreamEntry,
)
from multireel.bundle.timing import Timecode
from multireel.bundle.video import Profile


def make_manifest(**kwargs: Any) -> Manifest:
    """A small manifest with an overlay and free-form fields."""
    info = {"clock_source": "ptp", "flag": "on"}
    fields: dict[str, Any] = dict(
        bundle_id=uuid4(),
        recording_id="1.10",
        timecode_origin=ORIGIN,
        edit_rate=30,
        keyframe_interval=15,
        start_timecode=Timecode(1, 12, 0, 0, 30),
        frame_count=9000,
        streams={
            "front_cam": StreamEntry(
                Profile.from_name("hevc-main10-420"), label="no", info=info),
            "front_boxes": StreamEntry(
                Profile.from_name("prores-4444"), overlay_of="front_cam",
                alpha_mode="premultiplied", info=info),
        },
        provenance={"git": "abc123", "args": ["--rate", "30"]})
    fields.update(kwargs)
    return Manifest(**fields)


def test_round_trip(tmp_path: Path) -> None:
    """A written manifest reads back unchanged."""
    manifest = make_manifest()
    manifest.write(tmp_path)
    assert Manifest.read(tmp_path) == manifest


def test_ambiguous_strings_quoted(tmp_path: Path) -> None:
    """Strings that look like numbers or booleans stay strings."""
    make_manifest().write(tmp_path)
    data = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert data["recording_id"] == "1.10"
    assert data["streams"]["front_cam"]["label"] == "no"
    assert data["streams"]["front_cam"]["info"]["flag"] == "on"


def test_no_aliases(tmp_path: Path) -> None:
    """Shared objects are written out in full, without anchors."""
    make_manifest().write(tmp_path)
    text = (tmp_path / "manifest.yaml").read_text()
    assert "&" not in text and "*" not in text


def test_no_temporary_files_left(tmp_path: Path) -> None:
    """The temporary file is renamed to manifest.yaml."""
    make_manifest().write(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.yaml"]


def test_newer_minor_version(tmp_path: Path) -> None:
    """A newer minor version with unknown fields is accepted."""
    make_manifest().write(tmp_path)
    path = tmp_path / "manifest.yaml"
    data = yaml.safe_load(path.read_text())
    data["schema_version"] = "1.7"
    data["future_field"] = 1
    data["streams"]["front_cam"]["future_field"] = 2
    path.write_text(yaml.safe_dump(data))
    assert Manifest.read(tmp_path) == replace(
        make_manifest(), bundle_id=Manifest.read(tmp_path).bundle_id,
        schema_version="1.7")


def test_unknown_major_version(tmp_path: Path) -> None:
    """An unknown major version is rejected."""
    make_manifest(schema_version="2.0").write(tmp_path)
    with pytest.raises(ValueError):
        Manifest.read(tmp_path)


def test_missing_field(tmp_path: Path) -> None:
    """A missing required field is a ValueError."""
    make_manifest().write(tmp_path)
    path = tmp_path / "manifest.yaml"
    data = yaml.safe_load(path.read_text())
    del data["frame_count"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        Manifest.read(tmp_path)


def test_custom_profile(tmp_path: Path) -> None:
    """A custom profile is written as a mapping and reads back unchanged."""
    custom = Profile("hevc", "422", 12, False)
    manifest = make_manifest(streams={"cam": StreamEntry(custom)})
    manifest.write(tmp_path)
    data = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert data["streams"]["cam"]["profile"] == {
        "codec": "hevc", "chroma": "422", "bit_depth": 12, "alpha": False}
    assert Manifest.read(tmp_path) == manifest


def test_standard_profile_written_by_name(tmp_path: Path) -> None:
    """A profile equal to a standard one is written as its name."""
    lt = Profile("prores", "422", 10, False, "lt")
    make_manifest(streams={"cam": StreamEntry(lt)}).write(tmp_path)
    data = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert data["streams"]["cam"]["profile"] == "prores-422-lt"


def test_unknown_profile_name(tmp_path: Path) -> None:
    """A profile name that is not a standard profile is a ValueError."""
    make_manifest().write(tmp_path)
    path = tmp_path / "manifest.yaml"
    data = yaml.safe_load(path.read_text())
    data["streams"]["front_cam"]["profile"] = "h264"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="h264"):
        Manifest.read(tmp_path)
