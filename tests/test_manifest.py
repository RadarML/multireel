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
    read_manifest,
    write_manifest,
)
from multireel.bundle.timing import Timecode


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
            "front_cam": StreamEntry("hevc-main10-420", label="no", info=info),
            "front_boxes": StreamEntry(
                "prores-4444", overlay_of="front_cam",
                alpha_mode="premultiplied", info=info),
        },
        provenance={"git": "abc123", "args": ["--rate", "30"]})
    fields.update(kwargs)
    return Manifest(**fields)


def test_round_trip(tmp_path: Path) -> None:
    """A written manifest reads back unchanged."""
    manifest = make_manifest()
    write_manifest(tmp_path, manifest)
    assert read_manifest(tmp_path) == manifest


def test_ambiguous_strings_quoted(tmp_path: Path) -> None:
    """Strings that look like numbers or booleans stay strings."""
    write_manifest(tmp_path, make_manifest())
    data = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert data["recording_id"] == "1.10"
    assert data["streams"]["front_cam"]["label"] == "no"
    assert data["streams"]["front_cam"]["info"]["flag"] == "on"


def test_no_aliases(tmp_path: Path) -> None:
    """Shared objects are written out in full, without anchors."""
    write_manifest(tmp_path, make_manifest())
    text = (tmp_path / "manifest.yaml").read_text()
    assert "&" not in text and "*" not in text


def test_no_temporary_files_left(tmp_path: Path) -> None:
    """The temporary file is renamed to manifest.yaml."""
    write_manifest(tmp_path, make_manifest())
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.yaml"]


def test_newer_minor_version(tmp_path: Path) -> None:
    """A newer minor version with unknown fields is accepted."""
    write_manifest(tmp_path, make_manifest())
    path = tmp_path / "manifest.yaml"
    data = yaml.safe_load(path.read_text())
    data["schema_version"] = "1.7"
    data["future_field"] = 1
    data["streams"]["front_cam"]["future_field"] = 2
    path.write_text(yaml.safe_dump(data))
    assert read_manifest(tmp_path) == replace(
        make_manifest(), bundle_id=read_manifest(tmp_path).bundle_id,
        schema_version="1.7")


def test_unknown_major_version(tmp_path: Path) -> None:
    """An unknown major version is rejected."""
    write_manifest(tmp_path, make_manifest(schema_version="2.0"))
    with pytest.raises(ValueError):
        read_manifest(tmp_path)


def test_missing_field(tmp_path: Path) -> None:
    """A missing required field is a ValueError."""
    write_manifest(tmp_path, make_manifest())
    path = tmp_path / "manifest.yaml"
    data = yaml.safe_load(path.read_text())
    del data["frame_count"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        read_manifest(tmp_path)
