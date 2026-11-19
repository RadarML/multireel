"""End-to-end tests: write bundles, then decode and check every frame."""

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jaxtyping import Int64, UInt8
from synthetic import (
    ORIGIN,
    SyntheticSensor,
    decode_pattern,
    expected_samples,
    read_frames,
    render_pattern,
    sample_sensors,
    sample_streams,
    sample_sync,
)

from multireel import (
    EncodeOptions,
    Manifest,
    Stream,
    read_manifest,
    validate_bundle,
    write_bundle,
)
from multireel.bundle.manifest import stream_path
from multireel.bundle.video import profile_info


def write_sample(
    path: Path, edit_rate: Any = 30, scale: int = 1, **kwargs: Any
) -> tuple[Manifest, dict[str, SyntheticSensor], dict[str, Stream]]:
    """Write the sample bundle."""
    sensors = sample_sensors()
    streams = sample_streams(sensors, scale)
    manifest = write_bundle(
        path, streams, sample_sync(), recording_id="drive_0423_a",
        timecode_origin=ORIGIN, edit_rate=edit_rate,
        **kwargs)
    return manifest, sensors, streams


def check_frames(
    path: Path, manifest: Manifest, sensors: dict[str, SyntheticSensor],
    streams: dict[str, Stream],
) -> None:
    """Every decoded frame shows the sample the synchronization chose."""
    expected = expected_samples(manifest, sensors)
    assert any(None in shown for shown in expected.values())
    for stream_id, stream in streams.items():
        alpha = profile_info(stream.profile).alpha
        first = next(stream.frames(np.array([0])))
        height, width = first.shape[:2]
        frames = read_frames(
            stream_path(path, stream_id), width, height, alpha)
        assert [decode_pattern(f) for f in frames] == expected[stream_id]
        if alpha:
            no_data = [f for f, e in zip(frames, expected[stream_id])
                       if e is None]
            assert all((f[..., 3] < 8).all() for f in no_data)


@pytest.mark.parametrize("edit_rate", [10, 15, 24, 25, 30, 50, 60])
def test_every_edit_rate(tmp_path: Path, edit_rate: int) -> None:
    """Bundles at every edit rate validate and show the right samples."""
    manifest, sensors, streams = write_sample(tmp_path, edit_rate)
    assert read_manifest(tmp_path) == manifest
    assert validate_bundle(tmp_path) == []
    check_frames(tmp_path, manifest, sensors, streams)


def test_range_and_keyframe_interval(tmp_path: Path) -> None:
    """An explicit range and keyframe interval are respected."""
    t0 = ORIGIN.timestamp() + 720.5
    manifest, sensors, streams = write_sample(
        tmp_path, start=t0, end=t0 + 1.0, keyframe_interval=7)
    assert manifest.frame_count == 31
    assert str(manifest.start_timecode) == "01:12:00:15"
    assert validate_bundle(tmp_path) == []
    check_frames(tmp_path, manifest, sensors, streams)


def test_nvenc(tmp_path: Path) -> None:
    """HEVC profiles also validate when encoded with NVENC."""
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
        text=True).stdout
    if "hevc_nvenc" not in encoders or shutil.which("nvidia-smi") is None:
        pytest.skip("NVENC is not available.")
    manifest, sensors, streams = write_sample(
        tmp_path, scale=4, options=EncodeOptions(hevc_encoder="hevc_nvenc"))
    assert validate_bundle(tmp_path) == []
    check_frames(tmp_path, manifest, sensors, streams)


def test_failed_render_leaves_no_manifest(tmp_path: Path) -> None:
    """A render that fails part way leaves the directory without a manifest."""
    sensors = sample_sensors()
    streams = sample_streams(sensors)

    def broken(
        indices: Int64[np.ndarray, "K"],
    ) -> Iterator[UInt8[np.ndarray, "H W C"]]:
        yield render_pattern(int(indices[0]), 128, 64)
        raise RuntimeError("renderer crashed")

    streams["thermal"] = Stream(
        sensors["thermal"].metadata.timestamps, broken,
        profile="prores-422-lt")
    with pytest.raises(RuntimeError, match="renderer crashed"):
        write_bundle(tmp_path, streams, sample_sync(),
                     recording_id="drive_0423_a", timecode_origin=ORIGIN)
    assert not (tmp_path / "manifest.yaml").exists()


def test_existing_bundle(tmp_path: Path) -> None:
    """Writing over an existing bundle is refused."""
    write_sample(tmp_path)
    with pytest.raises(FileExistsError):
        write_sample(tmp_path)


def test_decreasing_sync(tmp_path: Path) -> None:
    """A synchronization that goes backwards is rejected."""
    sensors = sample_sensors()

    def backwards(timestamps: dict) -> dict:
        n = 4
        return {k: np.arange(n)[::-1] for k in timestamps}

    with pytest.raises(ValueError, match="decreasing"):
        write_bundle(tmp_path, sample_streams(sensors), backwards,
                     recording_id="drive_0423_a", timecode_origin=ORIGIN)


@pytest.mark.parametrize("bad", [
    {"overlay_of": "missing"}, {"overlay_of": "boxes"}])
def test_bad_overlay(tmp_path: Path, bad: dict) -> None:
    """Overlays must link to a stream that is not itself an overlay."""
    sensors = sample_sensors()
    streams = sample_streams(sensors)
    streams["cam2"] = Stream.from_sensor(
        sensors["cam"], lambda i: render_pattern(i, 128, 64, True),
        profile="prores-4444", **bad)
    with pytest.raises(ValueError):
        write_bundle(tmp_path, streams, sample_sync(),
                     recording_id="drive_0423_a", timecode_origin=ORIGIN)
