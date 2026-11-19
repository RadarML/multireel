"""End-to-end tests: write bundles, then decode and check every frame."""

import os
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
    Profile,
    Stream,
    read_manifest,
    write_bundle,
)
from multireel.bundle.manifest import stream_path
from multireel.bundle.video import profile_info
from multireel.validate import checks, validate_stream


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
    """Bundles at every edit rate show the right samples."""
    manifest, sensors, streams = write_sample(tmp_path, edit_rate)
    assert read_manifest(tmp_path) == manifest
    check_frames(tmp_path, manifest, sensors, streams)


def test_range_and_keyframe_interval(tmp_path: Path) -> None:
    """An explicit range and keyframe interval are respected."""
    t0 = ORIGIN.timestamp() + 720.5
    manifest, sensors, streams = write_sample(
        tmp_path, start=t0, end=t0 + 1.0, keyframe_interval=7)
    assert manifest.frame_count == 31
    assert str(manifest.start_timecode) == "01:12:00:15"
    check_frames(tmp_path, manifest, sensors, streams)


def test_nvenc(tmp_path: Path) -> None:
    """HEVC profiles are also valid when encoded with NVENC."""
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
        text=True).stdout
    if "hevc_nvenc" not in encoders or shutil.which("nvidia-smi") is None:
        pytest.skip("NVENC is not available.")
    manifest, sensors, streams = write_sample(
        tmp_path, scale=4, options=EncodeOptions(hevc_encoder="hevc_nvenc"))
    check_frames(tmp_path, manifest, sensors, streams)


def test_processes(tmp_path: Path) -> None:
    """Bundles written with one process per stream match threaded ones."""
    manifest, sensors, streams = write_sample(tmp_path, processes=True)
    assert read_manifest(tmp_path) == manifest
    check_frames(tmp_path, manifest, sensors, streams)


@pytest.mark.parametrize("processes", [False, True])
def test_progress(tmp_path: Path, processes: bool) -> None:
    """Progress counts every frame of every stream, without going back."""
    calls: list[tuple[dict[str, int], int]] = []
    manifest, _, streams = write_sample(
        tmp_path, processes=processes,
        progress=lambda done, total: calls.append((dict(done), total)))
    assert all(total == manifest.frame_count for _, total in calls)
    assert calls[-1][0] == dict.fromkeys(streams, manifest.frame_count)
    for before, after in zip(calls, calls[1:]):
        assert all(after[0][k] >= before[0][k] for k in streams)


def test_process_workers(tmp_path: Path) -> None:
    """Fewer processes than streams still write every stream."""
    manifest, sensors, streams = write_sample(
        tmp_path, processes=True, workers=2)
    check_frames(tmp_path, manifest, sensors, streams)


@pytest.mark.parametrize("profile", ["hevc-main10-420", "prores-4444"])
def test_bgr_channel_order(tmp_path: Path, profile: Profile) -> None:
    """BGR(A) frames are encoded with the same colors as RGB(A) frames."""
    rgb = np.array([200, 40, 20], dtype=np.uint8)
    alpha = profile_info(profile).alpha

    def solid(color: UInt8[np.ndarray, "C"]) -> UInt8[np.ndarray, "H W C"]:
        pixel = np.concatenate([color, [255]]) if alpha else color
        return np.broadcast_to(pixel.astype(np.uint8), (64, 128, len(pixel)))

    timestamps = sample_sensors()["cam"].metadata.timestamps
    streams = {
        "rgb": Stream.from_sensor(
            SyntheticSensor(timestamps), lambda i: solid(rgb),
            profile=profile),
        "bgr": Stream.from_sensor(
            SyntheticSensor(timestamps), lambda i: solid(rgb[::-1]),
            profile=profile, channel_order="bgr"),
    }
    write_bundle(tmp_path, streams, sample_sync(),
                 recording_id="drive_0423_a", timecode_origin=ORIGIN)
    decoded = {}
    for k in streams:
        frames = read_frames(stream_path(tmp_path, k), 128, 64, alpha)
        decoded[k] = frames[len(frames) // 2]
    assert np.array_equal(decoded["rgb"], decoded["bgr"])
    assert np.abs(decoded["rgb"][32, 64, :3].astype(int) - rgb).max() <= 4


def test_verify_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `decode`, stream validation does not decode frames."""
    manifest, _, _ = write_sample(tmp_path)
    monkeypatch.setattr(checks, "decode_errors", lambda *a, **k: ["broken"])
    file = stream_path(tmp_path, "cam")
    assert [v.rule for v in validate_stream(file, "cam", manifest)] == [
        "decode"]
    assert validate_stream(file, "cam", manifest, decode=False) == []


@pytest.mark.parametrize("processes", [False, True])
def test_failed_render_leaves_no_manifest(
    tmp_path: Path, processes: bool
) -> None:
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
                     recording_id="drive_0423_a", timecode_origin=ORIGIN,
                     processes=processes)
    assert not (tmp_path / "manifest.yaml").exists()


def test_process_exits_without_result(tmp_path: Path) -> None:
    """A stream process that dies without reporting raises in the parent."""
    sensors = sample_sensors()
    streams = sample_streams(sensors)

    def dies(
        indices: Int64[np.ndarray, "K"],
    ) -> Iterator[UInt8[np.ndarray, "H W C"]]:
        os._exit(3)
        yield render_pattern(0, 128, 64)

    streams["thermal"] = Stream(
        sensors["thermal"].metadata.timestamps, dies,
        profile="prores-422-lt")
    with pytest.raises(RuntimeError, match="'thermal' exited with code 3"):
        write_bundle(tmp_path, streams, sample_sync(),
                     recording_id="drive_0423_a", timecode_origin=ORIGIN,
                     processes=True)
    assert not (tmp_path / "manifest.yaml").exists()


def test_existing_bundle(tmp_path: Path) -> None:
    """Writing over an existing bundle is refused."""
    write_sample(tmp_path)
    with pytest.raises(FileExistsError):
        write_sample(tmp_path)


@pytest.mark.parametrize("processes", [False, True])
def test_decreasing_sync(tmp_path: Path, processes: bool) -> None:
    """A synchronization that goes backwards is rejected."""
    sensors = sample_sensors()

    def backwards(timestamps: dict) -> dict:
        n = 4
        return {k: np.arange(n)[::-1] for k in timestamps}

    with pytest.raises(ValueError, match="decreasing"):
        write_bundle(tmp_path, sample_streams(sensors), backwards,
                     recording_id="drive_0423_a", timecode_origin=ORIGIN,
                     processes=processes)


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
