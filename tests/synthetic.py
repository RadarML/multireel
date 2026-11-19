"""Synthetic sensors and renderers with known ground truth, for tests.

Each synthetic sample is its own index. `render_pattern` draws that index
into the frame as a block pattern that survives lossy encoding, and
`decode_pattern` reads it back. After decoding a bundle, every frame can be
checked against the sample that the synchronization chose.
"""

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from abstract_dataloader import abstract
from abstract_dataloader.generic import Nearest
from jaxtyping import Float64, UInt8

from multireel import FRAME_CLOCK, Manifest, Profile, Stream

ORIGIN = datetime(2026, 11, 18, 16, 0, 0, tzinfo=UTC)
"""Timecode origin of the sample bundles."""


BLOCK = 8


@dataclass
class SyntheticMetadata:
    """Timestamps of a `SyntheticSensor`, in float seconds."""

    timestamps: Float64[np.ndarray, "N"]


class SyntheticSensor(abstract.Sensor[int, SyntheticMetadata]):
    """Abstract dataloader sensor whose samples are their own indices.

    Args:
        timestamps: strictly increasing timestamps, in seconds.
    """

    def __init__(self, timestamps: Float64[np.ndarray, "N"]) -> None:
        super().__init__(SyntheticMetadata(timestamps), name="synthetic")

    def __getitem__(self, index: int | np.integer) -> int:
        """Return `index`."""
        return int(index)

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.metadata.timestamps)


def render_pattern(
    index: int, width: int, height: int, alpha: bool = False
) -> UInt8[np.ndarray, "H W C"]:
    """Draw a sample index into a frame as a block pattern.

    The first block is always white, and the following 24 blocks hold the
    bits of `index`. No-data frames are black, so their first block is
    black.
    """
    cols = width // BLOCK
    bits = np.zeros((height // BLOCK) * cols, dtype=bool)
    bits[0] = True
    bits[1:25] = [(index >> i) & 1 for i in range(24)]
    blocks = bits.reshape(-1, cols).astype(np.uint8) * 255
    gray = np.kron(blocks, np.ones((BLOCK, BLOCK), dtype=np.uint8))
    channels = [gray, gray, gray]
    if alpha:
        channels.append(np.full_like(gray, 255))
    return np.stack(channels, axis=-1)


def decode_pattern(frame: UInt8[np.ndarray, "H W C"]) -> int | None:
    """Read back a sample index drawn by `render_pattern`.

    Returns:
        The sample index, or `None` for a no-data frame.
    """
    gray = frame[..., :3].mean(axis=-1)
    centers = gray[BLOCK // 2::BLOCK, BLOCK // 2::BLOCK].reshape(-1) > 128
    if not centers[0]:
        return None
    return sum(int(bit) << i for i, bit in enumerate(centers[1:25]))


def read_frames(
    path: Path, width: int, height: int, alpha: bool = False
) -> UInt8[np.ndarray, "F H W C"]:
    """Decode every frame of a stream file to RGB, or RGBA."""
    channels = 4 if alpha else 3
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgba" if alpha else "rgb24", "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.uint8).reshape(
        -1, height, width, channels)


def sample_sensors() -> dict[str, SyntheticSensor]:
    """Sensors at several rates, starting 12 minutes after `ORIGIN`.

    `lidar` has a gap from 1.0 s to 1.6 s.
    """
    t0 = ORIGIN.timestamp() + 720.0123

    def at(rate: float, duration: float) -> Float64[np.ndarray, "N"]:
        return t0 + np.arange(int(duration * rate)) / rate

    lidar = at(10, 2.5)
    lidar = lidar[(lidar - t0 < 1.0) | (lidar - t0 >= 1.6)]
    return {
        "cam": SyntheticSensor(at(30, 2.5)),
        "radar": SyntheticSensor(at(20, 2.5)),
        "lidar": SyntheticSensor(lidar),
        "thermal": SyntheticSensor(at(15, 2.5)),
        "boxes": SyntheticSensor(at(10, 2.5)),
    }


def sample_streams(
    sensors: dict[str, SyntheticSensor], scale: int = 1
) -> dict[str, Stream]:
    """Streams covering every profile, with an overlay of `cam`.

    Frames are `128 * scale` by `64 * scale`, except `radar`, which is half
    that size. `thermal` is read sequentially.
    """

    def pattern(
        width: int, height: int, alpha: bool = False
    ) -> Callable[[int], UInt8[np.ndarray, "H W C"]]:
        return lambda index: render_pattern(
            index, width * scale, height * scale, alpha)

    return {
        "cam": Stream.from_sensor(
            sensors["cam"], pattern(128, 64), label="Front camera"),
        "radar": Stream.from_sensor(
            sensors["radar"], pattern(64, 32),
            profile=Profile.from_name("hevc-444")),
        "lidar": Stream.from_sensor(
            sensors["lidar"], pattern(128, 64),
            profile=Profile.from_name("prores-422-hq")),
        "thermal": Stream.from_sensor(
            sensors["thermal"], pattern(128, 64),
            profile=Profile.from_name("prores-422-lt"), sequential=True),
        "boxes": Stream.from_sensor(
            sensors["boxes"], pattern(128, 64, alpha=True),
            profile=Profile.from_name("prores-4444"), overlay_of="cam"),
    }


def sample_sync() -> Nearest:
    """Nearest-sample synchronization to the frame clock."""
    return Nearest(FRAME_CLOCK, tol=0.06)


def expected_samples(
    manifest: Manifest, sensors: dict[str, SyntheticSensor]
) -> dict[str, list[int | None]]:
    """Recompute the sample each frame should show, or `None` for no data."""
    first = manifest.start_timecode.to_frame(manifest.timecode_origin)
    clock = (first + np.arange(manifest.frame_count)) / manifest.edit_rate
    synced = sample_sync()({
        FRAME_CLOCK: clock,
        **{k: s.metadata.timestamps for k, s in sensors.items()}})
    expected: dict[str, list[int | None]] = {}
    for stream_id in sensors:
        shown: list[int | None] = [None] * manifest.frame_count
        for frame, sample in zip(synced[FRAME_CLOCK], synced[stream_id]):
            shown[int(frame)] = int(sample)
        expected[stream_id] = shown
    return expected
