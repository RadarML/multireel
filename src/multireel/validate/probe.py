"""Reading stream file properties with `ffprobe`.

The checks only look at the [`StreamProbe`][.] produced here, so they can be
tested with crafted probes instead of crafted files.
"""

import json
import struct
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Bool, Int64


@dataclass(frozen=True)
class Colr:
    """Contents of a QuickTime `colr` atom.

    Attributes:
        type: `nclc` or `nclx`.
        primaries: color primaries code; `1` is BT.709.
        transfer: transfer function code; `1` is BT.709.
        matrix: matrix code; `1` is BT.709.
        full_range: the full-range flag of an `nclx` atom, or `None` for
            `nclc`, which has no range flag.
    """

    type: str
    primaries: int
    transfer: int
    matrix: int
    full_range: bool | None


@dataclass(frozen=True)
class StreamProbe:
    """Properties of one stream file.

    Attributes:
        tracks: `(type, sample entry)` of each track, such as
            `[("video", "hvc1"), ("data", "tmcd")]`.
        codec: video codec name, such as `hevc` or `prores`.
        pixel_format: decoded pixel format, such as `yuv420p10le`.
        width: frame width, in pixels.
        height: frame height, in pixels.
        sample_aspect_ratio: pixel aspect ratio, such as `1:1`, or `None`
            if unset.
        field_order: `progressive`, an interlaced field order, or `None` if
            unset.
        rotation: display matrix rotation, in degrees.
        frame_rate: video frame rate, such as `30/1`.
        time_base: video track time base, such as `1/15360`.
        color: color fields of the first decoded frame: `color_range`,
            `color_primaries`, `color_transfer` and `color_space`.
        colr: the `colr` atom, or `None` if absent.
        timecode: start timecode of the `tmcd` track, or `None`.
        reel_name: reel name of the `tmcd` track, or `None`.
        tags: container metadata tags.
        durations: duration of each frame, in the time base.
        keyframes: whether each frame is a keyframe.
        has_reordering: whether decode order differs from presentation
            order, or the first frame is not presented at time zero, which
            needs composition offsets or an edit list.
    """

    tracks: list[tuple[str, str]]
    codec: str
    pixel_format: str
    width: int
    height: int
    sample_aspect_ratio: str | None
    field_order: str | None
    rotation: float
    frame_rate: str
    time_base: str
    color: Mapping[str, str]
    colr: Colr | None
    timecode: str | None
    reel_name: str | None
    tags: Mapping[str, str]
    durations: Int64[np.ndarray, "F"]
    keyframes: Bool[np.ndarray, "F"]
    has_reordering: bool


def _ffprobe(ffprobe: str, path: Path, *args: str) -> dict[str, Any]:
    """Run `ffprobe` with JSON output."""
    out = subprocess.run(
        [ffprobe, "-v", "error", "-of", "json", *args, str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _read_colr(path: Path) -> Colr | None:
    """Find and parse the `colr` atom in the `moov` box of a MOV file."""
    moov = b""
    with open(path, "rb") as f:
        while len(header := f.read(8)) == 8:
            size, kind = struct.unpack(">I4s", header)
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0] - 8
            if kind == b"moov":
                moov = f.read() if size == 0 else f.read(size - 8)
                break
            if size == 0:
                break
            f.seek(size - 8, 1)

    i = moov.find(b"colr")
    if i < 0:
        return None
    kind = moov[i + 4:i + 8].decode(errors="replace")
    primaries, transfer, matrix = struct.unpack(">HHH", moov[i + 8:i + 14])
    full_range = bool(moov[i + 14] >> 7) if kind == "nclx" else None
    return Colr(kind, primaries, transfer, matrix, full_range)


def probe(path: Path, ffprobe: str = "ffprobe") -> StreamProbe:
    """Read the properties of a stream file.

    Raises:
        subprocess.CalledProcessError: if `ffprobe` cannot read the file.
        ValueError: if the file has no video track.
    """
    info = _ffprobe(ffprobe, path, "-show_streams", "-show_format")
    streams = info["streams"]
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"{path} has no video track.")
    tmcd = next(
        (s for s in streams if s.get("codec_tag_string") == "tmcd"), {})

    packets = _ffprobe(
        ffprobe, path, "-select_streams", "v:0",
        "-show_entries", "packet=pts,dts,duration,flags")["packets"]
    pts = np.array([p["pts"] for p in packets], dtype=np.int64)
    dts = np.array([p["dts"] for p in packets], dtype=np.int64)

    frames = _ffprobe(
        ffprobe, path, "-select_streams", "v:0", "-read_intervals", "%+#1",
        "-show_entries",
        "frame=color_range,color_primaries,color_transfer,color_space",
    )["frames"]

    rotation = next(
        (float(d["rotation"]) for d in video.get("side_data_list", [])
         if "rotation" in d), 0.0)

    return StreamProbe(
        tracks=[(s["codec_type"], s.get("codec_tag_string", ""))
                for s in streams],
        codec=video["codec_name"],
        pixel_format=video["pix_fmt"],
        width=video["width"],
        height=video["height"],
        sample_aspect_ratio=video.get("sample_aspect_ratio"),
        field_order=video.get("field_order"),
        rotation=rotation,
        frame_rate=video["r_frame_rate"],
        time_base=video["time_base"],
        color=frames[0] if frames else {},
        colr=_read_colr(path),
        timecode=tmcd.get("tags", {}).get("timecode"),
        reel_name=tmcd.get("tags", {}).get("reel_name"),
        tags=info["format"].get("tags", {}),
        durations=np.array(
            [p["duration"] for p in packets], dtype=np.int64),
        keyframes=np.array(["K" in p["flags"] for p in packets], dtype=bool),
        has_reordering=bool(
            len(pts) > 0 and (
                pts[0] != 0 or np.any(pts != dts)
                or np.any(np.diff(pts) <= 0))),
    )


def decode_errors(path: Path, ffmpeg: str = "ffmpeg") -> list[str]:
    """Decode every frame in software and return any decoder errors."""
    out = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-f", "null", "-"],
        capture_output=True, text=True)
    return [line for line in out.stderr.splitlines() if line.strip()]
