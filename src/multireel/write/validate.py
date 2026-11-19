"""Checking stream files against the bundle spec.

[`validate_stream`][.validate_stream] reads a stream file's properties with
`ffprobe` into a [`StreamProbe`][.StreamProbe], checks them with
[`check_stream`][.check_stream], and decodes every frame with
[`decode_errors`][.decode_errors]. Each failed check is reported as a
`"<rule>: <message>"` string. The checks only look at the probe, so they can
be tested with crafted probes instead of crafted files.
"""

import json
import re
import struct
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Bool, Int64

from multireel.bundle import Manifest


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


PRORES_TAGS = {
    "proxy": "apco", "lt": "apcs", "standard": "apcn", "hq": "apch",
    "4444": "ap4h", "4444xq": "ap4x"}
"""QuickTime sample entry of each ProRes level."""


def check_stream(
    probe: StreamProbe, stream_id: str, manifest: Manifest
) -> list[str]:
    """Check the properties of one stream file against the spec.

    This covers every per-stream MUST except `decode`, which needs the file.
    `no-b-frames` also covers closed GOPs: without reordering, no HEVC frame
    can reference a frame before the keyframe that starts its GOP, and
    ProRes is intra-only.

    Returns:
        One `"<rule>: <message>"` string per failed check, such as
            `"hvc1-tag: HEVC sample entry is ['hev1']."`; empty if the
            stream is valid.
    """
    violations: list[str] = []

    def fail(rule: str, message: str) -> None:
        violations.append(f"{rule}: {message}")

    profile = manifest.streams[stream_id].profile
    fmt = probe.pixel_format
    depth = re.search(r"p(\d*)(?:le|be)?$", fmt)
    found = (
        probe.codec,
        next((c for c in ("420", "422", "444") if c in fmt), None),
        int(depth.group(1)) if depth and depth.group(1) else 8,
        fmt.startswith("yuva"))
    if found != (
            profile.codec, profile.chroma, profile.bit_depth, profile.alpha):
        fail("profile", f"{probe.codec} {fmt} does not match {profile}.")
    if profile.codec == "prores" and profile.level is not None:
        tag = next((t for k, t in probe.tracks if k == "video"), None)
        if tag != PRORES_TAGS.get(profile.level):
            fail("profile",
                 f"ProRes sample entry {tag} is not level {profile.level}.")

    rate = manifest.edit_rate
    if probe.frame_rate != f"{rate}/1":
        fail("edit-rate", f"Frame rate is {probe.frame_rate}, not {rate}.")
    num, den = map(int, probe.time_base.split("/"))
    if den % (num * rate) or (probe.durations != den // (num * rate)).any():
        fail("constant-frame-rate", "Frame durations are not 1/edit_rate.")
    if len(probe.durations) != manifest.frame_count:
        fail("frame-count",
             f"{len(probe.durations)} frames, not {manifest.frame_count}.")
    start = manifest.start_timecode
    found_start = [int(x) for x in re.split("[:;]", probe.timecode or "")
                   if x.isdigit()]
    if found_start != [start.hours, start.minutes, start.seconds,
                       start.frames]:
        fail("start-timecode", f"Starts at {probe.timecode}, not {start}.")
    if probe.has_reordering:
        fail("no-b-frames",
             "Frames are reordered, or the first frame is not at time 0.")
    if not probe.keyframes[::manifest.keyframe_interval].all():
        fail("keyframe-alignment",
             f"Not every frame at a multiple of "
             f"{manifest.keyframe_interval} is a keyframe.")

    w, h = probe.width, probe.height
    if w % 2 or h % 2 or w > 4096 or h > 4096:
        fail("dimensions", f"{w}x{h} is odd or larger than 4096.")
    if probe.sample_aspect_ratio not in (None, "1:1", "0:1"):
        fail("square-pixels",
             f"Pixel aspect ratio is {probe.sample_aspect_ratio}.")
    if probe.field_order not in (None, "progressive") or probe.rotation:
        fail("progressive",
             f"Field order {probe.field_order}, rotation {probe.rotation}.")
    expected = {
        "color_range": "tv", "color_primaries": "bt709",
        "color_transfer": "bt709", "color_space": "bt709"}
    if any(probe.color.get(k) != v for k, v in expected.items()):
        fail("color", f"Bitstream color is {dict(probe.color)}.")

    video = [tag for kind, tag in probe.tracks if kind == "video"]
    timecode = [tag for _, tag in probe.tracks if tag == "tmcd"]
    if len(video) != 1 or len(timecode) != 1 or len(probe.tracks) != 2:
        fail("tracks", f"Tracks are {probe.tracks}.")
    if probe.codec == "hevc" and video != ["hvc1"]:
        fail("hvc1-tag", f"HEVC sample entry is {video}.")
    if probe.timecode is None or ";" in probe.timecode:
        fail("timecode-track",
             f"Timecode {probe.timecode} is missing or drop-frame.")
    colr = probe.colr
    if colr is None or colr.type not in ("nclc", "nclx") or (
            colr.primaries, colr.transfer, colr.matrix) != (1, 1, 1) or (
            colr.full_range):
        fail("colr-tag", f"colr atom is {colr}.")

    expected_reel = manifest.reel_name(stream_id)
    if probe.reel_name != expected_reel:
        fail("reel-name", f"Reel name {probe.reel_name!r}, not "
             f"{expected_reel!r}.")
    expected_clip = manifest.clip_name(stream_id)
    if probe.tags.get("title") != expected_clip:
        fail("clip-name", f"Clip name {probe.tags.get('title')!r}, not "
             f"{expected_clip!r}.")
    for key in ("encoder", "encoder_version", "encoder_settings"):
        if not probe.tags.get(f"multireel_{key}"):
            fail("encoder-metadata", f"multireel_{key} is missing.")

    return violations


def validate_stream(
    path: Path,
    stream_id: str,
    manifest: Manifest,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    decode: bool = True,
) -> list[str]:
    """Check one stream file against the spec and its manifest.

    The manifest does not need to be written yet, so the writer can check
    each stream before finalizing a bundle.

    Args:
        path: stream file.
        stream_id: ID of the stream in `manifest`.
        manifest: manifest of the bundle the stream belongs to.
        ffmpeg: `ffmpeg` executable.
        ffprobe: `ffprobe` executable.
        decode: also decode every frame to check the `decode` rule. This
            takes most of the time; without it, only the container, stream
            and packet properties are checked.

    Returns:
        One `"<rule>: <message>"` string per failed check, as in
            [`check_stream`][^.]; empty if the stream is valid.
    """
    try:
        properties = probe(path, ffprobe)
    except (subprocess.CalledProcessError, ValueError) as e:
        return [f"decode: Cannot read {path}: {e}"]
    violations = check_stream(properties, stream_id, manifest)
    if decode and (errors := decode_errors(path, ffmpeg)):
        violations.append(f"decode: {'; '.join(errors)}")
    return violations
