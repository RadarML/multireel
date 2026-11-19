"""The per-stream rules of the bundle spec, and checking stream files."""

import re
import subprocess
from pathlib import Path

from ..bundle.manifest import Manifest
from ..bundle.video import clip_name, reel_name
from .probe import StreamProbe, decode_errors, probe

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

    expected_reel = reel_name(manifest.recording_id, stream_id)
    if probe.reel_name != expected_reel:
        fail("reel-name", f"Reel name {probe.reel_name!r}, not "
             f"{expected_reel!r}.")
    expected_clip = clip_name(
        manifest.recording_id, stream_id, manifest.streams[stream_id].label)
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
