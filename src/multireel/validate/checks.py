"""The rules of the bundle spec, and checking bundles against them."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from ..bundle.manifest import Manifest, read_manifest, stream_path
from ..bundle.timing import Timecode
from ..bundle.video import Profile, clip_name, profile_info, reel_name
from .probe import StreamProbe, decode_errors, probe

Rule = Literal[
    # Manifest
    "manifest-present",
    "manifest-yaml",
    "schema-version",
    "manifest-fields",
    "stream-id",
    "timecode-origin",
    "timecode-range",
    "overlay-link",
    # Video format
    "profile",
    "edit-rate",
    "constant-frame-rate",
    "frame-count",
    "start-timecode",
    "no-b-frames",
    "keyframe-alignment",
    "dimensions",
    "square-pixels",
    "progressive",
    "color",
    "tracks",
    "hvc1-tag",
    "timecode-track",
    "colr-tag",
    "reel-name",
    "clip-name",
    "encoder-metadata",
    "decode",
    "overlay-match",
]
"""A rule from the bundle spec, one per MUST.

`no-b-frames` also covers closed GOPs: without reordering, no HEVC frame can
reference a frame before the keyframe that starts its GOP, and ProRes is
intra-only.
"""


@dataclass(frozen=True)
class Violation:
    """One failed check.

    Attributes:
        rule: rule that was broken.
        message: what was found.
        stream: stream ID, or `None` for bundle-level rules.
    """

    rule: Rule
    message: str
    stream: str | None = None


def check_stream(
    probe: StreamProbe, stream_id: str, manifest: Manifest
) -> list[Violation]:
    """Check the properties of one stream file against the spec.

    This covers every per-stream rule except `decode`, which needs the file.

    Returns:
        Every failed check; empty if the stream is valid.
    """
    violations: list[Violation] = []

    def fail(rule: Rule, message: str) -> None:
        violations.append(Violation(rule, message, stream_id))

    entry = manifest.streams[stream_id]
    if entry.profile not in get_args(Profile):
        fail("profile", f"Unknown profile {entry.profile!r}.")
    else:
        info = profile_info(entry.profile)
        fmt = probe.pixel_format
        depth = re.search(r"p(\d*)(?:le|be)?$", fmt)
        found = (
            probe.codec,
            next((c for c in ("420", "422", "444") if c in fmt), None),
            int(depth.group(1)) if depth and depth.group(1) else 8,
            fmt.startswith("yuva"))
        if found != (info.codec, info.chroma, info.bit_depth, info.alpha):
            fail("profile",
                 f"{probe.codec} {fmt} does not match {entry.profile}.")

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
    expected_clip = clip_name(manifest.recording_id, stream_id, entry.label)
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
) -> list[Violation]:
    """Check one stream file against the spec and its manifest.

    The manifest does not need to be written yet, so the writer can check
    each stream before finalizing a bundle.

    Returns:
        Every failed check; empty if the stream is valid.
    """
    return _validate_file(path, stream_id, manifest, ffmpeg, ffprobe)[1]


def _validate_file(
    path: Path, stream_id: str, manifest: Manifest, ffmpeg: str, ffprobe: str
) -> tuple[StreamProbe | None, list[Violation]]:
    """Probe, check and decode one stream file."""
    try:
        properties = probe(path, ffprobe)
    except (subprocess.CalledProcessError, ValueError) as e:
        return None, [
            Violation("decode", f"Cannot read {path}: {e}", stream_id)]
    violations = check_stream(properties, stream_id, manifest)
    if errors := decode_errors(path, ffmpeg):
        violations.append(Violation("decode", "; ".join(errors), stream_id))
    return properties, violations


def _check_manifest_document(data: Any) -> list[Violation]:
    """Check that a parsed manifest has every required field."""
    required = {
        "bundle_id": str, "recording_id": str, "timecode_origin": str,
        "edit_rate": int, "keyframe_interval": int, "start_timecode": str,
        "frame_count": int, "streams": dict}
    violations = [
        Violation("manifest-fields", f"{key} is missing or not a {kind}.")
        for key, kind in required.items()
        if not isinstance(data.get(key), kind)]
    for stream_id, entry in (data.get("streams") or {}).items():
        if not isinstance(entry, dict) or not isinstance(
                entry.get("profile"), str):
            violations.append(Violation(
                "manifest-fields", "profile is missing or not a str.",
                str(stream_id)))
    return violations


def validate_bundle(
    path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> list[Violation]:
    """Check a bundle against the spec.

    Returns:
        Every failed check; empty if the bundle is valid.
    """
    path = Path(path)
    file = path / "manifest.yaml"
    if not file.exists():
        return [Violation("manifest-present", f"{file} does not exist.")]

    try:
        text = file.read_bytes().decode("utf-8")
        documents = list(yaml.safe_load_all(text))
        tokens = list(yaml.scan(text))
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        return [Violation("manifest-yaml", str(e))]
    if len(documents) != 1 or not isinstance(documents[0], dict):
        return [Violation("manifest-yaml", "Not a single YAML mapping.")]
    violations = [
        Violation("manifest-yaml", f"Uses {type(token).__name__}.")
        for token in tokens
        if isinstance(token, (yaml.AnchorToken, yaml.AliasToken,
                              yaml.TagToken))]

    data = documents[0]
    version = data.get("schema_version")
    if not isinstance(version, str) or version.split(".")[0] != "1":
        return [*violations, Violation(
            "schema-version", f"Unknown schema version {version!r}.")]
    violations += _check_manifest_document(data)
    if violations:
        return violations
    try:
        manifest = read_manifest(path)
    except ValueError as e:
        return [Violation("manifest-fields", str(e))]

    origin = manifest.timecode_origin
    if origin.tzinfo is None or origin.microsecond:
        violations.append(Violation(
            "timecode-origin", f"{origin} is not a whole number of seconds."))
    else:
        last = manifest.start_timecode.to_frame(origin) + (
            manifest.frame_count - 1)
        try:
            Timecode.from_frame(last, manifest.edit_rate, origin)
        except ValueError as e:
            violations.append(Violation("timecode-range", str(e)))
    if manifest.edit_rate not in (10, 15, 24, 25, 30, 50, 60):
        violations.append(Violation(
            "edit-rate", f"{manifest.edit_rate} is not an allowed rate."))

    probes: dict[str, StreamProbe] = {}
    for stream_id, entry in manifest.streams.items():
        base = entry.overlay_of
        if base is not None and (
                base not in manifest.streams
                or manifest.streams[base].overlay_of is not None):
            violations.append(Violation(
                "overlay-link", f"{base!r} is not a stream that is not an "
                "overlay.", stream_id))
        file = stream_path(path, stream_id)
        if re.fullmatch(r"[a-z0-9_-]+", stream_id) is None or (
                not file.exists()):
            violations.append(Violation(
                "stream-id", f"Invalid ID, or {file} is missing.", stream_id))
            continue
        properties, found = _validate_file(
            file, stream_id, manifest, ffmpeg, ffprobe)
        violations += found
        if properties is not None:
            probes[stream_id] = properties

    for stream_id, entry in manifest.streams.items():
        overlay, base = probes.get(stream_id), probes.get(
            entry.overlay_of or "")
        if overlay is not None and base is not None and (
                (overlay.width, overlay.height, len(overlay.durations))
                != (base.width, base.height, len(base.durations))):
            violations.append(Violation(
                "overlay-match", "Dimensions or frame count differ from "
                f"{entry.overlay_of!r}.", stream_id))

    return violations
