"""Manifest: the `manifest.yaml` schema, reading and writing."""

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import yaml

from .timing import Timecode
from .video import Profile

AlphaMode = Literal["straight", "premultiplied"]
"""How an overlay's alpha is stored."""


def stream_path(bundle: Path, stream_id: str) -> Path:
    """Return the path of a stream file, `streams/<stream_id>.mov`."""
    return bundle / "streams" / f"{stream_id}.mov"


@dataclass(frozen=True)
class StreamEntry:
    """Manifest fields of one stream.

    Attributes:
        profile: encoding profile.
        label: human-readable name for display.
        overlay_of: base stream ID, for overlays only.
        alpha_mode: how alpha is stored, for overlays only.
        info: free-form mapping, stored unchanged.
    """

    profile: Profile
    label: str | None = None
    overlay_of: str | None = None
    alpha_mode: AlphaMode = "straight"
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Manifest:
    """Contents of `manifest.yaml`.

    Attributes:
        bundle_id: unique ID of this bundle.
        recording_id: ID of the source recording.
        timecode_origin: absolute UTC time shown as timecode `01:00:00:00`.
        edit_rate: frames per second.
        keyframe_interval: keyframe interval `N`, in frames.
        start_timecode: timecode of the first frame.
        frame_count: number of frames in every stream.
        streams: stream entries, keyed by stream ID. Stream IDs use only
            lowercase letters, digits, `_` and `-`.
        provenance: free-form value, stored unchanged.
        schema_version: `"major.minor"` schema version.
    """

    bundle_id: UUID
    recording_id: str
    timecode_origin: datetime
    edit_rate: int
    keyframe_interval: int
    start_timecode: Timecode
    frame_count: int
    streams: Mapping[str, StreamEntry]
    provenance: Any = None
    schema_version: str = "1.0"


def read_manifest(bundle: Path) -> Manifest:
    """Read the manifest of a bundle.

    Fields that this schema version does not know are ignored.

    Raises:
        ValueError: if the major schema version is unknown, or a required
            field is missing.
    """
    with open(bundle / "manifest.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    version = str(data.get("schema_version", ""))
    if version.split(".")[0] != "1":
        raise ValueError(f"Unknown manifest schema version: {version!r}")

    try:
        edit_rate = int(data["edit_rate"])
        return Manifest(
            bundle_id=UUID(data["bundle_id"]),
            recording_id=data["recording_id"],
            timecode_origin=datetime.fromisoformat(data["timecode_origin"]),
            edit_rate=edit_rate,
            keyframe_interval=int(data["keyframe_interval"]),
            start_timecode=Timecode.parse(data["start_timecode"], edit_rate),
            frame_count=int(data["frame_count"]),
            streams={
                stream_id: StreamEntry(
                    profile=entry["profile"],
                    label=entry.get("label"),
                    overlay_of=entry.get("overlay_of"),
                    alpha_mode=entry.get("alpha_mode", "straight"),
                    info=entry.get("info", {}))
                for stream_id, entry in data["streams"].items()},
            provenance=data.get("provenance"),
            schema_version=version)
    except KeyError as e:
        raise ValueError(f"Manifest is missing field {e}") from e


class _Dumper(yaml.SafeDumper):
    """YAML dumper that never writes anchors or aliases."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def write_manifest(bundle: Path, manifest: Manifest) -> None:
    """Write the manifest of a bundle atomically.

    The manifest is written to a temporary file in `bundle`, then renamed to
    `manifest.yaml`. Strings that YAML could read as another type are
    quoted.
    """
    streams: dict[str, Any] = {}
    for stream_id, entry in manifest.streams.items():
        fields: dict[str, Any] = {}
        if entry.label is not None:
            fields["label"] = entry.label
        fields["profile"] = entry.profile
        if entry.overlay_of is not None:
            fields["overlay_of"] = entry.overlay_of
        if entry.alpha_mode != "straight":
            fields["alpha_mode"] = entry.alpha_mode
        if entry.info:
            fields["info"] = dict(entry.info)
        streams[stream_id] = fields

    origin = manifest.timecode_origin.astimezone(UTC)
    data: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "bundle_id": str(manifest.bundle_id),
        "recording_id": manifest.recording_id,
        "timecode_origin": origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "edit_rate": manifest.edit_rate,
        "keyframe_interval": manifest.keyframe_interval,
        "start_timecode": str(manifest.start_timecode),
        "frame_count": manifest.frame_count,
    }
    if manifest.provenance is not None:
        data["provenance"] = manifest.provenance
    data["streams"] = streams

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=bundle, suffix=".tmp", delete=False
    ) as f:
        yaml.dump(
            data, f, Dumper=_Dumper, sort_keys=False, allow_unicode=True)
    os.replace(f.name, bundle / "manifest.yaml")
