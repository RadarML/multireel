"""The bundle manifest, `manifest.yaml`."""

import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import yaml

from ._profile import Profile
from ._timecode import Timecode


@dataclass(frozen=True)
class StreamEntry:
    """Manifest fields of one stream.

    Attributes:
        profile: encoding profile.
        label: display name.
        overlay_of: base stream ID, for overlays.
        alpha_mode: how alpha is stored, for overlays.
        info: any extra information.
    """

    profile: Profile
    label: str | None = None
    overlay_of: str | None = None
    alpha_mode: Literal["straight", "premultiplied"] = "straight"
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Manifest:
    """Contents of `manifest.yaml`.

    Stream IDs may only use lowercase letters, digits, `_` and `-`.

    Attributes:
        bundle_id: unique ID of this bundle.
        recording_id: ID of the source recording.
        timecode_origin: UTC time shown as timecode `01:00:00:00`.
        edit_rate: frames per second.
        keyframe_interval: number of frames between keyframes.
        start_timecode: timecode of the first frame.
        frame_count: number of frames in every stream.
        streams: stream entries, by stream ID.
        provenance: any extra information.
        schema_version: schema version, as `"major.minor"`.

    Raises:
        ValueError: if a stream ID, overlay link or edit rate is invalid.
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

    def __post_init__(self) -> None:
        """Check the stream IDs, overlay links and edit rate."""
        if self.edit_rate not in (10, 15, 24, 25, 30, 50, 60):
            raise ValueError(f"Edit rate {self.edit_rate} is not allowed.")
        for stream_id, entry in self.streams.items():
            if re.fullmatch(r"[a-z0-9_-]+", stream_id) is None:
                raise ValueError(f"Invalid stream ID: {stream_id!r}")
            base = entry.overlay_of
            if base is not None and (
                    base not in self.streams
                    or self.streams[base].overlay_of is not None):
                raise ValueError(
                    f"Stream {stream_id!r} is an overlay of {base!r}, which "
                    f"is not a stream or is itself an overlay.")

    @staticmethod
    def stream_path(bundle: Path, stream_id: str) -> Path:
        """Return the path of a stream file, `streams/<stream_id>.mov`."""
        return bundle / "streams" / f"{stream_id}.mov"

    def reel_name(self, stream_id: str) -> str:
        """Return the reel name of a stream, `<recording_id>.<stream_id>`."""
        return f"{self.recording_id}.{stream_id}"

    def clip_name(self, stream_id: str) -> str:
        """Return the clip name of a stream, `<recording_id> <label>`.

        Streams without a label use their ID instead.
        """
        label = self.streams[stream_id].label
        return f"{self.recording_id} {label or stream_id}"

    @classmethod
    def read(cls, bundle: Path) -> "Manifest":
        """Read the manifest of a bundle.

        Unknown fields are ignored.

        Raises:
            ValueError: if the schema version is unsupported, or a field is
                missing or invalid.
        """
        with open(bundle / "manifest.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        version = str(data.get("schema_version", ""))
        if version.split(".")[0] != "1":
            raise ValueError(f"Unknown manifest schema version: {version!r}")

        try:
            edit_rate = int(data["edit_rate"])
            return cls(
                bundle_id=UUID(data["bundle_id"]),
                recording_id=data["recording_id"],
                timecode_origin=datetime.fromisoformat(
                    data["timecode_origin"]),
                edit_rate=edit_rate,
                keyframe_interval=int(data["keyframe_interval"]),
                start_timecode=Timecode.parse(
                    data["start_timecode"], edit_rate),
                frame_count=int(data["frame_count"]),
                streams={
                    stream_id: StreamEntry(
                        profile=Profile.from_manifest(entry["profile"]),
                        label=entry.get("label"),
                        overlay_of=entry.get("overlay_of"),
                        alpha_mode=entry.get("alpha_mode", "straight"),
                        info=entry.get("info", {}))
                    for stream_id, entry in data["streams"].items()},
                provenance=data.get("provenance"),
                schema_version=version)
        except KeyError as e:
            raise ValueError(f"Manifest is missing field {e}") from e

    def write(self, bundle: Path) -> None:
        """Write this manifest to a bundle.

        The file is written to a temporary file first, then renamed, so it is
        never left half-written.
        """
        streams: dict[str, Any] = {}
        for stream_id, entry in self.streams.items():
            fields: dict[str, Any] = {}
            if entry.label is not None:
                fields["label"] = entry.label
            fields["profile"] = entry.profile.to_manifest()
            if entry.overlay_of is not None:
                fields["overlay_of"] = entry.overlay_of
            if entry.alpha_mode != "straight":
                fields["alpha_mode"] = entry.alpha_mode
            if entry.info:
                fields["info"] = dict(entry.info)
            streams[stream_id] = fields

        origin = self.timecode_origin.astimezone(UTC)
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "bundle_id": str(self.bundle_id),
            "recording_id": self.recording_id,
            "timecode_origin": origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "edit_rate": self.edit_rate,
            "keyframe_interval": self.keyframe_interval,
            "start_timecode": str(self.start_timecode),
            "frame_count": self.frame_count,
        }
        if self.provenance is not None:
            data["provenance"] = self.provenance
        data["streams"] = streams

        # PyYAML writes an anchor and alias for any object that appears
        # twice, such as a dict shared by `info` and `provenance` values.
        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data: Any) -> bool:
                return True

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=bundle, suffix=".tmp", delete=False
        ) as f:
            yaml.dump(
                data, f, Dumper=NoAliasDumper, sort_keys=False,
                allow_unicode=True)
        os.replace(f.name, bundle / "manifest.yaml")
