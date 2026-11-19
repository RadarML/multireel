"""The bundle format as code.

This package holds the data types, constants and pure functions defined by
the bundle spec. Its modules follow the sections of the spec:

- `video`: encoding profiles and container naming rules.
- `timing`: frame grid and timecode. Frame indices are absolute, so frame
    `n` at edit rate `r` sits at `n/r` seconds since the UTC epoch.
- `manifest`: the `manifest.yaml` schema, reading and writing.
"""

from .manifest import (
    AlphaMode,
    Manifest,
    StreamEntry,
    read_manifest,
    stream_path,
    write_manifest,
)
from .timing import Timecode
from .video import Profile, ProfileInfo, clip_name, profile_info, reel_name

__all__ = [
    "AlphaMode",
    "Manifest",
    "Profile",
    "ProfileInfo",
    "StreamEntry",
    "Timecode",
    "clip_name",
    "profile_info",
    "read_manifest",
    "reel_name",
    "stream_path",
    "write_manifest",
]
