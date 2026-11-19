"""The bundle format as code.

This package holds the data types, constants and pure functions defined by
the bundle spec. Its modules follow the sections of the spec:

- `video`: encoding profiles and container naming rules.
- `timing`: frame grid and timecode. Frame indices are absolute, so frame
    `n` at edit rate `r` sits at `n/r` seconds since the UTC epoch.
- `manifest`: the `manifest.yaml` schema, reading and writing.
"""

from .manifest import (
    Manifest,
    StreamEntry,
    stream_path,
)
from .timing import Timecode
from .video import Profile, ProfileName, clip_name, reel_name

__all__ = [
    "Manifest",
    "Profile",
    "ProfileName",
    "StreamEntry",
    "Timecode",
    "clip_name",
    "reel_name",
    "stream_path",
]
