"""Multireel: a multimodal rendering framework for video editing.

Multireel renders timestamped sensor data into bundles of video streams that
line up with each other in a video editor.

- [`bundle`][.bundle]: the bundle format as code.
- [`write`][.write]: writing bundles from abstract dataloader sensors.
- [`encode`][.encode]: encoding frames into stream files with `ffmpeg`.
- [`validate`][.validate]: checking bundles against the spec, and the
    `multireel-validate` command.
"""

from .bundle import Manifest, Profile, read_manifest
from .encode import EncodeOptions
from .validate import validate_bundle
from .write import Stream, write_bundle

__all__ = [
    "EncodeOptions",
    "Manifest",
    "Profile",
    "Stream",
    "read_manifest",
    "validate_bundle",
    "write_bundle",
]
