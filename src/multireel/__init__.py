"""Multireel: a multimodal rendering framework for video editing.

Multireel renders timestamped sensor data into bundles of video streams that
line up with each other in a video editor.

- [`bundle`][.bundle]: the bundle format as code.
- [`stream`][.stream]: output streams, and how their frames are rendered.
- [`write`][.write]: writing bundles from streams: planning, encoding with
    `ffmpeg`, and checking stream files against the spec.
- [`render`][.render]: turning arrays into RGB frames, with colormaps.
- [`progress`][.progress]: rich progress bars for writing bundles.
"""

from .bundle import Manifest, Profile
from .stream import Stream
from .write import FRAME_CLOCK, BundleWriter, EncodeOptions

__all__ = [
    "FRAME_CLOCK",
    "BundleWriter",
    "EncodeOptions",
    "Manifest",
    "Profile",
    "Stream",
]
