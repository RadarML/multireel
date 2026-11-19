"""Multireel: render sensor data into video streams for a video editor.

Multireel writes bundles: sets of video streams that line up in time.

- [`bundle`][.bundle]: the bundle format.
- [`stream`][.stream]: output streams and how their frames are rendered.
- [`write`][.write]: writing bundles.
- [`render`][.render]: helpers that turn arrays into RGB frames.
- [`progress`][.progress]: progress bars for writing bundles.
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
