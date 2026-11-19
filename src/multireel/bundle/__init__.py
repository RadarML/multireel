"""The bundle format, as defined by the [bundle spec][?multireel-bundle-spec].

- [`Profile`][.Profile]: how a stream is encoded. See
    [encoding profiles][?encoding-profiles].
- [`Timecode`][.Timecode]: the [timecode][?timecode] of a frame.
- [`Manifest`][.Manifest] and [`StreamEntry`][.StreamEntry]: the
    [manifest][?manifest]. `Manifest` also gives each stream's file path and
    its reel and clip [names][?container].
"""

from .manifest import Manifest, StreamEntry
from .timing import Timecode
from .video import Profile, ProfileName

__all__ = [
    "Manifest",
    "Profile",
    "ProfileName",
    "StreamEntry",
    "Timecode",
]
