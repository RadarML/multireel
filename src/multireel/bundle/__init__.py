"""The bundle format as code.

Data types and pure functions for the parts of the
[bundle spec][?multireel-bundle-spec] that code needs to read or write:

- [`Profile`][.Profile]: the codec configuration of a stream, as in
    [encoding profiles][?encoding-profiles]. Use
    [`Profile.from_name`][.Profile.from_name] for a standard profile, named
    by a [`ProfileName`][.ProfileName].
- [`Timecode`][.Timecode]: non-drop-frame [timecode][?timecode] on the
    [frame grid][?frame-grid]. Frame indices are absolute, so frame `n` at
    edit rate `r` sits at `n/r` seconds since the UTC epoch.
- [`Manifest`][.Manifest] and [`StreamEntry`][.StreamEntry]: the
    [manifest][?manifest], read with [`Manifest.read`][.Manifest.read] and
    written with [`Manifest.write`][.Manifest.write]. `Manifest` also gives
    each stream's file path and its reel and clip [names][?container].
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
