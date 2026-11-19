"""Multireel: render sensor data into video streams for a video editor.

Multireel writes bundles: sets of video streams that line up in time, as
defined by the [bundle spec][?multireel-bundle-spec].

```python
from abstract_dataloader.generic import Nearest
from multireel import BundleWriter, Profile, Stream

streams = {
    "front_cam": Stream.from_sensor(trace["camera"], render_camera),
    "front_boxes": Stream.from_sensor(
        trace["detections"], render_boxes,
        profile=Profile.from_name("prores-4444"), overlay_of="front_cam"),
}
writer = BundleWriter(
    "out/drive_0423_a", streams,
    Nearest(BundleWriter.FRAME_CLOCK, tol=0.05),
    recording_id="drive_0423_a", timecode_origin=first_sample_time)
manifest = writer.write()
```

**Writing bundles.**

- [`Stream`][.Stream]: one output stream, and how its frames are rendered.
- [`BundleWriter`][.BundleWriter]: plans and writes a bundle.
- [`EncodeOptions`][.EncodeOptions]: encoder settings.

**Bundle format.**

- [`Profile`][.Profile] and [`ProfileName`][.ProfileName]: how a stream is
    encoded. See [encoding profiles][?encoding-profiles].
- [`Timecode`][.Timecode]: the [timecode][?timecode] of a frame.
- [`Manifest`][.Manifest] and [`StreamEntry`][.StreamEntry]: the
    [manifest][?manifest]. `Manifest` also gives each stream's file path and
    its reel and clip [names][?container].
- [`validate_stream`][.validate_stream]: check a stream file against the
    spec.

**Tools.**

- [`render`][.render]: helpers that turn arrays into RGB frames.
- [`progress`][.progress]: progress bars for writing bundles.
"""

from ._encode import EncodeOptions
from ._manifest import Manifest, StreamEntry
from ._profile import Profile, ProfileName
from ._stream import Stream
from ._timecode import Timecode
from ._validate import validate_stream
from ._writer import BundleWriter

__all__ = [
    "BundleWriter",
    "EncodeOptions",
    "Manifest",
    "Profile",
    "ProfileName",
    "Stream",
    "StreamEntry",
    "Timecode",
    "validate_stream",
]
