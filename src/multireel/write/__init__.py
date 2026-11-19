"""Writing bundles.

A [`BundleWriter`][.BundleWriter] writes a set of
[`Stream`][multireel.stream.Stream]s as a bundle:

```python
from abstract_dataloader.generic import Nearest
from multireel import FRAME_CLOCK, BundleWriter, Profile, Stream

streams = {
    "front_cam": Stream.from_sensor(trace["camera"], render_camera),
    "front_boxes": Stream.from_sensor(
        trace["detections"], render_boxes,
        profile=Profile.from_name("prores-4444"), overlay_of="front_cam"),
}
writer = BundleWriter(
    "out/drive_0423_a", streams, Nearest(FRAME_CLOCK, tol=0.05),
    recording_id="drive_0423_a", timecode_origin=first_sample_time)
manifest = writer.write()
```

Writing has two steps:

1. Creating a `BundleWriter` checks every argument and plans the bundle.
    Nothing is rendered yet.
2. [`write`][.BundleWriter.write] renders and encodes each stream, checks
    each file with [`validate_stream`][.validate_stream], and writes the
    manifest.

The synchronization chooses the sample shown in each frame. It receives the
timestamps of each stream, and the frame times under the key
[`FRAME_CLOCK`][.FRAME_CLOCK]. Frames that it drops show no data in every
stream. Each shown sample is rendered once.
"""

from .encode import EncodeOptions, Encoder
from .validate import StreamProbe, check_stream, decode_errors, validate_stream
from .writer import (
    FRAME_CLOCK,
    BundleWriter,
)

__all__ = [
    "FRAME_CLOCK",
    "BundleWriter",
    "EncodeOptions",
    "Encoder",
    "StreamProbe",
    "check_stream",
    "decode_errors",
    "validate_stream",
]
