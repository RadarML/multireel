"""Writing bundles from output streams.

A [`BundleWriter`][.BundleWriter] renders, encodes and validates a set of
[`Stream`][multireel.stream.Stream]s, then writes the bundle's manifest:

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

A [`Synchronization`][abstract_dataloader.spec.Synchronization] decides which
sample each frame shows. The writer calls it once, with the timestamps of
every stream keyed by stream ID, plus the instants of the bundle's frames
keyed by [`FRAME_CLOCK`][.FRAME_CLOCK]. The frame clock is usually the
reference. Frames that the result leaves out are no-data frames in every
stream.

Only samples that are actually shown are rendered, and each one is rendered
once, however many frames it is shown for.

Writing a bundle has two steps:

- [`BundleWriter`][.BundleWriter] plans the bundle when constructed: it
    picks the frame range, builds the manifest, and runs the
    synchronization to choose the sample each frame shows, keyed by
    [`FRAME_CLOCK`][.FRAME_CLOCK]. Every argument is checked here, before
    anything is rendered.
- [`BundleWriter.write`][.BundleWriter.write] writes each stream in its own
    forked process: it renders the stream, encodes it with an
    [`Encoder`][.Encoder] set up by [`EncodeOptions`][.EncodeOptions], and
    checks the file with [`validate_stream`][.validate_stream]. It reports
    progress as it goes.

[`validate_stream`][.validate_stream] reads a file's properties into a
[`StreamProbe`][.StreamProbe], checks them with
[`check_stream`][.check_stream], and decodes every frame with
[`decode_errors`][.decode_errors].
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
