# Multireel Bundle Spec

A bundle is a directory that holds the rendered video streams of one recording, together with a manifest that describes them. Each stream is a video file that can be imported directly into a video editor. All streams in a bundle share the same frame rate, start timecode and frame count, so they line up with each other in an editor without any extra synchronization.

A bundle looks like this on disk:

```
drive_0423_a/
├── manifest.yaml
└── streams/
    ├── front_cam.mov
    └── front_boxes.mov
```

The keywords MUST, SHOULD and MAY are used as defined in [RFC 2119](https://www.rfc-editor.org/info/rfc2119/).

## Definitions

| Term | Definition |
| --- | --- |
| Recording | The source data that bundles are rendered from, identified by its `recording_id`. |
| Source sample | One timestamped item of source data, such as a single camera image. Its timestamp is called the source timestamp. |
| Bundle | A directory that holds the streams rendered from one recording over one time range, together with a manifest. |
| Stream | One video file in a bundle, stored at `streams/<stream_id>.mov` and rendered from one sequence of source samples. |
| Manifest | The `manifest.yaml` file at the root of a bundle, which describes the bundle and its streams. |
| Profile | One of five named codec configurations that a stream can be encoded with. |
| Overlay | A stream with alpha that is composited over another stream, called its base stream. |
| Edit rate | The constant frame rate shared by every stream in a bundle. |
| Frame instant | The absolute time that a frame represents. Frame `n` at edit rate `r` sits at `n/r` seconds since the UTC epoch. |
| Timecode origin | The absolute UTC time shown as timecode 01:00:00:00. |
| Keyframe interval | The number of frames `N` such that every frame whose index is a multiple of `N` is a keyframe in every stream. |
| No-data frame | A frame that shows no source sample. |

## Video format

Every stream in a bundle is a constant-frame-rate MOV file at the bundle's edit rate, encoded with one of five profiles.

### Encoding profiles

Each stream MUST use exactly one of the following profiles.

| Profile | Codec | Chroma, bit depth | Alpha |
| --- | --- | --- | --- |
| `hevc-main10-420` | HEVC Main 10 | 4:2:0, 10-bit | No |
| `hevc-444` | HEVC Range Extensions | 4:4:4, 10-bit | No |
| `prores-422-lt` | Apple ProRes 422 LT | 4:2:2, 10-bit | No |
| `prores-422-hq` | Apple ProRes 422 HQ | 4:2:2, 10-bit | No |
| `prores-4444` | Apple ProRes 4444 | 4:4:4, 12-bit | Yes |

Overlay streams SHOULD use `prores-4444`. Other streams MAY use any profile.

**Decoding.** Every stream MUST decode correctly in software with stock FFmpeg.

### Frame rate and keyframes

**Edit rate.** All streams in a bundle share one constant edit rate, which MUST be one of 10, 15, 24, 25, 30, 50 or 60 fps, integer and non-drop-frame. Every frame MUST last exactly one frame period. Every stream in a bundle MUST start at the same timecode and contain the same number of frames. The [time model](#time-model) defines where each frame sits in time and what it shows.

**Group of Pictures (GOP) structure.** GOPs MUST be closed, so no frame may reference a frame outside its own GOP. B-frames MUST NOT be used, so decode order equals presentation order. Files MUST NOT need composition time offsets or edit lists.

**Keyframe alignment.** The bundle declares a single keyframe interval `N`, in frames. Every frame whose position in the stream, counting from 0, is a multiple of `N` MUST be a keyframe (IDR) in every stream. Other frames MAY also be keyframes. ProRes profiles are intra-only, so every ProRes frame is a keyframe.

### Picture format

**Dimensions.** Width and height MUST be even and at most 4096. Pixels MUST be square, and dimensions MUST stay constant for the whole stream. Video MUST be progressive, with an identity display matrix and no rotation.

**Color.** Video MUST be SDR, with BT.709 primaries, transfer function and matrix, in limited (video) range. This applies to all profiles.

### Container

Each stream is one QuickTime (MOV) file.

**Tracks.** The file MUST contain one video track and one `tmcd` timecode track, and no audio, subtitle or data tracks. HEVC streams MUST use the `hvc1` sample entry rather than `hev1`, because QuickTime and some editors refuse `hev1`. The timecode track MUST be non-drop-frame at the edit rate, and the first frame's timecode MUST equal the bundle's `start_timecode`.

**Color tags.** A `colr` atom of type `nclc` or `nclx` MUST state BT.709 primaries, transfer and matrix. An `nclx` atom MUST also state limited range. The color fields in the bitstream (the HEVC VUI or the ProRes frame header) MUST match it.

**Names and encoder metadata.** The reel name MUST be `<recording_id>.<stream_id>`. The clip name MUST be `<recording_id> <label>`, with the stream ID in place of the label if the stream has none. File metadata MUST record the encoder name, encoder version and full encoder settings. This spec does not fix which QuickTime metadata fields hold this information.

### Overlays

An overlay is a stream with alpha that is composited over a base stream.

- Overlay frame `k` is composited over base frame `k`.
- Alpha is straight (not premultiplied) unless the stream declares `alpha_mode: "premultiplied"` in the manifest.
- An overlay SHOULD name its base stream in the manifest with `overlay_of`.
- An overlay linked to a base stream MUST match the base stream's dimensions and frame count. It MUST be drawn in the base stream's image space, so if the base is undistorted or cropped, the overlay must be as well.

## Time model

The time model defines where each frame sits in absolute time and how frames map to timecode.

### Frame grid

**Clock domain.** All times in a bundle are expressed in one time base, which is seconds since the UTC epoch (Unix time). The source timestamps of every stream in a bundle are in this time base. This spec does not define which moment of a measurement a timestamp refers to, such as the start or middle of an exposure.

**Frame instants.** Frame `n` of a bundle at edit rate `r` sits at the instant `n/r` seconds since the UTC epoch. All bundles at all edit rates therefore share one absolute grid. Frames are instants, not intervals.

**Grid alignment.** Whole seconds are grid points at every allowed edit rate. A lower-rate grid is a subset of a higher-rate grid whenever the higher rate is a multiple of the lower one.

### Timecode

**Origin.** Each bundle has a timecode origin, which is the absolute UTC time shown as timecode 01:00:00:00. The manifest records it as `timecode_origin`. The origin MUST be a whole number of seconds, so that timecode stays on the frame grid at every edit rate. Bundles with the same `recording_id` MUST use the same origin.

**Frame timecode.** A frame's timecode is 01:00:00:00 plus the time from the origin to the frame instant, written as non-drop-frame timecode at the edit rate. Frames before the origin count down into 00:xx:xx:xx. Timecode MUST NOT wrap, so a bundle lies between one hour before the origin and 23 hours after it. The absolute time of any frame can be recovered from its timecode and the origin.

### Frame contents

**Sample selection.** Each stream is made from a sequence of timestamped source samples, and each frame shows at most one of them. This spec does not define which sample a frame shows. The writer chooses it. Bundles store no per-frame timing data.

**No-data frames.** Every stream covers the whole bundle range. Frames that show no sample are plain black in opaque streams and fully transparent in overlays. Bundles store no gap metadata, so gaps are visible only in the frame content.

## Manifest

Each bundle is described by a YAML file, `manifest.yaml`, at the bundle root. A bundle is a directory that can be moved or copied as a unit.

### File format

**YAML.** The manifest is a single YAML 1.2 document encoded as UTF-8. It MUST NOT use anchors, aliases or custom tags. Strings that YAML could read as another type, such as `1.10`, `no` or `on`, MUST be quoted.

**Paths and stream IDs.** Stream files are stored at `streams/<stream_id>.mov`. Paths in the manifest are relative to the bundle root. Other specs MAY add their own subdirectories. Stream IDs use only lowercase letters, digits, `_` and `-`, because they double as file names and file systems on macOS and Windows are case-insensitive.

**Versioning.** `schema_version` is a quoted `"major.minor"` string. Adding optional fields bumps the minor version, and any other change bumps the major version. Readers MUST reject an unknown major version. Readers MUST accept a newer minor version and ignore any fields they do not recognize.

**Completeness.** A directory is a bundle only if it contains `manifest.yaml`. The presence of `manifest.yaml` means that every stream file in the bundle is complete and conforms to this spec.

### Bundle fields

The top level of the manifest identifies the bundle and its recording, and places the bundle in time.

```yaml
schema_version: "1.0"
bundle_id: "6f1c2a9e-0b7d-4c41-9a53-2e8f7d1b4c60"
recording_id: "drive_0423_a"
timecode_origin: "2026-11-18T16:00:00Z"
edit_rate: 30
keyframe_interval: 15
start_timecode: "01:12:00:00"
frame_count: 9000
provenance: { }
streams: { }      # see Stream fields
```

All fields in the table below are required.

| Field | Type | Meaning |
| --- | --- | --- |
| `bundle_id` | UUID string | Unique per bundle |
| `recording_id` | string | Source recording |
| `timecode_origin` | ISO 8601 UTC string, whole seconds | Absolute time shown as 01:00:00:00 |
| `edit_rate` | integer | Frames per second |
| `keyframe_interval` | integer | Keyframe interval `N`, in frames |
| `start_timecode` | `"HH:MM:SS:FF"` string | Timecode of the first frame |
| `frame_count` | integer | Number of frames in every stream |

`start_timecode` and `frame_count` MUST match every stream file.

**Provenance.** `provenance` is optional and may hold any YAML value. This spec does not define its contents.

### Stream fields

`streams` maps each stream ID to that stream's fields. A stream's file is always `streams/<id>.mov`, so the path is not stored. The order of entries has no meaning.

```yaml
streams:
  front_cam:
    label: "Front camera"
    profile: "hevc-main10-420"
    info: { clock_source: "ptp", timestamp_reference: "exposure_mid" }
  front_boxes:
    profile: "prores-4444"
    overlay_of: "front_cam"
```

| Field | Required | Meaning |
| --- | --- | --- |
| `label` | No | Human-readable name for display |
| `profile` | Yes | [Encoding profile](#encoding-profiles) |
| `overlay_of` | No | Base stream ID, for overlays only |
| `alpha_mode` | No | `"straight"` (default) or `"premultiplied"`, for overlays only |
| `info` | No | Mapping of any structure, whose contents this spec does not define |

`overlay_of` MUST name another stream in the same bundle, and that stream MUST NOT itself be an overlay. Properties stored in the stream files, such as dimensions, are not repeated in the manifest.
