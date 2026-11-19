"""Encoding rendered frames into stream files with `ffmpeg`.

HEVC profiles use `libx265` or `hevc_nvenc`, and ProRes profiles use
`prores_ks`. Frames are piped to an `ffmpeg` subprocess as raw 8-bit RGB(A)
or BGR(A) video, and converted to BT.709 limited-range YUV by `ffmpeg`.

The encoder name, version and settings are stored in the file's metadata
under `multireel_encoder`, `multireel_encoder_version` and
`multireel_encoder_settings`. The clip name is stored as `title`, and the
reel name on the timecode track.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

ChannelOrder = Literal["rgb", "bgr"]
"""Order of the color channels in input frames.

Alpha, if any, always comes last. `bgr` suits sources such as OpenCV, which
decode to BGR, and saves reordering the channels before encoding.
"""

import numpy as np
from jaxtyping import UInt8

from .bundle.timing import Timecode
from .bundle.video import Profile


@dataclass(frozen=True)
class EncodeOptions:
    """Encoder settings shared by all streams in a bundle.

    Attributes:
        ffmpeg: `ffmpeg` executable.
        ffprobe: `ffprobe` executable, used to validate each stream.
        hevc_encoder: encoder for HEVC profiles.
        quality: constant-quality level for HEVC profiles: CRF for
            `libx265`, CQ for `hevc_nvenc`. Lower is better. ProRes quality
            is set by the profile's `level`.
    """

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    hevc_encoder: Literal["libx265", "hevc_nvenc"] = "libx265"
    quality: int = 20


@dataclass(frozen=True)
class StreamFormat:
    """Everything the encoder needs to know about one stream file.

    Attributes:
        width: frame width, in pixels.
        height: frame height, in pixels.
        profile: encoding profile.
        edit_rate: frames per second.
        keyframe_interval: keyframe interval `N`, in frames.
        start_timecode: timecode of the first frame.
        reel_name: reel name.
        clip_name: clip name.
        channel_order: order of the color channels in input frames.
    """

    width: int
    height: int
    profile: Profile
    edit_rate: int
    keyframe_interval: int
    start_timecode: Timecode
    reel_name: str
    clip_name: str
    channel_order: ChannelOrder = "rgb"


@cache
def _ffmpeg_version(ffmpeg: str) -> str:
    """Return the version string of an `ffmpeg` executable."""
    out = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, text=True, check=True)
    return out.stdout.split()[2]


def _codec_args(
    format: StreamFormat, options: EncodeOptions
) -> tuple[str, str, list[str]]:
    """Return the encoder name, its input pixel format, and its arguments.

    Raises:
        ValueError: if the encoder cannot produce the profile.
    """
    n = str(format.keyframe_interval)
    q = str(options.quality)
    profile = format.profile
    chroma, depth = profile.chroma, profile.bit_depth

    if profile.codec == "prores":
        if profile.level is None:
            raise ValueError(f"ProRes profile {profile} has no level.")
        # ProRes is always encoded from 10-bit input; 4444 decodes as
        # 12-bit regardless.
        alpha = "a" if profile.alpha else ""
        return "prores_ks", f"yuv{alpha}{chroma}p10le", [
            "-c:v", "prores_ks", "-profile:v", profile.level,
            "-vendor", "apl0"]

    if profile.alpha:
        raise ValueError("HEVC profiles cannot carry alpha.")

    if options.hevc_encoder == "hevc_nvenc":
        pix_fmt = {
            ("420", 8): "nv12", ("420", 10): "p010le",
            ("444", 8): "yuv444p", ("444", 10): "yuv444p16le",
        }.get((chroma, depth))
        if pix_fmt is None:
            raise ValueError(
                f"hevc_nvenc does not support {chroma} {depth}-bit.")
        nvenc_profile = "rext" if chroma == "444" else (
            "main10" if depth == 10 else "main")
        return "hevc_nvenc", pix_fmt, [
            "-c:v", "hevc_nvenc", "-profile:v", nvenc_profile,
            "-rc", "vbr", "-cq", q, "-b:v", "0", "-bf", "0", "-g", n,
            "-forced-idr", "1", "-no-scenecut", "1", "-tag:v", "hvc1"]

    pix_fmt = f"yuv{chroma}p" + (f"{depth}le" if depth > 8 else "")
    x265_profile = (
        ("main" if depth == 8 else f"main{depth}") if chroma == "420"
        else f"main{chroma}-{depth}")
    return "libx265", pix_fmt, [
        "-c:v", "libx265", "-profile:v", x265_profile,
        "-crf", q,
        "-x265-params",
        f"keyint={n}:min-keyint={n}:scenecut=0:bframes=0:open-gop=0"
        ":log-level=error",
        "-tag:v", "hvc1"]


class Encoder:
    """Encodes the frames of one stream into one MOV file.

    Frames are written in presentation order.

    Args:
        path: output MOV file.
        format: stream format.
        options: encoder settings.
    """

    def __init__(
        self, path: Path, format: StreamFormat, options: EncodeOptions
    ) -> None:
        self.channels = 4 if format.profile.alpha else 3
        self.shape = (format.height, format.width, self.channels)

        name, pix_fmt, codec = _codec_args(format, options)
        in_pix_fmt = format.channel_order + (
            "a" if self.channels == 4 else "24")
        version = _ffmpeg_version(options.ffmpeg)
        args = [
            options.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", in_pix_fmt,
            "-s", f"{format.width}x{format.height}",
            "-framerate", str(format.edit_rate), "-i", "-",
            "-vf", f"scale=out_color_matrix=bt709:out_range=tv,"
            f"format={pix_fmt},setparams=color_primaries=bt709:"
            "color_trc=bt709:colorspace=bt709:range=tv",
            *codec,
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv",
            "-timecode", str(format.start_timecode),
            "-metadata:s:v:0", f"reel_name={format.reel_name}",
            "-metadata", f"title={format.clip_name}",
            "-metadata", f"multireel_encoder={name}",
            "-metadata", f"multireel_encoder_version=ffmpeg {version}",
            "-metadata", f"multireel_encoder_settings={' '.join(codec)}",
            "-movflags", "+use_metadata_tags", "-f", "mov", str(path)]

        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stderr=self.stderr)

    def write(self, frame: UInt8[np.ndarray, "H W C"], count: int = 1) -> None:
        """Write a frame, repeated `count` times.

        Frames are 8-bit, in the stream's channel order, with alpha last
        if the profile has alpha.

        Repeats are how held samples and runs of no-data frames are written.

        Raises:
            ValueError: if the frame has the wrong shape or type.
        """
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected a uint8 frame of shape {self.shape}, got "
                f"{frame.dtype} {frame.shape}.")
        data = np.ascontiguousarray(frame).tobytes()
        assert self.process.stdin is not None
        try:
            for _ in range(count):
                self.process.stdin.write(data)
        except BrokenPipeError:
            self.process.wait()
            raise RuntimeError(f"ffmpeg failed: {self._errors()}") from None

    def close(self) -> None:
        """Finish encoding and finalize the file.

        Raises:
            RuntimeError: if `ffmpeg` exited with an error.
        """
        assert self.process.stdin is not None
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {self._errors()}")
        self.stderr.close()

    def _errors(self) -> str:
        """Return what `ffmpeg` wrote to stderr."""
        self.stderr.seek(0)
        return self.stderr.read().decode(errors="replace").strip()

    def abort(self) -> None:
        """Stop `ffmpeg` without finalizing the file."""
        self.process.kill()
        self.process.wait()
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        self.stderr.close()
