"""Encoding stream files with `ffmpeg`.

HEVC profiles are encoded with `libx265` or `hevc_nvenc`, and ProRes profiles
with `prores_ks`. Each file records its encoder and settings in its metadata.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

import numpy as np
from jaxtyping import UInt8

from multireel.bundle import Manifest, Profile


@dataclass(frozen=True)
class EncodeOptions:
    """Encoder settings for a bundle.

    `quality` is the CRF for `libx265`, or the CQ for `hevc_nvenc`. ProRes
    quality is set by the profile's level instead.

    Attributes:
        ffmpeg: `ffmpeg` executable.
        ffprobe: `ffprobe` executable.
        hevc_encoder: encoder for HEVC profiles.
        quality: HEVC quality level. Lower is better.
    """

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    hevc_encoder: Literal["libx265", "hevc_nvenc"] = "libx265"
    quality: int = 20

    def codec_args(
        self, profile: Profile, keyframe_interval: int
    ) -> tuple[str, str, list[str]]:
        """Return the `ffmpeg` encoder, pixel format and arguments.

        Raises:
            ValueError: if the profile is not supported.
        """
        n = str(keyframe_interval)
        q = str(self.quality)
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

        if self.hevc_encoder == "hevc_nvenc":
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


@cache
def _ffmpeg_version(ffmpeg: str) -> str:
    """Return the version string of an `ffmpeg` executable."""
    out = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, text=True, check=True)
    return out.stdout.split()[2]


class Encoder:
    """Encodes frames into one stream file with `ffmpeg`.

    The encoding settings come from the manifest. Use the encoder as a
    context manager. On exit, it finalizes the file with [`close`][.close],
    or stops `ffmpeg` with [`abort`][.abort] if an exception was raised.

    Args:
        path: output file.
        manifest: bundle manifest.
        stream_id: ID of the stream.
        width: frame width, in pixels.
        height: frame height, in pixels.
        options: encoder settings.
        channel_order: color channel order of the frames.

    Raises:
        ValueError: if the stream's profile is not supported.
    """

    def __init__(
        self, path: Path, manifest: Manifest, stream_id: str, width: int,
        height: int, options: EncodeOptions,
        channel_order: Literal["rgb", "bgr"] = "rgb",
    ) -> None:
        profile = manifest.streams[stream_id].profile
        self.channels = 4 if profile.alpha else 3
        self.shape = (height, width, self.channels)

        name, pix_fmt, codec = options.codec_args(
            profile, manifest.keyframe_interval)
        in_pix_fmt = channel_order + ("a" if self.channels == 4 else "24")
        version = _ffmpeg_version(options.ffmpeg)
        args = [
            options.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", in_pix_fmt,
            "-s", f"{width}x{height}",
            "-framerate", str(manifest.edit_rate), "-i", "-",
            "-vf", f"scale=out_color_matrix=bt709:out_range=tv,"
            f"format={pix_fmt},setparams=color_primaries=bt709:"
            "color_trc=bt709:colorspace=bt709:range=tv",
            *codec,
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv",
            "-timecode", str(manifest.start_timecode),
            "-metadata:s:v:0", f"reel_name={manifest.reel_name(stream_id)}",
            "-metadata", f"title={manifest.clip_name(stream_id)}",
            "-metadata", f"multireel_encoder={name}",
            "-metadata", f"multireel_encoder_version=ffmpeg {version}",
            "-metadata", f"multireel_encoder_settings={' '.join(codec)}",
            "-movflags", "+use_metadata_tags", "-f", "mov", str(path)]

        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stderr=self.stderr)

    def __enter__(self) -> Self:
        """Return this encoder."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        """Finalize the file, or stop `ffmpeg` if an exception was raised."""
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def write(self, frame: UInt8[np.ndarray, "H W C"], count: int = 1) -> None:
        """Write a frame `count` times.

        Raises:
            ValueError: if the frame has the wrong shape or type.
            RuntimeError: if `ffmpeg` fails.
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
