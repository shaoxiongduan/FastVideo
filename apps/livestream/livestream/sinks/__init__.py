"""Output sinks. `make_sink` is the one place a sink name maps to a class.

Adding a destination (LiveKit, an SFU, a file recorder, ...):
  1. Implement `StreamSink` (see `base.py` for the contract) in a new module.
  2. Add a branch to `make_sink` and a value for `SINK` in `.env.example`.
  3. Document it in the README's sink table.
"""

from __future__ import annotations

from .base import AudioFormat, StreamSink, VideoFormat
from .hls import HlsSink
from .noop import NoOpSink
from .rtmp import RtmpSink

__all__ = [
    "AudioFormat",
    "HlsSink",
    "NoOpSink",
    "RtmpSink",
    "StreamSink",
    "VideoFormat",
    "make_sink",
]


def make_sink(
    name: str,
    *,
    rtmp_url: str | None = None,
    rtmp_video_bitrate_k: int = 4500,
    hls_dir: str | None = None,
) -> StreamSink:
    """Build the sink named by config."""
    if name == "noop":
        return NoOpSink()
    if name == "hls":
        if not hls_dir:
            raise ValueError("the hls sink needs an output directory")
        return HlsSink(hls_dir, video_bitrate_k=rtmp_video_bitrate_k)
    if name == "rtmp":
        if not rtmp_url:
            raise ValueError("the rtmp sink needs an RTMP URL")
        return RtmpSink(rtmp_url, video_bitrate_k=rtmp_video_bitrate_k)
    raise ValueError(f"unknown sink {name!r}")
