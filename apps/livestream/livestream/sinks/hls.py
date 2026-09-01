"""HLS sink: the same ffmpeg encode as `rtmp.py`, written to a playlist on disk.

Why this exists: Cloudflare Tunnel proxies HTTP and TCP, while WebRTC media is
UDP, so a tunnelled demo cannot carry the model's WebRTC tracks to a browser
directly. HLS turns the paced stream into plain files that any HTTP server --
including the demo's own web app -- can serve through the tunnel, and every
browser plays without a player library.

It inherits `RtmpSink` wholesale: the two raw input pipes, the writer threads
that keep the event loop unblocked, the restart policy, and the encoder
settings are all the same hard-won machinery. Only the destination differs, so
only `_output_args` is overridden.

Latency is one segment plus the player's buffer -- a few seconds at the shipped
2 s segments. That is irrelevant here because clips are pre-built anyway; lower
latency would mean LL-HLS or a WebRTC SFU, both of which cost a service.

The playlist is a sliding window: `delete_segments` keeps the directory bounded
at roughly `hls_list_size` segments, so a stream can run for days without
filling the disk.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .rtmp import RtmpSink

logger = logging.getLogger("livestream.sinks.hls")

# One segment is the floor on how fresh a viewer's picture can be, and also the
# unit ffmpeg must finish before anything is playable at all. Two seconds is the
# usual compromise: short enough to start fast, long enough that segment churn
# does not dominate.
SEGMENT_SECONDS = 2
# Segments kept in the playlist. Six gives a viewer ~12 s of rewind buffer and
# bounds the directory; older segments are deleted from disk as they roll off.
PLAYLIST_SEGMENTS = 6


class HlsSink(RtmpSink):
    """Write the paced stream as an HLS playlist under `directory`."""

    def __init__(
        self,
        directory: str | Path,
        video_bitrate_k: int = 4500,
        *,
        playlist_name: str = "stream.m3u8",
    ) -> None:
        # RtmpSink's constructor checks for ffmpeg and sets up the writers; the
        # URL it stores is only ever read by `_output_args`, which this class
        # replaces, so the playlist path stands in for it.
        self._directory = Path(directory)
        self._playlist_name = playlist_name
        super().__init__(str(self._directory / playlist_name), video_bitrate_k=video_bitrate_k)

    @property
    def playlist_path(self) -> Path:
        """Where the web app should point the player."""
        return self._directory / self._playlist_name

    def _output_args(self) -> list[str]:
        # A restart must not leave a viewer's player reading a playlist that
        # references segments from before the gap, so the directory is cleared
        # each time ffmpeg is spawned.
        self._reset_directory()
        return [
            "-f",
            "hls",
            "-hls_time",
            str(SEGMENT_SECONDS),
            "-hls_list_size",
            str(PLAYLIST_SEGMENTS),
            # delete_segments bounds the directory; independent_segments lets a
            # player start on any segment; omit_endlist keeps the playlist live
            # rather than signalling the end of a VOD after every write.
            "-hls_flags",
            "delete_segments+independent_segments+omit_endlist+program_date_time",
            "-hls_segment_type",
            "mpegts",
            "-hls_segment_filename",
            str(self._directory / "seg_%05d.ts"),
            str(self.playlist_path),
        ]

    def _reset_directory(self) -> None:
        if self._directory.exists():
            shutil.rmtree(self._directory, ignore_errors=True)
        self._directory.mkdir(parents=True, exist_ok=True)
        logger.info("[hls] writing %s", self.playlist_path)
