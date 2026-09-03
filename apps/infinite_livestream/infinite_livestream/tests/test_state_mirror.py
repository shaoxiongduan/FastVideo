"""The web page's state mirror folds engine messages correctly.

`DemoState` is what every viewer's panel is rendered from, and it is pure: a
sequence of `(kind, data)` in, a snapshot out. That makes the behaviour worth
pinning here rather than discovering on a live stream -- particularly the
now-playing lifecycle, which is what the page uses to keep the title in step
with the picture.
"""

from __future__ import annotations

from infinite_livestream.webapp import DemoState


def clip(clip_id: str = "abcdef123456", prompt: str = "a lighthouse keeper", *, generated: bool = False,
         scene: int | None = None, scenes: int | None = None) -> dict:
    import json
    meta = {"group_id": "g1", "title": prompt, "author": "viewer", "generated": generated, "raw_prompt": prompt}
    if scene is not None:
        meta |= {"scene": scene, "scenes": scenes}
    return {"clip_id": clip_id, "prompt": prompt, "metadata": json.dumps(meta), "frames": 345,
            "seconds": 14.375, "seed": 1, "ready": True}


def test_queue_update_replaces_both_queues() -> None:
    state = DemoState()
    state.on_message("queue_update", {"generation": [clip("a")], "playout": [clip("b"), clip("c")]})
    assert [c["clip_id"] for c in state.generation] == ["a"]
    assert [c["clip_id"] for c in state.playout] == ["b", "c"]
    # Replacement, not accumulation: a queue that empties must render empty.
    state.on_message("queue_update", {"generation": [], "playout": []})
    assert state.generation == [] and state.playout == []


def test_generating_is_the_generation_front() -> None:
    """Builds consume the queue front-first, so the front is what is in flight."""
    state = DemoState()
    assert state.generating is None
    state.on_message("queue_update", {"generation": [clip("a"), clip("b")], "playout": []})
    generating = state.generating
    assert generating is not None and generating["clip_id"] == "a"


def test_now_playing_clears_when_a_clip_ends() -> None:
    state = DemoState()
    state.on_message("clip_started", {"clip": clip("a")})
    playing = state.now_playing
    assert playing is not None and playing["clip_id"] == "a"
    state.on_message("clip_finished", {"clip": clip("a"), "seconds_sent": 14.4})
    assert state.now_playing is None


def test_timeline_records_gaps_as_well_as_clips() -> None:
    """The page reads the timeline to name what the viewer is seeing *now*.

    Without the trailing None it would keep naming a clip that already ended.
    """
    state = DemoState()
    state.on_message("clip_started", {"clip": clip("a")})
    state.on_message("clip_finished", {"clip": clip("a"), "seconds_sent": 14.4})
    entries = list(state.timeline)
    assert [e["clip"] and e["clip"]["clip_id"] for e in entries] == ["a", None]
    assert entries[0]["at"] <= entries[1]["at"]


def test_state_update_without_playing_clears_now_playing() -> None:
    """The engine's own view wins: a restart must not strand a stale title."""
    state = DemoState()
    state.on_message("clip_started", {"clip": clip("a")})
    state.on_message("state_update", {"playing": False, "generation_queued": 0, "playout_queued": 0})
    assert state.now_playing is None
    assert state.connected is True


def test_only_filler_is_announced_in_chat_and_once_per_group() -> None:
    """Viewer submissions are echoed by the POST handler, so only filler here.

    And one line per group, not per scene: a six-scene story is still one
    thing somebody asked for.
    """
    state = DemoState()
    state.on_message("clip_queued", {"clip": clip("v", "viewer idea", generated=False)})
    assert list(state.chat) == []
    for scene in (1, 2, 3):
        state.on_message("clip_queued", {"clip": clip(f"f{scene}", "filler idea", generated=True,
                                                      scene=scene, scenes=3)})
    assert [c["author"] for c in state.chat] == ["filler"]


def test_failed_viewer_clips_are_reported_but_filler_is_not() -> None:
    state = DemoState()
    state.on_message("clip_failed", {"clip": clip("f", generated=True), "reason": "boom"})
    assert list(state.chat) == []
    state.on_message("clip_failed", {"clip": clip("v", "viewer idea", generated=False), "reason": "boom"})
    assert [c["kind"] for c in state.chat] == ["error"]


def test_snapshot_carries_everything_the_page_reads() -> None:
    state = DemoState()
    state.on_message("state_update", {"playing": False, "generation_queued": 1, "generation_capacity": 20,
                                      "playout_queued": 2, "playout_capacity": 10, "clips_played": 7,
                                      "width": 1344, "height": 768})
    snap = state.snapshot()
    assert set(snap) == {"connected", "now_playing", "timeline", "live_edge", "generating",
                         "generation", "playout", "stats", "chat"}
    assert snap["stats"]["clips_played"] == 7


def test_timeline_uses_the_stream_clock_not_wall_clock() -> None:
    """Timeline entries must be stamped in the same clock as the playlist.

    The page locates what is on screen by comparing EXT-X-PROGRAM-DATE-TIME
    against these timestamps. Wall clock is a fraction of a second early: a
    frame still has the pacer's buffer and ffmpeg's encoder to cross before it
    is stamped, so the sink supplies the corrected instant.
    """
    state = DemoState()
    state.stream_clock = lambda: 1000.0
    state.on_message("clip_started", {"clip": clip("a")})
    state.on_message("clip_finished", {"clip": clip("a"), "seconds_sent": 14.4})
    assert [e["at"] for e in state.timeline] == [1000.0, 1000.0]


def test_timeline_falls_back_to_wall_clock_before_ffmpeg_starts() -> None:
    """`stream_time` is None until the first ffmpeg is up; a timestamp is still
    better than none, and the page has no PDT to compare against yet anyway."""
    import time as _time

    state = DemoState()
    state.stream_clock = lambda: None
    before = _time.time()
    state.on_message("clip_started", {"clip": clip("a")})
    assert before <= state.timeline[0]["at"] <= _time.time()


def test_snapshot_no_longer_carries_server_now() -> None:
    """It existed only to drive the client's lag estimate, which is gone."""
    assert "server_now" not in DemoState().snapshot()


def test_live_edge_comes_from_the_sink() -> None:
    """The page needs it to locate itself when the player exposes no date."""
    state = DemoState()
    assert state.snapshot()["live_edge"] is None
    state.live_edge_clock = lambda: 4321.0
    assert state.snapshot()["live_edge"] == 4321.0
