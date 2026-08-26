# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the video arena: manifest loading, pairing, voting, aggregation.

Run with:  pytest apps/video_arena/tests -v
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from apps.video_arena.app import RANDOM_PROMPT, ArenaUI
from apps.video_arena.arena import Arena
from apps.video_arena.storage import VOTE_BOTH_BAD, VOTE_LEFT, VOTE_RIGHT, VOTE_TIE, VoteStore

MODELS = ["ckpt-a", "ckpt-b", "ckpt-c"]
PROMPT_IDS = ["p001", "p002", "p003"]


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    for mid in MODELS:
        d = tmp_path / "videos" / mid
        d.mkdir(parents=True)
        for pid in PROMPT_IDS:
            (d / f"{pid}.mp4").write_bytes(b"not-a-real-mp4")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({
            "name": "test arena",
            "models": [{"id": m, "name": f"Model {m}", "dir": f"videos/{m}"} for m in MODELS],
            "prompts": [{"id": p, "text": f"prompt text {p}"} for p in PROMPT_IDS],
        }))
    return path


def test_grid_is_fully_populated(manifest: Path) -> None:
    arena = Arena(manifest, anonymize_paths=False)
    assert len(arena.models) == len(MODELS)
    assert [p.id for p in arena.prompts] == PROMPT_IDS
    for pid in PROMPT_IDS:
        assert set(arena.videos[pid]) == set(MODELS)


def test_prompts_are_discovered_from_filenames_when_manifest_omits_them(manifest: Path) -> None:
    raw = json.loads(manifest.read_text())
    del raw["prompts"]
    manifest.write_text(json.dumps(raw))
    (manifest.parent / "prompts.json").write_text(json.dumps({"p001": "recovered text"}))

    arena = Arena(manifest, anonymize_paths=False)
    assert [p.id for p in arena.prompts] == PROMPT_IDS
    assert arena.prompts[0].text == "recovered text"
    assert arena.prompts[1].text == "p002"  # no sidecar entry: falls back to the id


def test_prompt_without_two_videos_is_dropped(manifest: Path) -> None:
    for mid in MODELS[1:]:
        (manifest.parent / "videos" / mid / "p003.mp4").unlink()
    arena = Arena(manifest, anonymize_paths=False)
    assert [p.id for p in arena.prompts] == ["p001", "p002"]


def test_arena_rejects_a_manifest_with_no_usable_prompts(manifest: Path) -> None:
    for mid in MODELS:
        for pid in PROMPT_IDS:
            (manifest.parent / "videos" / mid / f"{pid}.mp4").unlink()
    with pytest.raises(ValueError, match="no prompt has videos"):
        Arena(manifest, anonymize_paths=False)


def test_sampling_is_balanced_and_never_self_pairs(manifest: Path) -> None:
    arena = Arena(manifest, anonymize_paths=False)
    rng = random.Random(0)
    left_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, ...]] = Counter()
    for _ in range(600):
        b = arena.sample_battle(None, rng)
        assert b.left.id != b.right.id
        left_counts[b.left.id] += 1
        pair_counts[tuple(sorted((b.left.id, b.right.id)))] += 1

    # Every model should land on the left roughly a third of the time, and all three
    # unordered pairs should occur — position must not encode identity.
    assert len(pair_counts) == 3
    for c in left_counts.values():
        assert 0.25 < c / 600 < 0.42


def test_prompt_selector_pins_the_prompt(manifest: Path) -> None:
    arena = Arena(manifest, anonymize_paths=False)
    label = next(p.label for p in arena.prompts if p.id == "p002")
    for _ in range(20):
        assert arena.sample_battle(label, random.Random()).prompt.id == "p002"


def test_anonymized_paths_hide_the_model_id(manifest: Path) -> None:
    arena = Arena(manifest, anonymize_paths=True)
    b = arena.sample_battle(None, random.Random(0))
    for serve, real, model in ((b.left_serve, b.left_video, b.left), (b.right_serve, b.right_video, b.right)):
        assert model.id not in str(serve), "model id leaked into the served path"
        assert serve.resolve() == real.resolve(), "anonymized path must point at the real video"
    assert str(arena._anon_root) in arena.serve_roots


def test_vote_is_recorded_with_the_real_model_identities(manifest: Path, tmp_path: Path) -> None:
    arena = Arena(manifest, anonymize_paths=True)
    store = VoteStore(tmp_path / "votes.jsonl")
    ui = ArenaUI(arena, store, random.Random(0))

    state = ui.new_battle(RANDOM_PROMPT, 0)[0]
    battle = state["battle"]
    ui.cast_vote(VOTE_LEFT, state, "sess1", 0)

    (row, ) = store.load().to_dict("records")
    assert row["vote"] == VOTE_LEFT
    assert row["winner"] == battle.left.id
    assert (row["model_left"], row["model_right"]) == (battle.left.id, battle.right.id)
    assert row["video_left"] == str(battle.left_video)  # the real path, not the anon one
    assert row["session_id"] == "sess1"
    assert row["prompt_id"] == battle.prompt.id
    assert row["decision_ms"] >= 0


def test_vote_without_an_active_battle_raises(manifest: Path, tmp_path: Path) -> None:
    ui = ArenaUI(Arena(manifest, anonymize_paths=False), VoteStore(tmp_path / "v.jsonl"))
    with pytest.raises(Exception, match="No active round"):
        ui.cast_vote(VOTE_LEFT, {}, "s", 0)


def test_voting_reveals_both_models_and_stays_on_the_round(manifest: Path, tmp_path: Path) -> None:
    arena = Arena(manifest, anonymize_paths=False)
    ui = ArenaUI(arena, VoteStore(tmp_path / "v.jsonl"), random.Random(1))

    state = ui.new_battle(RANDOM_PROMPT, 0)[0]
    battle = state["battle"]
    out = ui.cast_vote(VOTE_RIGHT, state, "s", 0)

    assert out[0]["battle"].battle_id == battle.battle_id, "vote must not advance on its own"
    reveal = out[4]
    assert battle.left.name in reveal and battle.right.name in reveal, "both identities revealed"
    assert battle.right.name in reveal.split("\n")[0], "winner named first"
    # Vote buttons go dead, "next round" wakes up.
    assert all(u.get("interactive") is False for u in out[5:9])
    assert out[9].get("interactive") is True
    assert out[11] == 1, "rated counter increments"


def test_next_round_re_enables_voting_and_clears_the_reveal(manifest: Path, tmp_path: Path) -> None:
    ui = ArenaUI(Arena(manifest, anonymize_paths=False), VoteStore(tmp_path / "v.jsonl"), random.Random(1))
    state = ui.new_battle(RANDOM_PROMPT, 0)[0]
    ui.cast_vote(VOTE_RIGHT, state, "s", 0)

    out = ui.new_battle(RANDOM_PROMPT, 1)  # what the "next round" button triggers
    assert out[0]["battle"].battle_id != state["battle"].battle_id
    assert out[4] == "", "reveal is cleared"
    assert all(u.get("interactive") is True for u in out[5:9])
    assert out[9].get("interactive") is False


def test_leaderboard_recovers_the_stronger_model(manifest: Path, tmp_path: Path) -> None:
    arena = Arena(manifest, anonymize_paths=False)
    store = VoteStore(tmp_path / "votes.jsonl")
    ui = ArenaUI(arena, store, random.Random(7))

    winner = "ckpt-c"
    for _ in range(120):
        state = ui.new_battle(RANDOM_PROMPT, 0)[0]
        b = state["battle"]
        if winner in (b.left.id, b.right.id):
            vote = VOTE_LEFT if b.left.id == winner else VOTE_RIGHT
        else:
            vote = VOTE_TIE
        ui.cast_vote(vote, state, "s", 0)

    lb = store.leaderboard({m.id: m.name for m in arena.models})
    assert lb.iloc[0]["model"] == winner
    assert lb.iloc[0]["win_rate"] == 1.0
    assert lb.iloc[0]["elo"] > 1000 > lb.iloc[-1]["elo"]

    pw = store.pairwise()
    assert len(pw) == 3  # one row per unordered pair, both orderings merged
    assert pw["n"].sum() == 120


def test_ties_and_both_bad_are_counted_separately(manifest: Path, tmp_path: Path) -> None:
    store = VoteStore(tmp_path / "votes.jsonl")
    for vote in (VOTE_TIE, VOTE_TIE, VOTE_BOTH_BAD):
        store.record(model_left="ckpt-a", model_right="ckpt-b", vote=vote)
    lb = store.leaderboard().set_index("model")
    assert lb.loc["ckpt-a", "ties"] == 2
    assert lb.loc["ckpt-a", "both_bad"] == 1
    assert pd.isna(lb.loc["ckpt-a", "win_rate"])  # no decisive votes -> undefined
    assert lb.loc["ckpt-a", "elo"] == pytest.approx(1000.0)  # non-decisive votes cannot move Elo


def test_empty_store_yields_empty_tables(tmp_path: Path) -> None:
    store = VoteStore(tmp_path / "nothing.jsonl")
    assert store.load().empty
    assert store.leaderboard().empty
    assert store.pairwise().empty


def test_torn_final_line_is_tolerated(tmp_path: Path) -> None:
    store = VoteStore(tmp_path / "votes.jsonl")
    store.record(model_left="a", model_right="b", vote=VOTE_LEFT)
    with store.path.open("a") as f:
        f.write('{"model_left": "a", "model_r')  # simulate a crash mid-write
    assert len(store.load()) == 1
