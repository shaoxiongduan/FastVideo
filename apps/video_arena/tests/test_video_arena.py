# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the video arena: manifest loading, pairing, voting, aggregation.

Run with:  pytest apps/video_arena/tests -v
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from apps.video_arena.app import RANDOM_PROMPT, ArenaUI
from apps.video_arena.arena import Arena
from apps.video_arena.storage import (VOTE_BOTH_BAD, VOTE_LEFT, VOTE_RIGHT, VOTE_TIE, VoteStore,
                                      bradley_terry)

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


def test_players_are_anonymous_until_a_vote_then_carry_the_model_name(manifest: Path, tmp_path: Path) -> None:
    arena = Arena(manifest, anonymize_paths=False)
    ui = ArenaUI(arena, VoteStore(tmp_path / "v.jsonl"), random.Random(1))

    started = ui.new_battle(RANDOM_PROMPT, 0)
    state = started[0]
    battle = state["battle"]
    assert (started[1]["label"], started[2]["label"]) == ("A", "B"), "identities hidden before voting"

    out = ui.cast_vote(VOTE_RIGHT, state, "s", 0)
    assert out[0]["battle"].battle_id == battle.battle_id, "vote must not advance on its own"
    assert out[1]["label"] == battle.left.display, "left player relabelled with its checkpoint"
    assert out[2]["label"] == battle.right.display
    assert "value" not in out[1], "relabelling must not reload the video"
    # Vote buttons go dead, "next round" wakes up.
    assert all(u.get("interactive") is False for u in out[4:8])
    assert out[8].get("interactive") is True
    assert out[10] == 1, "rated counter increments"


def test_next_round_re_anonymizes_and_re_enables_voting(manifest: Path, tmp_path: Path) -> None:
    ui = ArenaUI(Arena(manifest, anonymize_paths=False), VoteStore(tmp_path / "v.jsonl"), random.Random(1))
    state = ui.new_battle(RANDOM_PROMPT, 0)[0]
    ui.cast_vote(VOTE_RIGHT, state, "s", 0)

    out = ui.next_round(1)
    assert out[0]["battle"].battle_id != state["battle"].battle_id
    assert (out[1]["label"], out[2]["label"]) == ("A", "B"), "labels hidden again"
    assert all(u.get("interactive") is True for u in out[4:8])
    assert out[8].get("interactive") is False


def test_next_round_returns_the_prompt_selector_to_random(manifest: Path, tmp_path: Path) -> None:
    arena = Arena(manifest, anonymize_paths=False)
    ui = ArenaUI(arena, VoteStore(tmp_path / "v.jsonl"), random.Random(3))

    pinned = next(p.label for p in arena.prompts if p.id == "p002")
    assert ui.new_battle(pinned, 0)[0]["battle"].prompt.id == "p002"

    out = ui.next_round(1)
    assert out[-1]["value"] == RANDOM_PROMPT, "dropdown resets so the next round is random"
    # Over many rounds the pinned prompt must not dominate.
    seen = {ui.next_round(0)[0]["battle"].prompt.id for _ in range(60)}
    assert len(seen) > 1


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

    lb = store.leaderboard({m.id: m.name for m in arena.models}, n_boot=0)
    assert lb.iloc[0]["model"] == winner
    assert lb.iloc[0]["win_rate"] == 1.0
    assert lb.iloc[0]["rating"] > 1000 > lb.iloc[-1]["rating"]

    pw = store.pairwise()
    assert len(pw) == 3  # one row per unordered pair, both orderings merged
    assert pw["n"].sum() == 120


def test_ties_and_both_bad_are_counted_separately(manifest: Path, tmp_path: Path) -> None:
    store = VoteStore(tmp_path / "votes.jsonl")
    for vote in (VOTE_TIE, VOTE_TIE, VOTE_BOTH_BAD):
        store.record(model_left="ckpt-a", model_right="ckpt-b", vote=vote)
    lb = store.leaderboard(n_boot=0).set_index("model")
    assert lb.loc["ckpt-a", "ties"] == 2
    assert lb.loc["ckpt-a", "both_bad"] == 1
    assert pd.isna(lb.loc["ckpt-a", "win_rate"])  # no decisive votes -> undefined
    # Two evenly-matched models with only draws between them must not separate.
    assert lb.loc["ckpt-a", "rating"] == pytest.approx(1000.0)
    assert lb.loc["ckpt-b", "rating"] == pytest.approx(1000.0)


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


# -- rating math -------------------------------------------------------------


def _log(tmp_path: Path, results: list[tuple[str, str, str]]) -> VoteStore:
    store = VoteStore(tmp_path / "votes.jsonl")
    for left, right, vote in results:
        store.record(model_left=left, model_right=right, vote=vote)
    return store


def test_rating_is_independent_of_vote_order(tmp_path: Path) -> None:
    """The whole reason for Bradley-Terry over sequential Elo: reshuffling must not matter."""
    rng = random.Random(5)
    rows = [(*rng.sample(["a", "b", "c"], 2), rng.choice([VOTE_LEFT, VOTE_RIGHT, VOTE_TIE])) for _ in range(150)]

    baseline = _log(tmp_path / "base", rows).leaderboard(n_boot=0).set_index("model")["rating"].to_dict()
    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        got = _log(tmp_path / f"s{seed}", shuffled).leaderboard(n_boot=0).set_index("model")["rating"].to_dict()
        assert got == baseline, f"order changed the ratings (seed {seed})"


def test_unregularized_two_model_fit_matches_the_closed_form(tmp_path: Path) -> None:
    """With two models and no regularization, BT has an exact answer: p_a/p_b == wins_a/wins_b."""
    rows = [("a", "b", VOTE_LEFT)] * 8 + [("a", "b", VOTE_RIGHT)] * 6
    fit = bradley_terry([(a, b, 1.0 if v == VOTE_LEFT else 0.0) for a, b, v in rows], ["a", "b"], reg=0.0)
    assert fit["a"] - fit["b"] == pytest.approx(400 * math.log10(8 / 6), abs=0.05)


def test_regularization_pulls_an_undefeated_model_back_from_infinity(tmp_path: Path) -> None:
    """A model that has only ever won has an infinite MLE; reg must keep it finite and sane."""
    battles = [("a", "b", 1.0)] * 10
    light = bradley_terry(battles, ["a", "b"], reg=0.5)
    heavy = bradley_terry(battles, ["a", "b"], reg=5.0)
    for fit in (light, heavy):
        assert math.isfinite(fit["a"]) and math.isfinite(fit["b"])
    assert light["a"] - light["b"] > heavy["a"] - heavy["b"] > 0, "more reg -> more shrinkage"
    assert heavy["a"] - heavy["b"] < 800


def test_rating_recovers_a_known_ordering(tmp_path: Path) -> None:
    strengths = {"strong": 0.8, "mid": 0.5, "weak": 0.2}
    rng = random.Random(4)
    rows = []
    for _ in range(900):
        x, y = rng.sample(list(strengths), 2)
        p_x = strengths[x] / (strengths[x] + strengths[y])
        rows.append((x, y, VOTE_LEFT if rng.random() < p_x else VOTE_RIGHT))
    lb = _log(tmp_path, rows).leaderboard(n_boot=0)
    assert list(lb["model"]) == ["strong", "mid", "weak"]


def test_confidence_interval_shrinks_as_votes_accumulate(tmp_path: Path) -> None:
    rng = random.Random(9)
    rows = [("a", "b", VOTE_LEFT if rng.random() < 0.65 else VOTE_RIGHT) for _ in range(400)]
    small = _log(tmp_path / "small", rows[:25]).leaderboard(n_boot=60).set_index("model")
    large = _log(tmp_path / "large", rows).leaderboard(n_boot=60).set_index("model")
    assert large.loc["a", "ci95"] < small.loc["a", "ci95"], "more votes -> tighter interval"
    # With 400 votes the 65/35 gap should clear its own error bar; with 25 it should not.
    assert large.loc["a", "ci95"] < abs(large.loc["a", "rating"] - large.loc["b", "rating"])
    assert small.loc["a", "ci95"] > large.loc["a", "ci95"] * 2
