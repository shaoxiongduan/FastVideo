# SPDX-License-Identifier: Apache-2.0
"""Append-only vote log plus the aggregations shown on the leaderboard tab.

Every vote is one JSON object on its own line, so the log survives crashes, can be
appended to from several worker processes, and is trivial to load later with
``pandas.read_json(path, lines=True)`` for offline analysis.
"""
from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Vote values written to the log. Keep these stable: analyses depend on them.
VOTE_LEFT = "left"
VOTE_RIGHT = "right"
VOTE_TIE = "tie"
VOTE_BOTH_BAD = "both_bad"
VOTES = (VOTE_LEFT, VOTE_RIGHT, VOTE_TIE, VOTE_BOTH_BAD)

ELO_START = 1000.0
ELO_K = 32.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VoteStore:
    """Thread-safe append-only JSONL writer + reader for arena votes."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, **fields: Any) -> dict[str, Any]:
        row = {"ts": utc_now(), **fields}
        line = json.dumps(row, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    def load(self) -> pd.DataFrame:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame(columns=["ts", "vote", "model_left", "model_right", "winner", "prompt_id", "voter"])
        rows = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # tolerate a torn final line
                    continue
        return pd.DataFrame(rows)

    # -- aggregation ---------------------------------------------------------

    def leaderboard(self, model_names: dict[str, str] | None = None) -> pd.DataFrame:
        """Per-model win/loss/tie counts, win rate, and a sequential Elo rating.

        Win rate counts decisive votes only (wins / (wins + losses)). Both ``tie`` and
        ``both_bad`` score 0.5 for Elo but are reported in separate columns, since
        "both good" and "both bad" mean very different things for a checkpoint.
        """
        df = self.load()
        cols = ["model", "name", "battles", "wins", "losses", "ties", "both_bad", "win_rate", "elo"]
        if df.empty:
            return pd.DataFrame(columns=cols)

        stats: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(("wins", "losses", "ties", "both_bad"), 0))
        elo: dict[str, float] = {}

        for _, r in df.sort_values("ts").iterrows():
            a, b, vote = r.get("model_left"), r.get("model_right"), r.get("vote")
            if not a or not b or vote not in VOTES:
                continue
            if vote == VOTE_LEFT:
                stats[a]["wins"] += 1
                stats[b]["losses"] += 1
                score_a = 1.0
            elif vote == VOTE_RIGHT:
                stats[b]["wins"] += 1
                stats[a]["losses"] += 1
                score_a = 0.0
            else:
                key = "ties" if vote == VOTE_TIE else "both_bad"
                stats[a][key] += 1
                stats[b][key] += 1
                score_a = 0.5

            ra, rb = elo.get(a, ELO_START), elo.get(b, ELO_START)
            expected_a = 1.0 / (1.0 + 10.0**((rb - ra) / 400.0))
            elo[a] = ra + ELO_K * (score_a - expected_a)
            elo[b] = rb + ELO_K * ((1.0 - score_a) - (1.0 - expected_a))

        names = model_names or {}
        out = []
        for mid, s in stats.items():
            decisive = s["wins"] + s["losses"]
            out.append({
                "model": mid,
                "name": names.get(mid, mid),
                "battles": sum(s.values()),
                "wins": s["wins"],
                "losses": s["losses"],
                "ties": s["ties"],
                "both_bad": s["both_bad"],
                "win_rate": round(s["wins"] / decisive, 3) if decisive else None,
                "elo": round(elo.get(mid, ELO_START), 1),
            })
        return pd.DataFrame(out, columns=cols).sort_values("elo", ascending=False, ignore_index=True)

    def pairwise(self) -> pd.DataFrame:
        """Head-to-head table: one row per unordered model pair."""
        df = self.load()
        if df.empty:
            return pd.DataFrame(columns=["model_a", "model_b", "a_wins", "b_wins", "ties", "both_bad", "n"])

        agg: dict[tuple[str, str],
                  dict[str, int]] = defaultdict(lambda: dict.fromkeys(("a_wins", "b_wins", "ties", "both_bad"), 0))
        for _, r in df.iterrows():
            left, right, vote = r.get("model_left"), r.get("model_right"), r.get("vote")
            if not left or not right or vote not in VOTES:
                continue
            a, b = sorted((left, right))  # canonical order so A|B and B|A merge
            winner = left if vote == VOTE_LEFT else right if vote == VOTE_RIGHT else None
            if winner == a:
                agg[(a, b)]["a_wins"] += 1
            elif winner == b:
                agg[(a, b)]["b_wins"] += 1
            else:
                agg[(a, b)]["ties" if vote == VOTE_TIE else "both_bad"] += 1

        rows = [{"model_a": a, "model_b": b, **c, "n": sum(c.values())} for (a, b), c in agg.items()]
        return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)

    def export_csv(self, out_path: str | os.PathLike) -> Path:
        out = Path(out_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.load().to_csv(out, index=False)
        return out
