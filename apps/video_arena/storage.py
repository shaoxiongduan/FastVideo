# SPDX-License-Identifier: Apache-2.0
"""Append-only vote log plus the aggregations shown on the leaderboard tab.

Every vote is one JSON object on its own line, so the log survives crashes, can be
appended to from several worker processes, and is trivial to load later with
``pandas.read_json(path, lines=True)`` for offline analysis.
"""
from __future__ import annotations

import json
import math
import os
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Vote values written to the log. Keep these stable: analyses depend on them.
VOTE_LEFT = "left"
VOTE_RIGHT = "right"
VOTE_TIE = "tie"
VOTE_BOTH_BAD = "both_bad"
VOTES = (VOTE_LEFT, VOTE_RIGHT, VOTE_TIE, VOTE_BOTH_BAD)
# Score awarded to the left-hand model. Non-decisive verdicts are half a win each.
SCORES = {VOTE_LEFT: 1.0, VOTE_RIGHT: 0.0, VOTE_TIE: 0.5, VOTE_BOTH_BAD: 0.5}

# Ratings are reported on the familiar Elo scale: centred on 1000, and 400 points is a
# 10:1 odds ratio, so P(i beats j) = 1 / (1 + 10 ** (-(R_i - R_j) / 400)).
RATING_CENTER = 1000.0
RATING_SCALE = 400.0

# Virtual ties each model plays against a fixed anchor when fitting Bradley-Terry. Without
# it a model that has only ever won (or only ever lost) has an infinite MLE rating.
BT_REGULARIZATION = 1.0
BT_BOOTSTRAP = 200  # resamples used for the confidence interval
BT_SEED = 0  # fixed so the same log always yields the same interval


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _model_id(value: Any) -> str | None:
    """Normalize a model id from a log row, or None if the row doesn't carry one.

    Rows written by different versions of the app are read back into one DataFrame, so a
    field absent from some rows arrives as NaN rather than None — and ``not float('nan')``
    is False, which would otherwise let NaN through as if it were a model id.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def bradley_terry(battles: Sequence[tuple[str, str, float]],
                  models: Sequence[str],
                  reg: float = BT_REGULARIZATION,
                  weights: Sequence[float] | None = None,
                  iters: int = 500,
                  tol: float = 1e-8) -> dict[str, float]:
    """Fit Bradley-Terry strengths by MM, returned on the Elo scale.

    ``battles`` is ``(model_a, model_b, score_a)`` with ``score_a`` 1.0 / 0.0 / 0.5. The
    result depends only on the *multiset* of battles, not their order — which is the whole
    reason this is used instead of sequential Elo updates, whose output shifts by more than
    the effect being measured when the same votes arrive in a different order.

    ``reg`` gives every model that many virtual ties against a fixed anchor of strength 1,
    keeping the estimate finite for a model that has only ever won or only ever lost.
    ``weights`` counts each battle that many times, which is what lets the bootstrap
    resample distinct outcome cells instead of individual votes. ``tol`` is on the log
    strengths, so 1e-8 settles the reported rating far below its displayed 0.1 point.
    """
    if not models:
        return {}
    wins: dict[str, float] = {m: 0.0 for m in models}
    played: dict[tuple[str, str], float] = defaultdict(float)
    for i, (a, b, score_a) in enumerate(battles):
        w = 1.0 if weights is None else float(weights[i])
        if w == 0.0:
            continue
        wins[a] += score_a * w
        wins[b] += (1.0 - score_a) * w
        played[(a, b)] += w
        played[(b, a)] += w

    # None stands for the fixed anchor below, whose strength is pinned at 1.0.
    opponents: dict[str, list[tuple[str | None, float]]] = defaultdict(list)
    for (left, right), n in played.items():
        opponents[left].append((right, n))
    if reg > 0:  # virtual ties against the anchor
        for m in models:
            wins[m] += reg / 2.0
            opponents[m].append((None, reg))

    p = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {}
        for m in models:
            denom = sum(n / (p[m] + (1.0 if o is None else p[o])) for o, n in opponents[m])
            # Floor keeps log() finite for a model that never won, which is only reachable
            # with reg=0 -- the MLE there is genuinely -inf, so we report a large finite gap.
            new[m] = max(wins[m] / denom, 1e-12) if denom > 0 else p[m]
        shift = max(abs(math.log(new[m]) - math.log(p[m])) for m in models)
        p = new
        if shift < tol:
            break

    log_mean = sum(math.log(v) for v in p.values()) / len(p)
    return {m: RATING_CENTER + RATING_SCALE * (math.log(v) - log_mean) / math.log(10.0) for m, v in p.items()}


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

    def verdicts(self) -> list[tuple[str, str, str]]:
        """Every usable vote as ``(model_left, model_right, vote)``.

        Malformed, self-paired and unknown-verdict rows are dropped here, so the counts
        and the rating fit always see exactly the same set of battles.
        """
        out = []
        for _, r in self.load().iterrows():
            a, b = _model_id(r.get("model_left")), _model_id(r.get("model_right"))
            vote = r.get("vote")
            if a is None or b is None or a == b or vote not in VOTES:
                continue
            out.append((a, b, str(vote)))
        return out

    def battles(self) -> list[tuple[str, str, float]]:
        """``verdicts`` as ``(model_left, model_right, score_left)`` for the rating fit."""
        return [(a, b, SCORES[v]) for a, b, v in self.verdicts()]

    def leaderboard(self, model_names: dict[str, str] | None = None, n_boot: int = BT_BOOTSTRAP) -> pd.DataFrame:
        """Per-model counts, win rate, and a Bradley-Terry rating on the Elo scale.

        Win rate counts decisive votes only (wins / (wins + losses)). Both ``tie`` and
        ``both_bad`` score 0.5 for the rating but are reported in separate columns, since
        "both good" and "both bad" mean very different things for a checkpoint.

        ``ci95`` is the half-width of a bootstrap 95% interval on the rating. When two
        checkpoints' intervals overlap, the vote count is not yet enough to separate them.
        """
        cols = ["model", "name", "battles", "wins", "losses", "ties", "both_bad", "win_rate", "rating", "ci95"]
        df = self.load()
        if df.empty and not model_names:
            return pd.DataFrame(columns=cols)

        stats: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(("wins", "losses", "ties", "both_bad"), 0))
        # Seed every known model so a checkpoint nobody has rated yet still shows up.
        for known in (model_names or {}):
            _ = stats[known]

        verdicts = self.verdicts()
        for a, b, vote in verdicts:
            if vote == VOTE_LEFT:
                stats[a]["wins"] += 1
                stats[b]["losses"] += 1
            elif vote == VOTE_RIGHT:
                stats[b]["wins"] += 1
                stats[a]["losses"] += 1
            else:
                key = "ties" if vote == VOTE_TIE else "both_bad"
                stats[a][key] += 1
                stats[b][key] += 1

        battles = [(a, b, SCORES[v]) for a, b, v in verdicts]
        models = sorted(stats)
        rating = bradley_terry(battles, models)
        ci = self._bootstrap_ci(battles, models, n_boot)

        names = model_names or {}
        out = []
        for mid in models:
            s = stats[mid]
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
                "rating": round(rating.get(mid, RATING_CENTER), 1),
                "ci95": ci.get(mid),
            })
        return pd.DataFrame(out, columns=cols).sort_values("rating", ascending=False, ignore_index=True)

    @staticmethod
    def _bootstrap_ci(battles: Sequence[tuple[str, str, float]], models: Sequence[str],
                      n_boot: int) -> dict[str, float | None]:
        """Half-width of a 95% bootstrap interval per model, or None when it can't be formed.

        Resampling n votes with replacement is the same as drawing multinomial counts over
        the distinct (pair, outcome) cells, of which there are at most 3 per model pair. So
        each refit costs O(models^2) no matter how many votes the log holds — without this
        the leaderboard got steadily slower as raters worked.
        """
        if n_boot <= 0 or len(battles) < 2:
            return {m: None for m in models}
        cells = Counter(battles)
        keys = list(cells)
        counts = np.array([cells[k] for k in keys], dtype=float)
        n = int(counts.sum())

        rng = np.random.default_rng(BT_SEED)
        draws = {m: np.empty(n_boot) for m in models}
        for i in range(n_boot):
            fit = bradley_terry(keys, models, weights=rng.multinomial(n, counts / n))
            for m in models:
                draws[m][i] = fit[m]
        return {m: round(float(np.percentile(d, 97.5) - np.percentile(d, 2.5)) / 2.0, 1) for m, d in draws.items()}

    def pairwise(self) -> pd.DataFrame:
        """Head-to-head table: one row per unordered model pair."""
        if self.load().empty:
            return pd.DataFrame(columns=["model_a", "model_b", "a_wins", "b_wins", "ties", "both_bad", "n"])

        agg: dict[tuple[str, str],
                  dict[str, int]] = defaultdict(lambda: dict.fromkeys(("a_wins", "b_wins", "ties", "both_bad"), 0))
        for left, right, vote in self.verdicts():
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
