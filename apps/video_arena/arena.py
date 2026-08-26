# SPDX-License-Identifier: Apache-2.0
"""Manifest loading and battle sampling for the side-by-side video arena.

An *arena* is a set of models (each a folder of videos) that were all run on the
same set of prompts. A *battle* pairs two randomly chosen models on one prompt and
randomly decides which one is shown on the left, so the rater cannot infer identity
from position.
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".avi")

# Used when a caller does not supply its own generator (pass one to make runs reproducible).
_DEFAULT_RNG = random.Random()


@dataclass
class ModelEntry:
    """One competitor: a named model/checkpoint plus the folder holding its videos."""

    id: str
    name: str
    video_dir: Path
    notes: str = ""

    @property
    def display(self) -> str:
        """Plain-text label (safe in both markdown and component labels).

        The id is appended only when the name doesn't already carry it, so a name like
        "MiniMax H3 — h3-ckpt-4000" isn't rendered as "... (h3-ckpt-4000)" twice.
        """
        return self.name if self.id in self.name else f"{self.name} ({self.id})"


@dataclass
class PromptEntry:
    id: str
    text: str
    # Per-model filename overrides: {model_id: "some_name.mp4"}.
    files: dict[str, str] = field(default_factory=dict)
    # Default filename tried in every model dir when there is no per-model override.
    file: str | None = None

    @property
    def label(self) -> str:
        head = " ".join(self.text.split())
        if len(head) > 90:
            head = head[:87] + "..."
        return f"[{self.id}] {head}"


@dataclass
class Battle:
    """A single side-by-side comparison presented to a rater."""

    battle_id: str
    prompt: PromptEntry
    left: ModelEntry
    right: ModelEntry
    left_video: Path
    right_video: Path
    # Paths actually handed to the UI (anonymized copies when enabled).
    left_serve: Path
    right_serve: Path


def _is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def _index_dir(d: Path) -> dict[str, Path]:
    """Map filename-stem -> path for every video directly inside ``d``."""
    if not d.is_dir():
        logger.warning("video dir does not exist: %s", d)
        return {}
    return {p.stem: p for p in sorted(d.iterdir()) if p.is_file() and _is_video(p)}


def _load_side_prompts(manifest_dir: Path) -> dict[str, str]:
    """Optional prompt-text sidecar, used when the manifest itself lists no prompts."""
    for name in ("prompts.json", "prompts.txt"):
        p = manifest_dir / name
        if not p.exists():
            continue
        if p.suffix == ".json":
            raw = json.loads(p.read_text())
            if isinstance(raw, dict):
                return {str(k): str(v) for k, v in raw.items()}
            if isinstance(raw, list):  # [{"id":..., "text":...}, ...] or ["a", "b", ...]
                out = {}
                for i, row in enumerate(raw):
                    if isinstance(row, dict):
                        out[str(row.get("id", i))] = str(row.get("text", row.get("prompt", "")))
                    else:
                        out[str(i)] = str(row)
                return out
        else:
            lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            return {str(i): ln for i, ln in enumerate(lines)}
    return {}


class Arena:
    """Holds the model roster, the prompt list, and the prompt x model video grid."""

    def __init__(self, manifest_path: str | os.PathLike, anonymize_paths: bool = True) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        raw = json.loads(self.manifest_path.read_text())
        root = self.manifest_path.parent

        self.name: str = raw.get("name", "Video Arena")
        self.description: str = raw.get("description", "")

        self.models: list[ModelEntry] = []
        for m in raw["models"]:
            mid = str(m["id"])
            d = Path(m.get("dir") or m["video_dir"]).expanduser()
            self.models.append(
                ModelEntry(id=mid,
                           name=str(m.get("name", mid)),
                           video_dir=d if d.is_absolute() else (root / d),
                           notes=str(m.get("notes", ""))))
        if len(self.models) < 2:
            raise ValueError(f"{self.manifest_path}: need at least 2 models, got {len(self.models)}")

        self._dir_index = {m.id: _index_dir(m.video_dir) for m in self.models}
        self.prompts = self._build_prompts(raw, root)
        self.videos = self._build_grid()

        # A prompt with fewer than two videos cannot form a battle.
        self.prompts = [p for p in self.prompts if len(self.videos.get(p.id, {})) >= 2]
        if not self.prompts:
            raise ValueError(f"{self.manifest_path}: no prompt has videos from >=2 models; "
                             "check the `dir` paths and the video filenames")

        self.anonymize_paths = anonymize_paths
        self._anon_root = Path(tempfile.mkdtemp(prefix="video_arena_anon_")) if anonymize_paths else None

    # -- construction helpers ------------------------------------------------

    def _build_prompts(self, raw: dict, root: Path) -> list[PromptEntry]:
        if raw.get("prompts"):
            out = []
            for i, p in enumerate(raw["prompts"]):
                if isinstance(p, str):
                    out.append(PromptEntry(id=str(i), text=p))
                else:
                    out.append(
                        PromptEntry(id=str(p.get("id", i)),
                                    text=str(p.get("text", p.get("prompt", ""))),
                                    files={
                                        str(k): str(v)
                                        for k, v in (p.get("files") or {}).items()
                                    },
                                    file=p.get("file")))
            return out

        # No explicit prompt list: derive ids from the video filenames themselves.
        stems: set[str] = set()
        for idx in self._dir_index.values():
            stems |= set(idx)
        texts = _load_side_prompts(root)
        return [PromptEntry(id=s, text=texts.get(s, s)) for s in sorted(stems)]

    def _build_grid(self) -> dict[str, dict[str, Path]]:
        grid: dict[str, dict[str, Path]] = {}
        for prompt in self.prompts:
            row = {m.id: path for m in self.models if (path := self._locate(prompt, m)) is not None}
            if row:
                grid[prompt.id] = row
        return grid

    def _locate(self, prompt: PromptEntry, model: ModelEntry) -> Path | None:
        override = prompt.files.get(model.id) or prompt.file
        if override:
            cand = Path(override)
            cand = cand if cand.is_absolute() else model.video_dir / cand
            if cand.exists():
                return cand
            # An override may name a stem rather than a full filename.
            return self._dir_index[model.id].get(cand.stem)
        return self._dir_index[model.id].get(prompt.id)

    # -- battle sampling -----------------------------------------------------

    def prompt_choices(self) -> list[str]:
        return [p.label for p in self.prompts]

    def prompt_by_label(self, label: str | None) -> PromptEntry | None:
        if not label:
            return None
        return next((p for p in self.prompts if p.label == label), None)

    def sample_battle(self, prompt_label: str | None = None, rng: random.Random | None = None) -> Battle:
        rng = rng or _DEFAULT_RNG
        prompt = self.prompt_by_label(prompt_label) or rng.choice(self.prompts)

        available = [m for m in self.models if m.id in self.videos[prompt.id]]
        # rng.sample also decides which of the pair lands on the left.
        left, right = rng.sample(available, 2)
        lv, rv = self.videos[prompt.id][left.id], self.videos[prompt.id][right.id]
        bid = uuid.uuid4().hex[:12]
        return Battle(battle_id=bid,
                      prompt=prompt,
                      left=left,
                      right=right,
                      left_video=lv,
                      right_video=rv,
                      left_serve=self._serve_path(bid, "L", lv),
                      right_serve=self._serve_path(bid, "R", rv))

    def _serve_path(self, battle_id: str, side: str, src: Path) -> Path:
        """Hide the real path from the browser so the model cannot be read off the video URL."""
        if self._anon_root is None:
            return src
        dst = self._anon_root / f"{battle_id}_{side}{src.suffix.lower()}"
        if not dst.exists():
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        return dst

    @property
    def serve_roots(self) -> list[str]:
        """Directories gradio must be allowed to serve files from."""
        roots = {str(m.video_dir.resolve()) for m in self.models}
        if self._anon_root is not None:
            roots.add(str(self._anon_root))
        return sorted(roots)

    def coverage(self) -> str:
        lines = [f"**{self.name}** — {len(self.models)} models × {len(self.prompts)} prompts"]
        if self.description:
            lines.append(f"\n{self.description}\n")
        for m in self.models:
            n = sum(1 for p in self.prompts if m.id in self.videos[p.id])
            note = f" — {m.notes}" if m.notes else ""
            lines.append(f"- `{m.id}` — {n}/{len(self.prompts)} videos — {m.name}{note}")
        return "\n".join(lines)
