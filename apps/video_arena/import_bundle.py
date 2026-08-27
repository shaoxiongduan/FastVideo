# SPDX-License-Identifier: Apache-2.0
"""Turn a published validation voting bundle into an arena manifest.

A bundle is a directory laid out as::

    <bundle>/
      manifest.jsonl        one row per held-out record: index, sample_id, prompt, arms{}
      arms.json             per-arm provenance: slug, display_name, checkpoint_step, ...
      arms/<slug>/videos/<index>_<sample_id>.mp4

Arms are discovered from the directory listing rather than read from ``arms.json``, so
dropping a new ``arms/<slug>/videos/`` folder in and re-running this picks the new
checkpoint up. Any arm missing from ``arms.json`` still works — it just gets its slug as
its display name.

    python -m apps.video_arena.import_bundle \\
        --bundle /mnt/lustre/vlm-shared/h3_fasth3_validation_voting/fast_h3_v1_family_heldout60 \\
        --out apps/video_arena/arenas/fasth3_v1_family.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from apps.video_arena.arena import VIDEO_EXTS

logger = logging.getLogger(__name__)


def discover_arms(bundle: Path) -> list[str]:
    """Every ``arms/<slug>/videos`` directory that actually holds videos."""
    root = bundle / "arms"
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist — is {bundle} really a bundle?")
    out = []
    for d in sorted(root.iterdir()):
        videos = d / "videos"
        if videos.is_dir() and any(p.suffix.lower() in VIDEO_EXTS for p in videos.iterdir()):
            out.append(d.name)
    return out


def load_arm_meta(bundle: Path) -> dict[str, dict]:
    path = bundle / "arms.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    entries = raw.get("arms", raw) if isinstance(raw, dict) else raw
    if isinstance(entries, dict):  # {slug: {...}}
        return {str(k): v for k, v in entries.items()}
    return {str(e.get("slug", e.get("name", i))): e for i, e in enumerate(entries)}


def load_prompts(bundle: Path, arms: list[str]) -> list[dict]:
    """Prompt text keyed by the video filename stem, which is shared across arms."""
    path = bundle / "manifest.jsonl"
    if not path.exists():
        logger.warning("%s has no manifest.jsonl — prompt ids will fall back to filenames", bundle)
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # Prefer the path this bundle recorded for an arm we actually have on disk.
        stem = None
        for arm in arms:
            rel = (row.get("arms") or {}).get(arm, {}).get("path")
            if rel:
                stem = Path(rel).stem
                break
        if stem is None:
            idx, sid = row.get("index"), row.get("sample_id")
            stem = f"{idx:03d}_{sid}" if idx is not None and sid else str(sid or idx)
        out.append({"id": stem, "text": row.get("prompt", "")})
    return out


def describe(slug: str, meta: dict) -> tuple[str, str]:
    name = str(meta.get("display_name") or meta.get("model_id") or slug)
    bits = []
    if meta.get("checkpoint_step") is not None:
        bits.append(f"step {meta['checkpoint_step']}")
    if meta.get("attention_backend"):
        bits.append(str(meta["attention_backend"]))
    if meta.get("model_id"):
        bits.append(str(meta["model_id"]))
    return name, " · ".join(bits)


def build(bundle: Path, arms: list[str] | None, name: str | None) -> dict:
    arms = arms or discover_arms(bundle)
    if len(arms) < 2:
        raise SystemExit(f"need at least 2 arms to run an arena, found {arms}")
    meta = load_arm_meta(bundle)
    prompts = load_prompts(bundle, arms)

    models = []
    for slug in arms:
        display, notes = describe(slug, meta.get(slug, {}))
        models.append({
            "id": slug,
            "name": display,
            "dir": str((bundle / "arms" / slug / "videos").resolve()),
            "notes": notes,
        })

    readme = bundle / "README.md"
    caveat = ""
    if readme.exists():
        for para in readme.read_text().split("\n\n"):
            if "caveat" in para.lower():
                caveat = " ".join(para.split())
                break

    return {
        "name": name or f"{bundle.name} arena",
        "description": caveat,
        "models": models,
        "prompts": prompts,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, help="published voting bundle directory")
    p.add_argument("--out", required=True, help="arena manifest to write")
    p.add_argument("--arms", nargs="+", default=None, help="restrict to these arm slugs (default: all found)")
    p.add_argument("--name", default=None, help="arena title shown in the UI")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    bundle = Path(args.bundle).expanduser().resolve()
    manifest = build(bundle, args.arms, args.name)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(manifest['models'])} arms x {len(manifest['prompts'])} prompts -> {out}")
    for m in manifest["models"]:
        n = sum(1 for p_ in Path(m["dir"]).iterdir() if p_.suffix.lower() in VIDEO_EXTS)
        print(f"  {m['id']:<10} {n:>4} videos  {m['name']}")


if __name__ == "__main__":
    main()
