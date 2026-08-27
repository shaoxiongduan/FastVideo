# SPDX-License-Identifier: Apache-2.0
"""Publish a generated arm into a shared voting bundle, in the h3-validation-voting layout.

    <bundle>/
      README.md
      arms.json
      manifest.jsonl
      arms/<slug>/videos/<index>_<sample_id>.mp4

Filenames are normalized to ``<index>_<sample_id>.mp4`` so every arm lines up on the same
stem and a consumer needs only one matching rule. Files are written world-readable on
purpose: the original bundle shipped its videos mode 0600, which left everyone but the
owner able to stat them but not open them, and the arena served 200s with zero bytes.

    python scripts/checkpoint_conversion/publish_arena_arm.py \\
        --bundle /mnt/lustre/vlm-shared/h3_arena_extra_arms_heldout60 \\
        --slug lightx2v_turbo_4step --name "lightx2v Turbo 4-step" \\
        --videos /mnt/lustre/vlm-s4duan/arena_arms/lightx2v_turbo_4step \\
        --notes "minimax_h3_fl2v_turbo_4step_v1.1_768p, 4 steps, 124 frames"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")
FILE_MODE = 0o644
DIR_MODE = 0o755


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_source(src: Path, cases: dict[str, int]) -> dict[int, Path]:
    """Map prompt index -> source file, accepting either naming convention.

    Generated arms are already ``<index>_<sample_id>.mp4``; arms pulled from elsewhere
    (a HF dataset, a vendor) are usually just ``<sample_id>.mp4``.
    """
    out: dict[int, Path] = {}
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        stem = p.stem
        idx = None
        if "_" in stem and stem.split("_", 1)[0].isdigit():
            head, rest = stem.split("_", 1)
            if rest in cases:
                idx = int(head)
        if idx is None and stem in cases:
            idx = cases[stem]
        if idx is None:
            print(f"  ! no prompt matches {p.name}, skipping")
            continue
        if idx in out:
            raise SystemExit(f"two files claim prompt {idx}: {out[idx].name} and {p.name}")
        out[idx] = p
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--videos", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--prompts", default="/mnt/lustre/vlm-s4duan/FastVideo/prompts.jsonl")
    p.add_argument("--link", action="store_true", help="hardlink instead of copying (same filesystem only)")
    args = p.parse_args()

    rows = [json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()]
    cases = {r["case_id"]: i for i, r in enumerate(rows)}

    bundle = Path(args.bundle).expanduser().resolve()
    dest = bundle / "arms" / args.slug / "videos"
    dest.mkdir(parents=True, exist_ok=True)

    src = Path(args.videos).expanduser().resolve()
    found = index_source(src, cases)
    print(f"{args.slug}: {len(found)}/{len(rows)} prompts found in {src}")

    published = {}
    for idx, path in sorted(found.items()):
        name = f"{idx:03d}_{rows[idx]['case_id']}{path.suffix.lower()}"
        target = dest / name
        if target.exists():
            target.unlink()
        if args.link:
            os.link(path, target)
        else:
            shutil.copy2(path, target)
        os.chmod(target, FILE_MODE)
        published[idx] = {"path": f"arms/{args.slug}/videos/{name}", "bytes": target.stat().st_size,
                          "sha256": sha256(target)}

    for d in (bundle, bundle / "arms", bundle / "arms" / args.slug, dest):
        os.chmod(d, DIR_MODE)

    # arms.json: append or replace this arm
    arms_path = bundle / "arms.json"
    arms = json.loads(arms_path.read_text()) if arms_path.exists() else {"schema_version": "h3-arena-arms-v1",
                                                                        "arms": []}
    arms["arms"] = [a for a in arms["arms"] if a.get("slug") != args.slug]
    arms["arms"].append({"slug": args.slug, "display_name": args.name, "notes": args.notes,
                         "published_file_count": len(published)})
    arms["arms"].sort(key=lambda a: a["slug"])
    arms["record_count"] = len(rows)
    arms_path.write_text(json.dumps(arms, indent=2) + "\n")
    os.chmod(arms_path, FILE_MODE)

    # manifest.jsonl: one row per prompt, arms merged in
    man_path = bundle / "manifest.jsonl"
    existing = {}
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["index"]] = row
    for idx, row in enumerate(rows):
        rec = existing.setdefault(idx, {"index": idx, "sample_id": row["case_id"], "prompt": row["prompt"],
                                        "arms": {}})
        rec["arms"].pop(args.slug, None)
        if idx in published:
            rec["arms"][args.slug] = published[idx]
    man_path.write_text("".join(json.dumps(existing[i], ensure_ascii=False) + "\n" for i in sorted(existing)))
    os.chmod(man_path, FILE_MODE)

    print(f"published {len(published)} videos -> {dest}")
    print(f"updated {arms_path.name} ({len(arms['arms'])} arms) and {man_path.name}")


if __name__ == "__main__":
    main()
