# SPDX-License-Identifier: Apache-2.0
"""Generate placeholder videos + a manifest so the arena UI can be run without a GPU.

Each mock "model" gets its own folder and its own visual signature (palette, motion
style, amount of frame-to-frame jitter) so that A/B differences are obvious while
rating, but nothing in the frames names the model — the mock exercises the real
anonymization path. Pass ``--label-models`` to stamp the model id on the frames when
you want to check that the post-vote reveal matches what was on screen.

    python -m apps.video_arena.make_mock_data --out apps/video_arena/mock
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

PROMPTS = [
    ("p001", "A red panda eating bamboo in a misty forest, shallow depth of field"),
    ("p002", "Timelapse of a city skyline at sunset, clouds racing past skyscrapers"),
    ("p003", "A paper boat drifting down a rain-soaked gutter, macro shot"),
    ("p004", "An astronaut planting a flag on a dusty orange planet, wide shot"),
    ("p005", "Close-up of coffee being poured into a glass cup, steam rising"),
    ("p006", "A neon-lit alley in the rain, reflections on wet asphalt, slow dolly in"),
]

# (name, base BGR-ish palette hue, motion style, jitter) — one entry per mock checkpoint.
STYLES = {
    "smooth": dict(hue=(60, 140, 220), wobble=0.0, jitter=0),
    "jittery": dict(hue=(220, 120, 60), wobble=0.35, jitter=6),
    "washed": dict(hue=(150, 150, 150), wobble=0.12, jitter=2),
}


def _frame(w: int, h: int, t: float, style: dict, prompt_id: str, seed: int, label: str | None) -> np.ndarray:
    rng = np.random.default_rng(seed + int(t * 1000))
    hue = np.array(style["hue"], dtype=np.float32)

    # Vertical gradient background tinted by the style palette.
    ramp = np.linspace(0.25, 1.0, h, dtype=np.float32)[:, None, None]
    img = np.broadcast_to(hue[None, None, :] * ramp, (h, w, 3)).astype(np.float32)
    if style["jitter"]:
        img = img + rng.normal(0.0, style["jitter"], img.shape).astype(np.float32)
    img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))

    # A ball on a circular path; "wobble" perturbs the path, "jitter" adds grain.
    wob = style["wobble"] * math.sin(t * 17.0)
    cx = int(w * (0.5 + 0.32 * math.cos(t * 2 * math.pi + wob)))
    cy = int(h * (0.5 + 0.28 * math.sin(t * 2 * math.pi + wob)))
    r = max(8, h // 12)
    cv2.circle(img, (cx, cy), r, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, tuple(int(c) for c in hue[::-1]), 3, lineType=cv2.LINE_AA)

    cv2.putText(img, prompt_id, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if label:  # debug only: normally nothing on screen identifies the model
        cv2.putText(img, label, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_video(path: Path, style: dict, prompt_id: str, seed: int, w: int, h: int, n_frames: int, fps: int,
                label: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 keeps the exact requested resolution (imageio defaults to yuv420p).
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=6, macro_block_size=1) as w_:
        for i in range(n_frames):
            w_.append_data(_frame(w, h, i / n_frames, style, prompt_id, seed, label))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="apps/video_arena/mock", help="output directory for videos + manifest")
    p.add_argument("--models",
                   nargs="+",
                   default=["h3-ckpt-1000", "h3-ckpt-4000"],
                   help="mock model ids; one folder is created per id")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--label-models", action="store_true", help="stamp the model id on frames (debugging only)")
    args = p.parse_args()

    out = Path(args.out).expanduser().resolve()
    style_names = list(STYLES)
    models = []
    for i, mid in enumerate(args.models):
        style = STYLES[style_names[i % len(style_names)]]
        vdir = out / "videos" / mid
        for j, (pid, _text) in enumerate(PROMPTS):
            write_video(vdir / f"{pid}.mp4",
                        style,
                        pid,
                        seed=1000 * i + j,
                        w=args.width,
                        h=args.height,
                        n_frames=args.frames,
                        fps=args.fps,
                        label=mid if args.label_models else None)
        models.append({
            "id": mid,
            "name": f"MiniMax H3 — {mid}",
            "dir": f"videos/{mid}",
            "notes": f"mock data, style={style_names[i % len(style_names)]}",
        })
        print(f"wrote {len(PROMPTS)} videos -> {vdir}")

    manifest = {
        "name": "MiniMax H3 checkpoint arena (mock data)",
        "description": "Placeholder videos generated by `make_mock_data.py` — replace with real samples.",
        "models": models,
        "prompts": [{
            "id": pid,
            "text": text
        } for pid, text in PROMPTS],
    }
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest -> {mpath}")


if __name__ == "__main__":
    main()
