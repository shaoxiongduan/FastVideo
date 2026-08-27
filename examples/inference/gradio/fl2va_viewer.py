# SPDX-License-Identifier: Apache-2.0
"""Browse FL2VA results: the two conditioning frames, then every checkpoint's output.

FL2VA is only interesting next to what it was conditioned on, so this puts the first and
last frames at the top and every checkpoint's clip underneath, each labelled with how far
its own endpoints landed from those frames. A working run reproduces both; a broken one
drifts, and the numbers say by how much without needing to trust an eyeball.

Expects the layout the FL2VA runner writes:

    <runs>/<checkpoint-tag>/<index>_<case_id>.mp4
    <frames>/<index>_<case_id>_{first,last}.png  +  index.json

    python examples/inference/gradio/fl2va_viewer.py --port 7864 --share
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

PER_ROW = 3


def endpoint_error(mp4: Path, first: Path, last: Path) -> tuple[float, float, float]:
    """(first-frame error, last-frame error, distance between the conditioning frames).

    The third number is the scale: it is how far apart the two conditioning frames are, so
    an error much smaller than it means the endpoint was actually reproduced.
    """
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vf", "select=eq(n\\,0)",
                    "-frames:v", "1", str(tmp / "gf.png")], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.4", "-i", str(mp4), "-update", "1",
                    str(tmp / "gl.png")], check=True)
    cf_img = Image.open(first).convert("RGB")
    size = cf_img.size

    def arr(p: Path) -> np.ndarray:
        im = Image.open(p).convert("RGB")
        if im.size != size:
            im = im.resize(size)
        return np.asarray(im, dtype=np.float32)

    cf = np.asarray(cf_img, dtype=np.float32)
    cl = arr(last)
    return (float(np.abs(arr(tmp / "gf.png") - cf).mean()),
            float(np.abs(arr(tmp / "gl.png") - cl).mean()),
            float(np.abs(cf - cl).mean()))


class Runs:
    """The FL2VA runs on disk: which checkpoints exist, and which prompts each rendered."""

    def __init__(self, runs_dir: Path, frames_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.frames_dir = frames_dir
        self.index = json.loads((frames_dir / "index.json").read_text())
        self.tags = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
        # prompt index -> {tag: mp4}
        self.videos: dict[str, dict[str, Path]] = {}
        for idx in self.index:
            found = {}
            for tag in self.tags:
                hit = next(iter((runs_dir / tag).glob(f"{int(idx):03d}_*.mp4")), None)
                if hit is not None:
                    found[tag] = hit
            if found:
                self.videos[idx] = found
        if not self.videos:
            raise SystemExit(f"no rendered clips under {runs_dir}")

    def label(self, idx: str) -> str:
        d = self.index[idx]
        head = " ".join(d["prompt"].split())[:80]
        return f"[{int(idx):03d}] {d['case_id']} — {head}..."

    def by_label(self, label: str) -> str:
        return next(i for i in self.videos if self.label(i) == label)


def build(runs: Runs, video_height: int = 380) -> gr.Blocks:

    @lru_cache(maxsize=256)
    def cached_error(mp4: str, first: str, last: str) -> tuple[float, float, float]:
        return endpoint_error(Path(mp4), Path(first), Path(last))

    def show(label: str):
        idx = runs.by_label(label)
        meta = runs.index[idx]
        first, last = Path(meta["first_frame"]), Path(meta["last_frame"])

        players = []
        for tag in runs.tags:
            mp4 = runs.videos[idx].get(tag)
            if mp4 is None:
                players.append(gr.update(value=None, label=f"{tag} — not rendered"))
                continue
            ef, el, scale = cached_error(str(mp4), str(first), str(last))
            players.append(
                gr.update(value=str(mp4),
                          label=f"{tag} · first {ef:.2f} / last {el:.2f} (scale {scale:.1f})"))

        return [str(first), str(last), gr.update(value=meta["prompt"], label=f"Prompt — {meta['case_id']}"),
                *players]

    with gr.Blocks(title="FL2VA results") as demo:
        gr.Markdown("# FL2VA results\n"
                    "Top row is what each clip was conditioned on. Each player's label reports how far its "
                    "own first/last frame landed from those images (mean absolute pixel difference), next to "
                    "**scale** — how far the two conditioning frames are from each other. Error far below "
                    "scale means the endpoint was reproduced.")

        prompt_dd = gr.Dropdown(choices=[runs.label(i) for i in runs.videos],
                                value=runs.label(next(iter(runs.videos))), label="Prompt")

        with gr.Row():
            first_img = gr.Image(label="conditioning: FIRST frame", height=video_height, interactive=False)
            last_img = gr.Image(label="conditioning: LAST frame", height=video_height, interactive=False)

        prompt_box = gr.Textbox(label="Prompt", lines=5, max_lines=5, interactive=False, show_copy_button=True)

        players = []
        for start in range(0, len(runs.tags), PER_ROW):
            with gr.Row():
                for tag in runs.tags[start:start + PER_ROW]:
                    players.append(gr.Video(label=tag, height=video_height, loop=True, autoplay=False,
                                            show_download_button=True, interactive=False))

        outputs = [first_img, last_img, prompt_box, *players]
        gr.on(triggers=[prompt_dd.change, demo.load], fn=show, inputs=prompt_dd, outputs=outputs)

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="/mnt/lustre/vlm-s4duan/arena_arms/fl2va_test")
    p.add_argument("--frames", default="/mnt/lustre/vlm-s4duan/arena_arms/fl2va_from_fal")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7864)
    p.add_argument("--share", action="store_true")
    p.add_argument("--video-height", type=int, default=380)
    args = p.parse_args()

    runs = Runs(Path(args.runs).resolve(), Path(args.frames).resolve())
    print(f"checkpoints: {runs.tags}")
    print(f"prompts with output: {sorted(int(i) for i in runs.videos)}")

    build(runs, video_height=args.video_height).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        allowed_paths=[str(runs.runs_dir), str(runs.frames_dir)])


if __name__ == "__main__":
    main()
