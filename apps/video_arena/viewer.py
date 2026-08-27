# SPDX-License-Identifier: Apache-2.0
"""Side-by-side viewer: one prompt, every model's video at once, names shown.

The opposite of the arena. The arena hides identities and asks for one verdict at a time;
this shows the whole row labelled, for looking at a prompt across every checkpoint and
spotting patterns. Nothing is recorded — use the arena for anything that needs a vote.

    python -m apps.video_arena.viewer --manifest apps/video_arena/arenas/fasth3_v1_family.json

Reads the same manifest as the arena, so a newly imported arm shows up here with no extra
work.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import gradio as gr
from starlette.middleware import Middleware

from apps.video_arena.app import VideoCacheHeaders
from apps.video_arena.arena import Arena

logger = logging.getLogger(__name__)

PER_ROW = 3

# Six players with six audio tracks at once is unusable, so play-all forces mute; un-mute
# a single player by hand to actually listen to it.
PLAY_ALL_JS = """() => {
  document.querySelectorAll('video').forEach(v => { v.muted = true; v.currentTime = 0; v.play(); });
}"""
PAUSE_ALL_JS = """() => { document.querySelectorAll('video').forEach(v => v.pause()); }"""
REWIND_JS = """() => { document.querySelectorAll('video').forEach(v => { v.currentTime = 0; }); }"""


def describe(path: Path) -> str:
    """frames / duration / size, so length differences between arms are visible, not guessed."""
    try:
        out = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
            "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ],
                             capture_output=True,
                             text=True,
                             timeout=20).stdout.split()
        w, h, nf, dur = out[0], out[1], out[2], float(out[3])
        return f"{w}x{h} · {nf}f · {dur:.2f}s · {path.stat().st_size / 2**20:.1f} MiB"
    except Exception:  # noqa: BLE001 - a probe failure must not blank the page
        return f"{path.stat().st_size / 2**20:.1f} MiB"


def build_viewer(arena: Arena, video_height: int = 420) -> gr.Blocks:

    def show(prompt_label: str):
        prompt = arena.prompt_by_label(prompt_label) or arena.prompts[0]
        updates = []
        for model in arena.models:
            path = arena.videos[prompt.id].get(model.id)
            if path is None:
                updates.append(gr.update(value=None, label=f"{model.display} — MISSING"))
            else:
                updates.append(gr.update(value=str(path), label=f"{model.display} · {describe(path)}"))
        return [gr.update(value=prompt.text, label=f"Prompt — {prompt.id}"), *updates]

    def step(prompt_label: str, delta: int) -> str:
        prompt = arena.prompt_by_label(prompt_label) or arena.prompts[0]
        idx = arena.prompts.index(prompt)
        return arena.prompts[(idx + delta) % len(arena.prompts)].label

    with gr.Blocks(title=f"{arena.name} — viewer") as demo:
        gr.Markdown(f"# {arena.name} — viewer\n"
                    f"{len(arena.models)} models × {len(arena.prompts)} prompts. "
                    "Names are shown and nothing is recorded.")

        with gr.Row():
            prev_btn = gr.Button("◀ Prev", scale=0)
            prompt_dd = gr.Dropdown(choices=arena.prompt_choices(),
                                    value=arena.prompts[0].label,
                                    label="Prompt",
                                    scale=8)
            next_btn = gr.Button("Next ▶", scale=0)

        prompt_box = gr.Textbox(label="Prompt", lines=5, max_lines=5, interactive=False, show_copy_button=True)

        with gr.Row():
            play_btn = gr.Button("▶ Play all (muted)")
            pause_btn = gr.Button("⏸ Pause all")
            rewind_btn = gr.Button("↺ Rewind all")
        gr.Markdown("*Play-all mutes every player — un-mute one to hear its audio.*")

        players = []
        for start in range(0, len(arena.models), PER_ROW):
            with gr.Row():
                for model in arena.models[start:start + PER_ROW]:
                    players.append(
                        gr.Video(label=model.display,
                                 height=video_height,
                                 loop=True,
                                 autoplay=False,
                                 show_download_button=True,
                                 interactive=False))

        outputs = [prompt_box, *players]
        gr.on(triggers=[prompt_dd.change, demo.load], fn=show, inputs=prompt_dd, outputs=outputs)
        prev_btn.click(lambda p: step(p, -1), inputs=prompt_dd, outputs=prompt_dd)
        next_btn.click(lambda p: step(p, 1), inputs=prompt_dd, outputs=prompt_dd)
        play_btn.click(None, None, None, js=PLAY_ALL_JS)
        pause_btn.click(None, None, None, js=PAUSE_ALL_JS)
        rewind_btn.click(None, None, None, js=REWIND_JS)

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true")
    p.add_argument("--video-height", type=int, default=420)
    p.add_argument("--video-cache-seconds", type=int, default=86400)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arena = Arena(args.manifest)
    logger.info("%s", arena.coverage().replace("**", ""))

    app_kwargs = {}
    if args.video_cache_seconds > 0:
        app_kwargs["middleware"] = [Middleware(VideoCacheHeaders, max_age=args.video_cache_seconds)]

    build_viewer(arena, video_height=args.video_height).launch(server_name=args.host,
                                                               server_port=args.port,
                                                               share=args.share,
                                                               allowed_paths=arena.serve_roots,
                                                               app_kwargs=app_kwargs)


if __name__ == "__main__":
    main()
