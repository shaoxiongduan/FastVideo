# SPDX-License-Identifier: Apache-2.0
"""Side-by-side ("arena") rating UI for comparing video-model checkpoints.

Two videos generated from the same prompt by two different checkpoints are shown
anonymously side by side. The rater picks a winner, calls it a tie, or marks both as
bad; the vote is appended to a JSONL log for offline analysis.

    python -m apps.video_arena --manifest apps/video_arena/mock/manifest.json

See ``apps/video_arena/README.md`` for the manifest format.
"""
from __future__ import annotations

import argparse
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from apps.video_arena.arena import Arena, Battle
from apps.video_arena.storage import VOTE_BOTH_BAD, VOTE_LEFT, VOTE_RIGHT, VOTE_TIE, VoteStore

logger = logging.getLogger(__name__)

RANDOM_PROMPT = "🎲 Random prompt"


def _prompt_md(battle: Battle) -> str:
    return f"### Prompt `{battle.prompt.id}`\n\n{battle.prompt.text}"


def _reveal_md(battle: Battle, vote: str) -> str:
    verdict = {
        VOTE_LEFT: f"**A wins** — {battle.left.display}",
        VOTE_RIGHT: f"**B wins** — {battle.right.display}",
        VOTE_TIE: "**Tie** — both good",
        VOTE_BOTH_BAD: "**Both bad**",
    }[vote]
    return f"{verdict}\n\n- A = {battle.left.display}\n- B = {battle.right.display}"


class ArenaUI:
    """Event handlers for the arena tab, kept out of ``build_demo`` so they are testable.

    ``new_battle`` returns a flat tuple lined up with ``battle_outputs`` in
    ``build_demo``; ``cast_vote`` returns that same tuple plus two extra outputs
    (rated counter, refreshed leaderboard).
    """

    def __init__(self, arena: Arena, store: VoteStore, rng: random.Random | None = None) -> None:
        self.arena = arena
        self.store = store
        self.rng = rng or random.Random()
        self.model_names = {m.id: m.name for m in arena.models}

    def leaderboard(self) -> pd.DataFrame:
        return self.store.leaderboard(self.model_names)

    def new_battle(self, prompt_label: str, n_rated: int) -> tuple[Any, ...]:
        label = None if prompt_label == RANDOM_PROMPT else prompt_label
        battle = self.arena.sample_battle(label, self.rng)
        return (
            {
                "battle": battle,
                "t0": time.time()
            },
            gr.update(value=str(battle.left_serve)),
            gr.update(value=str(battle.right_serve)),
            _prompt_md(battle),
            "",  # clear the reveal from the previous round
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),  # "next round" only becomes live after a vote
            f"Rated **{n_rated}** rounds this session.",
        )

    def cast_vote(self, vote: str, state: dict, session_id: str, n_rated: int) -> tuple[Any, ...]:
        if not state or "battle" not in state:
            raise gr.Error("No active round — pick a prompt to start one.")
        battle: Battle = state["battle"]

        self.store.record(
            arena=self.arena.name,
            session_id=session_id,
            battle_id=battle.battle_id,
            prompt_id=battle.prompt.id,
            prompt_text=battle.prompt.text,
            model_left=battle.left.id,
            model_right=battle.right.id,
            video_left=str(battle.left_video),
            video_right=str(battle.right_video),
            vote=vote,
            winner=battle.left.id if vote == VOTE_LEFT else battle.right.id if vote == VOTE_RIGHT else None,
            decision_ms=int((time.time() - state["t0"]) * 1000),
        )
        n_rated += 1

        # Stay on this round with the identities revealed; "next round" advances.
        return (
            state,
            gr.update(),
            gr.update(),
            gr.update(),
            _reveal_md(battle, vote),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            f"Rated **{n_rated}** rounds this session.",
            n_rated,
            self.leaderboard(),
        )


def build_demo(arena: Arena, store: VoteStore, rng: random.Random | None = None) -> gr.Blocks:
    ui = ArenaUI(arena, store, rng)

    with gr.Blocks(title=arena.name) as demo:
        session_id = gr.State(lambda: uuid.uuid4().hex[:16])
        battle_state = gr.State({})
        n_rated = gr.State(0)

        gr.Markdown(f"# {arena.name}\n"
                    "Two checkpoints, same prompt, names hidden until you vote.")

        with gr.Tabs():
            with gr.Tab("Arena"):
                prompt_dd = gr.Dropdown(choices=[RANDOM_PROMPT] + arena.prompt_choices(),
                                        value=RANDOM_PROMPT,
                                        label="Prompt")
                prompt_md = gr.Markdown()

                with gr.Row():
                    vid_a = gr.Video(label="A", autoplay=True, loop=True, show_download_button=False)
                    vid_b = gr.Video(label="B", autoplay=True, loop=True, show_download_button=False)

                with gr.Row():
                    btn_a = gr.Button("A is better", variant="primary")
                    btn_b = gr.Button("B is better", variant="primary")
                    btn_tie = gr.Button("Both good")
                    btn_bad = gr.Button("Both bad")

                reveal_md = gr.Markdown()
                next_btn = gr.Button("Next round", interactive=False)
                progress_md = gr.Markdown("Rated **0** rounds this session.")

            with gr.Tab("Leaderboard"):
                gr.Markdown("Elo starts at 1000 (K=32); *both good* and *both bad* both score 0.5 but are "
                            "counted in separate columns. `win_rate` uses decisive votes only.")
                lb_df = gr.Dataframe(value=ui.leaderboard, label="Leaderboard", interactive=False, wrap=True)
                pw_df = gr.Dataframe(value=store.pairwise, label="Head-to-head", interactive=False, wrap=True)
                refresh_btn = gr.Button("Refresh")
                refresh_btn.click(lambda: (ui.leaderboard(), store.pairwise()), outputs=[lb_df, pw_df])

            with gr.Tab("Votes"):
                gr.Markdown(f"Raw votes are appended to `{store.path}`.")
                votes_df = gr.Dataframe(value=lambda: store.load().tail(200),
                                        label="Recent votes (last 200)",
                                        interactive=False,
                                        wrap=True)
                with gr.Row():
                    votes_refresh = gr.Button("Refresh")
                    export_btn = gr.Button("Export CSV")
                export_file = gr.File(label="Exported CSV", interactive=False)
                votes_refresh.click(lambda: store.load().tail(200), outputs=votes_df)
                export_btn.click(lambda: str(store.export_csv(store.path.with_suffix(".csv"))), outputs=export_file)

            with gr.Tab("Setup"):
                gr.Markdown(arena.coverage())
                gr.Markdown(f"\nManifest: `{arena.manifest_path}`\n\n"
                            f"Anonymized video paths: `{arena.anonymize_paths}`")

        battle_outputs = [
            battle_state, vid_a, vid_b, prompt_md, reveal_md, btn_a, btn_b, btn_tie, btn_bad, next_btn, progress_md
        ]
        vote_outputs = battle_outputs + [n_rated, lb_df]
        vote_inputs = [battle_state, session_id, n_rated]

        for btn, vote in ((btn_a, VOTE_LEFT), (btn_b, VOTE_RIGHT), (btn_tie, VOTE_TIE), (btn_bad, VOTE_BOTH_BAD)):
            btn.click(lambda *a, _v=vote: ui.cast_vote(_v, *a), inputs=vote_inputs, outputs=vote_outputs)

        gr.on(triggers=[next_btn.click, prompt_dd.change, demo.load],
              fn=ui.new_battle,
              inputs=[prompt_dd, n_rated],
              outputs=battle_outputs)

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="apps/video_arena/mock/manifest.json", help="arena manifest JSON")
    p.add_argument("--votes", default="apps/video_arena/votes/votes.jsonl", help="where votes are appended")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public gradio.live tunnel")
    p.add_argument("--seed", type=int, default=None, help="seed the battle sampler (reproducible ordering)")
    p.add_argument("--no-anon-paths",
                   action="store_true",
                   help="serve videos from their real paths (model name becomes visible in the video URL)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arena = Arena(args.manifest, anonymize_paths=not args.no_anon_paths)
    store = VoteStore(args.votes)
    logger.info("%s", arena.coverage().replace("**", ""))
    logger.info("votes -> %s", store.path)

    demo = build_demo(arena, store, rng=random.Random(args.seed) if args.seed is not None else None)
    demo.launch(server_name=args.host,
                server_port=args.port,
                share=args.share,
                allowed_paths=arena.serve_roots + [str(Path(store.path).parent)])


if __name__ == "__main__":
    main()
