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


def _prompt_update(battle: Battle) -> dict:
    """Prompt goes in a fixed-height read-only box: real prompts run to a few thousand chars,
    and rendering them inline pushes the videos below the fold."""
    return gr.update(value=battle.prompt.text, label=f"Prompt — {battle.prompt.id}")


class ArenaUI:
    """Event handlers for the arena tab, kept out of ``build_demo`` so they are testable.

    ``new_battle`` returns a flat tuple lined up with ``battle_outputs`` in
    ``build_demo``. ``cast_vote`` returns that same tuple plus two extra outputs (rated
    counter, refreshed leaderboard); ``next_round`` appends the prompt-dropdown reset.
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
            gr.update(value=str(battle.left_serve), label="A"),  # anonymous again
            gr.update(value=str(battle.right_serve), label="B"),
            _prompt_update(battle),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),  # "next round" only becomes live after a vote
            f"Rated **{n_rated}** rounds this session.",
        )

    def next_round(self, n_rated: int) -> tuple[Any, ...]:
        """The "next round" button always goes back to a randomly chosen prompt."""
        return (*self.new_battle(RANDOM_PROMPT, n_rated), gr.update(value=RANDOM_PROMPT))

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

        # Stay on this round, but relabel each player with the checkpoint it came from;
        # "next round" is what advances.
        return (
            state,
            gr.update(label=battle.left.display),
            gr.update(label=battle.right.display),
            gr.update(),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            f"Rated **{n_rated}** rounds this session.",
            n_rated,
        )


def build_demo(arena: Arena,
               store: VoteStore,
               rng: random.Random | None = None,
               autoplay: bool = False,
               video_height: int = 460) -> gr.Blocks:
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
                prompt_md = gr.Textbox(label="Prompt", lines=6, max_lines=6, interactive=False, show_copy_button=True)

                # autoplay defaults off: these clips carry audio, and two players starting at
                # once talk over each other. Pass --autoplay for silent visual-only comparison.
                with gr.Row():
                    vid_a = gr.Video(label="A",
                                     autoplay=autoplay,
                                     loop=True,
                                     show_download_button=False,
                                     height=video_height)
                    vid_b = gr.Video(label="B",
                                     autoplay=autoplay,
                                     loop=True,
                                     show_download_button=False,
                                     height=video_height)

                with gr.Row():
                    btn_a = gr.Button("A is better", variant="primary")
                    btn_b = gr.Button("B is better", variant="primary")
                    btn_tie = gr.Button("Both good")
                    btn_bad = gr.Button("Both bad")

                next_btn = gr.Button("Next round", interactive=False)
                progress_md = gr.Markdown("Rated **0** rounds this session.")

            with gr.Tab("Leaderboard") as lb_tab:
                gr.Markdown("`rating` is a Bradley-Terry fit on the Elo scale (1000 = average, 400 points "
                            "= 10:1 odds), so it does not depend on the order votes arrived in. `ci95` is a "
                            "bootstrap 95% half-interval — **if two rows' intervals overlap, you do not yet "
                            "have enough votes to separate those checkpoints.** *Both good* and *both bad* "
                            "each score half a win but are counted separately; `win_rate` uses decisive "
                            "votes only.")
                lb_df = gr.Dataframe(value=ui.leaderboard, label="Leaderboard", interactive=False, wrap=True)
                pw_df = gr.Dataframe(value=store.pairwise, label="Head-to-head", interactive=False, wrap=True)
                refresh_btn = gr.Button("Refresh")
                # Refit on demand, not on every vote: the bootstrap costs ~0.2-2s.
                gr.on(triggers=[refresh_btn.click, lb_tab.select],
                      fn=lambda: (ui.leaderboard(), store.pairwise()),
                      outputs=[lb_df, pw_df])

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

        battle_outputs = [battle_state, vid_a, vid_b, prompt_md, btn_a, btn_b, btn_tie, btn_bad, next_btn, progress_md]
        vote_inputs = [battle_state, session_id, n_rated]

        for btn, vote in ((btn_a, VOTE_LEFT), (btn_b, VOTE_RIGHT), (btn_tie, VOTE_TIE), (btn_bad, VOTE_BOTH_BAD)):
            btn.click(lambda *a, _v=vote: ui.cast_vote(_v, *a), inputs=vote_inputs, outputs=battle_outputs + [n_rated])

        # `.input` (not `.change`) so that next_round resetting the dropdown below does
        # not itself count as picking a prompt and draw a second battle.
        gr.on(triggers=[prompt_dd.input, demo.load],
              fn=ui.new_battle,
              inputs=[prompt_dd, n_rated],
              outputs=battle_outputs)
        next_btn.click(ui.next_round, inputs=[n_rated], outputs=battle_outputs + [prompt_dd])

    return demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="apps/video_arena/mock/manifest.json", help="arena manifest JSON")
    p.add_argument("--votes", default="apps/video_arena/votes/votes.jsonl", help="where votes are appended")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public gradio.live tunnel")
    p.add_argument("--seed", type=int, default=None, help="seed the battle sampler (reproducible ordering)")
    p.add_argument("--autoplay",
                   action="store_true",
                   help="start both players automatically; leave off so their audio tracks don't overlap")
    p.add_argument("--video-height", type=int, default=460, help="max player height in px")
    p.add_argument("--no-anon-paths",
                   action="store_true",
                   help="serve videos from their real paths (model name becomes visible in the video URL)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arena = Arena(args.manifest, anonymize_paths=not args.no_anon_paths)
    store = VoteStore(args.votes)
    logger.info("%s", arena.coverage().replace("**", ""))
    logger.info("votes -> %s", store.path)

    demo = build_demo(arena,
                      store,
                      rng=random.Random(args.seed) if args.seed is not None else None,
                      autoplay=args.autoplay,
                      video_height=args.video_height)
    demo.launch(server_name=args.host,
                server_port=args.port,
                share=args.share,
                allowed_paths=arena.serve_roots + [str(Path(store.path).parent)])


if __name__ == "__main__":
    main()
