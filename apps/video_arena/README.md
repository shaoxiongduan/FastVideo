# Video Arena

A side-by-side ("arena") rating app for comparing video-model checkpoints — built for
picking between MiniMax H3 checkpoints, but nothing in it is H3-specific.

Two videos generated from the **same prompt** by **two different checkpoints** are shown
side by side. Which checkpoint is which, and which one lands on the left, are both
randomized, and the names stay hidden until the rater votes. Every vote is appended to a
JSONL log for offline analysis.

```
  Arena | Leaderboard | Votes | Setup
  ------------------------------------------------------------------
   Prompt [ 🎲 Random prompt                                     v ]

   ### Prompt p003
   A paper boat drifting down a rain-soaked gutter, macro shot

   ┌──────────── A ─────────────┐  ┌──────────── B ─────────────┐
   │                            │  │                            │
   └────────────────────────────┘  └────────────────────────────┘

   [ A is better ] [ B is better ] [ Both good ] [ Both bad ]

   [ Next round ]
```

Voting replaces the "A" / "B" player labels with the checkpoint each video came from
and disables the vote buttons; **Next round** draws the next pair and re-hides the
names. Picking a specific prompt starts a round on it right away, and **Next round**
then returns the selector to random — so pinning a prompt is a one-off, not a mode.

## Quick start (mock data, no GPU)

```bash
# 1. generate placeholder videos + a manifest
python -m apps.video_arena.make_mock_data --out apps/video_arena/mock

# 2. serve the arena
python -m apps.video_arena --manifest apps/video_arena/mock/manifest.json
# -> http://localhost:7860   (add --share for a public gradio.live tunnel)
```

## Letting other people rate

Three ways to expose the app. Numbers below are measured on this cluster by fetching the
six real clips of three sampled rounds (45 MiB total) through each route:

| route | page load | per round (2 clips, ~15 MiB) | notes |
|---|---|---|---|
| SSH port-forward | 8 ms | <0.1 s (386 MB/s) | no third party, no cap; rater needs cluster access |
| ngrok | 0.3 s | 0.3 s (30–60 MB/s) | needs an authtoken; monthly byte cap on the plan |
| gradio `--share` | 2.9 s | 40–65 s (0.13–0.65 MB/s) | zero setup; link expires; relay is the bottleneck |

Neither the app nor Lustre is the constraint — the same files serve at 386 MB/s over
plain local HTTP, and read off Lustre faster still. **`gradio --share` works but its relay
runs at well under 1 MB/s**, which is 40–65 s of loading per round. Fine to demo, unusable
for a rating session.

One caveat on ngrok: a long-lived agent session once degraded to a flat 2.00 MB/s across
every file size, and restarting `ngrok http` restored 30–60 MB/s. I could not account for
it. If throughput feels wrong, restart the agent before debugging anything else.

```bash
# SSH forward — best if raters have cluster access
ssh -N -L 7860:localhost:7860 <user>@<node>

# ngrok — run alongside the app, in a second shell
ngrok config add-authtoken <token>     # once
ngrok http 7860

# gradio's own tunnel
python -m apps.video_arena --manifest ... --share
```

Budget the bytes before you start: clips average 11 MB, so **one round costs ~15–22 MB**
and 400 rounds is 6–9 GB. Check your ngrok plan's monthly transfer allowance against that
number before a large session.

## Importing a published voting bundle

If your videos come from a `h3-validation-voting-ready-v1` bundle (`manifest.jsonl` +
`arms.json` + `arms/<slug>/videos/`), the manifest is generated for you:

```bash
python -m apps.video_arena.import_bundle \
  --bundle /mnt/lustre/vlm-shared/h3_fasth3_validation_voting/fast_h3_v1_family_heldout60 \
  --out apps/video_arena/arenas/fasth3_v1_family.json \
  --name "FastH3 v1-family (held-out 60)"

python -m apps.video_arena --manifest apps/video_arena/arenas/fasth3_v1_family.json \
  --votes apps/video_arena/votes/fasth3_v1_family.jsonl
```

Arms are discovered from the `arms/` directory listing, not read out of `arms.json`, so
**adding a checkpoint is: drop `arms/<new-slug>/videos/*.mp4` in and re-run the import.**
An arm missing from `arms.json` still works — it just gets its slug as its display name.
An arm that has only rendered some of the prompts is fine too: it is simply never paired
on a prompt it hasn't rendered.

## Using real checkpoints

Put one folder of videos per checkpoint, with **matching filenames across folders** —
the filename stem is the prompt id that pairs videos up:

```
samples/
├── manifest.json
├── h3-ckpt-1000/   p001.mp4  p002.mp4  p003.mp4 ...
├── h3-ckpt-4000/   p001.mp4  p002.mp4  p003.mp4 ...
└── h3-turbo-lora/  p001.mp4  p002.mp4  p003.mp4 ...
```

`manifest.json`:

```json
{
  "name": "MiniMax H3 checkpoint arena",
  "description": "Stage-2 checkpoints, 720p, 30 steps, cfg 5.0",
  "models": [
    {"id": "h3-ckpt-1000", "name": "H3 @ 1k steps",  "dir": "h3-ckpt-1000", "notes": "lr 1e-5"},
    {"id": "h3-ckpt-4000", "name": "H3 @ 4k steps",  "dir": "h3-ckpt-4000"},
    {"id": "h3-turbo-lora", "name": "H3 turbo LoRA", "dir": "/abs/path/also/works"}
  ],
  "prompts": [
    {"id": "p001", "text": "A red panda eating bamboo in a misty forest"},
    {"id": "p002", "text": "Timelapse of a city skyline at sunset"}
  ]
}
```

Then: `python -m apps.video_arena --manifest samples/manifest.json --votes votes/h3.jsonl`

### Manifest rules

| Field | Required | Notes |
|---|---|---|
| `models[].id` | yes | Stable key written into the vote log. Don't rename it mid-study. |
| `models[].dir` | yes | Absolute, or relative to the manifest file. |
| `models[].name` / `notes` | no | Shown on the reveal and the Setup tab only. |
| `prompts` | no | Omit it and prompt ids are discovered from the video filenames; prompt *text* is then read from a `prompts.json` / `prompts.txt` sidecar next to the manifest, falling back to the id. |
| `prompts[].file` | no | Filename to use in every model dir, when it isn't `{id}.mp4`. |
| `prompts[].files` | no | `{"model_id": "weird_name.mp4"}` per-model override. |

Any prompt with fewer than two videos is dropped (with the model list on the Setup tab
showing the coverage), so a partially-finished sampling run still works.

## Anonymization

- The rater sees only "A" / "B"; the player labels become the checkpoint names after the vote.
- Which model is A vs B is re-randomized every round, so position carries no signal.
- By default videos are served through per-battle symlinks with opaque names
  (`<battle_id>_L.mp4`) in a temp dir, so the checkpoint name is not visible in the
  video URL either. `--no-anon-paths` disables this and serves the real paths.
- The *real* paths and model ids are always what gets written to the vote log.

Caveat: the mock videos are visually distinguishable by design (different palette and
jitter per style). Real samples from checkpoints of the same model family generally
aren't, but if your checkpoints have an obvious tell (a watermark, a resolution
difference), anonymization can't fix that — normalize the videos first.

## Vote log

One JSON object per line, appended and `fsync`ed:

```json
{"ts": "2026-08-26T06:59:00+00:00", "arena": "...", "session_id": "a1b2...",
 "battle_id": "3f9c...", "prompt_id": "p003", "prompt_text": "...",
 "model_left": "h3-ckpt-1000", "model_right": "h3-ckpt-4000",
 "video_left": "/abs/.../p003.mp4", "video_right": "/abs/.../p003.mp4",
 "vote": "right", "winner": "h3-ckpt-4000", "decision_ms": 4821}
```

`session_id` is per browser session, so votes can be grouped by rater without
collecting names.

`vote` is one of `left` / `right` / `tie` / `both_bad`; `winner` is `null` for the
last two. Analyze it with:

```python
import pandas as pd
df = pd.read_json("votes/h3.jsonl", lines=True)
```

Or use the built-in aggregations:

```python
from apps.video_arena.storage import VoteStore
store = VoteStore("votes/h3.jsonl")
store.leaderboard()   # wins/losses/ties/both_bad, win_rate, rating, ci95
store.verdicts()      # [(model_left, model_right, vote), ...] after filtering bad rows
store.pairwise()      # head-to-head, both orderings merged
store.export_csv("votes/h3.csv")
```

### Ratings

`rating` is a **Bradley–Terry** maximum-likelihood fit, reported on the Elo scale: 1000
is the field average and 400 points is 10:1 odds, so

```
P(i beats j) = 1 / (1 + 10 ** (-(rating_i - rating_j) / 400))
```

It is fitted from the whole vote set at once, so it does **not** depend on the order
votes arrived in. (Classic sequential Elo does: on a 14-vote log, reshuffling the same
votes moved the gap between two checkpoints from −26 to +105 — it flipped which one was
ahead. That is why this is a batch fit, not incremental updates.)

`ci95` is the half-width of a bootstrap 95% interval. **If two rows' intervals overlap,
you don't have enough votes to separate those checkpoints yet** — that is the number to
look at before concluding a checkpoint is better. Each model is also given
`BT_REGULARIZATION` virtual ties against a fixed anchor, so a checkpoint that has only
ever won gets a large-but-finite rating instead of infinity.

`tie` and `both_bad` each count half a win, but are reported in separate columns because
"both good" and "both bad" mean very different things about a checkpoint. `win_rate`
uses decisive votes only.

The rating is recomputed from the log on demand — opening the **Leaderboard** tab or
hitting Refresh — not on every vote, since the bootstrap takes ~0.2–2 s.

## CLI

```
--manifest        arena manifest JSON (default: apps/video_arena/mock/manifest.json)
--votes           JSONL vote log      (default: apps/video_arena/votes/votes.jsonl)
--host / --port   bind address        (default: 0.0.0.0:7860)
--share           public gradio.live tunnel
--seed            seed the battle sampler for a reproducible battle order
--no-anon-paths   serve videos from their real paths
```

Several people can rate at once against one server; each browser session gets its own
`session_id` and battle stream, and writes are lock-protected.

## Tests

```bash
pytest apps/video_arena/tests -v   # CPU only, no video decoding
```

## Layout

| File | Purpose |
|---|---|
| `arena.py` | Manifest loading, prompt × model video grid, battle sampling, path anonymization |
| `storage.py` | Append-only JSONL vote log + leaderboard / head-to-head aggregation |
| `app.py` | Gradio UI (`ArenaUI` holds the handlers, `build_demo` wires them) |
| `make_mock_data.py` | Placeholder video + manifest generator for UI work without a GPU |
