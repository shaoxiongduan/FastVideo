# livestream — a chat-driven FastH3 broadcast

Viewers type prompts into a web page; the app rewrites them, generates clips
on FastVideo, and plays them back as one continuous stream on that same page.
When nobody is typing it keeps itself fed from a preset of idle prompts, so
the channel never goes dark.

```
chat ──▶ Director ──▶ PromptUpsampler (any OpenAI-compatible LLM)
            │
            ▼ enqueue / move / pop
          Engine ──▶ FastH3Backend ──▶ FastVideo (4 GPUs, VSA + FA4)
            │ frames + audio
            ▼
          Pacer ──▶ HlsSink ──▶ the page's <video>
```

The page is the only way in and the only way out: there is no Twitch or
YouTube integration and no RTMP target, because a chat box on the same origin
as the video is all this needs.

## One process, on purpose

The generator and the broadcast run in the same interpreter, so a finished
clip is handed to the pacer as the numpy arrays it already is. There is no
transport in between and therefore nothing that can shed frames: the pacer's
video and audio buffers move together or not at all.

`Engine` is the whole model side — the two clip queues, the build pump, and
the playout loop. It exposes three commands (`enqueue`, `pop`, `move`) and
broadcasts state as messages (`clip_queued`, `clip_generated`, `clip_started`,
`clip_finished`, `queue_update`, `state_update`) to in-process listeners.

## The two queues

A clip is requested, then built, then played, and each stage is a queue:

* the **generation queue** holds accepted requests; builds consume it front
  first, one at a time;
* the **playout queue** holds built clips waiting their turn. Each entry holds
  a fully decoded clip in host memory, so its capacity is also the memory
  budget.

Generation pauses while the playout queue is full — a finished build needs a
slot to land in, and that pause is the reservation that makes the later add
impossible to overflow. The director curates the front of the playout queue so
a viewer's clip plays before filler.

## Configuration

Two files, split by what they describe:

* `serve_configs/fasth3.yaml` — what the *checkpoint* is asked for: clip
  length, canvas, sparse-attention kernels, GPU count, compile policy. Tuned
  values, in version control.
* `.env` — what this *deployment* does: which LLM rewrites prompts, which
  preset it runs, where the playlist goes, who may run admin commands.
  Secrets and switches, not in version control.

`config.py` is the only reader of either; nothing else touches `os.environ`.

`LIVESTREAM_WEIGHTS_PATH` points at the FastH3 bundle. The bundle is checked
for completeness at startup, before any GPU work, so a missing component is a
one-line error rather than a loader traceback five minutes in.

## Running it

```bash
uv pip install -e apps/livestream
cp apps/livestream/.env.example .env      # keys, preset, sink
livestream-server
```

The web page and the HLS stream come up immediately; the model loads behind
them, so a viewer arriving during startup sees the page and a live black
stream rather than a refused connection.

## Tests

`livestream/tests/` runs on any machine:

```bash
pytest apps/livestream/livestream/tests -m "not gpu"
```

The `gpu` marker covers one test: the assertion that `clip_plan`'s copy of
MiniMax-H3's packing constants still matches FastVideo's. Importing the
upstream module needs a live CUDA driver, so run that one whenever the pinned
FastVideo version moves.

## Presets

A preset is one JSON bundle: the `style` block every rewritten scene is
written in, plus the `idle_prompts` that keep the stream fed. Drop a new one
into `livestream/presets/` and an admin can switch to it live with
`!switch <name>` — the folder is re-scanned per switch, so no restart.

## What is deliberately not here

The app targets FastH3 only. `clip_plan.py` is MiniMax-H3's geometry (24 fps,
17n+5 frame packing, a 5-15 s window, a 768 short edge) and `backend.py`'s
profile environment selects H3's sparse-attention kernels. A second checkpoint
-- LTX-2 packs 8n+1 frames at its own resolutions -- wants its own geometry
module and its own backend behind the same `submit()` seam, not edits to
these. Nothing above that seam (director, queues, pacer, sinks, web) is
model-specific.
