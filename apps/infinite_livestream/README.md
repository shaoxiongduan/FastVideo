# Infinite Livestream

Infinite Livestream is a chat-driven FastH3 broadcast. Viewers type prompts into a web
page, the app rewrites them with an LLM, generates clips with FastVideo, and
plays them back as one continuous HLS stream on that same page. When nobody is
typing it feeds itself from a preset of idle prompts, so the channel never goes
dark.

It lives in this monorepo under `apps/infinite_livestream/`.

```
chat -> Director -> PromptUpsampler (OpenAI-compatible LLM)
           |
           v enqueue / move / pop
        Engine -> FastH3Backend -> FastVideo
           |  frames + audio
           v
        Pacer -> HlsSink -> the page's <video>
```

Everything runs in a single process, and the page, the playlist and the chat
endpoint are served from one HTTP origin, so publishing the stream means
pointing a tunnel or reverse proxy at one port.

## Requirements

- FastH3 weights. Set `LIVESTREAM_WEIGHTS_PATH` to the model directory. It
  needs `transformer`, `text_encoder`, `tokenizer`, `processor`, `vae`,
  `audio_vae`, `scheduler`, `audio_scheduler` and `modular_model_index.json`.
- GPUs for the generator. `infinite_livestream/configs/infinite_livestream.yaml` ships `num_gpus: 4`;
  the count must divide the model's attention head count.
- An API key for an OpenAI-compatible endpoint. Prompt rewriting is on the
  path for the idle filler too, so the stream does not run without one.
- `ffmpeg` on `PATH`.

## Install

The app ships as part of FastVideo, under the `infinite-livestream` extra.
From a source checkout:

```bash
uv pip install -e ".[infinite-livestream]"
```

Or from PyPI:

```bash
uv pip install "fastvideo[infinite-livestream]"
```

Either way you also need the FastVideo runtime, which for this app means a
`fastvideo-kernel` build carrying the Blackwell VSA extension and
`flash-attn-4`, plus `ffmpeg` on `PATH`.

## Quick start

```bash
export OPENAI_API_KEY=...
export LIVESTREAM_WEIGHTS_PATH=/path/to/fasth3
infinite-livestream-server
```

The page and the HLS stream come up immediately on the configured port. The
model loads behind them, so a viewer arriving during startup sees the page and
a live black stream rather than a refused connection. Loading takes a few
minutes, most of it weight loading and the compile warm-up.

## Configuration

`infinite_livestream/configs/infinite_livestream.yaml` holds everything the app is configured with. Copy it
and pass your own with `--config`:

```bash
infinite-livestream-server --config my-config.yaml
```

| Block | Contents |
|---|---|
| `inference` | What the checkpoint is asked for: clip length, canvas, sparse-attention kernels, compile policy. |
| `runtime` | How it is hosted: GPU count, sharding, offload. |
| `upsampler` | Prompt rewriting: model, endpoint, how many clips one prompt may become. |
| `moderation` | Whether viewer prompts are checked, and against which endpoint. |
| `director` | Idle filler depth, per-viewer cooldown, chat command, filler directory. |
| `output` | Where the playlist is written (defaults under `$XDG_STATE_HOME`), and the video bitrate. |
| `web` | Bind address and port. |

Two settings stay in the environment. API keys, because a key in a
version-controlled file is a key that leaks, and the weights path, because it
is a property of the machine rather than of the deployment.

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required. Prompt rewriting runs for the idle filler too, so the stream does not start without it. |
| `LIVESTREAM_WEIGHTS_PATH` | Required. FastH3 model directory. `--weights` overrides it. |
| `MODERATION_API_KEY` | Optional. Falls back to `OPENAI_API_KEY`. |

## Clip geometry

Clip geometry is fixed by the checkpoint: 24 fps, frame counts of the form
`17n + 5`, lengths between 5 and 15 seconds, and a 768 pixel short edge.
`inference.clip_seconds: 14.375` is the longest clip it can produce, at 345
frames.

Keeping one clip length means one compiled shape. Setting
`inference.warmup_lengths: all` warms every legal length instead, which makes
startup slower but avoids a one-off compile stall on a viewer's first
odd-length clip.

## API Endpoints

| Route | Description |
|---|---|
| `GET /` | The watch page. |
| `GET /assets/<file>` | Logo and favicon. |
| `GET /hls/<file>` | Playlist and segments, written by `infinite_livestream/sink.py`. |
| `GET /healthz` | `{"connected": bool}`, true once the model is loaded. |
| `WS /state` | One JSON snapshot on connect, then one per change. |
| `POST /chat` | `{"author": str, "text": str}`. Returns 429 with `retry_after` when that viewer is still on cooldown. |

The cooldown is answered by `POST /chat` rather than reported later, so the
sender's page can disable its send box and count down. The chat feed is shared
by every viewer, so refusals are kept out of it.

## Idle fillers

When nobody is typing, the stream keeps itself fed from a list of prompts.
`director.fillers` names the directory holding `fillers.json`, and defaults to
the one that ships in `infinite_livestream/presets/`.

```json
{
  "style": "the look and tone every scene is written in",
  "idle_prompts": ["a lighthouse keeper teaching a seagull to play chess"]
}
```

`style` is applied to every rewritten scene, viewer prompts included.
`idle_prompts` feeds the filler; an empty list turns the filler off, as does
`director.idle_queue_target: 0`.

To change the stream's identity, copy the directory, edit `fillers.json` and
point `director.fillers` at it. The file is read once, at startup.

## Now-playing titles

Each viewer sits at a different point in the stream, so the server cannot say
what is on screen. It publishes which clip occupies which instant, and the page
locates itself against that.

Where a browser reports the date of the frame it is showing, the page uses it
directly. Most browsers report nothing, so the page falls back to the live edge
the server publishes minus how far behind its own buffer edge it is playing.
Append `?debug=1` to the page URL to see which source answered.

## Tests

```bash
pytest apps/infinite_livestream/infinite_livestream/tests -m "not gpu"
```

One test is marked `gpu`. It checks that `infinite_livestream/clip_plan.py`'s copy of
MiniMax-H3's packing constants still matches FastVideo's, and importing the
upstream module needs a live CUDA driver. Run it when the pinned FastVideo
version moves.

## Adding another model

`FastH3Backend.submit(frames, prompt, seed, height, width)` is the seam.
Everything above it, meaning the engine, director, queues, pacer, sink and web
app, is model-agnostic. Everything below it is MiniMax-H3 specific:
`clip_plan.py` is its geometry and `backend.py` selects its kernels.

A second checkpoint needs its own geometry module and its own backend behind
that seam. LTX-2, for example, packs `8n + 1` frames at different resolutions.

## Troubleshooting

**`ffmpeg not found on PATH`.** The sink checks for it at construction. Install
it, or see `apps/dreamverse/scripts/install_native_ffmpeg.sh`.

**The weights are incomplete.** Startup lists the missing components before
any GPU work begins. The model directory needs `transformer`, `text_encoder`,
`tokenizer`, `processor`, `vae`, `audio_vae`, `scheduler`, `audio_scheduler`
and `modular_model_index.json`.

**`FastH3's sm100a route needs fastvideo-kernel built with the Blackwell VSA
extension`.** Startup checks for the fast sparse-attention kernel before
loading any weights. Install a `fastvideo-kernel` build that carries it, or set
`vsa_kernel: triton` in the engine YAML to use the slower fallback.

**`FastH3's FA4 route needs the pinned flash-attn-4 package`.** Install it, or
set `fa4: false` in the engine YAML.

**A clip takes much longer than the others.** Each distinct clip length is a
separate compiled shape, and the first clip at a new length pays a one-off
compile cost. `warmup_lengths: all` pays all of them at startup instead.
