# Livestream Agent Notes

A chat-driven FastH3 broadcast: viewers type prompts into a web page, the app
rewrites them, generates clips on FastVideo, and plays them back as one
continuous HLS stream on that same page.

## Layout

- `livestream/` — the whole app. Flat on purpose; there is one implementation
  of each thing, so there are no `chat/` or `sinks/` packages to hold variants
  that do not exist.
- `livestream/presets/` — creative bundles (style + idle prompts), chosen at
  startup with `PRESET`.
- `livestream/web/` — the single page, plus the logo and favicon it references.
- `livestream/tests/` — runs anywhere with `-m "not gpu"`.
- `serve_configs/fasth3.yaml` — what the checkpoint is asked for.

## The seam that matters

`FastH3Backend.submit(frames, prompt, seed, height, width) -> (frames, audio)`.
Everything above it — engine, director, queues, pacer, sink, web — is
model-agnostic. Everything below is MiniMax-H3 specific: `clip_plan.py` is its
geometry (24 fps, 17n+5 packing, a 5–15 s window, a 768 short edge) and
`backend.py`'s profile environment selects its sparse-attention kernels.

**Adding a second checkpoint means a new geometry module and a new backend
behind that seam, not edits to these two.** LTX-2, for instance, packs 8n+1
frames at its own resolutions.

## Rules worth knowing before editing

- **The engine and the broadcast share a process.** A built clip is handed to
  the pacer as the arrays it already is. Do not introduce an encode or a
  transport between them: a lossy hop can shed video frames while audio flows
  on, which is exactly how a picture drifts behind its soundtrack.
- **ffmpeg derives every PTS from byte counts** across two raw pipes, so an
  entry dropped on one and not the other shifts sound against picture
  permanently. Both are gated on the same `_ensure_running`.
- **`_feed_clip` must wait out its tail.** The slice loop sleeps *before* each
  slice, so returning at the last slice gains `EMIT_FRAMES/FPS` per clip; the
  pacer drains at a flat rate, so that surplus accumulates in its buffer until
  it starts dropping.
- **One pinned clip length is one compiled shape.** Varying
  `warmup_lengths` reintroduces the dynamo recompile-limit problem that
  `_raise_dynamo_limits` guards against.
- **`config.py` is the only reader of `os.environ` and the only YAML parser.**
  Keep it that way; everything else takes a `Config`.
- **Do not compute where a frame will land on the clock — read it.** ffmpeg's
  PROGRAM-DATE-TIME base was measured at 7.0–8.8 s after the process starts,
  varying between runs, and is not the encoder settle. `stream_time()` anchors
  on segment 0 of the playlist ffmpeg itself writes.
- **Do not assume the browser can report its playback date.** Measured in a
  real browser, `hls.playingDate`, the fragment's `programDateTime` and
  `getStartDate` all come back empty. The page derives its position from the
  live edge minus its distance behind its own buffer edge instead. Both of
  those step about a segment apart, so their difference is a square wave and
  has to be averaged over more than a segment and carried on `currentTime`,
  which is the only smooth quantity available.
- **A refusal must reach the viewer who caused it, and nobody else.** The chat
  feed is shared. The cooldown is answered at `POST /chat` with a 429 so the
  sender's own page can lock its box; anything refused later goes through
  `Director.on_reject` and is rendered only for the viewer it names. A refusal
  that only reaches the log is a prompt that silently disappears — which is
  exactly what shipped once.
- **There is no runtime control surface.** The preset is fixed at startup, so
  nothing mutates the director's prompt list or the upsampler's style once
  running. Adding live control means re-adding the flush that keeps a switch
  from leaving a queue of old-style clips on air.

## Comment style

Prose explains *why*, never *what* — the signature already says what. The
repo's app code runs a low prose-to-code ratio; match it. Do not restate a
method's name in its docstring.

## Testing

```bash
pytest apps/livestream/livestream/tests -m "not gpu"
```

The one `gpu`-marked test asserts `clip_plan`'s copy of MiniMax-H3's packing
constants still matches FastVideo's; importing the upstream module needs a
live CUDA driver. Run it whenever the pinned FastVideo version moves.

`mypy` cannot run from a worktree whose directory name contains a hyphen — it
reports "not a valid Python package name" for the *directory*, not the code.
