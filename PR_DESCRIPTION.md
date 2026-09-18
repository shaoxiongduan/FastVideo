# [feat] Add standalone Infinite Livestream app

## Purpose

Add a FastH3 livestream app that turns viewer prompts into video with audio and fills idle time with preset prompts.

## Changes

- Add a browser viewer with chat, queue status, and continuous HLS playback.
- Add prompt moderation, rewriting, scene queues, and automatic idle fillers.
- Package the `infinite-livestream-server` CLI with YAML configuration and installation docs. Defaults target four GB200 GPUs.
- Handle startup failures and bounded FFmpeg shutdown, with local regression tests.

## Test Plan

Reproduce the local checks from the repository root with the app dependencies installed:

```bash
PYTHONPATH="$PWD/apps/infinite_livestream" python -m pytest \
  apps/infinite_livestream/infinite_livestream/tests \
  -m 'not gpu' -q -p no:cacheprovider --tb=short

pre-commit run --files \
  apps/infinite_livestream/README.md \
  apps/infinite_livestream/infinite_livestream/main.py \
  apps/infinite_livestream/infinite_livestream/sink.py \
  apps/infinite_livestream/infinite_livestream/tests/*.py
```

Also checked wheel packaging and CLI `--help`, and encoded/decoded a 12-second synthetic HLS stream with FFmpeg.
Tests remain local; CI routing is unchanged. Full GPU generation and SSIM were not rerun.

## Test Results

<details>
<summary>Test output</summary>

```text
68 passed, 1 deselected, 1 warning in 2.89s
Config class/default isolation preserved
HLS smoke: 6 segments, H.264 video and AAC audio decoded
FFmpeg shutdown: 0.009 seconds, no SIGKILL needed
```

</details>

Changed-file formatting, lint, spelling, and Markdown checks passed. Mypy passed on the same source files in an
isolated copy; the hyphenated worktree directory name prevents package discovery in the original checkout.
The new failure/shutdown tests also fail against the original code, and the prompt-flow tests catch disabled enqueueing.

## Checklist

- [ ] I ran `pre-commit run --all-files` and fixed all issues
- [x] I added or updated tests for my changes
- [x] I updated documentation if needed
- [x] I considered GPU memory impact of my changes

**For model/pipeline changes, also check:**

Not applicable: this app uses existing FastH3 support.

- [ ] I verified SSIM regression tests pass
- [ ] I updated the support matrix if adding a new model
