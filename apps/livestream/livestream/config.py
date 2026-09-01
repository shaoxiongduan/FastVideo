"""Configuration: the app's environment settings and the engine's YAML shape.

Two things are configured, and they come from different places on purpose.

* :class:`Config` -- what this *deployment* does: which chat sources feed it,
  which preset it runs, where the video goes, which LLM rewrites prompts. All
  from the environment (a `.env` file is loaded when present), because these
  are per-deployment secrets and switches.
* :class:`ModelConfig` -- what the *checkpoint* is asked for: clip geometry,
  sparse-attention kernels, GPU count, compile policy. From a YAML under
  `serve_configs/`, because these are tuned values that belong in version
  control next to the code that reads them.

`Config.load` and `load_model_config` are the only readers of either; nothing
else in the package touches `os.environ` or parses YAML.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from . import clip_plan


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- presets


class PresetError(ValueError):
    """A preset file is missing or malformed."""


def presets_dir() -> Path:
    return Path(__file__).parent / "presets"


def load_preset(name_or_path: str) -> dict:
    """Load and validate one preset: the creative bundle the stream runs.

    A bare name resolves against `presets/`; a value with a path separator or
    a `.json` suffix is used as a path. The format is `style` (the block every
    upsampled scene is written in) and `idle_prompts` (which may be empty,
    disabling the filler); other keys are ignored, so a preset can carry its
    own notes.
    """
    if "/" in name_or_path or name_or_path.endswith(".json"):
        path = Path(name_or_path)
    else:
        path = presets_dir() / f"{name_or_path}.json"
    if not path.is_file():
        raise PresetError(f"preset not found: {path}")
    try:
        preset = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PresetError(f"preset {path} is not valid JSON: {error}") from None
    style = preset.get("style")
    prompts = preset.get("idle_prompts")
    if not isinstance(style, str) or not style.strip():
        raise PresetError(f"preset {path} needs a non-empty string `style`")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise PresetError(f"preset {path} needs `idle_prompts` as a list of strings")
    return {
        "style": style.strip(),
        "idle_prompts": [p.strip() for p in prompts if p.strip()],
    }


# ------------------------------------------------------------ model config

# Component directories the T2VA pipeline loads. An incomplete bundle must kill
# startup, not surface as a loader traceback on the first clip.
REQUIRED_COMPONENTS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)

DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parents[1] / "serve_configs" / "fasth3.yaml"


@dataclass(frozen=True)
class ModelConfig:
    """Everything the engine YAML configures, validated once at load.

    The top-level fields are what the queues and the clip planner need;
    ``inference`` and ``runtime`` are the raw blocks, which the backend reads
    its engine knobs (attention kernels, compile flags, parallelism, offload
    policy) straight out of.
    """

    aspect: str
    clip_frames: int
    seed: int
    num_inference_steps: int
    queue_size: int
    generation_queue_size: int
    warmup_aspects: tuple[str, ...]
    warmup_frames: tuple[int, ...]
    inference: dict[str, Any]
    runtime: dict[str, Any]


def load_model_config(config_path: Path | None = None) -> ModelConfig:
    """Parse the engine YAML into a validated :class:`ModelConfig`.

    Raises:
        ValueError: If the configured aspect is not one the checkpoint offers,
            or a queue size is not positive.
    """
    path = config_path or DEFAULT_MODEL_CONFIG
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inference: dict[str, Any] = document.get("inference") or {}
    runtime: dict[str, Any] = document.get("runtime") or {}

    aspect = str(inference.get("aspect", "16:9"))
    if aspect not in clip_plan.ASPECT_CHOICES:
        raise ValueError(f"inference.aspect must be one of {list(clip_plan.ASPECT_CHOICES)}, got {aspect!r}")

    queue_size = int(inference.get("queue_size", 10))
    if queue_size < 1:
        raise ValueError(f"inference.queue_size must be positive, got {queue_size}")

    generation_queue_size = int(inference.get("generation_queue_size", 20))
    if generation_queue_size < 1:
        raise ValueError(f"inference.generation_queue_size must be positive, got {generation_queue_size}")

    clip_frames = clip_plan.frames_for_seconds(float(inference.get("clip_seconds", clip_plan.MAX_SECONDS)))

    return ModelConfig(
        aspect=aspect,
        clip_frames=clip_frames,
        seed=int(inference.get("seed", 1000)),
        num_inference_steps=int(inference.get("num_inference_steps", 5)),
        queue_size=queue_size,
        generation_queue_size=generation_queue_size,
        warmup_aspects=tuple(str(a) for a in (inference.get("warmup_aspects") or [aspect])),
        warmup_frames=_parse_warmup_lengths(inference.get("warmup_lengths"), clip_frames),
        inference=inference,
        runtime=runtime,
    )


def _parse_warmup_lengths(raw: Any, clip_frames: int) -> tuple[int, ...]:
    """Resolve ``inference.warmup_lengths`` to the frame counts load() warms.

    ``"default"`` (or nothing) warms only the session's default length;
    ``"all"`` warms every length the checkpoint can generate; a list of
    seconds warms those, snapped to legal lengths. The default length is
    always included -- it is the shape every plain enqueue uses.
    """
    if raw in (None, "", "default"):
        return (clip_frames, )
    if raw == "all":
        frames = set(clip_plan.legal_frame_counts())
    elif isinstance(raw, list | tuple):
        frames = {clip_plan.frames_for_seconds(float(seconds)) for seconds in raw}
    else:
        raise ValueError(f'inference.warmup_lengths must be "default", "all", or a list of seconds, got {raw!r}')
    frames.add(clip_frames)
    return tuple(sorted(frames))


def resolve_model_path(config: ModelConfig, weights_root: Path) -> Path:
    """The checkpoint directory inside the bundle; "." means the root itself."""
    subdir = str(config.runtime.get("checkpoint_dir", "."))
    if subdir in ("", "."):
        return weights_root
    return weights_root / subdir


def require_weights(root: Path, model_path: Path) -> None:
    """Fail startup loudly when the weights bundle is incomplete."""
    problems: list[str] = []
    if not model_path.is_dir():
        problems.append(f"checkpoint directory is missing: {model_path}")
    else:
        index = model_path / "modular_model_index.json"
        if not index.is_file():
            problems.append(f"modular_model_index.json is missing: {index}")
        for component in REQUIRED_COMPONENTS:
            if not (model_path / component).is_dir():
                problems.append(f"component directory is missing: {model_path / component}")
    if problems:
        raise FileNotFoundError(f"FastH3 weights bundle under {root} is incomplete:\n  " + "\n  ".join(problems))


# -------------------------------------------------------------- app config


@dataclass(frozen=True)
class Config:
    """One immutable snapshot of everything the app is configured with."""

    # The engine: where the weights live and which YAML shapes it
    weights_path: Path
    model_config_path: Path | None

    # Upsampling
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    max_chunks: int
    # Filler always wears the preset's style; a viewer's own request may pick
    # whatever look suits it. Set 0 to put every clip in the house style.
    viewer_free_style: bool

    # Preset: the creative bundle (style + premade idle prompts)
    preset_name: str
    style: str
    idle_prompts: tuple[str, ...]

    # Moderation (its own endpoint: the upsampling gateway may not expose
    # /moderations, so this can point at api.openai.com while upsampling
    # goes elsewhere)
    moderation_enabled: bool
    moderation_api_key: str
    moderation_base_url: str | None
    moderation_model: str

    # Idle filler
    idle_queue_target: int

    # Output: the HLS playlist the page plays, written by `sink.py`.
    hls_dir: str
    video_bitrate_k: int

    # The watch page: video, chat and the queue on one HTTP origin, so a
    # single tunnel publishes the whole thing.
    web_host: str
    web_port: int

    # Chat
    chat_command: str
    chat_cooldown_s: float

    @staticmethod
    def load(argv: list[str] | None = None) -> Config:
        """Read `.env` + environment, apply CLI overrides, and validate."""
        parser = argparse.ArgumentParser(description="Chat-driven FastH3 livestream (see README.md).")
        parser.add_argument("--env-file", default=None, help="path to a .env file")
        parser.add_argument("--weights", default=None, help="override LIVESTREAM_WEIGHTS_PATH")
        parser.add_argument(
            "--model-config",
            default=None,
            help="engine YAML (default serve_configs/fasth3.yaml)",
        )
        parser.add_argument("--preset", default=None, help="override PRESET")
        parser.add_argument("--port", default=None, type=int, help="override WEB_PORT")
        args = parser.parse_args(argv)

        if args.env_file:
            load_dotenv(args.env_file, override=True)
        else:
            load_dotenv()  # ./.env when present; no-op otherwise

        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        openai_base_url = os.environ.get("OPENAI_BASE_URL") or None

        preset_name = args.preset or os.environ.get("PRESET", "unhinged")
        try:
            preset = load_preset(preset_name)
        except PresetError as error:
            raise SystemExit(f"{error} (set PRESET or --preset)") from None

        weights = args.weights or os.environ.get("LIVESTREAM_WEIGHTS_PATH", "")
        model_config = args.model_config or os.environ.get("LIVESTREAM_MODEL_CONFIG") or None

        config = Config(
            weights_path=Path(weights).expanduser() if weights else Path(),
            model_config_path=Path(model_config).expanduser() if model_config else None,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            max_chunks=max(1, int(os.environ.get("MAX_CHUNKS", "6"))),
            viewer_free_style=_flag(os.environ.get("VIEWER_FREE_STYLE", "1")),
            preset_name=preset_name,
            style=preset["style"],
            idle_prompts=tuple(preset["idle_prompts"]),
            moderation_enabled=_flag(os.environ.get("MODERATION_ENABLED", "1")),
            moderation_api_key=os.environ.get("MODERATION_API_KEY") or openai_api_key,
            moderation_base_url=os.environ.get("MODERATION_BASE_URL") or openai_base_url,
            moderation_model=os.environ.get("MODERATION_MODEL", "omni-moderation-latest"),
            idle_queue_target=int(os.environ.get("IDLE_QUEUE_TARGET", "6")),
            hls_dir=os.environ.get("HLS_DIR", "./hls"),
            video_bitrate_k=int(os.environ.get("VIDEO_BITRATE_K", "4500")),
            web_host=os.environ.get("WEB_HOST", "0.0.0.0"),
            web_port=args.port or int(os.environ.get("WEB_PORT", "8081")),
            chat_command=os.environ.get("CHAT_COMMAND", "!prompt").strip(),
            chat_cooldown_s=float(os.environ.get("CHAT_COOLDOWN_S", "30")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Fail fast on contradictions instead of half-starting."""
        if not str(self.weights_path) or self.weights_path == Path():
            raise SystemExit("LIVESTREAM_WEIGHTS_PATH (or --weights) must point at the FastH3 bundle.")
        if not self.openai_api_key:
            raise SystemExit("OPENAI_API_KEY is required for prompt upsampling.")
        if not self.chat_command.startswith("!"):
            raise SystemExit("CHAT_COMMAND should start with '!' (e.g. !prompt).")


__all__ = [
    "Config",
    "ModelConfig",
    "PresetError",
    "load_model_config",
    "load_preset",
    "require_weights",
    "resolve_model_path",
]
