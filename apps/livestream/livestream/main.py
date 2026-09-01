"""Chat-driven FastH3 livestream: one process, from prompt to playlist.

Wiring, in dependency order:

  chat sources ──▶ Director ──▶ PromptUpsampler (OpenAI-compatible LLM)
                      │
                      ▼ enqueue / move / pop
                    Engine ──▶ FastH3Backend ──▶ FastVideo (4 GPUs)
                      │ frames + audio
                      ▼
                    Pacer ──▶ StreamSink (hls | rtmp | noop)
                                   │
                    webapp ────────┘  the page that plays it and feeds the chat

Everything is one asyncio process and one machine: the generator hands each
built clip straight to the pacer as the arrays it already is. The pacer and
sink start before the model finishes loading, so the stream is live (on black)
while the GPUs warm rather than refusing connections for five minutes.

Usage:
    cp .env.example .env      # keys, preset, sink, chat
    livestream-server         # everything from .env
    python -m livestream.main --sink noop --preset default
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import warnings

from .admin import AdminControl
from .chat import ChatPrompt, ChatSource, TwitchChat, WebChat, YouTubeChat
from .config import Config, load_model_config
from .director import Director
from .engine import MODEL_FPS, MODEL_SAMPLE_RATE, Engine
from .moderator import Moderator
from .overlay import StreamStatusOverlay
from .pacer import Pacer
from .sinks import AudioFormat, VideoFormat, make_sink
from .upsampler import PromptUpsampler
from .webapp import DemoWeb

logger = logging.getLogger("livestream")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_chat_sources(config: Config, commands: tuple[str, ...]) -> list[ChatSource]:
    """One source per configured platform. Add new platforms here.

    `commands` is every command word a source should deliver -- the viewer
    prompt command plus the admin commands; the router tells them apart.
    """
    sources: list[ChatSource] = []
    if config.twitch_channel:
        sources.append(TwitchChat(config.twitch_channel, commands))
    if config.youtube_video_id and config.youtube_api_key:
        sources.append(YouTubeChat(config.youtube_video_id, config.youtube_api_key, commands))
    return sources


async def serve(config: Config) -> None:
    """Build every component, wire them together, and run until one dies."""
    model_config = load_model_config(config.model_config_path)

    # Constructing the engine validates the weights bundle; it does not load
    # the model, so a broken bundle fails here in milliseconds rather than
    # after five minutes of GPU work.
    engine = Engine(config, model_config)

    upsampler = PromptUpsampler(
        api_key=config.openai_api_key,
        model=config.openai_model,
        style=config.style,
        free_viewer_style=config.viewer_free_style,
        max_chunks=config.max_chunks,
        base_url=config.openai_base_url,
    )
    moderator = Moderator(
        api_key=config.moderation_api_key,
        model=config.moderation_model,
        enabled=config.moderation_enabled,
        base_url=config.moderation_base_url,
    )
    if not moderator.enabled:
        logger.warning("moderation is DISABLED (MODERATION_ENABLED=0) — every chat "
                       "prompt reaches the upsampler unchecked")
    director = Director(
        engine,
        upsampler,
        moderator,
        cooldown_s=config.chat_cooldown_s,
        idle_prompts=config.idle_prompts,
        idle_queue_target=config.idle_queue_target,
    )
    admin = AdminControl(config.admin_users, upsampler, director)
    if config.admin_users:
        logger.info(
            "admin commands (%s) enabled for: %s",
            ", ".join(admin.commands),
            ", ".join(sorted(config.admin_users)),
        )
    else:
        logger.info("no admins configured (ADMIN_USERS empty) — admin commands off")

    def route_chat(prompt: ChatPrompt) -> None:
        """Admin commands to the admin handler; everything else is a prompt."""
        if not admin.handle(prompt):
            director.submit(prompt)

    chat_sources = build_chat_sources(config, (config.chat_command, *admin.commands))
    # The demo page is both a chat source and an output: viewers type into the
    # same page they watch on. It is built here, not in build_chat_sources,
    # because the web app needs the source instance to push submissions into.
    web: DemoWeb | None = None
    if config.web_enabled:
        web_chat = WebChat(config.chat_command, (config.chat_command, *admin.commands))
        web = DemoWeb(web_chat, config.hls_dir, host=config.web_host, port=config.web_port)
        engine.add_listener(web.listener)
        chat_sources.append(web_chat)
    if not chat_sources:
        logger.warning("no chat source configured — the stream will run, but nothing "
                       "will feed the queue")

    sink = make_sink(
        config.sink,
        rtmp_url=config.rtmp_url,
        rtmp_video_bitrate_k=config.rtmp_video_bitrate_k,
        hls_dir=config.hls_dir,
    )
    # The canvas is this deployment's own config, not something negotiated with
    # a remote, so the pacer and sink can start immediately -- which is what
    # puts a live (black) stream in front of a viewer during model load.
    width, height = engine.canvas
    overlay = (StreamStatusOverlay(engine, chat_command=config.chat_command)
               if config.overlay_enabled and not config.web_enabled else None)
    pacer = Pacer(
        sink,
        VideoFormat(width=width, height=height, fps=MODEL_FPS),
        AudioFormat(sample_rate=MODEL_SAMPLE_RATE, channels=1),
        overlay=overlay,
    )
    engine.attach_pacer(pacer)

    tasks: list[asyncio.Task] = [
        asyncio.create_task(pacer.run(), name="pacer"),
        asyncio.create_task(engine.run(), name="engine"),
        asyncio.create_task(director.run(), name="director"),
        asyncio.create_task(director.run_playout(), name="playout"),
    ]
    # Gated here because any finished task is a shutdown signal, and run_idle
    # returns immediately when the target is 0. A preset with no idle prompts
    # still gets the task: the filler idles until a `!switch` brings prompts.
    if config.idle_queue_target > 0:
        tasks.append(asyncio.create_task(director.run_idle(), name="idle-filler"))
    else:
        logger.info("idle filler off (IDLE_QUEUE_TARGET=0)")
    tasks += [asyncio.create_task(source.run(route_chat), name=f"chat-{source.name}") for source in chat_sources]
    if web is not None:
        tasks.append(asyncio.create_task(web.run(), name="webapp"))

    logger.info(
        "streaming %dx%d@%dfps to sink=%s (overlay %s, preset %r) — "
        "chat command %r on %s",
        width,
        height,
        MODEL_FPS,
        config.sink,
        "on" if overlay else "off",
        config.preset_name,
        config.chat_command,
        ", ".join(s.name for s in chat_sources) or "nothing",
    )
    try:
        # Run until a task dies (none should) or the process is interrupted.
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.exception() is not None:
                logger.error("task %s died: %s", task.get_name(), task.exception())
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for source in chat_sources:
            with contextlib.suppress(Exception):
                await source.close()
        await sink.stop()
        logger.info("shut down cleanly")


def cli() -> None:
    setup_logging()
    config = Config.load()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(config))


if __name__ == "__main__":
    cli()
