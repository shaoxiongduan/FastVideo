"""The demo's own watch page: video on the left, live chat and queue on the right.

Upstream's client targets Twitch and YouTube, where the page belongs to the
platform, so it burns status into the video pixels (`overlay/`). A private demo
owns its page, which is better in every way that matters here: the queue is real
text rather than baked-in pixels, it costs no encode, and it can show more than
fits legibly on a frame.

Everything the panel shows is read off the model's own message stream. The state
mirror registers with `Engine.add_listener` and rebuilds itself from
`state_update`, `queue_update` and the `clip_*` messages -- the same echo the
director and overlay already trust -- so this module adds no state of its own
that a reconnect could desynchronise, and needs no changes anywhere else.

One HTTP origin serves the whole demo: the page, the HLS segments, the state
websocket, and the chat endpoint. That is deliberate. Cloudflare Tunnel proxies
one port, so "publish the demo" is pointing `cloudflared` at this server and
nothing else.

Routes:
    GET  /              the page
    GET  /assets/<file> the logo and favicon the page references
    GET  /hls/<file>    the HLS playlist and segments (written by `sinks/hls.py`)
    GET  /healthz       whether the engine is loaded and commands take effect
    WS   /state         one JSON snapshot on connect, then one per change
    POST /chat          {"author": str, "text": str} -> a prompt for the director
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .chat.web import WebChat
from .group_tag import parse_group_tag

logger = logging.getLogger("livestream.webapp")

WEB_DIR = Path(__file__).parent / "web"
# Chat lines kept for late joiners. The panel is a live feed, not a log: enough
# to show the room is alive, not so much that a new viewer scrolls history.
CHAT_HISTORY = 60
# Longest prompt text echoed into the panel. The model accepts 800 characters;
# showing all of them would push the queue off the screen.
PROMPT_PREVIEW = 180

# How many clip transitions to keep for the playback timeline. A viewer sits a
# few HLS segments behind live, so the page needs the recent past to work out
# what is on *their* screen, not what the model has just started.
TIMELINE_ENTRIES = 40


def _clip_view(clip: dict[str, Any]) -> dict[str, Any]:
    """One queue entry, flattened for the page.

    Everything here comes from the clip's own `ClipInfo` plus the group tag the
    director wrote into its metadata, which the model echoes untouched.
    """
    tag = parse_group_tag(clip.get("metadata", "")) or {}
    # `prompt` on the wire is what the upsampler wrote -- long, styled, and not
    # what the viewer typed. The director keeps the original in the group tag,
    # so the panel shows that and keeps the rewrite for the hover text: a
    # viewer should recognise their own words in the queue.
    original = tag.get("raw_prompt") or clip.get("prompt") or ""
    return {
        "clip_id": clip.get("clip_id", ""),
        "title": tag.get("title") or "",
        "author": tag.get("author") or "",
        "scene": tag.get("scene"),
        "scenes": tag.get("scenes"),
        "generated": bool(tag.get("generated")),
        # The author of filler is the stream itself; surfacing "auto" as a name
        # invites viewers to read it as another person's request.
        "author_label": ("" if tag.get("generated") else (tag.get("author") or "")),
        "seconds": clip.get("seconds"),
        "ready": bool(clip.get("ready")),
        "prompt": original[:PROMPT_PREVIEW],
        "expanded": (clip.get("prompt") or "")[:PROMPT_PREVIEW],
    }


class DemoState:
    """The panel's view of the model, rebuilt purely from the message stream."""

    def __init__(self) -> None:
        self.generation: list[dict[str, Any]] = []
        self.playout: list[dict[str, Any]] = []
        self.now_playing: dict[str, Any] | None = None
        self.stats: dict[str, Any] = {}
        self.chat: deque[dict[str, Any]] = deque(maxlen=CHAT_HISTORY)
        self.connected = False
        # (wall clock when these frames were emitted, what they are). The page
        # matches its EXT-X-PROGRAM-DATE-TIME playback position against this
        # rather than trusting the live `now_playing`, which runs ahead of the
        # picture by the whole HLS pipeline.
        self.timeline: deque[dict[str, Any]] = deque(maxlen=TIMELINE_ENTRIES)

    @property
    def generating(self) -> dict[str, Any] | None:
        """What the GPUs are building right now.

        Builds consume the generation queue front-first, one at a time, so the
        front entry is the clip in flight -- there is no separate message that
        announces a build starting.
        """
        return self.generation[0] if self.generation else None

    def snapshot(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "now_playing": self.now_playing,
            "timeline": list(self.timeline),
            "server_now": time.time(),
            "generating": self.generating,
            "generation": self.generation,
            "playout": self.playout,
            "stats": self.stats,
            "chat": list(self.chat),
        }

    def note(self, kind: str, text: str, author: str = "") -> None:
        self.chat.append({"kind": kind, "author": author, "text": text, "at": time.time()})

    def on_message(self, kind: str, data: dict[str, Any]) -> None:
        """Fold one model message into the mirror. Never raises: it runs on the link."""
        if kind == "queue_update":
            self.generation = [_clip_view(c) for c in data.get("generation", [])]
            self.playout = [_clip_view(c) for c in data.get("playout", [])]
        elif kind == "state_update":
            self.connected = True
            self.stats = {
                "generation_queued": data.get("generation_queued"),
                "generation_capacity": data.get("generation_capacity"),
                "playout_queued": data.get("playout_queued"),
                "playout_capacity": data.get("playout_capacity"),
                "clips_played": data.get("clips_played"),
                "width": data.get("width"),
                "height": data.get("height"),
            }
            if not data.get("playing"):
                self.now_playing = None
        elif kind == "clip_started":
            self.now_playing = _clip_view(data.get("clip", {}))
            self.timeline.append({"at": time.time(), "clip": self.now_playing})
        elif kind == "clip_queued":
            # One line per group, not per scene: a six-scene story is still one
            # thing somebody asked for. Viewer submissions are already echoed
            # by the POST handler, so only filler is announced here.
            clip = _clip_view(data.get("clip", {}))
            scene = clip.get("scene")
            if clip["generated"] and (scene is None or scene == 1):
                self.note("filler", clip["prompt"], author="filler")
        elif kind in ("clip_finished", "clip_stopped"):
            self.now_playing = None
            # A gap is part of the timeline too, or the panel would keep naming
            # a clip that has already ended for the viewer.
            self.timeline.append({"at": time.time(), "clip": None})
        elif kind == "clip_failed":
            # Failures stay: a viewer whose request vanished deserves to know.
            clip = _clip_view(data.get("clip", {}))
            if not clip["generated"]:
                self.note("error", f"build failed: {clip['title'] or clip['clip_id'][:8]}")


class DemoWeb:
    """The web server: the page, the HLS files, the state socket, the chat box."""

    def __init__(self, chat: WebChat, hls_dir: Path | str, *, host: str = "0.0.0.0", port: int = 8081) -> None:
        self.state = DemoState()
        self._chat = chat
        self._hls_dir = Path(hls_dir)
        self._host, self._port = host, port
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        # One writer, never a task per update. Spawning a task per broadcast let
        # two snapshots race: a client could receive a newer state and then an
        # older one, and the page would render the stale timeline -- the panel
        # jumping back a clip and then forward again. The flag also coalesces
        # bursts into a single send.
        self._dirty: asyncio.Event | None = None
        self._send_lock: asyncio.Lock | None = None
        self._seq = 0
        self.app = self._build_app()

    # ------------------------------------------------------------------ wiring

    def listener(self, kind: str, data: dict[str, Any]) -> None:
        """Hand to `Engine.add_listener`. Folds, then pushes to the page."""
        try:
            self.state.on_message(kind, data)
        except Exception:
            logger.exception("[web] failed to fold %s", kind)
            return
        self.broadcast()

    def broadcast(self) -> None:
        """Mark the state dirty; the writer sends it. Safe from any thread."""
        loop, dirty = self._loop, self._dirty
        if loop is None or dirty is None:
            return
        loop.call_soon_threadsafe(dirty.set)

    async def _pump(self) -> None:
        """The only thing that sends state, so snapshots can never overtake."""
        assert self._dirty is not None
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            if not self._clients:
                continue
            await self._send(self.state.snapshot())

    async def _send(self, snapshot: dict[str, Any]) -> None:
        assert self._send_lock is not None
        async with self._send_lock:
            self._seq += 1
            snapshot["seq"] = self._seq
            payload = json.dumps(snapshot)
            for socket in list(self._clients):
                try:
                    await socket.send_text(payload)
                except Exception:
                    self._clients.discard(socket)

    # -------------------------------------------------------------------- app

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="FastH3 livestream")
        self._hls_dir.mkdir(parents=True, exist_ok=True)
        # The playlist must never be cached: it is rewritten every segment, and
        # a cached copy strands the player on segments that have been deleted.
        app.mount("/hls", StaticFiles(directory=str(self._hls_dir)), name="hls")

        @app.get("/")
        async def index() -> FileResponse:
            """The watch page: video, chat and the queue on one origin.

            Read from disk per request, so iterating on the markup needs no
            restart.
            """
            return FileResponse(WEB_DIR / "index.html")

        # The logo and the favicon the page references.
        app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")

        @app.get("/healthz")
        async def healthz() -> JSONResponse:
            return JSONResponse({"connected": self.state.connected})

        @app.post("/chat")
        async def chat(body: dict[str, Any]) -> JSONResponse:
            author = str(body.get("author") or "viewer")[:32]
            text = str(body.get("text") or "")[:800]
            if not text.strip():
                return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
            accepted = self._chat.submit(author, text)
            self.state.note("viewer" if accepted else "error",
                            text if accepted else f"dropped (queue full): {text[:80]}",
                            author=author)
            self.broadcast()
            return JSONResponse({"ok": accepted})

        @app.websocket("/state")
        async def state_socket(socket: WebSocket) -> None:
            await socket.accept()
            self._clients.add(socket)
            try:
                await self._send(self.state.snapshot())
                while True:
                    # The page never sends; this is just how a disconnect is noticed.
                    await socket.receive_text()
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                self._clients.discard(socket)

        @app.middleware("http")
        async def no_cache_playlist(request, call_next) -> Response:
            response = await call_next(request)
            if request.url.path.endswith(".m3u8") or request.url.path in ("/", ""):
                response.headers["Cache-Control"] = "no-store"
            return response

        return app

    async def run(self) -> None:
        """Serve until cancelled. One task in `main.py`'s gather."""
        import uvicorn

        self._loop = asyncio.get_running_loop()
        self._dirty = asyncio.Event()
        self._send_lock = asyncio.Lock()
        pump = asyncio.create_task(self._pump(), name="webapp-pump")
        config = uvicorn.Config(self.app, host=self._host, port=self._port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        logger.info("[web] serving the demo page on http://%s:%d", self._host, self._port)
        try:
            await server.serve()
        finally:
            pump.cancel()
