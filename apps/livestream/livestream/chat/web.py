"""Web chat source: prompts typed into the demo's own page rather than Twitch.

The shipped sources read a public platform because that is where a real
broadcast's audience is. A private demo has no such audience: the people
watching are the people holding the link, and the chat lives on the same page
as the video. So this source is fed directly, in-process, by the web app
(`webapp.py`) instead of polling anything.

It keeps the `ChatSource` contract exactly, which is what lets `main.py` treat
it like any other platform: the director, the per-author cooldown, the
moderation hook and the scene-group machinery are all unchanged and unaware
that the messages came from a form instead of IRC.

`submit` is deliberately non-async and never blocks: it is called from a
request handler, and a full queue drops the message rather than stalling the
web server. Dropping is right here -- the director already refuses prompts when
its backlog is full, and a viewer who sees nothing happen will simply type
again.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from .base import ChatPrompt, ChatSource, match_command

logger = logging.getLogger("livestream.chat.web")

# Messages held between the web request that accepted one and the director
# picking it up. Small on purpose: a deep queue would let a burst of typing
# commit the stream to minutes of stale prompts.
QUEUE_SIZE = 32


class WebChat(ChatSource):
    """Prompts submitted through the demo page's chat box."""

    name = "web"

    def __init__(self, command: str = "!prompt", commands: tuple[str, ...] = ()) -> None:
        self._command = command
        # Every word the router understands, admin commands included. The box is
        # the only way in on this deployment, so `!switch` has to work from it;
        # without this an admin command was delivered as a video prompt and the
        # stream cheerfully generated a clip of someone saying "!switch".
        self._commands = tuple(commands) or (command, )
        self._queue: asyncio.Queue[ChatPrompt] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._dropped = 0

    def submit(self, author: str, text: str, command: str | None = None) -> bool:
        """Accept one message from the web app. True when it was queued.

        Called from the request handler, so it never awaits: a full queue is
        reported to the caller (and the page tells the viewer) rather than
        applying backpressure to the web server.
        """
        text = text.strip()
        if not text:
            return False
        # A leading command word wins; anything else is a plain prompt, because
        # typing into this box is itself the request.
        matched = match_command(text, self._commands)
        word, body = matched if matched else (self._command, text)
        prompt = ChatPrompt(
            source=self.name,
            author=author or "viewer",
            text=body,
            command=command or word,
        )
        try:
            self._queue.put_nowait(prompt)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("[web] queue full, dropped prompt from %s (%d total)", prompt.author, self._dropped)
            return False
        return True

    async def run(self, on_prompt: Callable[[ChatPrompt], None]) -> None:
        logger.info("[web] chat source ready (queue %d)", QUEUE_SIZE)
        while True:
            prompt = await self._queue.get()
            on_prompt(prompt)
