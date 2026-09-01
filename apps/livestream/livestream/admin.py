"""Admin chat commands: live control of the stream from trusted names.

Admins are the usernames in `ADMIN_USERS`, matched case-insensitively.

The router in `main.py` offers every matched command here before the director
sees it, so an admin command costs no cooldown slot, moderation call or LLM
call -- while a `!prompt` from an admin is still just a prompt. A recognised
admin command from a non-admin is consumed and logged.

`!switch <preset>` swaps the creative preset live. The name is resolved
against `presets/` at that moment, so a new JSON dropped in the folder is
switchable without a restart. Style and idle prompts change immediately, and
both queues are flushed down to one buffer clip so the new identity reaches
the stream in about one clip's time rather than after a whole queue drains.
"""

from __future__ import annotations

import asyncio
import logging

from .chat import ChatPrompt
from .config import PresetError, available_presets, load_preset
from .director import Director
from .upsampler import PromptUpsampler

logger = logging.getLogger(__name__)

_SWITCH_COMMAND = "!switch"


class AdminControl:
    """Recognize and execute admin commands arriving from chat."""

    #: Every command word this handler owns; chat sources match on these
    #: (plus the viewer prompt command) so the router can dispatch here.
    commands: tuple[str, ...] = (_SWITCH_COMMAND, )

    def __init__(
        self,
        admin_users: frozenset[str],
        upsampler: PromptUpsampler,
        director: Director,
    ) -> None:
        self._admins = admin_users
        self._upsampler = upsampler
        self._director = director
        self._flush_task: asyncio.Task | None = None

    def is_admin(self, prompt: ChatPrompt) -> bool:
        """Whether the message's author is on the admin list."""
        author = prompt.author.lower()
        return author in self._admins or f"{prompt.source}:{author}" in self._admins

    def handle(self, prompt: ChatPrompt) -> bool:
        """Execute one admin command; True when the message was one.

        False means the message is not an admin command and should continue
        to the director as a viewer prompt. Synchronous and never raises —
        it is called inline from the chat sources' callback.
        """
        if prompt.command not in self.commands:
            return False
        if not self.is_admin(prompt):
            logger.info(
                "[admin] ignoring %s from non-admin %s@%s",
                prompt.command,
                prompt.author,
                prompt.source,
            )
            return True
        if prompt.command == _SWITCH_COMMAND:
            self._switch_preset(prompt)
        return True

    def _switch_preset(self, prompt: ChatPrompt) -> None:
        """`!switch <preset>`: re-point style + idle prompts at another preset."""
        name = prompt.text.split()[0]
        # Only bare names from the presets/ folder — never paths, so chat
        # input cannot point the loader at an arbitrary file.
        if name not in available_presets():
            logger.warning(
                "[admin] %s@%s: unknown preset %r (available: %s)",
                prompt.author,
                prompt.source,
                name,
                ", ".join(available_presets()) or "none",
            )
            return
        try:
            preset = load_preset(name)
        except PresetError as error:
            logger.warning(
                "[admin] %s@%s: preset %r failed to load: %s — keeping the "
                "current one",
                prompt.author,
                prompt.source,
                name,
                error,
            )
            return
        self._upsampler.set_style(preset["style"])
        self._director.set_idle_prompts(preset["idle_prompts"])
        # Handled here synchronously (chat callbacks run on the event loop);
        # the flush itself pops clips over the wire, so it runs as a task.
        # The reference keeps it from being garbage-collected mid-flight.
        self._flush_task = asyncio.create_task(self._director.flush_stale_clips(), name="preset-switch-flush")
        logger.info(
            "[admin] %s@%s switched preset to %r (%d idle prompts); "
            "flushing stale queued clips",
            prompt.author,
            prompt.source,
            name,
            len(preset["idle_prompts"]),
        )
