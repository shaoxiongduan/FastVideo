"""A `get_logger` that accepts structured keyword fields.

The generation code was written against a structured logger, so it calls
``logger.info("built the thing", frames=345, seconds=12.1)``. Standard
:mod:`logging` rejects unknown keywords, so this thin adapter folds them onto
the end of the message instead of forcing every call site to pre-format.

Deliberately not a logging.Logger subclass and not a LoggerAdapter: those
inherit a large surface we would have to keep consistent, where all that is
needed is five methods that flatten kwargs.
"""

from __future__ import annotations

import logging
from typing import Any


def _fold(message: str, fields: dict[str, Any]) -> str:
    if not fields:
        return message
    return f"{message} " + " ".join(f"{key}={value!r}" for key, value in fields.items())


class StructuredLogger:
    """Wraps one stdlib logger, folding keyword fields into the message."""

    __slots__ = ("_logger", )

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, message: str, **fields: Any) -> None:
        self._logger.debug(_fold(message, fields))

    def info(self, message: str, **fields: Any) -> None:
        self._logger.info(_fold(message, fields))

    def warning(self, message: str, **fields: Any) -> None:
        self._logger.warning(_fold(message, fields))

    def error(self, message: str, **fields: Any) -> None:
        self._logger.error(_fold(message, fields))

    def exception(self, message: str, **fields: Any) -> None:
        self._logger.exception(_fold(message, fields))


def get_logger(name: str) -> StructuredLogger:
    """A structured-field logger writing through the stdlib logger *name*."""
    return StructuredLogger(logging.getLogger(name))


__all__ = ["StructuredLogger", "get_logger"]
