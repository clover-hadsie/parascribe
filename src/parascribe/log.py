"""Logging configuration.

Operational logs (request ids, timings, counts) are content-free and emitted at
the configured ``log_level`` (default INFO). ``debug_logging`` forces DEBUG and is
the ONLY switch that permits content (transcript text, language) into the logs;
it logs a loud warning when enabled (invariant #3).

Call :func:`configure_logging` once at startup. It owns the ``parascribe`` logger
(its own handler, ``propagate=False``) so app logs appear regardless of how the
uvicorn/root loggers are configured.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parascribe.config import Settings

_LOGGER_NAME = "parascribe"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s rid=%(rid)s | %(message)s"
_PLAIN_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


class _RidFilter(logging.Filter):
    """Ensure every record has a ``rid`` attribute so the format never errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "rid"):
            record.rid = "-"
        return True


def configure_logging(settings: Settings) -> None:
    level = logging.DEBUG if settings.debug_logging else settings.log_level.upper()
    handler = logging.StreamHandler()
    handler.addFilter(_RidFilter())
    handler.setFormatter(logging.Formatter(_FORMAT))

    pkg = logging.getLogger(_LOGGER_NAME)
    pkg.handlers.clear()
    pkg.addHandler(handler)
    pkg.setLevel(level)
    pkg.propagate = False

    if settings.debug_logging:
        pkg.warning("debug_logging is ON: transcript content may appear in logs.")


def debug_enabled() -> bool:
    return logging.getLogger(_LOGGER_NAME).isEnabledFor(logging.DEBUG)
