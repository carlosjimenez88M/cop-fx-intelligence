"""
cop_fx.logger
=============
Drop-in coloured logger.  Import it anywhere — no relative imports needed.

Usage
-----
    from cop_fx.logger import get_logger

    log = get_logger(__name__)
    log.debug("starting…")
    log.info("rate fetched: %s", 4200.5)
    log.success("pipeline complete")   # bright green ✔
    log.warning("BanRep fallback used")
    log.error("fetch failed: %s", err)
    log.critical("unrecoverable error")

Colour map
----------
  DEBUG    — dim white
  INFO     — cyan
  SUCCESS  — bright green   (custom level 25, between INFO and WARNING)
  WARNING  — yellow
  ERROR    — red
  CRITICAL — bright red + bold
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# ANSI escape codes
# ---------------------------------------------------------------------------

_RESET = "\033[0m"

_COLOURS = {
    "DEBUG": "\033[2;37m",  # dim white
    "INFO": "\033[36m",  # cyan
    "SUCCESS": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[1;31m",  # bold bright red
}

# ---------------------------------------------------------------------------
# Custom SUCCESS level (25 — sits between INFO=20 and WARNING=30)
# ---------------------------------------------------------------------------

SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")


def _success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS):
        self._log(SUCCESS, message, args, **kwargs)


logging.Logger.success = _success  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Coloured formatter
# ---------------------------------------------------------------------------


class _ColouredFormatter(logging.Formatter):
    """Adds ANSI colour codes around the level name and resets afterwards."""

    _FMT = "%(asctime)s  {colour}%(levelname)-8s{reset}  %(name)s  —  %(message)s"
    _DATE_FMT = "%H:%M:%S"

    def __init__(self, *, use_colour: bool = True) -> None:
        super().__init__(datefmt=self._DATE_FMT)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        if self._use_colour:
            colour = _COLOURS.get(level, "")
            reset = _RESET
        else:
            colour = reset = ""

        formatter = logging.Formatter(
            fmt=self._FMT.format(colour=colour, reset=reset),
            datefmt=self._DATE_FMT,
        )
        return formatter.format(record)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_handler: logging.StreamHandler[TextIO] | None = None  # single shared handler


def _get_handler(stream: TextIO = sys.stdout) -> logging.StreamHandler[TextIO]:
    global _handler
    if _handler is None:
        # Colours on by default; set NO_COLOR=1 to disable (POSIX convention).
        use_colour = "NO_COLOR" not in os.environ
        _handler = logging.StreamHandler(stream)
        _handler.setFormatter(_ColouredFormatter(use_colour=use_colour))
    return _handler


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return (or create) a named coloured logger.

    Each unique ``name`` gets its own Logger instance with the shared
    coloured StreamHandler attached once — safe to call multiple times.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_get_handler())
        logger.propagate = False
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Module-level convenience logger used internally by cop_fx packages
# ---------------------------------------------------------------------------

root_logger = get_logger("cop_fx")
