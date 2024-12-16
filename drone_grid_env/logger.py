from __future__ import annotations

import warnings

from gymnasium.utils import colorize

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40


_level = INFO


def set_level(level: int) -> None:
    global _level
    _level = level


def debug(msg: str, *args: object) -> None:
    if _level <= DEBUG:
        print(colorize(f"DEBUG: {msg % args}", "blue"))


def info(msg: str, *args: object) -> None:
    if _level <= INFO:
        print(colorize(f"INFO: {msg % args}", "green"))


def warn(msg: str, *args: object, category: type[Warning] | None = None, stacklevel: int = 1) -> None:
    if _level <= WARN:
        warnings.warn(
            colorize(f"WARN: {msg % args}", "yellow"),
            category=category,
            stacklevel=stacklevel + 1,
        )


def error(msg: str, *args: object, category: type[Warning] | None = None, stacklevel: int = 1) -> None:
    if _level <= ERROR:
        warnings.warn(
            colorize(f"ERROR: {msg % args}", "red"),
            category=category,
            stacklevel=stacklevel + 1,
        )
