"""Consistent logging across the project.

:func:`get_logger` returns a logger that writes coloured-ish, timestamped lines
to stderr and, optionally, to a rotating file inside the project ``logs``
directory. Handlers are de-duplicated so repeated calls (common in notebooks)
do not multiply log output.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track which (logger-name, target) pairs already have a handler attached.
_configured: set[tuple[str, str]] = set()


def _level_from_env(default: int) -> int:
    raw = os.environ.get("FINAI_LOG_LEVEL")
    if not raw:
        return default
    resolved = logging.getLevelName(raw.strip().upper())
    return resolved if isinstance(resolved, int) else default


def get_logger(
    name: str = "finai",
    *,
    log_dir: str | os.PathLike[str] | None = None,
    filename: str | None = None,
    level: int | None = None,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name; use dotted names (``"finai.train"``) to namespace output.
    log_dir:
        When provided, a rotating file handler writing to
        ``<log_dir>/<filename>`` is attached in addition to the console handler.
    filename:
        File name for the file handler. Defaults to ``"<top-level name>.log"``.
    level:
        Logging level. Falls back to ``$FINAI_LOG_LEVEL`` then ``INFO``.
    """
    resolved_level = level if level is not None else _level_from_env(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    # Do not double-emit through the root logger's handlers.
    logger.propagate = False

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    console_key = (name, "console")
    if console_key not in _configured:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(formatter)
        logger.addHandler(console)
        _configured.add(console_key)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_name = filename or f"{name.split('.')[0]}.log"
        target = directory / file_name
        file_key = (name, str(target))
        if file_key not in _configured:
            file_handler = RotatingFileHandler(
                target, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            _configured.add(file_key)

    return logger


__all__ = ["get_logger"]
