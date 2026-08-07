"""Structured standard-library logging.

:func:`configure_logging` is idempotent and never emits secrets; it only formats
records the application explicitly logs. Modules obtain loggers via
:func:`get_logger` rather than configuring logging themselves.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_ROOT_LOGGER_NAME = "retail_clickstream_ai"

_configured = False


def configure_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure root logging once and return the project logger.

    Safe to call multiple times: the first call installs the handler/format;
    later calls only adjust the level.
    """
    global _configured
    if not _configured:
        logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        _configured = True
    else:
        logging.getLogger().setLevel(level)
    return get_logger()


def get_logger(name: str = _ROOT_LOGGER_NAME) -> logging.Logger:
    """Return a named project logger."""
    return logging.getLogger(name)
