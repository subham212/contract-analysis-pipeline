"""
logger.py
=========
Centralized logging configuration.
"""
from __future__ import annotations

import logging
import sys

from .config import settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger
