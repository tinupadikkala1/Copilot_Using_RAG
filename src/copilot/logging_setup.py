"""Logging configuration via dictConfig."""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path


def configure_logging(config_path: str | Path = "") -> None:
    """Configure structured JSON logging.

    If a JSON config file is provided, it is loaded as a dictConfig.
    Otherwise a sensible default is applied.
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            import json

            with open(path) as fh:
                cfg = json.load(fh)
            logging.config.dictConfig(cfg)
            return

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": "%(message)s",
                    "class": "logging.Formatter",
                },
                "simple": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "simple",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"],
            },
            "loggers": {
                "copilot": {
                    "level": "DEBUG",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "httpx": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "chromadb": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    )
