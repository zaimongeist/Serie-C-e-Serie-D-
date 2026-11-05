"""Structured (JSON lines) logging for every module.

Usage::

    from src.common.logging_config import get_logger
    log = get_logger(__name__)
    log.info("fetched", extra={"url": url, "status": 200})

Any ``extra`` keys are emitted as top-level JSON fields.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Serialise each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Compact, readable one-line formatter for interactive use."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        base = f"{datetime.fromtimestamp(record.created).strftime('%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


_CONFIGURED = False


def configure_logging(level: str | int | None = None, *, json_output: bool | None = None, force: bool = False) -> None:
    """Configure the root logger once. Env: ``SERIEC_LOG_LEVEL``, ``SERIEC_LOG_JSON``."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    if level is None:
        level = os.getenv("SERIEC_LOG_LEVEL", "INFO")
    if json_output is None:
        json_output = os.getenv("SERIEC_LOG_JSON", "0").lower() in {"1", "true", "yes"}
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else HumanFormatter())
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else str(level).upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
