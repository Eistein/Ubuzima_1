"""
logging_setup.py — one consistent, privacy-safe logger for UBUZIMA AI.

Centralises logging config so every module logs the same way, and so the app logs
*operational* facts (stage, latency, confidence band, error kind) WITHOUT ever logging
audio, transcripts, or answers. That last point is a privacy requirement, not a style
choice: the Terms tab promises that voice input is not retained, and a logger that printed
transcripts would break that promise.
"""

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once. Level from LOG_LEVEL env var (default INFO)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stdout,  # Railway captures stdout
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def log_turn(logger: logging.Logger, *, stage: str, latency_s: float | None = None,
             band: str | None = None, error_kind: str | None = None) -> None:
    """Log one operational event. NEVER pass transcript/answer/audio here."""
    parts = [f"stage={stage}"]
    if latency_s is not None:
        parts.append(f"latency_s={latency_s:.2f}")
    if band is not None:
        parts.append(f"band={band}")
    if error_kind is not None:
        parts.append(f"error_kind={error_kind}")
    logger.info(" ".join(parts))
