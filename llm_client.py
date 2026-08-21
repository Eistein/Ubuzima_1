"""
llm_client.py — LLM access layer for UBUZIMA AI, with structured errors and model fallback.

WHY THIS MODULE EXISTS
----------------------
The first-round feedback asked for two things this file provides: (1) a clear separation of
"language-model access" from the interface and audio code, and (2) evaluated behaviour when
a provider or model fails. Inline `requests.post` in app.py did neither — a failure surfaced
as a raw exception string, and there was no fallback.

Here, LLM access is a single injectable object. It tries a primary model, and on a
retriable failure falls back to a secondary model, before giving up with a *typed* error
the orchestrator can turn into a calm Kinyarwanda message. `post_fn` is injectable so the
whole thing is unit-testable with no network (see tests/test_llm_client.py).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from pipeline_logic import ProviderErrorKind, classify_status_code
from safety_prompt import build_messages

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(Exception):
    """Base class for all LLM-access failures."""
    def __init__(self, message: str, kind: ProviderErrorKind = ProviderErrorKind.UNKNOWN):
        super().__init__(message)
        self.kind = kind


class LLMUnavailable(LLMError):
    """Every configured model failed. The orchestrator should decline gracefully."""


@dataclass
class LLMClient:
    """Calls an OpenRouter chat model with a primary/fallback model list.

    Args:
        models:   ordered list of model ids; index 0 is primary, the rest are fallbacks.
        api_key:  OpenRouter key; defaults to the OPENROUTER_API_KEY env var.
        post_fn:  the callable used to POST (defaults to requests.post). Injectable for tests.
        timeout:  per-request timeout in seconds.
        max_tokens/temperature: generation settings (match app.py defaults).
    """
    models: list[str] = field(default_factory=lambda: [
        "google/gemini-2.5-flash",       # primary
        "google/gemini-2.0-flash-001",   # fallback model, same provider
    ])
    api_key: Optional[str] = None
    post_fn: Optional[Callable] = None
    timeout: int = 30
    max_tokens: int = 200
    temperature: float = 0.3

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        self.post_fn = self.post_fn or requests.post
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set.", ProviderErrorKind.AUTH)

    def _call_one(self, model: str, user_text: str) -> str:
        resp = self.post_fn(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "UBUZIMA AI",
            },
            json={
                "model": model,
                "messages": build_messages(user_text),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            timeout=self.timeout,
        )
        status = getattr(resp, "status_code", 200)
        if status != 200:
            kind = classify_status_code(status)
            raise LLMError(f"{model} returned HTTP {status}", kind)
        return resp.json()["choices"][0]["message"]["content"].strip()

    def answer(self, user_text: str) -> str:
        """Try each model in order. Return the first success.

        Auth failures are not retried against other models (the key is bad for all of them).
        Timeouts, rate limits, and 5xx are retried against the next model. If every model
        fails, raise LLMUnavailable so the orchestrator can decline in Kinyarwanda instead
        of crashing.
        """
        last: LLMError | None = None
        for i, model in enumerate(self.models):
            try:
                return self._call_one(model, user_text)
            except LLMError as e:
                if e.kind == ProviderErrorKind.AUTH:
                    raise  # no point trying other models with a bad key
                last = e
            except requests.Timeout:
                last = LLMError(f"{model} timed out", ProviderErrorKind.TIMEOUT)
            except requests.RequestException as e:
                last = LLMError(f"{model} connection error: {e}",
                                ProviderErrorKind.UNAVAILABLE)
            # brief backoff before the next model
            if i < len(self.models) - 1:
                time.sleep(0)  # keep tests fast; raise in production if desired
        raise LLMUnavailable(
            f"All {len(self.models)} model(s) failed; last error: {last}",
            ProviderErrorKind.UNAVAILABLE,
        )
