"""llm_service.py — Gemini reasoning layer (Section 4.1.5, LLMService in Figure 3.6).

Model policy: try Pro, fall back to Flash.

Gemini 2.5 Pro is on Google's free tier, but at much tighter limits than Flash —
single-digit requests per minute and tens-to-low-hundreds per day, and Google
changes these often. Pinning Pro alone means the demo dies the moment you hit the
daily cap, which is a bad thing to discover during a defence. Pinning Flash alone
means never finding out whether Pro writes better Kinyarwanda.

So: Pro for quality, Flash when Pro is rate-limited, and the app always tells you
which one actually answered. Same principle as the ASR adapter and the TTS voice —
the system never silently runs something other than what you configured.

The API key is a Hugging Face Space secret and is never committed (NFR4).
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)

PRIMARY = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
FALLBACK = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MAX_RETRIES = 2


class LLMConfigurationError(RuntimeError):
    pass


def _is_quota_error(e: Exception) -> bool:
    m = str(e).lower()
    return any(k in m for k in ("429", "quota", "rate limit", "resource_exhausted",
                                "resource exhausted"))


class LLMService:
    """Wraps the Gemini SDK behind the common invoke() signature."""

    def __init__(self, primary: str = PRIMARY, fallback: str = FALLBACK,
                 api_key: str = API_KEY):
        if not api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not set. In your Space: Settings → Variables and "
                "secrets → New secret → name GEMINI_API_KEY. Never put the key in code."
            )
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai

        # Don't assume a model string resolves. The SDK pinned in requirements.txt
        # predates the 2.5 line; if it cannot see these models, better to find out at
        # start-up than on the first user question.
        available = self._list_models()
        self.primary = self._pick(primary, available)
        self.fallback = self._pick(fallback, available)
        if not self.primary and not self.fallback:
            raise LLMConfigurationError(
                f"Neither {primary} nor {fallback} is available to this SDK. "
                f"Visible models: {sorted(available)[:12]}. Upgrade "
                "google-generativeai in requirements.txt."
            )
        self.primary = self.primary or self.fallback
        self.fallback = self.fallback or self.primary

        self._models = {}
        self.last_model = None
        self.calls = {}
        log.info("LLM ready — primary=%s fallback=%s", self.primary, self.fallback)

    def _list_models(self) -> set[str]:
        try:
            return {m.name.split("/")[-1] for m in self._genai.list_models()
                    if "generateContent" in getattr(m, "supported_generation_methods", [])}
        except Exception as e:
            log.warning("Could not list models (%s); trusting the configured strings", e)
            return set()

    @staticmethod
    def _pick(want: str, available: set[str]) -> str | None:
        if not available:
            return want                        # listing failed — trust and find out
        if want in available:
            return want
        # tolerate dated suffixes, e.g. gemini-2.5-pro-preview-06-05
        for m in sorted(available):
            if m.startswith(want):
                log.info("Resolved %s → %s", want, m)
                return m
        log.error("Model %r not available to this SDK", want)
        return None

    def _model(self, name: str):
        if name not in self._models:
            self._models[name] = self._genai.GenerativeModel(name)
        return self._models[name]

    @property
    def description(self) -> str:
        if not self.calls:
            return f"{self.primary} (fallback {self.fallback}) — no calls yet"
        split = ", ".join(f"{k}: {v}" for k, v in sorted(self.calls.items()))
        return f"{self.primary} → {self.fallback} | answered by — {split}"

    def invoke(self, prompted_text: str) -> tuple[str, str]:
        """Return (response_text, model_that_answered).

        Tries the primary model, then the fallback if the primary is rate-limited.
        A non-quota error on the primary is NOT swallowed — falling back on a real
        error would hide a genuine fault behind a quieter model.
        """
        chain = [self.primary] if self.primary == self.fallback else [self.primary, self.fallback]
        last: Exception | None = None

        for i, name in enumerate(chain):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = self._model(name).generate_content(
                        prompted_text,
                        generation_config={"temperature": 0.3, "max_output_tokens": 512},
                    )
                    text = (resp.text or "").strip()
                    if not text:
                        raise RuntimeError("empty response (possibly a safety block)")
                    self.last_model = name
                    self.calls[name] = self.calls.get(name, 0) + 1
                    return text, name
                except Exception as e:
                    last = e
                    if _is_quota_error(e):
                        if i < len(chain) - 1:
                            log.warning("%s rate-limited — falling back to %s",
                                        name, chain[i + 1])
                            break                      # straight to the next model
                        raise                          # nothing left to fall back to
                    log.warning("%s attempt %d/%d failed: %s", name, attempt,
                                MAX_RETRIES, e)
                    if attempt < MAX_RETRIES:
                        time.sleep(1.5 * attempt)
                    else:
                        raise                          # real error — surface it
        raise RuntimeError(f"All models failed: {last}")
