"""
pipeline_logic.py — pure decision logic for the UBUZIMA AI pipeline.

Every function here is a pure function of its inputs: no torch, no network, no Gradio, no
global state. That is the point. The orchestration in app.py's `safe_pipeline` interleaves
these decisions with I/O, which makes the decisions hard to test directly. Extracting them
here means the consent gate, the empty-audio guard, the confidence-band → action mapping,
and provider-error classification can each be unit-tested exhaustively and cannot silently
diverge from what the report claims.

app.py can adopt these by importing them and replacing the equivalent inline checks; the
behaviour is identical by construction (see tests/test_pipeline_logic.py).
"""

from enum import Enum


class Action(str, Enum):
    """What the orchestrator should do after ASR + confidence scoring."""
    REFUSE_NO_CONSENT = "refuse_no_consent"
    REFUSE_NO_AUDIO = "refuse_no_audio"
    DECLINE_LOW_CONFIDENCE = "decline_low_confidence"
    ANSWER = "answer"
    ANSWER_WITH_HEDGE = "answer_with_hedge"


def consent_ok(require_consent: bool, consent_given: bool) -> bool:
    """No audio is processed without affirmative consent when the gate is on."""
    return (not require_consent) or bool(consent_given)


def audio_present(audio_array) -> bool:
    """True only if there is actually a non-empty audio buffer to process."""
    if audio_array is None:
        return False
    try:
        return len(audio_array) > 0
    except TypeError:
        return False


def action_for_band(band: str) -> Action:
    """Map a confidence band to the orchestrator's action.

    high   -> answer
    medium -> answer, but hedge (tell the user to check the transcript)
    low / anything unexpected -> decline WITHOUT calling the LLM (fail-safe).
    """
    if band == "high":
        return Action.ANSWER
    if band == "medium":
        return Action.ANSWER_WITH_HEDGE
    # low, empty, or any unexpected value: decline. Fail closed.
    return Action.DECLINE_LOW_CONFIDENCE


class ProviderErrorKind(str, Enum):
    AUTH = "auth"            # bad/expired key, 401/403
    RATE_LIMIT = "rate_limit"  # 429
    TIMEOUT = "timeout"     # network timeout
    UNAVAILABLE = "unavailable"  # 5xx / connection error
    UNKNOWN = "unknown"


def classify_status_code(status_code: int) -> ProviderErrorKind:
    """Classify an HTTP status from the LLM provider into an actionable kind."""
    if status_code in (401, 403):
        return ProviderErrorKind.AUTH
    if status_code == 429:
        return ProviderErrorKind.RATE_LIMIT
    if 500 <= status_code <= 599:
        return ProviderErrorKind.UNAVAILABLE
    return ProviderErrorKind.UNKNOWN
