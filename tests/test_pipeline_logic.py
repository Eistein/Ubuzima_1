"""tests/test_pipeline_logic.py — pure pipeline decision logic (consent, audio, band, errors)."""
import numpy as np
from pipeline_logic import (
    Action, consent_ok, audio_present, action_for_band,
    ProviderErrorKind, classify_status_code,
)


class TestConsentGate:
    def test_blocks_without_consent(self):
        assert consent_ok(require_consent=True, consent_given=False) is False

    def test_allows_with_consent(self):
        assert consent_ok(require_consent=True, consent_given=True) is True

    def test_gate_off_always_allows(self):
        assert consent_ok(require_consent=False, consent_given=False) is True


class TestAudioPresent:
    def test_none_is_absent(self):
        assert audio_present(None) is False

    def test_empty_is_absent(self):
        assert audio_present(np.array([], dtype=np.float32)) is False

    def test_nonempty_is_present(self):
        assert audio_present(np.zeros(16000, dtype=np.float32)) is True


class TestActionForBand:
    def test_high_answers(self):
        assert action_for_band("high") is Action.ANSWER

    def test_medium_hedges(self):
        assert action_for_band("medium") is Action.ANSWER_WITH_HEDGE

    def test_low_declines(self):
        assert action_for_band("low") is Action.DECLINE_LOW_CONFIDENCE

    def test_unexpected_declines_fail_safe(self):
        # Fail closed: an unexpected band must NOT reach the LLM.
        assert action_for_band("garbled") is Action.DECLINE_LOW_CONFIDENCE
        assert action_for_band("") is Action.DECLINE_LOW_CONFIDENCE


class TestErrorClassification:
    def test_auth(self):
        assert classify_status_code(401) is ProviderErrorKind.AUTH
        assert classify_status_code(403) is ProviderErrorKind.AUTH

    def test_rate_limit(self):
        assert classify_status_code(429) is ProviderErrorKind.RATE_LIMIT

    def test_server_unavailable(self):
        assert classify_status_code(503) is ProviderErrorKind.UNAVAILABLE

    def test_unknown(self):
        assert classify_status_code(418) is ProviderErrorKind.UNKNOWN
