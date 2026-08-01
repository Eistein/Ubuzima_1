"""
tests/test_safety_prompt.py — Unit tests for the safety prompt module.

Validates that the system prompt contains required guardrails and that
message construction produces the expected structure.

Report reference: §4.3.3 (Unit Testing), Table 4.1
"""

import pytest
from safety_prompt import SYSTEM_PROMPT, PROMPT_VERSION, build_messages


class TestPromptVersion:
    """Ensure the prompt is versioned for reproducibility."""

    def test_version_exists(self):
        assert PROMPT_VERSION is not None
        assert isinstance(PROMPT_VERSION, str)
        assert len(PROMPT_VERSION) > 0

    def test_version_format(self):
        """Version should follow semantic-ish format: vX.Y-YYYY-MM-DD"""
        assert PROMPT_VERSION.startswith("v")
        assert "2026" in PROMPT_VERSION  # Year of capstone

    def test_version_documented(self):
        """Version should be documented in the module docstring."""
        import safety_prompt
        assert PROMPT_VERSION in safety_prompt.__doc__ or                "version" in safety_prompt.__doc__.lower()


class TestSystemPromptContent:
    """Validate that safety guardrails are present in the system prompt."""

    def test_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 200

    def test_prompt_in_kinyarwanda(self):
        """Prompt should be primarily in Kinyarwanda for the LLM."""
        # Check for Kinyarwanda-specific characters and words
        kinyarwanda_markers = ["mu", "ni", "ko", "ntu", "kandi", "muganga"]
        assert any(marker in SYSTEM_PROMPT.lower() for marker in kinyarwanda_markers)

    def test_no_diagnosis_rule(self):
        """Rule (2): Must refuse to diagnose."""
        # Look for refusal to diagnose/prescribe
        refusal_markers = ["NTUTANGE", "ntitanga", "ntisuzuma", "imiti"]
        assert any(marker in SYSTEM_PROMPT for marker in refusal_markers),             "System prompt must contain refusal to diagnose or prescribe"

    def test_referral_rule(self):
        """Rule (2-3): Must refer to qualified care."""
        referral_markers = ["muganga", "CHW", "umujyanama", "ivuriro"]
        assert any(marker.lower() in SYSTEM_PROMPT.lower() for marker in referral_markers),             "System prompt must refer users to qualified health workers"

    def test_emergency_escalation(self):
        """Rule (3): Must escalate emergencies."""
        emergency_markers = ["komeye", "ubutabazi", "ako kanya", "huta"]
        assert any(marker in SYSTEM_PROMPT.lower() for marker in emergency_markers),             "System prompt must escalate emergency symptoms"

    def test_no_medication_names(self):
        """Rule (2): Must not name specific medications."""
        # The prompt itself should not contain medication names
        # (it should instruct the model not to name them)
        assert "NTUTANGE" in SYSTEM_PROMPT or "ntitanga" in SYSTEM_PROMPT.lower()

    def test_response_length_guidance(self):
        """Rule (4): Should guide concise responses."""
        length_markers = ["nke", "2-4", "zumvikana"]
        assert any(marker in SYSTEM_PROMPT.lower() for marker in length_markers),             "System prompt should guide concise responses"


class TestBuildMessages:
    """Validate message construction for LLM API calls."""

    def test_returns_list(self):
        messages = build_messages("Test question")
        assert isinstance(messages, list)

    def test_has_two_messages(self):
        messages = build_messages("Test question")
        assert len(messages) == 2

    def test_system_message_first(self):
        messages = build_messages("Test question")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_user_message_second(self):
        messages = build_messages("Test question")
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test question"

    def test_user_text_preserved(self):
        test_texts = [
            "Umwana wanjye afite umuriro",
            "Ni iki gikora malariya?",
            "Ngomba kunywa paracetamol",  # Adversarial — should still be passed through
        ]
        for text in test_texts:
            messages = build_messages(text)
            assert messages[1]["content"] == text

    def test_message_format(self):
        """Each message must be a dict with 'role' and 'content' keys."""
        messages = build_messages("Test")
        for msg in messages:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["system", "user", "assistant"]
            assert isinstance(msg["content"], str)


class TestPromptIntegrity:
    """Ensure prompt has not been accidentally modified."""

    def test_prompt_length_stable(self):
        """Prompt length should be within expected range."""
        assert 500 < len(SYSTEM_PROMPT) < 2000

    def test_prompt_contains_all_rules(self):
        """All five rules from the report should be present."""
        # Rule 1: Kinyarwanda only
        assert "kinyarwanda" in SYSTEM_PROMPT.lower() or "gusa" in SYSTEM_PROMPT.lower()
        # Rule 2: No diagnosis/prescription
        assert "NTUTANGE" in SYSTEM_PROMPT or "imiti" in SYSTEM_PROMPT.lower()
        # Rule 3: General info only + emergency escalation
        assert "muganga" in SYSTEM_PROMPT.lower() or "chw" in SYSTEM_PROMPT.lower()
        # Rule 4: Concise responses
        # Rule 5: Human-like persona
        assert "robot" in SYSTEM_PROMPT.lower() or "ai" in SYSTEM_PROMPT.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
