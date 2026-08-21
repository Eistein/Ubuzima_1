"""tests/test_text_checks.py — output language/relevance guards (safety-prompt Rule 1)."""
from text_checks import (
    contains_digits, english_tokens, looks_kinyarwanda_only, is_relevant,
)


class TestDigits:
    def test_flags_digits(self):
        assert contains_digits("fata ibinini 2 ku munsi") is True

    def test_no_digits(self):
        assert contains_digits("fata ibinini bibiri ku munsi") is False


class TestEnglishTokens:
    def test_detects_english(self):
        assert "doctor" in english_tokens("Jya kwa doctor vuba")

    def test_pure_kinyarwanda_has_none(self):
        assert english_tokens("Jya kwa muganga vuba") == []


class TestKinyarwandaOnly:
    def test_clean_kinyarwanda_passes(self):
        ok, reasons = looks_kinyarwanda_only("Nywa amazi menshi kandi uruhuke.")
        assert ok is True and reasons == []

    def test_digits_fail_rule_1(self):
        ok, reasons = looks_kinyarwanda_only("Fata ibinini 3.")
        assert ok is False and any("digit" in r for r in reasons)

    def test_english_fails_rule_1(self):
        ok, reasons = looks_kinyarwanda_only("Please see a doctor.")
        assert ok is False and any("English" in r for r in reasons)

    def test_empty_fails(self):
        ok, reasons = looks_kinyarwanda_only("   ")
        assert ok is False and reasons == ["empty response"]


class TestRelevance:
    def test_reasonable_answer_ok(self):
        ok, _ = is_relevant("Malariya ni iki?",
                            "Malariya iterwa n'umubu; wirinde ukoresheje inzitiramibu.")
        assert ok is True

    def test_too_short_flagged(self):
        ok, reasons = is_relevant("Malariya ni iki?", "Yego.")
        assert ok is False and any("shorter" in r for r in reasons)

    def test_echo_flagged(self):
        q = "Malariya ni iki koko rero kandi"
        ok, reasons = is_relevant(q, q)
        assert ok is False and any("echoes" in r for r in reasons)
