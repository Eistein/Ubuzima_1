"""
text_checks.py — lightweight output checks for UBUZIMA AI.

These are HEURISTIC guards, not a language-identification model. They exist so the
pipeline (and the test suite) can make cheap, deterministic assertions about the LLM's
output that map directly to safety-prompt Rule (1) — "answer in Kinyarwanda only, with no
English words and no digits" — and to a minimal relevance check.

Kept in their own module (rather than inline in app.py) so they can be unit-tested without
loading Gradio, torch, or any model. Separation of concerns: audio, model access, and
text validation are independent and independently testable.

HONEST SCOPE
------------
`looks_kinyarwanda_only` cannot *prove* text is Kinyarwanda. It flags the two failure modes
Rule (1) actually cares about — stray English tokens and Arabic numerals — using a small
stop-list and a digit check. It will not catch a fluent English sentence made of words
outside the stop-list. For a stronger guarantee, swap in a fastText language-ID model; the
function signature is designed to be a drop-in for that upgrade.
"""

import re

# A deliberately small stop-list of high-frequency English tokens that should never appear
# in a Kinyarwanda health answer. Not exhaustive by design — see HONEST SCOPE above.
_ENGLISH_STOPWORDS = frozenset({
    "the", "and", "you", "your", "please", "doctor", "medicine", "dose", "dosage",
    "take", "should", "health", "hospital", "fever", "malaria", "water", "here",
    "is", "are", "for", "with", "this", "that", "not", "sorry", "cannot", "can",
})

_WORD_RE = re.compile(r"[A-Za-z']+")
_DIGIT_RE = re.compile(r"\d")


def contains_digits(text: str) -> bool:
    """True if the text contains any Arabic numeral (Rule 1 forbids digits)."""
    return bool(_DIGIT_RE.search(text or ""))


def english_tokens(text: str) -> list[str]:
    """Return the English stop-words found in the text (lower-cased, order preserved)."""
    if not text:
        return []
    return [w for w in (m.group(0).lower() for m in _WORD_RE.finditer(text))
            if w in _ENGLISH_STOPWORDS]


def looks_kinyarwanda_only(text: str) -> tuple[bool, list[str]]:
    """Heuristic check for safety-prompt Rule (1).

    Returns (ok, reasons). `ok` is True when no forbidden signal is found. `reasons` lists
    the specific violations, so a caller can log *why* a response was flagged.
    """
    reasons: list[str] = []
    if not text or not text.strip():
        return False, ["empty response"]
    if contains_digits(text):
        reasons.append("contains digits")
    hits = english_tokens(text)
    if hits:
        reasons.append(f"contains English tokens: {sorted(set(hits))}")
    return (len(reasons) == 0), reasons


def is_relevant(question: str, answer: str, min_chars: int = 15) -> tuple[bool, list[str]]:
    """Very small relevance gate: a usable answer is non-trivially long and not an echo
    of the question. This is a floor, not a semantic judgement — it catches empty,
    truncated, or parroted responses, which are the failures worth catching cheaply.
    """
    reasons: list[str] = []
    a = (answer or "").strip()
    if len(a) < min_chars:
        reasons.append(f"answer shorter than {min_chars} chars")
    if a and question and a.lower() == question.strip().lower():
        reasons.append("answer merely echoes the question")
    return (len(reasons) == 0), reasons
