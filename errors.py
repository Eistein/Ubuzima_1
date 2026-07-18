"""errors.py — every way UBUZIMA AI can refuse or fail, with what the user hears.

Design rules, in order of importance:

1. The user is a Kinyarwanda speaker who may not read English, and may not read at
   all. Every message therefore has a Kinyarwanda form that is safe to SPEAK
   aloud through the TTS, and an English form for the logs and for your examiner.
2. An error must say what to DO next, not what went wrong internally. "ASR
   confidence 0.31" helps you; "vuga urumva" helps her.
3. Nothing here ever guesses at health content. A failure never degrades into a
   half-answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    RETRY = "retry"        # user can fix it by trying again
    REFUSE = "refuse"      # deliberate policy refusal, working as intended
    FAULT = "fault"        # our fault — system problem


@dataclass(frozen=True)
class Message:
    code: str
    rw: str                # spoken to the user, in Kinyarwanda
    en: str                # shown under it + written to logs
    severity: Severity
    speak: bool = True     # synthesise the Kinyarwanda aloud?

    def as_markdown(self) -> str:
        icon = {Severity.RETRY: "🔁", Severity.REFUSE: "🛡️", Severity.FAULT: "⚠️"}[self.severity]
        return f"{icon} **{self.rw}**\n\n_{self.en}_  \n`{self.code}`"


# ─── input audio ─────────────────────────────────────────────────────────────
NO_AUDIO = Message(
    "NO_AUDIO",
    "Nta jwi nakiriye. Kanda kuri mikoro hanyuma uvuge ikibazo cyawe.",
    "No audio received. Tap the microphone and speak your question.",
    Severity.RETRY, speak=False,
)
AUDIO_TOO_SHORT = Message(
    "AUDIO_TOO_SHORT",
    "Ijwi ryawe ni rigufi cyane. Nyamuneka vuga ikibazo cyawe cyose, "
    "byibura amasegonda abiri.",
    "Recording shorter than 1.0 s. Ask the whole question — at least two seconds.",
    Severity.RETRY,
)
AUDIO_TOO_LONG = Message(
    "AUDIO_TOO_LONG",
    "Ijwi ryawe ni rirerire cyane. Nyamuneka baza ikibazo kimwe gusa, "
    "mu masegonda mirongo itatu.",
    "Recording longer than 30 s. Ask one question, under thirty seconds.",
    Severity.RETRY,
)
AUDIO_SILENT = Message(
    "AUDIO_SILENT",
    "Sinumvise ijwi na rimwe. Reba niba mikoro ikora, hanyuma wongere ugerageze.",
    "The recording is silent. Check the microphone permission and try again.",
    Severity.RETRY,
)
AUDIO_TOO_QUIET = Message(
    "AUDIO_TOO_QUIET",
    "Ijwi ryawe riri hasi cyane. Egera mikoro hanyuma uvuge urumva.",
    "Signal too quiet to transcribe reliably. Move closer and speak up.",
    Severity.RETRY,
)
AUDIO_CLIPPED = Message(
    "AUDIO_CLIPPED",
    "Ijwi ryawe rirenze urugero, ntirisobanutse. Tera intambwe usubire inyuma "
    "gato ku mikoro.",
    "Audio is clipping. Move slightly away from the microphone.",
    Severity.RETRY,
)

# ─── recognition ─────────────────────────────────────────────────────────────
ASR_EMPTY = Message(
    "ASR_EMPTY",
    "Sinashoboye kumva ibyo wavuze. Nyamuneka ongera uvuge buhoro kandi urumva.",
    "The recogniser returned an empty transcript. Speak slowly and clearly.",
    Severity.RETRY,
)
ASR_LOW_CONFIDENCE = Message(
    "ASR_LOW_CONFIDENCE",
    "Sinumvise neza ibyo wavuze. Bishobora kuba biterwa n'urusaku cyangwa "
    "ijwi ridasobanutse. Nyamuneka ongera ugerageze ahantu hatuje.",
    "ASR confidence below threshold — likely noise, distance, or non-Kinyarwanda "
    "speech. Retry somewhere quieter.",
    Severity.RETRY,
)
NOT_KINYARWANDA = Message(
    "NOT_KINYARWANDA",
    "UBUZIMA AI yumva Ikinyarwanda gusa. Nyamuneka baza ikibazo cyawe mu Kinyarwanda.",
    "This assistant understands Kinyarwanda only. Please ask your question in "
    "Kinyarwanda.",
    Severity.REFUSE,
)

# ─── reasoning ───────────────────────────────────────────────────────────────
LLM_UNAVAILABLE = Message(
    "LLM_UNAVAILABLE",
    "Ubu sinshobora gusubiza. Nyamuneka ongera ugerageze nyuma y'akanya.",
    "The reasoning service is unreachable. Try again shortly.",
    Severity.FAULT,
)
LLM_RATE_LIMIT = Message(
    "LLM_RATE_LIMIT",
    "Abantu benshi barimo kubaza icyarimwe. Nyamuneka tegereza akanya "
    "hanyuma wongere ugerageze.",
    "Rate limit reached on the LLM API. Wait a moment and retry.",
    Severity.FAULT,
)
LLM_BLOCKED = Message(
    "LLM_BLOCKED",
    "Sinshobora gusubiza kuri icyo kibazo. Nyamuneka baza ikindi kibazo "
    "cy'amakuru y'ubuzima.",
    "The model's own safety filter blocked this prompt or response.",
    Severity.REFUSE,
)
LLM_WRONG_LANGUAGE = Message(
    "LLM_WRONG_LANGUAGE",
    "Habaye ikibazo mu gusubiza mu Kinyarwanda. Nyamuneka ongera ugerageze.",
    "The model answered in a language other than Kinyarwanda and the retry also "
    "failed. Response withheld rather than served in the wrong language.",
    Severity.FAULT,
)

# ─── synthesis ───────────────────────────────────────────────────────────────
TTS_UNAVAILABLE = Message(
    "TTS_UNAVAILABLE",
    "", "Speech synthesis is unavailable; the answer is shown as text only.",
    Severity.FAULT, speak=False,
)

# ─── catch-all ───────────────────────────────────────────────────────────────
INTERNAL = Message(
    "INTERNAL",
    "Habaye ikibazo muri sisitemu. Nyamuneka ongera ugerageze.",
    "Unhandled internal error. See the Space logs.",
    Severity.FAULT,
)

ALL: dict[str, Message] = {
    m.code: m for m in (
        NO_AUDIO, AUDIO_TOO_SHORT, AUDIO_TOO_LONG, AUDIO_SILENT, AUDIO_TOO_QUIET,
        AUDIO_CLIPPED, ASR_EMPTY, ASR_LOW_CONFIDENCE, NOT_KINYARWANDA,
        LLM_UNAVAILABLE, LLM_RATE_LIMIT, LLM_BLOCKED, LLM_WRONG_LANGUAGE,
        TTS_UNAVAILABLE, INTERNAL,
    )
}
