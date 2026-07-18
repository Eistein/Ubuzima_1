"""safety_prompt.py — the Kinyarwanda health-safety layer.

Corresponds to Section 4.1.4 of the report and the SafetyPolicy class in Figure 3.6.

Design rule from Section 3.3.3, kept exactly: this layer NEVER alters the
transcription. It only prepends the system prompt, and — for queries that ask for
a diagnosis — short-circuits the LLM entirely with a fixed referral message.

NOTE FOR GANZA: you are the native speaker here, not me. Read every Kinyarwanda
string below and rewrite anything that does not sound like a real health worker.
Section 5.4 records that reviewer R2 valued the disclaimer because it sounded like
a responsible person rather than legal boilerplate — that property lives in this
file and nowhere else.
"""
from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# System prompt (Kinyarwanda). Prepended to every transcription before the LLM.
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Uri umufasha w'amakuru y'ubuzima witwa UBUZIMA AI. Uvugana n'abantu
bavuga Ikinyarwanda, cyane cyane abajyanama b'ubuzima (CHW) n'abaturage.

AMATEGEKO AGOMBA GUKURIKIZWA:
1. Subiza mu Kinyarwanda gusa, mu magambo yoroshye kandi asobanutse.
2. NTUTANGE isuzuma ry'uburwayi (diagnosis). Ntuvuge ngo "ufite iyi ndwara".
3. NTUTANGE imiti cyangwa ingano y'imiti (dosage).
4. Tanga amakuru rusange y'ubuzima no kwirinda indwara gusa.
5. Buri gihe usabe umuntu kujya ku kigo nderabuzima cyangwa kwa muganga
   iyo ari ikibazo cy'ubuzima kimwerekeyeho.
6. Niba utazi igisubizo, vuga ko utakizi. Ntuhimbe amakuru.
7. Gumana ijwi ry'icyubahiro n'impuhwe, nk'umujyanama w'ubuzima nyawe.

Igisubizo cyawe kigomba kuba kigufi (interuro 3 kugeza kuri 8)."""

# Appended to every response above MIN_LENGTH_FOR_DISCLAIMER characters (NFR3).
DISCLAIMER = (
    "Icyitonderwa: aya ni amakuru rusange y'ubuzima, si isuzuma rya muganga. "
    "Nyamuneka sura umuganga cyangwa ujye ku kigo nderabuzima kikwegereye "
    "kugira ngo ubone ubufasha bwihariye."
)
MIN_LENGTH_FOR_DISCLAIMER = 120

# Fixed refusal-and-referral response — used instead of calling the LLM at all.
REFUSAL_RESPONSES = {
    "diagnosis": (
        "Simbasha kukubwira indwara ufite, kandi sinshobora gutanga isuzuma. "
        "Ibyo ni akazi ka muganga wabigize umwuga. "
        "Nyamuneka jya ku kigo nderabuzima kikwegereye vuba bishoboka, "
        "cyane cyane niba ibimenyetso bikomeye cyangwa bimaze iminsi."
    ),
    "medication": (
        "Sinshobora kugutegeka imiti cyangwa kukubwira ingano yayo. "
        "Imiti igomba gutangwa na muganga cyangwa umuforomo nyuma yo kukwipimisha. "
        "Nyamuneka jya ku kigo nderabuzima cyangwa kuri farumasi wemewe."
    ),
    "emergency": (
        "Ibyo uvuze bishobora kuba ari ibyihutirwa. "
        "Nyamuneka jya ku bitaro cyangwa ku kigo nderabuzima ako kanya, "
        "cyangwa uhamagare 912. Ntutegereze."
    ),
}

# Trigger patterns. Kept deliberately conservative: a false refusal is a mild
# annoyance, a missed diagnostic answer is the failure mode Section 5.6 cares about.
_PATTERNS: dict[str, list[str]] = {
    "emergency": [
        r"\bamaraso menshi\b", r"\bntabwo ahumeka\b", r"\bntahumeka\b",
        r"\byataye ubwenge\b", r"\bkwiyahura\b", r"\bkwipfusha\b",
        r"\bumuriro mwinshi cyane\b", r"\bibyihutirwa\b",
    ],
    "diagnosis": [
        r"\bmfite iyihe ndwara\b", r"\bndwaye iki\b", r"\bnfite iki\b",
        r"\bnsuzum\w*", r"\bmbwira indwara\b", r"\bni iyihe ndwara\b",
        r"\bese mfite\b", r"\bnzi ko mfite\b",
    ],
    "medication": [
        r"\bnfate iyihe miti\b", r"\bimiti ki\b", r"\bnywe iyihe miti\b",
        r"\bingano y'umuti\b", r"\bdosage\b", r"\binshuro zingahe\b.*\bumuti\b",
    ],
}
_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in _PATTERNS.items()}


class SafetyPolicy:
    """Holds the system prompt template and the refusal-and-referral responses."""

    prompt_template = SYSTEM_PROMPT
    refusal_responses = REFUSAL_RESPONSES

    def classify(self, text: str) -> str | None:
        """Return 'emergency' | 'diagnosis' | 'medication' | None.

        Emergency is checked first: it outranks everything else.
        """
        if not text:
            return None
        for label in ("emergency", "diagnosis", "medication"):
            if any(p.search(text) for p in _COMPILED[label]):
                return label
        return None

    def apply(self, text: str) -> str:
        """Prepend the system prompt. The transcription itself is never modified."""
        return f"{self.prompt_template}\n\nIkibazo cy'umukoresha: {text}"

    def add_disclaimer(self, response: str) -> str:
        if len(response) >= MIN_LENGTH_FOR_DISCLAIMER and DISCLAIMER not in response:
            return f"{response}\n\n{DISCLAIMER}"
        return response
