"""kinya_text.py — text normalisation before synthesis.

A synthesiser reads what you hand it. Give it "912" or "38.5 °C" and it spells
characters or invents something. This turns machine text into words a Kinyarwanda
speaker would actually say.

╔═══════════════════════════════════════════════════════════════════════════════╗
║ GANZA — VERIFY EVERY STRING BELOW.                                            ║
║                                                                               ║
║ I am confident about DIGITS. I am NOT confident about CARDINALS: Kinyarwanda  ║
║ numerals agree with the noun class of what they count, so "kane" / "ine" and  ║
║ friends shift with context, and a flat lookup table cannot capture that. What ║
║ is here is a scaffold that will be right often and wrong sometimes. Lines     ║
║ marked VERIFY are the ones to read out loud and fix. This is five minutes of  ║
║ your time and it is not something I can do for you.                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import re

# Confident.
DIGITS = {
    "0": "zeru", "1": "rimwe", "2": "kabiri", "3": "gatatu", "4": "kane",
    "5": "gatanu", "6": "gatandatu", "7": "karindwi", "8": "umunani", "9": "icyenda",
}

# VERIFY — class agreement is not modelled here.
CARDINALS = {
    0: "zeru", 1: "rimwe", 2: "kabiri", 3: "gatatu", 4: "kane", 5: "gatanu",
    6: "gatandatu", 7: "karindwi", 8: "umunani", 9: "icyenda", 10: "icumi",
}
TENS = {  # VERIFY
    20: "makumyabiri", 30: "mirongo itatu", 40: "mirongo ine", 50: "mirongo itanu",
    60: "mirongo itandatu", 70: "mirongo irindwi", 80: "mirongo inani",
    90: "mirongo icyenda",
}
HUNDRED = "ijana"          # VERIFY
THOUSAND = "igihumbi"      # VERIFY — singular
THOUSANDS = "ibihumbi"     # VERIFY — plural, takes class-8 agreement below
AND = "na"                 # VERIFY — elides to n' before vowels in real speech
DECIMAL_SEP = "koma"       # VERIFY — Rwanda writes decimals with a comma
CURRENCY = "amafaranga y'u Rwanda"   # VERIFY

# THE REASON A FLAT TABLE CANNOT WORK, MADE CONCRETE:
# Kinyarwanda numerals agree with the noun class of what they count. The digit 5 is
# "gatanu" standing alone but "bitanu" counting ibihumbi (class 8) — as in
# "ibihumbi makumyabiri na bitanu" for 25,000. Same number, different word, because
# of what is being counted. This dict handles class 8 only, which covers ibihumbi
# and amafaranga. Every other class is unhandled. VERIFY ALL OF IT.
CLASS8 = {  # VERIFY
    1: "kimwe", 2: "bibiri", 3: "bitatu", 4: "bine", 5: "bitanu",
    6: "bitandatu", 7: "birindwi", 8: "umunani", 9: "icyenda", 10: "icumi",
}

# Expanded BEFORE digits are touched, so "38.5 °C" and "40%" come out right.
UNITS = [
    (r"°\s*C", " dogere selisiyusi"),
    (r"\bRWF\b|\bFRW\b", f" {CURRENCY} "),
    (r"%", f" ku {HUNDRED}"),
    (r"\bkm\b", " kilometero"),
    (r"\bkg\b", " kilogarama"),
    (r"\bml\b", " mililitiro"),
    (r"\bmg\b", " miligarama"),
]
ABBREVIATIONS = {
    "CHW": "umujyanama w'ubuzima",
    "RBC": "ikigo cy'igihugu cy'ubuzima",
    "HIV": "virusi itera sida",
    "VIH": "virusi itera sida",
    "TB": "igituntu",
    "ARV": "imiti igabanya ubukana",
}

# Numbers a Rwandan reads digit by digit, not as a cardinal.
PHONE_LIKE = {"912", "114", "116", "3029"}


def _class8(n: int) -> str:
    """Numeral agreeing with class 8 (ibihumbi, amafaranga). VERIFY."""
    if n in CLASS8:
        return CLASS8[n]
    if n < 100:
        tens, r = (n // 10) * 10, n % 10
        return TENS[tens] if r == 0 else f"{TENS[tens]} {AND} {CLASS8.get(r, CARDINALS[r])}"
    return _cardinal(n)


def _cardinal(n: int) -> str:
    """Spell 0-999,999. VERIFY — see the header."""
    if n >= 1000:
        th, r = n // 1000, n % 1000
        head = THOUSAND if th == 1 else f"{THOUSANDS} {_class8(th)}"
        return head if r == 0 else f"{head} {AND} {_cardinal(r)}"
    if n in CARDINALS:
        return CARDINALS[n]
    if n < 20:
        return f"{CARDINALS[10]} {AND} {CARDINALS[n - 10]}"
    if n < 100:
        t, r = (n // 10) * 10, n % 10
        return TENS[t] if r == 0 else f"{TENS[t]} {AND} {CARDINALS[r]}"
    if n < 1000:
        h, r = n // 100, n % 100
        head = HUNDRED if h == 1 else f"{HUNDRED} {CARDINALS[h]}"
        return head if r == 0 else f"{head} {AND} {_cardinal(r)}"
    return " ".join(DIGITS.get(d, d) for d in str(n))


def _number(m: re.Match) -> str:
    tok = m.group(0).replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+", m.group(0)) else m.group(0)
    if tok in PHONE_LIKE or (len(tok) >= 5 and "." not in tok and tok.isdigit()
                             and not 1000 <= int(tok) <= 999999):
        return " ".join(DIGITS.get(d, d) for d in tok if d.isdigit())
    if "." in tok or "," in tok:
        whole, frac = re.split(r"[.,]", tok, maxsplit=1)
        frac_spoken = " ".join(DIGITS.get(d, d) for d in frac)
        return f"{_cardinal(int(whole))} {DECIMAL_SEP} {frac_spoken}"
    return _cardinal(int(tok))


def normalise(text: str) -> str:
    if not text:
        return text
    for abbr, spoken in ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", spoken, text)
    for pat, spoken in UNITS:                      # units BEFORE digits
        text = re.sub(pat, spoken, text)
    text = re.sub(r"\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?", _number, text)
    text = re.sub(r"[*_#`>|]", " ", text)          # markdown the LLM leaks
    text = text.replace("&", f" {AND} ")
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


SENT_END = re.compile(r"(?<=[.!?])\s+")
CLAUSE = re.compile(r"(?<=[,;:])\s+")


def segment(text: str, max_chars: int = 180) -> list[tuple[str, float]]:
    """Split into (chunk, pause_after_seconds).

    VITS degrades on long inputs — the duration predictor drifts, which is the
    breath-and-pause artefact Section 5.4 recorded. Synthesising sentence by
    sentence and joining with real silence is both more stable and more natural,
    because a person pauses between sentences and the model will not do it for you.
    """
    out: list[tuple[str, float]] = []
    for sentence in SENT_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            out.append((sentence, 0.35))
            continue
        parts, buf = [], ""
        for clause in CLAUSE.split(sentence):
            if len(buf) + len(clause) + 1 <= max_chars:
                buf = f"{buf} {clause}".strip()
            else:
                if buf:
                    parts.append(buf)
                buf = clause
        if buf:
            parts.append(buf)
        for i, p in enumerate(parts):
            out.append((p, 0.35 if i == len(parts) - 1 else 0.15))
    return out
