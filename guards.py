"""guards.py — everything that must be true before a question reaches the LLM.

Three tiers, cheapest first, because the expensive one costs you NFR1 headroom:

  Tier 1  audio sanity        ~0 ms    silence, length, level, clipping
  Tier 2  transcript sanity   ~0 ms    empty, ASR confidence, Kinyarwanda word overlap
  Tier 3  spoken language ID  ~300 ms  facebook/mms-lid-126, optional, off by default

Tier 2 is the interesting one. Your Stage 2 eval reported mean ASR confidence of
0.959 on Afrivoice and 0.913 on Common Voice, with a confidence–WER Spearman of
-0.453 / -0.615 — i.e. the confidence signal is genuinely calibrated on your model.
A CTC decoder constrained to a Kinyarwanda vocabulary, fed English, emits
Kinyarwanda-shaped nonsense at low confidence. That is the cheap detector.

Turn Tier 3 on with ENABLE_LID=1 once you have measured what it costs you.
"""
from __future__ import annotations

import logging
import os
import re

import numpy as np
import soundfile as sf

import errors

log = logging.getLogger(__name__)

MIN_DURATION_S = 1.0
MAX_DURATION_S = 30.0
SILENCE_RMS = 1e-4
QUIET_RMS = 5e-3
CLIP_FRACTION = 0.01

# Tuned against your own eval: real Kinyarwanda sits at 0.91-0.96 mean confidence.
MIN_ASR_CONFIDENCE = float(os.environ.get("MIN_ASR_CONFIDENCE", "0.55"))
MIN_KIN_WORD_OVERLAP = float(os.environ.get("MIN_KIN_WORD_OVERLAP", "0.30"))
ENABLE_LID = os.environ.get("ENABLE_LID", "0") == "1"
LID_MODEL = os.environ.get("LID_MODEL", "facebook/mms-lid-126")
MIN_LID_CONFIDENCE = float(os.environ.get("MIN_LID_CONFIDENCE", "0.50"))
MIN_WORDS_FOR_LANGUAGE_CHECK = 5

# High-frequency Kinyarwanda function words and morphology. Cheap, no model needed.
# Extend this from your Afrivoice manifest rather than by hand.
_KIN_MARKERS = {
    "ni", "na", "no", "mu", "ku", "ya", "za", "cya", "bya", "rya", "wa", "ba",
    "ndi", "uri", "ari", "turi", "muri", "bari", "nta", "ese", "cyangwa",
    "kandi", "ariko", "kuko", "niba", "iyo", "aho", "uko", "ute", "iki", "iyi",
    "ibi", "izi", "uyu", "uwo", "abo", "bose", "byose", "cyane", "gato",
    "gute", "kuki", "ryari", "he", "nde", "yego", "oya", "murakoze",
    "muraho", "amakuru", "ubuzima", "umuti", "imiti", "muganga", "indwara",
    "kwivuza", "kwirinda", "umubyeyi", "umwana", "abana", "gufata", "kunywa",
    "kurya", "gukora", "kugira", "kuba", "kugera", "nshobora", "ushobora",
    "ashobora", "mfite", "ufite", "afite", "hari", "nkeneye", "mbwira",
    "miti", "nfate", "iyihe", "ndwaye", "urwaye", "arwaye", "umuriro",
    "umutwe", "inda", "amaraso", "ibimenyetso", "isuzuma", "kwipimisha",
    "ibitaro", "kigo", "nderabuzima", "malariya", "igituntu", "sida",
    "inzitiramubu", "umubu", "utwite", "gutwita", "konsa", "urukingo",
    "inkingo", "isuku", "amazi", "indyo", "intungamubiri", "nakwirinda",
    "nakora", "vuba", "cyane", "buri", "joro", "munsi", "ngombwa",
}
_WORD_RE = re.compile(r"[a-z']+")

_lid = None
_lid_ok: bool | None = None


# ─── Tier 1 ──────────────────────────────────────────────────────────────────
def check_audio(audio_path: str | None) -> errors.Message | None:
    """Return an error Message, or None if the audio is usable."""
    if not audio_path:
        return errors.NO_AUDIO
    try:
        data, sr = sf.read(audio_path, dtype="float32")
    except Exception as e:
        log.warning("Unreadable audio %s: %s", audio_path, e)
        return errors.NO_AUDIO

    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size == 0:
        return errors.AUDIO_SILENT

    duration = data.size / sr
    if duration < MIN_DURATION_S:
        return errors.AUDIO_TOO_SHORT
    if duration > MAX_DURATION_S:
        return errors.AUDIO_TOO_LONG

    rms = float(np.sqrt(np.mean(data ** 2)))
    if rms < SILENCE_RMS:
        return errors.AUDIO_SILENT
    if rms < QUIET_RMS:
        return errors.AUDIO_TOO_QUIET

    if float(np.mean(np.abs(data) >= 0.999)) > CLIP_FRACTION:
        return errors.AUDIO_CLIPPED
    return None


# ─── Tier 2 ──────────────────────────────────────────────────────────────────
def kinyarwanda_overlap(text: str) -> float:
    """Fraction of tokens that are recognisable Kinyarwanda function words."""
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _KIN_MARKERS)
    # Bantu agglutination: also count words carrying common Kinyarwanda prefixes.
    prefixed = sum(
        1 for w in words
        if len(w) > 4 and w[:2] in {"ku", "gu", "uk", "ab", "im", "in", "um", "ib", "ic", "ur"}
    )
    return min(1.0, (hits + 0.5 * prefixed) / len(words))


def check_transcript(text: str, confidence: float) -> errors.Message | None:
    if not (text or "").strip():
        return errors.ASR_EMPTY
    if confidence < MIN_ASR_CONFIDENCE:
        # Low confidence on a Kinyarwanda-only CTC head means either bad audio or
        # a language this model was never trained on. Distinguish with overlap.
        if kinyarwanda_overlap(text) < MIN_KIN_WORD_OVERLAP:
            return errors.NOT_KINYARWANDA
        return errors.ASR_LOW_CONFIDENCE
    # Only judge language from word overlap when there are enough words to judge
    # from. "Nfate iyihe miti?" is three words, perfectly good Kinyarwanda, and
    # carries almost no function-word signal — rejecting it would be a false
    # positive on exactly the kind of question the safety layer must catch.
    words = _WORD_RE.findall(text.lower())
    if len(words) >= MIN_WORDS_FOR_LANGUAGE_CHECK and \
            kinyarwanda_overlap(text) < MIN_KIN_WORD_OVERLAP:
        return errors.NOT_KINYARWANDA
    return None


# ─── Tier 3 ──────────────────────────────────────────────────────────────────
def _load_lid():
    """Load MMS-LID once. Disables itself if this checkpoint has no Kinyarwanda."""
    global _lid, _lid_ok
    if _lid_ok is not None:
        return _lid
    try:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        extractor = AutoFeatureExtractor.from_pretrained(LID_MODEL)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL)
        labels = set(model.config.id2label.values())
        if "kin" not in labels:
            log.error("%s does not cover Kinyarwanda (kin) — LID disabled. Try "
                      "facebook/mms-lid-256 or larger.", LID_MODEL)
            _lid_ok = False
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        _lid = (extractor, model, device)
        _lid_ok = True
        log.info("LID ready: %s (%d languages)", LID_MODEL, len(labels))
    except Exception as e:
        log.error("LID unavailable: %s", e)
        _lid_ok = False
    return _lid


def check_spoken_language(audio_path: str) -> tuple[errors.Message | None, dict]:
    """Definitive spoken-language check. Returns (error_or_None, detail)."""
    if not ENABLE_LID:
        return None, {"lid": "disabled"}
    lid = _load_lid()
    if lid is None:
        return None, {"lid": "unavailable"}

    import torch
    import torchaudio
    extractor, model, device = lid

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    inputs = extractor(wav.squeeze().numpy(), sampling_rate=16000,
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    top = int(probs.argmax())
    lang, conf = model.config.id2label[top], float(probs[top])
    detail = {"lid": lang, "lid_confidence": round(conf, 3)}

    if lang != "kin" and conf >= MIN_LID_CONFIDENCE:
        log.info("Rejected: detected %s at %.2f", lang, conf)
        return errors.NOT_KINYARWANDA, detail
    return None, detail
