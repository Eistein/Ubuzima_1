"""
UBUZIMA AI — Umufasha w'ubuzima uvuga mu Kinyarwanda
Production build for Railway (adapted from the Colab demo notebook).

Pipeline: badrex/w2v-bert-2.0-kinyarwanda-asr + your LoRA adapter (bundled
in ./adapter) -> Gemini 2.5 Flash via OpenRouter -> Meta MMS-TTS (Kinyarwanda)

Required environment variables (set these in Railway's dashboard, never
hardcode them):
    OPENROUTER_API_KEY   - from openrouter.ai/keys
    HF_TOKEN             - optional, only needed if badrex's model is gated
    CONTACT_EMAIL        - optional, shown in the Terms & Privacy tab

Local run:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY=sk-...
    python app.py
"""

import os
import re
import traceback
from pathlib import Path

import numpy as np
import requests
import librosa
import torch
import gradio as gr

# ==========================================================================
# GOVERNANCE SWITCHES
# ==========================================================================

# Affirmative consent before the microphone input is processed.
# Set to False if you would rather demo without the consent gate.
REQUIRE_CONSENT = True

# Shown in the Terms & Privacy tab. Set CONTACT_EMAIL in Railway.
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "d.ganza@alustudent.com")

# --- ASR confidence gating -------------------------------------------------
# Implemented in asr_confidence.py (report §5.2.5). Kept in its own module so the
# scoring logic can be unit-tested and calibrated without loading Gradio, and so
# app.py stays an orchestration layer rather than an algorithm dump.
from asr_confidence import (
    HIGH_CONF,
    LOW_CONF,
    confidence_band,
    ctc_confidence,
    resolve_blank_id,
)

# ==========================================================================
# UI TEXT — Kinyarwanda strings in ONE place. Edit freely; you're the native
# speaker. Nothing user-facing lives outside this block.
# ==========================================================================

TXT = {
    "tagline": "Umufasha w'ubuzima uvuga mu Kinyarwanda",
    "subtitle": "Baza ikibazo cy'ubuzima mu majwi, ubone igisubizo mu majwi mu Kinyarwanda",
    "live": "Irakora",
    "pill_asr": "ASR: badrex + LoRA",
    "pill_llm": "LLM: Gemini 2.5 Flash",
    "pill_wer": "WER: 6.76% ubuzima / 31.29% rusange",

    "tab_main": "\U0001FA7A Ubuzima AI",
    "tab_terms": "\U0001F512 Amasezerano n'Ibanga",

    "voice_card": "Baza amajwi hano",
    "voice_hint": "Kanda mikoro, uvuge ikibazo cyawe, hanyuma ukande \u201cTanga igisubizo\u201d.",
    "btn_run": "\u25B6 Tanga igisubizo",
    "btn_reset": "\u21BA Siba",

    "consent_label": "Ndemeye ko ijwi ryanjye rikoreshwa mu gutanga igisubizo (reba \u201cAmasezerano n'Ibanga\u201d).",
    "consent_note": "Uruhushya rurasabwa mbere yo gutangira. Soma amasezerano mu gace ka \u201cAmasezerano n'Ibanga\u201d haruguru.",
    "e_consent": "\u26A0\uFE0F Banza wemere uruhushya rwo gukoresha ijwi ryawe. Kanda agasanduku k'uruhushya hejuru y'iyi buto.",

    # confidence gate
    "conf_label": "Uko twumvise ijwi",
    "conf_high": "Twumvise neza",
    "conf_medium": "Ntitwumvise neza cyane \u2014 reba ko amagambo hepfo ari yo wavuze",
    "conf_low": "Ntitwumvise neza",
    "e_low_conf": ("\u26A0\uFE0F Ntitwumvise neza ijwi ryawe, bityo ntitwatanga igisubizo "
                   "kuko twashobora gusubiza ikibazo kitari cyo wabajije. "
                   "Ongera uvuge buhoro, mu kanwa gakuru, ahantu hatuje."),
    "hedge_medium": ("\u26A0\uFE0F Icyitonderwa: ntitwumvise neza ijwi ryawe. "
                     "Banza urebe ko amagambo twanditse ari yo wavuze mbere yo gukurikiza iki gisubizo."),

    "examples_head": "Ingero z'ibibazo washobora kubaza",

    "card1": "Icyo wavuze",
    "card1_ph": "Amagambo yawe azagaragara hano nyuma yo kumva ijwi\u2026",
    "card2": "Igisubizo cya UBUZIMA AI",
    "card2_ph": "Igisubizo mu Kinyarwanda kizagaragara hano\u2026",
    "card3": "Igisubizo mu majwi",

    # progress + errors
    "p_asr": "\U0001F3A4 Turimo kwandika icyo wavuze\u2026",
    "p_llm": "\U0001F4AD UBUZIMA AI iratekereza\u2026",
    "p_tts": "\U0001F50A Turimo gukora igisubizo mu majwi\u2026",
    "p_done": "\u2713 Byarangiye",
    "e_not_ready": "\u26A0\uFE0F Ijwi ntiryiteguye. Nyuma yo gukanda \u201cHagarika\u201d kuri mikoro, tegereza amasegonda 1-2, hanyuma ukande \u201cTanga igisubizo\u201d nanone.",
    "e_format": "\u26A0\uFE0F Ikibazo ku miterere y'ijwi: {e}",
    "e_empty": "\u26A0\uFE0F Nta jwi ryinjiye. Ongera ugerageze kuvuga.",
    "e_asr": "\u26A0\uFE0F Ikosa mu kwandika ijwi: {t}: {e}",
    "e_unclear": "(Ntitwabashije kumva neza ijwi. Gerageza kuvuga urerure gato kandi buhoro.)",
    "e_llm": "\u26A0\uFE0F Ikosa mu gutanga igisubizo: {t}: {e}",
    "e_tts": "[\u26A0\uFE0F Ikosa mu gukora ijwi: {e}]",

    "disc_strong": "\u26A0\uFE0F Iyi ni porototipe y'ubushakashatsi gusa.",
    "disc_body": "Si igikoresho cyo gusuzuma indwara. Buri gihe jya kwa muganga cyangwa CHW iyo ufite ikibazo cy'ubuzima.",
}

# ==========================================================================
# TERMS OF USE AND PRIVACY NOTICE
# Bilingual: Kinyarwanda for users, English for assessment and review.
# NOTE: review and correct the Kinyarwanda wording — you are the native speaker.
# ==========================================================================

TERMS_INTRO_RW = (
    "Aya masezerano asobanura icyo iyi porogaramu ari cyo, uko ijwi ryawe rikoreshwa, "
    "n'uburenganzira ufite. Nyamuneka soma mbere yo gukoresha serivisi."
)
TERMS_INTRO_EN = (
    "This notice explains what the service is, how your voice input is used, and what "
    "rights you have. Please read it before using the assistant."
)

TERMS_UPDATED = "Iheruka guhindurwa / Last updated: 24 Nyakanga 2026"

CLAUSES = [
    dict(
        n=1,
        title_rw="Icyo iyi porogaramu ari cyo",
        title_en="Purpose and scope",
        rw=("UBUZIMA AI ni umufasha utanga amakuru rusange y'ubuzima mu Kinyarwanda. "
            "Ikoresha ikoranabuhanga ryumva ijwi (ASR), urwego rusubiza (LLM), n'irivuga igisubizo (TTS). "
            "Ni porototipe y'ubushakashatsi yakozwe nk'umushinga wa kaminuza. "
            "Si igikoresho cyo kuvura kandi si serivisi ya muganga."),
        en=("UBUZIMA AI is an informational health assistant that answers general health questions in "
            "Kinyarwanda using speech recognition, a language model, and speech synthesis. It is a "
            "research prototype built as a university capstone project. It is not a medical device and "
            "is not a clinical service."),
        key=False,
    ),
    dict(
        n=2,
        title_rw="Si inama za muganga",
        title_en="Not medical advice",
        rw=("Iyi porogaramu ntisuzuma indwara, kandi ntitanga imiti cyangwa ingano y'imiti. "
            "Ibisubizo ni amakuru rusange gusa; ntibisimbura muganga cyangwa umujyanama w'ubuzima. "
            "Niba ikibazo gikomeye cyangwa kirushaho kwiyongera, jya kwa muganga cyangwa "
            "uhamagare ubutabazi ako kanya."),
        en=("The assistant does not diagnose conditions and does not prescribe or recommend medicines or "
            "dosages. Its responses are general health information only and are not a substitute for a "
            "qualified health professional. If your situation is urgent or worsening, contact a health "
            "facility or emergency services immediately. This matters because a spoken answer in your own "
            "language can feel authoritative; the assistant is designed to point you toward care, not to "
            "replace it."),
        key=True,
    ),
    dict(
        n=3,
        title_rw="Uko ijwi ryawe n'amakuru bikoreshwa",
        title_en="Voice and data use",
        rw=("Ijwi ryawe rikoreshwa gusa mu gutanga igisubizo. Ntiribikwa hamwe n'amakuru akuranga, "
            "kandi ntidukora dosiye y'umuntu ku giti cye. Amakuru yawe ntagurishwa kandi "
            "ntakoreshwa mu kwamamaza. Bimwe mu bikorwa bishobora gukorerwa ku bafatanyabikorwa "
            "bo hanze y'u Rwanda, kandi twohereza gusa ibya ngombwa."),
        en=("Your spoken input is processed only to generate a response. Audio is not retained alongside "
            "information that identifies you, and no profile of you is built. Your data is never sold and "
            "is not used for advertising. Some processing is carried out by service providers located "
            "outside Rwanda, and only the minimum necessary is transmitted. Temporary audio files created "
            "by the interface are cleared automatically."),
        key=True,
    ),
    dict(
        n=4,
        title_rw="Uruhushya rwawe",
        title_en="Consent",
        rw=("Mbere yo gutangira, usabwa kwemera ko ijwi ryawe rikoreshwa nk'uko byasobanuwe haruguru. "
            "Ushobora guhagarika gukoresha iyi porogaramu igihe icyo ari cyo cyose. "
            "Ntukoreshe ijwi cyangwa amakuru y'undi muntu utabanje kubimubaza, "
            "kandi ntutange amakuru wifuza ko atamenyekana."),
        en=("Affirmative consent is required before any audio is processed. You may stop using the service "
            "at any time, and withdrawing consent does not affect processing already carried out. Please "
            "do not submit another person's voice or personal information without their consent, and avoid "
            "sharing details you would not wish to be processed."),
        key=False,
    ),
    dict(
        n=5,
        title_rw="Umutekano",
        title_en="Security",
        rw=("Dukoresha uburyo bwa tekiniki bukwiye bwo kurinda iyi serivisi, harimo kubika neza "
            "imfunguzo za sisitemu ku buryo zitagaragara mu ikode. Nta serivisi yo kuri interineti "
            "ishobora kwemeza umutekano wuzuye."),
        en=("Reasonable technical and organisational measures protect the service, including keeping "
            "application credentials in environment variables rather than in source code, and restricting "
            "access to system configuration. No online service can be guaranteed completely secure, and "
            "this is stated plainly rather than implied otherwise."),
        key=False,
    ),
    dict(
        n=6,
        title_rw="Aho ubushobozi bugarukira",
        title_en="Limitations and accuracy",
        rw=("Ubushobozi bwo kumva ijwi buratandukana bitewe n'imvugo, ururimi rw'akarere, urusaku, "
            "n'ikibazo ubajije. Sisitemu ikora neza ku bibazo by'ubuzima (WER 6.76%) kurusha "
            "ibiganiro rusange (WER 31.29%). Ishobora kwibeshya. Koresha ubwenge bwawe, "
            "kandi ujye kwa muganga igihe bikenewe."),
        en=("Recognition accuracy varies with accent, dialect, background noise, and topic. Measured word "
            "error rate is 6.76% on health speech and 31.29% on general speech, so performance is "
            "noticeably weaker outside the health domain. The assistant can be wrong. A voice-first "
            "interface also does not serve deaf or hard-of-hearing users, and the system is limited to "
            "Kinyarwanda by design."),
        key=False,
    ),
    dict(
        n=7,
        title_rw="Uburenganzira bwawe",
        title_en="Your rights",
        rw=("Hakurikijwe Itegeko n\u00B0 058/2021 ryerekeye kurengera amakuru bwite n'ubuzima bwite, "
            "ufite uburenganzira bwo kumenya uko amakuru yawe akoreshwa, kuyabona, gusaba ko akosorwa "
            "cyangwa asibwa, kwanga ko akoreshwa, no kujuririra urwego rubishinzwe. "
            f"Kugira ngo ubikoreshe, twandikire kuri {CONTACT_EMAIL}."),
        en=("Under Rwanda's Law No. 058/2021 relating to the protection of personal data and privacy, you "
            "have the right to be informed about how your personal data is processed, to access it, to "
            "request rectification or erasure, to object to processing, and to appeal to the supervisory "
            f"authority. To exercise these rights, contact {CONTACT_EMAIL}. Rights that are never "
            "communicated cannot be exercised, which is why they are stated here."),
        key=True,
    ),
    dict(
        n=8,
        title_rw="Impinduka n'aho watubona",
        title_en="Changes and contact",
        rw=("Aya masezerano ashobora guhinduka; inyandiko iri muri iyi porogaramu ni yo ikurikizwa. "
            f"Ibibazo, ibirego cyangwa ibitekerezo: {CONTACT_EMAIL}."),
        en=("This notice may be updated, and the version published in the system governs your use. "
            f"Questions, complaints, or requests may be sent to {CONTACT_EMAIL}."),
        key=False,
    ),
]

# --------------------------------------------------------------------------
# Stage 1 — Config and auth (no interactive prompts — env vars only)
# --------------------------------------------------------------------------

ASR_ADAPTER = os.environ.get("ASR_ADAPTER_PATH", "./adapter")
ASR_BASE = "badrex/w2v-bert-2.0-kinyarwanda-asr"

print(f"Working directory: {os.getcwd()}")
print(f"Contents: {sorted(os.listdir('.'))}")
if Path(ASR_ADAPTER).exists():
    print(f"Adapter contents: {sorted(os.listdir(ASR_ADAPTER))}")

if not Path(ASR_ADAPTER).exists():
    raise RuntimeError(
        f"Adapter not found at {ASR_ADAPTER}. Commit your LoRA adapter folder "
        f"into the repo at this path, or set ASR_ADAPTER_PATH."
    )

if "OPENROUTER_API_KEY" not in os.environ:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it as an environment variable "
        "in Railway's dashboard — never hardcode it in this file."
    )

if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# --------------------------------------------------------------------------
# Stage 2 — Load ASR (loads once at process start)
# --------------------------------------------------------------------------

from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
from peft import PeftConfig, PeftModel

print("Loading ASR...")
asr_processor = Wav2Vec2BertProcessor.from_pretrained(ASR_ADAPTER)
asr_base = Wav2Vec2BertForCTC.from_pretrained(ASR_BASE, torch_dtype=torch.float32)

peft_config = PeftConfig.from_pretrained(ASR_ADAPTER)
peft_config.task_type = None
asr_model = PeftModel.from_pretrained(asr_base, ASR_ADAPTER, config=peft_config).to(DEVICE).eval()
print("ASR ready (badrex + your LoRA adapter)")

# CTC blank id — the pad token for Wav2Vec2-family models. Needed to exclude
# blank frames from the confidence computation.
CTC_BLANK_ID = resolve_blank_id(asr_processor, asr_base)
print(f"CTC blank id: {CTC_BLANK_ID} | confidence gate: low<{LOW_CONF} high>={HIGH_CONF}")

# --------------------------------------------------------------------------
# Stage 3 — Load TTS (Meta MMS-TTS Kinyarwanda)
# --------------------------------------------------------------------------

from transformers import VitsModel, AutoTokenizer

print("Loading MMS-TTS...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-kin").to(DEVICE).eval()
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kin")
TTS_LABEL = "Meta MMS-TTS"
print("MMS-TTS ready")

# --------------------------------------------------------------------------
# Stage 4 — LLM (Gemini 2.5 Flash via OpenRouter)
# --------------------------------------------------------------------------

LLM_MODEL = "google/gemini-2.5-flash"

# The safety prompt lives in safety_prompt.py so that app.py and
# eval/safety_eval.py send byte-identical input. If the two ever diverged, the
# refusal rate reported in the capstone would describe a system that is not the
# one deployed. Report §4.1.4 documents this module.
from safety_prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_messages


def llm_answer(user_text):
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "X-Title": "UBUZIMA AI",
        },
        json={
            "model": LLM_MODEL,
            "messages": build_messages(user_text),
            "max_tokens": 200,
            "temperature": 0.3,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------------------
# Stage 5 — Pipeline functions
# --------------------------------------------------------------------------

def transcribe(audio_array, sample_rate):
    if np.issubdtype(audio_array.dtype, np.integer):
        audio_array = audio_array.astype(np.float32) / np.iinfo(audio_array.dtype).max
    else:
        audio_array = audio_array.astype(np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)

    inputs = asr_processor.feature_extractor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    )
    print(f"[transcribe] feature extractor output keys: {list(inputs.keys())}")
    inputs = {
        k: v.to(DEVICE) for k, v in inputs.items()
        if k in ("input_features", "attention_mask")
    }
    with torch.no_grad():
        logits = asr_model(**inputs).logits

    # Confidence is computed from the same forward pass — no extra cost.
    conf, n_frames = ctc_confidence(logits, CTC_BLANK_ID)

    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    text = asr_processor.batch_decode(pred_ids)[0].strip()
    print(f"[transcribe] confidence={conf:.3f} non_blank_frames={n_frames} band={confidence_band(conf)}")
    return text, conf


def speak(text):
    clean = re.sub(r"[^a-z'\s]", " ", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None, None
    tok = tts_tokenizer(clean, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        wav = tts_model(**tok).waveform.cpu().numpy()[0]
    return wav, tts_model.config.sampling_rate


def confidence_badge_html(conf, band):
    """Small coloured badge shown next to the transcript. Visible on camera."""
    colour = {"high": "#3ddc84", "medium": "#e8a72c", "low": "#e8722c"}[band]
    label = {"high": TXT["conf_high"], "medium": TXT["conf_medium"],
             "low": TXT["conf_low"]}[band]
    return (
        f'<div class="conf-badge" style="border-color:{colour};color:{colour}">'
        f'<span class="conf-dot" style="background:{colour}"></span>'
        f'{TXT["conf_label"]}: <strong>{conf:.2f}</strong> &nbsp;—&nbsp; {label}'
        f'</div>'
    )


def safe_pipeline(audio_input, consent_given, progress=gr.Progress()):
    """End-to-end ASR -> LLM -> TTS with consent gate, confidence gate,
    progress reporting and graceful errors."""
    blank_badge = ""

    # Consent gate — no audio is processed without affirmative consent.
    if REQUIRE_CONSENT and not consent_given:
        return TXT["e_consent"], "\u2014", None, blank_badge

    if audio_input is None:
        return TXT["e_not_ready"], "\u2014", None, blank_badge

    try:
        sample_rate, audio_array = audio_input
    except Exception as e:
        return TXT["e_format"].format(e=e), "\u2014", None, blank_badge

    if audio_array is None or len(audio_array) == 0:
        return TXT["e_empty"], "\u2014", None, blank_badge

    progress(0.25, desc=TXT["p_asr"])
    try:
        transcript, confidence = transcribe(audio_array, sample_rate)
    except Exception as e:
        traceback.print_exc()
        return TXT["e_asr"].format(t=type(e).__name__, e=e), "\u2014", None, blank_badge

    band = confidence_band(confidence)
    badge = confidence_badge_html(confidence, band)

    if not transcript or len(transcript.strip()) < 2:
        return TXT["e_unclear"], "\u2014", None, badge

    # --- CONFIDENCE GATE (report §5.2.5) -----------------------------------
    # Below the low threshold we do NOT call the language model at all. A
    # garbled transcript would otherwise produce a fluent, confident answer to
    # a question the user never asked — the most dangerous failure mode in a
    # cascaded pipeline. Declining is safer than answering the wrong question.
    if band == "low":
        progress(1.0, desc=TXT["p_done"])
        return transcript, TXT["e_low_conf"], None, badge

    progress(0.5, desc=TXT["p_llm"])
    try:
        answer = llm_answer(transcript)
    except Exception as e:
        traceback.print_exc()
        return transcript, TXT["e_llm"].format(t=type(e).__name__, e=e), None, badge

    # Medium confidence: answer, but tell the user to check the transcript first.
    if band == "medium":
        answer = f"{answer}\n\n{TXT['hedge_medium']}"

    progress(0.85, desc=TXT["p_tts"])
    try:
        wav, sr = speak(answer)
        audio_out = (sr, (wav * 32767).astype(np.int16)) if wav is not None else None
    except Exception as e:
        traceback.print_exc()
        return transcript, f"{answer}\n\n{TXT['e_tts'].format(e=e)}", None, badge

    progress(1.0, desc=TXT["p_done"])
    return transcript, answer, audio_out, badge


print("Pipeline ready")
print(f"  ASR: badrex + LoRA")
print(f"  LLM: {LLM_MODEL} (via OpenRouter)")
print(f"  TTS: {TTS_LABEL}")
print(f"  Consent gate: {'ON' if REQUIRE_CONSENT else 'OFF'}")
print(f"  Confidence gate: low<{LOW_CONF} / medium / high>={HIGH_CONF}")
print(f"  Safety prompt  : {PROMPT_VERSION}")

# --------------------------------------------------------------------------
# Stage 6 — UI (dark theme, Kinyarwanda-first)
# --------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    ("\U0001F99F", "Malariya", "Ni iki gikora malariya kandi nigute twayirinda?"),
    ("\U0001F476", "Umwana ufite umuriro", "Umwana wanjye ufite umuriro mwinshi, nakora iki?"),
    ("\U0001F927", "Ubwandu", "Nigute ndinda ubwandu bw'ubuhumekero?"),
    ("\U0001F4A7", "Amazi n'ubuzima", "Ese kunywa amazi menshi bifite akamaro ki ku buzima?"),
    ("\U0001F915", "Umutwe ubabaza", "Mfite umutwe ubabaza kuva ejo, ni iki nakora?"),
    ("\U0001F930", "Ubuzima bw'ababyeyi", "Umugore utwite agomba kurya iki?"),
]

CUSTOM_CSS = """
:root {
  --uz-bg:#0f0f11; --uz-card:#17171b; --uz-card2:#1c1c21; --uz-border:#2a2a31;
  --uz-orange:#e8722c; --uz-orange-soft:#f0997b; --uz-indigo:#6366d9;
  --uz-text:#e7e7ea; --uz-muted:#9a9aa4; --uz-green:#3ddc84;
}
.gradio-container { background: var(--uz-bg) !important; max-width: 940px !important; }
.hero { background: var(--uz-card); border: 1px solid var(--uz-border); border-radius: 16px; padding: 26px 28px; margin-bottom: 18px; }
.hero-top { display:flex; align-items:center; justify-content:space-between; gap:16px; }
.hero-id { display:flex; align-items:center; gap:14px; }
.hero-logo { width:46px; height:46px; border-radius:12px; background:var(--uz-orange); display:flex; align-items:center; justify-content:center; font-size:22px; }
.hero h1 { font-size:1.9em !important; font-weight:700; margin:0 !important; color:var(--uz-orange) !important; letter-spacing:-0.5px; }
.hero .tagline { color:var(--uz-muted); font-size:0.95em; margin:2px 0 0; }
.live-badge { background:rgba(61,220,132,0.12); border:1px solid rgba(61,220,132,0.4); color:var(--uz-green); padding:6px 14px; border-radius:999px; font-size:0.82em; font-weight:600; white-space:nowrap; }
.live-badge .dot { display:inline-block; width:7px; height:7px; background:var(--uz-green); border-radius:50%; margin-right:6px; }
.status-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }
.pill { padding:7px 14px; border-radius:999px; font-size:0.82em; font-weight:500; border:1px solid var(--uz-border); }
.pill.stack { background:rgba(99,102,217,0.14); border-color:rgba(99,102,217,0.4); color:#b9baf5; }
.pill.wer { background:rgba(232,114,44,0.14); border-color:rgba(232,114,44,0.4); color:var(--uz-orange-soft); }
.pill .dot { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:7px; }
.pill.stack .dot { background:var(--uz-indigo); }
.pill.wer .dot { background:var(--uz-orange); }
.section-card, .gr-group { background:var(--uz-card) !important; border:1px solid var(--uz-border) !important; border-radius:14px !important; }
.card-head { display:flex; align-items:center; gap:12px; margin-bottom:2px; }
.card-num { width:26px; height:26px; border-radius:50%; background:var(--uz-card2); border:1px solid var(--uz-border); color:var(--uz-muted); display:flex; align-items:center; justify-content:center; font-size:0.85em; font-weight:600; }
.card-title { color:var(--uz-text); font-weight:600; font-size:1.02em; }
.voice-title { color:var(--uz-text); font-weight:600; font-size:1.15em; }
.voice-hint { color:var(--uz-muted); font-size:0.88em; margin:6px 0 2px; }
.consent-note { color:var(--uz-muted); font-size:0.8em; margin:2px 0 6px; font-style:italic; }
.conf-badge { display:inline-flex; align-items:center; gap:8px; border:1px solid var(--uz-border);
  border-radius:999px; padding:6px 14px; font-size:0.85em; margin:6px 0 2px; background:var(--uz-card2); }
.conf-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.examples-head { font-weight:600; color:var(--uz-text); margin:2px 0 12px; font-size:1.0em; }
.examples-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.example-card { background:var(--uz-card2); border:1px solid var(--uz-border); border-radius:12px; padding:14px 16px; transition:all 0.18s ease; }
.example-card:hover { border-color:var(--uz-orange); transform:translateY(-1px); }
.example-card .icon { font-size:1.35em; margin-bottom:4px; }
.example-card .topic { font-weight:600; font-size:0.9em; color:var(--uz-text); margin-bottom:3px; }
.example-card .question { font-size:0.85em; color:var(--uz-muted); font-style:italic; line-height:1.4; }
button.primary { background:var(--uz-orange) !important; border:none !important; color:#1a0d04 !important; font-weight:600 !important; font-size:1.02em !important; }
button.primary:hover { background:#f0842f !important; }
.disclaimer { margin-top:20px; padding:14px 18px; background:rgba(232,114,44,0.08); border-left:4px solid var(--uz-orange); border-radius:8px; font-size:0.88em; color:#f0c9a8; }
.disclaimer strong { color:var(--uz-orange); }
.credits { text-align:center; font-size:0.78em; color:var(--uz-muted); margin-top:14px; padding:8px; }

/* ---- Terms & Privacy tab ---- */
.terms-wrap { background:var(--uz-card); border:1px solid var(--uz-border); border-radius:16px; padding:26px 30px; }
.terms-wrap h2 { color:var(--uz-orange) !important; font-size:1.5em !important; margin:0 0 6px !important; }
.terms-intro { color:var(--uz-text); font-size:0.94em; line-height:1.6; margin:0 0 4px; }
.terms-intro.en { color:var(--uz-muted); font-style:italic; }
.terms-updated { color:var(--uz-muted); font-size:0.78em; margin:10px 0 20px; }
.clause { background:var(--uz-card2); border:1px solid var(--uz-border); border-left:4px solid var(--uz-border); border-radius:12px; padding:16px 20px; margin-bottom:12px; }
.clause.key { border-left-color:var(--uz-orange); background:rgba(232,114,44,0.06); }
.clause-title { display:flex; align-items:baseline; gap:10px; margin-bottom:8px; flex-wrap:wrap; }
.clause-n { width:24px; height:24px; min-width:24px; border-radius:50%; background:var(--uz-orange); color:#1a0d04; display:inline-flex; align-items:center; justify-content:center; font-size:0.8em; font-weight:700; }
.clause-rw { color:var(--uz-text); font-weight:600; font-size:1.02em; }
.clause-en-t { color:var(--uz-muted); font-size:0.85em; font-style:italic; }
.clause p { margin:0 0 8px; line-height:1.6; }
.clause .rw { color:var(--uz-text); font-size:0.92em; }
.clause .en { color:var(--uz-muted); font-size:0.86em; }
.key-badge { background:rgba(232,114,44,0.18); border:1px solid rgba(232,114,44,0.45); color:var(--uz-orange-soft); font-size:0.68em; padding:2px 8px; border-radius:999px; font-weight:600; letter-spacing:0.3px; }
.legal-foot { margin-top:18px; padding:14px 18px; background:var(--uz-card2); border:1px solid var(--uz-border); border-radius:10px; font-size:0.82em; color:var(--uz-muted); line-height:1.6; }
@media (max-width:768px){ .examples-grid{ grid-template-columns:1fr; } .hero h1{ font-size:1.5em !important; } .terms-wrap{ padding:18px; } }
"""


def hero_html():
    return f"""
    <div class="hero">
      <div class="hero-top">
        <div class="hero-id">
          <div class="hero-logo">\U0001FA7A</div>
          <div>
            <h1>UBUZIMA AI</h1>
            <p class="tagline">{TXT['tagline']}</p>
          </div>
        </div>
        <span class="live-badge"><span class="dot"></span>{TXT['live']}</span>
      </div>
      <p class="tagline" style="margin-top:14px;">{TXT['subtitle']}</p>
      <div class="status-row">
        <span class="pill stack"><span class="dot"></span>{TXT['pill_asr']}</span>
        <span class="pill stack"><span class="dot"></span>{TXT['pill_llm']}</span>
        <span class="pill stack"><span class="dot"></span>TTS: {TTS_LABEL}</span>
        <span class="pill wer"><span class="dot"></span>{TXT['pill_wer']}</span>
      </div>
    </div>
    """


def examples_html():
    cards = ""
    for icon, topic, q in EXAMPLE_QUESTIONS:
        cards += f"""
        <div class="example-card">
          <div class="icon">{icon}</div>
          <div class="topic">{topic}</div>
          <div class="question">"{q}"</div>
        </div>
        """
    return f'<div class="examples-head">{TXT["examples_head"]}</div><div class="examples-grid">{cards}</div>'


def card_head(num, title):
    return f'<div class="card-head"><div class="card-num">{num}</div><div class="card-title">{title}</div></div>'


def terms_html():
    """Dedicated Terms of Use and Privacy Notice — navigate here in the video."""
    body = ""
    for c in CLAUSES:
        badge = '<span class="key-badge">INGENZI / KEY</span>' if c["key"] else ""
        body += f"""
        <div class="clause{' key' if c['key'] else ''}">
          <div class="clause-title">
            <span class="clause-n">{c['n']}</span>
            <span class="clause-rw">{c['title_rw']}</span>
            <span class="clause-en-t">{c['title_en']}</span>
            {badge}
          </div>
          <p class="rw">{c['rw']}</p>
          <p class="en">{c['en']}</p>
        </div>
        """
    return f"""
    <div class="terms-wrap">
      <h2>\U0001F512 Amasezerano y'Ikoreshwa n'Itangazo ry'Ibanga</h2>
      <p class="terms-intro">{TERMS_INTRO_RW}</p>
      <p class="terms-intro en">Terms of Use and Privacy Notice \u2014 {TERMS_INTRO_EN}</p>
      <p class="terms-updated">{TERMS_UPDATED}</p>
      {body}
      <div class="legal-foot">
        <strong>Amategeko akurikizwa / Governing frameworks.</strong>
        Itegeko n\u00B0 058/2021 ryerekeye kurengera amakuru bwite n'ubuzima bwite (Republic of Rwanda);
        African Union Convention on Cyber Security and Personal Data Protection (2014);
        WHO guidance on the ethics and governance of artificial intelligence for health (2021);
        UNESCO Recommendation on the Ethics of Artificial Intelligence (2021).
        <br><br>
        Uyu mushinga wemejwe na Komite y'Ubushakashatsi bw'Imyitwarire ya ALU
        (ALU Senate Research Ethics Committee), nomero M26-BSE-112, 22 Nyakanga 2026.
      </div>
    </div>
    """


def footer_html():
    return f"""
    <div class="disclaimer">
      <strong>{TXT['disc_strong']}</strong> {TXT['disc_body']}
    </div>
    <div class="credits">
      badrex ASR + LoRA (Afrivoice + Common Voice) \u00B7 Gemini 2.5 Flash \u00B7 {TTS_LABEL}
      <br>URURIMI / UBUZIMA AI \u2014 African Leadership University, Kigali
    </div>
    """


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
)

# Force dark mode on load so the CSS palette always applies.
FORCE_DARK_JS = """
function() {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url.href);
  }
}
"""

# analytics_enabled=False stops Gradio telemetry; delete_cache clears temporary
# audio files, which is what makes the retention clause in the notice true.
_blocks_kwargs = dict(
    title="UBUZIMA AI",
    theme=theme,
    css=CUSTOM_CSS,
    js=FORCE_DARK_JS,
    analytics_enabled=False,
)
try:
    demo = gr.Blocks(**_blocks_kwargs, delete_cache=(1800, 1800))
except TypeError:
    # Older Gradio without delete_cache support.
    demo = gr.Blocks(**_blocks_kwargs)

with demo:
    gr.HTML(hero_html())

    with gr.Tabs():
        # ---------------- Tab 1: the assistant ----------------
        with gr.Tab(TXT["tab_main"]):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["section-card"]):
                        gr.HTML(f'<div class="voice-title">\U0001F3A4 {TXT["voice_card"]}</div>'
                                f'<div class="voice-hint">{TXT["voice_hint"]}</div>')
                        audio_in = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="",
                            show_label=False,
                            waveform_options=gr.WaveformOptions(
                                waveform_color="#e8722c",
                                waveform_progress_color="#f0997b",
                            ),
                        )
                        consent_box = gr.Checkbox(
                            label=TXT["consent_label"],
                            value=False,
                            interactive=True,
                        )
                        gr.HTML(f'<div class="consent-note">{TXT["consent_note"]}</div>')
                        with gr.Row():
                            submit = gr.Button(TXT["btn_run"], variant="primary", size="lg", scale=3)
                            clear = gr.Button(TXT["btn_reset"], scale=1)
                    gr.HTML(examples_html())

                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["section-card"]):
                        gr.HTML(card_head(1, TXT["card1"]))
                        transcript_out = gr.Textbox(
                            label="", show_label=False, lines=2, interactive=False,
                            placeholder=TXT["card1_ph"],
                        )
                        confidence_out = gr.HTML(value="")
                    with gr.Group(elem_classes=["section-card"]):
                        gr.HTML(card_head(2, TXT["card2"]))
                        answer_out = gr.Textbox(
                            label="", show_label=False, lines=5, interactive=False,
                            placeholder=TXT["card2_ph"],
                        )
                    with gr.Group(elem_classes=["section-card"]):
                        gr.HTML(card_head(3, TXT["card3"]))
                        audio_out = gr.Audio(label="", show_label=False, type="numpy", autoplay=True)

        # ---------------- Tab 2: Terms & Privacy (walk through on camera) ----------------
        with gr.Tab(TXT["tab_terms"]):
            gr.HTML(terms_html())

    gr.HTML(footer_html())

    submit.click(
        safe_pipeline,
        inputs=[audio_in, consent_box],
        outputs=[transcript_out, answer_out, audio_out, confidence_out],
        show_progress="full",
    )
    clear.click(
        lambda: (None, "", "", None, ""),
        outputs=[audio_in, transcript_out, answer_out, audio_out, confidence_out],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
