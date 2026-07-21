"""
UBUZIMA AI — Umufasha w'ubuzima uvuga mu Kinyarwanda
Production build for Railway (adapted from the Colab demo notebook).

Pipeline: badrex/w2v-bert-2.0-kinyarwanda-asr + your LoRA adapter (bundled
in ./adapter) -> Gemini 2.5 Flash via OpenRouter -> Meta MMS-TTS (Kinyarwanda)

Required environment variables (set these in Railway's dashboard, never
hardcode them):
    OPENROUTER_API_KEY   - from openrouter.ai/keys
    HF_TOKEN             - optional, only needed if badrex's model is gated

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

    "voice_card": "Baza amajwi hano",
    "voice_hint": "Kanda mikoro, uvuge ikibazo cyawe, hanyuma ukande \u201cTanga igisubizo\u201d.",
    "btn_run": "\u25B6 Tanga igisubizo",
    "btn_reset": "\u21BA Siba",

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

SYSTEM_PROMPT = (
    "URURIMI/UBUZIMA AI ni umufasha mu by'ubuzima utanga inama z'ibanze. "
    "AMABWIRIZA Y'INGENZI: "
    "(1) Subiza mu Kinyarwanda gusa, ntushyiremo amagambo y'icyongereza cyangwa imibare. "
    "(2) Andika interuro nke (2-4) zoroshye, zumvikana ku muntu wese. "
    "(3) Niba ikibazo gikomeye (umuriro mwinshi, kuruka amaraso, kubabara cyane k'umutima, "
    "ababyeyi batwite bafite ibibazo), vuga ko bagomba kujya kwa muganga vuba. "
    "(4) Ntiwivuge ko uri robot cyangwa AI — uvuge nk'umufasha w'ubuzima usanzwe. "
    "Urugero rwo gusubiza: 'Malariya iterwa n'imibu. Kugira ngo wirinde, "
    "koresha agasenge ko kurara, urindire ibibarafu mu rugo, kandi sura muganga "
    "iyo ugize ibimenyetso.'"
)


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
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
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
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    return asr_processor.batch_decode(pred_ids)[0].strip()


def speak(text):
    clean = re.sub(r"[^a-z'\s]", " ", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None, None
    tok = tts_tokenizer(clean, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        wav = tts_model(**tok).waveform.cpu().numpy()[0]
    return wav, tts_model.config.sampling_rate


def safe_pipeline(audio_input, progress=gr.Progress()):
    """End-to-end ASR -> LLM -> TTS with visible progress and graceful errors."""
    if audio_input is None:
        return TXT["e_not_ready"], "\u2014", None

    try:
        sample_rate, audio_array = audio_input
    except Exception as e:
        return TXT["e_format"].format(e=e), "\u2014", None

    if audio_array is None or len(audio_array) == 0:
        return TXT["e_empty"], "\u2014", None

    progress(0.25, desc=TXT["p_asr"])
    try:
        transcript = transcribe(audio_array, sample_rate)
    except Exception as e:
        traceback.print_exc()
        return TXT["e_asr"].format(t=type(e).__name__, e=e), "\u2014", None

    if not transcript or len(transcript.strip()) < 2:
        return TXT["e_unclear"], "\u2014", None

    progress(0.5, desc=TXT["p_llm"])
    try:
        answer = llm_answer(transcript)
    except Exception as e:
        traceback.print_exc()
        return transcript, TXT["e_llm"].format(t=type(e).__name__, e=e), None

    progress(0.85, desc=TXT["p_tts"])
    try:
        wav, sr = speak(answer)
        audio_out = (sr, (wav * 32767).astype(np.int16)) if wav is not None else None
    except Exception as e:
        traceback.print_exc()
        return transcript, f"{answer}\n\n{TXT['e_tts'].format(e=e)}", None

    progress(1.0, desc=TXT["p_done"])
    return transcript, answer, audio_out


print("Pipeline ready")
print(f"  ASR: badrex + LoRA")
print(f"  LLM: {LLM_MODEL} (via OpenRouter)")
print(f"  TTS: {TTS_LABEL}")

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
@media (max-width:768px){ .examples-grid{ grid-template-columns:1fr; } .hero h1{ font-size:1.5em !important; } }
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

with gr.Blocks(title="UBUZIMA AI", theme=theme, css=CUSTOM_CSS, js=FORCE_DARK_JS) as demo:
    gr.HTML(hero_html())

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
            with gr.Group(elem_classes=["section-card"]):
                gr.HTML(card_head(2, TXT["card2"]))
                answer_out = gr.Textbox(
                    label="", show_label=False, lines=5, interactive=False,
                    placeholder=TXT["card2_ph"],
                )
            with gr.Group(elem_classes=["section-card"]):
                gr.HTML(card_head(3, TXT["card3"]))
                audio_out = gr.Audio(label="", show_label=False, type="numpy", autoplay=True)

    gr.HTML(footer_html())

    submit.click(
        safe_pipeline,
        inputs=audio_in,
        outputs=[transcript_out, answer_out, audio_out],
        show_progress="full",
    )
    clear.click(
        lambda: (None, "", "", None),
        outputs=[audio_in, transcript_out, answer_out, audio_out],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
