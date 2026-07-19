"""
UBUZIMA AI — Kinyarwanda Voice Health Assistant
Production build for Railway (adapted from the Colab demo notebook).

Pipeline: badrex/w2v-bert-2.0-kinyarwanda-asr + your LoRA adapter (bundled
in ./adapter) -> Gemini 2.5 Flash via OpenRouter -> Meta MMS-TTS (Kinyarwanda)

Required environment variables (set these in Railway's dashboard, never
hardcode them):
    OPENROUTER_API_KEY   - from openrouter.ai/keys
    HF_TOKEN              - optional, only needed if badrex's model is gated

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

# --------------------------------------------------------------------------
# Stage 1 — Config and auth (no interactive prompts — env vars only)
# --------------------------------------------------------------------------

ASR_ADAPTER = os.environ.get("ASR_ADAPTER_PATH", "./adapter")
ASR_BASE = "badrex/w2v-bert-2.0-kinyarwanda-asr"

# Debug aid: print what actually made it into the container so a missing
# adapter shows up clearly in the Railway logs instead of a bare error.
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
from peft import PeftModel

print("Loading ASR...")
asr_processor = Wav2Vec2BertProcessor.from_pretrained(ASR_ADAPTER)
asr_base = Wav2Vec2BertForCTC.from_pretrained(ASR_BASE, torch_dtype=torch.float32)
asr_model = PeftModel.from_pretrained(asr_base, ASR_ADAPTER).to(DEVICE).eval()
print("ASR ready (badrex + your LoRA adapter)")

# --------------------------------------------------------------------------
# Stage 3 — Load TTS (MMS-TTS only — lighter and more reliable in containers
# than Coqui YourTTS, which needs espeak-ng and a large extra model download)
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
    # Some processor/transformers version combinations include extra keys
    # (e.g. input_ids, which belongs to text tokenization, not audio
    # features) in this output. Wav2Vec2BertForCTC.forward() only accepts
    # input_features and attention_mask, so filter explicitly rather than
    # passing the raw dict through.
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
        return (
            "⚠️ Audio not ready. After clicking Stop on the mic, wait 1-2 seconds, then click Process again.",
            "—", None,
        )

    try:
        sample_rate, audio_array = audio_input
    except Exception as e:
        return f"⚠️ Audio format issue: {e}", "—", None

    if audio_array is None or len(audio_array) == 0:
        return "⚠️ Empty audio. Try recording again.", "—", None

    progress(0.25, desc="🎙️ Transcribing your Kinyarwanda...")
    try:
        transcript = transcribe(audio_array, sample_rate)
    except Exception as e:
        traceback.print_exc()
        return f"⚠️ ASR error: {type(e).__name__}: {e}", "—", None

    if not transcript or len(transcript.strip()) < 2:
        return "(Couldn't understand the audio. Try speaking louder and a bit slower.)", "—", None

    progress(0.5, desc="💭 UBUZIMA AI is thinking...")
    try:
        answer = llm_answer(transcript)
    except Exception as e:
        traceback.print_exc()
        return transcript, f"⚠️ LLM error: {type(e).__name__}: {e}", None

    progress(0.85, desc="🔊 Generating spoken answer...")
    try:
        wav, sr = speak(answer)
        audio_out = (sr, (wav * 32767).astype(np.int16)) if wav is not None else None
    except Exception as e:
        traceback.print_exc()
        return transcript, f"{answer}\n\n[⚠️ TTS error: {e}]", None

    progress(1.0, desc="✓ Done")
    return transcript, answer, audio_out


print("Pipeline ready")
print(f"  ASR: badrex + LoRA")
print(f"  LLM: {LLM_MODEL} (via OpenRouter)")
print(f"  TTS: {TTS_LABEL}")

# --------------------------------------------------------------------------
# Stage 6 — UI
# --------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    ("🦟", "Malariya", "Ni iki gikora malariya kandi nigute twayirinda?"),
    ("👶", "Umwana ufite umuriro", "Umwana wanjye ufite umuriro mwinshi, nakora iki?"),
    ("🤧", "Ubwandu", "Nigute ndinda ubwandu bw'ubuhumekero?"),
    ("💧", "Amazi & ubuzima", "Ese kunywa amazi menshi bifite akamaro ki ku buzima?"),
    ("🤕", "Umutwe ubabaza", "Mfite umutwe ubabaza kuva ejo, ni iki nakora?"),
    ("🤰", "Ubuzima bw'ababyeyi", "Umugore utwite agomba kurya iki?"),
]

CUSTOM_CSS = """
:root {
  --primary: #00897b; --primary-dark: #00695c; --primary-light: #4db6ac;
  --accent: #ffa726; --bg-soft: #f0fdf9; --text-muted: #546e7a; --border: #e0f2f1;
}
.hero {
  background: linear-gradient(135deg, #00897b 0%, #00bfa5 60%, #4db6ac 100%);
  color: white; padding: 32px 28px; border-radius: 16px; margin-bottom: 20px;
  box-shadow: 0 8px 24px rgba(0, 137, 123, 0.18);
}
.hero h1 { font-size: 2.4em !important; font-weight: 700; margin: 0 0 8px 0 !important; color: white !important; letter-spacing: -0.5px; }
.hero p { font-size: 1.05em; margin: 0; opacity: 0.96; color: white !important; }
.hero .tagline { font-weight: 500; font-size: 1.15em; margin-bottom: 4px !important; }
.status-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.status-pill { background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28); color: white; padding: 6px 14px; border-radius: 999px; font-size: 0.85em; font-weight: 500; }
.status-pill .dot { display: inline-block; width: 7px; height: 7px; background: #76ff03; border-radius: 50%; margin-right: 6px; box-shadow: 0 0 6px #76ff03; }
.section-card { background: white; border: 1px solid var(--border); border-radius: 14px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.examples-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
.example-card { background: var(--bg-soft); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; transition: all 0.2s ease; }
.example-card:hover { background: white; border-color: var(--primary-light); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0, 137, 123, 0.08); }
.example-card .icon { font-size: 1.4em; margin-bottom: 4px; }
.example-card .topic { font-weight: 600; font-size: 0.85em; color: var(--primary-dark); margin-bottom: 4px; }
.example-card .question { font-size: 0.85em; color: var(--text-muted); font-style: italic; line-height: 1.4; }
button.lg.primary { background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%) !important; border: none !important; font-weight: 600 !important; font-size: 1.05em !important; padding: 14px 24px !important; }
button.lg.primary:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0, 137, 123, 0.28) !important; }
.howto { background: #e1f5fe; border-left: 4px solid #0288d1; padding: 12px 16px; border-radius: 8px; font-size: 0.9em; margin: 12px 0; color: #01579b; }
.disclaimer { margin-top: 24px; padding: 14px 18px; background: #fff8e1; border-left: 4px solid var(--accent); border-radius: 8px; font-size: 0.88em; color: #5d4037; }
.disclaimer strong { color: #e65100; }
.credits { text-align: center; font-size: 0.78em; color: var(--text-muted); margin-top: 14px; padding: 10px; }
.credits a { color: var(--primary); text-decoration: none; }
@media (max-width: 768px) { .examples-grid { grid-template-columns: 1fr; } .hero h1 { font-size: 1.8em !important; } }
"""


def hero_html():
    llm_short = LLM_MODEL.split("/")[-1]
    return f"""
    <div class="hero">
      <h1>🩺 UBUZIMA AI</h1>
      <p class="tagline">Kinyarwanda voice health assistant</p>
      <p>Speak a health question, get a spoken answer in Kinyarwanda</p>
      <div class="status-row">
        <span class="status-pill"><span class="dot"></span> ASR: badrex + LoRA</span>
        <span class="status-pill"><span class="dot"></span> LLM: {llm_short}</span>
        <span class="status-pill"><span class="dot"></span> TTS: {TTS_LABEL}</span>
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
    return f"""
    <div>
      <div style="font-weight:600;color:var(--primary-dark);margin:0 0 12px;">💡 Suggested questions to ask</div>
      <div class="examples-grid">{cards}</div>
    </div>
    """


def howto_html():
    return """
    <div class="howto">
      <strong>How to use:</strong>
      <ol style="margin:6px 0 0 18px;padding:0;">
        <li>Click <strong>Record</strong> on the microphone</li>
        <li>Speak your question in Kinyarwanda</li>
        <li>Click <strong>Stop</strong> on the mic, wait 1-2 seconds for the waveform to settle</li>
        <li>Click <strong>✨ Process question</strong></li>
      </ol>
    </div>
    """


def footer_html():
    return f"""
    <div class="disclaimer">
      <strong>⚠️ Research demo only.</strong> Not a medical diagnosis tool.
      Always consult a qualified healthcare professional for medical concerns.
    </div>
    <div class="credits">
      Built on
      <a href="https://huggingface.co/badrex/w2v-bert-2.0-kinyarwanda-asr" target="_blank">badrex/w2v-bert-2.0-kinyarwanda-asr</a>
      · Fine-tuned with LoRA on Afrivoice + Common Voice Kinyarwanda · TTS: {TTS_LABEL}
      <br><em>URURIMI / UBUZIMA AI capstone — African Leadership University, Kigali</em>
    </div>
    """


theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

with gr.Blocks(title="UBUZIMA AI", theme=theme, css=CUSTOM_CSS) as demo:
    gr.HTML(hero_html())
    gr.HTML(howto_html())

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### 🎙️ Speak your question")
                audio_in = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="",
                    show_label=False,
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#00897b",
                        waveform_progress_color="#00bfa5",
                    ),
                )
                with gr.Row():
                    submit = gr.Button("✨ Process question", variant="primary", size="lg", scale=3)
                    clear = gr.Button("Clear", scale=1)
            gr.HTML(examples_html())

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### 📝 What you said")
                transcript_out = gr.Textbox(
                    label="", show_label=False, lines=2, interactive=False,
                    placeholder="Your transcribed question will appear here...",
                )
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### 💬 UBUZIMA AI says")
                answer_out = gr.Textbox(
                    label="", show_label=False, lines=5, interactive=False,
                    placeholder="The AI's response in Kinyarwanda will appear here...",
                )
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### 🔊 Spoken answer")
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
    # server_name="0.0.0.0" + reading $PORT are both required for Railway.
    # No share=True / debug=True here — those are Colab-only conveniences
    # and don't belong in a production container.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
