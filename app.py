"""
UBUZIMA AI — Kinyarwanda Voice Health Assistant (Hugging Face Spaces build)
============================================================================

This is the PORTABLE deployment of UBUZIMA AI, adapted from the original
Colab research notebook (ubuzima_ai_platform.ipynb) for hosting on Hugging
Face Spaces with a permanent public URL.

WHAT'S DIFFERENT FROM THE FULL COLAB PIPELINE
-----------------------------------------------
The Colab notebook's TTS stage tries C4IR-RW/kinya-flex-tts first (3 speaker
voices, 24kHz, via the DeepKIN/MorphoKIN toolchain) and falls back to
facebook/mms-tts-kin if that's unavailable. kinya-flex-tts requires:
  - A licensed MorphoKIN binary (26GB download + a license file you must
    upload interactively — there is no way to bake this into a public,
    shareable Space)
  - A background daemon process started with `sudo`
  - Building NVIDIA apex from source
  - A CUDA GPU for reasonable latency

None of that is compatible with a public, shareable, always-on host. So this
Space runs ONLY the MMS-TTS fallback path — the same fallback your own
notebook already falls back to when kinya-flex-tts isn't available. Nothing
here is invented; it's the subset of your own pipeline that doesn't need a
license file or a GPU daemon.

Result: single voice instead of three, but a genuinely permanent URL that
works with zero manual setup for anyone who opens it.

The full 3-voice pipeline is what's demoed in the project video, run from
the original Colab notebook. See README.md and DEPLOYMENT.md for details.

ARCHITECTURE (same as Colab notebook)
--------------------------------------
  Ears  (ASR): akera/whisper-large-v3-kin-200h-v2   (unchanged)
  Brain (LLM): Google Gemini (gemini-2.5-flash)     (unchanged)
  Mouth (TTS): facebook/mms-tts-kin                 (fallback-only here)
"""

import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import librosa
import torch
import google.generativeai as genai
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
    VitsModel,
    AutoTokenizer,
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path("/tmp/ubuzima_ai")
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASR_MODEL_ID = "akera/whisper-large-v3-kin-200h-v2"
TTS_FALLBACK = "facebook/mms-tts-kin"
GEMINI_MODEL = "gemini-2.5-flash"

LANGUAGE = "swahili"  # closest Bantu language Whisper's tokenizer knows
SAMPLE_RATE = 16000
TTS_FALLBACK_SR = 16000

MAX_TOKENS = 500
TEMPERATURE = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

SYSTEM_PROMPT = """
Witwa UBUZIMA AI. Uri umufasha mu by'ubuzima utanga inama z'ibanze mu Kinyarwanda.

AMABWIRIZA:
1. Subiza mu Kinyarwanda gusa, mu nteruro 2-3 zoroshye. Ntukarenze amagambo 60.
""".strip()

EXAMPLES = [
    ("🦟", "Ni iki gikora malariya?", "Malaria"),
    ("🌡️", "Umwana wanjye afite umuriro, ni iki nakora?", "Umuriro w'umwana"),
    ("🤧", "Uburyo bwo kwirinda indwara z'ubuhumekero?", "Ubuhumekero"),
    ("💧", "Ni ryari nagomba kunywa amazi menshi?", "Amazi"),
    ("🤕", "Mfite umutwe ubabaza, nakora iki?", "Umutwe"),
    ("🤰", "Umugore utwite agomba kurya iki?", "Gusama"),
]

# ─────────────────────────────────────────────────────────────
# Auth — set GOOGLE_API_KEY as a Space secret (Settings → Repository secrets)
# ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Add it in this Space's Settings → "
        "Repository secrets (do NOT hardcode it in app.py)."
    )
genai.configure(api_key=GOOGLE_API_KEY)


# ─────────────────────────────────────────────────────────────
# ASR — SpeechRecognizer (identical to the Colab notebook's Cell 4)
# ─────────────────────────────────────────────────────────────
class SpeechRecognizer:
    """Kinyarwanda ASR using akera Whisper-large-v3."""

    def __init__(self):
        print(f"Loading ASR: {ASR_MODEL_ID}...")
        self.processor = WhisperProcessor.from_pretrained(ASR_MODEL_ID)
        self.processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task="transcribe")

        self.model = WhisperForConditionalGeneration.from_pretrained(
            ASR_MODEL_ID, torch_dtype=DTYPE,
        ).to(DEVICE).eval()

        # Fix outdated generation config from the akera checkpoint
        self.model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=LANGUAGE, task="transcribe"
        )
        print("✓ ASR loaded")

    def transcribe(self, audio_tuple) -> str:
        if audio_tuple is None:
            return ""

        sr, arr = audio_tuple
        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if np.max(np.abs(arr)) > 1.0:
            arr = arr / 32768.0
        if sr != SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)

        inputs = self.processor.feature_extractor(
            arr, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(DEVICE, dtype=DTYPE)

        with torch.no_grad():
            ids = self.model.generate(
                inputs.input_features,
                max_length=225,
                forced_decoder_ids=self.forced_decoder_ids,
            )

        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ─────────────────────────────────────────────────────────────
# LLM — GeminiAssistant (identical to the Colab notebook's Cell 5)
# ─────────────────────────────────────────────────────────────
class GeminiAssistant:
    """Health Q&A using Google Gemini (Kinyarwanda-capable, free tier)."""

    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            generation_config={"temperature": TEMPERATURE, "max_output_tokens": 1024},
        )
        print(f"✓ Gemini client ready (model={GEMINI_MODEL})")

    def ask(self, question: str) -> str:
        response = self.model.generate_content(question)
        return response.text.strip()


# ─────────────────────────────────────────────────────────────
# TTS — MMSFallbackTTS only (the portable subset of the Colab notebook's Cell 6)
# ─────────────────────────────────────────────────────────────
class MMSFallbackTTS:
    """Meta MMS-TTS Kinyarwanda — single-speaker, no license/GPU-daemon required."""

    def __init__(self):
        print(f"Loading MMS-TTS: {TTS_FALLBACK}...")
        self.tokenizer = AutoTokenizer.from_pretrained(TTS_FALLBACK)
        self.model = VitsModel.from_pretrained(TTS_FALLBACK).to(DEVICE).eval()
        self.sample_rate = TTS_FALLBACK_SR
        self.supports_speakers = False
        print("✓ MMS-TTS loaded (single speaker, 16 kHz)")

    def synthesize(self, text: str, speaker: int = 0):
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            waveform = self.model(**inputs).waveform[0]
        audio = waveform.cpu().numpy().astype(np.float32)
        return (self.sample_rate, audio)


# ─────────────────────────────────────────────────────────────
# Pipeline orchestrator (identical logic to the Colab notebook's Cell 7)
# ─────────────────────────────────────────────────────────────
class Timer:
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = round(time.perf_counter() - self._t0, 3)


class UbuzimaPipeline:
    def __init__(self, asr, llm, tts):
        self.asr, self.llm, self.tts = asr, llm, tts

    def run_voice(self, audio_tuple):
        if audio_tuple is None:
            return "⚠️ Nta majwi yumvikanye", "", None, ""

        latency = {}
        with Timer() as t:
            transcript = self.asr.transcribe(audio_tuple)
        latency["asr"] = t.elapsed

        if not transcript.strip():
            return "⚠️ Sinumva neza — ongera uvuge", "", None, ""

        with Timer() as t:
            answer = self.llm.ask(transcript)
        latency["llm"] = t.elapsed

        audio_out = None
        with Timer() as t:
            audio_out = self.tts.synthesize(answer)
        latency["tts"] = t.elapsed

        latency["total"] = round(sum(latency.values()), 3)
        lat_str = " · ".join(f"{k.upper()} {v * 1000:.0f}ms" for k, v in latency.items())
        return transcript, answer, audio_out, f"⏱ {lat_str}"

    def run_text(self, question):
        if not question or not question.strip():
            return "", "", None, ""

        latency = {}
        with Timer() as t:
            answer = self.llm.ask(question)
        latency["llm"] = t.elapsed

        with Timer() as t:
            audio_out = self.tts.synthesize(answer)
        latency["tts"] = t.elapsed

        latency["total"] = round(sum(latency.values()), 3)
        lat_str = " · ".join(f"{k.upper()} {v * 1000:.0f}ms" for k, v in latency.items())
        return question, answer, audio_out, f"⏱ {lat_str}"


# ─────────────────────────────────────────────────────────────
# Load everything (runs once at Space startup)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("UBUZIMA AI — starting up")
print(f"Device: {DEVICE}")
print("=" * 60)

asr = SpeechRecognizer()
llm = GeminiAssistant()
tts = MMSFallbackTTS()
pipeline = UbuzimaPipeline(asr=asr, llm=llm, tts=tts)
print("✓ Pipeline ready")


# ─────────────────────────────────────────────────────────────
# Gradio UI (same visual design as the Colab notebook's Cell 8,
# minus the speaker dropdown since MMS-TTS is single-voice)
# ─────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 960px !important; margin: 0 auto !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important; }
.hero-banner { background: linear-gradient(135deg, #00695C 0%, #004D40 60%, #00897B 100%);
    color: white; padding: 32px 24px; border-radius: 20px; text-align: center;
    margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,77,64,0.3); }
.hero-banner h1 { font-size: 2.6em; font-family: Cambria, Georgia, serif; margin: 0; letter-spacing: 1px; }
.hero-banner .subtitle { color: #B2DFDB; font-style: italic; margin: 6px 0 14px; font-size: 1.05em; }
.hero-banner .status-row { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
.pill { display: inline-block; background: rgba(255,255,255,0.15); backdrop-filter: blur(4px);
    padding: 5px 14px; border-radius: 20px; font-size: 0.78em; letter-spacing: 0.3px; }
.section-card { background: #FAFFFE; border: 1px solid #E0F2F1; border-radius: 14px; padding: 20px; margin: 8px 0; }
.example-btn { background: #F1F8F6 !important; border: 1.5px solid #C8E6C9 !important;
    border-radius: 12px !important; padding: 12px !important; text-align: left !important;
    transition: all 0.2s ease !important; font-size: 0.92em !important; }
.example-btn:hover { border-color: #00695C !important; background: #E0F2F1 !important;
    transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,105,92,0.12) !important; }
.result-question { border-left: 4px solid #FFA726 !important; background: #FFF8E1 !important;
    border-radius: 0 10px 10px 0 !important; }
.result-answer { border-left: 4px solid #00695C !important; background: #E0F2F1 !important;
    border-radius: 0 10px 10px 0 !important; }
.app-footer { text-align: center; color: #78909C; font-size: 0.78em; margin-top: 24px;
    padding: 16px; border-top: 1px solid #ECEFF1; }
.app-footer em { color: #B0BEC5; }
"""


def voice_handler(audio, progress=gr.Progress()):
    progress(0.15, desc="🎧 Ndumva ikibazo cyawe...")
    transcript, answer, audio_out, latency = pipeline.run_voice(audio)
    progress(1.0, desc="✓ Byarangiye")
    return transcript, answer, audio_out, latency


def text_handler(question, progress=gr.Progress()):
    progress(0.3, desc="💭 Ndategura igisubizo...")
    _, answer, audio_out, latency = pipeline.run_text(question)
    progress(1.0, desc="✓ Byarangiye")
    return answer, audio_out, latency


with gr.Blocks(css=CSS, title="UBUZIMA AI", theme=gr.themes.Soft()) as app:

    gr.HTML(f"""
        <div class="hero-banner">
            <h1>🩺 UBUZIMA AI</h1>
            <p class="subtitle">Umufasha mu by'ubuzima uvuga Ikinyarwanda</p>
            <div class="status-row">
                <span class="pill">● ASR: akera Whisper</span>
                <span class="pill">● LLM: Gemini (Google)</span>
                <span class="pill">● TTS: MMS-TTS</span>
                <span class="pill">● {'GPU' if DEVICE.type == 'cuda' else 'CPU'}</span>
            </div>
        </div>
    """)

    with gr.Tab("🎙️ Ijwi (Voice)", id="voice"):
        gr.HTML('<div class="section-card">')
        gr.Markdown("### Kanda ku buto ya microphone, uvuge ikibazo cyawe mu Kinyarwanda")

        with gr.Row():
            with gr.Column(scale=1):
                voice_input = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Vuga hano")
                voice_btn = gr.Button("✨ Kora igisubizo", variant="primary", size="lg")

            with gr.Column(scale=2):
                voice_transcript = gr.Textbox(label="📝 Ikibazo cyawe (transcript)", lines=2, elem_classes=["result-question"])
                voice_answer = gr.Textbox(label="💬 Igisubizo", lines=5, elem_classes=["result-answer"])
                voice_audio = gr.Audio(label="🔊 Ijwi ry'igisubizo", autoplay=True)
                voice_latency = gr.Markdown("")

        gr.HTML('</div>')
        voice_btn.click(voice_handler, inputs=[voice_input],
                         outputs=[voice_transcript, voice_answer, voice_audio, voice_latency])

    with gr.Tab("📝 Inyandiko (Text)", id="text"):
        gr.HTML('<div class="section-card">')
        gr.Markdown("### Andika ikibazo cyawe cyangwa uhitemo kimwe mu bibazo bikurikira")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(label="✍️ Ikibazo cyawe",
                                         placeholder="Urugero: Ni iki gikora malariya?", lines=3)
                text_btn = gr.Button("✨ Kora igisubizo", variant="primary", size="lg")

                gr.Markdown("#### 🏥 Ibibazo bishoboka")
                ex_btns = []
                for emoji, question, label in EXAMPLES:
                    btn = gr.Button(f"{emoji}  {question}", elem_classes=["example-btn"])
                    ex_btns.append((btn, question))

            with gr.Column(scale=2):
                text_question_display = gr.Textbox(label="📝 Ikibazo", lines=2, elem_classes=["result-question"], value="")
                text_answer = gr.Textbox(label="💬 Igisubizo", lines=6, elem_classes=["result-answer"])
                text_audio = gr.Audio(label="🔊 Ijwi ry'igisubizo", autoplay=True)
                text_latency = gr.Markdown("")

        gr.HTML('</div>')

        def text_submit(question):
            answer, audio, latency = text_handler(question)
            return question, answer, audio, latency

        text_btn.click(text_submit, inputs=[text_input],
                        outputs=[text_question_display, text_answer, text_audio, text_latency])

        for btn, question in ex_btns:
            def make_handler(q):
                def handler():
                    answer, audio, latency = text_handler(q)
                    return q, q, answer, audio, latency
                return handler
            btn.click(make_handler(question), inputs=[],
                      outputs=[text_input, text_question_display, text_answer, text_audio, text_latency])

    with gr.Tab("ℹ️ About"):
        gr.Markdown(f"""
### URURIMI / UBUZIMA AI

**Capstone project** by Ganza Didier, BSc Software Engineering (Data Science & ML),
African Leadership University, Kigali.

**Supervisor:** Emmanuel Adjei

---

#### This deployment vs. the full research pipeline

This Hugging Face Space runs a **portable subset** of the full pipeline so it can stay
online permanently with no manual setup. The full pipeline — including the 3-voice
`kinya-flex-tts` model — is demoed in the project video and runs from the original
Colab notebook, because it depends on a licensed MorphoKIN binary and a GPU that
can't be bundled into a public Space. See this project's `DEPLOYMENT.md` for details.

#### Architecture (this deployment)

```
🎤 User speaks Kinyarwanda
    → akera Whisper ASR (speech → text)
    → Google Gemini (question → Kinyarwanda answer)
    → Meta MMS-TTS (text → spoken Kinyarwanda)
🔊 User hears the answer
```

#### Tech Stack

| Layer | Technology |
|---|---|
| ASR | `{ASR_MODEL_ID}` |
| LLM | Google Gemini (`{GEMINI_MODEL}`) |
| TTS | `{TTS_FALLBACK}` (single voice) |
| UI | Gradio |
| Compute | Hugging Face Spaces ({'GPU: ' + str(DEVICE) if DEVICE.type == 'cuda' else 'CPU'}) |

#### Credits

- ASR: [akera](https://huggingface.co/akera/whisper-large-v3-kin-200h-v2)
- TTS (full pipeline, demoed in video): [C4IR Rwanda](https://huggingface.co/C4IR-RW) + [KiNLP](https://kinlp.com/)
- TTS (this deployment): [Meta MMS-TTS](https://huggingface.co/facebook/mms-tts-kin)
- LLM: [Google Gemini](https://ai.google.dev/)

---

⚠️ *Iri gikoresho ntabwo risimbura muganga. Iyo ufite ikibazo gikomeye, jya kwa muganga.*
        """)

    gr.HTML("""
        <div class="app-footer">
            <p>URURIMI / UBUZIMA AI · ALU Kigali · Ganza Didier · Supervisor: Emmanuel Adjei</p>
            <p><em>Iri gikoresho ntabwo risimbura muganga. Iyo ufite ikibazo gikomeye, jya kwa muganga.</em></p>
        </div>
    """)

if __name__ == "__main__":
    app.launch()
