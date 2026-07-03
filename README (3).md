# 🩺 UBUZIMA AI — Kinyarwanda Voice Health Assistant

A voice-first health-information assistant that lets someone speak or type a
health question **in Kinyarwanda** and get a spoken Kinyarwanda answer back.

Part of the **URURIMI AI** research initiative on Kinyarwanda speech
technology. Capstone project by **Ganza Didier**, BSc Software Engineering
(Data Science & ML), African Leadership University, Kigali.
**Supervisor:** Emmanuel Adjei.

> ⚠️ This tool does not replace a doctor. If you have a serious health
> concern, please see a medical professional.
> *(Iri gikoresho ntabwo risimbura muganga. Iyo ufite ikibazo gikomeye, jya
> kwa muganga.)*

---

## How it works

```
🎤 User speaks or types a question in Kinyarwanda
    → Whisper ASR converts speech → Kinyarwanda text
    → Google Gemini answers the question in Kinyarwanda
    → A text-to-speech model converts the answer → spoken Kinyarwanda
🔊 User hears the answer
```

| Stage | Model | Role |
|---|---|---|
| **Ears** (ASR) | [`akera/whisper-large-v3-kin-200h-v2`](https://huggingface.co/akera/whisper-large-v3-kin-200h-v2) | Speech → Kinyarwanda text |
| **Brain** (LLM) | Google Gemini (`gemini-2.5-flash`) | Question → Kinyarwanda answer |
| **Mouth** (TTS) | [`C4IR-RW/kinya-flex-tts`](https://huggingface.co/C4IR-RW/kinya-flex-tts) (full pipeline) → [`facebook/mms-tts-kin`](https://huggingface.co/facebook/mms-tts-kin) (fallback / hosted demo) | Text → spoken Kinyarwanda |

**A note on the LLM name:** the class in the notebook is currently named
`GeminiAssistant` (an earlier prototype used the Anthropic Claude API and was
named `ClaudeAssistant` — that name is kept as a backward-compatible alias).
The system that actually answers questions is Google Gemini, and every label
in this project (UI, docs, About tab) says so accurately.

---

## Two ways to run this project

This project has **two deployment targets**, because the full pipeline's
best-quality TTS model (`kinya-flex-tts`, 3 voices) depends on a licensed,
26GB toolchain that can only run in an environment you control — it cannot
be bundled into a public, always-on web host. See
[`DEPLOYMENT.md`](DEPLOYMENT.md) for the full explanation.

| | **Full pipeline** (this repo's notebook) | **Hosted demo** (`deploy/`) |
|---|---|---|
| TTS | `kinya-flex-tts`, 3 voices, 24kHz | `mms-tts-kin`, 1 voice, 16kHz |
| Where it runs | Google Colab (GPU) | Hugging Face Spaces (free, permanent URL) |
| Setup needed | MorphoKIN license file, ~26GB download | None — just open the link |
| Used for | The demo video (see below) | The always-on public link for graders/reviewers |

**🔗 Live permanent demo:** _[add your Hugging Face Spaces URL here after
deploying — see DEPLOYMENT.md]_

**🎥 Demo video:** _[add your video link here]_

---

## Repository contents

```
.
├── README.md                      ← you are here
├── DEPLOYMENT.md                  ← how to get a permanent URL (Spaces walkthrough)
├── ANALYSIS.md                    ← results, discussion, recommendations (draft — finalize with supervisor)
├── ubuzima_ai_platform.ipynb      ← full pipeline notebook (Colab, GPU, 3-voice TTS)
├── deploy/
│   ├── app.py                     ← portable Gradio app for Hugging Face Spaces
│   └── requirements.txt           ← dependencies for the hosted deployment
└── .gitignore
```

---

## Running the full pipeline (`ubuzima_ai_platform.ipynb`)

### Prerequisites
- A Google account with access to [Google Colab](https://colab.research.google.com/)
- A Colab GPU runtime (Runtime → Change runtime type → GPU; T4 is the free-tier option, A100 recommended if available)
- A **Google Gemini API key** — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- A **MorphoKIN license file** (`.dat`) — required only if you want the full
  3-voice `kinya-flex-tts` output. Without it, the notebook automatically
  falls back to MMS-TTS (still fully functional, single voice).

### Step by step
1. Open `ubuzima_ai_platform.ipynb` in Google Colab (File → Upload notebook, or open directly from GitHub).
2. Set the runtime to GPU (Runtime → Change runtime type → T4 GPU or better).
3. Add your Gemini key as a Colab secret:
   - Click the 🔑 key icon in the left sidebar.
   - Add a new secret named `GOOGLE_API_KEY` with your key as the value.
   - Toggle "Notebook access" on for this notebook.
4. Run **Cell 1** (system dependencies), then **restart the runtime** when prompted (Runtime → Restart session). This step only needs to happen once per Colab session.
5. Run every cell from the top in order. When you reach the MorphoKIN license-upload cell, either:
   - Upload your `.dat` license file when prompted (enables 3-voice TTS), **or**
   - Skip that cell — the pipeline will automatically use the MMS-TTS fallback (1 voice) instead.
6. The final cell launches the Gradio app with `share=True`, which prints a public `https://xxxxx.gradio.live` URL. This URL is **temporary** — it stops working when the Colab runtime disconnects or the notebook is closed. For a permanent URL, see [`DEPLOYMENT.md`](DEPLOYMENT.md).
7. Test the app: try all 6 example questions on the Text tab, then test the Voice tab with your microphone.

---

## Running the hosted demo locally (`deploy/app.py`)

This is the same portable app that's deployed to Hugging Face Spaces — useful if you want to test or modify it before pushing.

```bash
cd deploy
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-api-key"
python app.py
```

This opens a local Gradio server (default `http://127.0.0.1:7860`) running ASR + Gemini + MMS-TTS — no MorphoKIN, no license file, no `sudo` required. A GPU speeds up ASR but is not required.

---

## Getting a permanent URL instead of a `.gradio.live` link

Short answer: **deploy `deploy/app.py` to Hugging Face Spaces.** Full walkthrough, cost breakdown, and the reasoning behind this choice are in [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## Known limitations

- **LLM answers are general information, not medical diagnosis.** The system prompt explicitly constrains Gemini to short, general Kinyarwanda health guidance.
- **Kinyarwanda ASR accuracy** depends on the `akera` checkpoint's training data (200h); background noise and unclear speech will degrade transcription quality.
- **The hosted demo uses single-voice TTS** (MMS-TTS) rather than the 3-voice `kinya-flex-tts` model used in the full pipeline, for the portability reasons explained above.
- **Free-tier Hugging Face Spaces sleep after ~48 hours of inactivity** and take a short moment to wake up on the next visit. See `DEPLOYMENT.md` for what this means in practice.

---

## Credits

- ASR: [akera/whisper-large-v3-kin-200h-v2](https://huggingface.co/akera/whisper-large-v3-kin-200h-v2)
- TTS (full pipeline): [C4IR Rwanda](https://huggingface.co/C4IR-RW) + [KiNLP](https://kinlp.com/)
- TTS (hosted demo): [Meta MMS-TTS](https://huggingface.co/facebook/mms-tts-kin)
- LLM: [Google Gemini](https://ai.google.dev/)
- UI: [Gradio](https://gradio.app/)
