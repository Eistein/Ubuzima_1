---
title: UBUZIMA AI
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Kinyarwanda voice health assistant — ASR + LLM + TTS
---

# UBUZIMA AI

An end-to-end Kinyarwanda voice health assistant: speak a health question in
Kinyarwanda, get a spoken Kinyarwanda answer. Built as a capstone research project
at the African Leadership University, Kigali.

**This is a research prototype, not a medical device. It does not diagnose.**

| Stage | Component |
|---|---|
| ASR | `badrex/w2v-bert-2.0-kinyarwanda-asr` + UBUZIMA LoRA adapter (r=32, α=64) |
| Safety | Kinyarwanda system prompt; diagnostic and medication requests bypass the LLM |
| LLM | Google Gemini 2.5 Pro, falling back to 2.5 Flash on rate limit |
| TTS | DigitalUmuganda/Kinyarwanda_YourTTS_v1 (YourTTS, speaker-conditioned) |

---

## Setup

### 1. Publish your adapter

The Space cannot read Google Drive. Push the adapter to its own model repo first:

```bash
python scripts/push_adapter.py \
  --adapter-dir /content/drive/MyDrive/ururimi/badrex_lora_combined_100h/final \
  --repo-id     <your-username>/ubuzima-w2v-bert-lora
```

**If you skip this, the Space silently serves the base model and the reported WER
figures describe something the visitor is not using.** `asr_service.py` logs an
error in that case; the Telemetry tab shows which model is actually loaded.

### 2. Create the Space

Hugging Face → New Space → Gradio → **Hardware: CPU basic (free) is enough.**

A GPU is a speed optimisation here, not a requirement. Both local models are
non-autoregressive single-pass networks and the weights total ~3.5 GB against the
free tier's 16 GB. See FREE_HOSTING.md.

```bash
git clone https://huggingface.co/spaces/<your-username>/ubuzima-ai
cd ubuzima-ai
cp -r /path/to/these/files/* .
git add . && git commit -m "UBUZIMA AI" && git push
```

### 3. Set secrets

Space → Settings → Variables and secrets:

| Name | Kind | Value |
|---|---|---|
| `GEMINI_API_KEY` | **Secret** | your Google AI Studio key |
| `ADAPTER_ID` | Variable | `<your-username>/ubuzima-w2v-bert-lora` |
| `BASE_MODEL_ID` | Variable | `badrex/w2v-bert-2.0-kinyarwanda-asr` |
| `GEMINI_MODEL` | Variable | `gemini-2.5-pro` |
| `GEMINI_FALLBACK_MODEL` | Variable | `gemini-2.5-flash` |

`GEMINI_API_KEY` must be a **Secret**, never a Variable — Variables are visible to
anyone who can see the Space. If a key has ever been pushed to git, revoke and
regenerate it; rotating is cheap, a leaked key is not.

### 4. Record a conditioning voice (optional but this is the big one)

The TTS clones whatever voice you give it. Record ~1 minute of clean speech from a
real Kinyarwanda speaker (with their consent), upload it to the Space, and set
`SPEAKER_WAV` to its path. Without it the model falls back to the repo's default
conditioning audio, which is where most of the "robotic" quality comes from.

The residual flatness is tonal: Kinyarwanda marks tone in speech but not in
orthography, and this model was trained on graphemes with `use_phonemes: False`.
See the header of `tts_service.py`.

---

## Layout

```
app.py             Gradio UI + Config, Telemetry, Pipeline classes  (Figure 3.6)
errors.py          Bilingual error catalogue (Kinyarwanda + English)
guards.py          Audio, transcript and language guard chain
voice_clone.py     Create and audition a new TTS voice (CLI)
ubuzima_voice_clone.ipynb   Colab notebook — record and clone your voice, start here
BUILD.md           Step-by-step build guide — start here
FREE_HOSTING.md    Zero-cost hosting routes and what they cost you instead
asr_service.py     badrex W2V-BERT 2.0 + LoRA adapter               (Section 4.1.3)
safety_prompt.py   Kinyarwanda safety prompt + refusal responses    (Section 4.1.4)
llm_service.py     Gemini 2.5 Flash                                 (Section 4.1.5)
tts_service.py     Kinyarwanda synthesis                            (Section 4.1.6)
benchmark.py       WER/CER/RTF and end-to-end latency               (Chapter 5)
tone_probe.py      Measure the tonal gap — see TONE.md
scripts/           push_adapter.py
docs/              architecture and sequence diagrams, deployment guide
```

## Reproducing the benchmarks

```bash
# ASR accuracy — needs manifests/ exported from the training notebook
python benchmark.py --split test --domain both --adapter <your-adapter-id>

# End-to-end latency — Table 5.8
python benchmark.py --latency 50 --audio-dir ./samples
```

Put 10–15 second Kinyarwanda health questions in `samples/` before the latency run.

## Privacy

Recordings are transcribed on the Space GPU and not retained. The **transcribed
text** is sent to Google's Gemini API and leaves Rwanda. No identifiers are
collected. Voice is biometric data under Rwanda's Law N° 058/2021; this prototype
has no NCSA authorisation and is for academic evaluation only. Full notice on the
app's Privacy & Terms tab.

## Credits

Ganza Didier · BSc Software Engineering (Data Science & Applied ML), African
Leadership University · Supervisor: Emmanuel Adjei · MasterCard Foundation Scholar.

Built on badrex Labs' Kinyarwanda ASR, Digital Umuganda's Afrivoice, Mozilla Common
Voice, and the DeepKIN / KiNLP Kinyarwanda TTS work of Dr Antoine Nzeyimana's group.

Apache 2.0. Model weights follow their own upstream licences.
