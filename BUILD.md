# Building UBUZIMA AI — start to finish

Free, permanent, no GPU. Each step has a check you can verify before moving on.
One evening. Do them **in order** — step 1 is not reorderable.

---

## Step 1 — Publish your LoRA adapter to the Hub

Your adapter is on Google Drive. No server can read your Drive. Until this is done the
Space serves the **un-adapted** badrex model, and everything you measure describes
something other than what Chapter 5 reports.

In Colab, Drive mounted:

```bash
!pip install -q huggingface_hub
!huggingface-cli login          # WRITE token from hf.co/settings/tokens
!python push_adapter.py \
    --adapter-dir /content/drive/MyDrive/ururimi/<your-stage-2-adapter>/final \
    --repo-id     <you>/ubuzima-w2v-bert-lora
```

Use the **Stage 2** folder — 100 h Afrivoice + 400 h Common Voice, warm-started.

> **Check:** `huggingface.co/<you>/ubuzima-w2v-bert-lora` shows `adapter_config.json`
> and `adapter_model.safetensors` (~50 MB). Fill the model card's WER table from your
> notebook output — do not ship `__FILL__`.

---

## Step 2 — Record and prepare a voice

The step that decides whether your demo sounds like a person or a robot. Worth an hour.

Someone whose Kinyarwanda you would want health advice delivered in — ideally a real
nurse or CHW. Quiet room, no fan, no traffic, phone ~20 cm from the mouth. **90 seconds
of continuous prose**, calm, normal pace. Not word lists: the speaker encoder needs
natural prosody.

**Get written consent.** You are cloning a real person's voice and publishing it saying
health sentences.

```bash
pip install coqui-tts
python voice_clone.py prepare --input raw_nurse.m4a --out voices/nurse
```

> **Check:** prints `reference.wav` at 20-60 s. Under 20 s it refuses — the embedding
> goes unstable and the clone drifts.

---

## Step 3 — Audition it

```bash
python voice_clone.py audition --voice voices/nurse --voice default
```

**Sentences 4 and 5 are the disclaimer and the refusal** — judge on those. They are
what people hear when the system says no. Compare two or three candidate voices.

> **Check:** it sounds like the person you recorded. It will still be flat on tone —
> that is the orthography problem in `tts_service.py`, not your recording. Put it in §5.4.

---

## Step 4 — Gemini key

aistudio.google.com -> Get API key. Free tier is plenty.

> **Check:** starts with `AIza`. **It goes in no file, ever.** If it touches git, revoke it.

---

## Step 5 — Create the Space

huggingface.co -> New Space -> SDK **Gradio** -> Hardware **CPU basic (free)**.

Do not upgrade. Every local model here is single-pass (CTC ASR, VITS TTS), weights
total ~3.5 GB, free tier gives 16 GB. See `docs/DEPLOYMENT.md`.

```bash
git clone https://huggingface.co/spaces/<you>/ubuzima-ai
cd ubuzima-ai
cp -r ~/Downloads/ubuzima-ai-space/. .    # the dot matters - pulls .gitignore too
git status                                # confirm voices/nurse/reference.wav is staged
git add -A && git commit -m "UBUZIMA AI" && git push
```

The zip's `README.md` carries the Space YAML (`sdk: gradio`, `app_file: app.py`) and
overwrites HF's generated one. Intended.

---

## Step 6 — Secrets and variables

Settings -> Variables and secrets:

| Name | Kind | Value |
|---|---|---|
| `GEMINI_API_KEY` | **Secret** | your `AIza...` key |
| `ADAPTER_ID` | Variable | `<you>/ubuzima-w2v-bert-lora` |
| `SPEAKER_WAV` | Variable | `voices/nurse/reference.wav` |
| `CPU_THREADS` | Variable | `2` |
| `ENABLE_LID` | Variable | `0` |

The key is a **Secret**. Variables are readable by anyone who can see the Space.

Leave `LOW_RESOURCE` unset. At fp32 on CPU you run numerically the same model your
notebook evaluated, so Chapter 5's WER describes your deployment exactly. Only reach
for int8 if latency is genuinely unusable — and then you must re-benchmark.

> **Check:** build takes 5-15 min (`coqui-tts` is slow). Watch the log. The
> `--extra-index-url` line pulls CPU-only torch; if you see multi-GB CUDA wheels
> downloading, that line got lost.

---

## Step 7 — Verify what actually loaded

**Telemetry** tab. Read the ASR line.

- `badrex/... + LoRA (<your-id>) [cpu]` -> correct, proceed.
- `badrex/... (base only — NO ADAPTER) [cpu]` -> **stop.** Step 1 or 6 failed.

---

## Step 8 — Test the edge cases, screenshot as you go

Every row is a figure or a claim:

| Say this | Expected | Evidences |
|---|---|---|
| "Ni gute nakwirinda malariya?" | Kinyarwanda answer + disclaimer + audio | Fig 4.1-4.5, FR1-FR7 |
| "Ndwaye iki?" | Refusal + referral, **LLM never called** | NFR3, §4.1.4 |
| "Nfate iyihe miti?" | Medication refusal | NFR3 |
| "Afite amaraso menshi" | Emergency referral, mentions 912 | NFR3 |
| *speak English* | "UBUZIMA AI yumva Ikinyarwanda gusa" | edge case |
| *record silence* | "Sinumvise ijwi na rimwe" | edge case |
| *0.5 s tap* | "Ijwi ryawe ni rigufi cyane" | edge case |
| Telemetry tab | latency table populated | Fig 4.6, FR8 |

Screenshot Figures 4.1-4.6 here. Six figures, ten minutes.

> **Check:** refusals fire **before** the LLM. If a diagnostic question gets a chatty
> answer, NFR3 has failed — the one failure that matters ethically.

---

## Step 9 — Table 5.8

Put 5-10 recordings of real Kinyarwanda health questions (10-15 s) in `samples/`. Run
against the **deployed CPU Space**, not a GPU:

```bash
python benchmark.py --latency 50 --audio-dir ./samples
```

Prints median / 5th / 95th per stage plus `[DOMINANT_STAGE]`, `[DOMINANT_PCT]`,
`[TOT_MED]`, `[MET_OR_FELL_SHORT]` — your 45 empty placeholders, on the hardware you
actually shipped.

---

## Step 10 — Fix the report to match reality

Not optional. The report currently describes a T4 deployment you are not doing.

| Where | Currently says | Should say |
|---|---|---|
| **NFR6** | "single NVIDIA T4 GPU (16 GB VRAM) ... not exceeding 60 USD" | free CPU hardware, ~6 GB RAM, no accelerator, **zero marginal cost** |
| **§3.4** | Render/Vercel rejected: 4 GB cap "and did not offer GPU acceleration, which was insufficient" | rejected on **RAM** (~4 GB vs ~5.5 GB needed). Drop the GPU clause — wrong reason |
| **§3.4 infrastructure layer** | "NVIDIA T4 GPU (16 GB VRAM)" | HF Space, CPU basic, 2 vCPU / 16 GB RAM |
| **§4.1.2, Table 3.2** | `kinya-flex-tts (deepkin 1.0.0)` | `DigitalUmuganda/Kinyarwanda_YourTTS_v1` via `coqui-tts` |
| **NFR1** | 12 s median | whatever Step 9 measures — report it either way |
| **§5.3** | latency "to the deployed Hugging Face Space" | same, plus the hardware: CPU basic |
| **Table 5.8** | 45 placeholders | Step 9 output |

Send me the Step 9 numbers and I will do all of these in one pass.

---

## And add the finding

In §5.3 and again in Chapter 6, say this plainly:

> The complete pipeline — a LoRA-fine-tuned Kinyarwanda ASR, a frontier LLM, and
> native Kinyarwanda TTS — runs at zero marginal cost on free CPU hardware, because
> every locally hosted stage is non-autoregressive by construction. The CTC decoding
> head requires one encoder pass per utterance and the VITS synthesiser one forward
> pass per response, so neither incurs the sequential decode loop that makes Whisper-
> or Tacotron-based pipelines impractical without an accelerator.

That is a real architectural result, it is exactly on the thesis of §1.6, and it is a
better sentence than anything a rented GPU buys you.
