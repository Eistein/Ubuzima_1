# Deployment plan

This document covers two things: **why** the deployment is split into two
tracks, and **exactly how** to get a permanent public URL.

## Why not just deploy the full Colab pipeline?

The `share=True` URL that Colab's `app.launch()` produces
(`https://xxxxx.gradio.live`) is a temporary tunnel into your Colab
runtime. It dies as soon as the runtime disconnects — closing the tab,
hitting Colab's idle timeout, or the free-tier 12-hour session cap. That's
expected behavior, not a bug in the notebook; Colab was never meant to be a
permanent host.

The natural next thought is "just run the same notebook code somewhere
that stays on." The obstacle is the **full TTS stage**
(`C4IR-RW/kinya-flex-tts`, the 3-voice model), which needs:

- A **licensed MorphoKIN binary** (~26GB) tied to a license file you upload
  interactively — there's no way to bake a personal license into a public
  repository or container image.
- A background daemon process started with `sudo`.
- An NVIDIA `apex` build from source.
- A CUDA GPU for reasonable latency.

None of that fits into a shareable, always-on public host without either
(a) paying for and maintaining a dedicated GPU server yourself, indefinitely,
or (b) publishing your personal MorphoKIN license inside a public repo,
which you should not do.

So the deployment plan is split into two tracks that serve different
purposes:

| Track | Purpose | What it runs |
|---|---|---|
| **Full pipeline** (Colab notebook) | Demo video — shows the complete, best-quality system you built | ASR + Gemini + `kinya-flex-tts` (3 voices) |
| **Hosted demo** (`deploy/app.py` on Spaces) | Permanent public URL for anyone to try, anytime | ASR + Gemini + MMS-TTS (1 voice) |

This isn't a workaround or a downgrade of the project — it's the same
graceful fallback your own notebook already implements (`try: kinya-flex-tts
except: MMSFallbackTTS`), just applied at the deployment-architecture level
instead of only inside one Python `try/except` block. The README says so
explicitly, which is the more professional choice: silently deploying only
the fallback without explaining why, or silently claiming the hosted demo
has 3 voices when it doesn't, would both be worse than stating the tradeoff
plainly.

---

## How to deploy: Hugging Face Spaces

Hugging Face Spaces gives you a permanent URL
(`https://huggingface.co/spaces/<your-username>/<space-name>`) that stays
up without you keeping a notebook or terminal open. The free CPU tier is
enough for this app's size.

### 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space) (create a free account first if you don't have one).
2. Fill in:
   - **Space name:** e.g. `ubuzima-ai`
   - **License:** your choice (MIT is a common default for a capstone project)
   - **Select the Space SDK:** Gradio
   - **Space hardware:** CPU Basic (free) — this is enough for this app
   - **Visibility:** Public (so the grader/reviewer can open it without an account)
3. Click **Create Space**.

### 2. Add your Gemini API key as a secret

**Do not put your API key in any file you push to the repo.** Instead:

1. In your new Space, go to **Settings → Variables and secrets**.
2. Click **New secret**.
3. Name: `GOOGLE_API_KEY`
4. Value: your Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
5. Save.

The app reads this via `os.environ["GOOGLE_API_KEY"]` — Spaces injects it automatically at runtime.

### 3. Push the files

Every Space is a git repository. From your local machine (with this
project's `deploy/` folder):

```bash
git clone https://huggingface.co/spaces/<your-username>/ubuzima-ai
cd ubuzima-ai
cp /path/to/this/repo/deploy/app.py .
cp /path/to/this/repo/deploy/requirements.txt .
git add app.py requirements.txt
git commit -m "Initial UBUZIMA AI deployment"
git push
```

(If you'd rather not use git locally, the Space's web UI also has an "Add
file" button under the **Files** tab — you can drag and drop `app.py` and
`requirements.txt` there directly.)

### 4. Set the Space's own README frontmatter

Hugging Face Spaces reads its configuration from a YAML block at the top of
a `README.md` **inside the Space repo itself** (this is separate from this
project's main README.md — Spaces needs its own). Create or edit
`README.md` inside the Space repo with:

```yaml
---
title: UBUZIMA AI
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---
```

Commit and push this alongside `app.py`. Without this block, the Space
will show a "Configuration error."

### 5. Wait for the build, then test

- Go to your Space's **App** tab — you'll see build logs while it installs `requirements.txt` and starts `app.py`.
- First build typically takes a few minutes (downloading the Whisper and MMS-TTS model weights).
- Once it shows a green "Running" status, your permanent URL is live at:
  `https://huggingface.co/spaces/<your-username>/ubuzima-ai`
- Test all 6 example questions and the microphone input, the same way you tested the Colab version.

### 6. Add the URL to your README

Copy that URL into the `_[add your Hugging Face Spaces URL here]_`
placeholder in this project's main `README.md`, and into your Canvas
submission.

---

## What to expect from the free tier

- **It's genuinely free** for an app this size — no credit card required for CPU Basic.
- **It sleeps after ~48 hours of no visits**, and takes roughly 15–30 seconds to wake up on the next visit while it reloads the models into memory. This is normal — not a broken deployment. If you're demoing it live (e.g. to your supervisor), open the link a minute or two beforehand so it's already warm.
- **CPU-only inference is slower than the GPU-backed Colab version**, especially for the Whisper ASR step. Expect a few seconds of latency per question rather than the near-instant response you may see on a Colab A100. This is worth mentioning in your analysis/discussion as a known tradeoff, not something to hide.
- If you want GPU speed on the permanent deployment later, Spaces supports upgrading to a paid GPU tier (T4 from roughly $0.40/hour, billed only while the Space is actually running) from **Settings → Hardware**. This isn't required for a working submission — the free CPU tier is a legitimate, complete answer to "get a permanent URL."

---

## Redeploying after changes

Any time you edit `deploy/app.py` or `deploy/requirements.txt` in this
repo, copy the updated file(s) into your local clone of the Space repo and
`git push` again — Spaces automatically rebuilds and redeploys on every
push, the same way GitHub Pages or Vercel would.
