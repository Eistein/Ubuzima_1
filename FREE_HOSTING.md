# Hosting UBUZIMA AI for free

You do not have to pay anything. But the free routes each cost you something other
than money, and one of them costs you an NFR. Pick deliberately.

| Route | Cost | Public URL | GPU | Good for |
|---|---|---|---|---|
| **A. Spaces CPU basic** | £0 forever | permanent | no | NFR5, Figures 4.1–4.6, the link in your report |
| **B. Colab + `share=True`** | £0 | 72 h, changes | T4 | Table 5.8, live defence demo |
| **C. Community GPU grant** | £0 if granted | permanent | T4 | everything, if you get it |
| ~~ZeroGPU~~ | **$9/mo PRO** | — | — | ruled out |

Do **A + B + C together**. They cover different holes.

---

## A. Spaces CPU basic — your permanent public URL

Free tier is 2 vCPU / 16 GB RAM. Both models fit:

```
badrex W2V-BERT 2.0    583.7M params    2.33 GB fp32    0.58 GB int8
YourTTS (VITS + SE)    ~290M  params    1.16 GB fp32    0.29 GB int8
weights total                           3.49 GB
+ activations                          ~2.0  GB
CPU basic RAM                          16.0  GB   -> fits comfortably
```

Same steps as `BUILD.md`, but **skip the hardware upgrade** — stay on CPU basic. Then
add two variables:

| Name | Value |
|---|---|
| `CPU_THREADS` | `2` |

**Do not set `LOW_RESOURCE=1` unless you have measured and the latency is genuinely
unacceptable.** At fp32 on CPU you are running numerically the same model your
notebook evaluated, so Chapter 5's WER describes your deployment exactly. int8 would
force you to re-benchmark and report two sets of numbers, to save a few seconds you
may not need to save.

**The catch, and you must not paper over it:** int8 quantisation changes the model.
The 6.76% / 31.29% WER in Chapter 5 was measured on the fp16 GPU model. If you serve
the quantised model you have to re-run `benchmark.py` and report the quantised WER
*separately*. Quoting the GPU number for a quantised deployment would be exactly the
kind of quiet misdescription your NFR2 promises not to commit. The Telemetry tab
prints `int8 — RE-BENCHMARK BEFORE QUOTING WER` so you cannot forget.

Free Spaces sleep after ~48 h of inactivity and wake when someone visits — a cold
wake is slow but it is not gone. That satisfies NFR5.

---

## B. Colab + Gradio share link — where you measure Table 5.8

Colab gives you a free T4, which is exactly the hardware NFR6 specifies. Run the
real app there and measure on it:

```python
!git clone https://huggingface.co/spaces/<you>/ubuzima-ai && %cd ubuzima-ai
!pip install -q -r requirements.txt
import os
os.environ["ADAPTER_ID"]     = "<you>/ubuzima-w2v-bert-lora"
os.environ["GEMINI_API_KEY"] = "..."      # paste at runtime, never commit
os.environ["SPEAKER_WAV"]    = "voices/nurse/reference.wav"

from app import Pipeline, build_ui
build_ui(Pipeline()).launch(share=True)   # public URL, ~72 h
```

Then in a second cell:

```python
!python benchmark.py --latency 50 --audio-dir ./samples
```

That gives you Table 5.8 on a T4 — the configuration your report actually describes.

Use this same link for the live defence demo. Start it 20 minutes before you present
so the models are warm.

---

## C. Apply for a community GPU grant

Hugging Face gives free GPU upgrades to Spaces they consider worth supporting. You
apply from the Space's hardware settings, lower left, under sleep-time settings.

Apply on day one. An undergraduate capstone doing Kinyarwanda health ASR, with a
published LoRA adapter, real benchmark numbers, and an open Apache-2.0 repo, is
close to the archetype they fund. Worst case they say no and you have lost ten
minutes.

---

## What this does to your report — do not skip this

Your NFR6 currently reads: *"shall run on a single NVIDIA T4 GPU (16 GB VRAM)
allocated by Hugging Face Spaces, at a monthly cost not exceeding 60 USD"*, and §3.4
rejects Render and Vercel for lacking GPU.

If your public deployment runs on free CPU, that sentence no longer describes the
delivered system. You have two honest options, and one dishonest one.

**Honest option 1 — split the claim.** Rewrite NFR6 to something like: the system
shall run on a single T4 (16 GB VRAM) for benchmark measurement, and the public
demonstration deployment shall run on free CPU hardware at zero marginal cost.
Report latency for both in Table 5.8. Two columns, one honest table.

**Honest option 2 — report the miss.** Keep NFR6 as written, deploy on CPU, and
state plainly in §5.3 that the deployed configuration did not meet NFR1's 12-second
median, with the T4 figures given as the reference. A missed NFR with an explanation
costs you almost nothing. Your NFR2 already commits you to reporting WER "honestly
regardless of outcome" — same principle, same paragraph energy.

**The dishonest one** is quoting T4 latency next to a CPU deployment URL and letting
the reader assume they match. Do not.

---

## The part worth saying out loud at your defence

Option 1 is not a compromise. It is a finding, and it is *on thesis*.

Your Chapter 1 argues this matters because Kinyarwanda speakers are excluded from
voice AI by cost and infrastructure. If you can stand up and say "the entire
pipeline — a fine-tuned Kinyarwanda ASR, a frontier LLM, and native TTS — runs at
zero marginal cost on free hardware, at N seconds per turn", that is a stronger
result than "it runs on a GPU I rented".

The affordability claim in §1.6 stops being an argument and becomes a measurement.
Get the number, and say it.
