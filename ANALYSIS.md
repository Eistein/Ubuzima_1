# Analysis, Discussion & Recommendations

**Status: draft.** The rubric asks for these sections to be developed "with
the supervisor" — this document is a starting point grounded in what the
notebook and code actually do, not a finished submission. Sections marked
🔲 need input, numbers, or a decision from you and Emmanuel Adjei before
this goes in the final submission. Everything else here is a factual
description of what was built and verified by reading the code directly.

---

## 1. Analysis of results

### 1.1 What was proposed vs. what was built

🔲 *Paste the relevant scope/objectives from your original project proposal
here, then fill in the comparison table below with your supervisor.*

| Proposed objective | Status | Notes |
|---|---|---|
| Kinyarwanda speech-to-text | ✅ Built | `akera/whisper-large-v3-kin-200h-v2` via `transformers`, wrapped in a `SpeechRecognizer` class |
| Kinyarwanda-capable health Q&A | ✅ Built | Google Gemini (`gemini-2.5-flash`), constrained by a Kinyarwanda-only system prompt limited to 2–3 short sentences |
| Kinyarwanda text-to-speech | ✅ Built, with fallback | `kinya-flex-tts` (3 voices) with automatic fallback to `mms-tts-kin` (1 voice) if the licensed toolchain is unavailable |
| End-to-end voice pipeline | ✅ Built | `UbuzimaPipeline` orchestrates ASR → LLM → TTS with per-stage latency tracking |
| Web-accessible interface | ✅ Built | Gradio app with Voice and Text tabs, 6 example health questions, speaker selection |
| Permanent public deployment | ✅ Built | Hugging Face Spaces deployment (see `DEPLOYMENT.md`) |
| 🔲 *(any other objectives from your proposal)* | | |

### 1.2 A correction made during this round of work

One thing worth stating plainly in an honest analysis: the notebook's
documentation, UI labels, and a class name (`ClaudeAssistant`) originally
described the LLM backend as **Anthropic Claude**, while the actual running
code called **Google Gemini**. This has been corrected throughout — code,
labels, and docs now consistently describe Gemini, which is what the system
actually runs. Worth naming in your report as evidence of the kind of
verification the project underwent, rather than glossing over it.

🔲 *If your original proposal specifically named Claude as the LLM, decide
with your supervisor whether to (a) document Gemini as the implementation
decision that was actually made, with a brief rationale, or (b) treat
swapping in the real Anthropic API as a scope item before final submission.
This project intentionally did not decide this for you.*

### 1.3 Testing performed

🔲 *This section needs your actual testing evidence — the rubric weights
this at 4 points, the single largest line item, and wants "multiple
relevant testing strategies... across different inputs, edge cases, and
performance across hardware/software environments." Suggested structure:*

- **Functional testing** — the 6 built-in example questions (malaria,
  child fever, respiratory illness, hydration, headache, pregnancy
  nutrition), tested via both the Text tab and Voice tab (microphone
  input). 🔲 *Add screenshots and your observed transcript/answer/audio
  output for each.*
- **Edge cases** 🔲 *— e.g. silence/no speech, background noise, code-switched
  input (Kinyarwanda + English/French mixed), very short or very long
  questions, questions outside the health domain. Document what happened
  for each.*
- **Cross-environment testing** — 🔲 *run the same questions on (a) the
  Colab GPU pipeline and (b) the Hugging Face Spaces CPU deployment, and
  record the latency difference. The pipeline already reports per-stage
  latency (ASR/LLM/TTS/total) in the UI — screenshot those numbers from
  both environments as your hardware/software comparison.*
- **Voice comparison** 🔲 *— if you have access to a MorphoKIN license,
  compare `kinya-flex-tts` (3 voices) output quality/naturalness against
  the MMS-TTS fallback used in the hosted demo, with your own subjective
  or listener-based assessment.*

---

## 2. Discussion

### 2.1 Architectural decisions and why they matter

**Fallback-first design.** The TTS module was already written to
gracefully degrade — try `kinya-flex-tts`, fall back to `MMS-TTS` if
unavailable — rather than hard-failing. That same pattern was extended to
the deployment layer: rather than one deployment target that's either
fully faithful to the research (and unhostable publicly) or fully portable
(and a smaller demo), the project ships both, each honest about what it is.
🔲 *Discuss with your supervisor whether this fallback-oriented design
pattern is worth calling out as a deliberate engineering choice in your
report — it demonstrates thinking about production robustness, not just
getting a demo working once.*

**Kinyarwanda as a low-resource language in Whisper.** The ASR model sets
`language="swahili"` as the closest Bantu language in Whisper's built-in
tokenizer vocabulary, since Kinyarwanda isn't one of Whisper's native
language tokens — the actual transcription capability comes from the
`akera` fine-tune's training data (200 hours), not from Whisper's own
built-in Kinyarwanda support (there isn't any). 🔲 *This is worth a
sentence or two in your discussion as a concrete illustration of the
low-resource-language challenges your broader URURIMI AI research is
addressing.*

**Why Gemini rather than Anthropic Claude for this specific module.**
🔲 *Discuss and document your actual reasoning here* — e.g. cost (Gemini
has a usable free tier for a student project), Kinyarwanda output quality
in practice, availability, or simply what was easiest to get working first.
An honest, specific answer here is more valuable to a reader than a vague
one.

### 2.2 Impact

🔲 *This is where the rubric wants you to connect the milestone back to
real-world relevance — e.g. Kinyarwanda speakers without reliable internet
access to health information in their own language, rural health literacy,
reducing reliance on translated (and sometimes inaccurate) health content.
Ground this in whatever motivated the original URURIMI AI / UBUZIMA AI
proposal, and discuss with Emmanuel Adjei what's appropriate to claim here
— avoid overstating clinical impact, since this is explicitly not a
diagnostic tool (see the disclaimer in the app itself).*

---

## 3. Recommendations

### 3.1 To the community / future work

- **Expand the ASR training data.** The `akera` checkpoint is trained on
  200 hours; broader Kinyarwanda ASR coverage (accents, dialects,
  background-noise robustness) would directly improve transcription
  quality, which is the first and most error-compounding stage of the
  pipeline.
- **A Kinyarwanda-native evaluation of the LLM stage.** 🔲 *Consider
  whether a systematic accuracy/appropriateness review of Gemini's
  Kinyarwanda health answers (ideally by a Kinyarwanda-speaking clinician
  or health-literacy expert) is worth recommending as a next step before
  any real-world deployment beyond a capstone demo.*
- **Wider TTS voice licensing.** The best-quality TTS (`kinya-flex-tts`)
  is gated behind a license that makes public deployment impossible. A
  recommendation worth raising to the community/maintainers: an openly
  licensed, or at least redistribution-friendly, high-quality Kinyarwanda
  TTS model would remove this entire class of deployment constraint for
  future projects, not just this one.
- 🔲 *Add any other recommendations you and your supervisor think are
  relevant — e.g. offline/low-bandwidth deployment for areas with poor
  connectivity, integration with existing community health worker
  workflows, multi-turn conversation support (currently single-turn Q&A).*

### 3.2 To future maintainers of this specific project

- Rename the `ClaudeAssistant` alias away entirely once nothing external
  depends on the old name, to avoid any future confusion about which LLM
  is actually in use.
- Consider adding automated tests for the pipeline's fallback logic (ASR
  failure, TTS failure) rather than relying only on manual testing.
- If GPU-tier hosting becomes available/affordable, upgrading the Spaces
  deployment's hardware tier would close the latency gap with the Colab
  version without changing any application code.

---

## 4. Sign-off

🔲 *Once finalized with your supervisor, note the date and any changes made
from this draft here, for your own record and for the technical report.*
