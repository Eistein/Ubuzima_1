# UBUZIMA AI

An End-to-End Kinyarwanda Voice Health Assistant Integrating Automatic Speech Recognition, a Large Language Model, and Native Rwandan Text-to-Speech Synthesis

**Author:** Ganza Didier  
**Supervisor:** Mr. Emmanuel Adjei  
**Institution:** African Leadership University, Kigali, Rwanda  
**Date:** July 2026

**Live Demo:** [https://ubuzima1-production.up.railway.app](https://ubuzima1-production.up.railway.app)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Environment Setup](#environment-setup)
5. [Running the Application](#running-the-application)
6. [Reproducing the Safety Evaluation](#reproducing-the-safety-evaluation)
7. [Deployment](#deployment)
8. [Project Structure](#project-structure)
9. [Safety and Ethics](#safety-and-ethics)
10. [Limitations](#limitations)
11. [License](#license)
12. [Citation](#citation)
13. [Acknowledgements](#acknowledgements)

---

## Project Overview

UBUZIMA AI is a voice-first health information assistant for Kinyarwanda speakers. It allows users to ask health questions in spoken Kinyarwanda and receive spoken answers in their native language. The system is designed as a proof-of-concept for community health worker (CHW) support in Rwanda, where 54% of adults are literate in Kinyarwanda only.

### Key Features

- **Kinyarwanda Speech Recognition:** Fine-tuned badrex Wav2Vec2-BERT 2.0 with LoRA adaptation (health-domain WER: 6.76%)
- **Health-Safe LLM Reasoning:** Google Gemini 2.5 Flash (via OpenRouter) with Kinyarwanda safety prompt
- **Native Voice Synthesis:** Meta MMS-TTS (facebook/mms-tts-kin) for natural Kinyarwanda speech
- **Real-time Processing:** End-to-end latency of 5–11 seconds on modest hardware
- **Safety First:** Explicitly refuses diagnosis, prescription, and dosage requests
- **Consent Gate:** Affirmative consent required before audio processing
- **Confidence Gating:** Three-band confidence scoring (high/medium/low) with automatic refusal on low confidence

### Performance Metrics

| Metric | Value |
|--------|-------|
| Health-domain WER (Afrivoice) | 6.76% (95% CI: 6.44–7.11%) |
| General-domain WER (Common Voice) | 31.29% (95% CI: 30.86–31.72%) |
| Health-term Recall (in-domain) | 97.64% |
| Real-time Factor | < 0.03 |
| Safety Compliance (adversarial prompts) | 20/20 (100%) |
| End-to-end Latency | ~5–11 seconds |

---

## System Architecture

UBUZIMA AI follows a four-layer architecture:

1. **Presentation Layer:** Gradio web interface with microphone input and audio playback
2. **Orchestration Layer:** Python pipeline coordinator with safety prompts, confidence gating, and telemetry
3. **Model Services Layer:** ASR (badrex + LoRA), LLM (Gemini 2.5 Flash via OpenRouter), TTS (Meta MMS-TTS)
4. **Infrastructure Layer:** Railway deployment with CPU-only inference

```
User → Gradio UI → Orchestrator → ASR → Confidence Gate → Safety Filter → LLM → TTS → Audio Player
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip < 24.1 (required for fairseq compatibility)
- Git
- 8 GB RAM minimum (16 GB recommended)
- CUDA-capable GPU optional (for faster ASR inference)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Eistein/Ubuzima_1.git
cd Ubuzima_1
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

**IMPORTANT:** Install in this exact order to resolve dependency conflicts.

```bash
pip install "pip<24.1"
pip install "omegaconf==2.0.6" "hydra-core==1.0.7"
pip install -r requirements.txt
```

### Step 4: Download Model Weights

The ASR and TTS models are downloaded automatically on first run from Hugging Face. Ensure you have:

- Stable internet connection for initial model download (~3.5 GB total)
- The LoRA adapter weights are bundled in the `./adapter/` directory

---

## Environment Setup

Create a `.env` file in the project root (see `.env.example` for template):

```bash
# Required: OpenRouter API Key (for Gemini 2.5 Flash access)
# Obtain from: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-...

# Optional: Hugging Face Token (only needed if models are gated)
# Obtain from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_...

# Optional: Contact email shown in Terms & Privacy tab
CONTACT_EMAIL=your@email.com

# Optional: Confidence thresholds (defaults shown)
HIGH_CONF=0.90
LOW_CONF=0.75
```

**SECURITY WARNING:** Never commit API keys to version control. The `.env` file is listed in `.gitignore`.

---

## Running the Application

### Local Development

```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860` with a public shareable URL.

### Using the Pipeline

1. Open the browser interface
2. Check the consent box (required before audio processing)
3. Click the microphone and speak your Kinyarwanda health question
4. Click "Tanga igisubizo" (Get answer)
5. Review the transcription, read the response, and listen to the spoken answer

---

## Reproducing the Safety Evaluation

The safety evaluation validates that the system refuses diagnostic and prescription requests.

```bash
python eval/safety_eval.py
```

This runs 20 adversarial prompts across five categories:
- Direct diagnosis requests
- Medication dosage requests
- Prescription requests
- Emergency triage bypass attempts
- Authority-framing attacks

Expected output: 20/20 compliant (100% refusal rate).

---

## Deployment

### Railway (Recommended)

1. Fork this repository
2. Create account at [railway.app](https://railway.app)
3. Create new project from GitHub repository
4. Add environment variables in Railway dashboard:
   - `OPENROUTER_API_KEY`
   - `HF_TOKEN` (optional)
   - `CONTACT_EMAIL`
5. Deploy — Railway will automatically build from the Dockerfile

**Resource Requirements:**
- Minimum: 8 GB RAM, 2 vCPU
- Recommended: 16 GB RAM, 4 vCPU
- GPU not required

### Alternative Platforms

| Platform | Status | Reason |
|----------|--------|--------|
| Hugging Face Spaces | Supported | Requires GPU upgrade for model loading |
| Render | Not supported | RAM capped at 4 GB (insufficient) |
| Vercel | Not supported | RAM capped at 4 GB, no GPU |

**Why Railway was selected:** Direct control over runtime environment, ability to expose a permanent custom URL, and sufficient memory (8+ GB) to hold both ASR (~2.4 GB) and TTS (~1.1 GB) models simultaneously.

---

## Project Structure

```
Ubuzima_1/
├── app.py                    # Gradio interface and pipeline orchestrator
├── safety_prompt.py          # Kinyarwanda health-safety system prompt
├── asr_confidence.py         # ASR confidence scoring and gating
├── requirements.txt          # Pinned dependencies with explanatory comments
├── Dockerfile                # Container configuration for Railway
├── .env.example              # Environment variable template
├── .gitignore                # Excludes secrets and cache
├── README.md                 # This file
├── LICENSE                   # MIT License
├── adapter/                  # LoRA adapter weights (bundled)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── eval/                     # Evaluation scripts
│   └── safety_eval.py        # Safety refusal validation (20 adversarial prompts)
└── tests/                    # Unit tests
    ├── test_safety_prompt.py
    └── test_asr_confidence.py
```

---

## Safety and Ethics

### Health Safety

UBUZIMA AI is **NOT a medical device**. It is a research prototype for general health information only.

- ✓ Provides prevention, hygiene, nutrition, and vaccination information
- ✓ Refuses diagnostic requests
- ✓ Refuses prescription and dosage requests
- ✓ Refers users to qualified health workers and facilities
- ✗ Does **NOT** replace professional medical advice

### Data Privacy

- No audio recordings or transcripts are persisted
- Only anonymized latency metadata is logged
- Compliant with Rwanda Law No. 058/2021 on personal data protection
- All processing occurs in real-time; no data retention
- Affirmative consent required before any audio processing

### Usage Disclaimer

> **Research Demo Only. Not a medical diagnosis tool. Always consult a qualified healthcare professional for medical concerns.**

---

## Limitations

1. **ASR Cross-Domain Gap:** General-domain WER (31.29%) is significantly higher than health-domain WER (6.76%)
2. **TTS Tone:** Does not reproduce Kinyarwanda lexical tone (data representation limitation)
3. **Dialect Coverage:** Limited to standard Kinyarwanda; regional dialects not well represented
4. **Clinical Safety:** No formal evaluation with medical professionals; prompt-based guardrails are not foolproof
5. **Concurrent Users:** Not tested under load; single-user performance only characterized
6. **Scope:** Three of four proposed LoRA conditions were descoped due to time constraints

For detailed discussion, see Chapter 5 (Section 5.6) and Chapter 6 (Section 6.6) of the capstone report.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

Third-party components:
- badrex/w2v-bert-2.0-kinyarwanda-asr: CC-BY-4.0
- Meta MMS-TTS: Permissive research license
- Google Gemini API (via OpenRouter): Subject to respective terms of service

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{ganza2026ubuzima,
  title={UBUZIMA AI: An End-to-End Kinyarwanda Voice Health Assistant},
  author={Ganza, Didier},
  year={2026},
  institution={African Leadership University},
  address={Kigali, Rwanda},
  type={Bachelor's Capstone Project}
}
```

---

## Acknowledgements

- **Supervisor:** Mr. Emmanuel Adjei (African Leadership University)
- **Funding:** MasterCard Foundation Scholars Program at ALU
- **ASR Model:** badrex Labs (badrex/w2v-bert-2.0-kinyarwanda-asr)
- **TTS Model:** Meta AI (MMS-TTS)
- **Datasets:** Mozilla Common Voice, Digital Umuganda Afrivoice
- **Research Community:** KiNLP group at UMass Amherst, Sunbird AI, C4IR Rwanda

---

**Last Updated:** August 2026
