#!/usr/bin/env python3
"""push_adapter.py — publish your trained LoRA adapter to the Hugging Face Hub.

The Space cannot read your Google Drive. Until the adapter lives on the Hub as its
own model repo, the Space serves the UN-ADAPTED badrex model — which would make
every WER number in Chapter 5 a description of something other than what people
can actually try.

Run this once, from Colab, with the Drive still mounted:

    !python push_adapter.py \
        --adapter-dir /content/drive/MyDrive/ururimi/badrex_lora_combined_100h/final \
        --repo-id     <your-hf-username>/ubuzima-w2v-bert-lora

Then set ADAPTER_ID to that repo id in your Space secrets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

CARD = """---
license: apache-2.0
language: rw
library_name: peft
base_model: badrex/w2v-bert-2.0-kinyarwanda-asr
tags: [automatic-speech-recognition, kinyarwanda, lora, health]
---

# UBUZIMA AI — Kinyarwanda ASR LoRA adapter

A LoRA adapter over [badrex/w2v-bert-2.0-kinyarwanda-asr](https://huggingface.co/badrex/w2v-bert-2.0-kinyarwanda-asr),
fine-tuned on pooled Afrivoice (health-domain) and Common Voice (general-domain)
Kinyarwanda speech.

**Configuration:** r = 32, alpha = 64, dropout 0.05, target modules `linear_q` and
`linear_v`. 3,145,728 trainable parameters (0.54% of 583.6M). Feature-extractor and
feature-projection stems frozen.

## Results

| Test set | n clips | WER | CER |
|---|---|---|---|
| Afrivoice (in-domain, health) | 1,560 | __FILL__ | __FILL__ |
| Common Voice (cross-domain, general) | 16,205 | __FILL__ | __FILL__ |

> Fill these from your own benchmark run before publishing. Do not copy numbers
> from the report into this card without checking they describe *this* adapter.

**Contamination note.** The base badrex checkpoint was pretrained on Kinyarwanda
audio that included portions of the Digital Umuganda health corpus from which the
Afrivoice subset is drawn. The in-domain figure therefore reflects both this
adaptation and residual pretraining familiarity; the Common Voice figure is the
cleaner cross-domain measure.

## Usage

```python
from transformers import Wav2Vec2BertForCTC, AutoProcessor
from peft import PeftModel

base = "badrex/w2v-bert-2.0-kinyarwanda-asr"
model = PeftModel.from_pretrained(Wav2Vec2BertForCTC.from_pretrained(base), "__REPO__")
processor = AutoProcessor.from_pretrained("__REPO__")
```

## Intended use and limits

Research prototype for Kinyarwanda health-*information* delivery. Not a medical
device. Does not diagnose. Dialect coverage is limited to the standard Kinyarwanda
represented in the training corpora.

Author: Ganza Didier · African Leadership University · Supervisor: Emmanuel Adjei
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True, help="folder holding adapter_config.json")
    p.add_argument("--repo-id", required=True, help="e.g. ganzadidier/ubuzima-w2v-bert-lora")
    p.add_argument("--private", action="store_true")
    a = p.parse_args()

    src = Path(a.adapter_dir)
    cfg = src / "adapter_config.json"
    if not cfg.exists():
        raise SystemExit(f"No adapter_config.json in {src} — wrong folder?")

    have = {f.name for f in src.iterdir()}
    print("Found:", sorted(have))
    if not ({"adapter_model.safetensors", "adapter_model.bin"} & have):
        raise SystemExit("No adapter weights found next to adapter_config.json")
    if "preprocessor_config.json" not in have:
        print("WARNING: no processor saved here. The Space will fall back to the base "
              "processor — fine only if the tokenizer vocab is unchanged.")

    from huggingface_hub import HfApi, create_repo

    create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    (src / "README.md").write_text(CARD.replace("__REPO__", a.repo_id))
    HfApi().upload_folder(folder_path=str(src), repo_id=a.repo_id, repo_type="model")

    print(f"\n✓ Pushed to https://huggingface.co/{a.repo_id}")
    print(f"  Now set this in your Space:  ADAPTER_ID = {a.repo_id}")


if __name__ == "__main__":
    main()
