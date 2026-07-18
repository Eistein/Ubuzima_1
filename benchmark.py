#!/usr/bin/env python3
"""benchmark.py — reproduces the Chapter 5 measurements.

Two modes.

1. ASR accuracy (Sections 5.2.1-5.2.5):
       python benchmark.py --split test --domain both \
              --model badrex/w2v-bert-2.0-kinyarwanda-asr --adapter <your-adapter-id>

2. End-to-end latency (Section 5.3, Table 5.8) — this is the one you still need:
       python benchmark.py --latency 50 --audio-dir ./samples

   Mode 2 runs N warm requests through the real deployed pipeline, discards the
   cold start, and prints the median / 5th / 95th percentile per stage in exactly
   the shape Table 5.8 expects.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


# ─────────────────────────── Mode 2: latency ─────────────────────────────────
def run_latency(n: int, audio_dir: str, out: str, pin_model: str = "") -> None:
    import os
    if pin_model:
        # Pin BOTH so no fallback can slip a second model into the sample and
        # quietly blend two latency distributions into one median.
        os.environ["GEMINI_MODEL"] = pin_model
        os.environ["GEMINI_FALLBACK_MODEL"] = pin_model
        print(f"Pinned LLM to {pin_model} — no fallback, so Table 5.8 is clean.")
    from app import Pipeline

    clips = sorted(p for p in Path(audio_dir).glob("*")
                   if p.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"})
    if not clips:
        raise SystemExit(f"No audio files in {audio_dir}. Record 10-15s Kinyarwanda "
                         f"health questions and put them there.")

    pipe = Pipeline()
    print(f"Warming up (cold start excluded, per Section 5.3)...")
    pipe.run(str(clips[0]))
    pipe.telemetry.samples.clear()

    print(f"Running {n} warm requests over {len(clips)} clips...")
    for i in range(n):
        clip = clips[i % len(clips)]
        pipe.run(str(clip))
        print(f"  [{i+1}/{n}] {clip.name}", end="\r")
    print()

    s = pipe.telemetry.summary()
    Path(out).write_text(json.dumps(s, indent=2))

    names = {"asr": "ASR (LoRA-fine-tuned badrex)", "safety": "Safety prompt assembly",
             "llm": "LLM (Gemini 2.5 Flash)", "tts": "TTS (YourTTS)",
             "total": "End-to-end total"}
    print("\n" + "=" * 74)
    print("TABLE 5.8 — paste these into the report")
    print("=" * 74)
    print(f"{'Pipeline stage':<32}{'Median':>10}{'5th pct':>12}{'95th pct':>12}")
    for k in ("asr", "safety", "llm", "tts", "total"):
        if k not in s:
            continue
        d = s[k]
        print(f"{names[k]:<32}{d['median']:>10.2f}{d['p5']:>12.2f}{d['p95']:>12.2f}")

    med = s["total"]["median"]
    dom = max((k for k in s if k != "total"), key=lambda k: s[k]["median"])
    print("\n[DOMINANT_STAGE]   =", names[dom])
    print("[DOMINANT_PCT]     =", f"{100 * s[dom]['median'] / med:.0f}")
    print("[TOT_MED]          =", f"{med:.2f}")
    print("[MET_OR_FELL_SHORT]=", "met" if med <= 12 else "fell short of")
    print(f"\nFull JSON -> {out}")


# ─────────────────────────── Mode 1: ASR accuracy ────────────────────────────
def run_asr(model: str, adapter: str, split: str, domain: str, limit: int, out: str) -> None:
    import evaluate
    from asr_service import ASRService

    wer_metric, cer_metric = evaluate.load("wer"), evaluate.load("cer")
    svc = ASRService(base_model_id=model, adapter_id=adapter)
    print(f"Model under test: {svc.model_description}")

    manifest = Path(f"manifests/{domain}_{split}.jsonl")
    if not manifest.exists():
        raise SystemExit(
            f"Missing {manifest}.\n"
            "Export it from the training notebook — one JSON object per line with "
            '{"audio_path": ..., "transcription": ...}. Keep the speaker-disjoint '
            "filter applied, or the numbers are contaminated (Section 3.2.3)."
        )

    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]

    preds, refs, rtfs, per_clip = [], [], [], []
    for i, r in enumerate(rows, 1):
        import soundfile as sf
        dur = sf.info(r["audio_path"]).duration
        t0 = time.perf_counter()
        hyp = svc.invoke(r["audio_path"])
        wall = time.perf_counter() - t0
        preds.append(hyp)
        refs.append(r["transcription"])
        rtfs.append(wall / dur if dur else 0)
        per_clip.append({"audio_path": r["audio_path"], "ref": r["transcription"],
                         "hyp": hyp, "rtf": wall / dur if dur else 0})
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", end="\r")

    wer = 100 * wer_metric.compute(predictions=preds, references=refs)
    cer = 100 * cer_metric.compute(predictions=preds, references=refs)

    # 95% bootstrap CI, 500 resamples — matches Section 5.2.1
    import random
    boots = []
    idx = range(len(preds))
    for _ in range(500):
        sample = [random.choice(idx) for _ in idx]
        boots.append(100 * wer_metric.compute(
            predictions=[preds[i] for i in sample],
            references=[refs[i] for i in sample]))
    boots.sort()
    lo, hi = boots[12], boots[487]

    print(f"\n{domain} {split}: n={len(preds)}  WER={wer:.2f}%  "
          f"95% CI [{lo:.2f}, {hi:.2f}]  CER={cer:.2f}%  "
          f"RTF median={statistics.median(rtfs):.4f}")
    Path(out).write_text("\n".join(json.dumps(r) for r in per_clip))
    print(f"Per-clip predictions -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="UBUZIMA AI benchmarks")
    p.add_argument("--latency", type=int, metavar="N",
                   help="run N warm end-to-end requests and emit Table 5.8")
    p.add_argument("--audio-dir", default="./samples")
    p.add_argument("--pin-model", default="",
                   help="pin the LLM for a clean measurement, e.g. gemini-2.5-flash")
    p.add_argument("--model", default="badrex/w2v-bert-2.0-kinyarwanda-asr")
    p.add_argument("--adapter", default="", help="HF id of your published LoRA adapter")
    p.add_argument("--split", default="test")
    p.add_argument("--domain", default="both", choices=["afrivoice", "common_voice", "both"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()

    if a.latency:
        run_latency(a.latency, a.audio_dir, a.out or "results/latency.json", a.pin_model)
        return
    Path("results").mkdir(exist_ok=True)
    domains = ["afrivoice", "common_voice"] if a.domain == "both" else [a.domain]
    for d in domains:
        run_asr(a.model, a.adapter, a.split, d, a.limit, a.out or f"results/{d}_{a.split}.jsonl")


if __name__ == "__main__":
    main()
