#!/usr/bin/env python3
"""voice_clone.py — create a new Kinyarwanda voice for UBUZIMA AI.

DigitalUmuganda/Kinyarwanda_YourTTS_v1 is a YourTTS model: a speaker encoder turns
a reference recording into a d-vector, and the synthesiser conditions on it. So you
do not train anything. You hand it ~1 minute of someone's voice and it speaks in
that voice. This script prepares that minute properly and lets you audition it.

Usage
─────
    # 1. prepare a raw recording into a clean reference
    python voice_clone.py prepare --input raw_voice.m4a --out voices/nurse

    # 2. listen to the cloned voice saying real health sentences
    python voice_clone.py audition --voice voices/nurse

    # 3. compare candidate voices side by side
    python voice_clone.py audition --voice voices/nurse --voice voices/chw --voice default

Then set SPEAKER_WAV=voices/nurse/reference.wav in your Space.

Consent
───────
You are cloning a real person's voice. Get their written permission, tell them it
will be public on the internet saying health sentences, and keep the permission
note next to the audio. Appendix C is where that record belongs.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

SR_ENCODER = 16000          # speaker encoder expects 16 kHz
TARGET_SECONDS = 60.0
MIN_SECONDS = 20.0

# Real sentences from the system's actual output distribution — not "hello world".
AUDITION_SENTENCES = [
    "Muraho. Ndi UBUZIMA AI, umufasha w'amakuru y'ubuzima.",
    "Malariya iterwa n'umubu witwa anofele. Kwirinda ni ukuryama mu nzitiramubu.",
    "Umubyeyi utwite agomba kurya indyo yuzuye, akanywa amazi ahagije, "
    "kandi akajya kwipimisha buri kwezi.",
    "Icyitonderwa: aya ni amakuru rusange y'ubuzima, si isuzuma rya muganga. "
    "Nyamuneka sura umuganga cyangwa ujye ku kigo nderabuzima kikwegereye.",
    "Simbasha kukubwira indwara ufite. Nyamuneka jya ku kigo nderabuzima "
    "kikwegereye vuba bishoboka.",
]


# ─────────────────────────── prepare ─────────────────────────────────────────
def _load_mono_16k(path: Path) -> np.ndarray:
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception:
        import librosa
        data, sr = librosa.load(str(path), sr=None, mono=False)
        data = data.T if data.ndim > 1 else data
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR_ENCODER:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR_ENCODER)
    return data.astype("float32")


def _trim_silence(x: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Drop leading/trailing silence and collapse long internal gaps."""
    import librosa
    intervals = librosa.effects.split(x, top_db=top_db)
    if len(intervals) == 0:
        return x
    gap = np.zeros(int(0.15 * SR_ENCODER), dtype="float32")  # keep 150 ms breaths
    return np.concatenate([np.concatenate([x[s:e], gap]) for s, e in intervals])


def _normalise(x: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-6:
        return x
    gain = 10 ** (target_dbfs / 20) / rms
    y = x * gain
    peak = float(np.abs(y).max())
    return y / peak * 0.97 if peak > 0.97 else y


def cmd_prepare(args) -> None:
    src, out = Path(args.input), Path(args.out)
    if not src.exists():
        raise SystemExit(f"No such file: {src}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src.name} ...")
    x = _load_mono_16k(src)
    print(f"  raw: {x.size / SR_ENCODER:.1f}s")

    x = _trim_silence(x)
    print(f"  after silence trim: {x.size / SR_ENCODER:.1f}s")

    dur = x.size / SR_ENCODER
    if dur < MIN_SECONDS:
        raise SystemExit(
            f"Only {dur:.1f}s of speech. YourTTS needs about a minute; under "
            f"{MIN_SECONDS:.0f}s the d-vector is unstable and the clone drifts. "
            "Record more."
        )
    if dur > TARGET_SECONDS:
        x = x[: int(TARGET_SECONDS * SR_ENCODER)]
        print(f"  trimmed to {TARGET_SECONDS:.0f}s")

    x = _normalise(x)
    ref = out / "reference.wav"
    sf.write(str(ref), x, SR_ENCODER)

    # Also write 3 chunks — passing several files averages the d-vector, which is
    # usually steadier than one long take.
    n = 3
    step = x.size // n
    chunks = []
    for i in range(n):
        c = out / f"chunk_{i+1}.wav"
        sf.write(str(c), x[i * step:(i + 1) * step], SR_ENCODER)
        chunks.append(c)

    (out / "CONSENT.md").write_text(
        "# Voice consent record\n\n"
        "- Speaker name: \n- Date recorded: \n- Recorded by: \n"
        "- Permission given for: public research demonstration of UBUZIMA AI\n"
        "- Speaker understands the cloned voice will speak health-information "
        "sentences on a public website: YES / NO\n- Signature: \n\n"
        "Attach this to Appendix C of the report.\n"
    )

    rms_dbfs = 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9)
    print(f"\n✓ reference.wav  {x.size / SR_ENCODER:.1f}s @ {SR_ENCODER} Hz, "
          f"RMS {rms_dbfs:.1f} dBFS")
    print(f"✓ {n} chunks for d-vector averaging")
    print(f"✓ CONSENT.md — fill this in, it is not optional")
    print(f"\nNext:  python voice_clone.py audition --voice {out}")


# ─────────────────────────── audition ────────────────────────────────────────
def _synth():
    from huggingface_hub import snapshot_download
    from TTS.utils.synthesizer import Synthesizer
    import torch

    local = Path(snapshot_download("DigitalUmuganda/Kinyarwanda_YourTTS_v1"))

    def pick(*names):
        for n in names:
            if (local / n).exists():
                return str(local / n)
        return None

    return Synthesizer(
        tts_checkpoint=pick("best_model.pth", "model.pth"),
        tts_config_path=pick("config.json"),
        tts_speakers_file=pick("speakers.pth"),
        encoder_checkpoint=pick("SE_checkpoint.pth.tar") or "",
        encoder_config=pick("config_se.json") or "",
        use_cuda=torch.cuda.is_available(),
    ), local


def cmd_audition(args) -> None:
    synth, repo = _synth()
    sr = synth.tts_config.audio["sample_rate"]
    outdir = Path("voice_auditions")
    outdir.mkdir(exist_ok=True)

    for voice in args.voice:
        if voice == "default":
            ref, name = str(repo / "conditioning_audio.wav"), "default"
        else:
            vp = Path(voice)
            ref = str(vp / "reference.wav")
            name = vp.name
            if not Path(ref).exists():
                print(f"  skip {voice}: no reference.wav — run `prepare` first")
                continue

        print(f"\n── {name} ──")
        for i, text in enumerate(AUDITION_SENTENCES, 1):
            wav = np.asarray(synth.tts(text, speaker_wav=ref), dtype="float32")
            path = outdir / f"{name}_{i}.wav"
            sf.write(str(path), wav, sr)
            print(f"  {path}  ({len(wav)/sr:.1f}s)  {text[:50]}...")

    print(f"\nListen to everything in {outdir}/ and pick the voice that a Rwandan "
          f"health worker would actually sound like.")
    print("Sentence 4 is the disclaimer and sentence 5 is the refusal — those two "
          "matter most. They are what people hear when the system says no.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="clean a raw recording into a reference")
    pp.add_argument("--input", required=True)
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=cmd_prepare)

    pa = sub.add_parser("audition", help="hear the cloned voice say real sentences")
    pa.add_argument("--voice", action="append", required=True,
                    help="a folder from `prepare`, or the word 'default'")
    pa.set_defaults(func=cmd_audition)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
