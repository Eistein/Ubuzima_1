#!/usr/bin/env python3
"""tone_probe.py — measure the tonal gap in Kinyarwanda TTS.

WHY THIS EXISTS
───────────────
"The voice sounds robotic" is an impression. This turns it into a number, which is
what Chapter 5 needs and what an examiner will actually credit.

THE ARGUMENT, IN ONE PARAGRAPH
──────────────────────────────
Kinyarwanda tone is contrastive: Kimenyi documents minimal pairs distinguished by
tone alone — inda 'stomach' vs indá 'louse', umuryaango 'family' vs umuryáango
'door'. Standard Kinyarwanda orthography does not mark tone. So the two members of
each pair are the SAME STRING once written normally. A grapheme-input synthesiser
therefore receives identical input for two different words and must emit identical
audio. This is not an empirical weakness to be measured — it is a proof. The
measurement below quantifies how much contrast is destroyed.

Two modes:

  python tone_probe.py collapse
      Proves the orthographic collapse. Needs no model and no audio. Runs in 1s.

  python tone_probe.py measure --human recordings/ --out results/
      You record each pair member; this extracts F0, normalises to semitones over
      your own median, and reports human contrast vs synthesised contrast.

REFERENCES
──────────
Kimenyi, A. (2002). A Tonal Grammar of Kinyarwanda: An Autosegmental and Metrical
    Analysis. Lewiston, NY: Edwin Mellen Press.
Kimenyi, A. (n.d.). Kinyarwanda Tones Made Easy. California State University,
    Sacramento.
Muhirwe, J. (2010). Morphological analysis of tone marked Kinyarwanda text. In
    A. Yli-Jyrä et al. (Eds.), Finite-State Methods and Natural Language Processing
    (pp. 48-55). Berlin: Springer.
Myers, S. (2003). F0 timing in Kinyarwanda. Phonetica, 60(2), 71-97.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Minimal pairs. The first two are Kimenyi's, cited above.
#
# GANZA: you are the native speaker and I am not. Verify these two, then add
# eight more — health vocabulary if you can find any, since that is the domain
# your system speaks in. Ten pairs is a credible probe; two is an anecdote.
# ─────────────────────────────────────────────────────────────────────────────
MINIMAL_PAIRS = [
    {"a": "indá",       "a_gloss": "louse",  "b": "inda",       "b_gloss": "stomach",
     "source": "Kimenyi (n.d.)"},
    {"a": "umuryáango", "a_gloss": "door",   "b": "umuryaango", "b_gloss": "family",
     "source": "Kimenyi (n.d.)"},
    # {"a": "...", "a_gloss": "...", "b": "...", "b_gloss": "...", "source": "your source"},
]

TONE_MARKS = {"\u0301", "\u0300", "\u0302", "\u030C"}   # acute, grave, circumflex, caron


def strip_tone(s: str) -> str:
    """Write the word the way Kinyarwanda is actually written: without tone marks."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if c not in TONE_MARKS)


# ─────────────────────────── mode 1: the proof ───────────────────────────────
def cmd_collapse(args) -> None:
    print("\nORTHOGRAPHIC COLLAPSE — what the synthesiser actually receives")
    print("=" * 78)
    print(f"{'tone-marked':<16}{'gloss':<10}{'as written':<16}{'gloss':<10}{'collapse?'}")
    print("-" * 78)

    collapsed = 0
    for p in MINIMAL_PAIRS:
        wa, wb = strip_tone(p["a"]), strip_tone(p["b"])
        same = wa == wb
        collapsed += same
        print(f"{p['a']:<16}{p['a_gloss']:<10}{wa:<16}{p['b_gloss']:<10}"
              f"{'IDENTICAL ✗' if same else 'distinct ✓'}")
        print(f"{p['b']:<16}{p['b_gloss']:<10}{wb:<16}")
        print()

    n = len(MINIMAL_PAIRS)
    print("=" * 78)
    print(f"{collapsed}/{n} pairs collapse to the same input string.")
    print()
    print("For every collapsed pair the synthesiser is handed one string and must")
    print("return one waveform. It cannot distinguish the two words. Not 'does not")
    print("distinguish them well' — CANNOT. The information is absent from the input.")
    print()
    print("This is why conditioning on a better reference voice does not fix tone.")
    print("Speaker conditioning changes WHO is talking. It cannot recover WHAT was")
    print("never written down.")
    if collapsed == n and n < 5:
        print()
        print(f"⚠  Only {n} pairs. Add more before putting this in the report — ten")
        print("   pairs is a probe, two is an anecdote.")


# ─────────────────────────── mode 2: measurement ─────────────────────────────
def f0_semitones(path: str, fmin: float = 60, fmax: float = 400):
    """F0 track in semitones relative to the utterance's own median.

    Semitones-over-own-median makes contours comparable across speakers and across
    human vs synthetic audio, which raw Hz does not.
    """
    import librosa

    y, sr = librosa.load(path, sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                 frame_length=1024, hop_length=160)
    f0 = f0[voiced & ~np.isnan(f0)]
    if f0.size < 5:
        return None, None
    med = float(np.median(f0))
    return 12 * np.log2(f0 / med), med


def contour_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference between two semitone contours, time-normalised to 50 points."""
    if a is None or b is None:
        return float("nan")
    g = np.linspace(0, 1, 50)
    ra = np.interp(g, np.linspace(0, 1, len(a)), a)
    rb = np.interp(g, np.linspace(0, 1, len(b)), b)
    return float(np.sqrt(np.mean((ra - rb) ** 2)))


def cmd_measure(args) -> None:
    human = Path(args.human)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading synthesiser...")
    from huggingface_hub import snapshot_download
    from TTS.utils.synthesizer import Synthesizer
    import soundfile as sf
    import torch

    local = Path(snapshot_download("DigitalUmuganda/Kinyarwanda_YourTTS_v1"))

    def pick(*names):
        for n in names:
            hits = sorted(local.rglob(n))
            if hits:
                return str(hits[0])
        return None

    synth = Synthesizer(
        tts_checkpoint=pick("best_model.pth", "model.pth", "*.pth"),
        tts_config_path=pick("config.json"),
        tts_speakers_file=pick("speakers.pth", "speakers.json"),
        encoder_checkpoint=pick("SE_checkpoint.pth.tar", "*SE*.pth*") or "",
        encoder_config=pick("config_se.json") or "",
        use_cuda=torch.cuda.is_available(),
    )
    sr_out = synth.tts_config.audio["sample_rate"]
    ref = args.speaker_wav or pick("conditioning_audio.wav", "*.wav")

    rows = []
    for i, p in enumerate(MINIMAL_PAIRS, 1):
        written = strip_tone(p["a"])
        if strip_tone(p["b"]) != written:
            print(f"skip pair {i}: not orthographically identical")
            continue

        # human recordings: you saying each member, in a carrier phrase
        ha = human / f"pair{i}_a.wav"
        hb = human / f"pair{i}_b.wav"
        if not (ha.exists() and hb.exists()):
            print(f"skip pair {i}: need {ha.name} and {hb.name}")
            continue

        fa, meda = f0_semitones(str(ha))
        fb, medb = f0_semitones(str(hb))
        human_dist = contour_distance(fa, fb)

        # synth: ONE string, so synthesise once and again — any difference is
        # sampling noise, not tone.
        wav1 = np.asarray(synth.tts(written, speaker_wav=ref), dtype="float32")
        wav2 = np.asarray(synth.tts(written, speaker_wav=ref), dtype="float32")
        s1, s2 = out / f"pair{i}_synth_1.wav", out / f"pair{i}_synth_2.wav"
        sf.write(s1, wav1, sr_out)
        sf.write(s2, wav2, sr_out)
        fs1, _ = f0_semitones(str(s1))
        fs2, _ = f0_semitones(str(s2))
        synth_dist = contour_distance(fs1, fs2)

        rows.append({
            "pair": i,
            "written": written,
            "a": p["a"], "a_gloss": p["a_gloss"],
            "b": p["b"], "b_gloss": p["b_gloss"],
            "human_contrast_st": round(human_dist, 3),
            "synth_contrast_st": round(synth_dist, 3),
            "human_f0_range_st": round(float(np.ptp(fa)), 2) if fa is not None else None,
            "synth_f0_range_st": round(float(np.ptp(fs1)), 2) if fs1 is not None else None,
        })
        print(f"pair {i}  {written:<14} human contrast {human_dist:5.2f} st | "
              f"synth {synth_dist:5.2f} st")

    if not rows:
        raise SystemExit("Nothing measured. Record the pairs first — see --help.")

    (out / "tone_probe.json").write_text(json.dumps(rows, indent=2))

    hc = np.mean([r["human_contrast_st"] for r in rows])
    sc = np.mean([r["synth_contrast_st"] for r in rows])
    hr = np.mean([r["human_f0_range_st"] for r in rows if r["human_f0_range_st"]])
    srg = np.mean([r["synth_f0_range_st"] for r in rows if r["synth_f0_range_st"]])

    print("\n" + "=" * 72)
    print("TONE PROBE — paste into Chapter 5")
    print("=" * 72)
    print(f"pairs measured                       {len(rows)}")
    print(f"mean human tonal contrast            {hc:.2f} semitones")
    print(f"mean synthesised contrast            {sc:.2f} semitones  "
          f"({100*sc/hc:.0f}% of human)")
    print(f"mean human F0 range                  {hr:.2f} semitones")
    print(f"mean synthesised F0 range            {srg:.2f} semitones  "
          f"({100*srg/hr:.0f}% of human)")
    print()
    print(f"→ The synthesiser preserves {100*sc/hc:.0f}% of the tonal contrast that")
    print(f"  distinguishes these words in natural speech. Residual synth contrast is")
    print(f"  sampling noise: the input strings were identical.")
    print(f"\nJSON → {out/'tone_probe.json'}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collapse", help="prove the orthographic collapse (no model needed)")
    c.set_defaults(func=cmd_collapse)

    m = sub.add_parser("measure", help="F0 contrast, human vs synthesised")
    m.add_argument("--human", required=True,
                   help="folder of pair{N}_a.wav / pair{N}_b.wav — you saying each member")
    m.add_argument("--speaker-wav", default="", help="your reference.wav")
    m.add_argument("--out", default="results/tone")
    m.set_defaults(func=cmd_measure)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
