"""tts_service.py - Kinyarwanda speech synthesis (Section 4.1.6, TTSService in Figure 3.6).

Backend: DigitalUmuganda/Kinyarwanda_YourTTS_v1 - YourTTS (VITS + speaker encoder),
trained on 67.8 h of studio-quality Kinyarwanda. Unlike MMS it supports zero-shot
speaker conditioning: give it ~1 minute of a real Rwandan voice as `speaker_wav`
and it clones that speaker's timbre.

WHY IT STILL WILL NOT SOUND FULLY NATURAL - read before burning a week on it.

Kinyarwanda is tonal, but standard Kinyarwanda orthography does not mark tone, and
this model was trained on graphemes with `use_phonemes: False`. It never sees which
of two identically-spelled words carries which tone contour, and cannot reconstruct
that at inference. Published naturalness MOS for this model is 2.3/5 (Digital
Umuganda, AfricaNLP@ICLR 2023). The mbazaNLP FastPitch model card states outright
that it "does not always capture the Kinyarwanda tones" and recommends training
future models against a tonal dictionary.

The flatness you are hearing is a documented property of every open Kinyarwanda TTS
available today, not a mistake in your setup. Conditioning on a good reference voice
is the largest gain available without training a new model.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import kinya_text

log = logging.getLogger(__name__)

TTS_REPO = os.environ.get("TTS_REPO", "DigitalUmuganda/Kinyarwanda_YourTTS_v1")
# ~1 min of clean speech from the voice you want. The repo ships
# conditioning_audio.wav; replace it with a real Rwandan health-worker voice
# (recorded with consent) for a markedly better result.
SPEAKER_WAV = os.environ.get("SPEAKER_WAV", "")

# VITS inference knobs. These are real and they move the needle on how mechanical
# the delivery sounds. Tune them by ear with tts_tune.py — the defaults below are
# VITS defaults, not defaults chosen for Kinyarwanda.
LENGTH_SCALE = float(os.environ.get("TTS_LENGTH_SCALE", "1.0"))   # >1 = slower
NOISE_SCALE = float(os.environ.get("TTS_NOISE_SCALE", "0.667"))   # timbre variation
NOISE_SCALE_DP = float(os.environ.get("TTS_NOISE_SCALE_DP", "0.8"))  # rhythm variation


TARGET_DBFS = float(os.environ.get("TTS_TARGET_DBFS", "-20.0"))


class TTSUnavailable(RuntimeError):
    pass


def _match_loudness(w: np.ndarray, target_dbfs: float, strength: float = 0.7) -> np.ndarray:
    """Pull a chunk toward a target RMS level.

    Peak normalisation (what this used to do) leaves sentence-to-sentence loudness
    all over the place, because one loud consonant sets the peak for a whole chunk.
    Levelling on RMS is what makes concatenated sentences sound like one take.

    strength < 1 deliberately under-corrects, so genuine emphasis between sentences
    survives instead of being flattened into a compressor-brick.
    """
    rms = float(np.sqrt(np.mean(w ** 2)))
    if rms < 1e-6:
        return w
    full_gain = 10 ** (target_dbfs / 20) / rms
    return w * (full_gain ** strength)


class TTSService:
    def __init__(self, repo: str = TTS_REPO, speaker_wav: str = SPEAKER_WAV):
        self.repo = repo
        self.speaker_wav = speaker_wav
        self.voice_source = "unknown"     # surfaced on the Telemetry tab
        self._synth = None
        self.sample_rate = 22050
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            from huggingface_hub import snapshot_download
            from TTS.utils.synthesizer import Synthesizer

            local = Path(snapshot_download(self.repo))

            def pick(*names):
                for n in names:
                    p = local / n
                    if p.exists():
                        return str(p)
                return None

            model = pick("best_model.pth", "model.pth")
            config = pick("config.json")
            if not model or not config:
                raise TTSUnavailable(f"No model/config found in {self.repo}")

            self._synth = Synthesizer(
                tts_checkpoint=model,
                tts_config_path=config,
                tts_speakers_file=pick("speakers.pth"),
                encoder_checkpoint=pick("SE_checkpoint.pth.tar") or "",
                encoder_config=pick("config_se.json") or "",
                use_cuda=self._cuda(),
            )
            self._resolve_voice(pick("conditioning_audio.wav") or "")
            self.sample_rate = self._synth.tts_config.audio["sample_rate"]
            self._apply_inference_params()
            log.info("TTS ready: %s @ %d Hz", self.repo, self.sample_rate)
        except Exception as e:
            self._synth = None
            log.error("TTS backend unavailable: %s", e)

    def _resolve_voice(self, repo_default: str) -> None:
        """Work out which voice we are ACTUALLY using, and never guess.

        The bug this replaces: the old code only fell back to the repo default when
        SPEAKER_WAV was EMPTY. If SPEAKER_WAV was set but pointed at a file that did
        not exist — a typo, a relative path resolved from the wrong cwd, a file never
        committed to the Space — the string was non-empty, so no fallback fired, no
        check ran, and the bad path went straight into tts(). Depending on the build
        that either raised (and invoke()'s per-chunk try/except swallowed it) or let
        the synthesiser quietly pick a built-in speaker. Either way you would hear a
        voice that was not yours and be told nothing.
        """
        want = (self.speaker_wav or "").strip()
        if want:
            p = Path(want).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.is_file():
                self.speaker_wav = str(p)
                self.voice_source = f"cloned: {p.name}"
                log.info("Voice: cloned from %s", p)
                return
            log.error(
                "SPEAKER_WAV=%r does not exist (resolved to %s). NOT falling back "
                "silently — the voice you hear would not be the one you configured. "
                "Commit the file to the Space and check the path.", want, p)
            self.voice_source = f"MISCONFIGURED — {want} not found"

        if repo_default and Path(repo_default).is_file():
            self.speaker_wav = repo_default
            if self.voice_source == "unknown":
                self.voice_source = "repo default conditioning audio"
                log.warning("No SPEAKER_WAV set — using the repo's default voice. "
                            "Record a real Kinyarwanda voice; this is the single "
                            "largest perceptual win available.")
            else:
                self.voice_source += " → fell back to repo default"
        else:
            self.speaker_wav = ""
            self.voice_source = "NONE — built-in speaker, clone NOT active"
            log.error("No usable conditioning audio at all. YourTTS will use a "
                      "built-in speaker and your clone is not active.")

    def _apply_inference_params(self) -> None:
        """Set the VITS knobs, reporting which ones this build actually exposes."""
        m = getattr(self._synth, "tts_model", None)
        if m is None:
            return
        wanted = {
            "length_scale": LENGTH_SCALE,
            "inference_noise_scale": NOISE_SCALE,
            "noise_scale": NOISE_SCALE,
            "inference_noise_scale_dp": NOISE_SCALE_DP,
            "noise_scale_dp": NOISE_SCALE_DP,
            "inference_noise_scale_w": NOISE_SCALE_DP,
        }
        applied = []
        for k, v in wanted.items():
            if hasattr(m, k):
                setattr(m, k, v)
                applied.append(f"{k}={v}")
        log.info("VITS params applied: %s", ", ".join(applied) or "none exposed")

    @staticmethod
    def _cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._synth is not None

    @property
    def voice_description(self) -> str:
        return self.voice_source

    def invoke(self, text: str) -> str:
        """Normalise, segment, synthesise sentence by sentence, join with real pauses.

        The old version threw the whole 600-character response at the model in one
        call and truncated whatever did not fit. That is where the breath-and-pause
        artefacts in Section 5.4 came from: the duration predictor drifts over long
        inputs, and the model never inserts the silence a person puts between
        sentences. Synthesising per sentence and joining with measured pauses is more
        stable AND less mechanical.
        """
        if self._synth is None:
            raise TTSUnavailable(
                "TTS backend not loaded. Check `pip install coqui-tts` succeeded and "
                f"that {self.repo} downloaded."
            )
        text = kinya_text.normalise((text or "").strip())
        if not text:
            raise ValueError("TTSService.invoke() received empty text")

        pieces = []
        for chunk, pause in kinya_text.segment(text):
            try:
                w = np.asarray(self._synth.tts(chunk, speaker_wav=self.speaker_wav or None),
                               dtype=np.float32)
            except Exception as e:
                log.warning("Chunk failed, skipping: %r (%s)", chunk[:40], e)
                continue
            if w.ndim > 1:
                w = w.mean(axis=1)
            pieces.append(_match_loudness(w, TARGET_DBFS))
            pieces.append(np.zeros(int(pause * self.sample_rate), dtype=np.float32))

        if not pieces:
            raise TTSUnavailable("Every chunk failed to synthesise")

        wav = np.concatenate(pieces)
        wav = _match_loudness(wav, TARGET_DBFS, strength=1.0)   # whole-utterance level
        peak = float(np.abs(wav).max() or 1.0)
        if peak > 0.97:
            wav = wav / peak * 0.97

        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(out.name, wav, self.sample_rate)
        return out.name
