"""app.py — UBUZIMA AI: Gradio entry point and pipeline orchestrator.

Corresponds to Section 4.1.7 of the report. Holds the Config, Telemetry and
Pipeline classes drawn in Figure 3.6, and renders the interface captured in
Figures 4.1-4.6.

Tabs map to the functional requirements:
  Assistant   -> FR1-FR7   (ask, transcribe, reason, synthesise, replay)
  Telemetry   -> FR8, NFR1 (per-stage latency; feeds Table 5.8)
  Privacy     -> NFR8      (navigable EULA / privacy notice inside the app)
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field

import gradio as gr

import errors
import guards
from asr_service import ASRService
from llm_service import LLMService, LLMConfigurationError
from safety_prompt import SafetyPolicy
from tts_service import TTSService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ubuzima")

STAGES = ("asr", "safety", "llm", "tts", "total")


# ─────────────────────────────── Config ──────────────────────────────────────
@dataclass
class Config:
    """Populated from Hugging Face Space environment variables (Figure 3.6)."""
    import os as _os
    base_model_id: str = _os.environ.get("BASE_MODEL_ID", "badrex/w2v-bert-2.0-kinyarwanda-asr")
    adapter_id: str = _os.environ.get("ADAPTER_ID", "")
    gemini_model: str = _os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    @classmethod
    def from_env(cls) -> "Config":
        return cls()


# ────────────────────────────── Telemetry ────────────────────────────────────
@dataclass
class Telemetry:
    """Collects per-stage latency samples; exposes summary() used by Chapter 5."""
    samples: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def record(self, timings: dict[str, float]) -> None:
        self.samples.append(timings)

    def model_split(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.samples:
            m = s.get("_model")
            if m:
                out[m] = out.get(m, 0) + 1
        return out

    def record_error(self, code: str) -> None:
        self.errors.append(code)

    def summary(self) -> dict:
        """Median / 5th / 95th percentile per stage, in seconds.

        These are exactly the cells Table 5.8 of the report is waiting for.
        """
        if not self.samples:
            return {}
        out = {}
        for stage in STAGES:
            vals = sorted(s[stage] for s in self.samples if stage in s
                          and isinstance(s.get(stage), (int, float)))
            if not vals:
                continue
            out[stage] = {
                "n": len(vals),
                "median": statistics.median(vals),
                "p5": vals[max(0, int(0.05 * len(vals)) - 1)],
                "p95": vals[min(len(vals) - 1, int(0.95 * len(vals)))],
                "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
        return out

    def as_markdown(self) -> str:
        s = self.summary()
        if not s:
            return "_No requests recorded yet. Ask a question on the Assistant tab._"
        rows = [
            "| Pipeline stage | Median (s) | 5th pct | 95th pct | Mean | SD |",
            "|---|---|---|---|---|---|",
        ]
        names = {"asr": "ASR (LoRA badrex)", "safety": "Safety prompt assembly",
                 "llm": "LLM (Gemini 2.5 Flash)", "tts": "TTS (YourTTS)",
                 "total": "End-to-end total"}
        for stage in STAGES:
            if stage not in s:
                continue
            d = s[stage]
            rows.append(f"| {names[stage]} | {d['median']:.2f} | {d['p5']:.2f} | "
                        f"{d['p95']:.2f} | {d['mean']:.2f} | {d['std']:.2f} |")
        split = self.model_split()
        if len(split) > 1:
            rows.append("")
            rows.append("⚠️ **These samples mix models — " +
                        ", ".join(f"`{k}`×{v}" for k, v in sorted(split.items())) +
                        ". The LLM median below is a blend and is NOT a clean "
                        "Table 5.8 number. Pin one model and re-measure.**")
        n = s.get("total", {}).get("n", 0)
        med = s.get("total", {}).get("median", 0)
        verdict = "MET" if med <= 12 else "EXCEEDED"
        rows.append("")
        rows.append(f"**{n} warm requests. NFR1 target is 12 s median — currently {verdict} "
                    f"at {med:.2f} s.**")
        if s:
            dom = max((k for k in s if k != "total"), key=lambda k: s[k]["median"])
            pct = 100 * s[dom]["median"] / med if med else 0
            rows.append(f"**Dominant stage: {names[dom]} — {pct:.0f}% of median end-to-end.**")
        return "\n".join(rows)


# ─────────────────────────────── Pipeline ────────────────────────────────────
class Pipeline:
    """Coordinates the three model calls, the safety layer and telemetry."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()
        self.telemetry = Telemetry()
        self.policy = SafetyPolicy()
        self.asr = ASRService()
        try:
            self.llm = LLMService()
            self.llm_error = None
        except LLMConfigurationError as e:
            self.llm, self.llm_error = None, str(e)
            log.error("%s", e)
        self.tts = TTSService()

    def _fail(self, msg: "errors.Message", transcript: str = ""):
        """Render an error to the user — spoken in Kinyarwanda where it makes sense."""
        self.telemetry.record_error(msg.code)
        wav = None
        if msg.speak and msg.rw and self.tts.available:
            try:
                wav = self.tts.invoke(msg.rw)
            except Exception:
                pass                      # never let the error path raise
        return transcript, msg.rw, wav, msg.as_markdown()

    def run(self, audio_path: str):
        """Returns (transcript, response_text, wav_path, status_markdown)."""
        try:
            return self._run(audio_path)
        except Exception:
            log.exception("Unhandled failure")
            return self._fail(errors.INTERNAL)

    def _run(self, audio_path: str):
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        # ── Guard tier 1: is this audio usable at all? ──
        if err := guards.check_audio(audio_path):
            return self._fail(err)

        # ── Guard tier 3: is it even Kinyarwanda? (optional, ENABLE_LID=1) ──
        err, lid_detail = guards.check_spoken_language(audio_path)
        if err:
            return self._fail(err)

        # ── ASR ──
        try:
            s = time.perf_counter()
            transcript, conf = self.asr.invoke_with_confidence(audio_path)
            t["asr"] = time.perf_counter() - s
        except Exception:
            log.exception("ASR failed")
            return self._fail(errors.ASR_EMPTY)

        # ── Safety FIRST. A phrase that trips the safety patterns is by
        #    definition recognised Kinyarwanda, and a refusal must never be
        #    downgraded into "please speak Kinyarwanda". Safety outranks the
        #    language guard.
        s = time.perf_counter()
        label = self.policy.classify(transcript)
        prompted = self.policy.apply(transcript)
        t["safety"] = time.perf_counter() - s

        # ── Guard tier 2: did we get Kinyarwanda words we believe? ──
        if label is None:
            if err := guards.check_transcript(transcript, conf):
                return self._fail(err, transcript)

        # ── LLM (bypassed entirely for diagnosis / medication / emergency) ──
        s = time.perf_counter()
        if label:
            response = self.policy.refusal_responses[label]
            note = f"🛡️ Safety filter fired (`{label}`) — LLM bypassed."
        elif self.llm is None:
            return self._fail(errors.LLM_UNAVAILABLE, transcript)
        else:
            try:
                response, model_used = self.llm.invoke(prompted)
                t["_model"] = model_used
            except Exception as e:
                msg = str(e).lower()
                if "quota" in msg or "rate" in msg or "429" in msg:
                    return self._fail(errors.LLM_RATE_LIMIT, transcript)
                if "block" in msg or "safety" in msg:
                    return self._fail(errors.LLM_BLOCKED, transcript)
                return self._fail(errors.LLM_UNAVAILABLE, transcript)

            # Gemini sometimes drifts into English. One retry, then withhold —
            # a correct answer in the wrong language is still a failed answer here.
            if guards.kinyarwanda_overlap(response) < 0.20:
                log.warning("LLM answered off-language; retrying once")
                try:
                    response, _ = self.llm.invoke(
                        prompted + "\n\nSUBIZA MU KINYARWANDA GUSA. "
                                   "Ntukoreshe Icyongereza cyangwa Igifaransa.")
                except Exception:
                    return self._fail(errors.LLM_WRONG_LANGUAGE, transcript)
                if guards.kinyarwanda_overlap(response) < 0.20:
                    return self._fail(errors.LLM_WRONG_LANGUAGE, transcript)

            response = self.policy.add_disclaimer(response)
            note = (f"✅ Answered by `{t.get('_model', '?')}` under the Kinyarwanda "
                    f"safety prompt.")
        t["llm"] = time.perf_counter() - s

        # ── TTS (failure degrades to text, never to silence) ──
        wav = None
        try:
            s = time.perf_counter()
            wav = self.tts.invoke(response)
            t["tts"] = time.perf_counter() - s
        except Exception as e:
            self.telemetry.record_error(errors.TTS_UNAVAILABLE.code)
            t["tts"] = time.perf_counter() - s
            note += f"\n\n⚠️ {errors.TTS_UNAVAILABLE.en} ({e})"

        t["total"] = time.perf_counter() - t0
        self.telemetry.record(t)

        status = (f"{note}\n\nASR confidence {conf:.2f} · "
                  f"Kinyarwanda overlap {guards.kinyarwanda_overlap(transcript):.2f}"
                  f"{' · LID ' + str(lid_detail.get('lid')) if lid_detail.get('lid') not in (None, 'disabled') else ''}"
                  f"\n\nASR {t['asr']:.2f}s · LLM {t['llm']:.2f}s · "
                  f"TTS {t.get('tts', 0):.2f}s · **total {t['total']:.2f}s**")
        return transcript, response, wav, status


# ──────────────────────────────── UI ─────────────────────────────────────────
PRIVACY = """
## Privacy notice and terms of use

**UBUZIMA AI is a research prototype. It is not a medical device.**
It does not diagnose, does not prescribe, and must not be used to make a clinical
decision. Always consult a qualified health worker.

### What happens to your voice
* Your recording is transcribed on this Space's GPU and is **not** stored after the response is returned.
* The **transcribed text** is sent to Google's Gemini API for reasoning. It leaves Rwanda and is processed on Google infrastructure under Google's terms.
* No name, phone number, patient identifier or account is collected.
* Only anonymous timings and error codes are counted, to measure performance.

### What you should not send
Do not speak anyone's name, national ID, or any detail that identifies a specific
patient. Voice is biometric data under Rwanda's Law N° 058/2021 on the protection
of personal data and privacy.

### Legal status
This prototype has not been authorised by the National Cyber Security Authority for
processing sensitive personal data, and is deployed for academic evaluation only.
Any pilot beyond this study requires that authorisation.

### Licence
Source code released under Apache 2.0. Model weights follow their own upstream licences.
"""


def build_ui(pipeline: Pipeline) -> gr.Blocks:
    with gr.Blocks(title="UBUZIMA AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# UBUZIMA AI\n### Umufasha w'amakuru y'ubuzima mu Kinyarwanda")
        gr.Markdown(
            "Baza ikibazo cy'ubuzima mu Kinyarwanda — usubizwe mu Kinyarwanda. "
            "**Iki ni igikoresho cy'ubushakashatsi, si muganga.**"
        )

        with gr.Tab("Assistant"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                        label="Baza ikibazo cyawe (Kinyarwanda)")
                    go = gr.Button("Ohereza", variant="primary")
                with gr.Column():
                    transcript = gr.Textbox(label="Ibyo wavuze (transcription)", lines=3)
                    response = gr.Textbox(label="Igisubizo", lines=8)
                    audio_out = gr.Audio(label="Umva igisubizo", autoplay=False)
            status = gr.Markdown()
            go.click(pipeline.run, inputs=audio_in,
                     outputs=[transcript, response, audio_out, status])

        with gr.Tab("Telemetry"):
            gr.Markdown("### Per-stage latency across this session's warm requests")
            table = gr.Markdown()
            gr.Button("Refresh").click(lambda: pipeline.telemetry.as_markdown(), outputs=table)
            gr.Markdown(
                f"**ASR model:** `{pipeline.asr.model_description}`  \n"
                f"**LLM:** `{pipeline.llm.description if pipeline.llm else pipeline.llm_error}`  \n"
                f"**TTS backend loaded:** `{pipeline.tts.available}`  \n"
                f"**Voice:** `{pipeline.tts.voice_description}`"
            )

        with gr.Tab("Privacy & Terms"):
            gr.Markdown(PRIVACY)

    return demo


if __name__ == "__main__":
    import os as _os_main
    _port = int(_os_main.environ.get("PORT", "7860"))
    build_ui(Pipeline()).launch(server_name="0.0.0.0", server_port=_port)
