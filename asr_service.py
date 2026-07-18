"""asr_service.py — badrex Wav2Vec2-BERT 2.0 with the UBUZIMA LoRA adapter.

Corresponds to Section 4.1.3 of the report and the ASRService class in Figure 3.6.
The adapter is loaded on top of the frozen base checkpoint; if no adapter is
configured the service falls back to the base model and says so loudly, because
silently serving the un-adapted model would invalidate every WER figure in
Chapter 5.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, Wav2Vec2BertForCTC

log = logging.getLogger(__name__)

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "badrex/w2v-bert-2.0-kinyarwanda-asr")
# HF Hub id of YOUR published adapter, e.g. "ganzadidier/ubuzima-w2v-bert-lora"
ADAPTER_ID = os.environ.get("ADAPTER_ID", "").strip()
TARGET_SR = 16_000

# Set LOW_RESOURCE=1 when running on free CPU hardware. Applies dynamic int8
# quantisation to the Linear layers, which roughly halves CPU inference time and
# memory. It also changes the model: the WER in Chapter 5 was measured on the
# fp16 GPU model, so if you serve quantised you must re-benchmark and report BOTH
# numbers. Never quote the GPU WER for a quantised deployment.
LOW_RESOURCE = os.environ.get("LOW_RESOURCE", "0") == "1"
CPU_THREADS = int(os.environ.get("CPU_THREADS", "0"))


class ASRService:
    """Wraps the fine-tuned ASR model behind a single invoke() call."""

    def __init__(self, base_model_id: str = BASE_MODEL_ID, adapter_id: str = ADAPTER_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.adapter_id = adapter_id
        self.adapter_loaded = False

        log.info("Loading base ASR model %s onto %s", base_model_id, self.device)
        t0 = time.perf_counter()

        # The processor ships with the base checkpoint. If you saved a processor
        # alongside the adapter (the notebook does), prefer that one — the tokenizer
        # vocabulary must match what the adapter was trained against.
        processor_src = adapter_id or base_model_id
        try:
            self.processor = AutoProcessor.from_pretrained(processor_src)
        except Exception:
            log.warning("No processor at %s; falling back to base", processor_src)
            self.processor = AutoProcessor.from_pretrained(base_model_id)

        self.model = Wav2Vec2BertForCTC.from_pretrained(base_model_id, torch_dtype=self.dtype)
        self.model.config.ctc_zero_infinity = True

        if adapter_id:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_id)
            self.model = self.model.merge_and_unload()  # fold LoRA in for faster inference
            self.adapter_loaded = True
            log.info("LoRA adapter %s merged into base model", adapter_id)
        else:
            log.error(
                "ADAPTER_ID is not set — serving the UN-ADAPTED base model. "
                "The WER figures reported in Chapter 5 do NOT describe this configuration."
            )

        self.model.to(self.device).eval()

        self.quantised = False
        if self.device == "cpu":
            if CPU_THREADS:
                torch.set_num_threads(CPU_THREADS)
            log.warning("Running ASR on CPU. Expect several seconds per utterance; "
                        "NFR1's 12s median was specified for a T4.")
            if LOW_RESOURCE:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                self.quantised = True
                log.warning("int8 dynamic quantisation applied. This is NOT the model "
                            "Chapter 5 benchmarked — re-run benchmark.py and report "
                            "the quantised WER separately.")

        log.info("ASR ready in %.1fs (device=%s, quantised=%s)",
                 time.perf_counter() - t0, self.device, self.quantised)

    @property
    def model_description(self) -> str:
        base = (f"{BASE_MODEL_ID} + LoRA ({self.adapter_id})" if self.adapter_loaded
                else f"{BASE_MODEL_ID} (base only — NO ADAPTER)")
        tags = [self.device]
        if getattr(self, "quantised", False):
            tags.append("int8 — RE-BENCHMARK BEFORE QUOTING WER")
        return f"{base} [{', '.join(tags)}]"

    def _load_audio(self, audio_path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:                       # force mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        return waveform.squeeze().numpy()

    @torch.inference_mode()
    def invoke(self, audio_path: str) -> str:
        """Transcribe one Kinyarwanda utterance. Returns the decoded string."""
        audio = self._load_audio(audio_path)
        if audio.size == 0:
            return ""
        inputs = self.processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).to(self.device)
        if self.dtype == torch.float16:
            inputs["input_features"] = inputs["input_features"].half()
        logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0].strip()

    @torch.inference_mode()
    def invoke_with_confidence(self, audio_path: str) -> tuple[str, float]:
        """Transcribe and also return mean per-frame softmax confidence (Section 6.6)."""
        audio = self._load_audio(audio_path)
        if audio.size == 0:
            return "", 0.0
        inputs = self.processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).to(self.device)
        if self.dtype == torch.float16:
            inputs["input_features"] = inputs["input_features"].half()
        logits = self.model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values.mean().item()
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0].strip(), conf
