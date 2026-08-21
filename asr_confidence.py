"""
asr_confidence.py — runtime confidence scoring and gating for UBUZIMA AI.

Implements report §5.2.5: "a per-utterance confidence score from the ASR model,
computed as the geometric mean of the maximum softmax probability over non-blank
frames of the CTC output", together with the gate §5.2.5 says the orchestrator
applies.

This module is deliberately lean and dependency-light (torch only) so it can be
imported by app.py at start-up without pulling in evaluation tooling. The
calibration harness that chooses the thresholds lives in eval/, not here.

WHY THE GATE IS A SAFETY CONTROL, NOT INTERFACE POLISH
------------------------------------------------------
The characteristic failure of a cascaded ASR -> LLM -> TTS pipeline is not that
the recogniser errs. It is what happens next: a garbled transcript reaches the
language model, which answers the WRONG QUESTION fluently, and the user hears a
well-formed Kinyarwanda health answer to something they never asked. Declining to
answer is strictly safer than answering the wrong question.

HONEST CAVEAT
-------------
Mean max-softmax over CTC frames is a heuristic proxy for correctness, not a
calibrated probability; neural models are systematically over-confident. Report
Table 5.5 establishes that the score carries real information about error (a
monotonic rank relationship), which is what a threshold needs. It does not
establish that 0.90 means "90% likely correct". Proper calibration would require
temperature scaling fitted on a validation split, reported with a reliability
diagram and expected calibration error.
"""

import os

import torch

# --------------------------------------------------------------------------
# Thresholds
#
# HIGH_CONF = 0.90 is documented in report §5.2.5.
# LOW_CONF must match the value you calibrated in
# notebooks/01_asr_confidence_check.ipynb — keep the deployed number and the
# written method identical, and state both in the report.
# --------------------------------------------------------------------------

HIGH_CONF = float(os.environ.get("HIGH_CONF", 0.90))
LOW_CONF = float(os.environ.get("LOW_CONF", 0.75))


def ctc_confidence(logits, blank_id):
    """Per-utterance confidence from CTC logits.

    Geometric mean of the maximum softmax probability over non-blank frames.

    Geometric rather than arithmetic mean: the score behaves like a product of
    per-frame probabilities, so a single very uncertain frame correctly pulls the
    result down instead of being averaged away.

    Non-blank frames only: in CTC most frames are blank and are predicted with
    very high probability. Including them would push every score toward 1.0 and
    destroy the discrimination the threshold depends on.

    Args:
        logits:   [1, T, V] or [T, V] raw pre-softmax model output.
        blank_id: CTC blank token id (the pad token for Wav2Vec2-family models).

    Returns:
        (confidence in [0, 1], number of non-blank frames used)
    """
    if logits.dim() == 3:
        logits = logits[0]

    probs = torch.softmax(logits.float(), dim=-1)
    max_probs, pred_ids = probs.max(dim=-1)

    mask = pred_ids != blank_id
    n_kept = int(mask.sum().item())
    if n_kept == 0:
        # Nothing but blanks — the model heard no speech at all.
        return 0.0, 0

    kept = max_probs[mask].clamp_min(1e-12)
    return float(torch.exp(torch.log(kept).mean()).item()), n_kept


def resolve_blank_id(processor, model=None):
    """Find the CTC blank id. For Wav2Vec2-family models this is the pad token."""
    tokenizer = getattr(processor, "tokenizer", None)
    for candidate in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
    ):
        if candidate is not None:
            return int(candidate)
    raise ValueError(
        "Could not determine the CTC blank id. Inspect "
        "processor.tokenizer.pad_token_id and pass it explicitly."
    )


def confidence_band(conf):
    """Return 'high', 'medium' or 'low'."""
    if conf >= HIGH_CONF:
        return "high"
    if conf >= LOW_CONF:
        return "medium"
    return "low"


def should_call_llm(band):
    """The gate. Only explicitly high- or medium-confidence transcripts reach the LLM.

    Fail-safe (allow-list). This is written as ``band in {"high", "medium"}`` rather
    than ``band != "low"`` on purpose: an unexpected value — an empty string, or a
    band the classifier was never meant to emit — is treated as unsafe and blocked,
    instead of being allowed through by a deny-list default. For a safety control,
    "when in doubt, block" must be the behaviour of the code, not just the comment.
    """
    return band in {"high", "medium"}
