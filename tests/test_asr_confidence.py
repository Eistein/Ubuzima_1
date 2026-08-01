"""
tests/test_asr_confidence.py — Unit tests for ASR confidence scoring.

Validates the geometric-mean confidence computation, blank-id resolution,
and confidence band classification.

Report reference: §4.3.3 (Unit Testing), §5.2.5 (Confidence Calibration)
"""

import os
import pytest
import torch

from asr_confidence import (
    ctc_confidence,
    resolve_blank_id,
    confidence_band,
    should_call_llm,
    HIGH_CONF,
    LOW_CONF,
)


class TestThresholds:
    """Validate threshold configuration."""

    def test_high_conf_defined(self):
        assert HIGH_CONF is not None
        assert 0 < HIGH_CONF < 1

    def test_low_conf_defined(self):
        assert LOW_CONF is not None
        assert 0 < LOW_CONF < 1

    def test_high_greater_than_low(self):
        assert HIGH_CONF > LOW_CONF

    def test_default_values(self):
        """Default thresholds should match report §5.2.5."""
        assert HIGH_CONF == 0.90 or os.environ.get("HIGH_CONF") is not None
        assert LOW_CONF == 0.75 or os.environ.get("LOW_CONF") is not None


class TestCTCConfidence:
    """Test the core confidence computation."""

    def test_perfect_prediction(self):
        """If model is 100% certain, confidence should be ~1.0."""
        # Create logits where one class has very high probability
        logits = torch.zeros(10, 5)
        logits[:, 2] = 100.0  # Class 2 is certain
        conf, n_frames = ctc_confidence(logits, blank_id=0)
        assert conf > 0.99
        assert n_frames == 10  # All frames non-blank

    def test_all_blank(self):
        """If all frames are blank, confidence should be 0.0."""
        logits = torch.zeros(10, 5)
        logits[:, 0] = 100.0  # Class 0 (blank) is certain
        conf, n_frames = ctc_confidence(logits, blank_id=0)
        assert conf == 0.0
        assert n_frames == 0

    def test_mixed_prediction(self):
        """Moderate certainty should give moderate confidence."""
        logits = torch.zeros(10, 5)
        # Some frames certain, some uncertain
        logits[:5, 1] = 2.0   # ~88% probability
        logits[5:, 2] = 1.0   # ~73% probability
        conf, n_frames = ctc_confidence(logits, blank_id=0)
        assert 0.5 < conf < 0.95
        assert n_frames == 10

    def test_batch_dimension_handled(self):
        """Should handle [1, T, V] shape by squeezing."""
        logits = torch.zeros(1, 10, 5)
        logits[0, :, 1] = 100.0
        conf, n_frames = ctc_confidence(logits, blank_id=0)
        assert conf > 0.99
        assert n_frames == 10

    def test_output_range(self):
        """Confidence must always be in [0, 1]."""
        torch.manual_seed(42)
        for _ in range(10):
            logits = torch.randn(20, 10)
            conf, _ = ctc_confidence(logits, blank_id=0)
            assert 0.0 <= conf <= 1.0

    def test_non_blank_frames_count(self):
        """n_frames should equal number of non-blank predictions."""
        logits = torch.zeros(10, 5)
        logits[:3, 0] = 100.0   # Blank
        logits[3:7, 1] = 100.0  # Non-blank
        logits[7:, 0] = 100.0   # Blank
        conf, n_frames = ctc_confidence(logits, blank_id=0)
        assert n_frames == 4  # Frames 3,4,5,6


class TestConfidenceBand:
    """Test three-band classification."""

    def test_high_band(self):
        assert confidence_band(0.95) == "high"
        assert confidence_band(1.0) == "high"
        assert confidence_band(HIGH_CONF) == "high"

    def test_medium_band(self):
        assert confidence_band(0.80) == "medium"
        assert confidence_band(LOW_CONF) == "medium"

    def test_low_band(self):
        assert confidence_band(0.50) == "low"
        assert confidence_band(0.0) == "low"
        assert confidence_band(LOW_CONF - 0.01) == "low"

    def test_boundary_high_medium(self):
        """At exactly HIGH_CONF, should be high."""
        assert confidence_band(HIGH_CONF) == "high"

    def test_boundary_medium_low(self):
        """At exactly LOW_CONF, should be medium."""
        assert confidence_band(LOW_CONF) == "medium"


class TestShouldCallLLM:
    """Test the safety gate decision."""

    def test_high_confidence_allowed(self):
        assert should_call_llm("high") is True

    def test_medium_confidence_allowed(self):
        assert should_call_llm("medium") is True

    def test_low_confidence_blocked(self):
        """Low confidence transcripts must NOT reach the LLM."""
        assert should_call_llm("low") is False

    def test_invalid_band_blocked(self):
        """Unknown bands should default to safe (blocked)."""
        # This is a safety decision: when in doubt, block
        assert should_call_llm("unknown") is False
        assert should_call_llm("") is False


class TestResolveBlankId:
    """Test blank token ID resolution."""

    def test_mock_processor(self):
        """Test with a mock processor object."""
        class MockTokenizer:
            pad_token_id = 0

        class MockProcessor:
            tokenizer = MockTokenizer()

        blank_id = resolve_blank_id(MockProcessor())
        assert blank_id == 0

    def test_fallback_to_model_config(self):
        """Test fallback when tokenizer has no pad_token_id."""
        class MockTokenizer:
            pass  # No pad_token_id

        class MockConfig:
            pad_token_id = 1

        class MockModel:
            config = MockConfig()

        class MockProcessor:
            tokenizer = MockTokenizer()

        blank_id = resolve_blank_id(MockProcessor(), MockModel())
        assert blank_id == 1

    def test_raises_when_not_found(self):
        """Should raise ValueError when blank id cannot be determined."""
        class EmptyProcessor:
            pass

        with pytest.raises(ValueError):
            resolve_blank_id(EmptyProcessor())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
