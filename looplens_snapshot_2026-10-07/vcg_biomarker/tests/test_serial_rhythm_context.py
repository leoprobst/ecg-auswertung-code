import numpy as np
import pytest

from vcg_biomarker.core.temporal_research import serial_rhythm_context


def descriptor(pattern, confidence=.9):
    return {"rhythm_context": {"pattern": pattern, "confidence": confidence}}


@pytest.mark.parametrize("value", [None, np.nan, np.inf, 0, -1, 1.1, "invalid"])
def test_invalid_confidence_is_unknown_not_stable(value):
    result = serial_rhythm_context(descriptor("regular", value), descriptor("regular"))
    assert result["comparison"] == "unknown"
    assert result["before"]["confidence"] is None


def test_same_pattern_does_not_certify_activation_or_pacing():
    result = serial_rhythm_context(descriptor("regular"), descriptor("regular"))
    assert result["comparison"] == "same_recorded_pattern"
    assert not result["clinical_rhythm_stability_verified"]
    assert result["pacing_transition"] == "unknown"


def test_change_and_absence_are_separate():
    assert serial_rhythm_context(descriptor("regular"), descriptor("af_compatible_pattern"))["comparison"] == "recorded_pattern_change"
    assert serial_rhythm_context({}, {})["comparison"] == "unknown"
    assert serial_rhythm_context(descriptor("invented"), descriptor("invented"))["comparison"] == "unknown"
