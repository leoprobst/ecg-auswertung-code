"""Point gates and segment gates must share one calibration.

The per-wave measurement thresholds were introduced so that P and T -- whose
boundaries are genuinely harder to place than a QRS onset -- are not held to
the QRS's strictness and abstained away.  That calibration was applied to the
*segment* gates and missed every *point* gate, which kept a hard-coded 0.35.

The two are the same scale: ``segment_confidence("p")`` is defined as
``min(p_onset.confidence, p_offset.confidence)``.  So a stricter point gate
makes the segment gate unreachable -- the marker is admitted and then loses
its samples, returning nothing.  Measured on 60 LUDB records, 200 of 457
admitted beats (43.8 % of P onsets) were discarded that way, and the whole
per-wave change was inert for every marker that resolves a landmark through
``_sample``.

Nothing in the suite noticed, because the markers abstain rather than fail:
a silently emptied measurement looks exactly like an honest abstention.  These
tests therefore assert the invariant directly -- what a segment gate admits, a
point gate must not throw away.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import catalog_biomarkers as catalog
from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
    fiducial_confidence_threshold,
    measurement_confidence_threshold,
)


LANDMARK_WAVES = {
    "p_onset": "p",
    "p_peak": "p",
    "p_offset": "p",
    "r_onset": "qrs",
    "r_peak": "qrs",
    "r_offset": "qrs",
    "q_peak": "qrs",
    "s_peak": "qrs",
    "t_onset": "t",
    "t_peak": "t",
    "t_offset": "t",
    "u_onset": "u",
    "u_offset": "u",
}


@pytest.mark.parametrize("landmark, wave", sorted(LANDMARK_WAVES.items()))
def test_a_landmark_is_judged_at_its_own_waves_strictness(landmark, wave):
    assert fiducial_confidence_threshold(landmark) == pytest.approx(
        measurement_confidence_threshold(wave)
    )


def test_an_unknown_landmark_falls_back_to_the_strictest_value():
    """A new landmark must be calibrated on purpose, not inherit a loose gate."""

    assert fiducial_confidence_threshold("sigma_notch") == pytest.approx(0.35)


def test_p_and_t_landmarks_are_not_held_to_the_qrs_threshold():
    """The regression this module exists for."""

    for landmark in ("p_onset", "p_peak", "p_offset",
                     "t_onset", "t_peak", "t_offset"):
        assert fiducial_confidence_threshold(landmark) < 0.35


def _context(confidence: float) -> catalog.FeatureContext:
    fs, n = 500, 300
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    vcg = np.column_stack(
        [np.cos(theta), np.sin(theta), 0.1 * np.sin(2.0 * theta)]
    )
    ecg = np.column_stack([vcg[:, index % 3] for index in range(12)])
    beat = BeatDelineation(
        p_onset=MarkerPoint(sample=5, confidence=confidence, source="test"),
        p_peak=MarkerPoint(sample=15, confidence=confidence, source="test"),
        p_offset=MarkerPoint(sample=25, confidence=confidence, source="test"),
        r_onset=MarkerPoint(sample=70, confidence=1.0, source="test"),
        r_peak=MarkerPoint(sample=110, confidence=1.0, source="test"),
        r_offset=MarkerPoint(sample=145, confidence=1.0, source="test"),
        t_onset=MarkerPoint(sample=170, confidence=confidence, source="test"),
        t_peak=MarkerPoint(sample=205, confidence=confidence, source="test"),
        t_offset=MarkerPoint(sample=295, confidence=confidence, source="test"),
    )
    delineation = RecordDelineation(
        beats=[beat], fs=fs, n_samples=n, ecg_id=1
    )
    return catalog.FeatureContext(delineation, ecg, vcg, fs)


@pytest.mark.parametrize(
    "landmark", ["p_onset", "p_offset", "t_onset", "t_offset"]
)
def test_what_the_segment_gate_admits_the_point_gate_keeps(landmark):
    """The exact defect: admitted at 0.15, then discarded at 0.35."""

    confidence = 0.20
    wave = LANDMARK_WAVES[landmark]
    context = _context(confidence)

    # Precondition: this beat really is admitted at segment level.
    assert context.delineation.segment_confidence(wave) >= (
        measurement_confidence_threshold(wave)
    )
    # Therefore its samples must be resolvable.
    assert context._sample(landmark) is not None


def test_a_caller_can_still_ask_for_a_stricter_gate():
    """The override stays available -- it just is not the silent default."""

    context = _context(0.20)
    assert context._sample("p_onset", minimum_confidence=0.35) is None
    assert context._sample("p_onset", minimum_confidence=0.10) is not None


def test_a_qrs_landmark_keeps_the_strict_gate():
    """The fix must not loosen the wave that was calibrated strict."""

    context = _context(0.20)
    assert fiducial_confidence_threshold("r_onset") == pytest.approx(0.35)
    assert context._sample("r_onset") is not None  # built at confidence 1.0
