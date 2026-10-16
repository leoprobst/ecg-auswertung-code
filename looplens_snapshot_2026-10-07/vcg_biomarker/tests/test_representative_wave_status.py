"""The consensus beat takes its wave status from the beats, not from the medoid."""
from __future__ import annotations

from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
    WaveStatus,
)

FS = 500


def _marker(sample: int, *, reviewed: bool = False) -> MarkerPoint:
    return MarkerPoint(sample=sample, confidence=0.9, source="manual_review" if reviewed else "test")


def _beat(r: int, *, p: bool = True, p_status: WaveStatus = WaveStatus.PRESENT) -> BeatDelineation:
    return BeatDelineation(
        r_peak=_marker(r),
        p_onset=_marker(r - 110) if p else None,
        p_peak=_marker(r - 82) if p else None,
        p_offset=_marker(r - 55) if p else None,
        r_onset=_marker(r - 22), r_offset=_marker(r + 32), j_point=_marker(r + 32),
        t_onset=_marker(r + 70), t_peak=_marker(r + 140), t_offset=_marker(r + 225),
        p_status=p_status, qrs_status=WaveStatus.PRESENT, t_status=WaveStatus.PRESENT,
    )


def test_medoid_with_rejected_p_no_longer_removes_the_record_p() -> None:
    beats = tuple(_beat(500 + 400 * i) for i in range(7))
    medoid = _beat(500, p=False, p_status=WaveStatus.NOT_ASSESSABLE)
    representative = RecordDelineation._build_representative_beat(beats[1:] + (medoid,), medoid, FS)
    assert representative.p_onset is not None and representative.p_offset is not None
    assert representative.resolved_wave_status("p") == WaveStatus.PRESENT
    assert representative.segment_confidence("p") > 0.0


def test_minority_p_is_not_promoted() -> None:
    # Rhythm guards (e.g. atrial fibrillation) mark most beats without a
    # discrete P; two beats with P boundaries must not make it "present".
    beats = tuple(_beat(500 + 400 * i, p=i < 2, p_status=WaveStatus.PRESENT if i < 2
                        else WaveStatus.REPLACED_BY_ATRIAL_ACTIVITY) for i in range(7))
    medoid = beats[4]
    representative = RecordDelineation._build_representative_beat(beats, medoid, FS)
    assert representative.resolved_wave_status("p") != WaveStatus.PRESENT


def test_manual_review_of_the_medoid_is_kept() -> None:
    beats = tuple(_beat(500 + 400 * i) for i in range(5))
    medoid = _beat(500, p_status=WaveStatus.NOT_ASSESSABLE)
    medoid.p_onset = _marker(390, reviewed=True)
    representative = RecordDelineation._build_representative_beat(beats[1:] + (medoid,), medoid, FS)
    assert representative.p_status == WaveStatus.NOT_ASSESSABLE


def test_report_atrial_activity_removes_p_from_record_and_consensus() -> None:
    beats = [_beat(500 + 400 * i) for i in range(6)]
    record = RecordDelineation(beats=beats, fs=FS, n_samples=3500, ecg_id=1)
    assert record.representative_beat.p_onset is not None
    changed = record.suppress_discrete_p("discrete_p_suppressed_by_report")
    assert changed == 6
    assert record.representative_beat.p_onset is None
    assert record.representative_beat.resolved_wave_status("p") != WaveStatus.PRESENT
    assert not record.is_segment_evaluable("p")
    assert "discrete_p_suppressed_by_report" in record.rhythm.flags
    # QRS and T are untouched.
    assert record.representative_beat.t_offset is not None
    assert record.representative_beat.r_onset is not None


def test_report_suppression_keeps_manually_reviewed_p() -> None:
    beats = [_beat(500 + 400 * i) for i in range(4)]
    beats[0].p_onset = _marker(390, reviewed=True)
    record = RecordDelineation(beats=beats, fs=FS, n_samples=2200, ecg_id=2)
    assert record.suppress_discrete_p("discrete_p_suppressed_by_report") == 3
    assert beats[0].p_onset is not None
