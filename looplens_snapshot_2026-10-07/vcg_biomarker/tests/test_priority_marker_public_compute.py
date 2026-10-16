"""Exercise public registry compute with real delineation/context objects."""
import numpy as np
import pytest
from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.delineation.base import (
    BeatDelineation, MarkerPoint, RecordDelineation, RhythmAssessment, RhythmClass, WaveStatus,
)


def recording():
    signal = np.zeros((2200, 3))
    beats = []
    for r in (400, 900, 1400):
        def point(offset):
            return MarkerPoint(r + offset, 1., "synthetic")
        beats.append(BeatDelineation(r_peak=point(0), p_onset=point(-110),
            p_peak=point(-82), p_offset=point(-55), r_onset=point(-22),
            r_offset=point(32), j_point=point(32), t_onset=point(70),
            t_peak=point(140), t_offset=point(225), p_status=WaveStatus.PRESENT,
            qrs_status=WaveStatus.PRESENT, t_status=WaveStatus.PRESENT))
        signal[r-110:r-55+1, 0] = .2*np.sin(np.linspace(0, np.pi, 56))
        signal[r-22:r+33, 0] = np.sin(np.linspace(0, np.pi, 55))
        signal[r+70:r+226, 1] = .3*np.sin(np.linspace(0, np.pi, 156))
    return RecordDelineation(beats=beats, fs=500, n_samples=len(signal), ecg_id=0,
        quality_score=1., rhythm=RhythmAssessment(rhythm_class=RhythmClass.REGULAR)), signal


@pytest.mark.parametrize("name,expected", [
    ("t_energy_temporal_concentration", 1.5),
    ("p_early_late_directional_operator_overlap", 1.),
])
def test_public_compute_uses_real_context_and_fiducials(name, expected):
    record, signal = recording()
    result = BiomarkerRegistry.get(name).compute(record, vcg_xyz=signal, fs=500)
    assert result.value == pytest.approx(expected, rel=1e-3), result.computation_notes
    assert "clinical_claim_allowed=False" in result.computation_notes


def test_public_atrial_compute_rejects_af_pattern():
    record, signal = recording()
    record.rhythm = RhythmAssessment(rhythm_class=RhythmClass.AF_COMPATIBLE)
    result = BiomarkerRegistry.get("p_early_late_directional_operator_overlap").compute(
        record, vcg_xyz=signal, fs=500)
    assert result.value is None
