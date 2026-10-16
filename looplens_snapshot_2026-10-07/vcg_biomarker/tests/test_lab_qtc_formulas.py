from types import SimpleNamespace
import pytest

from vcg_biomarker.mimic.ecg_lab import _representative_intervals


def delineation(pairs, irregular=False):
    beat = SimpleNamespace(**dict.fromkeys(('p_onset','p_offset','r_onset','r_offset','t_peak','t_offset','u_onset','u_offset')))
    from vcg_biomarker.mimic.ecg_lab import WaveStatus
    beat.resolved_wave_status = lambda _: WaveStatus.PRESENT
    return SimpleNamespace(representative_beat=beat, segment_confidence=lambda _: 0.,
        rhythm=SimpleNamespace(requires_beatwise_rate_correction=irregular),
        qt_rr_measurements=lambda: [SimpleNamespace(qt_ms=qt, preceding_rr_seconds=rr, confidence=.9) for qt, rr in pairs])


@pytest.mark.parametrize('rr,expected', [(1.,400.),(.8,426.25),(.5,505.)])
def test_hodges_units_and_preceding_rr(rr, expected):
    rows = _representative_intervals(delineation([(400.,rr)]))
    value = next(r for r in rows if r['interval'] == 'QTc Hodges')
    assert value['value_ms'] == pytest.approx(expected)
    assert value['rr_seconds_median'] == rr
    assert value['uncertainty_kind'] == 'between_beat_iqr_not_measurement_ci'


def test_hodges_respects_existing_pair_gate():
    rows = _representative_intervals(delineation([(400.,1.)], irregular=True))
    corrections = [r for r in rows if r['interval'].startswith('QTc ')]
    assert len(corrections) == 4
    assert all(r['status'] == 'not_evaluable' for r in corrections)
