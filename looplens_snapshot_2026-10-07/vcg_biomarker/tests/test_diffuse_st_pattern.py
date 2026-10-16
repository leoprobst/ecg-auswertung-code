import pytest
from vcg_biomarker.core.data_loader import LEAD_NAMES_12
from vcg_biomarker.mimic.ischemia_patterns import diffuse_st_pattern


def signals(count=6, elevation=.05):
    st = {lead: {'j': 0.} for lead in LEAD_NAMES_12}
    for lead in ('I', 'II', 'III', 'aVL', 'aVF', 'V4')[:count]:
        st[lead]['j'] = -.1
    st['aVR']['j'] = elevation
    return st


@pytest.mark.parametrize('count,elevation,positive', [(6,.05,True),(5,.05,False),(6,0,False),(6,-.01,False)])
def test_boundaries(count, elevation, positive):
    result = diffuse_st_pattern(signals(count,elevation), quality_ok=True)
    assert result['positive'] is positive
    assert result['clinical_diagnosis'] is False


def test_missing_not_negative_and_positive_evidence_preserved():
    st = signals(5)
    del st['V6']
    assert diffuse_st_pattern(st, quality_ok=True)['positive'] is None
    st['V4']['j'] = -.1
    assert diffuse_st_pattern(st, quality_ok=True)['positive'] is True


@pytest.mark.parametrize('bad', [True, None, float('nan'), '0.1'])
def test_invalid_elevation_not_accepted(bad):
    st = signals(elevation=bad)
    assert diffuse_st_pattern(st, quality_ok=True)['positive'] is None


def test_quality_and_conduction_exclusion():
    assert diffuse_st_pattern(signals(), quality_ok=False)['positive'] is None
    assert diffuse_st_pattern(signals(), quality_ok=True, excluded=True)['positive'] is None


def test_classical_lab_integration_does_not_change_default(monkeypatch):
    import numpy as np
    from vcg_biomarker.mimic import classical
    from vcg_biomarker.core.delineation.base import BeatDelineation, RecordDelineation, MarkerPoint
    beat = BeatDelineation(r_peak=MarkerPoint(500, 1., 'test'),
        p_onset=MarkerPoint(300, 1., 'test'), p_peak=MarkerPoint(325, 1., 'test'),
        p_offset=MarkerPoint(350, 1., 'test'), r_onset=MarkerPoint(450, 1., 'test'),
        r_offset=MarkerPoint(530, 1., 'test'), j_point=MarkerPoint(530, 1., 'test'),
        t_onset=MarkerPoint(550, 1., 'test'), t_peak=MarkerPoint(620, 1., 'test'),
        t_offset=MarkerPoint(700, 1., 'test'))
    delineation = RecordDelineation(beats=[beat], fs=500, n_samples=1000, ecg_id=1)
    original = classical._st_t_metrics
    def metrics(*args, **kwargs):
        result = original(*args, **kwargs)
        for lead, values in signals().items():
            result[lead]['j'] = values['j']
        return result
    monkeypatch.setattr(classical, '_st_t_metrics', metrics)
    engine = classical.ClassicalECGCriteria()
    ecg = np.zeros((1000, 12))
    normal = engine.compute(ecg, delineation)
    extended = engine.compute(ecg, delineation, extended_measurements=True)
    key = 'lab_diffuse_st_depression_avr_v1_pattern'
    # Audit 5.1: the pattern is computed in every run, not only in the lab,
    # and stays exploratory; the extended flag changes nothing else.
    assert normal[key].positive is True and normal[key].exploratory is True
    assert extended == normal
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_measurement_cells, classical_rule_view
    displayed = classical_measurement_cells(key, extended[key].to_dict())
    assert 'exploratives Muster erfüllt' in displayed[2]
    assert 'academic.oup.com' in displayed[2]
    assert classical_rule_view(extended[key].to_dict())['eligible'] is False
