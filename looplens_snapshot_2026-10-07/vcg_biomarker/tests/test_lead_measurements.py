import numpy as np
import pytest

from vcg_biomarker.core.delineation.base import BeatDelineation, MarkerPoint, RecordDelineation
from vcg_biomarker.mimic import classical
from vcg_biomarker.mimic.lead_measurements import empty_lead_measurements, store_lead_measurements


def fixture():
    def point(sample):
        return MarkerPoint(sample=sample, confidence=1., source="synthetic")
    beat = BeatDelineation(p_onset=point(50), p_peak=point(65), p_offset=point(80),
        r_onset=point(110), r_peak=point(125), r_offset=point(145),
        t_onset=point(170), t_peak=point(210), t_offset=point(260))
    record = RecordDelineation(beats=[beat], fs=500, n_samples=350, ecg_id=1)
    signal = np.zeros((350, 12))
    signal[60:70] = .1
    signal[120:130] = 1.
    signal[130:140] = -.4
    signal[200:220] = .3
    return signal, record


def test_exports_all_leads_without_recomputing_qrs(monkeypatch):
    signal, record = fixture()
    calls = []
    original = classical._qrs_amplitudes
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(classical, "_qrs_amplitudes", counted)
    sink = {}
    classical.ClassicalECGCriteria().compute(signal, record, measurement_sink=sink)
    assert len(calls) == 1
    assert len(sink) == 96
    assert sink["lead_measurement_III_qrs_r_height_mv"] == pytest.approx(1.)
    assert sink["lead_measurement_aVR_qrs_s_depth_mv"] == pytest.approx(.4)
    assert sink["lead_measurement_aVL_t_t_max_mv"] == pytest.approx(.3)


def test_extended_local_export_reuses_primitives_and_preserves_criteria(monkeypatch):
    signal, record = fixture()
    engine = classical.ClassicalECGCriteria()
    normal = {}
    expected = engine.compute(signal, record, measurement_sink=normal)
    calls = []
    original = classical._st_t_metrics
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(classical, '_st_t_metrics', counted)
    extended = {}
    actual = engine.compute(signal, record, measurement_sink=extended, extended_measurements=True)
    assert len(calls) == 1
    assert len(extended) == 156
    # Audit 5.1: the diffuse ST pattern is no longer lab-only; both runs carry it.
    assert set(actual) == set(expected)
    assert 'lab_diffuse_st_depression_avr_v1_pattern' in expected
    assert {key: actual[key] for key in expected} == expected
    np.testing.assert_allclose([extended[k] for k in normal], list(normal.values()), equal_nan=True)
    assert extended['lead_measurement_I_st_st60_mv'] == signal[175, 0]


def test_extended_export_does_not_report_clipped_record_end():
    signal, record = fixture()
    sink = {}
    classical.ClassicalECGCriteria().compute(signal[:160], record, measurement_sink=sink,
                                             extended_measurements=True)
    assert np.isfinite(sink['lead_measurement_I_st_st20_mv'])
    assert np.isnan(sink['lead_measurement_I_st_st60_mv'])
    assert np.isnan(sink['lead_measurement_I_st_st80_mv'])


def test_flat_lead_and_failed_segment_never_become_measured_zero():
    signal = np.arange(120).reshape(10, 12).astype(float)
    signal[:, 0] = 0
    primitives = {"qrs": {"I": {"r_height": 1.}, "II": {"r_height": 2.}}}
    sink = {}
    store_lead_measurements(sink, signal, primitives, {"qrs": True})
    assert np.isnan(sink["lead_measurement_I_qrs_r_height_mv"])
    assert sink["lead_measurement_II_qrs_r_height_mv"] == 2.
    store_lead_measurements(sink, signal, primitives, {"qrs": False})
    assert all(np.isnan(x) for x in sink.values())


def test_baseline_offset_does_not_change_single_lead_measurements():
    signal, record = fixture()
    first, second = {}, {}
    engine = classical.ClassicalECGCriteria()
    engine.compute(signal, record, measurement_sink=first)
    engine.compute(signal + np.arange(12), record, measurement_sink=second)
    assert first.keys() == second.keys()
    np.testing.assert_allclose(list(first.values()), list(second.values()), atol=1e-12, equal_nan=True)


def test_no_representative_beat_has_a_complete_missing_measurement_schema():
    record = RecordDelineation(beats=[], fs=500, n_samples=350, ecg_id=1)
    sink = {}
    classical.ClassicalECGCriteria().compute(np.zeros((350, 12)), record, measurement_sink=sink)
    assert sink.keys() == empty_lead_measurements().keys()
    assert all(np.isnan(x) for x in sink.values())


def test_measurement_only_path_skips_unused_q_wave_and_criteria_calculations(monkeypatch):
    signal, record = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("Unused initial-Q metrics were calculated")
    monkeypatch.setattr(classical, "_initial_q_metric", forbidden)
    sink = {}
    result = classical.ClassicalECGCriteria().compute(signal, record, measurement_sink=sink, measurements_only=True)
    assert result == {}
    assert sink["lead_measurement_I_qrs_r_height_mv"] == 1.
