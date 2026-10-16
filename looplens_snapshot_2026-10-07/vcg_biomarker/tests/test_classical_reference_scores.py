"""Audit 2026-09-23, section 5.1: published reference scores."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vcg_biomarker.core.data_loader import LEAD_MAP, LEAD_NAMES_12
from vcg_biomarker.mimic import classical

FS = 500


def _beat(*, qrs_ms=100.0, qt_ms=400.0, p_status="present"):
    point = lambda sample: SimpleNamespace(sample=sample, confidence=0.9)  # noqa: E731
    return SimpleNamespace(
        p_onset=point(30), p_offset=point(80),
        r_onset=point(100), r_offset=point(100 + int(qrs_ms * FS / 1000) - 1),
        t_peak=point(260), t_offset=point(300),
        qrs_duration_ms=lambda fs: qrs_ms,
        qt_interval_ms=lambda fs: qt_ms,
        resolved_wave_status=lambda wave: SimpleNamespace(value=p_status),
    )


def _qrs(**overrides):
    values = {lead: {"r_height": 1.0, "s_depth": 0.5, "peak_to_peak": 1.5} for lead in LEAD_NAMES_12}
    for lead, fields in overrides.items():
        values[lead].update(fields)
    return values


def _st(**overrides):
    values = {lead: {"j": 0.0, "st20": 0.0, "st60": 0.0, "st80": 0.0, "t_peak": 0.3,
                     "t_max": 0.3, "t_min": 0.0} for lead in LEAD_NAMES_12}
    for lead, fields in overrides.items():
        values[lead].update(fields)
    return values


def _p_wave(evaluable=True):
    return {lead: {"fiducial_evaluable": float(evaluable)} for lead in LEAD_NAMES_12}


def _ecg():
    return np.zeros((600, 12))


def _early_lateral_r():
    """A narrow normal V5/V6 R wave peaking 20 ms after QRS onset."""
    ecg = _ecg()
    for lead in ("V5", "V6"):
        ecg[100:120, LEAD_MAP[lead]] = 0.2
        ecg[110, LEAD_MAP[lead]] = 1.0
    return ecg


def test_frontal_axis_difference_is_folded_and_abstains_without_an_axis():
    assert classical._frontal_axis_difference(60.0, -90.0) == 150.0
    assert classical._frontal_axis_difference(170.0, -170.0) == 20.0
    assert classical._frontal_axis_difference(None, 10.0) is None
    assert classical._frontal_axis_difference(float("nan"), 10.0) is None


def test_bounded_points_decide_only_when_unknown_components_cannot_change_it():
    assert classical._bounded_points([(3, True), (2, True)], threshold=5) == (5.0, True)
    # 2 known, 3 open: could reach 5 -> undecided, and no point value.
    assert classical._bounded_points([(3, None), (1, True), (1, True)], threshold=5) == (None, None)
    # 1 known, 3 open: cannot reach 5 -> negative even though incomplete.
    assert classical._bounded_points([(3, None), (1, True), (1, False)], threshold=5) == (None, False)
    # 5 known, 3 open: positive whatever the open component scores.
    assert classical._bounded_points([(3, True), (2, True), (3, None)], threshold=5) == (None, True)


def _romhilt(ecg, qrs, st, *, machine_axis=0.0, st_evaluable=True, qrs_ms=100.0, p_status="present"):
    return classical.ClassicalECGCriteria()._romhilt_estes(
        ecg, qrs, _p_wave(), st, _beat(qrs_ms=qrs_ms, p_status=p_status), np.zeros(12), FS,
        qrs_ms=qrs_ms, machine_qrs_axis_deg=machine_axis, paced_or_lbbb=False,
        qrs_evaluable=True, t_evaluable=True, st_evaluable=st_evaluable,
    )["romhilt_estes_points"]


def test_romhilt_estes_scores_every_component_from_its_source_definition():
    ecg = _ecg()
    # P-terminal negativity in V1: -0.15 mV over the last 25 samples = 50 ms.
    ecg[56:81, LEAD_MAP["V1"]] = -0.15
    # R peak in V5 at 30 samples after QRS onset = 60 ms (>=50 ms).
    ecg[100:150, LEAD_MAP["V5"]] = 0.2
    ecg[130, LEAD_MAP["V5"]] = 3.2
    qrs = _qrs(V5={"r_height": 3.2, "s_depth": 0.1})
    st = _st(V5={"st60": -0.10, "t_peak": -0.20})
    full = _romhilt(ecg, qrs, st, machine_axis=-45.0)
    # voltage 3 + strain 3 + atrial 3 + axis 2 + QRS>=90 1 + intrinsicoid 1
    assert full.value == 13.0 and full.positive is True and full.evaluable

    plain = _romhilt(_early_lateral_r(), _qrs(), _st(), machine_axis=60.0, qrs_ms=80.0)
    assert plain.value == 0.0 and plain.positive is False


def test_romhilt_estes_bounds_an_unmeasurable_strain_arm():
    ecg = _ecg()
    ecg[100:150, LEAD_MAP["V5"]] = 0.2
    ecg[130, LEAD_MAP["V5"]] = 1.0
    # QRS 100 ms (1) + intrinsicoid (1) known, strain (3) unknown: 2..5 -> undecided.
    undecided = _romhilt(ecg, _qrs(), _st(), st_evaluable=False)
    assert undecided.value is None and undecided.positive is None and not undecided.evaluable
    # Narrow QRS: 1 known + 3 open < 5 -> negative without the strain arm.
    negative = _romhilt(ecg, _qrs(), _st(), st_evaluable=False, qrs_ms=80.0)
    assert negative.value is None and negative.positive is False and negative.evaluable


def test_romhilt_estes_absent_p_wave_scores_no_atrial_points():
    result = _romhilt(_early_lateral_r(), _qrs(), _st(), machine_axis=60.0, qrs_ms=80.0,
                      p_status="absent")
    assert result.value == 0.0


def _daniel(ecg, st, q_metrics, *, rr_ms=500.0, qrs_ms=120.0):
    return classical.ClassicalECGCriteria()._daniel_pe_score(
        ecg, q_metrics, st, _beat(qrs_ms=qrs_ms), np.zeros(12), FS,
        qrs_ms=qrs_ms, rr_ms=rr_ms, paced_or_lbbb=False,
        qrs_evaluable=True, t_evaluable=True,
    )["daniel_pe_score"]


def _q_metrics(depth_iii=0.0):
    values = {lead: {"duration_ms": 0.0, "depth_mv": 0.0} for lead in LEAD_NAMES_12}
    values["III"] = {"duration_ms": 30.0, "depth_mv": depth_iii}
    return values


def test_daniel_score_reproduces_the_published_component_table():
    ecg = _ecg()
    lead_i, v1 = LEAD_MAP["I"], LEAD_MAP["V1"]
    # Lead I: R then S of -0.30 mV. V1: r, S, then terminal R' of 0.40 mV.
    ecg[100:115, lead_i] = 0.8
    ecg[115:159, lead_i] = -0.30
    ecg[100:105, v1] = 0.1
    ecg[105:130, v1] = -0.6
    ecg[130:159, v1] = 0.4
    st = _st(
        V1={"t_min": -0.25, "t_max": 0.0},   # >2 mm -> 2
        V2={"t_min": -0.15, "t_max": 0.0},   # 1-2 mm -> 2
        V3={"t_min": -0.05, "t_max": 0.0},   # <1 mm -> 1
        V4={"t_min": -0.10, "t_max": 0.02},
        III={"t_min": -0.20, "t_max": 0.0},
    )
    result = _daniel(ecg, st, _q_metrics(depth_iii=0.20))
    # rate 2 + complete RBBB 3 + V1-V4 4 + V1 2 + V2 2 + V3 1 + Q(III) 1
    # + T(III) 1 + S1Q3T3 2 = 18
    assert result.value == 18.0 and result.positive is True

    incomplete = _daniel(ecg, st, _q_metrics(depth_iii=0.20), qrs_ms=104.0)
    assert incomplete.value == 17.0

    normal = _daniel(_ecg(), _st(), _q_metrics(), rr_ms=900.0, qrs_ms=90.0)
    assert normal.value == 0.0 and normal.positive is False


def test_daniel_score_without_rate_stays_bounded():
    result = _daniel(_ecg(), _st(), _q_metrics(), rr_ms=None, qrs_ms=90.0)
    # Only the 2 rate points are open: 0..2 < 10 -> negative, no point value.
    assert result.value is None and result.positive is False


def test_p_terminal_negativity_measures_the_terminal_run():
    ecg = _ecg()
    ecg[40:55, LEAD_MAP["V1"]] = 0.1
    ecg[56:81, LEAD_MAP["V1"]] = -0.12
    depth, duration = classical._p_terminal_negativity(ecg, _beat(), "V1", np.zeros(12), FS)
    assert depth == 0.12 and duration == 50.0
    flat = classical._p_terminal_negativity(_ecg(), _beat(), "V1", np.zeros(12), FS)
    assert flat == (0.0, 0.0)


def test_reference_scores_report_indices_and_the_avr_t_wave():
    criteria = classical.ClassicalECGCriteria()
    beat = _beat(qrs_ms=100.0, qt_ms=400.0)
    qrs = _qrs()
    results = criteria._reference_scores(
        _ecg(), qrs, _q_metrics(), _p_wave(), _st(aVR={"t_peak": 0.08}), beat, np.zeros(12), FS,
        rr_ms=800.0, machine_qrs_axis_deg=60.0, machine_t_axis_deg=40.0,
        paced_or_lbbb=False, qrs_evaluable=True, p_evaluable=True,
        t_evaluable=True, st_evaluable=True,
    )
    assert results["t_amplitude_avr"].positive is True
    assert results["iceb_qt_qrs_ratio"].value == 4.0
    assert results["tpeak_tend_qt_ratio"].value == 80.0 / 400.0
    assert results["frontal_qrs_t_angle_device"].value == 20.0
    assert results["qrs_voltage_sum_12lead"].value == 18.0
    assert results["qrs_voltage_duration_product_12lead"].value == 1800.0
    # P 30..80 (100 ms), PR segment 80..100 (40 ms).
    assert results["macruz_index"].value == 2.5
    for key in ("iceb_qt_qrs_ratio", "tpeak_tend_qt_ratio", "frontal_qrs_t_angle_device",
                "qrs_voltage_sum_12lead", "macruz_index"):
        assert results[key].positive is None

    wide = criteria._reference_scores(
        _ecg(), qrs, _q_metrics(), _p_wave(), _st(aVR={"t_peak": 0.08}),
        _beat(qrs_ms=130.0), np.zeros(12), FS,
        rr_ms=800.0, machine_qrs_axis_deg=None, machine_t_axis_deg=40.0,
        paced_or_lbbb=False, qrs_evaluable=True, p_evaluable=True,
        t_evaluable=True, st_evaluable=True,
    )
    # The source cohort excluded QRS >120 ms; the recorder axis is missing.
    assert not wide["t_amplitude_avr"].evaluable
    assert not wide["frontal_qrs_t_angle_device"].evaluable
