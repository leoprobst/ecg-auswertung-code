"""Audit 2026-09-23, classical criteria S3-S5."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vcg_biomarker.core.data_loader import LEAD_NAMES_12
from vcg_biomarker.mimic import classical


def _qrs(**overrides):
    values = {lead: {"r_height": 1.0, "s_depth": 0.5, "peak_to_peak": 1.5} for lead in LEAD_NAMES_12}
    for lead, fields in overrides.items():
        values[lead].update(fields)
    return values


def _lvh(qrs, sex="M", qrs_ms=100.0):
    beat = SimpleNamespace(qrs_duration_ms=lambda fs: qrs_ms)
    return classical.ClassicalECGCriteria()._lvh_criteria(qrs, beat, 500, sex, excluded=False)


def test_s3_missing_lead_makes_voltage_criteria_unknown_not_negative():
    complete = _lvh(_qrs())
    assert all(item.positive is False and item.evaluable for item in complete.values())
    # Missing V6: Sokolow and Peguero... Sokolow needs V6, Cornell does not.
    missing_v6 = _lvh(_qrs(V6={"r_height": np.nan, "s_depth": np.nan}))
    assert missing_v6["sokolow_lyon"].positive is None
    assert not missing_v6["sokolow_lyon"].evaluable
    assert missing_v6["peguero_lo_presti"].positive is None  # deepest S over all leads
    assert missing_v6["cornell_voltage"].evaluable
    missing_v3 = _lvh(_qrs(V3={"s_depth": np.nan}))
    assert missing_v3["cornell_voltage"].positive is None
    assert missing_v3["cornell_product"].positive is None
    # Without a QRS duration the product is unknown, not negative.
    no_duration = _lvh(_qrs(), qrs_ms=None)
    assert no_duration["cornell_product"].positive is None
    assert not no_duration["cornell_product"].evaluable
    # A present positive stays positive.
    tall = _lvh(_qrs(V1={"s_depth": 2.0}, V5={"r_height": 2.0}))
    assert tall["sokolow_lyon"].positive is True


def test_s3_ratio_of_a_missing_amplitude_is_unknown():
    assert np.isnan(classical._safe_ratio(np.nan, 1.0))
    assert np.isnan(classical._safe_ratio(1.0, np.inf))
    assert classical._safe_ratio(2.0, 1e-12) == classical.RATIO_LIMIT
    assert classical._t_inversion_rs_component({"t_peak": -0.3, "r_to_s_ratio": np.nan}) is None


def _q(values, fs=500):
    lead = np.asarray(values, dtype=float)
    beat = SimpleNamespace(r_onset=SimpleNamespace(sample=0),
                           r_offset=SimpleNamespace(sample=len(lead) - 1))
    return classical._initial_q_metric(lead, beat, 0.0, fs)


def test_s4_negative_baseline_residue_before_a_late_onset_is_no_q():
    # 20 ms of -5 uV residue (one quantisation step), then a plain R.
    residue = [-0.005] * 10 + [0.3, 1.0, 0.4, 0.0]
    assert _q(residue)["duration_ms"] == 0
    # A real Q: 30 ms at -0.15 mV from the lead's own onset.
    real = [0.0, 0.0] + [-0.15] * 15 + [0.5, 1.0, 0.0]
    metric = _q(real)
    assert metric["duration_ms"] == 30.0
    assert metric["depth_mv"] == 0.15


def test_s4_v2_v3_q_needs_a_visible_depth():
    qrs = _qrs()
    shallow = {lead: {"duration_ms": 0.0, "depth_mv": 0.0} for lead in LEAD_NAMES_12}
    shallow["V2"] = {"duration_ms": 30.0, "depth_mv": 0.03}
    deep = dict(shallow, V2={"duration_ms": 30.0, "depth_mv": 0.08})
    criteria = classical.ClassicalECGCriteria()
    assert criteria._scar_criteria(qrs, shallow, excluded=False)["pathological_q_waves"].positive is False
    assert criteria._scar_criteria(qrs, deep, excluded=False)["pathological_q_waves"].positive is True


def test_s5_lbbb_requires_delayed_lateral_r_peak():
    beat = SimpleNamespace(qrs_duration_ms=lambda fs: 150,
                           r_onset=SimpleNamespace(sample=0), r_offset=SimpleNamespace(sample=74))
    qrs = _qrs(V1={"r_height": 0.1, "s_depth": 1.5}, V6={"s_depth": 0.02})
    q = {lead: {"duration_ms": 0.0, "depth_mv": 0.0} for lead in LEAD_NAMES_12}
    outcomes = []
    for peak_sample in (20, 40):          # 40 ms against 80 ms at 500 Hz
        signal = np.zeros((75, 12))
        for lead in (10, 11):             # V5, V6: broad R peaking at the sample
            signal[1:70, lead] = np.interp(np.arange(1, 70), [1, peak_sample, 69], [0.1, 1.5, 0.1])
        result = classical.ClassicalECGCriteria()._conduction_criteria(
            signal, qrs, q, beat, np.zeros(12), 500,
            paced=False, qrs_evaluable=True, p_evaluable=False)["lbbb_pattern"]
        assert "R-peak time >60 ms in V5 and V6" in result.definition
        outcomes.append(result.positive)
    assert outcomes == [False, True]


def test_s2_comparators_use_device_intervals_and_keep_own_for_sensitivity():
    import pandas as pd

    from vcg_biomarker.mimic.delineation_sensitivity import (
        apply_device_interval_comparators,
        evaluate_interval_criterion_sensitivity,
    )

    frame = pd.DataFrame({
        "label": [1, 0, 1, 0],
        # Own QTcF 470 (below 480), device QT 440 at 60/min -> QTcF 440 ...
        "classical_qtc_gt_480": [470.0, 470.0, 470.0, 470.0],
        "classical_qtc_gt_480_positive": [0.0, 0.0, 0.0, 0.0],
        "machine_qt_interval_ms": [500.0, 440.0, np.nan, 490.0],
        "machine_heart_rate_bpm": [60.0, 60.0, 60.0, 60.0],
        # Own QRS 124 (wide), device 110 (narrow); last row has no device QRS.
        "classical_qrs_widening_ge_120": [124.0, 124.0, 124.0, 124.0],
        "classical_qrs_widening_ge_120_positive": [1.0, 1.0, 1.0, 1.0],
        "machine_qrs_duration_ms": [110.0, 130.0, 110.0, np.nan],
        "classical_cornell_product": [248.0, 248.0, 248.0, 248.0],
        "classical_cornell_product_positive": [1.0, 1.0, 1.0, np.nan],
        "classical_qtc_fridericia": [470.0] * 4,
    })
    device = apply_device_interval_comparators(frame)
    assert device["classical_qtc_gt_480_positive"].tolist()[:2] == [1.0, 0.0]
    # No device QT: the own value and rule stay.
    assert device.loc[2, "classical_qtc_gt_480"] == 470.0
    assert device.loc[2, "classical_qtc_gt_480_interval_source"] == "own_delineation"
    assert device["classical_qrs_widening_ge_120_positive"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert device.loc[3, "classical_qrs_widening_ge_120_interval_source"] == "own_delineation"
    # Cornell: voltage 2.0 mV (248/124) times device QRS.
    assert device.loc[0, "classical_cornell_product"] == 220.0
    assert device["classical_cornell_product_positive"].tolist()[:2] == [0.0, 1.0]
    assert np.isnan(device.loc[3, "classical_cornell_product_positive"])
    assert device.loc[0, "classical_qtc_fridericia"] == 500.0
    # Own values are kept for the sensitivity arm, which now reads them.
    assert device["own_classical_qtc_gt_480"].tolist() == [470.0] * 4
    sensitivity = evaluate_interval_criterion_sensitivity(
        device, endpoint_id="e", evaluation_split="validation")
    qtc = sensitivity.set_index("criterion_id").loc["qtc_gt_480"]
    assert qtc["primary_comparator_intervals"] == "device"
    assert qtc["own_positive_rate"] == 0.0
    # The input frame is not modified.
    assert "own_classical_qtc_gt_480" not in frame


def test_s6_limb_lead_reversal_is_excluded_before_any_marker(monkeypatch):
    from vcg_biomarker.mimic import feature_record

    quality = SimpleNamespace(score=0.8, tier="medium", usable=True,
                              reasons=("limb_lead_reversal_or_dextrocardia_suspected",),
                              limb_lead_reversal_suspected=True)
    loaded = SimpleNamespace(quality=quality, original_sampling_rate=500, repair=None,
                             signal_millivolts=np.zeros((5000, 12)))

    class Loader:
        def __init__(self, **_):
            pass

        def load(self, *_args, **_kwargs):
            return loaded

        def prepare_paths(self, *_args, **_kwargs):
            raise AssertionError("an excluded recording must not be processed")

    monkeypatch.setattr(feature_record, "MimicWaveformLoader", Loader)
    row = {"subject_id": 1, "study_id": 2, "split": "train", "ecg_time": None,
           "waveform_path": "x"}
    result = feature_record._extract_record_worker((row, {"powerline_hz": 60.0}))
    assert result["extraction_status"] == "lead_placement_excluded"
    assert result["error_type"] == "LeadPlacementExclusion"
    assert "limb_lead_reversal" in result["error_message"]


def test_s7_hf_band_snr_is_measured_against_tp_noise():
    import pandas as pd

    from vcg_biomarker.mimic.hf_qrs_quality import (
        band_limited_quantisation_rms_uv,
        hf_qrs_signal_quality,
    )
    from vcg_biomarker.mimic.validation import _hf_qrs_snr_fields

    fs = 500
    rng = np.random.default_rng(4)
    xyz = rng.normal(scale=0.002, size=(5000, 3))       # 2 uV white noise
    point = lambda sample: SimpleNamespace(sample=sample)
    beats = []
    for start in range(200, 4600, 500):
        t = np.arange(50) / fs
        xyz[start:start + 50] += 0.05 * np.sin(2 * np.pi * 110 * t)[:, None]  # 110 Hz burst
        beats.append(SimpleNamespace(p_onset=point(start - 80), r_onset=point(start),
                                     r_offset=point(start + 49), t_offset=point(start + 250)))
    quality = hf_qrs_signal_quality(xyz, beats, fs)
    assert quality["context_hf_qrs_snr_beat_count"] == len(beats)
    assert quality["context_hf_qrs_single_beat_snr_db"] > 15
    assert quality["context_hf_qrs_averaged_snr_db"] == pytest_approx(
        quality["context_hf_qrs_single_beat_snr_db"] + 10 * np.log10(len(beats)))
    assert 0 < band_limited_quantisation_rms_uv(fs) < 2
    # No TP window, no noise estimate: abstain rather than guess.
    no_tp = [SimpleNamespace(p_onset=None, r_onset=b.r_onset, r_offset=b.r_offset,
                             t_offset=b.t_offset) for b in beats]
    assert np.isnan(hf_qrs_signal_quality(xyz, no_tp, fs)["context_hf_qrs_averaged_snr_db"])
    frame = pd.DataFrame({"context_hf_qrs_averaged_snr_db": [5.0, 12.0, 20.0, np.nan]})
    fields = _hf_qrs_snr_fields(frame, "terminal_qrs_hf_energy_fraction",
                                np.array([True, True, True, True]))
    assert fields["hf_qrs_band_snr_median_db"] == 12.0
    assert fields["hf_qrs_band_snr_measured"] == 3
    assert fields["hf_qrs_band_reportable"] is True
    assert _hf_qrs_snr_fields(frame, "qrs_duration_ms", np.ones(4, bool)) == {}


def pytest_approx(value):
    import pytest

    return pytest.approx(value)


def test_s8_software_notch_only_where_the_device_did_not_notch():
    from vcg_biomarker.core.preprocessing import prepare_signal_paths
    from vcg_biomarker.mimic.signals import device_notched_at

    assert device_notched_at("60 Hz notch Baseline filter", 60.0)
    assert not device_notched_at("50 Hz notch Baseline filter", 60.0)
    assert not device_notched_at("<not specified>", 60.0)
    assert not device_notched_at("Baseline filter", 60.0)
    assert not device_notched_at(None, 60.0)
    fs = 500
    t = np.arange(5000) / fs
    ecg = np.tile((0.1 * np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 1.2 * t))[:, None], (1, 12))
    notched = prepare_signal_paths(ecg, fs, powerline=60.0)
    kept = prepare_signal_paths(ecg, fs, powerline=60.0, notch=False)
    residual = lambda signal: np.abs(np.fft.rfft(signal[:, 0]))[600]  # 60 Hz bin
    assert residual(notched["diagnostic"]) < 0.05 * residual(kept["diagnostic"])
    default = prepare_signal_paths(ecg, fs, powerline=60.0, notch=True)
    np.testing.assert_array_equal(default["diagnostic"], notched["diagnostic"])
