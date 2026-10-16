from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import a108_robustness as m
from vcg_biomarker.core.delineation.base import RhythmClass


def signal(fs=500):
    x = np.zeros((fs, 12))
    q = (int(.2 * fs), int(.3 * fs))
    t = (q[1], int(.7 * fs))
    x[q[0]:q[1], 0] = 1
    x[t[0]:t[1] + 1, 1] = .5
    return x, q, t, (0, int(.1 * fs))


def test_representative_window_provenance_is_taken_from_computation():
    x, q, t, pr = signal()
    ctx = SimpleNamespace(fs=500, ecg_12lead=x, beat=SimpleNamespace(), _primitive_cache={},
        delineation=SimpleNamespace(rhythm=SimpleNamespace(discrete_p_not_evaluable=True)),
        is_segment_evaluable=lambda segment: segment != 'pr',
        bounds=lambda segment: {'qrs': q, 't': t, 'pr': pr}[segment])
    value = m.measure(ctx, kind='representative', field='preqrs_area')
    assert value.value is not None
    assert value.measurement_windows['baseline'] == m.preqrs_baseline(q[0], 500)
    assert value.measurement_windows['qrs'] == q
    assert value.measurement_windows['baseline_estimator'] == 'arithmetic_mean'
    assert value.measurement_windows['baseline_kind'] == 'preqrs'
    assert m.measure(ctx, kind='representative', field='tp_area').measurement_windows is None
    from vcg_biomarker.core.biomarkers import BiomarkerRegistry
    marker = BiomarkerRegistry.get('qrs_t_area_angle_direct_8lead_preqrs_deg')
    exported = marker._result(value.value, value.confidence, value.notes,
                              value.per_lead, value.measurement_windows)
    assert exported.measurement_windows == value.measurement_windows


@pytest.mark.parametrize("fs", [250, 500, 1000])
def test_native_vectors_offset_scale_and_resampling(fs):
    x, q, t, b = signal(fs)
    base, status = m.native_metrics(x, q, t, b, fs)
    changed, _ = m.native_metrics(3 * x + np.arange(12), q, t, b, fs)
    assert status == "ok"
    assert 88 < base["area"] <= 90
    assert base["early_late_angle"] == pytest.approx(0, abs=1e-6)
    assert base["late_early_ratio"] == pytest.approx(1)
    for key in ("area", "integral", "early_late_angle", "late_early_ratio"):
        assert changed[key] == pytest.approx(base[key], abs=1e-6)
    assert changed["stt_area"] == pytest.approx(3 * base["stt_area"])


@pytest.mark.parametrize("end", [348, 349, 350, 351])
def test_exact_half_duration_no_duplicated_midpoint_mass(end):
    x, q, _, b = signal()
    x[q[1]:end + 1, 1] = .5
    result, _ = m.native_metrics(x, q, (q[1], end), b, 500)
    assert result["late_early_ratio"] == pytest.approx(1)


@pytest.mark.parametrize("failure", ["nan", "zero", "baseline_short", "baseline_noise", "boundaries"])
def test_native_abstains_instead_of_imputing(failure):
    x, q, t, b = signal()
    if failure == "nan":
        x[120, 0] = np.nan
    elif failure == "zero":
        x[:] = 0
    elif failure == "baseline_short":
        b = (0, 5)
    elif failure == "baseline_noise":
        x[:51] = np.random.default_rng(3).normal(size=(51, 12))
    else:
        t = (q[1] + 1, t[1])
    result, reason = m.native_metrics(x, q, t, b, 500)
    assert not result and reason != "ok"


def test_degenerate_leave_one_out_not_silently_zero():
    x, q, t, b = signal()
    result, _ = m.native_metrics(x, q, t, b, 500)
    assert result["drop_I"] is not None  # J sample contains second lead
    assert result["drop_II"] is None
    assert "drop_max_abs" not in result


def context(n=105, duration=130):
    return SimpleNamespace(
        fs=500,
        delineation=SimpleNamespace(n_samples=duration * 500,
            rhythm=SimpleNamespace(rhythm_class=RhythmClass.REGULAR, confidence=.9),
            beats=[SimpleNamespace(morphology_cluster_id=0, beat_type="sinus", beat_type_source="test")
                   for _ in range(n)]))


def test_minimum_counts_morphology_and_adjacency(monkeypatch):
    rows = {i: {"area": float(i), "rr_ms": float(800 + i)} for i in range(105)}
    monkeypatch.setattr(m, "_record", lambda ctx: rows)
    ctx = context()
    assert m.measure(ctx, kind="long_rmssd", field="area").value == 1
    assert m.measure(ctx, kind="long_rr_slope", field="area").value == pytest.approx(1)
    ctx.delineation.n_samples = 10 * 500
    assert m.measure(ctx, kind="long_rmssd", field="area").value is None
    ctx.delineation.n_samples = 130 * 500
    ctx.delineation.beats[10].morphology_cluster_id = 1
    assert m.measure(ctx, kind="long_rmssd", field="area").value is None
    ctx.delineation.beats[10].morphology_cluster_id = 0
    for i in range(2, 103, 2):
        rows.pop(i)
    assert m.measure(ctx, kind="long_rmssd", field="area").value is None
    rows.clear()
    rows.update({i: {"area": float(i)} for i in range(5)})
    assert m.measure(ctx, kind="median", field="area").value is None
    rows[5] = {"area": 5.}
    assert m.measure(ctx, kind="median", field="area").value == 2.5
    assert m.measure(ctx, kind="mad", field="area").value == 1.5


def test_postectopy_requires_reference_and_counts(monkeypatch):
    ctx = context(20)
    rows = {i: {"area": float(i)} for i in range(20)}
    monkeypatch.setattr(m, "_record", lambda ctx: rows)
    for i in (1, 5, 9):
        ctx.delineation.beats[i].morphology_cluster_id = 1
        rows.pop(i)
    assert m.measure(ctx, kind="post", field="area").value is None
    for i in (1, 5, 9):
        ctx.delineation.beats[i].beat_type = "pvc"
    assert m.measure(ctx, kind="post", field="area").value is not None
    ctx.delineation.beats[1].beat_type_source = ""
    assert m.measure(ctx, kind="post", field="area").value is None


def test_registry_discovery_only_unique_formulas_and_units():
    from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import ADVANCED_SPECS
    from vcg_biomarker.mimic.feature_roles import feature_profile
    rows = [s for s in ADVANCED_SPECS if s.family == "A108 explorative Messrobustheit"]
    assert len(rows) == 34
    assert len({r.name for r in rows}) == len(rows)
    for row in rows:
        role = feature_profile(row.name)
        assert not role.core_sentinel_eligible
        assert role.discovery_eligible
        assert not role.combination_eligible


def _beat(r_peak, fs=500, pr_confidence=0.0):
    onset, offset, t_end = r_peak - int(.04 * fs), r_peak + int(.06 * fs), r_peak + int(.40 * fs)
    thresholds = {"qrs": 1.0, "t": 1.0, "pr": pr_confidence}
    return SimpleNamespace(
        r_peak=SimpleNamespace(sample=r_peak),
        r_onset=SimpleNamespace(sample=onset),
        r_offset=SimpleNamespace(sample=offset),
        t_offset=SimpleNamespace(sample=t_end),
        p_offset=SimpleNamespace(sample=onset - int(.05 * fs)),
        morphology_cluster_id=0,
        segment_confidence=lambda w, _t=thresholds: _t[w],
    )


def _record_context(n_beats=4, fs=500, pr_confidence=0.0):
    length = fs * (n_beats + 1)
    x = np.zeros((length, 12))
    beats = []
    for k in range(n_beats):
        r = fs // 2 + k * fs
        b = _beat(r, fs, pr_confidence)
        x[b.r_onset.sample:b.r_offset.sample + 1, 0] = 1.0
        x[b.r_offset.sample:b.t_offset.sample + 1, 1] = 0.5
        beats.append(b)
    return SimpleNamespace(
        fs=fs,
        _record_ecg_12lead=x,
        _primitive_cache={},
        delineation=SimpleNamespace(
            n_samples=length,
            beats=beats,
            rhythm=SimpleNamespace(rhythm_class=RhythmClass.REGULAR, confidence=.9,
                                   discrete_p_not_evaluable=True),
        ),
    )


def test_preqrs_baseline_needs_no_p_wave_and_counts_its_beats():
    ctx = _record_context(n_beats=4)
    # No P offset with confidence, atrial fibrillation pattern: the PR path
    # has nothing, the pre-QRS path has every beat.
    assert m.measure(ctx, kind="count", field="area").value == 0
    assert m.measure(ctx, kind="count_preqrs", field="area").value == 4
    assert m.measure(ctx, kind="median", field="area").value is None
    median = m.measure(ctx, kind="median_preqrs", field="area")
    assert 88 < median.value <= 90
    assert "minimum=3" in median.notes
    assert m.measure(ctx, kind="mad_preqrs", field="area").value == pytest.approx(0.0)
    assert m.measure(_record_context(n_beats=2), kind="median_preqrs", field="area").value is None


def test_beat_windows_are_opt_in_and_preserve_summary():
    ctx = _record_context(n_beats=4)
    standard = m.measure(ctx, kind='median_preqrs', field='area')
    assert standard.measurement_windows is None
    captured = _record_context(n_beats=4)
    captured._record_acquisition = {'lab_capture_measurement_windows': True}
    result = m.measure(captured, kind='median_preqrs', field='area')
    assert result.value == standard.value
    contract = result.measurement_windows
    assert contract['eligible_beats'] == contract['stored_beats'] == 4
    assert contract['truncated'] is False
    assert contract['signal_path'] == 'record_feature_context_ecg_12lead'
    assert [item['beat_index'] for item in contract['items']] == [0, 1, 2, 3]
    assert contract['items'][0]['baseline'] == m.preqrs_baseline(captured.delineation.beats[0].r_onset.sample, captured.fs)
    assert contract['reason_counts'] == {'included': 4}
    assert len(contract['beat_audit']) == 4


@pytest.mark.parametrize('failure,reason', [
    ('cluster', 'not_primary_morphology_cluster'),
    ('quality', 'segment_quality_below_threshold'),
    ('baseline', 'baseline_short_or_unstable'),
    ('zero', 'degenerate_area_vector'),
])
def test_beat_exclusion_audit_records_first_failing_gate(failure, reason):
    ctx = _record_context(n_beats=4)
    ctx._record_acquisition = {'lab_capture_measurement_windows': True}
    beat = ctx.delineation.beats[1]
    if failure == 'cluster':
        beat.morphology_cluster_id = 1
    elif failure == 'quality':
        beat.segment_confidence = lambda wave: 0.
    elif failure == 'baseline':
        start, end = m.preqrs_baseline(beat.r_onset.sample, ctx.fs)
        ctx._record_ecg_12lead[start:end+1] = np.random.default_rng(2).normal(size=(end-start+1, 12))
    else:
        ctx._record_ecg_12lead[beat.r_onset.sample:beat.t_offset.sample+1] = 0.
    result = m.measure(ctx, kind='median_preqrs', field='area')
    audit = result.measurement_windows
    assert audit['reason_counts'] == {'included': 3, reason: 1}
    assert audit['beat_audit'][1]['reason'] == reason
    assert audit['beat_audit'][1]['included'] is False
    assert [item['beat_index'] for item in audit['items']] == [0, 2, 3]


def test_global_rhythm_rejection_is_not_normal_or_empty_audit():
    ctx = _record_context(n_beats=4)
    ctx._record_acquisition = {'lab_capture_measurement_windows': True}
    result = m.measure(ctx, kind='median', field='area')
    assert result.value is None
    assert result.measurement_windows['reason_counts'] == {'rhythm_no_discrete_p': 4}
    assert result.measurement_windows['eligible_beats'] == 0


@pytest.mark.parametrize("fs", [250, 500, 1000])
def test_preqrs_window_is_a_fixed_physical_duration(fs):
    onset = fs  # one second into the record
    start, end = m.preqrs_baseline(onset, fs)
    assert (end - start) / fs == pytest.approx(0.024, abs=1.5 / fs)
    assert (onset - end) / fs == pytest.approx(0.004, abs=1.0 / fs)
    assert m.preqrs_baseline(5, fs) is None


def test_preqrs_specifications_sit_beside_the_pr_definitions():
    from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import ADVANCED_SPECS
    names = {s.name for s in ADVANCED_SPECS if s.family == "A108 explorative Messrobustheit"}
    for method in ("area", "integral"):
        assert f"qrs_t_{method}_angle_direct_8lead_preqrs_deg" in names
        assert f"qrs_t_{method}_angle_direct_8lead_beat_median_preqrs_deg" in names
        assert f"qrs_t_{method}_angle_direct_8lead_beat_median_deg" in names
    assert {"direct8_pr_eligible_beat_count", "direct8_preqrs_eligible_beat_count"} <= names
    contract = m.measurement_contract()
    assert contract["preqrs_beat_summary_minimum"] == 3
    assert contract["beat_summary_minimum"] == 6
