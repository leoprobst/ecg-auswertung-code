"""The QRS-T coupling extensions: beat-level Fréchet and native-lead mirror symmetry."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import distance

from vcg_biomarker.core.biomarkers import qrs_t_coupling_extension_biomarkers as m
from vcg_biomarker.core.biomarkers.discovery_v3_biomarkers import _discrete_frechet


def test_batched_frechet_equals_the_representative_recurrence():
    rng = np.random.default_rng(2)
    curves = [(rng.normal(size=(64, 3)), rng.normal(size=(64, 3))) for _ in range(5)]
    cost = np.stack([distance.cdist(q, t) for q, t in curves])
    batched = m.discrete_frechet_batch(cost)
    for index, (q, t) in enumerate(curves):
        assert batched[index] == pytest.approx(_discrete_frechet(distance.cdist(q, t)))
    # A non-finite cost matrix abstains for that beat only.
    cost[2, 10, 10] = np.nan
    again = m.discrete_frechet_batch(cost)
    assert np.isnan(again[2]) and np.isfinite(again[[0, 1, 3, 4]]).all()


def _point(sample, confidence=1.0):
    return SimpleNamespace(sample=sample, confidence=confidence)


def _beat(r, fs=500, cluster=0, t_confidence=1.0):
    return SimpleNamespace(
        morphology_cluster_id=cluster,
        r_onset=_point(r - int(.04 * fs)),
        r_offset=_point(r + int(.06 * fs)),
        t_onset=_point(r + int(.20 * fs), t_confidence),
        t_offset=_point(r + int(.40 * fs), t_confidence),
        r_peak=_point(r),
    )


def _record(n_beats=5, fs=500, clusters=None, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    length = fs * (n_beats + 1)
    xyz = np.zeros((length, 3))
    beats = []
    for k in range(n_beats):
        r = fs // 2 + k * fs
        cluster = 0 if clusters is None else clusters[k]
        beat = _beat(r, fs, cluster)
        q0, q1 = beat.r_onset.sample, beat.r_offset.sample
        t0, t1 = beat.t_onset.sample, beat.t_offset.sample
        phase = np.linspace(0, np.pi, q1 - q0 + 1)
        xyz[q0:q1 + 1] = np.column_stack((np.sin(phase), 0.4 * np.sin(2 * phase), 0.2 * np.sin(phase)))
        phase_t = np.linspace(0, np.pi, t1 - t0 + 1)
        scale = 1.0 + jitter * rng.normal()
        xyz[t0:t1 + 1] = scale * np.column_stack((-0.3 * np.sin(phase_t), 0.1 * np.sin(phase_t), 0.05 * np.sin(2 * phase_t)))
        beats.append(beat)
    return SimpleNamespace(
        fs=fs,
        _record_vcg_xyz=xyz,
        ecg_12lead=None,
        delineation=SimpleNamespace(beats=beats),
        is_segment_evaluable=lambda segment: True,
    )


def test_beat_median_needs_three_primary_beats_and_counts_them():
    ctx = _record(n_beats=5)
    assert m._frechet_beat_count(ctx).value == 5
    median = m._frechet_beat_median(ctx)
    assert median.value is not None and median.value > 0
    assert "contributing_beats=5" in median.notes
    # Identical beats: no scatter.
    assert m._frechet_beat_mad(ctx).value == pytest.approx(0.0)
    # Secondary-morphology beats are not summarised.
    mixed = _record(n_beats=5, clusters=[0, 1, 1, 1, 0])
    assert m._frechet_beat_count(mixed).value == 2
    assert m._frechet_beat_median(mixed).value is None
    assert "fewer than 3" in m._frechet_beat_median(mixed).notes
    # Without a record VCG the count itself abstains rather than reading zero.
    empty = _record(n_beats=3)
    empty._record_vcg_xyz = None
    assert m._frechet_beat_count(empty).value is None
    # An unreadable representative segment must not hide usable record beats.
    unreadable = _record(n_beats=3)
    unreadable.is_segment_evaluable = lambda segment: segment != "t"
    assert m._frechet_beat_count(unreadable).value == 3
    assert m._frechet_beat_median(unreadable).value is not None
    unreadable_without_beats = _record(n_beats=0)
    unreadable_without_beats.is_segment_evaluable = lambda segment: False
    assert m._frechet_beat_count(unreadable_without_beats).value is None


def test_beat_scatter_is_seen_when_the_t_loop_changes_between_beats():
    steady = m._frechet_beat_mad(_record(n_beats=8, jitter=0.0))
    # Scaling one curve leaves the normalised Fréchet distance unchanged, so
    # the scatter must come from a change of shape, not of size.
    scaled = m._frechet_beat_mad(_record(n_beats=8, jitter=0.5, seed=4))
    assert steady.value == pytest.approx(0.0)
    assert scaled.value == pytest.approx(0.0, abs=1e-9)
    ctx = _record(n_beats=8)
    xyz = ctx._record_vcg_xyz
    for k, beat in enumerate(ctx.delineation.beats):
        if k % 2:
            t0, t1 = beat.t_onset.sample, beat.t_offset.sample
            phase = np.linspace(0, np.pi, t1 - t0 + 1)
            xyz[t0:t1 + 1, 2] += 0.6 * np.sin(3 * phase)
    assert m._frechet_beat_mad(ctx).value > 0.01


def test_native_lead_mirror_symmetry_is_baseline_free_and_bounded():
    fs = 500
    length = fs
    signal = np.zeros((length, 12))
    q0, q1, t0, t1 = 100, 150, 200, 350
    phase = np.linspace(0, np.pi, q1 - q0 + 1)
    phase_t = np.linspace(0, np.pi, t1 - t0 + 1)
    for lead in range(12):
        signal[q0:q1 + 1, lead] = np.sin(phase) * (1 + 0.1 * lead)
        signal[t0:t1 + 1, lead] = -0.4 * np.sin(phase_t) * (1 + 0.1 * lead)
    ctx = SimpleNamespace(
        ecg_12lead=signal,
        bounds=lambda segment: {"qrs": (q0, q1), "t": (t0, t1)}[segment],
        is_segment_evaluable=lambda segment: True,
    )
    value = m._mirror_symmetry_direct_8lead(ctx)
    assert value.value is not None and 0.0 <= value.value < 1.0
    # A perfect mirror (T = -QRS shape) gives zero; adding an offset per lead
    # changes nothing because both curves are centred.
    shifted = SimpleNamespace(
        ecg_12lead=signal + np.arange(12) * 0.3,
        bounds=ctx.bounds,
        is_segment_evaluable=ctx.is_segment_evaluable,
    )
    assert m._mirror_symmetry_direct_8lead(shifted).value == pytest.approx(value.value)
    missing = SimpleNamespace(ecg_12lead=None, bounds=ctx.bounds, is_segment_evaluable=ctx.is_segment_evaluable)
    assert m._mirror_symmetry_direct_8lead(missing).value is None


def test_extension_markers_are_registered_as_discovery_reserve_only():
    from vcg_biomarker.core.biomarkers import BiomarkerRegistry
    from vcg_biomarker.mimic.feature_roles import feature_profile
    from vcg_biomarker.mimic.marker_plausibility import declared_range

    for spec in m.QRS_T_COUPLING_EXTENSION_SPECS:
        assert spec.name in BiomarkerRegistry.available()
        profile = feature_profile(spec.name)
        assert profile.post_outcome_hypothesis
        assert profile.discovery_eligible
        assert not profile.core_sentinel_eligible
        assert not profile.combination_eligible
        assert declared_range(spec.name, spec.unit) is not None
