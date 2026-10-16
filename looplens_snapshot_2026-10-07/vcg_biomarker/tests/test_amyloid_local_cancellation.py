"""Temporal signal tests; no disease labels or tissue inference."""
from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import local_qrs_cancellation


def reference(points, fs):
    """Independent explicit integration of each window, including fractional ends."""
    span = .024 * fs
    points = np.asarray(points, dtype=float)
    magnitudes = np.linalg.norm(points, axis=1)
    numerator = denominator = 0.
    for start in range(int(np.floor(len(points) - 1 - span)) + 1):
        end = start + span
        knots = np.unique(np.r_[np.arange(start, np.floor(end) + 1), end])
        xyz = np.column_stack([np.interp(knots, np.arange(len(points)), points[:, axis]) for axis in range(3)])
        mag = np.interp(knots, np.arange(len(points)), magnitudes)
        numerator += np.linalg.norm(np.trapezoid(xyz, x=knots, axis=0))
        denominator += np.trapezoid(mag, x=knots)
    return 1 - numerator / denominator


@pytest.mark.parametrize("fs", [250., 360., 500., 1000.])
def test_prefix_integrals_match_explicit_windows(fs):
    points = np.random.default_rng(505).normal(size=(101, 3))
    assert local_qrs_cancellation(points, fs) == pytest.approx(reference(points, fs), abs=2e-14)


def test_fixed_direction_has_no_cancellation_even_with_amplitude_modulation():
    u = np.linspace(0., 1., 101)
    points = (1 + 3 * u ** 2)[:, None] * np.array([[1., 2., -1.]])
    assert local_qrs_cancellation(points, 500) == pytest.approx(0., abs=1e-14)


def test_alternating_directions_differ_from_slow_reversal_despite_equal_global_cancellation():
    fast = np.zeros((101, 3))
    fast[:, 0] = np.where(np.arange(101) % 2, -1., 1.)
    slow = np.zeros_like(fast)
    slow[:50, 0], slow[51:, 0] = 1., -1.
    np.testing.assert_allclose(np.trapezoid(fast, axis=0), np.trapezoid(slow, axis=0), atol=1e-14)
    assert local_qrs_cancellation(fast, 500) == pytest.approx(1.)
    assert local_qrs_cancellation(slow, 500) < .15


def test_common_gain_rotation_and_polarity_invariance_not_anisotropic_transfer():
    points = np.random.default_rng(17).normal(size=(61, 3)) + [1, 0, 0]
    rotation, _ = np.linalg.qr(np.random.default_rng(18).normal(size=(3, 3)))
    expected = local_qrs_cancellation(points, 500)
    assert local_qrs_cancellation(-1e200 * points @ rotation, 500) == pytest.approx(expected)
    assert local_qrs_cancellation(points * [10, 1, 1], 500) != pytest.approx(expected)


def test_physical_time_resampling_converges():
    def signal(fs):
        time = np.arange(round(.12 * fs) + 1) / fs
        return np.column_stack((np.cos(2 * np.pi * 12 * time), np.sin(2 * np.pi * 12 * time), time * 0))
    assert local_qrs_cancellation(signal(500), 500) == pytest.approx(local_qrs_cancellation(signal(1000), 1000), abs=.002)


def test_fixed_additive_noise_breaks_apparent_attenuation_invariance():
    noise = np.random.default_rng(512).normal(scale=.05, size=(101, 3))
    directed = np.tile([1., 0., 0.], (101, 1))
    # Gain invariance applies to the complete signal, not only its biological
    # component when the measurement noise floor stays fixed.
    assert local_qrs_cancellation(.01 * directed + noise, 500) > local_qrs_cancellation(directed + noise, 500) + .2


@pytest.mark.parametrize("points, fs", [(np.zeros((30, 3)), 500), (np.ones((11, 3)), 500),
    (np.ones((30, 2)), 500), (np.full((30, 3), np.nan), 500),
    (np.ones((30, 3)), 100), (np.ones((30, 3)), np.inf), (np.ones((30, 3)), 0)])
def test_invalid_or_unobservable_support_abstains(points, fs):
    with pytest.raises(ValueError):
        local_qrs_cancellation(points, fs)


def test_context_registry_freeze_and_cache(monkeypatch):
    from vcg_biomarker.core.biomarkers import pathology_mechanism_biomarkers as module
    from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
    from vcg_biomarker.mimic.feature_roles import feature_profile
    point = lambda n: SimpleNamespace(sample=n, confidence=1.)
    signal = np.random.default_rng(66).normal(size=(61, 3))
    ctx = SimpleNamespace(beat=SimpleNamespace(r_onset=point(0), r_offset=point(60)),
        fs=500, _primitive_cache={}, origin_corrected_vcg=lambda: signal)
    monkeypatch.setattr(module, "_context", lambda *args: ctx)
    name = "qrs_local24ms_cancellation_fraction"
    marker = module.MechanismBiomarker(next(s for s in module.SPECS if s.name == name))
    first = marker.compute(None)
    assert first.value is not None and marker.compute(None).value == first.value
    assert len(ctx._primitive_cache) == 1
    assert "clinical_claim_allowed=False" in first.computation_notes
    ctx._primitive_cache.clear()
    ctx.beat.r_offset.sample = 300
    assert marker.compute(None).value is None
    row = next(r for r in marker_freeze_payload()["markers"] if r["marker_id"] == name)
    assert row["segment_dependencies"] == ["qrs"]
    assert row["panel_ids"] == ["discovery_reserve"]
    assert any("05_AMYLOID_INFILTRATION.md" in r for r in row["references"])
    assert feature_profile(name).post_outcome_hypothesis
    assert not feature_profile(name).combination_eligible
