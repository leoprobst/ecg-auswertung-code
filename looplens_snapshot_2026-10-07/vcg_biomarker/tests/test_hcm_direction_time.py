"""Synthetic chronology contracts, not clinical HCM validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import direction_time_coupling


def rotating(n=101):
    u = np.linspace(0., 1., n)
    return np.column_stack((np.cos(np.pi * u), np.sin(np.pi * u), np.zeros(n)))


def test_fixed_direction_is_zero_despite_changing_amplitude():
    u = np.linspace(0., 1., 101)
    points = (1 + 4 * u ** 3)[:, None] * np.array([[1., 2., -3.]])
    result = direction_time_coupling(points)
    assert result == pytest.approx({"time_linear": 0., "time_quadratic": 0.}, abs=1e-14)


def test_quadratic_detects_symmetric_return_that_cancels_linear_moment():
    points = np.zeros((101, 3))
    points[:, 0] = 1.
    points[25:76, 0] = -1.
    result = direction_time_coupling(points)
    assert result["time_linear"] < 1e-14
    assert result["time_quadratic"] > .8


def test_agrees_with_explicit_weighted_unit_vector_reference():
    points = rotating() * np.linspace(.3, 2., 101)[:, None]
    h = np.ones(len(points))
    h[[0, -1]] = .5
    norms = np.linalg.norm(points, axis=1)
    w = h * norms ** 2
    w /= w.sum()
    directions = points / norms[:, None]
    phase = np.linspace(0., 1., len(points))
    centered = phase - np.average(phase, weights=w)
    variance = np.average(centered ** 2, weights=w)
    basis = [centered / np.sqrt(variance)]
    p2 = centered ** 2 - variance - np.average(centered ** 3, weights=w) / variance * centered
    basis.append(p2 / np.sqrt(np.average(p2 ** 2, weights=w)))
    expected = {name: np.linalg.norm(np.average(p[:, None] * directions, axis=0, weights=w))
                for name, p in zip(("time_linear", "time_quadratic"), basis)}
    assert direction_time_coupling(points) == pytest.approx(expected)


def test_gain_rotation_global_polarity_and_time_reversal_invariance():
    points = rotating()
    rng = np.random.default_rng(904)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    expected = direction_time_coupling(points)
    assert direction_time_coupling(-1e200 * points @ rotation) == pytest.approx(expected)
    assert direction_time_coupling(points[::-1]) == pytest.approx(expected)
    assert direction_time_coupling(points * [3, 1, 1]) != pytest.approx(expected)


def test_chronology_not_recoverable_from_unordered_directional_operator():
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import directional_operator
    points = rotating()
    shuffled = points.copy()
    shuffled[1:-1] = np.random.default_rng(444).permutation(shuffled[1:-1])
    np.testing.assert_allclose(directional_operator(points), directional_operator(shuffled), atol=1e-14)
    assert direction_time_coupling(points) != pytest.approx(direction_time_coupling(shuffled))


def test_parseval_bound_and_resampling_convergence():
    for seed in range(10):
        result = direction_time_coupling(np.random.default_rng(seed).normal(size=(100, 3)))
        assert all(0 <= x <= 1 for x in result.values())
        assert sum(x * x for x in result.values()) <= 1 + 1e-12
    assert direction_time_coupling(rotating(101)) == pytest.approx(direction_time_coupling(rotating(201)), abs=3e-4)


@pytest.mark.parametrize("points", [np.zeros((20, 3)), np.ones((11, 3)), np.ones((20, 2)),
    np.full((20, 3), np.nan), np.full((20, 3), np.inf),
    np.vstack((np.ones((1, 3)), np.zeros((19, 3))))])
def test_unobservable_segments_abstain(points):
    with pytest.raises(ValueError):
        direction_time_coupling(points)


def test_zero_samples_are_not_compressed_or_divided_by_zero():
    points = rotating()
    points[40:60] = 0
    result = direction_time_coupling(points)
    assert all(np.isfinite(list(result.values())))
    assert result != pytest.approx(direction_time_coupling(np.delete(points, np.arange(40, 60), axis=0)))


def test_shared_context_computation_and_abstention(monkeypatch):
    from vcg_biomarker.core.biomarkers import pathology_mechanism_biomarkers as module
    point = lambda n: SimpleNamespace(sample=n, confidence=1.)
    ctx = SimpleNamespace(beat=SimpleNamespace(t_onset=point(0), t_offset=point(100)),
                          fs=500, _primitive_cache={}, origin_corrected_vcg=lambda: rotating())
    monkeypatch.setattr(module, "_context", lambda *args: ctx)
    original = module.direction_time_coupling
    calls = []
    def count(points):
        calls.append(1)
        return original(points)
    monkeypatch.setattr(module, "direction_time_coupling", count)
    markers = [module.MechanismBiomarker(s) for s in module.SPECS if s.component.startswith("time_")]
    assert len(markers) == 2
    for marker in markers:
        result = marker.compute(None)
        assert result.value is not None
        assert "clinical_claim_allowed=False" in result.computation_notes
    assert len(calls) == 1
    ctx._primitive_cache.clear()
    ctx.beat.t_offset.sample = 500
    assert all(marker.compute(None).value is None for marker in markers)
    assert len(calls) == 1


def test_freeze_roles_and_t_only_dependency():
    from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
    from vcg_biomarker.mimic.feature_roles import feature_profile
    payload = marker_freeze_payload()
    for name in ("t_direction_time_linear_coupling", "t_direction_time_quadratic_coupling"):
        row = next(m for m in payload["markers"] if m["marker_id"] == name)
        assert row["segment_dependencies"] == ["t"]
        assert row["panel_ids"] == ["discovery_reserve"]
        assert any("04_HCM.md" in r for r in row["references"])
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis and not profile.combination_eligible
