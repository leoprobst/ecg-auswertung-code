from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import SPECS, directional_operator, operator_pair, _metrics


def line(vector):
    return np.tile(vector, (20, 1)).astype(float)


def test_alignment_and_orthogonality_share_zero_commutator_not_overlap():
    same = operator_pair(line([1, 0, 0]), line([1, 0, 0]))
    perpendicular = operator_pair(line([1, 0, 0]), line([0, 1, 0]))
    assert same == {"commutator": 0., "overlap": 1.}
    assert perpendicular == {"commutator": 0., "overlap": 0.}
    diagonal = operator_pair(line([1, 0, 0]), line([1, 1, 0]))
    assert diagonal["commutator"] == pytest.approx(1 / np.sqrt(2))
    assert diagonal["overlap"] == pytest.approx(.5)


def test_common_rotation_gain_and_polarity_invariance_not_general_transfer():
    rng = np.random.default_rng(60)
    q, s = rng.normal(size=(30, 3)), rng.normal(size=(20, 3)) * [3, 1, 2]
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    expected = operator_pair(q, s)
    transformed = operator_pair(30 * q @ rotation, -.03 * s @ rotation)
    assert transformed == pytest.approx(expected)
    assert operator_pair(q * [20, 1, 1], s * [20, 1, 1]) != pytest.approx(expected)


def test_no_covariance_centering_of_constant_st():
    assert np.trace(directional_operator(line([1, 2, 3]))) == pytest.approx(1)


@pytest.mark.parametrize("points", [np.zeros((20, 3)), np.full((20, 3), np.nan), np.ones((3, 3)), np.ones((20, 2))])
def test_invalid_or_unobservable_operator_rejected(points):
    with pytest.raises(ValueError):
        directional_operator(points)


def test_registration_freeze_discovery_only_and_no_curated_displacement():
    from vcg_biomarker.mimic.panels import default_feature_panels
    from vcg_biomarker.mimic.feature_roles import feature_profile
    from vcg_biomarker.core.biomarkers import BiomarkerRegistry
    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    for spec in SPECS:
        assert spec.name in BiomarkerRegistry.available()
        assert spec.name in panels["discovery_reserve"]
        assert spec.name not in panels["primary_core"] | panels["sentinel"]
        profile = feature_profile(spec.name)
        assert profile.post_outcome_hypothesis and not profile.combination_eligible
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240


def context(st_scale=.2):
    point = lambda value: SimpleNamespace(sample=value, confidence=1.)
    beat = SimpleNamespace(r_onset=point(0), r_offset=point(19), j_point=point(20), t_onset=point(35))
    signal = np.vstack([line([1, 0, 0]), line([st_scale, st_scale, 0])])
    return SimpleNamespace(beat=beat, fs=500, _primitive_cache={}, origin_corrected_vcg=lambda: signal)


def test_feature_context_reuses_one_operator_pair_and_rejects_weak_st():
    ctx = context()
    result = _metrics(ctx)
    assert _metrics(ctx) is result
    assert result["overlap"] == pytest.approx(.5)
    with pytest.raises(ValueError, match="ST_energy"):
        _metrics(context(.0001))


def test_invalid_fiducial_never_gets_clamped():
    ctx = context()
    ctx.beat.r_onset.sample = -.1
    with pytest.raises(ValueError):
        _metrics(ctx)


@pytest.mark.parametrize("gain", [1., 1e200, 1e300])
def test_relative_quality_gate_does_not_overflow_on_finite_amplitudes(gain):
    ctx = context()
    signal = ctx.origin_corrected_vcg() * gain
    ctx.origin_corrected_vcg = lambda: signal
    assert _metrics(ctx) == pytest.approx(_metrics(context()))


@pytest.mark.parametrize("gain", [0., 1e-15, 1e200])
def test_weak_signal_gate_still_abstains_after_numeric_scaling(gain):
    ctx = context(.0001)
    signal = ctx.origin_corrected_vcg() * gain
    ctx.origin_corrected_vcg = lambda: signal
    with pytest.raises(ValueError, match="ST_energy"):
        _metrics(ctx)


def test_early_late_overlap_needs_no_st_and_is_not_polarity_sensitive():
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import _early_late_overlap
    ctx = context()
    ctx.beat.r_offset.sample = 39
    ctx.beat.j_point = None
    assert _early_late_overlap(ctx) == pytest.approx(.5)
    assert _early_late_overlap(ctx) == pytest.approx(.5)
    assert len(ctx._primitive_cache) == 1


def test_early_late_rejects_vanishing_second_half():
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import _early_late_overlap
    ctx = context(.00001)
    ctx.beat.r_offset.sample = 39
    with pytest.raises(ValueError, match="weak_half"):
        _early_late_overlap(ctx)


@pytest.mark.parametrize("vector, expected", [([1, 0, 0], 0.), ([-1, 0, 0], 1.),
                                             ([0, 1, 0], 0.), ([-1, 1, 0], .5)])
def test_opposed_energy_known_vectors(vector, expected):
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import opposed_t_energy_fraction
    assert opposed_t_energy_fraction(line([1, 0, 0]), line(vector)) == pytest.approx(expected)


def test_opposed_energy_is_not_an_alias_of_mean_angle_or_polarity_blind_operator():
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import opposed_t_energy_fraction
    q = line([1, 0, 0])
    mixed = np.vstack([line([2, 0, 0]), line([-1, 0, 0])])
    # Both mean T axes point along +X; second-moment overlaps are also both 1.
    assert np.trapezoid(mixed[:, 0]) > 0
    assert operator_pair(q, mixed)["overlap"] == operator_pair(q, q)["overlap"]
    assert opposed_t_energy_fraction(q, q) == 0
    assert opposed_t_energy_fraction(q, mixed) == pytest.approx(.2)


def test_opposed_energy_rotation_gain_and_independent_polarity_contracts():
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import opposed_t_energy_fraction
    rng = np.random.default_rng(903)
    q = rng.normal(size=(30, 3)) + [2, 0, 0]
    t = rng.normal(size=(50, 3)) - [1, 0, 0]
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    expected = opposed_t_energy_fraction(q, t)
    assert opposed_t_energy_fraction(-q @ rotation * 1e200, -t @ rotation * 2e200) == pytest.approx(expected)
    assert opposed_t_energy_fraction(q, -t) != pytest.approx(expected)


@pytest.mark.parametrize("failure", ["cancelled_qrs", "weak_t", "nan", "shape", "zero"])
def test_opposed_energy_quality_abstentions(failure):
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import opposed_t_energy_fraction
    q, t = line([1, 0, 0]), line([-1, 1, 0])
    if failure == "cancelled_qrs":
        q = np.vstack([q, -q])
    elif failure == "weak_t":
        t *= 1e-5
    elif failure == "nan":
        t[4, 0] = np.nan
    elif failure == "shape":
        t = t[:, :2]
    else:
        t[:] = 0
    with pytest.raises(ValueError):
        opposed_t_energy_fraction(q, t)


def test_opposed_energy_context_boundaries_cache_and_result_contract(monkeypatch):
    from vcg_biomarker.core.biomarkers import pathology_mechanism_biomarkers as module
    point = lambda n: SimpleNamespace(sample=n, confidence=1.)
    signal = np.vstack([np.tile([1., 0, 0], (30, 1)), np.tile([-1., 1, 0], (30, 1))])
    ctx = SimpleNamespace(beat=SimpleNamespace(r_onset=point(0), r_offset=point(29),
        t_onset=point(30), t_offset=point(59)), fs=500, _primitive_cache={},
        origin_corrected_vcg=lambda: signal)
    monkeypatch.setattr(module, "_context", lambda *args: ctx)
    marker = module.MechanismBiomarker(next(s for s in SPECS if s.component == "opposed_t_energy"))
    assert marker.compute(None).value == pytest.approx(.5)
    assert "polarity_blind=False" in marker.compute(None).computation_notes
    assert len(ctx._primitive_cache) == 1
    ctx._primitive_cache.clear()
    ctx.beat.t_offset.sample = 80
    result = marker.compute(None)
    assert result.value is None and result.confidence == 0
    assert "invalid_qrs_t_boundaries" in result.computation_notes


def test_rwt_proxy_does_not_claim_normal_mass_or_confirmed_remodeling():
    from vcg_biomarker.mimic.label_registry import definitions_by_id
    for name in ("concentric_remodeling_proxy", "concentric_remodeling_rwt"):
        definition = definitions_by_id()[name]
        assert "geometry proxy" in definition.display_name
        assert "LV mass not constrained" in definition.positive_rule
        assert "does not exclude eccentric LVH" in definition.negative_rule


def test_opposed_energy_freeze_declares_both_segments_and_hypothesis_reference():
    from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
    row = next(m for m in marker_freeze_payload()["markers"] if m["marker_id"] == "qrs_t_opposed_energy_fraction")
    assert row["segment_dependencies"] == ["qrs", "t"]
    assert any("03_HYPERTENSIVE_LVH.md" in r for r in row["references"])
    assert row["initial_evidence_tier"] == "exploratory"
