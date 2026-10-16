import numpy as np
import pytest
from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import t_energy_concentration
from vcg_biomarker.core.biomarkers import BiomarkerRegistry


def test_uniform_and_concentrated_reference():
    x = np.ones((21, 3))
    assert t_energy_concentration(x) == pytest.approx(1.)
    x[:] = 0
    x[10] = [1., 0., 0.]
    assert t_energy_concentration(x) == pytest.approx(20.)


def test_gain_rotation_and_polarity_invariance():
    rng = np.random.default_rng(83)
    x = rng.normal(size=(100, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert t_energy_concentration(x) == pytest.approx(t_energy_concentration(-1e150*x@rotation))


@pytest.mark.parametrize("x", [np.zeros((20, 3)), np.ones((3, 3)), np.full((20, 3), np.nan)])
def test_bad_segments_abstain(x):
    with pytest.raises(ValueError):
        t_energy_concentration(x)


def test_registered_marker_abstains_without_delineation():
    from vcg_biomarker.core.delineation.base import RecordDelineation
    marker = BiomarkerRegistry.get("t_energy_temporal_concentration")
    result = marker.compute(RecordDelineation(beats=[], fs=500, n_samples=100, ecg_id=0))
    assert result.value is None


def test_freeze_has_formula_dependency_and_discovery_membership():
    from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
    marker = next(row for row in marker_freeze_payload()["markers"]
                  if row["marker_id"] == "t_energy_temporal_concentration")
    assert marker["panel_ids"] == ["discovery_reserve"]
    assert "t" in marker["segment_dependencies"]
    assert "||T||^4" in marker["formula"]
    assert marker["initial_evidence_tier"] == "exploratory"
    assert marker["mathematical_plausibility_range"]["lower"] == 1.
    assert marker["mathematical_plausibility_range"]["upper"] is None


def test_context_qualified_boundaries_and_cache():
    from types import SimpleNamespace as NS
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import _t_energy_concentration
    point = lambda sample: NS(sample=sample, confidence=1.)
    ctx = NS(beat=NS(t_onset=point(10), t_offset=point(40)), fs=500.,
             _primitive_cache={}, origin_corrected_vcg=lambda: np.ones((60, 3)))
    assert _t_energy_concentration(ctx) == pytest.approx(1.)
    assert len(ctx._primitive_cache) == 1
    assert _t_energy_concentration(ctx) == pytest.approx(1.)
    ctx._primitive_cache.clear()
    ctx.beat.t_offset.confidence = 0.
    with pytest.raises(ValueError, match="boundaries"):
        _t_energy_concentration(ctx)


def test_time_stretch_invariance_for_same_normalized_shape():
    def triangle(n):
        t = np.linspace(0., 1., n)
        return np.column_stack([1 - np.abs(2*t-1), np.zeros(n), np.zeros(n)])
    assert t_energy_concentration(triangle(1001)) == pytest.approx(1.8, rel=1e-5)
    assert t_energy_concentration(triangle(2001)) == pytest.approx(1.8, rel=1e-5)
