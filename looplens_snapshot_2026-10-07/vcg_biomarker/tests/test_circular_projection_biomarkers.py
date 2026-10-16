"""The projections that give the angular markers a scale a screen can order.

Twenty-four extracted markers are coordinates on a full turn. A rank-based
screen cannot order them: -179 and +179 degrees are two degrees of arc apart
and sit at opposite ends of the rank list.

Measured over 1.168 marker-endpoint pairs from the existing partitions before
this module was written: 297 pairs (25,4 %) separate better under a faithful
representation by more than 0,05 AUC, 92 of them invisible to the current
screen. Every one of the twenty-four markers has at least one such pair.

These tests pin the properties that make the projections trustworthy -- exact
agreement with the source, identical missingness, continuity across the seam,
invariance of the doubled angle under a half turn -- and the boundaries that
keep them additive.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.circular_projection_biomarkers import (
    CIRCULAR_PROJECTION_SPECS,
    CIRCULAR_SOURCE_MARKERS,
    FRONTAL_ORIENTATION_SOURCES,
    TRANSVERSE_AZIMUTH_SOURCES,
    CircularProjectionBiomarker,
    circular_projection_marker_ids,
    project_angle,
)


# --- the projection itself ------------------------------------------------


def test_every_source_marker_gets_exactly_two_projections():
    per_source: dict[str, int] = {}
    for spec in CIRCULAR_PROJECTION_SPECS:
        per_source[spec.source] = per_source.get(spec.source, 0) + 1
    assert set(per_source) == set(CIRCULAR_SOURCE_MARKERS)
    assert set(per_source.values()) == {2}
    assert len(CIRCULAR_PROJECTION_SPECS) == 2 * len(CIRCULAR_SOURCE_MARKERS)


def test_the_projection_is_continuous_across_the_seam():
    """The defect, stated as the property that repairs it.

    The raw angles -179,9 and +179,9 are 0,2 degrees apart in the body and 359,8
    apart on the scale the screen reads. Their projections must be adjacent.
    """

    for component in ("cos", "sin"):
        left = project_angle(-179.9, component)
        right = project_angle(179.9, component)
        assert abs(left - right) < 0.01, component


def test_the_raw_scale_is_what_makes_that_necessary():
    """If this ever fails the seam has gone away and the module is redundant."""

    assert abs(-179.9 - 179.9) > 350.0


def test_projections_are_bounded():
    for degrees in np.linspace(-180.0, 180.0, 721):
        for component in ("cos", "sin"):
            value = project_angle(float(degrees), component)
            assert -1.0 <= value <= 1.0


def test_the_two_projections_determine_the_angle():
    """Nothing is lost: cosine and sine together recover the original angle."""

    for degrees in np.linspace(-179.0, 179.0, 359):
        recovered = np.degrees(
            np.arctan2(
                project_angle(float(degrees), "sin"),
                project_angle(float(degrees), "cos"),
            )
        )
        assert abs(recovered - degrees) < 1e-6


def test_a_doubled_angle_is_invariant_under_a_half_turn():
    """The property that repairs an axis rather than a direction.

    ``loop_ellipse_orientation_deg`` is atan2 over an eigenvector, and an
    eigenvector is defined only up to sign, so the same ellipse can be
    reported 180 degrees apart. Measured on the existing partitions, 12,8 % of
    recordings land on the opposite branch from the other 87,2 %.
    """

    for degrees in (-10.0, 0.0, 37.5, 90.0, 123.4):
        for component in ("cos", "sin"):
            here = project_angle(degrees, component, 2)
            there = project_angle(degrees + 180.0, component, 2)
            assert here == pytest.approx(there, abs=1e-9), (degrees, component)


def test_a_single_angle_would_not_repair_an_axis():
    """States why the doubling is there rather than assuming it."""

    assert project_angle(-10.0, "cos", 1) != pytest.approx(
        project_angle(170.0, "cos", 1), abs=1e-6
    )


def test_a_missing_or_non_finite_angle_projects_to_nothing():
    for bad in (None, float("nan"), float("inf"), float("-inf")):
        for component in ("cos", "sin"):
            assert project_angle(bad, component) is None


# --- registration and panel boundaries ------------------------------------


def test_all_projections_are_registered_and_computable():
    available = set(BiomarkerRegistry.available())
    for name in circular_projection_marker_ids():
        assert name in available, name
        assert isinstance(BiomarkerRegistry.get(name), CircularProjectionBiomarker)


def test_every_source_marker_exists():
    """A projection of a marker that is not extracted would be a dead column."""

    available = set(BiomarkerRegistry.available())
    for source in CIRCULAR_SOURCE_MARKERS:
        assert source in available, source


def test_the_curated_panels_are_unchanged():
    """The whole module must be additive: fixed-size panels, so a marker
    admitted to the candidate pool would push another one out."""

    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert len(panels["classical_criteria"]) == 53
    new = set(circular_projection_marker_ids())
    for panel_id in ("primary_core", "sentinel", "classical_criteria"):
        assert not (new & panels[panel_id]), panel_id
    assert new <= panels["discovery_reserve"]


def test_projections_are_post_outcome_and_out_of_the_combination_search():
    """The cost of the tier, pinned rather than left implicit: a cosine and a
    sine carry a whole direction only when something can use them together,
    and the combination search is closed to post-outcome markers."""

    from vcg_biomarker.mimic.feature_roles import feature_profile

    for name in circular_projection_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.discovery_eligible, name
        assert not profile.core_sentinel_eligible, name
        assert not profile.combination_eligible, name


def test_no_source_marker_was_altered_or_retired():
    """Additive means the sources keep their identity and their panels."""

    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert "svg_azimuth" in panels["sentinel"]
    assert "loop_ellipse_orientation_deg" in panels["sentinel"]
    for source in CIRCULAR_SOURCE_MARKERS:
        assert BiomarkerRegistry.get(source) is not None
        assert not isinstance(
            BiomarkerRegistry.get(source), CircularProjectionBiomarker
        )


def test_names_do_not_collide_with_anything_else():
    ids = circular_projection_marker_ids()
    assert len(set(ids)) == len(ids)
    assert not (set(ids) & set(CIRCULAR_SOURCE_MARKERS))


# --- the planes are not interchangeable -----------------------------------


def test_the_transverse_sources_share_one_convention():
    """All twenty-three azimuths are atan2(Z, X), so cosine is leftward and
    sine is posterior for every one of them. If a new azimuth marker ever used
    a different convention this would have to be revisited."""

    from vcg_biomarker.core.vcg_geometry import vcg_azimuth_deg

    vector = np.array([3.0, -1.0, 4.0])
    expected = np.degrees(np.arctan2(vector[2], vector[0]))
    assert vcg_azimuth_deg(vector) == pytest.approx(expected)
    magnitude = float(np.hypot(vector[0], vector[2]))
    assert project_angle(vcg_azimuth_deg(vector), "cos") == pytest.approx(
        vector[0] / magnitude
    )
    assert project_angle(vcg_azimuth_deg(vector), "sin") == pytest.approx(
        vector[2] / magnitude
    )


def test_the_orientation_marker_is_labelled_frontal_not_transverse():
    """It is atan2(Y, X) over an eigenvector, so its sine points inferior, not
    posterior. Getting this wrong would mislabel a whole axis."""

    specs = [
        spec
        for spec in CIRCULAR_PROJECTION_SPECS
        if spec.source in FRONTAL_ORIENTATION_SOURCES
    ]
    assert len(specs) == 2
    assert {spec.plane for spec in specs} == {"frontal"}
    assert {spec.multiple for spec in specs} == {2}
    labels = {spec.component: spec.axis_label for spec in specs}
    assert "inferior" in labels["sin"]
    assert "left" in labels["cos"]


def test_the_transverse_specs_are_direction_not_axis():
    specs = [
        spec
        for spec in CIRCULAR_PROJECTION_SPECS
        if spec.source in TRANSVERSE_AZIMUTH_SOURCES
    ]
    assert len(specs) == 2 * len(TRANSVERSE_AZIMUTH_SOURCES)
    assert {spec.multiple for spec in specs} == {1}
    assert {spec.plane for spec in specs} == {"transverse"}


# --- the folded diagnostic must fold angles on the circle ------------------


def test_the_folded_diagnostic_knows_which_markers_wrap():
    from vcg_biomarker.mimic.statistics import _is_circular_feature

    assert _is_circular_feature("svg_azimuth")
    assert _is_circular_feature("loop_ellipse_orientation_deg")
    assert not _is_circular_feature("svg_magnitude")
    assert not _is_circular_feature("svg_leftward")


def test_folding_an_angle_on_the_line_was_arithmetic_on_a_scale_that_does_not_exist():
    """The bug this fixes, stated as a measurement.

    Controls clustered near the seam, cases spread away from it. Folded on the
    line the reference is meaningless; folded on the circle the association is
    there.
    """

    from vcg_biomarker.mimic.statistics import fit_marker_model

    rng = np.random.default_rng(11)
    controls = rng.normal(180.0, 8.0, 400)
    controls = ((controls + 180.0) % 360.0) - 180.0   # straddles the seam
    cases = rng.uniform(-90.0, 90.0, 200)
    values = np.concatenate([controls, cases])
    labels = np.concatenate([np.zeros(400, dtype=int), np.ones(200, dtype=int)])

    model = fit_marker_model(values, labels, "svg_azimuth")
    assert model.folded_auc > 0.9
    assert model.folded_effect_gain > 0.3


def test_the_circular_fold_is_only_applied_to_circular_markers():
    """The same values under an ordinary marker id must take the linear path,
    so nothing else changes behaviour."""

    from vcg_biomarker.mimic.statistics import fit_marker_model

    rng = np.random.default_rng(11)
    controls = ((rng.normal(180.0, 8.0, 400) + 180.0) % 360.0) - 180.0
    cases = rng.uniform(-90.0, 90.0, 200)
    values = np.concatenate([controls, cases])
    labels = np.concatenate([np.zeros(400, dtype=int), np.ones(200, dtype=int)])

    circular = fit_marker_model(values, labels, "svg_azimuth")
    linear = fit_marker_model(values, labels, "svg_magnitude")
    assert circular.folded_auc != pytest.approx(linear.folded_auc, abs=1e-6)


def test_the_circular_reference_is_a_resultant_direction_not_a_median():
    """A median of angles needs a branch chosen first, which is the very thing
    that goes wrong. Pinned so a later simplification cannot reintroduce it."""

    import inspect

    from vcg_biomarker.mimic import statistics

    source = inspect.getsource(statistics._circular_fold)
    assert "arctan2" in source
    # The docstring says why a median is wrong here, so only the code after it
    # is checked for one actually being taken.
    body = source.split('"""')[-1]
    assert "median" not in body


def test_the_diagnostic_still_decides_nothing():
    """Unchanged from the folded diagnostic: no gate, no FDR family, no claim
    level -- for the circular path either."""

    import inspect

    from vcg_biomarker.mimic import discovery_statistics, validation

    for module in (validation, discovery_statistics):
        source = inspect.getsource(module)
        for guard in ("folded_auc", "folded_effect_gain", "_circular_fold"):
            assert guard not in source, f"{module.__name__} acts on {guard}"


# --- documentation carries the measurement --------------------------------


def test_the_module_docstring_carries_the_numbers_that_justify_it():
    from vcg_biomarker.core.biomarkers import circular_projection_biomarkers

    doc = circular_projection_biomarkers.__doc__ or ""
    assert "25,4 %" in doc
    assert "1.168" in doc
    assert "12,8 %" in doc


def test_every_projection_names_its_source_in_its_formula():
    for spec in CIRCULAR_PROJECTION_SPECS:
        definition = BiomarkerRegistry.definition(spec.name)
        assert spec.source in definition.formula, spec.name
        assert definition.pathological_direction == "data_driven"
        assert "No diagnostic cutoff is claimed." in definition.pathological_note
