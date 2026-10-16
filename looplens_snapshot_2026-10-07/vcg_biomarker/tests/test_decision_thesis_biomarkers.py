"""Five markers derived from the treatment-decision theses.

The theses are in THESEN_ENTSCHEIDUNGSMARKER_2026-09-12.md at the project
root. These tests pin the geometry the markers rest on -- the forward Dower
lead vectors that reproduce the project's inverse Dower matrix, the
criterion's own contiguity and thresholds -- and the behaviour of each marker
on constructed vectors, before any endpoint is looked at.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers import decision_thesis_biomarkers as m
from vcg_biomarker.core.biomarkers.decision_thesis_biomarkers import (
    CONTIGUOUS_GROUPS,
    DECISION_THESIS_SPECS,
    DOWER_LEAD_VECTORS,
    LEAD_VECTORS,
    MINIMUM_VECTOR_SHARE,
    THRESHOLDS_MV,
    VISIBILITY_CEILING,
    DecisionThesisBiomarker,
    criterion_visibility,
    decision_thesis_marker_ids,
)


# --- registration and panel boundaries ------------------------------------


def test_five_markers_are_registered():
    ids = decision_thesis_marker_ids()
    assert len(ids) == 5
    available = set(BiomarkerRegistry.available())
    for name in ids:
        assert name in available, name
        assert isinstance(BiomarkerRegistry.get(name), DecisionThesisBiomarker)


def test_the_curated_panels_are_unchanged():
    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert len(panels["classical_criteria"]) == 53
    new = set(decision_thesis_marker_ids())
    for panel_id in ("primary_core", "sentinel", "classical_criteria"):
        assert not (new & panels[panel_id]), panel_id
    assert new <= panels["discovery_reserve"]


def test_the_markers_are_post_outcome_and_out_of_the_combination_search():
    from vcg_biomarker.mimic.feature_roles import feature_profile

    for name in decision_thesis_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.discovery_eligible, name
        assert not profile.core_sentinel_eligible, name
        assert not profile.combination_eligible, name


def test_every_spec_names_its_thesis_and_its_refutation():
    for spec in DECISION_THESIS_SPECS:
        assert "Thesis E" in spec.thesis, spec.name
        assert "Refuted if" in spec.thesis, spec.name
        definition = BiomarkerRegistry.definition(spec.name)
        assert "No diagnostic cutoff is claimed." in definition.pathological_note


def test_the_theses_paper_exists_and_names_the_markers():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "THESEN_ENTSCHEIDUNGSMARKER_2026-09-12.md").read_text(encoding="utf-8")
    for name in decision_thesis_marker_ids():
        assert name in text, name
    # the withdrawn first form of the opposition marker is recorded
    assert "terminal" in text and "40 ms" in text


# --- the geometry the criterion lives in ----------------------------------


def test_the_forward_dower_matrix_reproduces_the_projects_inverse_dower():
    """The lead vectors are the ones the VCG was derived through: their
    pseudo-inverse is the project's inverse Dower matrix, channel for
    channel."""

    from vcg_biomarker.core.vcg_transform import INVERSE_DOWER_CHANNELS, INVERSE_DOWER_MATRIX

    forward = np.array([DOWER_LEAD_VECTORS[name] for name in INVERSE_DOWER_CHANNELS])
    assert np.abs(np.linalg.pinv(forward) - INVERSE_DOWER_MATRIX).max() < 1e-3


def test_the_contiguity_is_the_criterions_own():
    from vcg_biomarker.mimic.classical import CONTIGUOUS_GROUPS as CRITERION_GROUPS

    assert tuple(tuple(g) for g in CONTIGUOUS_GROUPS) == tuple(tuple(g) for g in CRITERION_GROUPS)


def test_the_thresholds_are_the_criterions_with_v2_v3_doubled():
    assert THRESHOLDS_MV["V2"] == pytest.approx(0.20)
    assert THRESHOLDS_MV["V3"] == pytest.approx(0.20)
    for lead, threshold in THRESHOLDS_MV.items():
        if lead not in ("V2", "V3"):
            assert threshold == pytest.approx(0.10), lead


def test_the_derived_limb_leads_obey_einthoven_and_goldberger():
    one, two, three = LEAD_VECTORS["I"], LEAD_VECTORS["II"], LEAD_VECTORS["III"]
    assert np.allclose(three, two - one)
    assert np.allclose(LEAD_VECTORS["aVR"], -(one + two) / 2.0)
    assert np.allclose(LEAD_VECTORS["aVL"], one - two / 2.0)
    assert np.allclose(LEAD_VECTORS["aVF"], (two + three) / 2.0)


def test_visibility_is_a_property_of_direction_only():
    d = np.array([0.3, 0.5, -0.8])
    assert criterion_visibility(d) == pytest.approx(criterion_visibility(d * 40.0))
    assert criterion_visibility(np.zeros(3)) is None
    assert criterion_visibility(d) >= 0.0


def test_visibility_ranks_directions_the_way_the_lead_system_does():
    """Inferior and lateral injury vectors are seen almost as well as the
    best possible direction; posterior and rightward ones are seen as
    depression only and get almost nothing; and the doubled V2-V3 threshold
    makes a purely anterior vector less visible than an inferior one."""

    inferior = criterion_visibility(np.array([0.0, 1.0, 0.0])) / VISIBILITY_CEILING
    lateral = criterion_visibility(np.array([1.0, 0.0, 0.0])) / VISIBILITY_CEILING
    anterior = criterion_visibility(np.array([0.0, 0.0, -1.0])) / VISIBILITY_CEILING
    posterior = criterion_visibility(np.array([0.0, 0.0, 1.0])) / VISIBILITY_CEILING
    rightward = criterion_visibility(np.array([-1.0, 0.0, 0.0])) / VISIBILITY_CEILING
    assert inferior > 0.9 and lateral > 0.9
    assert posterior < 0.2 and rightward < 0.2
    assert anterior < inferior


def test_the_ceiling_is_the_exact_maximum_over_the_sphere():
    """The exact pairwise maximum must bound a dense sample of directions and
    be approached by it."""

    n = 20000
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5**0.5) * i
    points = np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])
    sampled = max(criterion_visibility(p) for p in points)
    assert sampled <= VISIBILITY_CEILING + 1e-9
    assert sampled >= 0.995 * VISIBILITY_CEILING


# --- a stub context -------------------------------------------------------


class _Ctx:
    """Stub carrying the vectors the five calculators read."""

    def __init__(self, st, qrs_max=(1.0, 0.0, 0.0), qrs_area=None, initial=None, qrs_points=None, fs=500):
        self._st = None if st is None else np.asarray(st, dtype=float)
        self._qrs_max = np.asarray(qrs_max, dtype=float)
        self._qrs_area = None if qrs_area is None else np.asarray(qrs_area, dtype=float)
        self._initial = None if initial is None else np.asarray(initial, dtype=float)
        self._qrs_points = None if qrs_points is None else np.asarray(qrs_points, dtype=float)
        self.fs = fs

    def vcg_segment_origin_corrected(self, segment):
        return self._qrs_points if segment == "qrs" else None


def _run(calculator, ctx):
    original_vec = m._vector_array
    original_win = m._window_vector

    def vector_array(c, seg, mode):
        if seg == "st":
            return c._st
        if seg == "qrs" and mode == "max":
            return c._qrs_max
        if seg == "qrs" and mode == "area":
            return c._qrs_area
        return None

    try:
        m._vector_array = vector_array
        m._window_vector = lambda c, seg, length_ms=20, end_aligned=False: c._initial
        return calculator(ctx)
    finally:
        m._vector_array = original_vec
        m._window_vector = original_win


# --- thesis E1: visibility and hidden magnitude ---------------------------


def test_a_posterior_injury_is_mostly_hidden_and_an_inferior_one_is_not():
    posterior = _Ctx(st=[0.0, 0.0, 0.3])
    inferior = _Ctx(st=[0.0, 0.3, 0.0])
    vis_p = _run(m._visibility_fraction, posterior).value
    vis_i = _run(m._visibility_fraction, inferior).value
    assert vis_p < 0.2 and vis_i > 0.9
    hidden_p = _run(m._hidden_magnitude, posterior).value
    hidden_i = _run(m._hidden_magnitude, inferior).value
    assert hidden_p == pytest.approx(0.3 * (1.0 - vis_p))
    assert hidden_p > 0.24 and hidden_i < 0.03


def test_the_hidden_magnitude_scales_with_the_vector_and_the_fraction_does_not():
    small = _Ctx(st=[0.0, 0.0, 0.2])
    large = _Ctx(st=[0.0, 0.0, 0.6])
    assert _run(m._visibility_fraction, small).value == pytest.approx(_run(m._visibility_fraction, large).value)
    assert _run(m._hidden_magnitude, large).value == pytest.approx(3.0 * _run(m._hidden_magnitude, small).value)


def test_a_tiny_st_vector_abstains_rather_than_reporting_a_direction():
    """Half-availability by design: a normal complex has no injury current,
    and the direction of a vector below five per cent of the QRS maximum is
    noise."""

    ctx = _Ctx(st=[0.0, 0.0, 0.5 * MINIMUM_VECTOR_SHARE * 0.5], qrs_max=[0.5, 0.0, 0.0])
    for calculator in (m._visibility_fraction, m._hidden_magnitude, m._qrs_opposition, m._initial_qrs_to_st_angle):
        value = _run(calculator, ctx)
        assert value.value is None, calculator.__name__
        assert "too small" in (value.notes or ""), calculator.__name__


# --- thesis E2: the magnitude beyond discordance --------------------------


def test_a_purely_discordant_st_vector_has_nothing_beyond_discordance():
    ctx = _Ctx(st=[-0.3, 0.0, 0.0], qrs_area=[2.0, 0.0, 0.0])
    assert _run(m._nondiscordant_magnitude, ctx).value == pytest.approx(0.0, abs=1e-12)


def test_an_orthogonal_or_concordant_st_vector_is_kept_whole():
    orthogonal = _Ctx(st=[0.0, 0.0, 0.3], qrs_area=[2.0, 0.0, 0.0])
    concordant = _Ctx(st=[0.3, 0.0, 0.0], qrs_area=[2.0, 0.0, 0.0])
    assert _run(m._nondiscordant_magnitude, orthogonal).value == pytest.approx(0.3)
    assert _run(m._nondiscordant_magnitude, concordant).value == pytest.approx(0.3)


def test_the_discordant_part_is_removed_and_the_rest_survives():
    """ST = -0.3 along the QRS plus 0.4 across it: the 0.3 is what strain or
    a block could produce, the 0.4 is what they could not."""

    ctx = _Ctx(st=[-0.3, 0.4, 0.0], qrs_area=[5.0, 0.0, 0.0])
    assert _run(m._nondiscordant_magnitude, ctx).value == pytest.approx(0.4)


def test_the_decomposition_does_not_gate_on_the_st_magnitude():
    """A small ST vector has a small magnitude beyond discordance -- that is
    information, not noise, so this marker keeps it."""

    ctx = _Ctx(st=[0.0, 0.001, 0.0], qrs_max=[1.0, 0.0, 0.0], qrs_area=[2.0, 0.0, 0.0])
    assert _run(m._nondiscordant_magnitude, ctx).value == pytest.approx(0.001)


def test_a_degenerate_qrs_area_vector_abstains():
    ctx = _Ctx(st=[0.0, 0.3, 0.0], qrs_area=[0.0, 0.0, 0.0])
    assert _run(m._nondiscordant_magnitude, ctx).value is None


# --- thesis E3: the S wave as the injury vector sees it -------------------


def _loop(points):
    return np.asarray(points, dtype=float)


def test_the_opposition_is_the_deepest_excursion_against_the_st_direction():
    points = _loop([[0, 0, 0], [0.2, 0, 0], [1.0, 0.1, 0], [0.3, 0, -0.8], [0, 0, -0.2], [0, 0, 0]])
    ctx = _Ctx(st=[0.0, 0.0, 0.5], qrs_points=points)   # injury points posterior
    assert _run(m._qrs_opposition, ctx).value == pytest.approx(0.8)


def test_a_loop_that_never_runs_against_the_injury_has_no_opposition():
    """Grade 3: the forces opposing the injury vector are absorbed."""

    points = _loop([[0, 0, 0], [0.5, 0, 0.1], [1.0, 0.2, 0.4], [0.4, 0.1, 0.6], [0, 0, 0.2]])
    ctx = _Ctx(st=[0.0, 0.0, 0.5], qrs_points=points)
    assert _run(m._qrs_opposition, ctx).value == pytest.approx(0.0)


def test_the_opposition_is_window_free_and_scales_with_the_loop():
    points = _loop([[0, 0, 0], [0.2, 0, 0], [1.0, 0.1, 0], [0.3, 0, -0.8], [0, 0, -0.2], [0, 0, 0]])
    a = _run(m._qrs_opposition, _Ctx(st=[0.0, 0.0, 0.5], qrs_points=points)).value
    b = _run(m._qrs_opposition, _Ctx(st=[0.0, 0.0, 0.5], qrs_points=points * 2.0)).value
    assert b == pytest.approx(2.0 * a)
    # the deepest point sits in the middle of the loop, not in any terminal window
    assert a == pytest.approx(0.8)


# --- thesis E4: the initial QRS against the ST vector ---------------------


def test_the_initial_angle_reads_alignment_and_opposition():
    aligned = _Ctx(st=[0.0, 0.0, 0.5], initial=[0.0, 0.0, 0.4])
    opposed = _Ctx(st=[0.0, 0.0, 0.5], initial=[0.0, 0.0, -0.4])
    assert _run(m._initial_qrs_to_st_angle, aligned).value == pytest.approx(0.0, abs=1e-6)
    assert _run(m._initial_qrs_to_st_angle, opposed).value == pytest.approx(180.0)


def test_the_initial_angle_has_no_seam():
    a = [np.cos(np.radians(179.0)), 0.0, np.sin(np.radians(179.0))]
    b = [np.cos(np.radians(-179.0)), 0.0, np.sin(np.radians(-179.0))]
    ctx = _Ctx(st=np.asarray(a) * 0.5, initial=np.asarray(b) * 0.5)
    assert _run(m._initial_qrs_to_st_angle, ctx).value == pytest.approx(2.0, abs=1e-6)


def test_a_tiny_initial_vector_abstains():
    ctx = _Ctx(st=[0.0, 0.0, 0.5], qrs_max=[1.0, 0.0, 0.0], initial=[0.0, 0.0, 0.5 * MINIMUM_VECTOR_SHARE * 0.5])
    value = _run(m._initial_qrs_to_st_angle, ctx)
    assert value.value is None
    assert "initial" in (value.notes or "")


# --- documentation carries the measurements -------------------------------


def test_the_module_records_the_replaced_first_form():
    doc = m.__doc__ or ""
    assert "terminal 40 ms" in doc
    assert "replaced before any endpoint was looked at" in doc


def test_the_module_records_the_mechanism_check():
    doc = m.__doc__ or ""
    assert "0,211" in doc and "0,537" in doc
    assert "AUC 0,756" in doc


def test_the_module_records_the_refuted_cross_prediction():
    """Thesis E3's right-bundle-branch-block half ran the other way; the
    marker stays because its acute half is untested, and the spec says so."""

    doc = m.__doc__ or ""
    assert "refuted" in doc and "0,43" in doc and "0,67" in doc
    spec = next(s for s in DECISION_THESIS_SPECS if s.name == "qrs_opposition_to_st_vector_mv")
    assert "refuted in the pilot" in spec.thesis
    assert "Refuted if acute occlusion does not fall below controls" in spec.thesis


def test_the_module_records_the_secondary_end_of_the_decomposition():
    doc = m.__doc__ or ""
    assert "0,545 against 0,927" in doc
    assert "26,9 against 19,8" in doc
