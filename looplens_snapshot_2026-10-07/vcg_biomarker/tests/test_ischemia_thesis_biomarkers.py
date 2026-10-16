"""Three markers derived from the acute-ischaemia theses.

The theses are in THESEN_AKUTE_ISCHAEMIE_2026-09-12.md at the project root.
A fourth marker, the ST nondipolar fraction, was built, measured and
withdrawn -- no dynamic range, wrong direction -- and these tests pin that
it stays withdrawn.

Measured on 2.964 recordings: the acute endpoints are nearly empty in the
pilot cohort (five, seven and nineteen positives), so the acute end of every
thesis waits for the full run. The cross-prediction was testable and held:
the stall-to-ST angle runs 123 degrees in strict echo hypertrophy against
89 in controls, 96 in old infarction, 62 in probable acute infarction.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.ischemia_thesis_biomarkers import (
    ISCHEMIA_THESIS_SPECS,
    MINIMUM_BEATS,
    MINIMUM_VECTOR_SHARE,
    IschemiaThesisBiomarker,
    ischemia_thesis_marker_ids,
)


# --- registration and panel boundaries ------------------------------------


def test_three_markers_are_registered():
    ids = ischemia_thesis_marker_ids()
    assert len(ids) == 3
    available = set(BiomarkerRegistry.available())
    for name in ids:
        assert name in available, name
        assert isinstance(BiomarkerRegistry.get(name), IschemiaThesisBiomarker)


def test_the_withdrawn_marker_stays_withdrawn():
    """No dynamic range: 99,6-99,8 % of the eight-lead ST deviation is dipolar
    in everyone, and the residual ran the wrong way. Pinned so it is not
    re-added without re-measuring."""

    assert "lead8_st_nondipolar_fraction" not in set(BiomarkerRegistry.available())


def test_the_curated_panels_are_unchanged():
    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert len(panels["classical_criteria"]) == 53
    new = set(ischemia_thesis_marker_ids())
    for panel_id in ("primary_core", "sentinel", "classical_criteria"):
        assert not (new & panels[panel_id]), panel_id
    assert new <= panels["discovery_reserve"]


def test_the_markers_are_post_outcome_and_out_of_the_combination_search():
    from vcg_biomarker.mimic.feature_roles import feature_profile

    for name in ischemia_thesis_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.discovery_eligible, name
        assert not profile.core_sentinel_eligible, name
        assert not profile.combination_eligible, name


def test_every_spec_names_its_thesis_and_its_refutation():
    for spec in ISCHEMIA_THESIS_SPECS:
        assert "Thesis O" in spec.thesis
        assert "Refuted if" in spec.thesis, spec.name
        definition = BiomarkerRegistry.definition(spec.name)
        assert "No diagnostic cutoff is claimed." in definition.pathological_note


def test_no_marker_carries_an_amplitude_or_an_interval():
    for spec in ISCHEMIA_THESIS_SPECS:
        assert spec.unit in ("degrees", "fraction"), spec.name


def test_the_theses_paper_exists_and_names_the_markers():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "THESEN_AKUTE_ISCHAEMIE_2026-09-12.md").read_text(encoding="utf-8")
    for name in ischemia_thesis_marker_ids():
        assert name in text, name
    assert "lead8_st_nondipolar_fraction" in text   # the withdrawal is recorded


# --- the alignments -------------------------------------------------------


def _angle(a, b):
    from vcg_biomarker.core.biomarkers import ischemia_thesis_biomarkers as m

    return m._angle_deg(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def test_the_angle_is_between_the_two_vectors():
    assert _angle([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert _angle([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert _angle([1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)


def test_the_angle_has_no_seam():
    """The reason the markers are angles between vectors rather than
    differences of azimuths: two directions two degrees apart across the
    +/-180 seam must read as two degrees."""

    a = [np.cos(np.radians(179.0)), 0.0, np.sin(np.radians(179.0))]
    b = [np.cos(np.radians(-179.0)), 0.0, np.sin(np.radians(-179.0))]
    assert _angle(a, b) == pytest.approx(2.0, abs=1e-6)


def test_the_angle_is_invariant_to_amplitude():
    a, b = [0.3, -0.5, 0.8], [0.1, 0.9, -0.2]
    assert _angle(a, b) == pytest.approx(_angle(np.array(a) * 40.0, np.array(b) * 0.01))


def test_a_zero_vector_has_no_angle():
    assert _angle([0, 0, 0], [1, 0, 0]) is None


class _Ctx:
    """Stub for the two alignment markers: stall, ST, terminal QRS, QRS max."""

    def __init__(self, stall, st, terminal, qrs_max=(1.0, 0.0, 0.0), fs=500):
        self._stall = None if stall is None else np.asarray(stall, dtype=float)
        self._st = None if st is None else np.asarray(st, dtype=float)
        self._terminal = None if terminal is None else np.asarray(terminal, dtype=float)
        self._qrs_max = np.asarray(qrs_max, dtype=float)
        self.fs = fs


def _stall_to_st(ctx):
    from vcg_biomarker.core.biomarkers import ischemia_thesis_biomarkers as m
    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as t

    original_stall = t._stall_vector
    original_vec = m._vector_array
    try:
        t._stall_vector = lambda c: c._stall
        m._vector_array = lambda c, seg, mode: c._st if seg == "st" else c._qrs_max
        return m._stall_to_st_angle(ctx)
    finally:
        t._stall_vector = original_stall
        m._vector_array = original_vec


def _terminal_to_st(ctx):
    from vcg_biomarker.core.biomarkers import ischemia_thesis_biomarkers as m

    original_vec = m._vector_array
    original_win = m._window_vector
    try:
        m._vector_array = lambda c, seg, mode: c._st if seg == "st" else c._qrs_max
        m._window_vector = lambda c, seg, length_ms=40, end_aligned=False: c._terminal
        return m._terminal_qrs_to_st_angle(ctx)
    finally:
        m._vector_array = original_vec
        m._window_vector = original_win


def test_aligned_stall_and_injury_give_a_small_angle():
    """The acute-occlusion configuration: both vectors on the same region."""

    value = _stall_to_st(_Ctx(stall=[0, 0, 1], st=[0.05, 0, 0.3], terminal=None))
    assert value.value is not None and value.value < 15.0


def test_opposed_stall_and_injury_give_a_large_angle():
    """The secondary-change configuration: the ST opposes, the stall stays
    posterobasal."""

    value = _stall_to_st(_Ctx(stall=[0, 0, 1], st=[0, 0, -0.3], terminal=None))
    assert value.value == pytest.approx(180.0)


def test_a_tiny_st_vector_abstains_rather_than_returning_a_direction():
    """The correct half-availability: a normal complex has no injury current."""

    value = _stall_to_st(_Ctx(stall=[0, 0, 1], st=[0, 0, 0.5 * MINIMUM_VECTOR_SHARE * 0.5], terminal=None))
    assert value.value is None
    assert "too small" in (value.notes or "")


def test_the_terminal_alignment_reads_the_same_way():
    small = _terminal_to_st(_Ctx(stall=None, st=[0, 0.3, 0], terminal=[0, 0.4, 0.05]))
    large = _terminal_to_st(_Ctx(stall=None, st=[0, 0.3, 0], terminal=[0, -0.4, 0.05]))
    assert small.value < 15.0
    assert large.value > 165.0


def test_no_stall_means_no_alignment():
    value = _stall_to_st(_Ctx(stall=None, st=[0, 0, 0.3], terminal=None))
    assert value.value is None
    assert "stall" in (value.notes or "")


# --- beat-to-beat ST direction --------------------------------------------


def _dispersion(directions):
    from vcg_biomarker.core.biomarkers import ischemia_thesis_biomarkers as m

    class _C:
        pass

    ctx = _C()
    setattr(ctx, "_ischemia_thesis_cache", {("beat_st_directions",): [np.asarray(d, float) for d in directions]})
    return m._st_direction_dispersion(ctx)


def test_a_stable_injury_disperses_by_nothing():
    value = _dispersion([[0, 0, 1]] * 5)
    assert value.value == pytest.approx(0.0, abs=1e-9)


def test_a_turning_injury_disperses():
    value = _dispersion([[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    assert value.value is not None and value.value > 0.25


def test_the_dispersion_is_circular_and_seam_free():
    """Directions at -179 and +179 degrees in the transverse plane are two
    degrees apart and must disperse by almost nothing."""

    a = [np.cos(np.radians(179.0)), 0.0, np.sin(np.radians(179.0))]
    b = [np.cos(np.radians(-179.0)), 0.0, np.sin(np.radians(-179.0))]
    value = _dispersion([a, b] * 3)
    assert value.value < 0.01


def test_too_few_beats_abstain():
    value = _dispersion([[0, 0, 1]] * (MINIMUM_BEATS - 1))
    assert value.value is None
    assert "three beats" in (value.notes or "")


# --- documentation carries the measurements -------------------------------


def test_the_module_records_the_withdrawal_and_the_cross_prediction():
    from vcg_biomarker.core.biomarkers import ischemia_thesis_biomarkers as m

    doc = m.__doc__ or ""
    assert "withdrawn" in doc
    assert "99,6-99,8 %" in doc
    assert "123 degrees" in doc
    assert "62 in probable acute infarction" in doc


def test_the_module_states_the_acute_end_is_untested():
    from vcg_biomarker.core.biomarkers import ischemia_thesis_biomarkers as m

    doc = m.__doc__ or ""
    assert "five positives" in doc
    assert "waits for the full run" in doc
