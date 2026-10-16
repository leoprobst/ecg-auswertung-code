"""Five markers derived from the propagation theses.

The theses are in THESEN_ERREGUNGSAUSBREITUNG_2026-09-12.md at the project
root. These tests pin the properties that make the markers trustworthy and
the measurements that shaped them -- including the one implementation the
data rejected (the first stall-direction rule returned a value on 11,6 % of
recordings) and the thesis whose sign the data reversed.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.thesis_derived_biomarkers import (
    MINIMUM_BEATS,
    MINIMUM_STALL_MAGNITUDE_SHARE,
    THESIS_DERIVED_SPECS,
    ThesisDerivedBiomarker,
    thesis_derived_marker_ids,
)


# --- registration and panel boundaries ------------------------------------


def test_five_markers_are_registered():
    ids = thesis_derived_marker_ids()
    assert len(ids) == 5
    available = set(BiomarkerRegistry.available())
    for name in ids:
        assert name in available, name
        assert isinstance(BiomarkerRegistry.get(name), ThesisDerivedBiomarker)


def test_the_curated_panels_are_unchanged():
    from vcg_biomarker.mimic.panels import default_feature_panels

    panels = {p.panel_id: set(p.feature_ids) for p in default_feature_panels()}
    assert len(panels["primary_core"]) == 80
    assert len(panels["sentinel"]) == 240
    assert len(panels["classical_criteria"]) == 53
    new = set(thesis_derived_marker_ids())
    for panel_id in ("primary_core", "sentinel", "classical_criteria"):
        assert not (new & panels[panel_id]), panel_id
    assert new <= panels["discovery_reserve"]


def test_the_markers_are_post_outcome_and_out_of_the_combination_search():
    from vcg_biomarker.mimic.feature_roles import feature_profile

    for name in thesis_derived_marker_ids():
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis, name
        assert profile.discovery_eligible, name
        assert not profile.core_sentinel_eligible, name
        assert not profile.combination_eligible, name


def test_every_spec_names_its_thesis_and_its_refutation():
    for spec in THESIS_DERIVED_SPECS:
        assert "Thes" in spec.thesis
        assert "Refuted if" in spec.thesis, spec.name
        definition = BiomarkerRegistry.definition(spec.name)
        assert "No diagnostic cutoff is claimed." in definition.pathological_note
        assert "refuting observation" in definition.pathological_note


def test_the_theses_paper_exists_at_the_root():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    paper = root / "THESEN_ERREGUNGSAUSBREITUNG_2026-09-12.md"
    assert paper.is_file()
    text = paper.read_text(encoding="utf-8")
    for name in thesis_derived_marker_ids():
        assert name in text, name


# --- a stub context -------------------------------------------------------


class _Ctx:
    def __init__(self, points, fs=500):
        self._points = np.asarray(points, dtype=float)
        self.fs = fs

    def vcg_segment_origin_corrected(self, segment):
        return self._points if segment in ("qrs", "t") else None


def _stall(points):
    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as m

    return m._stall_vector(_Ctx(points))


# --- stall direction ------------------------------------------------------


def _loop_with_a_stall(direction, n=80):
    """A loop that sweeps a large arc quickly and lingers where it points
    along ``direction``."""

    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    t = np.linspace(0.0, np.pi, n)
    # a half-circle in the plane spanned by direction and a perpendicular
    perp = np.cross(direction, [0.0, 0.0, 1.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(direction, [0.0, 1.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    points = np.outer(np.sin(t), direction) + np.outer(1.0 - np.cos(t), perp) * 0.4
    # linger: repeat the sample where the loop points most along direction
    peak = int(np.argmax(points @ direction))
    linger = np.repeat(points[peak : peak + 1], 12, axis=0)
    linger += np.random.default_rng(0).normal(0.0, 1e-4, linger.shape)
    return np.vstack([points[:peak], linger, points[peak:]])


def test_the_stall_vector_points_where_the_loop_lingers():
    for direction in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]):
        vector = _stall(_loop_with_a_stall(direction))
        assert vector is not None
        assert float(np.dot(vector, direction)) > 0.9


def test_the_stall_vector_is_a_unit_vector():
    vector = _stall(_loop_with_a_stall([0.3, -0.5, 0.8]))
    assert vector is not None
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-9)


def test_the_stall_is_sought_only_where_the_vector_is_large():
    """The rule the data forced.

    A loop that is slowest near the origin -- as most loops are at onset and
    offset -- must not report the origin's direction. The first rule did,
    and returned a value on 11,6 % of recordings; this one searches only
    where the magnitude clears the gate.
    """

    # slow crawl near the origin, then a fast large sweep
    crawl = np.linspace(0.0, 0.02, 20)[:, None] * np.array([1.0, 0.0, 0.0])
    t = np.linspace(0.0, np.pi, 60)
    sweep = np.column_stack([np.sin(t), 1.0 - np.cos(t), np.zeros_like(t)])
    points = np.vstack([crawl, sweep + crawl[-1]])
    vector = _stall(points)
    assert vector is not None
    # the crawl points along +X near the origin; the sweep's slowest large
    # part is its far end, which points along +Y
    assert float(vector[1]) > 0.5


def test_the_stall_vector_is_invariant_to_amplitude():
    base = _loop_with_a_stall([0.0, 0.0, 1.0])
    assert np.allclose(_stall(base), _stall(base * 12.0), atol=1e-9)


def test_a_flat_loop_has_no_stall():
    assert _stall(np.zeros((40, 3))) is None


def test_a_loop_that_is_large_only_at_its_edges_has_no_interior_stall():
    """The gate is relative to the maximum and the search is interior only.

    A loop whose magnitude is large only in the first and last tenth and
    tiny in between offers no interior sample above the gate, so there is
    no stall with a defined direction. (A uniformly scaled loop, by contrast,
    keeps the same eligible set -- that is the amplitude invariance tested
    above, and the reason the gate is relative.)
    """

    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as m

    n = 100
    points = np.zeros((n, 3))
    edge = np.linspace(0.0, 1.0, 8)
    points[:8, 0] = edge
    points[-8:, 0] = edge[::-1]
    points[8:-8, 0] = 0.5 * MINIMUM_STALL_MAGNITUDE_SHARE * 0.5  # well below the gate
    points[8:-8, 1] = np.linspace(0.0, 0.001, n - 16)             # moving, but tiny
    assert m._stall_vector(_Ctx(points)) is None


# --- beat-to-beat scatter -------------------------------------------------


class _Point:
    def __init__(self, sample, confidence=1.0):
        self.sample = int(sample)
        self.confidence = float(confidence)


class _Beat:
    def __init__(self, onset, offset):
        self.r_onset = _Point(onset)
        self.r_offset = _Point(offset)


class _Delineation:
    def __init__(self, beats):
        self.beats = beats


class _RecordCtx:
    def __init__(self, record, beats, fs=500):
        self._record_vcg_xyz = np.asarray(record, dtype=float)
        self.delineation = _Delineation(beats)
        self.fs = fs


def _scatter(record, beats):
    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as m

    return m._arc_fraction_beatwise_variability(_RecordCtx(record, beats))


def _uniform_beat(n=40):
    t = np.linspace(0.0, np.pi, n)
    return np.column_stack([np.sin(t), 1.0 - np.cos(t), 0.2 * t])


def test_identical_beats_scatter_by_nothing():
    beat = _uniform_beat()
    record = np.vstack([beat] * 5)
    beats = [_Beat(i * 40, i * 40 + 39) for i in range(5)]
    value = _scatter(record, beats)
    assert value.value == pytest.approx(0.0, abs=1e-9)


def test_beats_with_different_time_structure_scatter():
    fast_late = _uniform_beat()
    # same path, traversed early-fast / late-slow
    warped = np.concatenate([np.linspace(0, 0.9, 30), np.linspace(0.9, 1.0, 10)])
    idx = (warped * 39).astype(int)
    early_fast = fast_late[idx]
    record = np.vstack([fast_late, early_fast, fast_late, early_fast])
    beats = [_Beat(i * 40, i * 40 + 39) for i in range(4)]
    value = _scatter(record, beats)
    assert value.value is not None and value.value > 0.05


def test_too_few_beats_abstain():
    beat = _uniform_beat()
    record = np.vstack([beat] * (MINIMUM_BEATS - 1))
    beats = [_Beat(i * 40, i * 40 + 39) for i in range(MINIMUM_BEATS - 1)]
    value = _scatter(record, beats)
    assert value.value is None
    assert "three beats" in (value.notes or "")


def test_a_low_confidence_beat_is_left_out():
    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as m

    beat = _uniform_beat()
    record = np.vstack([beat] * 4)
    beats = [_Beat(i * 40, i * 40 + 39) for i in range(4)]
    beats[2].r_onset = _Point(80, confidence=0.0)
    ctx = _RecordCtx(record, beats)
    assert len(m._beatwise_arc_fractions(ctx)) == 3


# --- T-loop arc fraction --------------------------------------------------


def test_the_t_arc_fraction_reads_the_t_segment():
    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as m

    points = np.zeros((41, 3))
    points[:21, 0] = np.linspace(0.0, 0.9, 21)
    points[21:, 0] = np.linspace(0.9, 1.0, 20)
    value = m._t_arc_length_fraction_at_half_time(_Ctx(points))
    assert value.value is not None and value.value > 0.8


def test_the_t_arc_fraction_abstains_without_a_t_segment():
    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as m

    class _NoT:
        fs = 500

        def vcg_segment_origin_corrected(self, segment):
            return None

    value = m._t_arc_length_fraction_at_half_time(_NoT())
    assert value.value is None


# --- documentation carries the measurements -------------------------------


def test_the_module_records_the_rule_the_data_forced():
    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as m

    doc = m.__doc__ or ""
    assert "11,6 %" in doc
    assert "97,7 %" in doc


def test_the_module_records_the_posterior_stall_and_the_reversed_sign():
    from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as m

    doc = m.__doc__ or ""
    assert "posterior" in doc and "Durrer" in doc
    assert "sign reversed" in doc
    assert "not supported" in doc
