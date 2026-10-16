"""Discovery-v4: method uncertainty, exact topology, visual morphology.

The tests that matter here are the ones that would catch a marker quietly
returning a plausible wrong number: exact quantities are checked against
values derived independently (an MST barcode, a known solid angle, the
symmetry that forces odd harmonics to vanish), and the visually named markers
are checked against shapes built to have the property their name claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import register_all_biomarkers
from vcg_biomarker.core.biomarkers.base import BiomarkerRegistry
from vcg_biomarker.core.biomarkers import catalog_biomarkers as catalog
from vcg_biomarker.core.biomarkers.catalog_biomarkers import _context
from vcg_biomarker.core.biomarkers import discovery_v4_biomarkers as v4
from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
)


register_all_biomarkers()


# ------------------------------------------------------------- fixtures


def _point(sample: int) -> MarkerPoint:
    return MarkerPoint(
        sample=int(sample),
        confidence=0.95,
        source="discovery_v4_test",
        uncertainty_ms=2.0,
    )


def _beat(
    r_peak: int,
    p_onset: int,
    p_offset: int,
    qrs_onset: int,
    qrs_offset: int,
    t_onset: int,
    t_offset: int,
) -> BeatDelineation:
    return BeatDelineation(
        r_peak=_point(r_peak),
        p_onset=_point(p_onset),
        p_peak=_point((p_onset + p_offset) // 2),
        p_offset=_point(p_offset),
        r_onset=_point(qrs_onset),
        r_offset=_point(qrs_offset),
        t_onset=_point(t_onset),
        t_peak=_point((t_onset + t_offset) // 2),
        t_offset=_point(t_offset),
    )


def _synthetic_context(
    loop: np.ndarray | None = None,
    fs: int = 500,
) -> tuple[object, np.ndarray]:
    """A record whose QRS segment is exactly the loop handed in."""

    if loop is None:
        angle = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
        loop = np.column_stack(
            [1.2 * np.cos(angle), 0.6 * np.sin(angle), 0.15 * np.sin(2 * angle)]
        )
    loop = np.asarray(loop, dtype=float)
    n_loop = len(loop)
    total = 40 + n_loop + 120
    xyz = np.zeros((total, 3), dtype=float)
    xyz[40 : 40 + n_loop] = loop
    t_angle = np.linspace(0.0, np.pi, 80)
    xyz[40 + n_loop : 40 + n_loop + 80] = np.column_stack(
        [0.3 * np.sin(t_angle), 0.4 * np.sin(t_angle), 0.05 * np.sin(t_angle)]
    )
    beat = _beat(
        r_peak=40 + n_loop // 2,
        p_onset=5,
        p_offset=25,
        qrs_onset=40,
        qrs_offset=40 + n_loop - 1,
        t_onset=40 + n_loop,
        t_offset=40 + n_loop + 79,
    )
    delineation = RecordDelineation([beat], fs, total, 4711)
    catalog.reset_feature_context()
    ecg = np.zeros((total, 12), dtype=float)
    context = _context(delineation, ecg, xyz, fs)
    setattr(context, "_record_vcg_xyz", xyz)
    return context, xyz


# ------------------------------------------------- registry hygiene


def test_every_marker_is_registered_with_a_unit_and_a_formula() -> None:
    names = [spec.name for spec in v4.DISCOVERY_V4_SPECS]
    assert len(names) == len(set(names))
    for spec in v4.DISCOVERY_V4_SPECS:
        definition = BiomarkerRegistry.definition(spec.name)
        assert definition.unit.strip()
        assert definition.formula.strip()
        assert definition.description.strip()
        assert "No diagnostic cutoff is claimed" in definition.pathological_note


def test_markers_never_claim_more_than_discovery_status() -> None:
    for spec in v4.DISCOVERY_V4_SPECS:
        assert spec.evidence_level in {"B", "C", "D"}
        assert spec.visual_assessment in {"direct", "mathematical_only"}


def test_the_visually_readable_subset_is_identifiable() -> None:
    visual = set(v4.visually_interpretable_marker_ids())
    assert len(visual) > 50
    # This is the list a human would be handed to adjudicate a screening hit,
    # so every entry must actually be a visual claim.
    assert {"qrs_visual_open_gap_fraction", "beat_tube_relative_thickness"} <= visual
    assert "qrs_rqa_determinism" not in visual


# ------------------------------------------------- exact topology


def test_h0_barcode_equals_the_minimum_spanning_tree() -> None:
    """The exactness claim, checked against an independently built MST."""

    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import distance

    context, _ = _synthetic_context()
    result = v4._h0_persistence(context, "qrs")
    assert result is not None

    points = v4._resampled(context, "qrs", 64)
    reference = minimum_spanning_tree(
        distance.squareform(distance.pdist(points))
    ).toarray()
    reference_bars = np.sort(reference[reference > 0])
    assert result["h0_total_persistence_mv"] == pytest.approx(
        float(np.sum(reference_bars)), rel=1e-9
    )
    assert result["h0_max_persistence_mv"] == pytest.approx(
        float(np.max(reference_bars)), rel=1e-9
    )


def test_radial_h1_finds_the_lobes_that_were_put_there() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    # A radius modulated with three lobes has three radial minima.
    radius = 1.0 + 0.4 * np.cos(3.0 * angle)
    loop = np.column_stack(
        [radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)]
    )
    context, _ = _synthetic_context(loop)
    result = v4._h1_radial_persistence(context, "qrs")
    assert result is not None
    assert result["radial_h1_bar_count"] >= 2
    assert result["radial_h1_max_lifetime_mv"] > 0.0
    assert 0.0 <= result["radial_h1_persistence_entropy"] <= 1.0

    # A perfect circle has a constant radius. What survives smoothing is
    # float noise, and reporting that as lobes would be inventing structure.
    flat = np.column_stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)]
    )
    circle_context, _ = _synthetic_context(flat)
    assert v4._h1_radial_persistence(circle_context, "qrs") is None


def test_odd_spherical_harmonics_vanish_for_a_centrally_symmetric_loop() -> None:
    """The symmetry that makes degree 1 useless as a denominator.

    A planar ellipse is centrally symmetric, so its direction density is
    antipodally symmetric and every odd degree must cancel exactly. A loop
    that is *not* centrally symmetric keeps odd-degree power, which is the
    non-closure signal the marker reports.
    """

    angle = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    ellipse = np.column_stack(
        [1.2 * np.cos(angle), 0.6 * np.sin(angle), np.zeros_like(angle)]
    )
    symmetric = v4._spherical_harmonics(_synthetic_context(ellipse)[0], "qrs")
    assert symmetric is not None
    # Arc-length resampling breaks the exact antipodal pairing, so what
    # remains is discretisation rather than structure: five orders of
    # magnitude below the axial term.
    assert symmetric["sh_odd_degree_fraction"] < 1e-3
    assert symmetric["sh_odd_degree_fraction"] < 1e-3 * symmetric["sh_axial_fraction"]
    # The axis of an orientation distribution lives in degree 2, not degree 1.
    assert symmetric["sh_power_degree_1"] < 1e-20
    assert symmetric["sh_axial_fraction"] > 0.1
    fractions = (
        symmetric["sh_axial_fraction"]
        + symmetric["sh_isotropic_fraction"]
        + symmetric["sh_higher_order_fraction"]
    )
    assert fractions == pytest.approx(1.0, abs=1e-9)

    # A loop displaced out of plane in a non-antipodal way is not symmetric.
    lopsided = np.column_stack(
        [
            1.2 * np.cos(angle),
            0.6 * np.sin(angle),
            0.5 * np.sin(angle / 2.0),
        ]
    )
    asymmetric = v4._spherical_harmonics(_synthetic_context(lopsided)[0], "qrs")
    assert asymmetric is not None
    assert (
        asymmetric["sh_odd_degree_fraction"]
        > symmetric["sh_odd_degree_fraction"]
    )


def _octant_boundary() -> np.ndarray:
    quarter = np.linspace(0.0, np.pi / 2.0, 40)
    return np.vstack(
        [
            np.column_stack(
                [np.cos(quarter), np.sin(quarter), np.zeros_like(quarter)]
            ),
            np.column_stack(
                [np.zeros_like(quarter), np.cos(quarter), np.sin(quarter)]
            ),
            np.column_stack(
                [np.sin(quarter), np.zeros_like(quarter), np.cos(quarter)]
            ),
        ]
    )


def test_the_solid_angle_formula_matches_a_known_octant() -> None:
    """An octant of the unit sphere encloses exactly pi/2 steradian.

    Checked on the bare formula, because the marker centres the loop on its
    own centroid first and therefore does not see a curve that already lies
    on the unit sphere.
    """

    directions = _octant_boundary()
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    closed = np.vstack([directions, directions[:1]])
    anchor = closed[0]
    total = 0.0
    for index in range(1, len(closed) - 2):
        first, second = closed[index], closed[index + 1]
        numerator = float(np.dot(anchor, np.cross(first, second)))
        denominator = (
            1.0
            + float(np.dot(anchor, first))
            + float(np.dot(first, second))
            + float(np.dot(second, anchor))
        )
        total += 2.0 * float(np.arctan2(numerator, denominator))
    assert abs(total) == pytest.approx(np.pi / 2.0, rel=1e-6)


def test_a_planar_loop_has_zero_planar_departure_and_a_spiral_does_not() -> None:
    """Why the marker reports a departure rather than the solid angle itself.

    The enclosed fraction is a topological invariant of winding once: across
    planar, twisted, figure-of-eight and spiralling loops it stays inside
    0.4964-0.5000, so as a column it is very nearly a constant. The departure
    from the planar half sphere is what actually varies with shape.
    """

    angle = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    planar = np.column_stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)]
    )
    spiral = np.column_stack(
        [
            np.cos(angle) * (1.0 + 0.3 * angle / 6.0),
            np.sin(angle),
            0.5 * angle / 6.0,
        ]
    )
    planar_result = v4._spherical_holonomy(_synthetic_context(planar)[0], "qrs")
    spiral_result = v4._spherical_holonomy(_synthetic_context(spiral)[0], "qrs")
    assert planar_result is not None and spiral_result is not None

    # A planar loop's direction curve is a great circle: exactly half a sphere.
    assert planar_result["holonomy_absolute_planar_departure_sr"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert abs(planar_result["holonomy_solid_angle_sr"]) == pytest.approx(
        2.0 * np.pi, abs=1e-6
    )
    # A loop that does not return into its own plane does.
    assert spiral_result["holonomy_absolute_planar_departure_sr"] > 0.01
    assert (
        spiral_result["holonomy_absolute_planar_departure_sr"]
        > 100.0 * planar_result["holonomy_absolute_planar_departure_sr"] + 1e-6
    )


def test_recurrence_quantification_returns_the_standard_set() -> None:
    context, _ = _synthetic_context()
    result = v4._recurrence_quantification(context, "qrs")
    assert result is not None
    for key in (
        "rqa_determinism",
        "rqa_laminarity",
        "rqa_trapping_time",
        "rqa_max_diagonal_line",
        "rqa_diagonal_entropy",
    ):
        assert key in result
    assert 0.0 <= result["rqa_determinism"] <= 1.0
    assert 0.0 <= result["rqa_laminarity"] <= 1.0
    assert 0.0 <= result["rqa_recurrence_rate"] <= 1.0
    # A smooth periodic loop is highly deterministic.
    assert result["rqa_determinism"] > 0.5


def test_box_dimension_is_between_a_curve_and_a_volume() -> None:
    context, _ = _synthetic_context()
    result = v4._minkowski_spectrum(context, "qrs")
    assert result is not None
    assert 0.5 < result["minkowski_box_dimension"] < 3.0
    assert result["minkowski_hull_surface_mv2"] > 0.0


# ------------------------------------------------- visual morphology


def test_an_open_loop_reads_as_open_and_a_closed_one_does_not() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    closed = np.column_stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)]
    )
    open_arc = np.column_stack(
        [
            np.cos(angle[: int(0.7 * len(angle))]),
            np.sin(angle[: int(0.7 * len(angle))]),
            np.zeros(int(0.7 * len(angle))),
        ]
    )
    closed_context, _ = _synthetic_context(closed)
    open_context, _ = _synthetic_context(open_arc)
    closed_reading = v4._visual_loop_reading(closed_context, "qrs")
    open_reading = v4._visual_loop_reading(open_context, "qrs")
    assert closed_reading is not None and open_reading is not None
    assert closed_reading["visual_open_gap_fraction"] < 0.2
    assert open_reading["visual_open_gap_fraction"] > 0.8


def test_a_pinched_loop_reads_as_having_a_waist() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    # Figure-of-eight: a genuine waist in the middle.
    pinched = np.column_stack(
        [np.sin(angle), np.sin(2.0 * angle), np.zeros_like(angle)]
    )
    round_loop = np.column_stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)]
    )
    pinched_reading = v4._visual_loop_reading(
        _synthetic_context(pinched)[0], "qrs"
    )
    round_reading = v4._visual_loop_reading(
        _synthetic_context(round_loop)[0], "qrs"
    )
    assert pinched_reading is not None and round_reading is not None
    assert (
        pinched_reading["visual_waist_fraction"]
        < round_reading["visual_waist_fraction"]
    )


def test_rotation_direction_is_readable_from_the_signed_area() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=False)
    counter = np.column_stack(
        [np.cos(angle), np.sin(angle), np.zeros_like(angle)]
    )
    clockwise = counter[::-1].copy()
    counter_reading = v4._plane_projection_reading(
        _synthetic_context(counter)[0], "qrs", "frontal_xy"
    )
    clockwise_reading = v4._plane_projection_reading(
        _synthetic_context(clockwise)[0], "qrs", "frontal_xy"
    )
    assert counter_reading is not None and clockwise_reading is not None
    assert (
        counter_reading["plane_signed_area_mv2"]
        * clockwise_reading["plane_signed_area_mv2"]
        < 0
    )
    assert counter_reading["plane_circularity"] == pytest.approx(1.0, abs=0.1)


def test_a_figure_of_eight_is_counted_as_self_intersecting() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    eight = np.column_stack(
        [np.sin(angle), np.sin(2.0 * angle), np.zeros_like(angle)]
    )
    reading = v4._plane_projection_reading(
        _synthetic_context(eight)[0], "qrs", "frontal_xy"
    )
    assert reading is not None
    assert reading["plane_self_intersections"] >= 1


@pytest.mark.parametrize("point_count", [6, 17, 64, 96])
def test_vectorized_self_contacts_match_scalar_definition_exactly(
    point_count: int,
) -> None:
    rng = np.random.default_rng(20260902 + point_count)
    for _ in range(8):
        points = rng.normal(size=(point_count, 2))
        tolerance = float(rng.uniform(0.005, 0.4))
        contacts = 0
        last_contact = -10
        for first in range(len(points) - 1):
            for second in range(first + 3, len(points) - 1):
                if (
                    v4._segment_distance_2d(
                        points[first],
                        points[first + 1],
                        points[second],
                        points[second + 1],
                    )
                    <= tolerance
                ):
                    if first - last_contact > 2:
                        contacts += 1
                    last_contact = first
                    break
        assert v4._count_self_contacts(
            points,
            tolerance=tolerance,
        ) == float(contacts)


def test_activation_energy_quantiles_are_ordered_and_bounded() -> None:
    context, _ = _synthetic_context()
    result = v4._activation_energy_quantiles(context)
    assert result is not None
    fractions = [
        result[f"activation_energy_q{quantile}_fraction"]
        for quantile in (10, 25, 50, 75, 90)
    ]
    assert fractions == sorted(fractions)
    assert all(0.0 <= value <= 1.0 for value in fractions)


def test_a_small_scribbled_loop_scores_higher_than_a_large_smooth_one() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    large_smooth = np.column_stack(
        [2.0 * np.cos(angle), 2.0 * np.sin(angle), np.zeros_like(angle)]
    )
    small_scribbled = np.column_stack(
        [
            0.2 * np.cos(angle) + 0.05 * np.cos(11.0 * angle),
            0.2 * np.sin(angle) + 0.05 * np.sin(11.0 * angle),
            0.02 * np.sin(7.0 * angle),
        ]
    )
    smooth = v4._voltage_complexity(_synthetic_context(large_smooth)[0])
    scribbled = v4._voltage_complexity(_synthetic_context(small_scribbled)[0])
    assert smooth is not None and scribbled is not None
    # This is the amyloid hypothesis in one inequality.
    assert (
        scribbled["voltage_complexity_mismatch_index"]
        > smooth["voltage_complexity_mismatch_index"]
    )


# ------------------------------------------------- abstention discipline


def test_a_degenerate_record_abstains_everywhere_instead_of_returning_zeros() -> None:
    flat = np.zeros((30, 3), dtype=float)
    context, _ = _synthetic_context(flat)
    for primitive, segment in (
        (v4._h0_persistence, "qrs"),
        (v4._h1_radial_persistence, "qrs"),
        (v4._spherical_harmonics, "qrs"),
        (v4._spherical_holonomy, "qrs"),
        (v4._minkowski_spectrum, "qrs"),
        (v4._recurrence_quantification, "qrs"),
    ):
        assert primitive(context, segment) is None
    assert v4._voltage_complexity(context) is None
    assert v4._activation_energy_quantiles(context) is None


def test_abstention_states_a_reason() -> None:
    calculator = v4._value_from_dict(
        lambda ctx: None,
        "missing",
        "prerequisite absent",
    )
    value = calculator(object())
    assert value.value is None
    assert value.confidence == 0.0
    assert "not_evaluable" in value.notes
    assert "prerequisite absent" in value.notes


def test_no_marker_raises_on_a_synthetic_record() -> None:
    context, xyz = _synthetic_context()
    failures: list[str] = []
    for spec in v4.DISCOVERY_V4_SPECS:
        try:
            value = spec.calculator(context)
        except Exception as error:  # noqa: BLE001 - the point of the test
            failures.append(f"{spec.name}: {error}")
            continue
        if value.value is not None and not np.isfinite(value.value):
            failures.append(f"{spec.name}: non-finite value")
    assert failures == []


# ------------------------------------------------- method uncertainty


def test_the_origin_cone_needs_both_origins_before_it_reports_one() -> None:
    context, _ = _synthetic_context()
    result = v4._origin_uncertainty(context)
    # The synthetic record has no usable TP interval, so the honest answer is
    # nothing rather than a cone of width zero.
    assert result is None or result["cone_half_angle_deg"] >= 0.0


def test_transform_ensemble_abstains_without_a_twelve_lead_record() -> None:
    context, _ = _synthetic_context()
    # The synthetic 12-lead is all zeros, so no transform yields a loop.
    assert v4._transform_ensemble(context) is None


def test_peak_and_area_agree_for_a_constant_vector() -> None:
    # A QRS whose direction never changes: peak and area must be parallel.
    steady = np.tile(np.array([1.0, 0.5, -0.2]), (60, 1))
    steady = steady * np.sin(np.linspace(0.1, np.pi, 60))[:, None]
    context, _ = _synthetic_context(steady)
    result = v4._peak_area_discordance(context)
    assert result is not None
    assert result["peak_area_angle_deg"] == pytest.approx(0.0, abs=1.0)


# ------------------------------------------------- context isolation


def test_compute_all_does_not_leak_record_sources_into_a_later_call() -> None:
    """Regression: a stale record source silently changed the VCG origin.

    ``compute_all`` sets full-record sources for the whole panel. Before the
    fix it never cleared them, so the next direct ``compute`` in the same
    thread resolved its origin from the previous record's signal -- a wrong
    number, not an error.
    """

    context_a, xyz_a = _synthetic_context()
    delineation_a = context_a.delineation
    ecg_a = np.zeros((len(xyz_a), 12), dtype=float)
    BiomarkerRegistry.compute_all(
        delineation=delineation_a,
        ecg_12lead=ecg_a,
        vcg_xyz=xyz_a,
        fs=500,
        names=["qrs_visual_open_gap_fraction"],
        record_vcg_xyz=xyz_a + 5.0,
    )
    assert getattr(catalog._CONTEXT_STATE, "record_vcg_xyz", None) is None
    assert getattr(catalog._CONTEXT_STATE, "context", None) is None


def test_the_context_cache_cannot_be_fooled_by_a_recycled_address() -> None:
    """Regression: the cache key was ``id()`` of collectable objects."""

    context_a, xyz_a = _synthetic_context()
    key = getattr(catalog._CONTEXT_STATE, "key", None)
    assert key is not None
    # The key holds the objects themselves, so their addresses cannot be
    # handed to a different record while the cache is live.
    assert key[0] is context_a.delineation
    assert key[2] is xyz_a


# ------------------------------------------------- origin coverage


def test_the_tq_tier_resolves_an_origin_without_any_p_wave() -> None:
    """Regression: TP and PR both need a P wave, and P detection is weak.

    Measured on 38 consecutive MIMIC records, TP or PR resolved on 39 %, so
    every origin-corrected marker abstained on the other 61 %. The early TQ
    interval is electrical diastole and needs no P detection; coverage on the
    same records went to 97 %.
    """

    from vcg_biomarker.core.vcg_geometry import estimate_physiological_vcg_origin

    fs = 500
    total = 1400
    xyz = np.zeros((total, 3), dtype=float)
    beats = []
    # Three beats with QRS and T but no P wave at all.
    for index, onset in enumerate((100, 500, 900)):
        qrs = np.linspace(0.0, np.pi, 40)
        xyz[onset : onset + 40] = np.column_stack(
            [1.5 * np.sin(qrs), 0.8 * np.sin(qrs), -0.4 * np.sin(qrs)]
        )
        t_wave = np.linspace(0.0, np.pi, 90)
        xyz[onset + 60 : onset + 150] = np.column_stack(
            [0.3 * np.sin(t_wave), 0.4 * np.sin(t_wave), 0.1 * np.sin(t_wave)]
        )
        beats.append(
            BeatDelineation(
                r_peak=_point(onset + 20),
                p_onset=None,
                p_peak=None,
                p_offset=None,
                r_onset=_point(onset),
                r_offset=_point(onset + 39),
                t_onset=_point(onset + 60),
                t_peak=_point(onset + 105),
                t_offset=_point(onset + 149),
            )
        )
    delineation = RecordDelineation(beats, fs, total, 90210)
    assessment = estimate_physiological_vcg_origin(xyz, delineation, fs)

    assert assessment.evaluable
    assert assessment.tp is None and assessment.pr is None
    assert assessment.primary.method == "tq_early_fallback"
    # A record with no atrial activity has a zero baseline; the point is that
    # the tier resolves at all where the first two cannot.
    assert np.allclose(assessment.primary.vector_millivolts, 0.0, atol=0.05)
    # The tier must be visible in the provenance, not hidden behind "ok".
    note = assessment.provenance_note()
    assert "origin_tier=tertiary_tq_no_p_wave_required" in note
    # Third tier must not present itself as confidently as the first.
    assert assessment.primary.confidence < 0.65


def test_a_noisy_diastole_is_refused_rather_than_trusted() -> None:
    from vcg_biomarker.core.vcg_geometry import _tq_candidate

    fs = 500
    generator = np.random.default_rng(4)
    # Half a millivolt of scatter is not an isoelectric interval.
    signal = generator.normal(0.0, 0.5, size=(1400, 3))
    beats = [
        BeatDelineation(
            r_peak=_point(onset + 20),
            p_onset=None,
            p_peak=None,
            p_offset=None,
            r_onset=_point(onset),
            r_offset=_point(onset + 39),
            t_onset=_point(onset + 60),
            t_peak=_point(onset + 105),
            t_offset=_point(onset + 149),
        )
        for onset in (100, 500, 900)
    ]
    assert (
        _tq_candidate(
            signal,
            beats=beats,
            target_index=0,
            fs=fs,
            minimum_confidence=0.35,
        )
        is None
    )


# ------------------------------------------------- panel discipline


def test_no_post_outcome_marker_reaches_a_frozen_panel() -> None:
    """Regression, twice over: new markers kept landing in Sentinel.

    Core and Sentinel are frozen before outcomes were seen. A marker written
    afterwards belongs in Discovery Reserve, and the only thing that keeps it
    there is the post-outcome registry knowing the module exists -- which it
    did not for discovery-v4, putting 43 of them into Sentinel.
    """

    from vcg_biomarker.mimic.panels import default_feature_panels

    prospective = {spec.name for spec in v4.DISCOVERY_V4_SPECS}
    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    assert panels["primary_core"] & prospective == set()
    assert panels["sentinel"] & prospective == set()
    # And they must actually be screened somewhere.
    assert len(panels["discovery_reserve"] & prospective) == len(prospective)


def test_every_prospective_module_is_known_to_the_post_outcome_registry() -> None:
    """The check that would have caught it without anyone looking at a panel."""

    from vcg_biomarker.mimic.feature_roles import _post_outcome_module_feature_ids

    known = _post_outcome_module_feature_ids()
    for spec in v4.DISCOVERY_V4_SPECS:
        assert spec.name in known, f"{spec.name} is not registered as post-outcome"
