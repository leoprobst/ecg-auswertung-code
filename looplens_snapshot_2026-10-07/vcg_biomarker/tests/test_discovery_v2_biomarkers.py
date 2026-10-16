"""Scientific and numerical guards for the prospective Discovery-v2 block."""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers import discovery_v2_biomarkers as v2
from vcg_biomarker.core.biomarkers.catalog_biomarkers import _context, reset_feature_context
from vcg_biomarker.core.biomarkers.discovery_v2_biomarkers import (
    DISCOVERY_V2_SPECS,
    DISCOVERY_V2_VERSION,
    INDEPENDENT_8_LEADS,
    _lead8_spectrum,
    _segment,
    register_discovery_v2_biomarkers,
)
from vcg_biomarker.core.data_loader import LEAD_MAP
from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
    RhythmAssessment,
    WaveStatus,
)
from vcg_biomarker.mimic.feature_roles import feature_profile
from vcg_biomarker.mimic.panels import default_feature_panels


def _point(sample: int) -> MarkerPoint:
    return MarkerPoint(sample=int(sample), confidence=1.0, source="synthetic")


def _record(
    *,
    fs: int = 500,
    n_beats: int = 10,
    t_angle_step_deg: float = 5.0,
    add_nondipolar_t: bool = True,
) -> tuple[RecordDelineation, np.ndarray, np.ndarray, int]:
    beat_samples = int(round(0.8 * fs))
    total = beat_samples * n_beats
    vcg = np.zeros((total, 3), dtype=float)
    ecg = np.zeros((total, 12), dtype=float)
    beats = []

    for beat_index in range(n_beats):
        offset = beat_index * beat_samples
        p_on, p_off = offset + int(0.10 * fs), offset + int(0.20 * fs)
        q_on, q_off = offset + int(0.26 * fs), offset + int(0.35 * fs)
        t_on, t_peak, t_off = (
            offset + int(0.41 * fs),
            offset + int(0.50 * fs),
            offset + int(0.62 * fs),
        )
        beat = BeatDelineation(
            r_peak=_point(offset + int(0.30 * fs)),
            p_onset=_point(p_on),
            p_peak=_point(offset + int(0.15 * fs)),
            p_offset=_point(p_off),
            q_peak=_point(offset + int(0.28 * fs)),
            r_onset=_point(q_on),
            r_offset=_point(q_off),
            s_peak=_point(offset + int(0.34 * fs)),
            j_point=_point(q_off),
            t_onset=_point(t_on),
            t_peak=_point(t_peak),
            t_offset=_point(t_off),
            p_status=WaveStatus.PRESENT,
            qrs_status=WaveStatus.PRESENT,
            t_status=WaveStatus.PRESENT,
            morphology_cluster_id=0,
        )
        beats.append(beat)

        # Visually non-trivial but deterministic P, QRS and T loops.
        p_phase = np.linspace(0.0, np.pi, p_off - p_on + 1)
        vcg[p_on : p_off + 1] += np.column_stack(
            (0.10 * np.sin(p_phase), 0.025 * np.sin(2 * p_phase), 0.01 * np.sin(3 * p_phase))
        )
        q_phase = np.linspace(0.0, 2.0 * np.pi, q_off - q_on + 1)
        vcg[q_on : q_off + 1] += np.column_stack(
            (
                0.9 * np.sin(q_phase),
                0.45 * np.sin(2.0 * q_phase) + 0.08 * (1.0 - np.cos(q_phase)),
                0.18 * np.sin(3.0 * q_phase),
            )
        )
        t_phase = np.linspace(0.0, np.pi, t_off - t_on + 1)
        angle = np.radians(beat_index * t_angle_step_deg)
        radial = 0.34 * np.sin(t_phase)
        # A small non-planar beat-dependent displacement makes the peak cloud
        # genuinely three-dimensional without changing delineation.
        z_factor = 0.04 * np.sin(2.0 * angle) + 0.015 * np.cos(3.0 * angle)
        vcg[t_on : t_off + 1] += np.column_stack(
            (
                radial * np.cos(angle),
                radial * np.sin(angle) + 0.035 * np.sin(2.0 * t_phase),
                radial * z_factor + 0.018 * np.sin(3.0 * t_phase),
            )
        )

        # Eight algebraically independent inputs: I, II, V1..V6.  Three
        # morphology bases produce a dipolar signal; the optional fourth base
        # is deliberately orthogonal and must increase lambda4..lambda8.
        local = slice(offset, offset + beat_samples)
        x, y, z = vcg[local].T
        basis = [x, y, z]
        fourth = np.zeros(beat_samples)
        fourth[t_on - offset : t_off - offset + 1] = 0.09 * np.sin(5.0 * t_phase)
        weights = np.asarray(
            [
                [1.0, 0.2, 0.1, 0.2],
                [0.2, 1.0, 0.1, -0.3],
                [0.8, -0.2, 0.3, 0.5],
                [-0.4, 0.9, 0.1, -0.4],
                [0.5, 0.3, -0.8, 0.7],
                [-0.6, 0.1, 0.9, -0.6],
                [0.1, -0.7, 0.6, 0.8],
                [0.3, 0.4, 0.7, -0.2],
            ]
        )
        source = np.column_stack((*basis, fourth if add_nondipolar_t else np.zeros_like(fourth)))
        independent = source @ weights.T
        for column, lead in enumerate(INDEPENDENT_8_LEADS):
            ecg[local, LEAD_MAP[lead]] = independent[:, column]
        # Algebraically dependent leads are populated but must never enter the
        # eight-lead SVD.
        ecg[local, LEAD_MAP["III"]] = ecg[local, LEAD_MAP["II"]] - ecg[local, LEAD_MAP["I"]]
        ecg[local, LEAD_MAP["aVR"]] = -0.5 * (
            ecg[local, LEAD_MAP["I"]] + ecg[local, LEAD_MAP["II"]]
        )
        ecg[local, LEAD_MAP["aVL"]] = ecg[local, LEAD_MAP["I"]] - 0.5 * ecg[local, LEAD_MAP["II"]]
        ecg[local, LEAD_MAP["aVF"]] = ecg[local, LEAD_MAP["II"]] - 0.5 * ecg[local, LEAD_MAP["I"]]

    delineation = RecordDelineation(
        beats=beats,
        fs=fs,
        n_samples=total,
        ecg_id=1,
        quality_score=1.0,
        rhythm=RhythmAssessment(confidence=1.0),
    )
    return delineation, ecg, vcg, fs


def _compute(names, record=None):
    delineation, ecg, vcg, fs = record or _record()
    return {
        result.name: result
        for result in BiomarkerRegistry.compute_all(
            delineation=delineation,
            ecg_12lead=ecg,
            vcg_xyz=vcg,
            fs=fs,
            names=list(names),
            record_ecg_12lead=ecg,
            record_vcg_xyz=vcg,
        )
    }


def test_specs_are_unique_scalar_lowercase_and_provenanced() -> None:
    names = [spec.name for spec in DISCOVERY_V2_SPECS]
    assert len(names) >= 100
    assert len(names) == len(set(names))
    assert all(name == name.casefold() for name in names)
    assert not any("sai" in name for name in names)  # central GEH agent owns SAI
    for spec in DISCOVERY_V2_SPECS:
        assert spec.unit not in {"mixed", "vector", "array", "categorical"}, spec.name
        assert spec.formula.strip(), spec.name
        assert spec.evidence_level in {"A", "B", "C", "D"}, spec.name
        assert spec.evidence_note.strip(), spec.name
        assert spec.references, spec.name
        assert all(reference.startswith("https://") for reference in spec.references), spec.name


def test_every_spec_is_registered_and_registration_is_idempotent() -> None:
    names = {spec.name for spec in DISCOVERY_V2_SPECS}
    assert names <= set(BiomarkerRegistry.available())
    register_discovery_v2_biomarkers()
    assert names <= set(BiomarkerRegistry.available())
    for name in list(names)[:20]:
        definition = BiomarkerRegistry.definition(name)
        assert DISCOVERY_V2_VERSION in definition.notes
        assert definition.references


def test_entire_block_is_discovery_only() -> None:
    names = {spec.name for spec in DISCOVERY_V2_SPECS}
    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    assert not names.intersection(panels["primary_core"])
    assert not names.intersection(panels["sentinel"])
    assert names <= panels["discovery_reserve"]
    for name in names:
        profile = feature_profile(name)
        assert profile.discovery_eligible
        assert profile.post_outcome_hypothesis
        assert not profile.core_sentinel_eligible
        assert not profile.combination_eligible


def test_no_discovery_v2_marker_raises_on_nontrivial_synthetic_record() -> None:
    results = _compute([spec.name for spec in DISCOVERY_V2_SPECS])
    errors = {
        name: result.computation_notes
        for name, result in results.items()
        if str(result.computation_notes).startswith("error:")
    }
    assert not errors
    assert len(results) == len(DISCOVERY_V2_SPECS)
    required = (
        "qrs_area_vector_magnitude_mv_ms",
        "peak_svg_magnitude_mv",
        "lead8_t_wave_residuum_fraction",
        "lead8_qrs_nondipolar_fraction",
        "tt_prime_mean_deg",
        "vg_rvpo_mv_ms",
        "qrs_t_interloop_distance_mean_mv",
        "t_peak_cloud_convex_hull_volume_mv3",
    )
    assert all(results[name].value is not None for name in required)


def test_area_components_have_millivolt_millisecond_dimensions() -> None:
    record = _record()
    delineation, _, vcg, fs = record
    results = _compute(
        [
            "qrs_area_vector_x_mv_ms",
            "qrs_area_vector_y_mv_ms",
            "qrs_area_vector_z_mv_ms",
            "qrs_area_vector_magnitude_mv_ms",
        ],
        record,
    )
    beat = delineation.representative_beat
    start, end = beat.r_onset.sample, beat.r_offset.sample
    expected = np.trapezoid(vcg[start : end + 1], dx=1000.0 / fs, axis=0)
    observed = np.asarray(
        [
            results["qrs_area_vector_x_mv_ms"].value,
            results["qrs_area_vector_y_mv_ms"].value,
            results["qrs_area_vector_z_mv_ms"].value,
        ]
    )
    assert observed == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert results["qrs_area_vector_magnitude_mv_ms"].value == pytest.approx(
        np.linalg.norm(expected), rel=1e-10
    )


def test_peak_svg_is_vector_sum_of_spatial_qrs_and_t_peaks() -> None:
    record = _record()
    delineation, _, vcg, _ = record
    results = _compute(
        ["peak_svg_x_mv", "peak_svg_y_mv", "peak_svg_z_mv", "peak_svg_magnitude_mv"],
        record,
    )
    beat = delineation.representative_beat
    qrs = vcg[beat.r_onset.sample : beat.r_offset.sample + 1]
    t_wave = vcg[beat.t_onset.sample : beat.t_offset.sample + 1]
    expected = qrs[np.argmax(np.linalg.norm(qrs, axis=1))] + t_wave[
        np.argmax(np.linalg.norm(t_wave, axis=1))
    ]
    observed = np.asarray(
        [results["peak_svg_x_mv"].value, results["peak_svg_y_mv"].value, results["peak_svg_z_mv"].value]
    )
    assert observed == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert results["peak_svg_magnitude_mv"].value == pytest.approx(np.linalg.norm(expected))


def test_true_lead8_svd_excludes_the_four_dependent_leads() -> None:
    base = _record()
    delineation, ecg, vcg, fs = base
    contaminated = ecg.copy()
    rng = np.random.default_rng(42)
    for lead in ("III", "aVR", "aVL", "aVF"):
        contaminated[:, LEAD_MAP[lead]] += rng.normal(0.0, 1000.0, len(contaminated))
    names = ["lead8_t_wave_residuum_fraction", "lead8_qrs_nondipolar_fraction"]
    clean = _compute(names, base)
    dirty = _compute(names, (delineation, contaminated, vcg, fs))
    for name in names:
        assert dirty[name].value == pytest.approx(clean[name].value, rel=1e-12, abs=1e-12)


def test_true_lead8_twr_detects_a_fourth_independent_component() -> None:
    name = "lead8_t_wave_residuum_fraction"
    rank3 = _compute([name], _record(add_nondipolar_t=False))[name].value
    rank4 = _compute([name], _record(add_nondipolar_t=True))[name].value
    assert rank3 is not None and rank4 is not None
    assert rank3 < 1e-10
    assert rank4 > rank3 + 1e-5


def test_tt_prime_tracks_known_beatwise_t_axis_rotation() -> None:
    result = _compute(["tt_prime_mean_deg"], _record(t_angle_step_deg=8.0))[
        "tt_prime_mean_deg"
    ]
    assert result.value is not None
    assert result.value == pytest.approx(8.0, abs=1.5)


def test_dynamic_vcg_crosspoint_alignment_removes_beatwise_baseline_drift() -> None:
    record = _record(t_angle_step_deg=8.0)
    delineation, ecg, vcg, fs = record
    drifted = vcg.copy()
    beat_samples = len(vcg) // len(delineation.beats)
    for beat_index in range(len(delineation.beats)):
        start = beat_index * beat_samples
        stop = start + beat_samples
        drifted[start:stop] += np.asarray(
            [0.7 * beat_index, -0.3 * beat_index, 0.2 * ((-1) ** beat_index)]
        )
    names = [
        "tt_prime_mean_deg",
        "r_peak_cloud_rms_radius_mv",
        "t_peak_cloud_rms_radius_mv",
        "qrs_t_interloop_distance_mean_mv",
    ]
    original = _compute(names, record)
    changed = _compute(names, (delineation, ecg, drifted, fs))
    for name in names:
        assert changed[name].value == pytest.approx(
            original[name].value,
            rel=2e-9,
            abs=2e-9,
        ), name
    assert original["t_peak_cloud_rms_radius_mv"].confidence < 0.65
    assert "published peak-cloud protocol used 30" in original[
        "t_peak_cloud_rms_radius_mv"
    ].computation_notes


def test_vg_rvpo_uses_the_published_projection_coefficients() -> None:
    names = [
        "qrs_area_vector_x_mv_ms", "qrs_area_vector_y_mv_ms", "qrs_area_vector_z_mv_ms",
        "t_area_vector_x_mv_ms", "t_area_vector_y_mv_ms", "t_area_vector_z_mv_ms",
        "vg_rvpo_mv_ms",
    ]
    result = _compute(names)
    vg = np.asarray(
        [
            result[f"qrs_area_vector_{axis}_mv_ms"].value
            + result[f"t_area_vector_{axis}_mv_ms"].value
            for axis in "xyz"
        ]
    )
    coefficients = np.asarray([-0.807526151172, 0.404508497187, 0.376555628452])
    assert result["vg_rvpo_mv_ms"].value == pytest.approx(float(coefficients @ vg), rel=1e-12)


def test_shape_descriptors_are_invariant_to_rigid_similarity_transform() -> None:
    record = _record()
    delineation, ecg, vcg, fs = record
    angle = np.radians(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = 2.7 * (vcg @ rotation.T) + np.asarray([4.0, -3.0, 2.0])
    names = [
        "qrs_normalized_open_loop_gap",
        "qrs_loop_graph_fiedler_value",
        "qrs_loop_graph_spectral_entropy",
        "qrs_gauss_writhe_proxy",
        "qrs_gauss_absolute_entanglement",
    ]
    original = _compute(names, record)
    changed = _compute(names, (delineation, ecg, transformed, fs))
    for name in names:
        assert original[name].value is not None, name
        assert changed[name].value == pytest.approx(original[name].value, rel=2e-6, abs=2e-6), name


def test_origin_dependent_markers_are_invariant_to_acquisition_dc_offset() -> None:
    record = _record()
    delineation, ecg, vcg, fs = record
    shifted = vcg + np.asarray([7.5, -4.25, 2.0])
    names = [
        "peak_svg_x_mv",
        "peak_svg_magnitude_mv",
        "peak_area_svg_angle_deg",
        "t_vti_3d_mv_ms",
        "tt_prime_mean_deg",
        "t_direction_absolute_solid_angle_sr",
        "t_octant_occupancy_entropy",
    ]
    original = _compute(names, record)
    changed = _compute(names, (delineation, ecg, shifted, fs))
    for name in names:
        assert original[name].value is not None, name
        assert changed[name].value == pytest.approx(
            original[name].value,
            rel=2e-9,
            abs=2e-9,
        ), name


def test_expensive_primitives_are_memoized_on_record_context() -> None:
    delineation, ecg, vcg, fs = _record()
    reset_feature_context()
    ctx = _context(delineation, ecg, vcg, fs)
    assert _segment(ctx, "qrs") is _segment(ctx, "qrs")
    assert _lead8_spectrum(ctx, "t") is _lead8_spectrum(ctx, "t")
    cache = getattr(ctx, "_discovery_v2_cache")
    assert ("segment", "qrs") in cache
    assert ("lead8_spectrum", "t") in cache


@pytest.mark.parametrize("self_pair", [False, True])
def test_vectorized_gauss_quadrature_matches_scalar_definition(
    self_pair: bool,
) -> None:
    rng = np.random.default_rng(20260903 + int(self_pair))
    first = rng.normal(size=(73, 3)).cumsum(axis=0)
    second = rng.normal(size=(61, 3)).cumsum(axis=0)
    a = v2._closed_resampled(first)
    b = a if self_pair else v2._closed_resampled(second)
    assert a is not None and b is not None
    da, db = np.diff(a, axis=0), np.diff(b, axis=0)
    ma = 0.5 * (a[:-1] + a[1:])
    mb = 0.5 * (b[:-1] + b[1:])
    terms = []
    for first_index in range(len(da)):
        start = first_index + 2 if self_pair else 0
        for second_index in range(start, len(db)):
            if (
                self_pair
                and first_index == 0
                and second_index == len(db) - 1
            ):
                continue
            delta = ma[first_index] - mb[second_index]
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-6:
                continue
            terms.append(
                float(
                    np.dot(
                        delta,
                        np.cross(da[first_index], db[second_index]),
                    )
                    / norm**3
                )
            )
    scale = (2.0 if self_pair else 1.0) / (4.0 * np.pi)
    measured = v2._gauss_pair_integrals(
        first,
        second,
        self_pair=self_pair,
    )
    assert measured is not None
    assert measured["signed"] == pytest.approx(
        scale * sum(terms),
        rel=1e-13,
        abs=1e-13,
    )
    assert measured["absolute"] == pytest.approx(
        scale * sum(abs(value) for value in terms),
        rel=1e-13,
        abs=1e-13,
    )


def test_qrs_and_qrst_vti_are_not_re_registered_as_exact_aliases() -> None:
    names = {spec.name for spec in DISCOVERY_V2_SPECS}
    assert "qrs_vti_3d_mv_ms" not in names
    assert "qrst_vti_3d_mv_ms" not in names
    assert {"p_vti_3d_mv_ms", "t_vti_3d_mv_ms", "st_vti_3d_mv_ms"} <= names
    # Existing magnitude_area_qrs and magnitude_area_qrst already implement
    # the same root-sum-square integral in mV*s.
    assert {"magnitude_area_qrs", "magnitude_area_qrst"} <= set(BiomarkerRegistry.available())
