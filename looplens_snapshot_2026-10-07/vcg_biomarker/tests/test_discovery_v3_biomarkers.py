"""Scientific, dimensional, numerical, and policy guards for Discovery-v3."""

from __future__ import annotations

import numpy as np
import pytest
from collections import Counter

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers.catalog_biomarkers import _context, reset_feature_context
from vcg_biomarker.core.biomarkers.discovery_v3_biomarkers import (
    DISCOVERY_V3_SPECS,
    DISCOVERY_V3_VERSION,
    _disease_geometry,
    _discrete_frechet,
    _qrs_t_distances,
    _radial_h0_persistence,
    register_discovery_v3_biomarkers,
)
from vcg_biomarker.core.data_loader import LEAD_MAP
from vcg_biomarker.core.vcg_geometry import vector_angle_deg
from vcg_biomarker.mimic.feature_roles import feature_profile
from vcg_biomarker.mimic.panels import default_feature_panels
from vcg_biomarker.tests.test_discovery_v2_biomarkers import _record


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


def test_v3_has_exactly_180_unique_scalar_provenanced_specs() -> None:
    names = [spec.name for spec in DISCOVERY_V3_SPECS]
    assert len(names) == 180
    assert len(names) == len(set(names))
    assert all(name == name.casefold() for name in names)
    assert all(spec.unit not in {"mixed", "vector", "array", "categorical"} for spec in DISCOVERY_V3_SPECS)
    assert all(spec.formula.strip() and spec.description.strip() for spec in DISCOVERY_V3_SPECS)
    assert all(spec.evidence_level in {"B", "C", "D"} for spec in DISCOVERY_V3_SPECS)
    assert all(spec.evidence_note.strip() for spec in DISCOVERY_V3_SPECS)
    assert all(spec.references and all(ref.startswith("https://") for ref in spec.references) for spec in DISCOVERY_V3_SPECS)
    assert Counter(spec.visual_assessment for spec in DISCOVERY_V3_SPECS) == {
        "direct": 126,
        "visual_correlate": 46,
        "mathematical_only": 8,
    }
    assert sum(spec.visually_interpretable for spec in DISCOVERY_V3_SPECS) == 126


def test_known_exact_aliases_were_not_reintroduced() -> None:
    names = {spec.name for spec in DISCOVERY_V3_SPECS}
    forbidden = {
        "qrs_t_dtw_distance",
        "qrs_t_normalized_dtw_distance",
        "qrs_t_directional_wasserstein",
        "qrs_t_wasserstein_radial_proxy",
        "qrs_initial_20ms_vector",
        "qrs_terminal_20ms_vector",
        "t_loop_dispersion",
        "t_pca_ratio_lambda2_lambda1",
        "persistent_homology_loop_lifetime",
    }
    assert not names.intersection(forbidden)
    assert not any("teigen" in name for name in names)
    assert not any("lobe_balance" in name for name in names)
    assert "qrs_initial_20ms_force_magnitude_mv" not in names
    assert "qrs_terminal_20ms_force_magnitude_mv" not in names


def test_registration_is_idempotent_and_v3_is_discovery_only() -> None:
    names = {spec.name for spec in DISCOVERY_V3_SPECS}
    register_discovery_v3_biomarkers()
    assert names <= set(BiomarkerRegistry.available())
    panels = {panel.panel_id: set(panel.feature_ids) for panel in default_feature_panels()}
    assert names.isdisjoint(panels["primary_core"])
    assert names.isdisjoint(panels["sentinel"])
    assert names <= panels["discovery_reserve"]
    for name in names:
        profile = feature_profile(name)
        assert profile.post_outcome_hypothesis
        assert profile.discovery_eligible
        assert not profile.core_sentinel_eligible
        assert not profile.combination_eligible
        definition = BiomarkerRegistry.definition(name)
        assert DISCOVERY_V3_VERSION in definition.notes
        assert "Discovery Reserve only" in definition.notes


def test_all_180_calculators_are_failure_isolated_on_nontrivial_record() -> None:
    results = _compute(spec.name for spec in DISCOVERY_V3_SPECS)
    errors = {
        name: result.computation_notes
        for name, result in results.items()
        if "error:" in str(result.computation_notes)
    }
    assert len(results) == 180
    assert not errors
    required = {
        "qrs_frontal_xy_signed_rotation_turns",
        "qrs_frontal_xy_grid_occupancy_fraction",
        "qrs_temporal_color_gradient_r2",
        "qrs_t_discrete_frechet_mv",
        "qrs_initial_10ms_force_x_mv",
        "qrs_late_burst_index",
        "qrs_terminal40_xyz_octant_000_fraction",
        "t_early_trajectory_magnitude_mv",
        "arvc_rtrms_qrs_mv",
        "tavplan_mean_uv",
        "qrs_radial_h0_max_persistence_fraction",
    }
    assert all(results[name].value is not None for name in required)


def test_origin_dependent_v3_metrics_are_invariant_to_acquisition_dc_offsets() -> None:
    record = _record()
    names = (
        "qrs_frontal_xy_signed_rotation_turns",
        "qrs_horizontal_xz_grid_occupancy_fraction",
        "qrs_t_discrete_frechet_mv",
        "qrs_initial_10ms_force_x_mv",
        "tavplan_mean_uv",
        "arvc_rtrms_qrs_mv",
    )
    baseline = _compute(names, record)
    delineation, ecg, vcg, fs = record
    shifted_vcg = vcg + np.asarray([1.7, -0.8, 2.2])
    shifted_ecg = ecg + np.linspace(-0.4, 0.5, ecg.shape[1])[None, :]
    shifted = _compute(names, (delineation, shifted_ecg, shifted_vcg, fs))
    for name in names:
        assert shifted[name].value == pytest.approx(baseline[name].value, rel=1e-8, abs=1e-8), name


def test_plane_inscription_distinguishes_clockwise_from_counterclockwise() -> None:
    record = _record()
    delineation, ecg, vcg, fs = record
    counterclockwise = vcg.copy()
    clockwise = vcg.copy()
    for beat in delineation.beats:
        start, end = beat.r_onset.sample, beat.r_offset.sample
        phase = np.linspace(0.0, 2.0 * np.pi, end - start + 1)
        curve = np.column_stack((np.cos(phase), np.sin(phase), 0.1 * np.sin(2.0 * phase)))
        counterclockwise[start : end + 1] = curve
        clockwise[start : end + 1] = curve[::-1]
    names = ("qrs_frontal_xy_signed_rotation_turns", "qrs_frontal_xy_clockwise_path_fraction")
    ccw = _compute(names, (delineation, ecg, counterclockwise, fs))
    cw = _compute(names, (delineation, ecg, clockwise, fs))
    assert ccw[names[0]].value > 0.8
    assert cw[names[0]].value < -0.8
    assert ccw[names[1]].value < 0.1
    assert cw[names[1]].value > 0.9


def test_discrete_frechet_uses_order_not_only_point_cloud_proximity() -> None:
    first = np.asarray([[0.0], [1.0], [2.0]])
    same = distance = np.abs(first - first.T)
    reversed_cost = np.abs(first - first[::-1].T)
    assert _discrete_frechet(same) == pytest.approx(0.0)
    assert _discrete_frechet(reversed_cost) == pytest.approx(2.0)


def test_qrs_t_expensive_distance_primitive_is_record_cached() -> None:
    delineation, ecg, vcg, fs = _record()
    reset_feature_context()
    ctx = _context(delineation, ecg, vcg, fs)
    first = _qrs_t_distances(ctx)
    second = _qrs_t_distances(ctx)
    assert first is second
    assert first is not None
    assert first["sliced_wasserstein_mean_mv"] >= 0.0


def test_radial_lower_star_is_genuine_union_find_persistence() -> None:
    flat = _radial_h0_persistence(np.ones(8))
    two_valleys = _radial_h0_persistence(np.asarray([0.0, 1.0, 0.0, 1.0]))
    assert flat.size == 0
    assert np.max(two_valleys) == pytest.approx(1.0)


def test_st_j_offset_components_are_sample_exact_and_not_recomputed_magnitudes() -> None:
    record = _record()
    delineation, ecg, vcg, fs = record
    modified = vcg.copy()
    beat = delineation.representative_beat
    for offset_ms in (20, 40, 60, 80):
        index = beat.j_point.sample + int(round(offset_ms * fs / 1000.0))
        modified[index] = np.asarray([offset_ms / 100.0, -offset_ms / 200.0, offset_ms / 400.0])
    names = [f"st_j_plus_{offset}ms_{axis}_mv" for offset in (20, 40, 60, 80) for axis in ("x", "y", "z")]
    results = _compute(names, (delineation, ecg, modified, fs))
    for offset_ms in (20, 40, 60, 80):
        expected = (offset_ms / 100.0, -offset_ms / 200.0, offset_ms / 400.0)
        for axis, value in zip(("x", "y", "z"), expected):
            assert results[f"st_j_plus_{offset_ms}ms_{axis}_mv"].value == pytest.approx(value)


def test_arvc_rpd_construction_matches_published_lead_equation() -> None:
    record = _record()
    delineation, ecg, vcg, fs = record
    modified = np.zeros_like(ecg)
    beat = delineation.representative_beat
    qrs_slice = slice(beat.r_onset.sample, beat.r_offset.sample + 1)
    t_slice = slice(beat.t_onset.sample, beat.t_offset.sample + 1)
    modified[qrs_slice, LEAD_MAP["V5"]] = -2.0
    modified[qrs_slice, LEAD_MAP["II"]] = -3.0
    modified[qrs_slice, LEAD_MAP["V1"]] = 4.0
    modified[t_slice, LEAD_MAP["V5"]] = 1.0
    modified[t_slice, LEAD_MAP["II"]] = -2.0
    modified[t_slice, LEAD_MAP["V1"]] = 3.0
    results = _compute(("arvc_rtrms_qrs_mv", "arvc_rpd_qrs_t_angle_deg"), (delineation, modified, vcg, fs))
    q_vector = np.asarray([-2.0, -3.0, -2.0])
    t_vector = np.asarray([1.0, -2.0, -1.5])
    assert results["arvc_rtrms_qrs_mv"].value == pytest.approx(np.linalg.norm(q_vector))
    assert results["arvc_rpd_qrs_t_angle_deg"].value == pytest.approx(vector_angle_deg(q_vector, t_vector))


def test_rvh_literature_ratios_use_peak_directional_forces() -> None:
    delineation, ecg, vcg, fs = _record()
    modified = vcg.copy()
    beat = delineation.representative_beat
    q_start, q_end = beat.r_onset.sample, beat.r_offset.sample
    q_count = q_end - q_start + 1
    phase = np.linspace(0.0, 2.0 * np.pi, q_count)
    modified[q_start : q_end + 1] = np.column_stack(
        (3.0 * np.sin(phase), np.sin(2.0 * phase), 2.0 * np.sin(phase + 0.4))
    )
    reset_feature_context()
    ctx = _context(delineation, ecg, modified, fs)
    metrics = _disease_geometry(ctx)
    assert metrics is not None
    qrs = ctx.vcg_segment_origin_corrected("qrs")
    assert qrs is not None
    expected_pa = np.max(np.maximum(qrs[:, 2], 0.0)) / np.max(np.maximum(-qrs[:, 2], 0.0))
    expected_rl = np.max(np.maximum(-qrs[:, 0], 0.0)) / np.max(np.maximum(qrs[:, 0], 0.0))
    assert metrics["rvh_qrs_posterior_anterior_force_ratio"] == pytest.approx(expected_pa)
    assert metrics["rvh_qrs_right_left_force_ratio"] == pytest.approx(expected_rl)


def test_tavplan_is_zero_for_a_plane_and_increases_with_out_of_plane_bulge() -> None:
    record = _record()
    delineation, ecg, vcg, fs = record
    planar = vcg.copy()
    bulged = vcg.copy()
    for beat in delineation.beats:
        start, end = beat.t_onset.sample, beat.t_offset.sample
        phase = np.linspace(0.0, np.pi, end - start + 1)
        plane_curve = np.column_stack((0.4 * np.sin(phase), 0.2 * np.sin(2 * phase), np.zeros_like(phase)))
        planar[start : end + 1] = plane_curve
        bulged[start : end + 1] = plane_curve
        bulged[start : end + 1, 2] = 0.08 * np.sin(3 * phase)
    flat = _compute(("tavplan_mean_uv",), (delineation, ecg, planar, fs))["tavplan_mean_uv"].value
    curved = _compute(("tavplan_mean_uv",), (delineation, ecg, bulged, fs))["tavplan_mean_uv"].value
    assert flat == pytest.approx(0.0, abs=1e-9)
    assert curved > 1.0
