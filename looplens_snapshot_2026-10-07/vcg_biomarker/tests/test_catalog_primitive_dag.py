from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from vcg_biomarker.core.biomarkers import catalog_biomarkers as catalog
from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
)


def test_late_t_areal_sweep_is_a_true_fraction_for_self_crossing_loops():
    # A signed shoelace denominator can cancel almost to zero while an open
    # late sub-path still encloses area.  A017 consequently observed 125 for a
    # quantity declared as a fraction.
    first = np.linspace(0.0, 2.0 * np.pi, 90)
    second = np.linspace(2.0 * np.pi, 0.0, 90)
    points = np.column_stack(
        [
            np.r_[np.cos(first), 0.8 * np.cos(second)],
            np.r_[np.sin(first), 0.8 * np.sin(second)],
            np.r_[0.1 * np.sin(2 * first), 0.1 * np.sin(2 * second)],
        ]
    )
    ctx = SimpleNamespace(
        vcg_xyz=points,
        vcg_segment=lambda segment: points if segment == "t" else None,
        bounds=lambda segment: (0, len(points) - 1) if segment == "t" else None,
    )
    result = catalog._t_loop_extra(ctx, "t_loop_late_area_fraction")
    assert result is not None and result.value is not None
    assert 0.0 <= result.value <= 1.0


def _context() -> catalog.FeatureContext:
    catalog.reset_feature_context()
    fs = 500
    n = 300
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    vcg = np.column_stack(
        [
            0.9 * np.cos(theta) + 0.08 * np.cos(5.0 * theta),
            0.5 * np.sin(theta) + 0.05 * np.sin(3.0 * theta),
            0.15 * np.sin(2.0 * theta),
        ]
    )
    ecg = np.column_stack(
        [vcg[:, index % 3] + 0.01 * index for index in range(12)]
    )
    beat = BeatDelineation(
        p_onset=MarkerPoint(sample=5, confidence=1.0, source="test"),
        p_offset=MarkerPoint(sample=25, confidence=1.0, source="test"),
        r_onset=MarkerPoint(sample=70, confidence=1.0, source="test"),
        r_peak=MarkerPoint(sample=110, confidence=1.0, source="test"),
        r_offset=MarkerPoint(sample=145, confidence=1.0, source="test"),
        t_onset=MarkerPoint(sample=170, confidence=1.0, source="test"),
        t_peak=MarkerPoint(sample=205, confidence=1.0, source="test"),
        t_offset=MarkerPoint(sample=295, confidence=1.0, source="test"),
    )
    delineation = RecordDelineation(
        beats=[beat],
        fs=fs,
        n_samples=n,
        ecg_id=1,
    )
    return catalog.FeatureContext(delineation, ecg, vcg, fs)


def test_derivative_and_fourier_primitives_are_evaluated_once(monkeypatch):
    ctx = _context()
    smooth_calls = 0
    resample_calls = 0
    original_smooth = catalog._smooth_derivative_input
    original_resample = catalog._resample_curve

    def counted_smooth(*args, **kwargs):
        nonlocal smooth_calls
        smooth_calls += 1
        return original_smooth(*args, **kwargs)

    def counted_resample(*args, **kwargs):
        nonlocal resample_calls
        resample_calls += 1
        return original_resample(*args, **kwargs)

    monkeypatch.setattr(catalog, "_smooth_derivative_input", counted_smooth)
    monkeypatch.setattr(catalog, "_resample_curve", counted_resample)

    for name in (
        "loop_bending_energy",
        "loop_curvature_entropy",
        "loop_total_curvature",
        "loop_total_torsion",
        "loop_curvature_speed_correlation",
    ):
        catalog._math_loop_feature_for_segment(ctx, name, "qrs")
    for name in (
        "loop_fourier_harmonic1_energy",
        "loop_fourier_harmonic2_energy",
        "loop_fourier_complexity_ratio",
        "loop_fourier_reconstruction_error_k3",
    ):
        catalog._math_loop_feature_for_segment(ctx, name, "qrs")

    assert smooth_calls == 1
    assert resample_calls == 1


def test_graph_and_sliding_plane_bundles_do_not_repeat_quadratic_work(monkeypatch):
    ctx = _context()
    nonlocal_calls = 0
    pca_calls = 0
    original_nonlocal = catalog._nonlocal_distances
    original_pca = catalog._pca

    def counted_nonlocal(*args, **kwargs):
        nonlocal nonlocal_calls
        nonlocal_calls += 1
        return original_nonlocal(*args, **kwargs)

    def counted_pca(*args, **kwargs):
        nonlocal pca_calls
        pca_calls += 1
        return original_pca(*args, **kwargs)

    monkeypatch.setattr(catalog, "_nonlocal_distances", counted_nonlocal)
    monkeypatch.setattr(catalog, "_pca", counted_pca)

    for name in (
        "loop_graph_bottleneck_distance",
        "loop_graph_branchiness",
        "loop_graph_cycle_count_proxy",
        "loop_knn_graph_length",
        "loop_local_density_variance",
        "loop_revisit_index",
    ):
        catalog._math_loop_feature_for_segment(ctx, name, "qrs")
    assert nonlocal_calls == 1

    catalog._math_loop_feature_for_segment(
        ctx, "loop_sliding_plane_angle_mean", "qrs"
    )
    calls_after_first_sliding_marker = pca_calls
    for name in (
        "loop_sliding_plane_angle_std",
        "loop_plane_twist_integral",
        "loop_plane_outlier_fraction",
        "loop_best_plane_residual_p95",
    ):
        catalog._math_loop_feature_for_segment(ctx, name, "qrs")
    assert pca_calls == calls_after_first_sliding_marker


def test_eight_lead_svd_is_shared_across_scalar_markers(monkeypatch):
    ctx = _context()
    calls = 0
    original_svd = np.linalg.svd

    def counted_svd(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted_svd)
    for name in (
        "<segment>_lead8_effective_rank",
        "<segment>_lead8_lambda1..lambda8",
        "<segment>_lead8_nondipolar_fraction",
        "residual_rms_<lead>",
    ):
        result = catalog._pca_svd_feature(ctx, name)
        assert result is not None

    assert calls == 1


def test_segment_gate_and_dependency_resolution_are_record_scoped(
    monkeypatch,
):
    ctx = _context()
    confidence_calls = 0
    evaluability_calls = 0
    original_confidence = ctx.delineation.segment_confidence
    original_evaluable = ctx.delineation.is_segment_evaluable

    def counted_confidence(*args, **kwargs):
        nonlocal confidence_calls
        confidence_calls += 1
        return original_confidence(*args, **kwargs)

    def counted_evaluable(*args, **kwargs):
        nonlocal evaluability_calls
        evaluability_calls += 1
        return original_evaluable(*args, **kwargs)

    monkeypatch.setattr(
        ctx.delineation,
        "segment_confidence",
        counted_confidence,
    )
    monkeypatch.setattr(
        ctx.delineation,
        "is_segment_evaluable",
        counted_evaluable,
    )

    for _ in range(50):
        assert ctx.segment_confidence("qrs") > 0.0
        assert ctx.is_segment_evaluable("qrs")
        assert ctx.vcg_segment("qrs") is not None

    assert confidence_calls == 1
    # FeatureContext applies the same gate from its cached confidence instead
    # of re-entering RecordDelineation (whose mutation signature is costly).
    assert evaluability_calls == 0

    # A distinct explicit threshold is a distinct gate, but is likewise
    # evaluated only once in this computation.
    for _ in range(10):
        assert ctx.is_segment_evaluable("qrs", 0.35)
    assert evaluability_calls == 0
