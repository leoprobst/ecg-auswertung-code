from __future__ import annotations

import base64
import zlib

import numpy as np
import pytest

from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
)
from vcg_biomarker.core.delineation.benchmark_suite import (
    DELINEATION_RELEASE_POLICY_VERSION,
    DELINEATION_SUITE_VERSION,
    delineation_implementation_hash,
    evaluate_delineation_release_gate,
)
from vcg_biomarker.mimic.acceptance import _check_delineation_release_gate
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.storage import atomic_write_json
from vcg_biomarker.core.biomarkers import advanced_vcg_biomarkers as advanced
from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.biomarkers import catalog_biomarkers as catalog_features
from vcg_biomarker.core.biomarkers.catalog_biomarkers import _literature_feature
from vcg_biomarker.core.vcg_geometry import (
    GEH_FORMULA_VERSION,
    PHYSIONET_GEH_REFERENCE,
    compute_geh_area_metrics,
    estimate_physiological_vcg_origin,
    vcg_azimuth_deg,
    vcg_elevation_deg,
)


# Exact XYZ_M[221:405] (MATLAB q=222 through T-end=405) from PhysioNet's
# official GEH v1.0.0 example record 90757.mat, compressed only to keep the
# repository fixture small.  XYZ_M is published in microvolts.
_PHYSIONET_90757_QRST_UV_ZLIB_B64 = (
    "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"
)


def _official_90757_qrst_mv() -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(_PHYSIONET_90757_QRST_UV_ZLIB_B64))
    return np.frombuffer(raw, dtype="<f8").reshape(184, 3) / 1000.0


def _point(sample: int, confidence: float = 0.95) -> MarkerPoint:
    return MarkerPoint(
        sample=sample,
        confidence=confidence,
        source="scientific_contract_test",
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


def test_physionet_90757_official_geh_golden_values() -> None:
    """Golden contract against the official PhysioNet GEH MATLAB fixture."""

    metrics = compute_geh_area_metrics(
        _official_90757_qrst_mv(),
        qrs_onset=0,
        qrs_offset=41,
        t_offset=183,
        fs=500,
    )

    assert GEH_FORMULA_VERSION == "physionet-geh-v1.0.0"
    assert "physionet.org/content/geh/1.0.0" in PHYSIONET_GEH_REFERENCE
    assert metrics.qrs_area_vector_uv_ms == pytest.approx(
        [15296.460, 22736.145, 16549.055], abs=1e-6
    )
    assert metrics.jt_area_vector_uv_ms == pytest.approx(
        [17341.595, 8007.945, -2181.085], abs=1e-6
    )
    assert metrics.qrst_area_vector_uv_ms == pytest.approx(
        [32638.055, 30744.090, 14367.970], abs=1e-6
    )
    assert metrics.svg_magnitude_uv_ms == pytest.approx(
        47083.757985445736, abs=1e-6
    )
    assert metrics.sai_qrs_mv_ms == pytest.approx(66.536570, abs=1e-8)
    assert metrics.sai_jt_mv_ms == pytest.approx(44.364435, abs=1e-8)
    assert metrics.sai_qrst_mv_ms == pytest.approx(110.901005, abs=1e-8)
    assert vcg_azimuth_deg(metrics.qrst_area_vector_uv_ms) == pytest.approx(
        23.76012098785626, abs=1e-10
    )
    assert vcg_elevation_deg(metrics.qrst_area_vector_uv_ms) == pytest.approx(
        49.234404365831715, abs=1e-10
    )


def test_registered_geh_and_sai_biomarkers_match_physionet_golden() -> None:
    fs = 500
    qrst = _official_90757_qrst_mv()
    xyz = np.zeros((224, 3), dtype=float)
    xyz[40:224] = qrst
    beat = _beat(58, 5, 15, 40, 81, 100, 223)
    delineation = RecordDelineation([beat], fs, len(xyz), 90757)

    expected = {
        "svg_magnitude": 47083.757985445736,
        "sum_absolute_qrs_integral_mv_ms": 66.536570,
        "sum_absolute_jt_integral_mv_ms": 44.364435,
        "sum_absolute_qrst_integral_mv_ms": 110.901005,
    }
    for name, golden in expected.items():
        result = BiomarkerRegistry.get(name).compute(
            delineation,
            vcg_xyz=xyz,
            fs=fs,
        )
        assert result.value == pytest.approx(golden, abs=1e-6)
        assert "origin=pr_fallback" in result.computation_notes


def test_legacy_geh_markers_share_origin_and_area_computation(monkeypatch) -> None:
    fs = 500
    xyz = np.zeros((224, 3), dtype=float)
    xyz[40:224] = _official_90757_qrst_mv()
    delineation = RecordDelineation(
        [_beat(58, 5, 15, 40, 81, 100, 223)], fs, len(xyz), 90757
    )
    catalog_features.reset_feature_context()
    calls = {"origin": 0, "geh": 0}
    original_origin = catalog_features.estimate_physiological_vcg_origin
    original_geh = catalog_features.compute_geh_area_metrics

    def counted_origin(*args, **kwargs):
        calls["origin"] += 1
        return original_origin(*args, **kwargs)

    def counted_geh(*args, **kwargs):
        calls["geh"] += 1
        return original_geh(*args, **kwargs)

    monkeypatch.setattr(
        catalog_features, "estimate_physiological_vcg_origin", counted_origin
    )
    monkeypatch.setattr(catalog_features, "compute_geh_area_metrics", counted_geh)

    for name in (
        "svg_magnitude",
        "svg_azimuth",
        "svg_elevation",
        "spatial_qrs_t_angle_mean",
    ):
        assert BiomarkerRegistry.get(name).compute(
            delineation, vcg_xyz=xyz, fs=fs
        ).value is not None
    assert calls == {"origin": 1, "geh": 1}


def test_sai_is_samplewise_absolute_and_jt_includes_st() -> None:
    xyz = np.zeros((12, 3), dtype=float)
    xyz[2:6, 0] = [1.0, -1.0, 1.0, -1.0]
    xyz[6:9, 1] = 2.0  # ST contribution before the visual T samples.
    xyz[9:11, 2] = 3.0
    metrics = compute_geh_area_metrics(
        xyz,
        qrs_onset=2,
        qrs_offset=5,
        t_offset=10,
        fs=1000,
    )

    assert metrics.sai_qrs_mv_ms == pytest.approx(3.0)
    assert metrics.jt_area_vector_uv_ms[1] > 0.0
    assert metrics.sai_qrst_mv_ms > (
        np.sum(np.abs(metrics.qrst_area_vector_uv_ms)) / 1000.0
    )
    assert metrics.qrs_area_vector_uv_ms + metrics.jt_area_vector_uv_ms == (
        pytest.approx(metrics.qrst_area_vector_uv_ms)
    )


@pytest.mark.parametrize(
    ("vector", "azimuth", "elevation"),
    [
        ([1.0, 0.0, 0.0], 0.0, 90.0),
        ([0.0, 0.0, 1.0], 90.0, 90.0),
        ([0.0, 0.0, -1.0], -90.0, 90.0),
        ([0.0, 1.0, 0.0], None, 0.0),
        ([0.0, -1.0, 0.0], None, 180.0),
    ],
)
def test_vcg_axis_basis_vectors(vector, azimuth, elevation) -> None:
    actual_azimuth = vcg_azimuth_deg(np.asarray(vector))
    if azimuth is None:
        assert actual_azimuth is None
    else:
        assert actual_azimuth == pytest.approx(azimuth)
    assert vcg_elevation_deg(np.asarray(vector)) == pytest.approx(elevation)


def test_origin_prefers_tp_and_reports_tp_pr_sensitivity() -> None:
    fs = 1000
    beats = [
        _beat(200, 100, 140, 180, 225, 270, 340),
        _beat(600, 500, 540, 580, 625, 670, 740),
    ]
    delineation = RecordDelineation(beats, fs, 900, 1)
    tp = np.asarray([0.10, -0.20, 0.05])
    pr = np.asarray([0.40, 0.10, -0.10])
    xyz = np.tile(tp, (900, 1))
    xyz[144:181] = pr
    xyz[544:581] = pr

    assessment = estimate_physiological_vcg_origin(xyz, delineation, fs)

    assert assessment.evaluable
    assert assessment.primary.method.startswith("tp_")
    assert assessment.primary.vector_millivolts == pytest.approx(tp)
    assert assessment.tp is not None
    assert assessment.pr is not None
    expected_distance = np.linalg.norm(tp - pr) * 1000.0
    assert assessment.numeric_metadata()["origin_tp_pr_distance_uv"] == (
        pytest.approx(expected_distance)
    )
    assert "tp_pr_sensitivity=available" in assessment.provenance_note()


def test_origin_qualified_pr_fallback_and_fail_closed_abstention() -> None:
    fs = 1000
    beat = _beat(220, 100, 140, 190, 245, 290, 370)
    delineation = RecordDelineation([beat], fs, 500, 1)
    pr = np.asarray([0.22, -0.08, 0.04])
    xyz = np.tile(pr, (500, 1))

    fallback = estimate_physiological_vcg_origin(xyz, delineation, fs)
    assert fallback.evaluable
    assert fallback.primary.method == "pr_fallback"
    assert fallback.primary.vector_millivolts == pytest.approx(pr)

    no_p = BeatDelineation(
        r_peak=_point(220),
        r_onset=_point(190),
        r_offset=_point(245),
        t_onset=_point(290),
        t_offset=_point(370),
    )
    abstained = estimate_physiological_vcg_origin(
        xyz, RecordDelineation([no_p], fs, 500, 2), fs
    )
    assert not abstained.evaluable
    assert abstained.primary is None
    assert "not_evaluable" in abstained.provenance_note()


def _passing_release_results() -> dict[str, dict]:
    implementation_hash = delineation_implementation_hash()

    def ludb_boundaries() -> dict:
        return {
            location: {"mean_absolute": 1.0, "p95_absolute": 2.0}
            for location in ("onset", "peak", "offset")
        }

    def qtdb_boundaries() -> dict:
        return {
            location: {"mae_ms": 1.0, "p95_abs_ms": 2.0}
            for location in ("onset", "peak", "offset")
        }

    common = {
        "status": "complete",
        "suite_version": DELINEATION_SUITE_VERSION,
        "implementation_hash": implementation_hash,
    }
    ludb_waves = {
        wave: {
            "sensitivity": 0.99,
            "positive_predictive_value": 0.99,
            "boundary_errors_ms": ludb_boundaries(),
        }
        for wave in ("p", "qrs", "t")
    }
    qtdb_waves = {
        wave: {
            "sensitivity": 0.99,
            "boundary_errors_ms": qtdb_boundaries(),
        }
        for wave in ("p", "qrs", "t")
    }
    return {
        "ludb": {
            **common,
            "dataset_spec_version": "1.0.1",
            "records_requested": 200,
            "records_completed": 200,
            "records_quarantined": 0,
            "groups": {
                "all": {
                    "record_evaluable_rate": {"p": 0.99, "qrs": 0.99, "t": 0.99},
                    "evaluable_waves": ludb_waves,
                }
            },
        },
        "qtdb": {
            **common,
            "dataset_spec_version": "1.0.0",
            "records_requested": 105,
            "records_completed": 105,
            "records_quarantined": 0,
            "waves": qtdb_waves,
        },
        "nstdb": {
            **common,
            "dataset_spec_version": "1.0.0",
            "records_requested": 12,
            "records_quarantined": 0,
            "windows_completed": 24,
            "summary": {
                "clean_reference": {
                    "qrs_sensitivity_median": 0.99,
                    "qrs_ppv_median": 0.99,
                },
                "noise_exposure": {
                    "qrs_sensitivity_median": 0.90,
                    "qrs_ppv_median": 0.90,
                },
            },
        },
    }


def test_release_gate_is_performance_version_and_full_run_bound() -> None:
    passing = _passing_release_results()
    assert evaluate_delineation_release_gate(passing)["passed"] is True

    capped = evaluate_delineation_release_gate(passing, max_records=10)
    assert capped["passed"] is False
    assert "global:uncapped_record_run" in capped["failures"]

    wrong_version = _passing_release_results()
    wrong_version["qtdb"]["dataset_spec_version"] = "0.0.0"
    failed = evaluate_delineation_release_gate(wrong_version)
    assert failed["passed"] is False
    assert "qtdb:dataset_spec_version" in failed["failures"]

    failed_status = _passing_release_results()
    failed_status["nstdb"]["status"] = "failed_no_windows"
    failed = evaluate_delineation_release_gate(failed_status)
    assert "nstdb:status" in failed["failures"]


def test_acceptance_independently_rechecks_delineation_gate(tmp_path) -> None:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "structured.csv",
        echo_study_list_path=tmp_path / "study.csv",
        echo_record_list_path=tmp_path / "record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    results = _passing_release_results()
    release_gate = evaluate_delineation_release_gate(results)
    path = config.resources_root / "delineation" / "benchmark_suite.json"
    atomic_write_json(
        path,
        {
            "suite_version": DELINEATION_SUITE_VERSION,
            "release_policy_version": DELINEATION_RELEASE_POLICY_VERSION,
            "implementation_hash": delineation_implementation_hash(),
            "required_datasets_missing": [],
            "run_configuration": {
                "max_records": None,
                "max_seconds_per_record": 120,
                "record_capped": False,
            },
            "results": results,
            "release_gate": release_gate,
        },
    )

    observed, expected, evidence = _check_delineation_release_gate(config)
    assert observed == expected
    assert evidence == str(path.resolve())

    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    payload["run_configuration"]["max_records"] = 10
    payload["run_configuration"]["record_capped"] = True
    atomic_write_json(path, payload)
    with pytest.raises(RuntimeError, match="not current and passing"):
        _check_delineation_release_gate(config)


def test_beatwise_svg_pairs_qrs_and_jt_by_beat_identity(monkeypatch) -> None:
    identities = ((0, 100), (1, 200), (2, 300), (3, 400))
    qrs = {
        identities[0]: np.asarray([1.0, 0.0, 0.0]),
        identities[1]: np.asarray([0.0, 0.0, 2.0]),  # no matching JT
        identities[2]: np.asarray([0.0, 1.0, 0.0]),
        identities[3]: np.asarray([0.0, 0.0, 1.0]),
    }
    jt = {
        identities[0]: np.asarray([1.0, 0.0, 0.0]),
        identities[2]: np.asarray([2.0, -1.0, 0.0]),
        identities[3]: np.asarray([2.0, 0.0, -1.0]),
    }

    monkeypatch.setattr(
        advanced,
        "_beat_area_vectors_by_identity",
        lambda _ctx, segment: qrs if segment == "qrs" else jt,
    )
    result = advanced._beatwise_svg_direction_dispersion(object())

    assert result.value == pytest.approx(0.0)
    assert result.per_lead == {"paired_beats": 3.0}
    assert "identity-matched" in result.notes


def test_hf_qrs_filters_xyz_axes_before_spatial_energy() -> None:
    fs = 1000
    samples = np.arange(300, dtype=float)
    phase = 2.0 * np.pi * 120.0 * samples / fs
    # Its vector magnitude is exactly constant. Filtering that magnitude
    # would erase the 120-Hz rotating vector even though both physical axes
    # contain strong high-frequency energy.
    rotating_xyz = np.column_stack(
        (np.sin(phase), np.cos(phase), np.zeros_like(phase))
    )

    class WidebandContext:
        def __init__(self):
            self.fs = fs

        def vcg_segment_wideband(self, segment):
            assert segment == "qrs"
            return rotating_xyz

    result = _literature_feature(
        WidebandContext(), "qrs_fragmentation_high_frequency_energy"
    )

    assert result is not None
    assert result.value is not None
    assert result.value > 500_000.0
    assert result.per_lead["x_uv2"] > 200_000.0
    assert result.per_lead["y_uv2"] > 200_000.0
    assert "Axiswise XYZ 80-150 Hz" in result.notes

    unsupported = WidebandContext()
    unsupported.fs = 300
    abstained = _literature_feature(
        unsupported, "qrs_fragmentation_high_frequency_energy"
    )
    assert abstained.value is None
    assert "fixed 80-150 Hz" in abstained.notes
