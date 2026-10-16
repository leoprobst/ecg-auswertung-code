from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import pytest

from vcg_biomarker.core.cache import ResultCache
from vcg_biomarker.core.parallel_pipeline import _process_record_batch
from vcg_biomarker.mimic.statistics import MarkerModel
from vcg_biomarker.mimic.models import ValidationSpec
from vcg_biomarker.mimic.storage import DuckDBCatalog
from vcg_biomarker.mimic.validation import (
    _SharedValidationFeatureMatrix,
    _load_endpoint_frame,
    _load_endpoint_validation_checkpoint,
    _marker_verification_bias_records,
    _write_endpoint_validation_checkpoint,
)


def test_sensitivity_transform_reuses_the_primary_kors_result(monkeypatch) -> None:
    from vcg_biomarker.core.vcg_transform import VCGResult, VCGTransformRegistry
    from vcg_biomarker.mimic.signals import compute_vcg_sensitivity_transforms

    signal = np.column_stack(
        [
            np.sin(np.linspace(0.0, 4.0 * np.pi, 300) + lead * 0.1)
            for lead in range(12)
        ]
    )
    primary = VCGTransformRegistry.transform(signal, "kors")
    calls: list[str] = []
    original_transform = VCGTransformRegistry.transform

    def counted_transform(ecg, method):
        calls.append(str(method))
        return original_transform(ecg, method)

    monkeypatch.setattr(VCGTransformRegistry, "transform", counted_transform)

    result = compute_vcg_sensitivity_transforms(
        signal,
        methods=("inverse_dower", "pca_kors_aligned"),
        kors_result=VCGResult(
            vx=primary.vx.copy(),
            vy=primary.vy.copy(),
            vz=primary.vz.copy(),
            method_name=primary.method_name,
            channels_used=list(primary.channels_used),
            references=list(primary.references),
        ),
    )

    assert set(result) == {"inverse_dower", "pca_kors_aligned"}
    assert calls == ["inverse_dower"]

    incompatible = VCGResult(
        vx=primary.vx[:-1],
        vy=primary.vy[:-1],
        vz=primary.vz[:-1],
        method_name="kors",
        channels_used=list(primary.channels_used),
        references=list(primary.references),
    )
    with pytest.raises(ValueError, match="Kors transform"):
        compute_vcg_sensitivity_transforms(
            signal,
            methods=("pca_kors_aligned",),
            kors_result=incompatible,
        )


def _write_feature_cache(root: Path) -> str:
    for split, rows in (
        (
            "train",
            [{"subject_id": 1, "study_id": 11, "marker": 2.0}],
        ),
        (
            "validation",
            [{"subject_id": 4, "study_id": 44, "marker": 3.0}],
        ),
    ):
        target = root / f"split={split}" / "features.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(target, index=False)
    return str(root / "split=*" / "features.parquet")


def test_shared_feature_matrix_decodes_block_once_and_keeps_missing_rows(
    tmp_path: Path,
) -> None:
    feature_glob = _write_feature_cache(tmp_path / "features")
    matrix = _SharedValidationFeatureMatrix(feature_glob, ("marker",))
    base = pd.DataFrame(
        {
            "split": ["train", "train", "validation"],
            "subject_id": [1, 2, 4],
            "study_id": [11, 22, 44],
        }
    )

    first = matrix.attach(base)
    second = matrix.attach(base.iloc[[0]])

    assert len(first) == 3
    assert first["marker"].tolist()[0] == 2.0
    assert np.isnan(first.loc[1, "marker"])
    assert second.loc[0, "marker"] == 2.0
    assert matrix.decoded_block_count == 1


def test_shared_feature_matrix_bounded_projection_does_not_fill_wide_frame(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    for split, subject_id, study_id in (
        ("train", 1, 11),
        ("validation", 2, 22),
    ):
        path = root / f"split={split}" / "features.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject_id": [subject_id],
                "study_id": [study_id],
                "marker_a": [1.0],
                "marker_b": [2.0],
                "marker_c": [3.0],
            }
        ).to_parquet(path, index=False)
    matrix = _SharedValidationFeatureMatrix(
        str(root / "split=*" / "features.parquet"),
        ("marker_a", "marker_b", "marker_c"),
    )
    base = pd.DataFrame(
        {
            "split": ["train", "validation"],
            "subject_id": [1, 2],
            "study_id": [11, 22],
        }
    )

    projected = matrix.attach_columns(base, ("marker_b",))

    assert list(projected.columns) == [
        "split",
        "subject_id",
        "study_id",
        "marker_b",
    ]
    assert projected["marker_b"].tolist() == [2.0, 2.0]
    assert matrix.decoded_block_count == 0
    assert matrix.maximum_projected_columns == 1
    assert matrix.projected_read_count == 1


def test_validation_marker_checkpoints_never_attach_the_full_matrix(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.mimic.validation import (
        _load_or_evaluate_marker_chunks,
    )

    feature_ids = tuple(f"marker_{index:03d}" for index in range(260))
    root = tmp_path / "features"
    for split, subjects, studies, labels in (
        ("train", [1, 2], [11, 22], [0.0, 1.0]),
        ("validation", [3, 4], [33, 44], [0.0, 1.0]),
    ):
        path = root / f"split={split}" / "features.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "subject_id": subjects,
            "study_id": studies,
            "signal_quality_score": [0.75, 0.85],
            **{
                feature_id: np.asarray(labels) + index * 1e-4
                for index, feature_id in enumerate(feature_ids)
            },
        }
        pd.DataFrame(payload).to_parquet(path, index=False)
    matrix = _SharedValidationFeatureMatrix(
        str(root / "split=*" / "features.parquet"),
        feature_ids,
    )
    universe = pd.DataFrame(
        {
            "split": ["train", "train", "validation", "validation"],
            "subject_id": [1, 2, 3, 4],
            "study_id": [11, 22, 33, 44],
            "eligibility_state": ["eligible"] * 4,
            "label_state": ["negative", "positive"] * 2,
        }
    )
    verified = universe.copy()
    verified["label"] = [0, 1, 0, 1]
    train = verified.loc[verified["split"].eq("train")].reset_index(drop=True)
    validation = verified.loc[
        verified["split"].eq("validation")
    ].reset_index(drop=True)
    specification = ValidationSpec(
        run_id="bounded-matrix",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={"primary_core": "1"},
        development_bootstrap_iterations=0,
        random_seed=17,
    )

    evaluations = _load_or_evaluate_marker_chunks(
        endpoint_id="endpoint",
        feature_ids=feature_ids,
        panel_map={feature_id: "primary_core" for feature_id in feature_ids},
        train=train,
        validation=validation,
        endpoint_universe=universe,
        shared_feature_matrix=matrix,
        specification=specification,
        bootstrap_iterations=0,
        worker_count=1,
        checkpoint_root=tmp_path / "checkpoints",
        input_hash="bounded-input",
        endpoint_index=1,
        endpoint_total=1,
        progress=None,
    )

    assert len(evaluations) == len(feature_ids)
    # One one-column common-quality read is reused by all three marker
    # checkpoints; no marker projection exceeds the checkpoint width.
    assert matrix.projected_read_count == 4
    assert matrix.maximum_projected_columns == 128
    assert matrix.decoded_block_count == 0


def test_blockwise_validation_fails_closed_without_quality_score(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.mimic.validation import (
        _load_or_evaluate_marker_chunks,
    )

    root = tmp_path / "features"
    for split, subjects, studies in (
        ("train", [1, 2], [11, 22]),
        ("validation", [3, 4], [33, 44]),
    ):
        path = root / f"split={split}" / "features.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject_id": subjects,
                "study_id": studies,
                "marker": [0.0, 1.0],
            }
        ).to_parquet(path, index=False)
    universe = pd.DataFrame(
        {
            "split": ["train", "train", "validation", "validation"],
            "subject_id": [1, 2, 3, 4],
            "study_id": [11, 22, 33, 44],
            "eligibility_state": ["eligible"] * 4,
            "label_state": ["negative", "positive"] * 2,
        }
    )
    verified = universe.assign(label=[0, 1, 0, 1])
    matrix = _SharedValidationFeatureMatrix(
        str(root / "split=*" / "features.parquet"),
        ("marker",),
    )
    specification = ValidationSpec(
        run_id="missing-quality",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={"primary_core": "1"},
        development_bootstrap_iterations=0,
        random_seed=23,
    )

    with pytest.raises(
        RuntimeError,
        match="lacks required common feature columns.*signal_quality_score",
    ):
        _load_or_evaluate_marker_chunks(
            endpoint_id="endpoint",
            feature_ids=("marker",),
            panel_map={"marker": "primary_core"},
            train=verified.iloc[:2].reset_index(drop=True),
            validation=verified.iloc[2:].reset_index(drop=True),
            endpoint_universe=universe,
            shared_feature_matrix=matrix,
            specification=specification,
            bootstrap_iterations=0,
            worker_count=1,
            checkpoint_root=tmp_path / "checkpoints",
            input_hash="missing-quality-input",
            endpoint_index=1,
            endpoint_total=1,
            progress=None,
        )


def test_blockwise_validation_keeps_transform_sensitivity_columns(
    tmp_path: Path,
) -> None:
    from vcg_biomarker.mimic.validation import (
        _load_or_evaluate_marker_chunks,
    )

    feature_id = "spatial_qrs_t_angle_deg"
    sensitivity_id = f"sensitivity_inverse_dower__{feature_id}"
    root = tmp_path / "features"
    universe_parts = []
    for split_index, split in enumerate(("train", "validation")):
        labels = np.tile(np.asarray([0, 1], dtype=np.int8), 10)
        subjects = np.arange(20) + split_index * 100
        studies = subjects + 1_000
        path = root / f"split={split}" / "features.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject_id": subjects,
                "study_id": studies,
                "signal_quality_score": np.linspace(0.4, 0.9, 20),
                feature_id: labels + np.linspace(0.0, 0.01, 20),
                sensitivity_id: labels + np.linspace(0.01, 0.0, 20),
            }
        ).to_parquet(path, index=False)
        universe_parts.append(
            pd.DataFrame(
                {
                    "split": split,
                    "subject_id": subjects,
                    "study_id": studies,
                    "eligibility_state": "eligible",
                    "label_state": np.where(
                        labels == 1, "positive", "negative"
                    ),
                }
            )
        )
    universe = pd.concat(universe_parts, ignore_index=True)
    verified = universe.copy()
    verified["label"] = verified["label_state"].eq("positive").astype(int)
    train = verified.loc[verified["split"].eq("train")].reset_index(drop=True)
    validation = verified.loc[
        verified["split"].eq("validation")
    ].reset_index(drop=True)
    matrix = _SharedValidationFeatureMatrix(
        str(root / "split=*" / "features.parquet"),
        (feature_id,),
    )
    specification = ValidationSpec(
        run_id="transform-projection",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={"primary_core": "1"},
        development_bootstrap_iterations=0,
        random_seed=19,
    )

    evaluations = _load_or_evaluate_marker_chunks(
        endpoint_id="endpoint",
        feature_ids=(feature_id,),
        panel_map={feature_id: "primary_core"},
        train=train,
        validation=validation,
        endpoint_universe=universe,
        shared_feature_matrix=matrix,
        specification=specification,
        bootstrap_iterations=0,
        worker_count=1,
        checkpoint_root=tmp_path / "checkpoints",
        input_hash="transform-input",
        endpoint_index=1,
        endpoint_total=1,
        progress=None,
    )

    sensitivity = [
        row
        for row in evaluations[0].stability_records
        if row.get("analysis_type") == "vcg_transform_sensitivity"
    ]
    assert len(sensitivity) == 1
    assert sensitivity[0]["transform_method"] == "inverse_dower"
    assert evaluations[0].train_record is not None
    assert evaluations[0].validation_record is not None
    assert evaluations[0].train_record["mean_signal_quality"] == pytest.approx(
        0.65
    )
    assert evaluations[0].validation_record[
        "low_signal_quality_fraction"
    ] == pytest.approx(0.4)
    assert matrix.maximum_projected_columns == 2


def test_marker_quality_is_decoded_once_per_chunk_or_reused(
    monkeypatch,
) -> None:
    from vcg_biomarker.mimic import validation as validation_module

    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    train = pd.DataFrame(
        {
            "subject_id": np.arange(4),
            "label": labels,
            "signal_quality_score": [0.5, 0.6, 0.7, 0.8],
            "marker_a": labels.astype(float),
            "marker_b": labels.astype(float) + 0.1,
        }
    )
    validation = train.assign(subject_id=np.arange(10, 14))
    specification = ValidationSpec(
        run_id="quality-cache",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={"primary_core": "1"},
        development_bootstrap_iterations=0,
        random_seed=29,
    )
    original_numeric_feature = validation_module._numeric_feature
    quality_calls: list[int] = []

    def counted_numeric_feature(frame, feature_id):
        if feature_id == "signal_quality_score":
            quality_calls.append(len(frame))
        return original_numeric_feature(frame, feature_id)

    monkeypatch.setattr(
        validation_module,
        "_numeric_feature",
        counted_numeric_feature,
    )
    arguments = {
        "endpoint_id": "endpoint",
        "feature_ids": ("marker_a", "marker_b"),
        "panel_map": {
            "marker_a": "primary_core",
            "marker_b": "primary_core",
        },
        "train": train,
        "validation": validation,
        "specification": specification,
        "bootstrap_iterations": 0,
        "worker_count": 1,
    }

    validation_module._evaluate_endpoint_markers(**arguments)
    assert quality_calls == [4, 4]

    quality_calls.clear()
    validation_module._evaluate_endpoint_markers(
        **arguments,
        train_quality=train["signal_quality_score"].to_numpy(),
        validation_quality=validation["signal_quality_score"].to_numpy(),
    )
    assert quality_calls == []


def test_bounded_projection_preserves_categorical_quality_tier(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    for split, subject_id, study_id, tier in (
        ("train", 1, 11, "high"),
        ("validation", 2, 22, "low"),
    ):
        path = root / f"split={split}" / "features.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject_id": [subject_id],
                "study_id": [study_id],
                "marker": [1.0],
                "signal_quality_tier": [tier],
            }
        ).to_parquet(path, index=False)
    matrix = _SharedValidationFeatureMatrix(
        str(root / "split=*" / "features.parquet"),
        ("marker",),
    )
    base = pd.DataFrame(
        {
            "split": ["train", "train", "validation"],
            "subject_id": [1, 9, 2],
            "study_id": [11, 99, 22],
        }
    )

    projected = matrix.attach_columns(base, ("signal_quality_tier",))

    assert projected["signal_quality_tier"].tolist() == ["high", None, "low"]
    assert matrix.maximum_projected_columns == 1


def test_reference_blind_endpoint_left_join_counts_missing_feature(
    tmp_path: Path,
) -> None:
    feature_glob = _write_feature_cache(tmp_path / "features")
    matrix = _SharedValidationFeatureMatrix(feature_glob, ("marker",))
    cohort_path = tmp_path / "cohort.parquet"
    master_path = tmp_path / "master.parquet"
    cohort = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "study_id": [11, 22, 33],
            "ecg_time": pd.to_datetime(["2024-01-01"] * 3),
            "split": ["train"] * 3,
            "label_id": ["endpoint"] * 3,
            "label_state": ["positive", "negative", "unverified"],
            "eligibility_state": ["eligible", "eligible", "unverified"],
            "confidence": [1.0, 1.0, 0.0],
            "time_delta_hours": [0.0, 0.0, np.nan],
            "time_precision": ["exact", "exact", "unknown"],
            "evidence_ids_json": ["[]"] * 3,
            "evidence_summary": [""] * 3,
            "source_tables": [""] * 3,
        }
    )
    cohort.to_parquet(cohort_path, index=False)
    master = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "study_id": [11, 22, 33],
            "age": [60.0] * 3,
            "sex": ["M"] * 3,
            "race": ["UNKNOWN"] * 3,
            "waveform_path": [""] * 3,
            "machine_report": [""] * 3,
            "machine_cart_id": [""] * 3,
            "machine_bandwidth": [""] * 3,
            "machine_filtering": [""] * 3,
            "machine_metadata_temporal_status": ["unknown"] * 3,
            "machine_metadata_usable_for_primary": [False] * 3,
            "hadm_id": [np.nan] * 3,
            "ed_stay_id": [np.nan] * 3,
            "nearest_echo_measurement_id": [np.nan] * 3,
            "nearest_echo_delta_hours": [np.nan] * 3,
            "machine_qrs_duration_ms": [100.0] * 3,
            "machine_qt_interval_ms": [410.0] * 3,
            "machine_pr_interval_ms": [180.0] * 3,
            "machine_p_onset_ms": [10.0] * 3,
            "machine_p_end_ms": [90.0] * 3,
            "machine_qrs_onset_ms": [110.0] * 3,
            "machine_qrs_end_ms": [210.0] * 3,
            "machine_p_duration_ms": [80.0] * 3,
            "signal_quality_score": [1.0] * 3,
        }
    )
    master.to_parquet(master_path, index=False)

    frame = _load_endpoint_frame(
        DuckDBCatalog(tmp_path / "catalog.duckdb", threads=1),
        cohort_path=cohort_path,
        master_path=master_path,
        reports_path=None,
        context_path=None,
        feature_glob=feature_glob,
        endpoint_id="endpoint",
        report_label_id=None,
        splits=("train",),
        shared_feature_matrix=matrix,
    )

    assert len(frame) == 3
    assert frame["marker"].notna().sum() == 1
    assert frame["machine_qt_interval_ms"].eq(410.0).all()
    assert frame["machine_pr_interval_ms"].eq(180.0).all()
    assert frame["machine_p_duration_ms"].eq(80.0).all()
    model = MarkerModel(
        feature_id="marker",
        direction=1,
        mean=0.0,
        standard_deviation=1.0,
        thresholds={
            "specificity_95": 1.0,
            "specificity_90": 1.0,
            "sensitivity_90": 1.0,
        },
        n_train=1,
        n_positive_train=1,
        n_negative_train=0,
    )
    bounds = _marker_verification_bias_records(
        endpoint_id="endpoint",
        panel_id="primary_core",
        feature_id="marker",
        endpoint_universe=frame,
        model=model,
    )
    assert len(bounds) == 3
    assert bounds[0]["n_reference_blind_universe"] == 3
    assert bounds[0]["n_marker_missing"] == 2
    assert bounds[0]["verification_rate"] == 2 / 3


def test_completed_endpoint_checkpoint_is_manifest_not_duplicate(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "endpoint.json"
    marker_root = tmp_path / "endpoint" / "marker_chunks"
    marker_root.mkdir(parents=True)
    (marker_root / "markers-0000-0128.json").write_text(
        "intermediate",
        encoding="utf-8",
    )
    groups = {
        "marker_records": [{"feature_id": "marker", "auc": 0.75}],
        "failures": [],
    }
    _write_endpoint_validation_checkpoint(
        checkpoint,
        input_hash="input",
        endpoint_id="endpoint",
        result_groups=groups,
        group_offsets={"marker_records": 0, "failures": 0},
    )

    manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert "result_groups" not in manifest
    assert manifest["checkpoint_type"] == "completed_endpoint_manifest"
    group_entry = next(entry for entry in manifest["result_group_manifest"] if not entry.get("inline_empty"))
    assert group_entry["encoding"] == "gzip-json-v1"
    assert group_entry["relative_path"].endswith(".json.gz")
    assert not marker_root.exists()
    loaded = _load_endpoint_validation_checkpoint(
        checkpoint,
        input_hash="input",
        endpoint_id="endpoint",
    )
    assert loaded is not None
    assert loaded["result_groups"] == groups
    artifact = checkpoint.parent / group_entry["relative_path"]
    artifact.write_bytes(artifact.read_bytes() + b"synthetic corruption")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _load_endpoint_validation_checkpoint(checkpoint, input_hash="input", endpoint_id="endpoint")


def test_cache_consolidation_streams_without_pandas_concat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = ResultCache(tmp_path)
    cache.store_chunk([{"ecg_id": 1, "bm_x": 1.0}], 0)
    cache.store_chunk(
        [
            {"ecg_id": 1, "bm_x": 2.0},
            {"ecg_id": 2, "bm_x": 3.0},
        ],
        1,
    )
    assert not (tmp_path / "cache_index.parquet").exists()

    def fail_concat(*args, **kwargs):
        raise AssertionError("pandas concat fallback used")

    monkeypatch.setattr(pd, "concat", fail_concat)
    cache.consolidate()

    result = pd.read_parquet(tmp_path / "batch_results.parquet")
    assert result.sort_values("ecg_id")["bm_x"].tolist() == [2.0, 3.0]
    assert (tmp_path / "cache_index.parquet").exists()
    assert not (tmp_path / "chunks").exists()


def test_record_worker_returns_only_lean_scalar_ipc(monkeypatch) -> None:
    from vcg_biomarker.core import preprocessing
    from vcg_biomarker.core.biomarkers import base as biomarker_base
    from vcg_biomarker.core.delineation import fusion_delineator
    from vcg_biomarker.core.pathologies import registry as pathology_registry
    from vcg_biomarker.core import vcg_transform

    signal = np.zeros((12, 50), dtype=np.float32)
    monkeypatch.setattr(
        preprocessing,
        "preprocess_record",
        lambda raw, fs: {
            "cleaned": raw,
            "delineation": raw,
            "morphology": raw,
            "r_peaks": np.array([20]),
        },
    )
    monkeypatch.setattr(
        preprocessing,
        "construct_robust_morphology_record",
        lambda raw, delineation: (
            raw,
            {"status": "ok", "used_beat_count": 1},
        ),
    )
    monkeypatch.setattr(
        vcg_transform.VCGTransformRegistry,
        "transform",
        lambda raw, method: SimpleNamespace(
            xyz=np.zeros((3, 50), dtype=np.float32)
        ),
    )

    class FakeDelineator:
        def __init__(self, **kwargs):
            pass

        def delineate_record(self, **kwargs):
            return SimpleNamespace(quality_score=0.9)

    monkeypatch.setattr(
        fusion_delineator,
        "FusionDelineator",
        FakeDelineator,
    )
    monkeypatch.setattr(
        biomarker_base.BiomarkerRegistry,
        "compute_all",
        lambda **kwargs: [SimpleNamespace(name="marker", value=1.25)],
    )
    monkeypatch.setattr(
        pathology_registry.PathologyRegistry,
        "check_all",
        lambda codes: {"target": True},
    )

    result = _process_record_batch(
        [
            (
                7,
                signal,
                500,
                {},
                "kors",
                None,
                None,
                {},
                20.0,
            )
        ]
    )[0]

    assert result == {
        "ecg_id": 7,
        "error": None,
        "quality_score": 0.9,
        "morphology_template_status": "ok",
        "morphology_template_used_beat_count": 1,
        "biomarker_dict": {"marker": 1.25},
        "pathology_labels": {"target": True},
    }
