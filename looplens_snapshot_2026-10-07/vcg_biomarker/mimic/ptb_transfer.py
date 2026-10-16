from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .models import FrozenScoreSpec, TestLock
from .statistics import (
    MarkerModel,
    apply_frozen_score,
    enforce_inference_policy,
    evaluate_marker,
    fit_frozen_score,
    fit_marker_model,
    inference_status,
    marker_bootstrap_intervals,
)
from .storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_fingerprint,
    file_sha256,
)


PTB_REVALIDATION_VERSION = "1.3.0"
PTB_FREEZE_VERSION = "1.1.0"
PTB_TEST_VERSION = "1.0.0"
PTB_IMPLEMENTATION_HASH_VERSION = "1.0.0"
DEFAULT_PTB_TRANSFER_RUN_ID = "ptb-frozen-development-v3"


@dataclass(frozen=True)
class PTBEndpointSpec:
    endpoint_id: str
    display_name: str
    source_columns: tuple[str, ...]
    components: tuple[str, ...]
    mimic_endpoint_ids: tuple[str, ...]


@dataclass(frozen=True)
class PTBRevalidationOptions:
    run_id: str = DEFAULT_PTB_TRANSFER_RUN_ID
    bootstrap_iterations: int = 1000
    random_seed: int = 20260725
    force: bool = False


PTB_ENDPOINT_SPECS: tuple[PTBEndpointSpec, ...] = (
    PTBEndpointSpec(
        endpoint_id="ptb_ischemia",
        display_name="PTB-XL ischemia/MI label union",
        source_columns=(
            "path_ISCA",
            "path_ISCI",
            "path_AMI",
            "path_IMI",
            "path_LMI",
            "path_PMI",
        ),
        components=(
            "qrs_t_hausdorff_distance",
            "frontal_xy_qrs_t_angle_deg",
            "t_downstroke_velocity_energy",
            "st_loop_convex_hull_volume_3d",
        ),
        mimic_endpoint_ids=(
            "troponin_positive_any",
            "acute_troponin_change_proxy",
            "acute_mi_high_specificity",
            "acute_mi_probable",
            "omi_revascularization_proxy",
        ),
    ),
    PTBEndpointSpec(
        endpoint_id="ptb_lvh",
        display_name="PTB-XL LVH",
        source_columns=("path_LVH",),
        components=(
            "qrs_t_hausdorff_distance",
            "t_downstroke_velocity_energy",
            "cornell_voltage_surrogate",
            "svg_qrs_area_ratio",
        ),
        mimic_endpoint_ids=(
            "echo_lvh_strict",
            "echo_lvh_sensitive",
            "lv_wall_thickening",
            "concentric_remodeling_proxy",
        ),
    ),
    PTBEndpointSpec(
        endpoint_id="ptb_mi_scar",
        display_name="PTB-XL regional myocardial infarction label union",
        source_columns=(
            "path_AMI",
            "path_IMI",
            "path_LMI",
            "path_PMI",
        ),
        components=(
            "qrst_loop_frontal_xy_winding_number",
            "qrs_loop_area_best_fit_plane",
            "qrs_t_vector_correlation_distance",
            "st_loop_area_best_fit_plane",
        ),
        mimic_endpoint_ids=(
            "old_mi_icd",
            "mi_or_revascularization_history",
            "regional_wall_motion_abnormality",
        ),
    ),
    PTBEndpointSpec(
        endpoint_id="ptb_rvh",
        display_name="PTB-XL RVH",
        source_columns=("path_RVH",),
        components=(
            "qrs_terminal_20ms_vector",
            "horizontal_xz_qrs_t_angle_deg",
            "t_loop_planarity",
        ),
        mimic_endpoint_ids=(
            "echo_rv_dilation",
            "echo_rv_systolic_dysfunction",
            "elevated_rvsp_or_pasp",
            "echo_pulmonary_hypertension",
            "right_heart_strain_proxy",
        ),
    ),
    PTBEndpointSpec(
        endpoint_id="ptb_lae",
        display_name="PTB-XL LAO/LAE",
        source_columns=("path_LAO/LAE",),
        components=(
            "p_loop_width_major",
            "p_loop_terminal_force_vector",
            "p_loop_template_instability",
        ),
        mimic_endpoint_ids=(
            "echo_lae",
            "new_onset_af_30d",
            "new_onset_af_90d",
            "new_onset_af_1y",
        ),
    ),
)


def _load_ptb_feature_cache_provenance(
    metadata_path: Path,
    feature_cache: Path,
) -> dict[str, object]:
    candidates = (
        feature_cache.with_suffix(feature_cache.suffix + ".provenance.json"),
        feature_cache.parent / "feature_extraction.provenance.json",
    )
    provenance_path = next(
        (path for path in candidates if path.is_file()),
        None,
    )
    legacy = {
        "status": "unavailable_for_legacy_cache",
        "publication_eligible": False,
        "feature_cache_role": (
            "read_only_legacy_feasibility_cache_bound_by_sha256"
        ),
        "provenance_path": None,
        "reason": (
            "Re-derive PTB-XL v1.0.3 features with the current extractor and "
            "write a bound extraction provenance sidecar before freezing."
        ),
    }
    if provenance_path is None:
        return legacy
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            **legacy,
            "status": "invalid_feature_provenance",
            "provenance_path": str(provenance_path),
            "reason": f"Unreadable provenance sidecar: {exc}",
        }
    expected = {
        "feature_cache_sha256": file_sha256(feature_cache),
        "metadata_sha256": file_sha256(metadata_path),
        "ptbxl_version": "1.0.3",
        "source_record_count": 21_799,
        "feature_record_count": 21_799,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    extraction_hash = str(
        payload.get("feature_extraction_code_hash", "")
    ).strip().lower()
    if len(extraction_hash) != 64 or any(
        character not in "0123456789abcdef"
        for character in extraction_hash
    ):
        mismatches["feature_extraction_code_hash"] = {
            "expected": "64-character SHA-256",
            "observed": extraction_hash,
        }
    if mismatches:
        return {
            **legacy,
            "status": "invalid_feature_provenance",
            "provenance_path": str(provenance_path.resolve()),
            "provenance_sha256": file_sha256(provenance_path),
            "mismatches": mismatches,
            "reason": "Feature provenance does not bind the official source/cache.",
        }
    return {
        "status": "verified_current_extraction",
        "publication_eligible": True,
        "feature_cache_role": "current_rederived_ptbxl_v1_0_3_cache",
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": file_sha256(provenance_path),
        "feature_extraction_code_hash": extraction_hash,
        "ptbxl_version": "1.0.3",
        "source_record_count": 21_799,
        "feature_record_count": 21_799,
    }


def run_ptb_revalidation(
    config: MimicProjectConfig,
    *,
    ptb_root: Path | str,
    feature_cache: Path | str,
    options: PTBRevalidationOptions = PTBRevalidationOptions(),
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    _validate_run_id(options.run_id)
    ptb_root = Path(ptb_root).resolve()
    feature_cache = Path(feature_cache).resolve()
    metadata_path = ptb_root / "ptbxl_database.csv"
    required = (metadata_path, feature_cache)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PTB-XL revalidation inputs are missing: {missing}")
    if options.bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")
    feature_provenance = _load_ptb_feature_cache_provenance(
        metadata_path,
        feature_cache,
    )

    run_root = config.work_root / "ptb_revalidation" / options.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "revalidate.manifest.json"
    code_hash = ptb_implementation_hash()
    package_versions = _ptb_package_versions()
    specification = {
        "split_policy": {
            "train": [1, 2, 3, 4, 5, 6, 7, 8],
            "validation": [9],
            "locked_test": [10],
        },
        "endpoint_specs": [asdict(item) for item in PTB_ENDPOINT_SPECS],
        "bootstrap_iterations": int(options.bootstrap_iterations),
        "random_seed": int(options.random_seed),
        "score_aggregation": "unweighted_mean_train_direction_aligned_zscores",
        "complete_case_required": True,
        "feature_cache_role": feature_provenance["feature_cache_role"],
        "feature_extraction_code_provenance": feature_provenance["status"],
        "publication_eligible_feature_provenance": bool(
            feature_provenance["publication_eligible"]
        ),
    }
    input_hash = content_hash(
        {
            "metadata": file_fingerprint(
                metadata_path,
                include_sha256=True,
            ),
            "feature_cache": file_fingerprint(
                feature_cache,
                include_sha256=True,
            ),
            "feature_cache_provenance": feature_provenance,
            "specification": specification,
            "code_hash": code_hash,
            "package_versions": package_versions,
        }
    )
    config_hash = scientific_config_hash(config)
    if not options.force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash,
            config_hash,
            PTB_REVALIDATION_VERSION,
        ):
            return previous

    if progress:
        progress("Load PTB-XL folds 1-9 only; fold 10 remains sealed")
    frame = _load_ptb_frame(
        metadata_path,
        feature_cache,
        folds=tuple(range(1, 10)),
    )
    _assert_patient_fold_integrity(frame)
    frame["split"] = np.where(
        frame["strat_fold"].eq(9),
        "validation",
        "train",
    )

    marker_records: list[dict] = []
    score_records: list[dict] = []
    marker_payloads: list[dict] = []
    score_payloads: list[dict] = []
    failures: list[dict] = []
    cohort_records: list[dict] = []
    for endpoint_index, endpoint in enumerate(PTB_ENDPOINT_SPECS, start=1):
        if progress:
            progress(
                f"PTB development {endpoint_index}/{len(PTB_ENDPOINT_SPECS)}: "
                f"{endpoint.endpoint_id}"
            )
        endpoint_frame = _with_endpoint_label(frame, endpoint)
        endpoint_frame = endpoint_frame.loc[
            endpoint_frame["label"].notna()
        ].copy()
        endpoint_frame["label"] = endpoint_frame["label"].astype(np.int8)
        train = endpoint_frame.loc[
            endpoint_frame["split"].eq("train")
        ].reset_index(drop=True)
        validation = endpoint_frame.loc[
            endpoint_frame["split"].eq("validation")
        ].reset_index(drop=True)
        for split_name, split_frame in (
            ("train", train),
            ("validation", validation),
        ):
            cohort_records.append(
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "split": split_name,
                    "ecg_count": int(len(split_frame)),
                    "subject_count": int(split_frame["subject_id"].nunique()),
                    "positive_ecg_count": int(split_frame["label"].sum()),
                    "negative_ecg_count": int(
                        split_frame["label"].eq(0).sum()
                    ),
                    "positive_subject_count": int(
                        split_frame.loc[
                            split_frame["label"].eq(1),
                            "subject_id",
                        ].nunique()
                    ),
                    "negative_subject_count": int(
                        split_frame.loc[
                            split_frame["label"].eq(0),
                            "subject_id",
                        ].nunique()
                    ),
                }
            )
        try:
            for component_index, component in enumerate(endpoint.components):
                model = fit_marker_model(
                    train[component],
                    train["label"],
                    component,
                )
                marker_payloads.append(
                    {
                        "endpoint_id": endpoint.endpoint_id,
                        "mimic_endpoint_ids": endpoint.mimic_endpoint_ids,
                        "model": model.to_dict(),
                    }
                )
                marker_records.extend(
                    _ptb_split_records(
                        endpoint.endpoint_id,
                        component,
                        model,
                        train,
                        validation,
                        options,
                        seed_offset=component_index * 10,
                        analysis_type="single_marker",
                    )
                )

            score_spec = fit_frozen_score(
                train,
                train["label"],
                score_id=f"{endpoint.endpoint_id}__ptb_frozen_score",
                endpoint_id=endpoint.endpoint_id,
                components=endpoint.components,
                origin="ptbxl_folds_1_8",
                version=PTB_REVALIDATION_VERSION,
                subject_ids=train["subject_id"],
            )
            train_score = apply_frozen_score(train, score_spec)
            score_model = fit_marker_model(
                train_score,
                train["label"],
                score_spec.score_id,
            )
            split_records = _ptb_split_records(
                endpoint.endpoint_id,
                score_spec.score_id,
                score_model,
                train.assign(_score=train_score),
                validation.assign(
                    _score=apply_frozen_score(validation, score_spec)
                ),
                options,
                seed_offset=50_000 + endpoint_index * 100,
                analysis_type="predefined_combination",
                value_column="_score",
            )
            for record in split_records:
                record.update(
                    {
                        "components": "|".join(endpoint.components),
                        "component_count": len(endpoint.components),
                        "component_directions": "|".join(
                            str(value) for value in score_spec.directions
                        ),
                        "normalization": "ptb_train_folds_1_8_zscore",
                        "aggregation": score_spec.aggregation,
                        "exploratory_score": True,
                        "origin": score_spec.origin,
                    }
                )
            score_records.extend(split_records)
            validation_record = next(
                row
                for row in split_records
                if row["evaluation_split"] == "validation"
            )
            score_payloads.append(
                {
                    "ptb_endpoint_id": endpoint.endpoint_id,
                    "display_name": endpoint.display_name,
                    "mimic_endpoint_ids": endpoint.mimic_endpoint_ids,
                    "score_spec": score_spec.to_dict(),
                    "marker_model": score_model.to_dict(),
                    "ptb_validation_status": validation_record[
                        "inference_status"
                    ],
                    "ptb_validation_roc_auc": validation_record["roc_auc"],
                    "frozen_for_transfer": True,
                    "selection_note": (
                        "Predefined before MIMIC outcome review; freezing does "
                        "not imply successful PTB or clinical validation."
                    ),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "endpoint_id": endpoint.endpoint_id,
                    "stage": "ptb_development_fit",
                    "reason": str(exc),
                }
            )

    outputs = {
        "ptb_marker_results": pd.DataFrame(marker_records),
        "ptb_score_results": pd.DataFrame(score_records),
        "ptb_cohort_summary": pd.DataFrame(cohort_records),
        "ptb_failure_report": pd.DataFrame(failures),
    }
    output_paths: list[str] = []
    for name, output_frame in outputs.items():
        path = run_root / f"{name}.parquet"
        atomic_write_parquet(output_frame, path)
        output_paths.append(str(path.resolve()))

    scores_path = run_root / "ptb_frozen_score_candidates.json"
    markers_path = run_root / "ptb_trained_marker_models.json"
    spec_path = run_root / "ptb_validation_spec.json"
    provenance_path = run_root / "ptb_development_provenance.json"
    atomic_write_json(
        scores_path,
        {
            "score_type": "ptb_predefined_exploratory_score",
            "development_split": "strat_fold_1_8",
            "selection_split": "strat_fold_9",
            "locked_test_read": False,
            "scores": score_payloads,
        },
    )
    atomic_write_json(
        markers_path,
        {
            "model_type": "ptb_train_fitted_univariate_marker",
            "locked_test_read": False,
            "models": marker_payloads,
        },
    )
    atomic_write_json(spec_path, specification)
    atomic_write_json(
        provenance_path,
        {
            "run_id": options.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_hash": code_hash,
            "implementation_hash_version": (
                PTB_IMPLEMENTATION_HASH_VERSION
            ),
            "package_versions": package_versions,
            "input_hash": input_hash,
            "project_config_hash": config_hash,
            "operational_config_hash": operational_config_hash(config),
            "metadata_path": str(metadata_path),
            "feature_cache_path": str(feature_cache),
            "feature_cache_role": feature_provenance["feature_cache_role"],
            "feature_extraction_code_provenance": feature_provenance,
            "publication_eligible_feature_provenance": bool(
                feature_provenance["publication_eligible"]
            ),
            "analysis_implementation_hash": code_hash,
            "folds_read": list(range(1, 10)),
            "locked_fold_10_read": False,
        },
    )
    output_paths.extend(
        str(path.resolve())
        for path in (scores_path, markers_path, spec_path, provenance_path)
    )
    manifest = StageManifest.complete(
        stage=f"ptb-revalidate:{options.run_id}",
        stage_version=PTB_REVALIDATION_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(output_paths),
        row_counts={name: len(value) for name, value in outputs.items()},
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            "PTB-XL folds 1-8 train, fold 9 validation.",
            "PTB-XL fold 10 was not read.",
            f"feature_cache_provenance={feature_provenance['status']}",
            (
                "PTB results are feasibility-only and cannot be frozen"
                if not feature_provenance["publication_eligible"]
                else "PTB feature extraction provenance verified for freeze"
            ),
            "Scores are small complete-case means of train-aligned z-scores.",
        ),
    )
    manifest.save(manifest_path)
    return manifest


def freeze_ptb_revalidation(
    config: MimicProjectConfig,
    *,
    run_id: str = DEFAULT_PTB_TRANSFER_RUN_ID,
) -> TestLock:
    run_root = config.work_root / "ptb_revalidation" / run_id
    payload = _ptb_freeze_payload(config, run_id)
    freeze_hash = content_hash(payload)
    manifest_path = run_root / "ptb_freeze_manifest.json"
    lock_path = run_root / "ptb_test_lock.json"
    if lock_path.exists():
        existing = _load_lock(lock_path)
        if existing.freeze_hash != freeze_hash:
            raise RuntimeError(
                "A different PTB freeze already exists; start a new run ID."
            )
        return existing
    manifest = {
        "freeze_version": PTB_FREEZE_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "freeze_hash": freeze_hash,
        "payload": payload,
        "ptb_fold_10_status": "sealed_unread",
    }
    lock = TestLock(run_id=run_id, freeze_hash=freeze_hash)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(lock_path, lock.to_dict())
    return lock


def evaluate_ptb_locked_test(
    config: MimicProjectConfig,
    *,
    run_id: str = DEFAULT_PTB_TRANSFER_RUN_ID,
    bootstrap_iterations: int = 2000,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    run_root = config.work_root / "ptb_revalidation" / run_id
    manifest_path = run_root / "ptb_freeze_manifest.json"
    lock_path = run_root / "ptb_test_lock.json"
    if not manifest_path.exists() or not lock_path.exists():
        raise FileNotFoundError("Freeze PTB revalidation before reading fold 10")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_payload = _ptb_freeze_payload(config, run_id)
    current_hash = content_hash(current_payload)
    if current_hash != frozen.get("freeze_hash"):
        raise RuntimeError("PTB locked test refused: frozen inputs changed")
    lock = _load_lock(lock_path)
    lock.assert_evaluable(current_hash)
    lock.begin_evaluation()
    atomic_write_json(lock_path, lock.to_dict())

    provenance = json.loads(
        (run_root / "ptb_development_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if progress:
        progress("PTB fold 10 access started and permanently audited")
    frame = _load_ptb_frame(
        Path(provenance["metadata_path"]),
        Path(provenance["feature_cache_path"]),
        folds=(10,),
    )
    _assert_patient_fold_integrity(frame)
    score_payload = json.loads(
        (run_root / "ptb_frozen_score_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    marker_payload = json.loads(
        (run_root / "ptb_trained_marker_models.json").read_text(
            encoding="utf-8"
        )
    )
    endpoint_lookup = {
        item.endpoint_id: item for item in PTB_ENDPOINT_SPECS
    }
    score_records: list[dict] = []
    marker_records: list[dict] = []
    failures: list[dict] = []
    for index, payload in enumerate(score_payload.get("scores", [])):
        endpoint_id = str(payload["ptb_endpoint_id"])
        endpoint = endpoint_lookup[endpoint_id]
        endpoint_frame = _with_endpoint_label(frame, endpoint)
        endpoint_frame = endpoint_frame.loc[
            endpoint_frame["label"].notna()
        ].copy()
        endpoint_frame["label"] = endpoint_frame["label"].astype(np.int8)
        try:
            score_spec = _load_score(payload["score_spec"])
            model = MarkerModel.from_dict(payload["marker_model"])
            score = apply_frozen_score(endpoint_frame, score_spec)
            record = _ptb_metric_record(
                endpoint_id,
                score_spec.score_id,
                model,
                endpoint_frame["subject_id"],
                score,
                endpoint_frame["label"],
                split="locked_test",
                iterations=int(bootstrap_iterations),
                seed=20260725 + index,
                analysis_type="predefined_combination",
            )
            record.update(
                {
                    "components": "|".join(score_spec.components),
                    "component_count": len(score_spec.components),
                    "component_directions": "|".join(
                        str(value) for value in score_spec.directions
                    ),
                    "normalization": "ptb_train_folds_1_8_zscore",
                    "aggregation": score_spec.aggregation,
                    "exploratory_score": True,
                    "origin": score_spec.origin,
                }
            )
            score_records.append(record)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "stage": "ptb_locked_score",
                    "reason": str(exc),
                }
            )

    for index, payload in enumerate(marker_payload.get("models", [])):
        endpoint_id = str(payload["endpoint_id"])
        endpoint = endpoint_lookup[endpoint_id]
        endpoint_frame = _with_endpoint_label(frame, endpoint)
        endpoint_frame = endpoint_frame.loc[
            endpoint_frame["label"].notna()
        ].copy()
        endpoint_frame["label"] = endpoint_frame["label"].astype(np.int8)
        try:
            model = MarkerModel.from_dict(payload["model"])
            marker_records.append(
                _ptb_metric_record(
                    endpoint_id,
                    model.feature_id,
                    model,
                    endpoint_frame["subject_id"],
                    endpoint_frame[model.feature_id],
                    endpoint_frame["label"],
                    split="locked_test",
                    iterations=int(bootstrap_iterations),
                    seed=20270725 + index,
                    analysis_type="single_marker",
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "stage": "ptb_locked_marker",
                    "reason": str(exc),
                }
            )

    outputs = {
        "ptb_final_test_scores": pd.DataFrame(score_records),
        "ptb_final_test_markers": pd.DataFrame(marker_records),
        "ptb_final_test_failures": pd.DataFrame(failures),
    }
    output_paths: list[str] = []
    for name, output_frame in outputs.items():
        path = run_root / f"{name}.parquet"
        atomic_write_parquet(output_frame, path)
        output_paths.append(str(path.resolve()))
    artifact_hash = content_hash(
        {
            "freeze_hash": current_hash,
            "outputs": {
                Path(path).name: file_sha256(path) for path in output_paths
            },
        }
    )
    audit_path = run_root / "ptb_test_evaluation_audit.json"
    atomic_write_json(
        audit_path,
        {
            "run_id": run_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "freeze_hash": current_hash,
            "artifact_hash": artifact_hash,
            "folds_read": [10],
            "row_counts": {
                name: len(output_frame)
                for name, output_frame in outputs.items()
            },
        },
    )
    output_paths.append(str(audit_path.resolve()))
    lock.consume(artifact_hash)
    atomic_write_json(lock_path, lock.to_dict())
    stage = StageManifest.complete(
        stage=f"ptb-evaluate-test:{run_id}",
        stage_version=PTB_TEST_VERSION,
        input_hash=current_hash,
        config_hash=scientific_config_hash(config),
        output_paths=tuple(output_paths),
        row_counts={
            name: len(output_frame)
            for name, output_frame in outputs.items()
        },
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            "PTB-XL fold 10 consumed exactly once.",
            "All score directions, scaling and thresholds came from folds 1-8.",
        ),
    )
    stage.save(run_root / "ptb-evaluate-test.manifest.json")
    return stage


def load_ptb_frozen_transfer_scores(
    config: MimicProjectConfig,
    *,
    run_id: str = DEFAULT_PTB_TRANSFER_RUN_ID,
) -> tuple[list[dict], dict]:
    run_root = config.work_root / "ptb_revalidation" / run_id
    manifest_path = run_root / "ptb_freeze_manifest.json"
    if not manifest_path.exists():
        return [], {"status": "not_available", "run_id": run_id}
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_payload = _ptb_freeze_payload(config, run_id)
    current_hash = content_hash(current_payload)
    if current_hash != frozen.get("freeze_hash"):
        raise RuntimeError(
            "PTB transfer scores cannot be loaded because their freeze changed"
        )
    scores_path = run_root / "ptb_frozen_score_candidates.json"
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    scores = [
        row
        for row in payload.get("scores", [])
        if row.get("frozen_for_transfer", False)
    ]
    return scores, {
        "status": "verified_frozen",
        "run_id": run_id,
        "freeze_hash": current_hash,
        "freeze_manifest_path": str(manifest_path.resolve()),
        "score_path": str(scores_path.resolve()),
        "score_file_sha256": file_sha256(scores_path),
        "locked_fold_10_read_for_transfer": False,
    }


def _load_ptb_frame(
    metadata_path: Path,
    feature_cache: Path,
    *,
    folds: tuple[int, ...],
) -> pd.DataFrame:
    if not folds or any(fold not in range(1, 11) for fold in folds):
        raise ValueError(f"Invalid PTB fold selection: {folds}")
    all_components = tuple(
        dict.fromkeys(
            component
            for endpoint in PTB_ENDPOINT_SPECS
            for component in endpoint.components
        )
    )
    all_labels = tuple(
        dict.fromkeys(
            source
            for endpoint in PTB_ENDPOINT_SPECS
            for source in endpoint.source_columns
        )
    )
    relation = (
        f"read_parquet('{_sql_path(feature_cache)}')"
        if feature_cache.suffix.casefold() == ".parquet"
        else f"read_csv_auto('{_sql_path(feature_cache)}', header=true)"
    )
    component_sql = ",\n            ".join(
        f'c.{_quoted_identifier("bm_" + component)} '
        f"AS {_quoted_identifier(component)}"
        for component in all_components
    )
    label_sql = ",\n            ".join(
        f"c.{_quoted_identifier(label)} AS {_quoted_identifier(label)}"
        for label in all_labels
    )
    folds_sql = ", ".join(str(int(fold)) for fold in folds)
    from .storage import DuckDBCatalog
    with DuckDBCatalog(feature_cache.parent / ".transfer-read.duckdb", memory_limit_gb=2,
                       threads=2).connect(isolated=True) as connection:
        columns = {
            row[0]
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {relation}"
            ).fetchall()
        }
        required = {
            "ecg_id",
            *(f"bm_{component}" for component in all_components),
            *all_labels,
        }
        missing = sorted(required - columns)
        if missing:
            raise KeyError(
                "PTB feature cache lacks required frozen-panel columns: "
                f"{missing}"
            )
        return connection.execute(
            f"""
            SELECT
                c.ecg_id,
                TRY_CAST(d.patient_id AS BIGINT) AS subject_id,
                TRY_CAST(d.strat_fold AS INTEGER) AS strat_fold,
                {component_sql},
                {label_sql}
            FROM {relation} c
            JOIN read_csv_auto(
                '{_sql_path(metadata_path)}',
                header=true
            ) d USING (ecg_id)
            WHERE TRY_CAST(d.strat_fold AS INTEGER) IN ({folds_sql})
              AND TRY_CAST(d.patient_id AS BIGINT) IS NOT NULL
            ORDER BY strat_fold, subject_id, ecg_id
            """
        ).fetchdf()


def _with_endpoint_label(
    frame: pd.DataFrame,
    endpoint: PTBEndpointSpec,
) -> pd.DataFrame:
    output = frame.copy()
    source = output.loc[:, list(endpoint.source_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    positive = source.eq(1).any(axis=1)
    completely_assessed = source.notna().all(axis=1)
    output["label"] = np.where(
        positive,
        1.0,
        np.where(completely_assessed, 0.0, np.nan),
    )
    return output


def _assert_patient_fold_integrity(frame: pd.DataFrame) -> None:
    overlap = frame.groupby("subject_id")["strat_fold"].nunique()
    if bool((overlap > 1).any()):
        raise RuntimeError(
            "PTB-XL patient leakage detected across stratified folds"
        )


def _ptb_split_records(
    endpoint_id: str,
    feature_id: str,
    model: MarkerModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    options: PTBRevalidationOptions,
    *,
    seed_offset: int,
    analysis_type: str,
    value_column: Optional[str] = None,
) -> list[dict]:
    records = []
    column = value_column or feature_id
    for split_index, (split, split_frame) in enumerate(
        (("train", train), ("validation", validation))
    ):
        records.append(
            _ptb_metric_record(
                endpoint_id,
                feature_id,
                model,
                split_frame["subject_id"],
                split_frame[column],
                split_frame["label"],
                split=split,
                iterations=int(options.bootstrap_iterations),
                seed=options.random_seed + seed_offset + split_index,
                analysis_type=analysis_type,
            )
        )
    return records


def _ptb_metric_record(
    endpoint_id: str,
    feature_id: str,
    model: MarkerModel,
    subject_ids: Iterable,
    values: Iterable,
    labels: Iterable,
    *,
    split: str,
    iterations: int,
    seed: int,
    analysis_type: str,
) -> dict:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=float
    )
    binary = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(
        dtype=float
    )
    subjects = np.asarray(subject_ids)
    metrics = evaluate_marker(
        numeric,
        binary,
        direction=model.direction,
        thresholds=model.thresholds,
    )
    complete = (
        np.isfinite(numeric)
        & np.isfinite(binary)
        & np.isin(binary, (0, 1))
    )
    positive_subjects = int(
        pd.Series(subjects[complete & (binary == 1)]).nunique()
    )
    negative_subjects = int(
        pd.Series(subjects[complete & (binary == 0)]).nunique()
    )
    status = inference_status(
        positive_subjects,
        negative_subjects,
    )
    metrics = enforce_inference_policy(metrics, status)
    uncertainty = marker_bootstrap_intervals(
        subjects,
        numeric,
        binary,
        direction=model.direction,
        thresholds=model.thresholds,
        iterations=iterations if status != "descriptive_only" else 0,
        seed=seed,
    )
    return {
        "endpoint_id": endpoint_id,
        "feature_id": feature_id,
        "analysis_type": analysis_type,
        "evaluation_split": split,
        "metric_scope": (
            "ptb_in_sample_train"
            if split == "train"
            else (
                "ptb_locked_test_final"
                if split == "locked_test"
                else "ptb_validation_fixed_train_spec"
            )
        ),
        "direction": model.direction,
        "threshold_source": "ptb_train_folds_1_8",
        "missing_count": int(np.sum(~np.isfinite(numeric))),
        "missing_fraction": float(np.mean(~np.isfinite(numeric))),
        "subject_count": int(pd.Series(subjects).nunique()),
        "positive_subject_count": positive_subjects,
        "negative_subject_count": negative_subjects,
        "inference_status": status,
        **metrics,
        **uncertainty,
    }


def _ptb_freeze_payload(
    config: MimicProjectConfig,
    run_id: str,
) -> dict:
    run_root = config.work_root / "ptb_revalidation" / run_id
    required = {
        "revalidation_manifest": run_root / "revalidate.manifest.json",
        "score_candidates": run_root / "ptb_frozen_score_candidates.json",
        "marker_models": run_root / "ptb_trained_marker_models.json",
        "validation_spec": run_root / "ptb_validation_spec.json",
        "development_provenance": (
            run_root / "ptb_development_provenance.json"
        ),
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PTB freeze inputs are missing: {missing}")
    provenance = json.loads(
        required["development_provenance"].read_text(encoding="utf-8")
    )
    current_code_hash = ptb_implementation_hash()
    if provenance.get("code_hash") != current_code_hash:
        raise RuntimeError(
            "PTB freeze refused: code changed after revalidation"
        )
    if provenance.get("locked_fold_10_read") is not False:
        raise RuntimeError(
            "PTB freeze refused: development provenance is not blind to fold 10"
        )
    feature_provenance = provenance.get("feature_extraction_code_provenance")
    if (
        provenance.get("publication_eligible_feature_provenance") is not True
        or not isinstance(feature_provenance, dict)
        or feature_provenance.get("status")
        != "verified_current_extraction"
    ):
        raise RuntimeError(
            "PTB freeze refused: the legacy feature cache has no verified "
            "current extraction provenance. Re-derive all PTB-XL v1.0.3 "
            "features and bind metadata/cache/code hashes first."
        )
    package_versions = _ptb_package_versions()
    if provenance.get("package_versions") != package_versions:
        raise RuntimeError(
            "PTB freeze refused: statistical package versions changed after "
            "revalidation"
        )
    return {
        "freeze_version": PTB_FREEZE_VERSION,
        "run_id": run_id,
        "code_hash": current_code_hash,
        "project_config_hash": provenance.get("project_config_hash", ""),
        "implementation_hash_version": PTB_IMPLEMENTATION_HASH_VERSION,
        "package_versions": package_versions,
        "endpoint_specs_hash": content_hash(
            [asdict(item) for item in PTB_ENDPOINT_SPECS]
        ),
        "tracked_file_sha256": {
            name: file_sha256(path) for name, path in required.items()
        },
        "test_policy": {
            "development_folds": list(range(1, 9)),
            "selection_fold": 9,
            "locked_test_fold": 10,
            "one_time": True,
        },
    }


def _load_score(payload: dict) -> FrozenScoreSpec:
    values = dict(payload)
    for key in (
        "components",
        "directions",
        "means",
        "standard_deviations",
    ):
        values[key] = tuple(values[key])
    return FrozenScoreSpec(**values)


def ptb_implementation_hash() -> str:
    mimic_root = Path(__file__).resolve().parent
    return content_hash(
        {
            "version": PTB_IMPLEMENTATION_HASH_VERSION,
            "ptb_transfer": file_sha256(__file__),
            "statistics": file_sha256(mimic_root / "statistics.py"),
            "models": file_sha256(mimic_root / "models.py"),
        }
    )


def _ptb_package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "duckdb",
        "pyarrow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _load_lock(path: Path) -> TestLock:
    return TestLock(**json.loads(path.read_text(encoding="utf-8")))


def _validate_run_id(run_id: str) -> None:
    if (
        not run_id
        or not run_id.replace("-", "").replace("_", "").isalnum()
        or run_id in {".", ".."}
    ):
        raise ValueError(f"Unsafe run ID: {run_id!r}")


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")
