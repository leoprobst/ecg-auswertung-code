from __future__ import annotations

import html
import importlib.metadata
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .clinical_utility import validate_clinical_utility_payload
from .config import MimicProjectConfig
from .feature_roles import feature_analysis_role, is_candidate_biomarker
from .locking import _load_test_lock
from .statistics import MarkerModel
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_fingerprint,
    source_tree_hash,
    verify_output_integrity,
)
from .validation import (
    _load_endpoint_frame,
    _report_comparator_for_endpoint,
)
from .user_guidance import annotate_result_frame


REPORT_STAGE_VERSION = "1.12.0"
MIMIC_DUA_NOTICE = (
    "Contains derived MIMIC data. Keep local and handle under the PhysioNet "
    "Credentialed Health Data Use Agreement. Do not upload patient-level data "
    "to external services."
)


def _scientific_backend_hash() -> str:
    from .code_scope import scientific_source_hash

    package_root = Path(__file__).resolve().parents[1]
    return content_hash({
        "mimic": scientific_source_hash(package_root / "mimic"),
        "core": source_tree_hash(package_root / "core"),
    })


def generate_run_reports(
    config: MimicProjectConfig,
    run_id: str,
    *,
    feature_glob: Optional[str] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    run_root = config.runs_root / run_id
    if not run_root.exists():
        raise FileNotFoundError(f"Validation run does not exist: {run_root}")
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    manifest_path = report_root / "report.manifest.json"

    source_paths = _report_source_paths(config, run_root)
    clinical_utility_bundle = resolve_clinical_utility_report_paths(run_root)
    acceptance_json_path, _ = resolve_acceptance_audit_paths(run_root)
    input_hash = content_hash(
        {
            name: (
                file_fingerprint(path) if path is not None and path.exists() else None
            )
            for name, path in source_paths.items()
        }
        | {
            "run_id": run_id,
            "feature_glob": feature_glob,
            # Scientific backend only: GUI, tests and run machinery
            # (code_scope) cannot change a report value.
            "code_hash": _scientific_backend_hash(),
            "acceptance_audit_json": (
                file_fingerprint(acceptance_json_path)
                if acceptance_json_path is not None
                else None
            ),
            "clinical_utility_assessment": (
                file_fingerprint(clinical_utility_bundle["assessment"])
                if clinical_utility_bundle is not None
                else None
            ),
            "clinical_utility_manifest": (
                file_fingerprint(clinical_utility_bundle["manifest"])
                if clinical_utility_bundle is not None
                else None
            ),
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, REPORT_STAGE_VERSION):
            return previous

    if progress:
        progress("Sammle Run-, Daten-, Label- und Validierungsberichte")
    tables = _load_report_tables(config, run_id, source_paths)
    clinical_summary = tables["clinical_utility_summary"].iloc[0].to_dict()
    review_candidates, discordance, review_diagnostics = _build_review_candidates(
        config,
        run_id,
        tables,
        feature_glob=feature_glob,
        progress=progress,
    )
    tables["review_candidates"] = review_candidates
    tables["discordance_report"] = discordance
    tables["review_candidate_diagnostics"] = review_diagnostics
    tables["run_manifest"] = _run_manifest_frame(
        config,
        run_id,
        tables,
        source_paths,
    )

    output_paths: list[str] = []
    for name, frame in tables.items():
        parquet_path = report_root / f"{name}.parquet"
        csv_path = report_root / f"{name}.csv"
        atomic_write_parquet(_parquet_safe(frame), parquet_path)
        _atomic_write_csv(frame, csv_path)
        output_paths.extend(
            (str(parquet_path.resolve()), str(csv_path.resolve()))
        )

    metadata_path = report_root / "report_metadata.json"
    atomic_write_json(
        metadata_path,
        {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "canonical_format": "parquet",
            "dua_notice": MIMIC_DUA_NOTICE,
            "clinical_scope": (
                "Research-only clinical reference comparison; no individual "
                "diagnosis and no treatment recommendation."
            ),
            "clinical_utility_assessment_status": clinical_summary.get(
                "assessment_status", "not_assessed"
            ),
            "clinical_utility_claim_level": clinical_summary.get(
                "claim_level", "hypothesis_only"
            ),
            "decision_process_utility_demonstrated": bool(
                clinical_summary.get(
                    "decision_process_utility_demonstrated", False
                )
            ),
            "patient_benefit_demonstrated": bool(
                clinical_summary.get("patient_benefit_demonstrated", False)
            ),
            "clinical_utility_disclaimer": clinical_summary.get(
                "disclaimer",
                "No clinical-utility assessment is available; no utility claim is allowed.",
            ),
            "result_tiers": {
                "primary_core": "pre-specified primary panel",
                "sentinel": "separately FDR-controlled exploratory panel",
                "discovery_reserve": "exploratory; replication required",
            },
        },
    )
    output_paths.append(str(metadata_path.resolve()))

    if progress:
        progress("Erzeuge lokale Excel- und HTML-Zusammenfassungen")
    excel_path = report_root / f"{run_id}_reports.xlsx"
    html_path = report_root / f"{run_id}_report.html"
    _atomic_write_excel(tables, excel_path)
    atomic_write_text(html_path, _render_html_report(run_id, tables))
    output_paths.extend((str(excel_path.resolve()), str(html_path.resolve())))

    manifest = StageManifest.complete(
        stage=f"report:{run_id}",
        stage_version=REPORT_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(output_paths),
        row_counts={name: len(frame) for name, frame in tables.items()},
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            MIMIC_DUA_NOTICE,
            "Parquet is canonical; CSV, Excel and HTML are convenience exports.",
        ),
    )
    manifest.save(manifest_path)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    catalog.register_stage(manifest, manifest_path)
    return manifest


def _report_source_paths(
    config: MimicProjectConfig,
    run_root: Path,
) -> dict[str, Optional[Path]]:
    feature_failure_path = _run_feature_failure_path(run_root)
    _, acceptance_table_path = resolve_acceptance_audit_paths(run_root)
    clinical_utility = resolve_clinical_utility_report_paths(run_root)
    return {
        "acceptance_audit": acceptance_table_path,
        "clinical_utility_summary": (
            clinical_utility["summary"]
            if clinical_utility is not None
            else None
        ),
        "clinical_utility_gates": (
            clinical_utility["gates"]
            if clinical_utility is not None
            else None
        ),
        "data_inventory": config.work_root / "data_inventory.parquet",
        "ecg_master_index_summary": (
            config.master_root / "ecg_master_index_summary.parquet"
        ),
        "label_summary": config.labels_root / "label_summary.parquet",
        "split_summary": config.labels_root / "split_summary.parquet",
        "label_window_sensitivity_summary": (
            config.labels_root / "label_window_sensitivity_summary.parquet"
        ),
        "feature_failure_report": feature_failure_path,
        "screening_results": run_root / "screening_results.parquet",
        "classical_baseline_results": (
            run_root / "classical_baseline_results.parquet"
        ),
        "machine_report_baseline_results": (
            run_root / "machine_report_baseline_results.parquet"
        ),
        "vcg_vs_classical_comparison": (
            run_root / "vcg_vs_classical_comparison.parquet"
        ),
        "descriptive_validation_ranking": (
            run_root / "descriptive_validation_ranking.parquet"
        ),
        "baseline_registry": run_root / "baseline_registry.parquet",
        "methodology_registry": run_root / "methodology_registry.parquet",
        "combination_results": run_root / "combination_results.parquet",
        "overfitting_stability_report": (
            run_root / "overfitting_stability_report.parquet"
        ),
        "ptb_transfer_matrix": run_root / "ptb_transfer_matrix.parquet",
        "adjusted_model_results": run_root / "adjusted_model_results.parquet",
        "decision_curve_results": run_root / "decision_curve_results.parquet",
        "all_ecg_clustered_results": (
            run_root / "all_ecg_clustered_results.parquet"
        ),
        "performance_curve_points": (
            run_root / "performance_curve_points.parquet"
        ),
        "score_distribution_bins": (
            run_root / "score_distribution_bins.parquet"
        ),
        "operating_point_results": (
            run_root / "operating_point_results.parquet"
        ),
        "pathology_marker_explorer": (
            run_root / "pathology_marker_explorer.parquet"
        ),
        "marker_pathology_explorer": (
            run_root / "marker_pathology_explorer.parquet"
        ),
        "endpoint_summary_explorer": (
            run_root / "endpoint_summary_explorer.parquet"
        ),
        "criterion_candidates": run_root / "criterion_candidates.parquet",
        "criterion_thresholds": run_root / "criterion_thresholds.parquet",
        "validation_failure_report": (
            run_root / "validation_failure_report.parquet"
        ),
        "validation_label_counts": (
            run_root / "validation_label_counts.parquet"
        ),
        "final_test_report": run_root / "final_test_report.parquet",
        "final_test_combination_results": (
            run_root / "final_test_combination_results.parquet"
        ),
        "final_test_classical_baselines": (
            run_root / "final_test_classical_baselines.parquet"
        ),
        "final_test_machine_report_baselines": (
            run_root / "final_test_machine_report_baselines.parquet"
        ),
        "final_test_label_counts": (
            run_root / "final_test_label_counts.parquet"
        ),
        "final_test_failures": run_root / "final_test_failures.parquet",
        "final_test_adjusted_models": (
            run_root / "final_test_adjusted_models.parquet"
        ),
        "final_test_decision_curves": (
            run_root / "final_test_decision_curves.parquet"
        ),
        "final_test_window_sensitivity": (
            run_root / "final_test_window_sensitivity.parquet"
        ),
        "final_test_all_ecg_clustered": (
            run_root / "final_test_all_ecg_clustered.parquet"
        ),
    }


def resolve_clinical_utility_report_paths(
    run_root: Path,
) -> Optional[dict[str, Path]]:
    """Resolve only a complete, tamper-evident utility bundle for reporting."""

    root = run_root / "clinical_utility"
    if not root.exists():
        return None
    paths = {
        "assessment": root / "clinical_utility_assessment.json",
        "gates": root / "clinical_utility_gates.parquet",
        "summary": root / "clinical_utility_summary.parquet",
        "report": root / "clinical_utility_assessment.md",
        "manifest": root / "clinical_utility_assessment.manifest.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Clinical-utility report bundle is incomplete: " + ", ".join(missing)
        )
    manifest = StageManifest.load(paths["manifest"])
    if (
        manifest.status != "complete"
        or not manifest.stage.startswith("clinical-utility-assessment:")
        or not manifest.output_integrity
        or not verify_output_integrity(manifest.output_integrity)
    ):
        raise RuntimeError("Clinical-utility report bundle failed integrity checks")
    expected_outputs = {
        str(paths[name].resolve())
        for name in ("assessment", "gates", "summary", "report")
    }
    if {str(Path(path).resolve()) for path in manifest.output_paths} != expected_outputs:
        raise RuntimeError("Clinical-utility manifest output scope is inconsistent")
    try:
        payload = json.loads(paths["assessment"].read_text(encoding="utf-8"))
        validate_clinical_utility_payload(payload)
        summary = pd.read_parquet(paths["summary"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Clinical-utility assessment is not reportable") from exc
    if len(summary) != 1:
        raise RuntimeError("Clinical-utility summary must contain exactly one row")
    row = summary.iloc[0]
    expected = {
        "candidate_id": payload["candidate_id"],
        "claim_level": payload["claim_level"],
        "claim_rank": payload["claim_rank"],
        "decision_process_utility_demonstrated": payload[
            "decision_process_utility_demonstrated"
        ],
        "patient_benefit_demonstrated": payload[
            "patient_benefit_demonstrated"
        ],
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Clinical-utility summary disagrees with assessment")
    return paths


def resolve_acceptance_audit_paths(
    run_root: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """Return the strongest complete development acceptance audit available."""

    for stem in ("acceptance_audit", "acceptance_pre_freeze_audit"):
        json_path = run_root / f"{stem}.json"
        parquet_path = run_root / f"{stem}.parquet"
        if json_path.is_file() and parquet_path.is_file():
            return json_path, parquet_path
    return None, None


def _run_feature_failure_path(run_root: Path) -> Optional[Path]:
    """Resolve QC from the immutable feature artifact used by this run."""
    provenance_path = run_root / "development_provenance.json"
    if not provenance_path.is_file():
        return None
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        feature_root = Path(
            str(provenance["feature_integrity"]["root"])
        ).resolve()
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    candidate = feature_root / "feature_failure_report.parquet"
    return candidate if candidate.is_file() else None


def _load_report_tables(
    config: MimicProjectConfig,
    run_id: str,
    source_paths: dict[str, Optional[Path]],
) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, path in source_paths.items():
        if path is not None and path.exists():
            tables[name] = _annotate_feature_role(pd.read_parquet(path))
        elif name == "final_test_report":
            tables[name] = pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "status": "sealed_not_evaluated",
                        "message": (
                            "Locked test has not been consumed. This is not a "
                            "final performance result."
                        ),
                    }
                ]
            )
        elif name == "clinical_utility_summary":
            tables[name] = pd.DataFrame(
                [
                    {
                        "assessment_status": "not_assessed",
                        "candidate_id": "not_assessed",
                        "use_id": "not_prespecified",
                        "framework_scope": "not_assessed",
                        "claim_level": "hypothesis_only",
                        "claim_rank": 0,
                        "next_claim_level": "analytical_validity_supported",
                        "decision_process_utility_demonstrated": False,
                        "patient_benefit_demonstrated": False,
                        "blocking_gates_for_demonstrated_utility": (
                            '["clinical_utility_assessment_missing"]'
                        ),
                        "blocking_gates_for_patient_benefit": (
                            '["clinical_utility_assessment_missing"]'
                        ),
                        "disclaimer": (
                            "No verified clinical-utility assessment is available; "
                            "no clinical-utility or patient-benefit claim is allowed."
                        ),
                    }
                ]
            )
        elif name == "clinical_utility_gates":
            tables[name] = pd.DataFrame(
                [
                    {
                        "gate_id": "clinical_utility_assessment_present",
                        "domain": "governance",
                        "passed": False,
                        "status": "not_assessed",
                        "missing_requirements": (
                            '["verified clinical-utility assessment"]'
                        ),
                        "rationale": (
                            "No verified run-scoped assessment bundle was found."
                        ),
                    }
                ]
            )
        else:
            tables[name] = pd.DataFrame(
                [{"status": "not_available", "source": name}]
            )
    return tables


def _annotate_feature_role(frame: pd.DataFrame) -> pd.DataFrame:
    result = annotate_result_frame(frame)
    if "feature_id" in result:
        result["feature_analysis_role"] = result["feature_id"].map(
            feature_analysis_role
        )
    return result


def _build_review_candidates(
    config: MimicProjectConfig,
    run_id: str,
    tables: dict[str, pd.DataFrame],
    *,
    feature_glob: Optional[str],
    progress: Optional[Callable[[str], None]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_root = config.runs_root / run_id
    lock_path = run_root / "test_lock.json"
    locked_consumed = (
        lock_path.exists() and _load_test_lock(lock_path).is_consumed
    )
    evaluation_split = "locked_test" if locked_consumed else "validation"
    screening = (
        tables["final_test_report"]
        if locked_consumed
        else tables["screening_results"]
    )
    if (
        screening.empty
        or "evaluation_split" not in screening
        or "feature_id" not in screening
        or "panel_id" not in screening
    ):
        return _empty_review_result(
            "not_generated",
            "Screening results do not contain reviewable marker columns.",
        )

    selected = screening.loc[
        (screening["evaluation_split"] == evaluation_split)
        & screening["panel_id"].isin(("primary_core", "sentinel"))
        & screening["roc_auc"].notna()
        & screening["feature_id"].map(is_candidate_biomarker)
    ].copy()
    if selected.empty:
        return _empty_review_result(
            "not_generated",
            f"No reviewable {evaluation_split} marker result is available.",
        )
    selected["_inference_priority"] = (
        selected.get(
            "inference_status",
            pd.Series("descriptive_only", index=selected.index),
        )
        .map(
            {
                "formal_inference": 0,
                "underpowered": 1,
                "descriptive_only": 2,
            }
        )
        .fillna(3)
    )
    selected = (
        selected.sort_values(
            [
                "endpoint_id",
                "_inference_priority",
                "roc_auc",
                "missingness",
            ],
            ascending=[True, True, False, True],
        )
        .groupby("endpoint_id", as_index=False)
        .head(1)
    )
    marker_payload_path = run_root / "trained_marker_models.json"
    if not marker_payload_path.exists():
        return _empty_review_result(
            "not_generated",
            f"Trained marker payload is missing: {marker_payload_path}",
        )
    marker_payload = json.loads(
        marker_payload_path.read_text(encoding="utf-8")
    )
    model_lookup = {
        (row["endpoint_id"], row["feature_id"]): MarkerModel.from_dict(row)
        for row in marker_payload.get("models", [])
    }
    classical_table = (
        tables.get("final_test_classical_baselines", pd.DataFrame())
        if locked_consumed
        else tables["classical_baseline_results"]
    )
    if locked_consumed:
        final_classical_path = run_root / "final_test_classical_baselines.parquet"
        if final_classical_path.exists():
            classical_table = pd.read_parquet(final_classical_path)

    resolved_feature_glob = feature_glob or _default_feature_glob(
        config,
        evaluation_split,
        run_id=run_id,
    )
    cohort_path = config.labels_root / "endpoint_index_cohorts.parquet"
    master_path = config.master_root / "ecg_master_index.parquet"
    reports_path = config.evidence_root / "machine_report_labels.parquet"
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    rows: list[dict] = []
    diagnostics: list[dict] = []
    review_context_columns = (
        "troponin_max_url_ratio",
        "troponin_count_standard",
        "acute_troponin_change_proxy_value",
        "potassium_nearest",
        "potassium_delta_hours",
        "potassium_count_6h",
        "magnesium_nearest",
        "total_calcium_nearest",
        "ionized_calcium_nearest",
        "creatinine_nearest",
        "hemoglobin_nearest",
        "lactate_nearest",
        "preferred_lvef",
        "preferred_lvef_method",
        "lv_mass_index",
        "la_volume_index",
        "rv_basal_diameter_cm",
        "rv_function_category",
        "tapse_cm",
        "derived_rvsp_or_pasp_mmhg",
        "rvsp_or_pasp_method",
        "proc_pci",
        "proc_coronary_angiography",
        "proc_cabg",
        "proc_valve_procedure",
        "pci_day_delta",
        "cath_day_delta",
        "med_qt_prolonging_drug",
        "med_antiarrhythmic_class_i_iii",
        "med_digoxin",
        "med_vasopressor",
        "med_antiplatelet",
        "med_heparin",
        "med_nitrate",
        "med_electrolyte_replacement",
        "medication_assessed",
    )
    for index, marker in enumerate(selected.itertuples(index=False), start=1):
        endpoint_id = str(marker.endpoint_id)
        feature_id = str(marker.feature_id)
        marker_inference_status = str(
            getattr(marker, "inference_status", "not_recorded")
        )
        model = model_lookup.get((endpoint_id, feature_id))
        if model is None:
            diagnostics.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": feature_id,
                    "status": "skipped",
                    "reason": "No matching train-fitted marker model.",
                    "error_type": None,
                    "error_message": None,
                }
            )
            continue
        if progress:
            progress(
                f"Review-Candidates {index:,}/{len(selected):,}: {endpoint_id}"
            )
        classical_id = _best_classical_id(
            classical_table,
            endpoint_id,
            evaluation_split,
        )
        selected_columns = [
            feature_id,
            "extraction_status",
            "signal_quality_score",
            "signal_quality_tier",
            "fiducial_confidence",
        ]
        if classical_id:
            selected_columns.append(f"{classical_id}_positive")
        report_label_id = _report_comparator_for_endpoint(endpoint_id)
        try:
            frame = _load_endpoint_frame(
                catalog,
                cohort_path=cohort_path,
                master_path=master_path,
                reports_path=reports_path if reports_path.exists() else None,
                context_path=(
                    config.evidence_root
                    / "clinical_context_features.parquet"
                ),
                feature_glob=resolved_feature_glob,
                endpoint_id=endpoint_id,
                report_label_id=report_label_id,
                splits=(evaluation_split,),
                feature_columns=tuple(selected_columns),
                context_columns=review_context_columns,
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": feature_id,
                    "status": "error",
                    "reason": "Endpoint frame could not be loaded.",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:2000],
                }
            )
            continue
        if frame.empty or feature_id not in frame:
            diagnostics.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": feature_id,
                    "status": "skipped",
                    "reason": "No assessable feature rows for review.",
                    "error_type": None,
                    "error_message": None,
                }
            )
            continue
        values = pd.to_numeric(frame[feature_id], errors="coerce").to_numpy()
        threshold = float(model.thresholds["specificity_90"])
        aligned = model.direction * values
        evaluable = np.isfinite(aligned)
        vcg_positive = evaluable & (aligned >= threshold)
        label_positive = frame["label_state"].eq("positive").to_numpy()
        label_negative = frame["label_state"].eq("negative").to_numpy()
        report_states = frame["report_reference_state"].fillna(
            "not_available"
        ).astype(str).to_numpy()
        classical_prediction = np.full(len(frame), None, dtype=object)
        if classical_id and f"{classical_id}_positive" in frame:
            raw_classical = frame[f"{classical_id}_positive"]
            classical_prediction[raw_classical.notna().to_numpy()] = (
                raw_classical.dropna().astype(bool).to_numpy()
            )

        categories = [
            (
                "clinical_positive_classical_negative_vcg_positive",
                label_positive
                & (classical_prediction == False)  # noqa: E712
                & vcg_positive,
            ),
            (
                "clinical_positive_report_negative_vcg_positive",
                label_positive
                & (report_states == "negative")
                & vcg_positive,
            ),
            (
                "clinical_positive_report_not_mentioned_vcg_positive",
                label_positive
                & (report_states == "not_mentioned")
                & vcg_positive,
            ),
            (
                "clinical_negative_classical_positive_vcg_negative",
                label_negative
                & (classical_prediction == True)  # noqa: E712
                & ~vcg_positive
                & evaluable,
            ),
            (
                "clinical_negative_vcg_positive",
                label_negative & vcg_positive,
            ),
        ]
        for category, category_mask in categories:
            candidate_indices = np.flatnonzero(category_mask)
            if len(candidate_indices) == 0:
                continue
            distances = aligned[candidate_indices] - threshold
            order = np.argsort(-np.abs(distances), kind="stable")[:50]
            for local_index in candidate_indices[order]:
                item = frame.iloc[local_index]
                rows.append(
                    {
                        "subject_id": int(item.subject_id),
                        "study_id": int(item.study_id),
                        "split": evaluation_split,
                        "ecg_time": item.ecg_time,
                        "endpoint_id": endpoint_id,
                        "clinical_label_state": item.label_state,
                        "label_confidence": item.confidence,
                        "label_time_precision": item.time_precision,
                        "label_evidence_ids": item.evidence_ids_json,
                        "label_evidence": item.evidence_summary,
                        "source_tables": item.source_tables,
                        "time_delta_hours": item.time_delta_hours,
                        "lab_context": _compact_context_json(
                            item,
                            (
                                "troponin_max_url_ratio",
                                "troponin_count_standard",
                                "acute_troponin_change_proxy_value",
                                "potassium_nearest",
                                "potassium_delta_hours",
                                "potassium_count_6h",
                                "magnesium_nearest",
                                "total_calcium_nearest",
                                "ionized_calcium_nearest",
                                "creatinine_nearest",
                                "hemoglobin_nearest",
                                "lactate_nearest",
                            ),
                        ),
                        "echo_context": _compact_context_json(
                            item,
                            (
                                "nearest_echo_measurement_id",
                                "nearest_echo_delta_hours",
                                "preferred_lvef",
                                "preferred_lvef_method",
                                "lv_mass_index",
                                "la_volume_index",
                                "rv_basal_diameter_cm",
                                "rv_function_category",
                                "tapse_cm",
                                "derived_rvsp_or_pasp_mmhg",
                                "rvsp_or_pasp_method",
                            ),
                        ),
                        "procedure_context": _compact_context_json(
                            item,
                            (
                                "proc_pci",
                                "proc_coronary_angiography",
                                "proc_cabg",
                                "proc_valve_procedure",
                                "pci_day_delta",
                                "cath_day_delta",
                            ),
                        ),
                        "medication_context": _compact_context_json(
                            item,
                            (
                                "med_qt_prolonging_drug",
                                "med_antiarrhythmic_class_i_iii",
                                "med_digoxin",
                                "med_vasopressor",
                                "med_antiplatelet",
                                "med_heparin",
                                "med_nitrate",
                                "med_electrolyte_replacement",
                                "medication_assessed",
                            ),
                        ),
                        "hadm_id": item.hadm_id,
                        "ed_stay_id": item.ed_stay_id,
                        "echo_measurement_id": item.nearest_echo_measurement_id,
                        "echo_delta_hours": item.nearest_echo_delta_hours,
                        "machine_report": item.machine_report,
                        "machine_report_state": report_states[local_index],
                        "classical_criterion": classical_id,
                        "classical_positive": classical_prediction[local_index],
                        "vcg_feature_id": feature_id,
                        "vcg_feature_analysis_role": feature_analysis_role(
                            feature_id
                        ),
                        "vcg_marker_inference_status": marker_inference_status,
                        "vcg_value": values[local_index],
                        "vcg_direction": model.direction,
                        "threshold_aligned": threshold,
                        "threshold_original": model.direction * threshold,
                        "threshold_distance": aligned[local_index] - threshold,
                        "signal_quality_score": item.get(
                            "signal_quality_score",
                            item.master_signal_quality_score,
                        ),
                        "signal_quality_tier": item.get(
                            "signal_quality_tier",
                            None,
                        ),
                        "fiducial_confidence": item.get(
                            "fiducial_confidence",
                            None,
                        ),
                        "waveform_path": item.waveform_path,
                        "discordance_category": category,
                        "review_status": "unreviewed",
                        "locked_test_review_post_hoc": locked_consumed,
                        "dua_notice": MIMIC_DUA_NOTICE,
                    }
                )
        diagnostics.append(
            {
                "endpoint_id": endpoint_id,
                "feature_id": feature_id,
                "feature_analysis_role": feature_analysis_role(feature_id),
                "inference_status": marker_inference_status,
                "status": "complete",
                "reason": "Review candidate scan completed.",
                "error_type": None,
                "error_message": None,
                "assessable_rows": int(evaluable.sum()),
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return (
            _empty_review_frame(),
            _empty_discordance_frame(),
            pd.DataFrame(diagnostics)
            if diagnostics
            else _empty_review_diagnostics_frame(),
        )
    confidence_rank = candidates["label_confidence"].map(
        {"high": 3, "medium": 2, "low": 1, "none": 0}
    ).fillna(0)
    quality = pd.to_numeric(
        candidates["signal_quality_score"],
        errors="coerce",
    ).fillna(0)
    candidates["review_priority"] = (
        (3 if locked_consumed else 0)
        + confidence_rank
        + quality
        + np.minimum(np.abs(candidates["threshold_distance"]), 5.0) / 5.0
    )
    candidates = candidates.sort_values(
        ["review_priority", "endpoint_id", "study_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    candidates = annotate_result_frame(candidates)
    discordance = (
        candidates.groupby(
            ["endpoint_id", "split", "discordance_category"],
            dropna=False,
        )
        .agg(
            candidate_count=("study_id", "size"),
            subject_count=("subject_id", "nunique"),
            median_threshold_distance=("threshold_distance", "median"),
            median_signal_quality=("signal_quality_score", "median"),
        )
        .reset_index()
    )
    discordance["interpretation"] = (
        "Review queue only; discordance is not evidence of diagnostic error."
    )
    discordance = annotate_result_frame(discordance)
    return candidates, discordance, pd.DataFrame(diagnostics)


def _run_manifest_frame(
    config: MimicProjectConfig,
    run_id: str,
    tables: dict[str, pd.DataFrame],
    source_paths: dict[str, Optional[Path]],
) -> pd.DataFrame:
    lock_path = config.runs_root / run_id / "test_lock.json"
    lock = _load_test_lock(lock_path) if lock_path.exists() else None
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "project_mode": "MIMIC clinical validation",
                "research_use_only": True,
                "individual_diagnosis": False,
                "project_config_json": json.dumps(
                    config.to_dict(),
                    sort_keys=True,
                ),
                "code_hash": source_tree_hash(
                    Path(__file__).resolve().parents[1]
                ),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "source_release_versions_json": json.dumps(
                    {
                        "mimic_iv": config.mimic_iv_release,
                        "mimic_iv_ed": config.mimic_ed_release,
                        "mimic_iv_ecg": config.mimic_ecg_release,
                        "mimic_iv_echo": config.mimic_echo_release,
                    },
                    sort_keys=True,
                ),
                "package_versions_json": json.dumps(
                    _package_versions(),
                    sort_keys=True,
                ),
                "test_lock_status": (
                    "consumed"
                    if lock is not None and lock.is_consumed
                    else "frozen"
                    if lock is not None and lock.is_frozen
                    else "not_frozen"
                ),
                "freeze_hash": lock.freeze_hash if lock is not None else None,
                "source_fingerprints_json": json.dumps(
                    {
                        name: (
                            file_fingerprint(path)
                            if path is not None and path.exists()
                            else None
                        )
                        for name, path in source_paths.items()
                    },
                    sort_keys=True,
                ),
                "report_row_counts_json": json.dumps(
                    {name: len(frame) for name, frame in tables.items()},
                    sort_keys=True,
                ),
                "dua_notice": MIMIC_DUA_NOTICE,
            }
        ]
    )


def _package_versions() -> dict[str, str]:
    packages = (
        "duckdb",
        "neurokit2",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "wfdb",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _best_classical_id(
    table: pd.DataFrame,
    endpoint_id: str,
    split: str,
) -> Optional[str]:
    if table.empty or "endpoint_id" not in table:
        return None
    candidates = table.loc[
        (table["endpoint_id"] == endpoint_id)
        & (table["evaluation_split"] == split)
        & table["roc_auc"].notna()
    ]
    if candidates.empty:
        return None
    return str(
        candidates.sort_values("roc_auc", ascending=False).iloc[0][
            "baseline_id"
        ]
    )


def _default_feature_glob(
    config: MimicProjectConfig,
    evaluation_split: str,
    *,
    run_id: str,
) -> str:
    if evaluation_split != "locked_test":
        return str(
            config.features_root
            / "primary_core_sentinel"
            / "development"
            / f"split={evaluation_split}"
            / "part-*.parquet"
        )
    freeze_path = config.runs_root / run_id / "freeze_manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(
            "Locked-test review requires the run freeze manifest."
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_hash = str(freeze.get("freeze_hash", ""))
    development_root_text = str(
        freeze.get("payload", {})
        .get("feature_integrity", {})
        .get("root", "")
    )
    if not freeze_hash or not development_root_text:
        raise RuntimeError("Freeze manifest lacks feature provenance")
    development_root = Path(development_root_text)
    return str(
        development_root.parent
        / f"locked-test-{freeze_hash[:12]}"
        / "split=locked_test"
        / "part-*.parquet"
    )


def _empty_review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "clinical_target_de",
            "clinical_category_de",
            "subject_id",
            "study_id",
            "split",
            "endpoint_id",
            "discordance_category",
            "review_priority",
            "review_status",
            "dua_notice",
        ]
    )


def _empty_discordance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "clinical_target_de",
            "clinical_category_de",
            "endpoint_id",
            "split",
            "discordance_category",
            "candidate_count",
            "subject_count",
            "interpretation",
        ]
    )


def _empty_review_result(
    status: str,
    reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _empty_review_frame(),
        _empty_discordance_frame(),
        pd.DataFrame(
            [
                {
                    "endpoint_id": None,
                    "feature_id": None,
                    "status": status,
                    "reason": reason,
                    "error_type": None,
                    "error_message": None,
                }
            ]
        ),
    )


def _empty_review_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "endpoint_id",
            "feature_id",
            "status",
            "reason",
            "error_type",
            "error_message",
            "assessable_rows",
        ]
    )


def _compact_context_json(row: pd.Series, fields: tuple[str, ...]) -> str:
    values = {}
    for field in fields:
        value = row.get(field)
        if value is None or value is pd.NA:
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if pd.isna(value):
            continue
        values[field] = value
    return json.dumps(
        values,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) > 0:
        return frame
    return pd.DataFrame({"status": pd.Series(dtype="string")})


def _atomic_write_csv(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    try:
        frame.to_csv(temporary_name, index=False)
        atomic_replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


# The workbook is a convenience export beside the canonical Parquet; openpyxl
# holds every cell of every sheet in memory until the file is written. With
# the old limit of one million rows per sheet the 10.000-record preview of
# 2026-09-19 -- screening results 308,000 rows, operating points and the two
# explorer tables of the same order -- reached 16.6 GiB and was stopped by
# the memory guard. The row count of those tables does not shrink in the full
# run (it is markers x endpoints, not records), so the cap is what makes the
# stage runnable at all. Truncated sheets are listed on their own sheet with
# the full row count; the complete tables are the Parquet and CSV files.
EXCEL_SHEET_ROW_LIMIT = 20_000


def _atomic_write_excel(
    tables: dict[str, pd.DataFrame],
    target: Path,
    *,
    row_limit: int = EXCEL_SHEET_ROW_LIMIT,
) -> None:
    temporary = target.parent / f".{target.stem}.{os.getpid()}.tmp.xlsx"
    truncated: list[dict[str, object]] = []
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            for name, frame in tables.items():
                if len(frame) > row_limit:
                    truncated.append(
                        {
                            "table": name,
                            "rows_in_workbook": int(row_limit),
                            "rows_total": int(len(frame)),
                            "complete_table": f"{name}.parquet / {name}.csv",
                        }
                    )
                export = frame.head(row_limit).copy()
                for column in export.select_dtypes(
                    include=["datetimetz"]
                ).columns:
                    export[column] = export[column].dt.tz_localize(None)
                export.to_excel(
                    writer,
                    sheet_name=_excel_sheet_name(name),
                    index=False,
                )
                del export
            if truncated:
                pd.DataFrame(truncated).to_excel(
                    writer, sheet_name="truncated_tables", index=False
                )
        atomic_replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _excel_sheet_name(name: str) -> str:
    cleaned = "".join(
        character if character not in r"[]:*?/\\" else "_"
        for character in name
    )
    return cleaned[:31] or "report"


def _render_html_report(
    run_id: str,
    tables: dict[str, pd.DataFrame],
) -> str:
    sections = []
    preferred = (
        "run_manifest",
        "acceptance_audit",
        "data_inventory",
        "ecg_master_index_summary",
        "label_summary",
        "split_summary",
        "label_window_sensitivity_summary",
        "methodology_registry",
        "screening_results",
        "classical_baseline_results",
        "vcg_vs_classical_comparison",
        "combination_results",
        "adjusted_model_results",
        "decision_curve_results",
        "overfitting_stability_report",
        "ptb_transfer_matrix",
        "all_ecg_clustered_results",
        "discordance_report",
        "review_candidates",
        "feature_failure_report",
        "final_test_report",
        "final_test_adjusted_models",
        "final_test_decision_curves",
    )
    for name in preferred:
        frame = tables.get(name)
        if frame is None:
            continue
        preview = frame.head(200)
        sections.append(
            f"<section><h2>{html.escape(name.replace('_', ' ').title())}</h2>"
            f"<p>{len(frame):,} rows; preview limited to {len(preview):,}.</p>"
            f"{preview.to_html(index=False, border=0, escape=True)}</section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(run_id)} - MIMIC Clinical Validation</title>
<style>
body {{ margin: 0; font: 14px/1.45 Arial, sans-serif; color: #1d242b; background: #f5f7f8; }}
header {{ padding: 24px 32px; background: #18332d; color: white; }}
main {{ padding: 20px 32px 48px; }}
section {{ margin: 0 0 28px; }}
h1 {{ margin: 0 0 6px; font-size: 26px; letter-spacing: 0; }}
h2 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }}
p {{ margin: 4px 0 12px; }}
.notice {{ color: #f3d98d; max-width: 1000px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ border: 1px solid #d8dee2; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #e8eeec; position: sticky; top: 0; }}
tr:nth-child(even) td {{ background: #fafbfb; }}
</style>
</head>
<body>
<header>
<h1>{html.escape(run_id)}</h1>
<p>MIMIC clinical reference comparison, research use only.</p>
<p class="notice">{html.escape(MIMIC_DUA_NOTICE)}</p>
</header>
<main>{''.join(sections)}</main>
</body>
</html>
"""
