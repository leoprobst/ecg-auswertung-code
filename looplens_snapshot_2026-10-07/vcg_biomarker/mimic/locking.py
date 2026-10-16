from __future__ import annotations

import glob
import importlib.metadata
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .acceptance import refresh_pre_freeze_acceptance_audit
from .adjusted_models import (
    FrozenLogisticModel,
    _endpoint_design_frame,
    _repeated_marker_design,
    decision_curve,
    evaluate_logistic_model,
    predict_logistic_model,
)
from .baseline_registry import BaselineRegistry
from .config import MimicProjectConfig
from .clinical_utility import ScopeFreezeAttestation
from .event_equivalence import (
    verify_clinical_event_evidence_provenance,
)
from .feature_extraction import verify_feature_artifacts
from .label_registry import default_label_definitions
from .labels import (
    development_label_attestation_path,
    verify_development_label_attestation,
)
from .methodology_registry import MethodologyRegistry
from .models import (
    FrozenScoreSpec,
    ScoreReleaseStage,
    TestLock,
    ValidationSpec,
)
from .panels import freeze_panel_registry
from .ptb_transfer import load_ptb_frozen_transfer_scores
from .snapshots import (
    load_artifact_snapshot,
    verify_artifact_snapshot,
)
from .statistics import (
    MarkerModel,
    apply_fdr_by_family,
    apply_frozen_score,
    classification_metrics,
    enforce_inference_policy,
    evaluate_marker,
    inference_status,
    marker_bootstrap_intervals,
)
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    feature_parquet_clause,
    operational_config_hash,
    scientific_config_hash,
    file_sha256,
    source_tree_hash,
)
from .validation import (
    LOCKED_TEST_MARKER_SHORTLIST_VERSION,
    PRIMARY_ADJUSTED_MARKERS,
    _binary_baseline_intervals,
    _load_endpoint_frame,
    _numeric_feature,
    _quoted_identifier,
    _report_comparator_for_endpoint,
    _event_evidence_provenance_summary,
    verify_validation_input_payload_files,
)
from .validation_provenance import (
    locked_evaluation_implementation_hash,
    verify_validation_code_provenance,
)


FREEZE_VERSION = "2.7.0"
LOCKED_EVALUATION_VERSION = "2.3.0"
CLINICAL_UTILITY_SCOPE_ATTESTATION = (
    "clinical_utility_scope_attestation.json"
)

FROZEN_DEVELOPMENT_TABLES = (
    "screening_results.parquet",
    "classical_baseline_results.parquet",
    "machine_report_baseline_results.parquet",
    "vcg_vs_classical_comparison.parquet",
    "descriptive_validation_ranking.parquet",
    "baseline_registry.parquet",
    "methodology_registry.parquet",
    "combination_results.parquet",
    "overfitting_stability_report.parquet",
    "validation_failure_report.parquet",
    "validation_label_counts.parquet",
    "adjusted_model_results.parquet",
    "decision_curve_results.parquet",
    "all_ecg_clustered_results.parquet",
    "performance_curve_points.parquet",
    "score_distribution_bins.parquet",
    "operating_point_results.parquet",
    "ptb_transfer_matrix.parquet",
    "criterion_candidates.parquet",
    "criterion_thresholds.parquet",
    "cross_fit_confirmations.parquet",
    "cross_fit_results.parquet",
    "cross_fit_plan.parquet",
    "verification_bias_bounds.parquet",
)


@dataclass(frozen=True)
class LockedEvaluationOptions:
    run_id: str
    feature_glob: Optional[str] = None


def _locked_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    if isinstance(value, (int, np.integer)):
        return int(value) == 1
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    return False


def _locked_finite_or_none(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _load_and_validate_locked_test_marker_shortlist(
    run_root: Path,
    *,
    expected_run_id: str,
    require_nonempty: bool,
) -> dict:
    """Validate the pre-Test marker x endpoint contract without Test access."""
    shortlist_path = run_root / "locked_test_marker_shortlist.json"
    marker_models_path = run_root / "trained_marker_models.json"
    cross_fit_models_path = run_root / "cross_fit_marker_models.json"
    criteria_path = run_root / "criterion_candidates.parquet"
    for path in (
        shortlist_path,
        marker_models_path,
        cross_fit_models_path,
        criteria_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Locked-test shortlist input is missing: {path}")
    payload = json.loads(shortlist_path.read_text(encoding="utf-8"))
    if payload.get("shortlist_version") != LOCKED_TEST_MARKER_SHORTLIST_VERSION:
        raise RuntimeError(
            "Freeze refused: locked-test marker shortlist version is unsupported."
        )
    if str(payload.get("development_run_id", "")) != expected_run_id:
        raise RuntimeError(
            "Freeze refused: locked-test marker shortlist belongs to another run."
        )
    if payload.get("locked_test_read") is not False:
        raise RuntimeError(
            "Freeze refused: shortlist does not attest that Test remained unread."
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Locked-test marker shortlist entries must be a list")
    if int(payload.get("entry_count", -1)) != len(entries):
        raise RuntimeError("Locked-test marker shortlist entry count is inconsistent")
    expected_hash = content_hash(
        {
            "shortlist_version": LOCKED_TEST_MARKER_SHORTLIST_VERSION,
            "entries": entries,
        }
    )
    if str(payload.get("shortlist_hash", "")) != expected_hash:
        raise RuntimeError("Locked-test marker shortlist hash is inconsistent")
    if require_nonempty and not entries:
        raise RuntimeError(
            "Freeze refused: no marker x endpoint hypothesis passed the "
            "pre-Test replication, adjustment, multiplicity, and assessability "
            "gates. Test remains sealed."
        )

    marker_payload = json.loads(marker_models_path.read_text(encoding="utf-8"))
    model_lookup: dict[tuple[str, str, str], dict] = {}
    for raw_model in marker_payload.get("models", []):
        model = dict(raw_model)
        key = (
            str(model.get("endpoint_id", "")),
            str(model.get("panel_id", "")),
            str(model.get("feature_id", "")),
        )
        if key in model_lookup:
            raise RuntimeError(f"Duplicate train-fitted marker model: {key}")
        model_lookup[key] = model

    criteria = pd.read_parquet(criteria_path)
    required_columns = {
        "criterion_id",
        "endpoint_id",
        "panel_id",
        "feature_id",
        "stage",
        "locked_test_eligible",
        "adjustment_confirmed",
        "project_wide_multiplicity_confirmed",
        "cross_fit_confirmed",
        "cross_fit_evaluated_in_both_folds",
        "cross_fit_direction_consistent",
        "verification_bias_gate_passed",
        "train_assessability_gate_passed",
        "validation_assessability_gate_passed",
        "train_missingness_sensitivity_gate_passed",
        "validation_missingness_sensitivity_gate_passed",
    }
    missing_columns = sorted(required_columns - set(criteria.columns))
    if missing_columns:
        raise RuntimeError(
            "Freeze refused: criterion catalog lacks locked-test evidence "
            f"columns: {missing_columns}"
        )
    criterion_lookup = {
        str(row.get("criterion_id", "")): row
        for row in criteria.to_dict(orient="records")
    }
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Locked-test marker shortlist entry must be an object")
        if str(entry.get("candidate_kind", "")) != "single_marker":
            raise RuntimeError(
                "Locked evaluation currently accepts only explicit single-marker "
                "hypotheses; score candidates need their own frozen contract."
            )
        key = (
            str(entry.get("endpoint_id", "")),
            str(entry.get("panel_id", "")),
            str(entry.get("feature_id", "")),
        )
        if not all(key) or key in seen:
            raise RuntimeError(f"Invalid or duplicate locked-test shortlist key: {key}")
        seen.add(key)
        model = model_lookup.get(key)
        if model is None or content_hash(model) != entry.get("marker_model_hash"):
            raise RuntimeError(
                "Freeze refused: shortlist marker model is missing or changed for "
                f"{key}."
            )
        criterion = criterion_lookup.get(str(entry.get("criterion_id", "")))
        if criterion is None:
            raise RuntimeError(f"Shortlisted criterion is absent from catalog: {key}")
        criterion_key = (
            str(criterion.get("endpoint_id", "")),
            str(criterion.get("panel_id", "")),
            str(criterion.get("feature_id", "")),
        )
        gates = (
            str(criterion.get("stage", "")) == "validation_replicated",
            _locked_bool(criterion.get("locked_test_eligible", False)),
            _locked_bool(criterion.get("adjustment_confirmed", False)),
            _locked_bool(
                criterion.get("project_wide_multiplicity_confirmed", False)
            ),
            _locked_bool(criterion.get("cross_fit_confirmed", False)),
            _locked_bool(
                criterion.get("cross_fit_evaluated_in_both_folds", False)
            ),
            _locked_bool(
                criterion.get("cross_fit_direction_consistent", False)
            ),
            _locked_bool(
                criterion.get("verification_bias_gate_passed", False)
            ),
            _locked_bool(
                criterion.get("train_assessability_gate_passed", False)
            ),
            _locked_bool(
                criterion.get("validation_assessability_gate_passed", False)
            ),
            _locked_bool(
                criterion.get(
                    "train_missingness_sensitivity_gate_passed", False
                )
            ),
            _locked_bool(
                criterion.get(
                    "validation_missingness_sensitivity_gate_passed", False
                )
            ),
        )
        if criterion_key != key or not all(gates):
            raise RuntimeError(
                "Freeze refused: shortlisted criterion no longer satisfies its "
                f"pre-Test evidence gates: {key}."
            )
        criterion_evidence = {
            "criterion_id": str(criterion.get("criterion_id", "")),
            "train_q_value": _locked_finite_or_none(
                criterion.get("train_q_value")
            ),
            "validation_q_value": _locked_finite_or_none(
                criterion.get("q_value")
            ),
            "train_global_q_value": _locked_finite_or_none(
                criterion.get("train_global_q_value")
            ),
            "validation_global_q_value": _locked_finite_or_none(
                criterion.get("validation_global_q_value")
            ),
            "train_hierarchical_q_value": _locked_finite_or_none(
                criterion.get("train_hierarchical_q_value")
            ),
            "validation_hierarchical_q_value": _locked_finite_or_none(
                criterion.get("validation_hierarchical_q_value")
            ),
            "adjustment_model_id": criterion.get("adjustment_model_id"),
            "adjustment_incremental_auc_ci_low": _locked_finite_or_none(
                criterion.get("adjustment_incremental_auc_ci_low")
            ),
            "train_test_yield": _locked_finite_or_none(
                criterion.get("train_test_yield")
            ),
            "validation_test_yield": _locked_finite_or_none(
                criterion.get("validation_test_yield")
            ),
            "train_missingness_auc_lower_bound": _locked_finite_or_none(
                criterion.get("train_missingness_auc_lower_bound")
            ),
            "validation_missingness_auc_lower_bound": _locked_finite_or_none(
                criterion.get("validation_missingness_auc_lower_bound")
            ),
        }
        if content_hash(criterion_evidence) != entry.get("criterion_evidence_hash"):
            raise RuntimeError(
                "Freeze refused: shortlist evidence differs from criterion "
                f"catalog for {key}."
            )
    if int(payload.get("endpoint_count", -1)) != len(
        {str(entry["endpoint_id"]) for entry in entries}
    ):
        raise RuntimeError("Locked-test marker shortlist endpoint count is inconsistent")
    return payload


def _models_for_locked_shortlist(
    shortlist: dict,
    marker_rows: list[dict],
) -> dict[str, list[tuple[str, MarkerModel]]]:
    model_lookup = {
        (
            str(row.get("endpoint_id", "")),
            str(row.get("panel_id", "")),
            str(row.get("feature_id", "")),
        ): row
        for row in marker_rows
    }
    output: dict[str, list[tuple[str, MarkerModel]]] = {}
    for entry in shortlist.get("entries", []):
        key = (
            str(entry["endpoint_id"]),
            str(entry["panel_id"]),
            str(entry["feature_id"]),
        )
        row = model_lookup.get(key)
        if row is None or content_hash(row) != entry.get("marker_model_hash"):
            raise RuntimeError(f"Frozen shortlist model mismatch: {key}")
        output.setdefault(key[0], []).append((key[1], MarkerModel.from_dict(row)))
    return output


def _clinical_utility_scope_freeze(run_root: Path) -> dict:
    """Validate an optional, pre-outcome clinical-use scope attestation.

    A technical internal replication may be frozen without such an
    attestation.  That absence is itself sealed into the freeze payload and
    permanently prevents this run from being represented as having a
    prespecified clinical-utility scope.
    """

    path = run_root / CLINICAL_UTILITY_SCOPE_ATTESTATION
    if not path.is_file():
        return {
            "status": "not_prespecified",
            "clinical_utility_claim_scope_prespecified": False,
            "reason": (
                "No pre-outcome intended-use/comparator/threshold attestation "
                "was present at freeze. Internal replication remains allowed; "
                "clinical-utility claims do not."
            ),
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("attestation must be a JSON object")
        attestation = ScopeFreezeAttestation(**raw)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Freeze refused: clinical-utility scope attestation is invalid."
        ) from exc
    if not (
        attestation.frozen_before_evaluation
        and attestation.locked_test_data_attested_unread
        and attestation.locked_test_outcomes_attested_unread
    ):
        raise RuntimeError(
            "Freeze refused: clinical-utility scope was not attested before "
            "evaluation with locked data and outcomes unread."
        )
    return {
        "status": "prespecified_and_bound",
        "clinical_utility_claim_scope_prespecified": True,
        "attestation_id": attestation.attestation_id,
        "candidate_id": attestation.candidate_id,
        "use_id": attestation.use_id,
        "intended_use_sha256": attestation.intended_use_sha256,
        "comparator_sha256": attestation.comparator_sha256,
        "thresholds_sha256": attestation.thresholds_sha256,
        "evidence_registry_sha256": attestation.evidence_registry_sha256,
        "attestation_sha256": attestation.attestation_sha256,
    }


def freeze_validation_run(
    config: MimicProjectConfig,
    run_id: str,
) -> TestLock:
    run_root = config.runs_root / run_id
    validation_manifest_path = run_root / "validate.manifest.json"
    marker_models_path = run_root / "trained_marker_models.json"
    cross_fit_models_path = run_root / "cross_fit_marker_models.json"
    score_specs_path = run_root / "frozen_score_candidates.json"
    score_releases_path = run_root / "score_releases.json"
    adjusted_specs_path = run_root / "adjusted_model_specs.json"
    all_ecg_specs_path = run_root / "all_ecg_clustered_models.json"
    validation_spec_path = run_root / "validation_spec.json"
    provenance_path = run_root / "development_provenance.json"
    acceptance_path = run_root / "acceptance_pre_freeze_audit.json"
    locked_shortlist_path = run_root / "locked_test_marker_shortlist.json"
    required = (
        validation_manifest_path,
        marker_models_path,
        cross_fit_models_path,
        score_specs_path,
        score_releases_path,
        adjusted_specs_path,
        all_ecg_specs_path,
        validation_spec_path,
        provenance_path,
        acceptance_path,
        locked_shortlist_path,
        *(run_root / name for name in FROZEN_DEVELOPMENT_TABLES),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Development validation is incomplete; freeze refused: {missing}"
        )

    _load_and_validate_locked_test_marker_shortlist(
        run_root,
        expected_run_id=run_id,
        require_nonempty=True,
    )

    refresh_pre_freeze_acceptance_audit(
        config,
        run_id=run_id,
    )
    panel_registry_path = config.features_root / "feature_panels.json"
    if not panel_registry_path.exists():
        freeze_panel_registry(panel_registry_path)
    payload = _freeze_payload(
        config,
        run_id,
        panel_registry_path=panel_registry_path,
    )
    freeze_hash = content_hash(payload)
    freeze_manifest_path = run_root / "freeze_manifest.json"
    lock_path = run_root / "test_lock.json"
    if lock_path.exists():
        existing = _load_test_lock(lock_path)
        if existing.freeze_hash != freeze_hash:
            raise RuntimeError(
                "A different immutable freeze already exists. Explicitly invalidate "
                "that test lock with a documented reason before creating a new run."
            )
        _update_score_release_stage(
            score_releases_path,
            ScoreReleaseStage.FROZEN,
            freeze_hash=freeze_hash,
        )
        return existing

    freeze_manifest = {
        "freeze_version": FREEZE_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "freeze_hash": freeze_hash,
        "payload": payload,
        "platform_code_hash_at_freeze": source_tree_hash(
            Path(__file__).resolve().parents[1]
        ),
        "locked_test_status": "sealed_unread",
        "scientific_scope": (
            "Internal MIMIC locked test; not independent multicenter validation"
        ),
    }
    lock = TestLock(run_id=run_id, freeze_hash=freeze_hash)
    atomic_write_json(freeze_manifest_path, freeze_manifest)
    atomic_write_json(lock_path, lock.to_dict())
    _update_score_release_stage(
        score_releases_path,
        ScoreReleaseStage.FROZEN,
        freeze_hash=freeze_hash,
    )
    return lock


def evaluate_locked_test(
    config: MimicProjectConfig,
    options: LockedEvaluationOptions,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    run_root = config.runs_root / options.run_id
    lock_path = run_root / "test_lock.json"
    freeze_manifest_path = run_root / "freeze_manifest.json"
    if not lock_path.exists() or not freeze_manifest_path.exists():
        raise FileNotFoundError(
            "Locked test evaluation requires an immutable freeze. Run freeze first."
        )
    lock = _load_test_lock(lock_path)
    frozen = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    panel_registry_path = config.features_root / "feature_panels.json"
    current_payload = _freeze_payload(
        config,
        options.run_id,
        panel_registry_path=panel_registry_path,
    )
    current_hash = content_hash(current_payload)
    if current_hash != frozen.get("freeze_hash"):
        raise RuntimeError(
            "Locked test refused: the current freeze payload differs from the "
            "sealed manifest."
        )
    lock.assert_evaluable(current_hash)
    shortlist = _load_and_validate_locked_test_marker_shortlist(
        run_root,
        expected_run_id=options.run_id,
        require_nonempty=True,
    )

    frozen_feature_integrity = frozen["payload"]["feature_integrity"]
    development_feature_root = Path(frozen_feature_integrity["root"])
    feature_glob = options.feature_glob or str(
        development_feature_root.parent
        / f"locked-test-{current_hash[:12]}"
        / "split=locked_test"
        / "part-*.parquet"
    )
    feature_paths = sorted(glob.glob(feature_glob))
    if not feature_paths:
        raise FileNotFoundError(
            "Locked-test feature files are missing. Blind feature extraction is "
            f"allowed, but expected files matching {feature_glob!r}."
        )
    current_feature_integrity = verify_feature_artifacts(
        feature_paths,
    )
    if current_feature_integrity.get(
        "source_code_hash"
    ) != frozen_feature_integrity.get("source_code_hash"):
        raise RuntimeError(
            "Locked-test features were produced by a different implementation "
            "than the frozen development features."
        )
    if current_feature_integrity.get("panel_hash") != frozen_feature_integrity.get(
        "panel_hash"
    ):
        raise RuntimeError(
            "Locked-test features use a different feature panel than the "
            "frozen development analysis."
        )
    _assert_locked_test_feature_scope(
        current_feature_integrity,
        expected_freeze_hash=current_hash,
    )

    validation_spec = _load_validation_spec(
        run_root / "validation_spec.json"
    )
    marker_payload = json.loads(
        (run_root / "trained_marker_models.json").read_text(encoding="utf-8")
    )
    adjusted_payload = json.loads(
        (run_root / "adjusted_model_specs.json").read_text(encoding="utf-8")
    )
    all_ecg_payload = json.loads(
        (run_root / "all_ecg_clustered_models.json").read_text(
            encoding="utf-8"
        )
    )
    marker_rows = marker_payload.get("models", [])
    models_by_endpoint = _models_for_locked_shortlist(shortlist, marker_rows)
    scores_by_endpoint: dict[str, list[dict]] = {}
    shortlisted_features = {
        str(entry["endpoint_id"]): {
            str(candidate["feature_id"])
            for candidate in shortlist["entries"]
            if str(candidate["endpoint_id"]) == str(entry["endpoint_id"])
        }
        for entry in shortlist["entries"]
    }
    adjusted_by_endpoint: dict[str, list[dict]] = {}
    for row in adjusted_payload.get("models", []):
        endpoint_id = str(row["model"]["endpoint_id"])
        if str(row.get("vcg_feature_id", "")) not in shortlisted_features.get(
            endpoint_id, set()
        ):
            continue
        adjusted_by_endpoint.setdefault(endpoint_id, []).append(row)
    all_ecg_by_endpoint: dict[str, list[dict]] = {}
    for row in all_ecg_payload.get("models", []):
        endpoint_id = str(row["model"]["endpoint_id"])
        if str(row.get("vcg_feature_id", "")) not in shortlisted_features.get(
            endpoint_id, set()
        ):
            continue
        all_ecg_by_endpoint.setdefault(endpoint_id, []).append(row)

    cohort_path = (
        config.labels_root
        / "sealed"
        / "locked_endpoint_index_cohorts.parquet"
    )
    master_path = config.master_root / "ecg_master_index.parquet"
    reports_path = config.evidence_root / "machine_report_labels.parquet"
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    results: list[dict] = []
    combinations: list[dict] = []
    classical: list[dict] = []
    machine_reports: list[dict] = []
    label_counts: list[dict] = []
    failures: list[dict] = []
    adjusted_results: list[dict] = []
    decision_curve_results: list[dict] = []
    window_sensitivity_results: list[dict] = []
    all_ecg_results: list[dict] = []
    shortlist_endpoint_ids = set(shortlisted_features)
    unknown_endpoints = shortlist_endpoint_ids - set(validation_spec.endpoint_ids)
    if unknown_endpoints:
        raise RuntimeError(
            "Locked-test shortlist contains endpoints outside validation_spec: "
            f"{sorted(unknown_endpoints)}"
        )
    endpoints = tuple(
        endpoint_id
        for endpoint_id in validation_spec.endpoint_ids
        if endpoint_id in shortlist_endpoint_ids
    )
    lock.begin_evaluation()
    atomic_write_json(lock_path, lock.to_dict())
    for endpoint_index, endpoint_id in enumerate(endpoints, start=1):
        if progress:
            progress(
                f"Locked Test {endpoint_index:,}/{len(endpoints):,}: "
                f"{endpoint_id}"
            )
        report_label_id = _report_comparator_for_endpoint(endpoint_id)
        frame = _load_endpoint_frame(
            catalog,
            cohort_path=cohort_path,
            master_path=master_path,
            reports_path=reports_path if reports_path.exists() else None,
            context_path=(
                config.evidence_root / "clinical_context_features.parquet"
            ),
            feature_glob=feature_glob,
            endpoint_id=endpoint_id,
            report_label_id=report_label_id,
            splits=("locked_test",),
        )
        if frame.empty:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "stage": "locked_test",
                    "reason": "no assessable locked-test rows",
                }
            )
            continue
        frame["label"] = (frame["label_state"] == "positive").astype(np.int8)
        label_counts.append(
            {
                "endpoint_id": endpoint_id,
                "split": "locked_test",
                "n_total": len(frame),
                "n_positive": int(frame["label"].sum()),
                "n_negative": int((frame["label"] == 0).sum()),
                "subject_count": int(frame["subject_id"].nunique()),
            }
        )
        for model_index, (panel_id, model) in enumerate(
            models_by_endpoint.get(endpoint_id, [])
        ):
            values = _numeric_feature(frame, model.feature_id)
            record = _locked_marker_record(
                endpoint_id,
                panel_id,
                model.feature_id,
                frame,
                values,
                model,
                validation_spec,
                seed_offset=model_index,
            )
            results.append(record)
            if panel_id == "classical":
                fixed = _locked_classical_record(
                    endpoint_id,
                    model.feature_id,
                    frame,
                    validation_spec,
                )
                if fixed is not None:
                    classical.append(fixed)
        window_feature_id = PRIMARY_ADJUSTED_MARKERS.get(endpoint_id)
        window_model = next(
            (
                model
                for _, model in models_by_endpoint.get(endpoint_id, [])
                if model.feature_id == window_feature_id
            ),
            None,
        )
        locked_window_path = (
            config.labels_root
            / "sealed"
            / "locked_endpoint_window_sensitivity_cohorts.parquet"
        )
        if window_model is not None and locked_window_path.exists():
            window_sensitivity_results.extend(
                _locked_window_sensitivity_records(
                    catalog,
                    cohort_path=locked_window_path,
                    feature_glob=feature_glob,
                    endpoint_id=endpoint_id,
                    feature_id=str(window_feature_id),
                    model=window_model,
                    specification=validation_spec,
                )
            )
        for score_index, payload in enumerate(
            scores_by_endpoint.get(endpoint_id, [])
        ):
            try:
                specification = _load_frozen_score(payload["score_spec"])
                score_model = MarkerModel.from_dict(payload["marker_model"])
                values = apply_frozen_score(frame, specification)
                record = _locked_marker_record(
                    endpoint_id,
                    "predefined_score",
                    specification.score_id,
                    frame,
                    values,
                    score_model,
                    validation_spec,
                    seed_offset=100_000 + score_index,
                )
                record.update(
                    {
                        "analysis_type": payload.get(
                            "analysis_type",
                            "predefined_combination",
                        ),
                        "components": "|".join(specification.components),
                        "component_count": len(specification.components),
                        "component_directions": "|".join(
                            str(value) for value in specification.directions
                        ),
                        "normalization": payload.get(
                            "normalization",
                            "frozen_train_zscore_complete_case",
                        ),
                        "aggregation": specification.aggregation,
                        "exploratory_score": specification.exploratory_score,
                        "origin": specification.origin,
                        "threshold_source": payload.get(
                            "threshold_source",
                            "mimic_train",
                        ),
                        "ptb_endpoint_id": payload.get("ptb_endpoint_id"),
                        "mimic_refit_performed": payload.get(
                            "mimic_refit_performed"
                        ),
                    }
                )
                combinations.append(record)
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(
                    {
                        "endpoint_id": endpoint_id,
                        "stage": "locked_score",
                        "reason": str(exc),
                    }
                )
        for repeated_payload_row in all_ecg_by_endpoint.get(endpoint_id, []):
            try:
                repeated_model = FrozenLogisticModel.from_dict(
                    repeated_payload_row["model"]
                )
                repeated_frame = _load_locked_all_ecg_frame(
                    catalog,
                    label_path=(
                        config.labels_root
                        / "sealed"
                        / "locked_clinical_label_evidence.parquet"
                    ),
                    feature_glob=feature_glob,
                    endpoint_id=endpoint_id,
                    feature_id=repeated_payload_row["vcg_feature_id"],
                )
                repeated_design = _repeated_marker_design(
                    repeated_frame,
                    repeated_payload_row["vcg_feature_id"],
                    int(repeated_payload_row["vcg_direction"]),
                )
                all_ecg_results.append(
                    _locked_all_ecg_record(
                        repeated_design,
                        repeated_model,
                        feature_id=repeated_payload_row["vcg_feature_id"],
                        specification=validation_spec,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(
                    {
                        "endpoint_id": endpoint_id,
                        "stage": "locked_all_ecg_clustered",
                        "reason": str(exc),
                    }
                )
        if report_label_id and "report_reference_state" in frame:
            report_record = _locked_report_record(
                endpoint_id,
                report_label_id,
                frame,
                validation_spec,
            )
            if report_record is not None:
                machine_reports.append(report_record)
        for adjusted_payload_row in adjusted_by_endpoint.get(endpoint_id, []):
            try:
                model = FrozenLogisticModel.from_dict(
                    adjusted_payload_row["model"]
                )
                design = _endpoint_design_frame(
                    frame,
                    endpoint_id=endpoint_id,
                    vcg_feature_id=adjusted_payload_row["vcg_feature_id"],
                    vcg_direction=int(adjusted_payload_row["vcg_direction"]),
                    classical_feature_id=adjusted_payload_row.get(
                        "classical_feature_id"
                    ),
                )
                metrics = evaluate_logistic_model(design, model)
                adjusted_results.append(
                    {
                        "endpoint_id": endpoint_id,
                        "model_id": model.model_id,
                        "model_role": model.model_role,
                        "evaluation_split": "locked_test",
                        "metric_scope": "locked_test_final",
                        "predictors": "|".join(model.predictor_names),
                        "parameter_count": len(model.coefficients),
                        "events_per_parameter_train": (
                            model.events_per_parameter
                        ),
                        "baseline_model_id": model.baseline_model_id,
                        "vcg_feature_id": (
                            adjusted_payload_row["vcg_feature_id"]
                            if "vcg_marker" in model.predictor_names
                            else None
                        ),
                        "automatic_feature_selection": False,
                        **metrics,
                    }
                )
                if metrics.get("decision_curve_eligible", False):
                    decision_curve_results.extend(
                        decision_curve(
                            design,
                            model,
                            thresholds=np.arange(0.05, 0.51, 0.05),
                            evaluation_split="locked_test",
                        )
                    )
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(
                    {
                        "endpoint_id": endpoint_id,
                        "stage": "locked_adjusted_model",
                        "reason": str(exc),
                    }
                )

    final_results = apply_fdr_by_family(
        pd.DataFrame(results),
        family_columns=("endpoint_id", "panel_id", "evaluation_split"),
        alpha_by_panel={
            "primary_core": validation_spec.primary_fdr_alpha,
            "classical": validation_spec.primary_fdr_alpha,
            "sentinel": validation_spec.primary_fdr_alpha,
            "discovery_reserve": validation_spec.discovery_fdr_alpha,
        },
    )
    outputs = {
        "final_test_report": final_results,
        "final_test_combination_results": pd.DataFrame(combinations),
        "final_test_classical_baselines": pd.DataFrame(classical),
        "final_test_machine_report_baselines": pd.DataFrame(machine_reports),
        "final_test_label_counts": pd.DataFrame(label_counts),
        "final_test_failures": pd.DataFrame(failures),
        "final_test_adjusted_models": pd.DataFrame(adjusted_results),
        "final_test_decision_curves": pd.DataFrame(decision_curve_results),
        "final_test_window_sensitivity": pd.DataFrame(
            window_sensitivity_results
        ),
        "final_test_all_ecg_clustered": pd.DataFrame(all_ecg_results),
    }
    output_paths = []
    for name, frame in outputs.items():
        path = run_root / f"{name}.parquet"
        atomic_write_parquet(frame, path)
        output_paths.append(str(path.resolve()))

    artifact_hash = content_hash(
        {
            "freeze_hash": current_hash,
            "outputs": {
                Path(path).name: file_sha256(path) for path in output_paths
            },
        }
    )
    audit_path = run_root / "test_evaluation_audit.json"
    audit = {
        "run_id": options.run_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "freeze_hash": current_hash,
        "artifact_hash": artifact_hash,
        "locked_test_marker_shortlist_hash": shortlist["shortlist_hash"],
        "locked_test_marker_hypothesis_count": shortlist["entry_count"],
        "locked_test_endpoint_count": shortlist["endpoint_count"],
        "feature_files": [str(Path(path).resolve()) for path in feature_paths],
        "row_counts": {name: len(frame) for name, frame in outputs.items()},
        "post_hoc_review_policy": (
            "Locked-test review is post hoc and cannot alter this frozen run."
        ),
    }
    atomic_write_json(audit_path, audit)
    output_paths.append(str(audit_path.resolve()))
    lock.consume(artifact_hash)
    atomic_write_json(lock_path, lock.to_dict())

    manifest = StageManifest.complete(
        stage=f"evaluate-test:{options.run_id}",
        stage_version=LOCKED_EVALUATION_VERSION,
        input_hash=current_hash,
        config_hash=scientific_config_hash(config),
        output_paths=tuple(output_paths),
        row_counts={name: len(frame) for name, frame in outputs.items()},
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            "Locked test consumed exactly once.",
            (
                "Only hypotheses in locked_test_marker_shortlist.json were "
                "evaluated."
            ),
            "All directions, scaling and thresholds came from the frozen train run.",
        ),
    )
    manifest_path = run_root / "evaluate-test.manifest.json"
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    return manifest


def invalidate_test_lock(
    config: MimicProjectConfig,
    run_id: str,
    reason: str,
) -> TestLock:
    if not reason.strip():
        raise ValueError("A documented invalidation reason is required")
    lock_path = config.runs_root / run_id / "test_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"No test lock exists for run {run_id!r}")
    lock = _load_test_lock(lock_path)
    if lock.is_consumed:
        raise RuntimeError(
            "A consumed locked-test run remains immutable; create a new run ID."
        )
    lock.invalidate(reason.strip())
    atomic_write_json(lock_path, lock.to_dict())
    return lock


def _locked_marker_record(
    endpoint_id: str,
    panel_id: str,
    feature_id: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    model: MarkerModel,
    specification: ValidationSpec,
    *,
    seed_offset: int,
) -> dict:
    labels = frame["label"].to_numpy(dtype=np.int8)
    metrics = evaluate_marker(
        values,
        labels,
        direction=model.direction,
        thresholds=model.thresholds,
    )
    n_positive = int(metrics["n_positive"])
    n_negative = int(metrics["n_negative"])
    status = inference_status(
        n_positive,
        n_negative,
        minimum_inference=specification.min_class_inference,
        minimum_primary=specification.min_class_primary,
    )
    metrics = enforce_inference_policy(metrics, status)
    record = {
        "endpoint_id": endpoint_id,
        "panel_id": panel_id,
        "feature_id": feature_id,
        "analysis_type": "single_marker",
        "evaluation_split": "locked_test",
        "metric_scope": "locked_test_final",
        "threshold_source": "frozen_train",
        "direction_source": "frozen_train",
        "direction": model.direction,
        "missingness": metrics.get("overall_missingness", np.nan),
        "inference_status": status,
        "exploratory_score": False,
        **metrics,
    }
    record.update(
        marker_bootstrap_intervals(
            frame["subject_id"].to_numpy(),
            values,
            labels,
            direction=model.direction,
            thresholds=model.thresholds,
            iterations=(
                specification.final_bootstrap_iterations
                if status != "descriptive_only"
                else 0
            ),
            seed=specification.random_seed + seed_offset,
        )
    )
    return record


def _locked_window_sensitivity_records(
    catalog: DuckDBCatalog,
    *,
    cohort_path: Path,
    feature_glob: str,
    endpoint_id: str,
    feature_id: str,
    model: MarkerModel,
    specification: ValidationSpec,
) -> list[dict]:
    cohort_sql = str(cohort_path.resolve()).replace("\\", "/").replace(
        "'",
        "''",
    )
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    feature_column = _quoted_identifier(feature_id)
    read_features = feature_parquet_clause(sql_path=feature_sql)
    with catalog.connect(isolated=True) as connection:
        frame = connection.execute(
            f"""
            SELECT
                c.subject_id,
                c.study_id,
                c.ecg_time,
                c.split,
                c.label_state,
                c.confidence,
                c.time_delta_hours,
                c.window_profile,
                c.window_before_hours,
                c.window_after_hours,
                f.{feature_column} AS {feature_column}
            FROM read_parquet('{cohort_sql}') c
            JOIN {read_features} f USING (subject_id, study_id)
            WHERE c.label_id = ?
              AND c.split = 'locked_test'
              AND c.eligibility_state = 'eligible'
              AND c.label_state IN ('positive', 'negative')
            ORDER BY
                c.window_before_hours,
                c.subject_id,
                c.study_id
            """,
            [endpoint_id],
        ).fetchdf()
    if frame.empty:
        return []
    frame["label"] = frame["label_state"].eq("positive").astype(np.int8)
    records = []
    for profile_values, profile_frame in frame.groupby(
        ["window_profile", "window_before_hours", "window_after_hours"],
        sort=True,
        dropna=False,
    ):
        profile, before, after = profile_values
        profile_frame = profile_frame.reset_index(drop=True)
        values = _numeric_feature(profile_frame, feature_id)
        record = _locked_marker_record(
            endpoint_id,
            "window_sensitivity",
            feature_id,
            profile_frame,
            values,
            model,
            specification,
            seed_offset=(
                300_000
                + int(content_hash(f"{endpoint_id}:{profile}")[:6], 16)
            ),
        )
        record.update(
            {
                "analysis_type": "label_window_sensitivity",
                "sensitivity_dimension": "clinical_time_window",
                "window_profile": profile,
                "window_before_hours": float(before),
                "window_after_hours": float(after),
                "reference_model_source": "frozen_standard_endpoint_train",
                "threshold_source": "frozen_standard_endpoint_train",
            }
        )
        records.append(record)
    return records


def _load_locked_all_ecg_frame(
    catalog: DuckDBCatalog,
    *,
    label_path: Path,
    feature_glob: str,
    endpoint_id: str,
    feature_id: str,
) -> pd.DataFrame:
    label_sql = str(label_path.resolve()).replace("\\", "/").replace("'", "''")
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    feature_column = _quoted_identifier(feature_id)
    read_features = feature_parquet_clause(sql_path=feature_sql)
    with catalog.connect(isolated=True) as connection:
        frame = connection.execute(
            f"""
            SELECT
                l.subject_id,
                l.study_id,
                l.ecg_time,
                l.split,
                l.label_state,
                f.{feature_column} AS {feature_column}
            FROM read_parquet('{label_sql}') l
            JOIN {read_features} f USING (subject_id, study_id)
            WHERE l.label_id = ?
              AND l.split = 'locked_test'
              AND l.eligibility_state = 'eligible'
              AND l.label_state IN ('positive', 'negative')
            ORDER BY l.subject_id, l.ecg_time, l.study_id
            """,
            [endpoint_id],
        ).fetchdf()
    frame["label"] = frame["label_state"].eq("positive").astype(np.int8)
    return frame


def _locked_all_ecg_record(
    design: pd.DataFrame,
    model: FrozenLogisticModel,
    *,
    feature_id: str,
    specification: ValidationSpec,
) -> dict:
    probabilities, labels, valid = predict_logistic_model(design, model)
    if len(labels) == 0:
        raise ValueError("No evaluable locked repeated-ECG rows")
    subjects = design.loc[valid, "subject_id"].to_numpy()
    positive_subjects = int(pd.Series(subjects[labels == 1]).nunique())
    negative_subjects = int(pd.Series(subjects[labels == 0]).nunique())
    status = inference_status(
        positive_subjects,
        negative_subjects,
        minimum_inference=specification.min_class_inference,
        minimum_primary=specification.min_class_primary,
    )
    probability_metrics = evaluate_logistic_model(design, model)
    marker_metrics = evaluate_marker(probabilities, labels)
    uncertainty = marker_bootstrap_intervals(
        subjects,
        probabilities,
        labels,
        direction=1,
        thresholds={},
        iterations=(
            specification.final_bootstrap_iterations
            if status != "descriptive_only"
            else 0
        ),
        seed=(
            specification.random_seed
            + int(content_hash(f"{model.endpoint_id}:all-ecg:locked")[:8], 16)
        ),
    )
    return {
        "endpoint_id": model.endpoint_id,
        "model_id": model.model_id,
        "model_role": model.model_role,
        "evaluation_split": "locked_test",
        "metric_scope": "all_ecg_locked_test_final",
        "inference_method": (
            "frozen_binomial_glm_with_subject_cluster_bootstrap_evaluation"
        ),
        "vcg_feature_id": feature_id,
        "n_ecgs": int(len(labels)),
        "n_subjects": int(pd.Series(subjects).nunique()),
        "positive_subjects": positive_subjects,
        "negative_subjects": negative_subjects,
        "inference_status": status,
        "roc_auc": marker_metrics["roc_auc"],
        "average_precision": marker_metrics["average_precision"],
        "pr_auc": marker_metrics["pr_auc"],
        "brier_score": probability_metrics["brier_score"],
        "log_loss": probability_metrics["log_loss"],
        "calibration_intercept": probability_metrics[
            "calibration_intercept"
        ],
        "calibration_slope": probability_metrics["calibration_slope"],
        "decision_curve_eligible": False,
        **uncertainty,
    }


def _locked_classical_record(
    endpoint_id: str,
    feature_id: str,
    frame: pd.DataFrame,
    specification: ValidationSpec,
) -> Optional[dict]:
    column = f"{feature_id}_positive"
    if column not in frame:
        return None
    valid = frame[column].notna().to_numpy()
    if not valid.any():
        return None
    labels = frame.loc[valid, "label"].to_numpy(dtype=np.int8)
    predictions = frame.loc[valid, column].astype(bool).to_numpy()
    metrics = classification_metrics(labels, predictions)
    marker_metrics = evaluate_marker(predictions.astype(float), labels)
    status = inference_status(
        int(np.sum(labels == 1)),
        int(np.sum(labels == 0)),
        minimum_inference=specification.min_class_inference,
        minimum_primary=specification.min_class_primary,
    )
    uncertainty = _binary_baseline_intervals(
        frame.loc[valid, "subject_id"].to_numpy(),
        predictions,
        labels,
        specification,
        status=status,
        seed_key=f"{endpoint_id}:{feature_id}:locked_test",
        final=True,
    )
    return {
        "endpoint_id": endpoint_id,
        "baseline_id": feature_id,
        "baseline_type": "classical_rule",
        "evaluation_split": "locked_test",
        "threshold_source": "predefined_clinical_rule",
        "n_total": len(labels),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "missingness": float(1.0 - np.mean(valid)),
        "roc_auc": marker_metrics["roc_auc"],
        "average_precision": marker_metrics["average_precision"],
        "pr_auc": marker_metrics["pr_auc"],
        "inference_status": status,
        **metrics,
        **uncertainty,
    }


def _locked_report_record(
    endpoint_id: str,
    report_label_id: str,
    frame: pd.DataFrame,
    specification: ValidationSpec,
) -> Optional[dict]:
    report_states = frame["report_reference_state"].astype("string")
    valid = report_states.isin(
        ("positive", "negative", "not_mentioned")
    ).to_numpy()
    if not valid.any():
        return None
    labels = frame.loc[valid, "label"].to_numpy(dtype=np.int8)
    states = report_states.loc[valid].astype(str).to_numpy()
    all_states = report_states.dropna().astype(str).to_numpy()
    predictions = states == "positive"
    metrics = classification_metrics(labels, predictions)
    marker_metrics = evaluate_marker(predictions.astype(float), labels)
    status = inference_status(
        int(np.sum(labels == 1)),
        int(np.sum(labels == 0)),
        minimum_inference=specification.min_class_inference,
        minimum_primary=specification.min_class_primary,
    )
    uncertainty = _binary_baseline_intervals(
        frame.loc[valid, "subject_id"].to_numpy(),
        predictions,
        labels,
        specification,
        status=status,
        seed_key=f"{endpoint_id}:report:{report_label_id}:locked_test",
        final=True,
    )
    return {
        "endpoint_id": endpoint_id,
        "baseline_id": f"machine_report_{report_label_id}",
        "baseline_type": "machine_report_positive_only",
        "evaluation_split": "locked_test",
        "threshold_source": "frozen_report_parser",
        "n_total": len(labels),
        "n_source_rows": len(frame),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "missingness": float(1.0 - np.mean(valid)),
        "roc_auc": marker_metrics["roc_auc"],
        "average_precision": marker_metrics["average_precision"],
        "pr_auc": marker_metrics["pr_auc"],
        "positive_count": int(np.sum(all_states == "positive")),
        "negative_count": int(np.sum(all_states == "negative")),
        "uncertain_count": int(np.sum(all_states == "uncertain")),
        "not_mentioned_count": int(
            np.sum(all_states == "not_mentioned")
        ),
        "unavailable_count": int(report_states.isna().sum()),
        "uncertain_excluded": True,
        "inference_status": status,
        **metrics,
        **uncertainty,
    }


def _freeze_payload(
    config: MimicProjectConfig,
    run_id: str,
    *,
    panel_registry_path: Path,
) -> dict:
    run_root = config.runs_root / run_id
    package_root = Path(__file__).resolve().parents[1]
    locked_shortlist = _load_and_validate_locked_test_marker_shortlist(
        run_root,
        expected_run_id=run_id,
        require_nonempty=True,
    )
    clinical_utility_scope = _clinical_utility_scope_freeze(run_root)
    current_code_hash = locked_evaluation_implementation_hash()
    validation_manifest = json.loads(
        (run_root / "validate.manifest.json").read_text(encoding="utf-8")
    )
    validation_spec = json.loads(
        (run_root / "validation_spec.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (run_root / "development_provenance.json").read_text(encoding="utf-8")
    )
    validation_code_verification = verify_validation_code_provenance(
        run_root
    )
    baseline_registry = BaselineRegistry()
    methodology_registry = MethodologyRegistry()
    if (
        validation_manifest.get("status") != "complete"
        or validation_manifest.get("stage") != f"validate:{run_id}"
    ):
        raise RuntimeError(
            "Freeze refused: validation stage manifest is incomplete or "
            "belongs to another run."
        )
    if provenance.get("input_hash") != validation_manifest.get("input_hash"):
        raise RuntimeError(
            "Freeze refused: validation provenance and stage manifest refer "
            "to different inputs."
        )
    validation_input_payload = provenance.get("validation_input_payload")
    if not isinstance(validation_input_payload, dict):
        raise RuntimeError(
            "Freeze refused: validation has no immutable input payload."
        )
    if (
        content_hash(validation_input_payload)
        != validation_manifest.get("input_hash")
    ):
        raise RuntimeError(
            "Freeze refused: stored validation inputs do not reproduce the "
            "validation manifest hash."
        )
    if (
        provenance.get("preview") is True
        or provenance.get("analysis_completeness") != "complete_snapshot"
        or provenance.get("feature_scope") != "development_only"
        or validation_input_payload.get("analysis_scope")
        != "development_only"
    ):
        raise RuntimeError(
            "Freeze refused: preview or partial validation results are never "
            "freezable. Run validation on the complete development snapshot."
        )
    validation_input_verification = (
        verify_validation_input_payload_files(validation_input_payload)
    )
    label_attestation = verify_development_label_attestation(config)
    if (
        provenance.get("development_label_attestation_hash")
        != label_attestation.get("attestation_hash")
        or validation_input_payload.get(
            "development_label_attestation_hash"
        )
        != label_attestation.get("attestation_hash")
    ):
        raise RuntimeError(
            "Freeze refused: development label attestation changed after "
            "validation."
        )
    event_evidence_provenance = (
        verify_clinical_event_evidence_provenance(config)
    )
    event_evidence_summary = _event_evidence_provenance_summary(
        event_evidence_provenance
    )
    if (
        provenance.get("clinical_event_evidence_provenance")
        != event_evidence_summary
        or validation_input_payload.get(
            "clinical_event_evidence_provenance"
        )
        != event_evidence_summary
    ):
        raise RuntimeError(
            "Freeze refused: standardized clinical-event evidence "
            "provenance changed after validation."
        )
    if provenance.get("locked_test_labels_read") is not False:
        raise RuntimeError(
            "Freeze refused: development provenance does not explicitly "
            "confirm that locked-test labels remained unread."
        )
    current_config_hash = scientific_config_hash(config)
    if provenance.get("project_config_hash") != current_config_hash:
        raise RuntimeError(
            "Freeze refused: development validation used a different project "
            "configuration. Re-run validate before freezing."
        )
    if (
        provenance.get("baseline_registry_hash")
        != baseline_registry.registry_hash
    ):
        raise RuntimeError(
            "Freeze refused: the classical ECG baseline registry or its "
            "pinned source metadata changed after validation."
        )
    if (
        provenance.get("methodology_registry_hash")
        != methodology_registry.registry_hash
    ):
        raise RuntimeError(
            "Freeze refused: the pinned scientific methodology registry "
            "changed after validation."
        )
    feature_integrity = provenance.get("feature_integrity", {})
    if provenance.get("feature_scope") != "development_only":
        raise RuntimeError(
            "Freeze refused: development validation must use a complete "
            "development-only feature extraction."
        )
    if feature_integrity.get("status") != "verified":
        raise RuntimeError(
            "Freeze refused: development feature artifacts were not verified."
        )
    verified_integrity = verify_feature_artifacts(
        [Path(path) for path in provenance.get("feature_files", [])]
    )
    if (
        verified_integrity["artifact_hash"]
        != feature_integrity.get("artifact_hash")
    ):
        raise RuntimeError(
            "Freeze refused: feature artifact integrity changed after validation."
        )
    feature_manifest = json.loads(
        Path(verified_integrity["manifest_path"]).read_text(encoding="utf-8")
    )
    if (
        feature_manifest.get("status") != "complete"
        or feature_manifest.get("input_hash")
        != verified_integrity.get("stage_input_hash")
    ):
        raise RuntimeError(
            "Freeze refused: feature stage manifest is incomplete or "
            "inconsistent with its partition index."
        )
    snapshot_payload = provenance.get("feature_snapshot", {})
    snapshot_id = str(snapshot_payload.get("snapshot_id", ""))
    if not snapshot_id:
        raise RuntimeError(
            "Freeze refused: development validation has no immutable feature "
            "snapshot."
        )
    feature_snapshot_path = config.snapshots_root / f"{snapshot_id}.json"
    feature_snapshot = load_artifact_snapshot(feature_snapshot_path)
    feature_snapshot.assert_freezable()
    if (
        feature_snapshot.snapshot_hash
        != snapshot_payload.get("snapshot_hash")
        or feature_snapshot.input_hash
        != verified_integrity.get("stage_input_hash")
    ):
        raise RuntimeError(
            "Freeze refused: feature snapshot provenance is inconsistent."
        )
    snapshot_verification = verify_artifact_snapshot(feature_snapshot)
    if not snapshot_verification["valid"]:
        raise RuntimeError(
            "Freeze refused: feature snapshot integrity verification failed: "
            f"{snapshot_verification}"
        )
    tracked_files = {
        "scientific_analysis_contract": (
            package_root / "SCIENTIFIC_HARDENING_2026-08-12.md"
        ),
        "reproducibility_contract": package_root / "REPRODUCIBILITY.md",
        "hashed_environment_lock": (
            package_root / "requirements-lock-win-py312-hashed.txt"
        ),
        "panel_registry": panel_registry_path,
        "subject_splits": config.master_root / "subject_splits.parquet",
        "labels_manifest": config.labels_root / "labels.manifest.json",
        "development_label_attestation": (
            development_label_attestation_path(config)
        ),
        "development_clinical_label_evidence": (
            config.labels_root / "clinical_label_evidence.parquet"
        ),
        "development_clinical_labels_wide": (
            config.labels_root / "clinical_labels_wide.parquet"
        ),
        "development_endpoint_index_cohorts": (
            config.labels_root / "endpoint_index_cohorts.parquet"
        ),
        "development_endpoint_window_sensitivity_cohorts": (
            config.labels_root
            / "endpoint_window_sensitivity_cohorts.parquet"
        ),
        "development_label_definitions": (
            config.labels_root / "label_definitions.json"
        ),
        "locked_test_seal": (
            config.labels_root / "sealed" / "locked_test_seal.json"
        ),
        "locked_endpoint_index_cohorts": (
            config.labels_root
            / "sealed"
            / "locked_endpoint_index_cohorts.parquet"
        ),
        "locked_endpoint_window_sensitivity_cohorts": (
            config.labels_root
            / "sealed"
            / "locked_endpoint_window_sensitivity_cohorts.parquet"
        ),
        "locked_clinical_label_evidence": (
            config.labels_root
            / "sealed"
            / "locked_clinical_label_evidence.parquet"
        ),
        "validation_manifest": run_root / "validate.manifest.json",
        "trained_marker_models": run_root / "trained_marker_models.json",
        "cross_fit_marker_models": run_root / "cross_fit_marker_models.json",
        "locked_test_marker_shortlist": (
            run_root / "locked_test_marker_shortlist.json"
        ),
        "frozen_score_candidates": run_root / "frozen_score_candidates.json",
        "criterion_registry": run_root / "criterion_registry.json",
        "adjusted_model_specs": run_root / "adjusted_model_specs.json",
        "all_ecg_clustered_models": (
            run_root / "all_ecg_clustered_models.json"
        ),
        "validation_spec": run_root / "validation_spec.json",
        "development_provenance": run_root / "development_provenance.json",
        "development_acceptance_audit": (
            run_root / "acceptance_pre_freeze_audit.json"
        ),
        "feature_manifest": Path(verified_integrity["manifest_path"]),
        "feature_partition_index": Path(verified_integrity["index_path"]),
        "feature_snapshot": feature_snapshot_path,
        "baseline_source_lock": baseline_registry.source_lock_path,
        "methodology_source_lock": methodology_registry.source_lock_path,
        **{
            f"development_{Path(name).stem}": run_root / name
            for name in FROZEN_DEVELOPMENT_TABLES
        },
    }
    if event_evidence_provenance.get("status") == "direct_event_backed":
        tracked_files["clinical_event_evidence_provenance"] = (
            config.evidence_root / "evidence_provenance.json"
        )
    else:
        tracked_files[
            "clinical_event_evidence_equivalence_attestation"
        ] = (
            config.work_root
            / "provenance"
            / "clinical_event_evidence_equivalence.json"
        )
    validation_attestation_path = validation_code_verification.get(
        "attestation_path"
    )
    if validation_attestation_path:
        tracked_files["validation_code_scope_attestation"] = Path(
            str(validation_attestation_path)
        )
    ptb_transfer = provenance.get("ptb_transfer", {})
    if ptb_transfer.get("status") == "verified_frozen":
        _, current_ptb_provenance = load_ptb_frozen_transfer_scores(
            config,
            run_id=str(ptb_transfer["run_id"]),
        )
        if (
            current_ptb_provenance.get("freeze_hash")
            != ptb_transfer.get("freeze_hash")
            or current_ptb_provenance.get("score_file_sha256")
            != ptb_transfer.get("score_file_sha256")
        ):
            raise RuntimeError(
                "Freeze refused: the PTB frozen-transfer artifact changed "
                "after MIMIC validation."
            )
        tracked_files["ptb_transfer_freeze_manifest"] = Path(
            current_ptb_provenance["freeze_manifest_path"]
        )
        tracked_files["ptb_transfer_scores"] = Path(
            current_ptb_provenance["score_path"]
        )
    if clinical_utility_scope["clinical_utility_claim_scope_prespecified"]:
        tracked_files["clinical_utility_scope_attestation"] = (
            run_root / CLINICAL_UTILITY_SCOPE_ATTESTATION
        )
    missing = [str(path) for path in tracked_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Freeze inputs are missing: {missing}")
    return {
        "freeze_version": FREEZE_VERSION,
        "run_id": run_id,
        "project_config_hash": current_config_hash,
        "operational_config_hash": operational_config_hash(config),
        "code_hash": current_code_hash,
        "development_validation_implementation": (
            validation_code_verification
        ),
        "label_definitions_hash": content_hash(
            [definition.to_dict() for definition in default_label_definitions()]
        ),
        "baseline_registry_hash": baseline_registry.registry_hash,
        "methodology_registry_hash": methodology_registry.registry_hash,
        "validation_input_hash": validation_manifest["input_hash"],
        "validation_input_verification": validation_input_verification,
        "development_label_attestation_hash": label_attestation[
            "attestation_hash"
        ],
        "clinical_event_evidence_provenance": event_evidence_summary,
        "clinical_utility_scope": clinical_utility_scope,
        "split_hash": validation_spec["split_hash"],
        "feature_integrity": verified_integrity,
        "tracked_file_sha256": {
            name: file_sha256(path) for name, path in tracked_files.items()
        },
        "package_versions": _package_versions(),
        "test_policy": {
            "one_time": True,
            "threshold_source": "train",
            "selection_source": "validation",
            "selection_contract": "explicit_frozen_marker_by_endpoint_shortlist",
            "marker_hypothesis_count": locked_shortlist["entry_count"],
            "endpoint_count": locked_shortlist["endpoint_count"],
            "shortlist_hash": locked_shortlist["shortlist_hash"],
            "score_candidates_in_locked_test": False,
            "locked_test_review": "post_hoc_only",
            "clinical_utility_interpretation": (
                "internal_replication_only; external validation and prospective "
                "impact evidence are required for clinical-utility claims"
            ),
        },
    }


def _assert_locked_test_feature_scope(
    integrity: dict,
    *,
    expected_freeze_hash: str,
) -> None:
    root = Path(str(integrity.get("root", "")))
    manifest_path = Path(str(integrity.get("manifest_path", "")))
    index_path = Path(str(integrity.get("index_path", "")))
    if root.name != f"locked-test-{expected_freeze_hash[:12]}":
        raise RuntimeError(
            "Locked-test feature root is not bound to the active freeze hash."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notes = {str(note) for note in manifest.get("notes", ())}
    required_notes = {
        "locked_test_features_included=True",
        "locked_test_only=True",
        "locked_test_selection=frozen_locked_cohort_only",
        f"freeze_hash={expected_freeze_hash}",
    }
    if not required_notes <= notes:
        raise RuntimeError(
            "Locked-test feature provenance is incomplete or references a "
            "different freeze."
        )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    relative_paths = {
        str(item.get("relative_path", ""))
        for item in index_payload.get("partitions", ())
    }
    if not relative_paths or any(
        not path.startswith("split=locked_test/")
        for path in relative_paths
    ):
        raise RuntimeError(
            "Locked-test extraction artifact contains a non-test partition."
        )


def _load_validation_spec(path: Path) -> ValidationSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["endpoint_ids"] = tuple(data["endpoint_ids"])
    data["threshold_targets"] = tuple(data.get("threshold_targets", ()))
    return ValidationSpec(**data)


def _load_frozen_score(data: dict) -> FrozenScoreSpec:
    payload = dict(data)
    for key in (
        "components",
        "directions",
        "means",
        "standard_deviations",
    ):
        payload[key] = tuple(payload[key])
    return FrozenScoreSpec(**payload)


def _update_score_release_stage(
    path: Path,
    stage: ScoreReleaseStage,
    *,
    freeze_hash: str,
    locked_test_artifact_hash: str = "",
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Score release registry is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    releases = payload.get("releases")
    if not isinstance(releases, list):
        raise ValueError(f"Invalid score release registry: {path}")
    for release in releases:
        if not isinstance(release, dict):
            raise ValueError(f"Invalid score release entry: {path}")
        release["stage"] = stage.value
        release["freeze_hash"] = freeze_hash
        release["locked_test_artifact_hash"] = locked_test_artifact_hash
        release["primary_case_display_allowed"] = stage in {
            ScoreReleaseStage.LOCKED_TEST_VALIDATED,
            ScoreReleaseStage.EXTERNAL_VALIDATED,
        }
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, payload)


def _load_test_lock(path: Path) -> TestLock:
    return TestLock(**json.loads(path.read_text(encoding="utf-8")))


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "pyarrow",
        "duckdb",
        "wfdb",
        "neurokit2",
        "scikit-learn",
        "scipy",
        "statsmodels",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
