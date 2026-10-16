"""Separate fail-closed development-run release, never a clinical approval."""
from __future__ import annotations

import json
from pathlib import Path

from .result_contract import true
from .storage import file_sha256

SCIENTIFIC_RELEASE_VERSION = "2.2.0"
REQUIRED_RELEASE_GATES = ("reference_and_temporal_contract", "patient_and_label_semantics",
    "sampling_and_missingness", "numerical_family_coverage", "software_suite",
    "real_pilot_gui", "resource_scaling_and_storage", "workspace_isolation",
    "production_path")

REQUIRED_GATE_CHECKS = {
    "reference_and_temporal_contract": (
        "independent_modality_separate_from_clock", "observed_reference_required",
        "temporal_uncertainty_reported", "reference_blind_index_preserved"),
    "patient_and_label_semantics": (
        "patient_splits_disjoint", "censoring_preserved", "empty_echo_not_negative",
        "current_label_attestation_verified", "locked_test_not_evaluated"),
    "sampling_and_missingness": (
        "sampling_design_declared", "missingness_denominators_verified",
        "paired_comparison_population_verified", "enrichment_metrics_guarded"),
    "numerical_family_coverage": (
        "all_registered_features_accounted_for", "no_unresolved_bound_violations",
        "alias_audit_passed", "current_feature_freeze_verified"),
    "software_suite": (
        "source_bound_test_results", "no_test_failures", "no_unexpected_skips",
        "no_unexpected_warnings", "static_checks_passed"),
    "real_pilot_gui": (
        "real_pilot_results_reviewed", "real_ecg_lab_reviewed",
        "operating_points_unambiguous", "missing_evidence_visible",
        "required_display_modes_reviewed"),
    "resource_scaling_and_storage": (
        "ram_reserve_observed", "spill_budget_enforced",
        "concurrency_bound_verified", "disk_capacity_sufficient",
        "scale_up_down_verified", "resume_verified"),
    "workspace_isolation": (
        "input_artifacts_unchanged", "output_paths_isolated",
        "pilot_workspace_integrity_verified", "gui_independent_execution_verified"),
    "production_path": (
        "development_only", "patient_grouped", "endpoint_blind_sampling",
        "adjusted_models_executed", "score_search_executed",
        "calibration_evaluated_or_explicitly_abstained"),
}

DELTA_ACCEPTANCE_CHECKS = (
    "historical_pilots_preserved", "changes_reviewed", "current_labels_verified",
    "current_marker_implementation_exercised", "production_models_exercised",
    "runtime_exercised", "no_old_features_adopted", "no_locked_test_evaluation",
)


def reviewed_delta_acceptance_valid(run_root: Path, fingerprint: str | None = None) -> bool:
    """Explicit alternative to repeating broad QC pilots after bounded changes.

    This does not label historic runs as current pilots or reuse their features.
    All nine scientific gates remain mandatory. The reviewed change assessment
    and current delta reports are immutable, local, hash-bound evidence.
    """
    if not scientific_release_valid(run_root, fingerprint):
        return False
    try:
        release = json.loads((run_root / "scientific_release.json").read_text(encoding="utf-8"))
        item = release["technical_acceptance"]
        path = (run_root / item["relative_path"]).resolve()
        if (Path(item["relative_path"]).is_absolute() or not path.is_relative_to(run_root.resolve())
                or file_sha256(path) != item["sha256"]):
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("version") != "reviewed-delta-1.0"
                or payload.get("run_id") != run_root.name
                or payload.get("run_fingerprint") != release["run_fingerprint"]
                or payload.get("status") != "PASS"
                or payload.get("full_run_started") is not False
                or any(payload.get("checks", {}).get(key) is not True for key in DELTA_ACCEPTANCE_CHECKS)):
            return False
        proofs = payload.get("evidence", {})
        for key in ("historical_1000", "historical_10000", "change_review", "current_delta", "runtime"):
            proof = proofs[key]
            local = (run_root / proof["relative_path"]).resolve()
            if (Path(proof["relative_path"]).is_absolute() or not local.is_relative_to(run_root.resolve())
                    or not local.is_file() or file_sha256(local) != proof["sha256"]):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def gate_content_passed(payload: dict, gate: str, fingerprint: str) -> bool:
    """File existence is not a scientific test; legacy summaries fail closed."""
    if (not isinstance(payload, dict) or gate not in REQUIRED_GATE_CHECKS
            or payload.get("gate") != gate or payload.get("status") != "PASS"
            or payload.get("run_fingerprint") != fingerprint):
        return False
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        return False
    return all(checks.get(name) is True for name in REQUIRED_GATE_CHECKS[gate])


def scientific_release_valid(run_root: Path, fingerprint: str | None = None) -> bool:
    """A technical pilot GO or an unbound checkbox cannot authorize full work."""
    try:
        payload = json.loads((run_root/"scientific_release.json").read_text(encoding="utf-8"))
        if (payload.get("version") != SCIENTIFIC_RELEASE_VERSION
                or payload.get("run_id") != run_root.name
                or payload.get("status") != "approved_for_development_full_run"
                or not payload.get("run_fingerprint")
                or payload.get("clinical_approval") is not False
                or payload.get("locked_test_read") is not False):
            return False
        if fingerprint is not None and payload["run_fingerprint"] != fingerprint:
            return False
        gates = payload.get("gates", {})
        evidence = payload.get("gate_evidence", {})
        for name in REQUIRED_RELEASE_GATES:
            if not true(gates.get(name)) or name not in evidence:
                return False
            item = evidence[name]
            path = (run_root/item["relative_path"]).resolve()
            if (Path(item["relative_path"]).is_absolute() or not path.is_relative_to(run_root.resolve())
                    or not path.is_file() or file_sha256(path) != item.get("sha256")):
                return False
            if not gate_content_passed(
                json.loads(path.read_text(encoding="utf-8")), name,
                payload["run_fingerprint"],
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
