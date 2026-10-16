"""Conservative marker-endpoint evidence contract for clinical presentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from .models import ClinicalEvidenceTier
from .result_contract import scientific_result_contract, true, UNQUALIFIED_DIASTOLIC_ENDPOINTS


EVIDENCE_CONTRACT_VERSION = "1.2.1"


def _first(frame: pd.DataFrame) -> dict:
    return frame.iloc[0].to_dict() if frame is not None and not frame.empty else {}


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _classical_increment_confirmed(
    adjusted: pd.DataFrame,
    endpoint_id: str,
    marker_id: str,
) -> bool:
    if adjusted.empty:
        return False
    required = {"endpoint_id", "vcg_feature_id", "evaluation_split", "model_id"}
    if not required <= set(adjusted.columns):
        return False
    scoped = adjusted.loc[
        adjusted["endpoint_id"].astype(str).eq(endpoint_id)
        & adjusted["vcg_feature_id"].astype(str).eq(marker_id)
        & adjusted["evaluation_split"].astype(str).eq("validation")
        & adjusted["model_id"].astype(str).eq("classical_plus_vcg")
    ]
    row = _first(scoped)
    return bool(
        row
        and str(row.get("incremental_comparison_status")) == "formal_inference"
        and np.isfinite(_number(row.get("incremental_auc")))
        and _number(row.get("incremental_auc")) > 0
        and np.isfinite(_number(row.get("incremental_auc_ci_low")))
        and _number(row.get("incremental_auc_ci_low")) > 0
    )


def _calibration_passed(
    adjusted: pd.DataFrame,
    endpoint_id: str,
    marker_id: str,
) -> bool:
    if adjusted.empty:
        return False
    required = {"endpoint_id", "vcg_feature_id", "evaluation_split", "model_id"}
    if not required <= set(adjusted.columns):
        return False
    scoped = adjusted.loc[
        adjusted["endpoint_id"].astype(str).eq(endpoint_id)
        & adjusted["vcg_feature_id"].astype(str).eq(marker_id)
        & adjusted["evaluation_split"].astype(str).eq("validation")
        & adjusted["model_id"].astype(str).eq("classical_plus_vcg")
    ]
    row = _first(scoped)
    slope = _number(row.get("calibration_slope")) if row else np.nan
    intercept = _number(row.get("calibration_intercept")) if row else np.nan
    return bool(
        np.isfinite(slope)
        and np.isfinite(intercept)
        and 0.8 <= slope <= 1.2
        and abs(intercept) <= 0.2
    )


def _subgroup_stability_passed(
    screening: pd.DataFrame,
    endpoint_id: str,
    marker_id: str,
) -> bool:
    required = {"endpoint_id", "feature_id", "analysis_type", "evaluation_split", "subgroup_variable",
                "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "n_positive", "n_negative"}
    if screening.empty or not required <= set(screening):
        return False
    scoped = screening.loc[
        screening["endpoint_id"].astype(str).eq(endpoint_id)
        & screening["feature_id"].astype(str).eq(marker_id)
        & screening["analysis_type"].astype(str).eq("subgroup_stability")
        & screening["evaluation_split"].astype(str).eq("validation")
    ].copy()
    if scoped.empty or "roc_auc" not in scoped:
        return False
    scoped["roc_auc"] = pd.to_numeric(scoped["roc_auc"], errors="coerce")
    scoped = scoped.loc[scoped["roc_auc"].notna()]
    if len(scoped) < 2:
        return False
    for name in ("roc_auc_ci_low", "roc_auc_ci_high", "n_positive", "n_negative"):
        scoped[name] = pd.to_numeric(scoped[name], errors="coerce")
    if (scoped[["roc_auc_ci_low", "roc_auc_ci_high"]].isna().any().any()
            or (scoped[["n_positive", "n_negative"]].fillna(0) < 100).any().any()):
        return False
    # Require at least one genuinely paired subgroup variable and reject a
    # clinically material AUC spread larger than ten points.
    paired = False
    for _, group in scoped.groupby("subgroup_variable", sort=True):
        if len(group) < 2:
            continue
        paired = True
        if float(group["roc_auc_ci_high"].max() - group["roc_auc_ci_low"].min()) > 0.10 + 1e-12:
            return False
    return paired


def _net_benefit_passed(decision: pd.DataFrame, endpoint_id: str, marker_id: str) -> bool:
    required = {"endpoint_id", "vcg_feature_id", "model_id", "evaluation_split",
                "clinical_utility_evidence_eligible", "net_benefit",
                "treat_all_net_benefit", "treat_none_net_benefit"}
    if decision.empty or not required <= set(decision):
        return False
    scoped = decision.loc[
        decision["endpoint_id"].astype(str).eq(endpoint_id)
        & decision["vcg_feature_id"].astype(str).eq(marker_id)
        & decision["model_id"].astype(str).eq("classical_plus_vcg")
        & decision["evaluation_split"].astype(str).eq("validation")
        & decision["clinical_utility_evidence_eligible"].map(true)
    ]
    if scoped.empty:
        return False
    benefit = pd.to_numeric(scoped["net_benefit"], errors="coerce")
    comparator = pd.concat(
        [
            pd.to_numeric(scoped["treat_all_net_benefit"], errors="coerce"),
            pd.to_numeric(scoped["treat_none_net_benefit"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    return bool((benefit > comparator).mean() >= 0.8)


def build_candidate_evidence_contract(
    *,
    criteria: pd.DataFrame,
    reporting_set: pd.DataFrame,
    adjusted_models: pd.DataFrame,
    screening: pd.DataFrame,
    decision_curves: pd.DataFrame,
) -> pd.DataFrame:
    """One typed row per criterion; absent evidence is a failed gate."""

    columns = (
        "criterion_id",
        "endpoint_id",
        "marker_id",
        "cohort_id",
        "evaluation_split",
        "model_id",
        "effect_measure",
        "evidence_tier",
        "effect",
        "ci_low",
        "ci_high",
        "q_value",
        "n",
        "events",
        "availability",
        "gate_status",
        "analytical_validity_passed",
        "availability_passed",
        "repeatability_passed",
        "replication_passed",
        "adjustment_passed",
        "incremental_classical_value_passed",
        "transform_robustness_passed",
        "subgroup_stability_passed",
        "prespecified_threshold_passed",
        "calibration_passed",
        "net_benefit_passed",
        "external_validation_passed",
        "failed_gates",
        "abstention_reason",
        "run_id",
        "snapshot_hash",
        "contract_version",
    )
    if criteria.empty:
        return pd.DataFrame(columns=columns)
    reporting = (
        reporting_set.set_index("marker_id", drop=False)
        if not reporting_set.empty and "marker_id" in reporting_set
        else pd.DataFrame()
    )
    rows = []
    for candidate in criteria.to_dict(orient="records"):
        endpoint = str(candidate.get("endpoint_id", ""))
        marker = str(candidate.get("feature_id", ""))
        report = {}
        if not reporting.empty and marker in reporting.index:
            value = reporting.loc[marker]
            report = value.iloc[0].to_dict() if isinstance(value, pd.DataFrame) else value.to_dict()
        analytical = bool(
            true(candidate.get("assessability_gate_passed"))
            and true(candidate.get("missingness_sensitivity_gate_passed"))
            and str(candidate.get("inference_status")) == "formal_inference"
        )
        availability = bool(
            np.isfinite(_number(report.get("estimable_stratum_share")))
            and _number(report.get("estimable_stratum_share")) >= 0.5
        )
        repeatability = bool(
            np.isfinite(_number(report.get("icc")))
            and _number(report.get("icc")) >= 0.75
            and np.isfinite(_number(report.get("repeatability_coefficient")))
        )
        replicated = str(candidate.get("stage", "")) in {
            "validation_replicated",
            "frozen",
            "locked_test_validated",
            "external_validated",
        }
        adjusted = true(candidate.get("adjustment_confirmed", False))
        incremental = _classical_increment_confirmed(adjusted_models, endpoint, marker)
        transform = str(report.get("transform_agreement_class", "")) in {
            "transform_independent",
            "interchangeable_after_conversion",
            "convertible_with_declared_error",
        }
        subgroup = _subgroup_stability_passed(screening, endpoint, marker)
        threshold = bool(
            str(candidate.get("threshold_source", "")) == "train"
            and np.isfinite(_number(candidate.get("specificity_95_threshold")))
        )
        calibration = _calibration_passed(adjusted_models, endpoint, marker)
        net_benefit = _net_benefit_passed(decision_curves, endpoint, marker)
        external = (str(candidate.get("stage", "")) == "external_validated"
                    and true(candidate.get("independent_external_cohort_confirmed"))
                    and true(candidate.get("external_prespecified_evaluation_confirmed")))
        candidate_gates = {
            "analytical_validity": analytical,
            "availability": availability,
            "repeatability": repeatability,
            "replication": replicated,
            "adjustment": adjusted,
            "incremental_classical_value": incremental,
            "transform_robustness": transform,
            "subgroup_stability": subgroup,
        }
        if endpoint in UNQUALIFIED_DIASTOLIC_ENDPOINTS:
            # Current echo evidence has no verified rhythm/applicability
            # contract. Keep proxy associations, but do not promote them to
            # clinical criteria solely through statistical success.
            candidate_gates["echo_reference_applicability"] = False
        guideline_candidate = all(candidate_gates.values())
        clinically_validated = bool(
            guideline_candidate and threshold and calibration and net_benefit and external
        )
        if clinically_validated:
            tier = ClinicalEvidenceTier.CLINICALLY_VALIDATED
        elif guideline_candidate:
            tier = ClinicalEvidenceTier.GUIDELINE_CANDIDATE
        elif replicated:
            tier = ClinicalEvidenceTier.REPLICATED_ASSOCIATION
        else:
            tier = ClinicalEvidenceTier.EXPLORATORY
        all_gates = {
            **candidate_gates,
            "prespecified_threshold": threshold,
            "calibration": calibration,
            "net_benefit": net_benefit,
            "external_validation": external,
        }
        failed = [name for name, passed in all_gates.items() if not passed]
        rows.append(
            {
                "criterion_id": candidate.get("criterion_id"),
                "endpoint_id": endpoint,
                "marker_id": marker,
                "cohort_id": candidate.get("cohort_id", "MIMIC-IV-ECG"),
                "evaluation_split": "validation",
                "model_id": candidate.get("model_id", "single_marker_roc_auc"),
                "effect_measure": "roc_auc",
                "evidence_tier": tier.value,
                "effect": candidate.get("roc_auc"),
                "ci_low": candidate.get("roc_auc_ci_low"),
                "ci_high": candidate.get("roc_auc_ci_high"),
                "q_value": candidate.get("q_value"),
                "n": candidate.get("n_observations"),
                "events": candidate.get("n_positive"),
                "availability": report.get("estimable_stratum_share"),
                "gate_status": (
                    "all_internal_candidate_gates_passed"
                    if guideline_candidate
                    else "required_gates_missing"
                ),
                "analytical_validity_passed": analytical,
                "availability_passed": availability,
                "repeatability_passed": repeatability,
                "replication_passed": replicated,
                "adjustment_passed": adjusted,
                "incremental_classical_value_passed": incremental,
                "transform_robustness_passed": transform,
                "subgroup_stability_passed": subgroup,
                "prespecified_threshold_passed": threshold,
                "calibration_passed": calibration,
                "net_benefit_passed": net_benefit,
                "external_validation_passed": external,
                "failed_gates": "|".join(failed),
                "abstention_reason": "" if tier != ClinicalEvidenceTier.EXPLORATORY else "required_evidence_not_established",
                "run_id": candidate.get("development_run_id"),
                "snapshot_hash": candidate.get("snapshot_hash"),
                "contract_version": EVIDENCE_CONTRACT_VERSION,
            }
        )
    output = pd.DataFrame(rows, columns=columns)
    for index, candidate in enumerate(criteria.to_dict("records")):
        low, high = _number(candidate.get("roc_auc_ci_low")), _number(candidate.get("roc_auc_ci_high"))
        effect = _number(candidate.get("roc_auc"))
        output.loc[index, "effect_ci_passed"] = bool(np.isfinite(low) and np.isfinite(high) and 0 <= low <= effect <= high <= 1)
        output.loc[index, "reference_kind"] = str(candidate.get("reference_kind", "unverified_or_unknown"))
        output.loc[index, "independent_reference_passed"] = true(candidate.get("independent_reference_passed"))
        output.loc[index, "analysis_completeness"] = str(candidate.get("analysis_completeness", "unknown"))
        contract = scientific_result_contract(output.iloc[index].to_dict()).to_dict()
        for key, value in contract.items():
            output.loc[index, key] = value
        output.loc[index, "evidence_tier"] = contract["evidence_stage"]
        output.loc[index, "gate_status"] = ("all_internal_candidate_gates_passed" if contract["evidence_stage"] in {"guideline_candidate", "clinically_validated"} else "required_gates_missing")
        output.loc[index, "abstention_reason"] = contract["failed_gates"]
    return output


__all__ = [
    "EVIDENCE_CONTRACT_VERSION",
    "build_candidate_evidence_contract",
]
