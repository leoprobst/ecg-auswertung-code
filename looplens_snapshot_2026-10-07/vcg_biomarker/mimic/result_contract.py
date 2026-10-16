"""Fail-closed scientific presentation, independent of statistical strength."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Any

import numpy as np

RESULT_CONTRACT_VERSION = "1.2.0"
DOCUMENTATION_REFERENCE_ENDPOINTS = frozenset({
    "regional_wall_motion_documentation_proxy",
    "regional_wall_motion_abnormality_explicit",
})
UNQUALIFIED_DIASTOLIC_ENDPOINTS = frozenset({
    "diastolic_relaxation_abnormality", "restrictive_filling_pattern",
    "elevated_filling_pressure_proxy",
})
CANDIDATE_GATES = ("analytical_validity", "availability", "repeatability",
                   "replication", "adjustment", "incremental_classical_value",
                   "transform_robustness", "subgroup_stability", "effect_ci",
                   "independent_reference")
CLINICAL_GATES = ("prespecified_threshold", "calibration", "net_benefit", "external_validation")


def true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


@dataclass(frozen=True)
class ScientificResultContract:
    evidence_stage: str
    reference_kind: str
    claim_ceiling: str
    analysis_scope: str
    failed_gates: str
    clinical_claim_allowed: bool
    contract_version: str = RESULT_CONTRACT_VERSION
    independent_modality_observed_n: int | None = None
    clock_alignment_confirmed_n: int | None = None
    timing_evidence_note: str = "not_reported"

    def to_dict(self) -> dict:
        return asdict(self)


def scientific_result_contract(row: Mapping[str, Any]) -> ScientificResultContract:
    def observed_count(name):
        value = row.get(name)
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return int(number) if np.isfinite(number) and number >= 0 and number.is_integer() else None
    modality_n = observed_count("n_independent_modality_observed")
    clock_n = observed_count("n_reference_alignment_confirmed")
    gates = {name: true(row.get(f"{name}_passed")) for name in (*CANDIDATE_GATES, *CLINICAL_GATES)}
    reference = str(row.get("reference_kind", "unverified_or_unknown"))
    independent = reference in {"timed_quantitative", "blinded_adjudication", "independent_outcome"}
    documentation_only = str(row.get("endpoint_id", "")) in DOCUMENTATION_REFERENCE_ENDPOINTS
    if documentation_only:
        independent = False
        reference = "echo_documentation_proxy_not_disease_exclusion"
    gates["independent_reference"] = independent and true(row.get("independent_reference_passed"))
    echo_applicable = str(row.get("endpoint_id", "")) not in UNQUALIFIED_DIASTOLIC_ENDPOINTS
    if not echo_applicable:
        gates["echo_reference_applicability"] = False
    scope = str(row.get("analysis_completeness", row.get("analysis_scope", "unknown")))
    preview = scope not in {"complete_snapshot", "external_prespecified_evaluation"} or true(row.get("preview"))
    if not preview and all(gates.values()):
        stage = "clinically_validated"
    elif not preview and echo_applicable and all(gates[x] for x in CANDIDATE_GATES):
        stage = "guideline_candidate"
    elif not preview and gates["replication"] and gates["effect_ci"]:
        stage = "replicated_association"
    else:
        stage = "exploratory"
    ceiling = ("documentation_association_only" if documentation_only else
               "agreement_only" if reference.startswith("same_ecg") else
               "association_only" if not independent else "independent_reference_comparison")
    return ScientificResultContract(stage, reference, ceiling, scope,
                                    "|".join([key for key, passed in gates.items() if not passed]
                                             + (["preview_or_unknown_scope"] if preview else [])),
                                    stage == "clinically_validated",
                                    independent_modality_observed_n=modality_n,
                                    clock_alignment_confirmed_n=clock_n,
                                    timing_evidence_note=("not_reported" if clock_n is None else
                                        "no_clock_alignment_confirmed" if clock_n == 0 else
                                        "some_clock_alignments_confirmed_not_necessarily_entire_cohort"))


def legacy_display_grade(row: Mapping[str, Any]) -> str:
    if str(row.get("reference_independence")) == "same_ecg_technical_comparison":
        return "T"
    stage = scientific_result_contract(row).evidence_stage
    return {"clinically_validated": "A", "guideline_candidate": "B",
            "replicated_association": "C"}.get(stage, "D")
