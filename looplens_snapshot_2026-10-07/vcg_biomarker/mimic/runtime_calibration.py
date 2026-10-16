"""Attempt eligibility for runtime planning, not proof of workload equivalence."""
from __future__ import annotations

import math
import statistics
import re
from typing import Any, Mapping


FEATURE_TIMING_HASH_FIELDS = (
    "implementation_sha256", "scientific_config_sha256", "panel_sha256",
    "input_population_sha256", "sampling_design_sha256", "resources_sha256",
)


def build_feature_timing_contract(stage_inputs: Mapping[str, Any], *,
                                  scientific_config_sha256: str,
                                  scheduler_sha256: str,
                                  resource_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Describe an actual selected development cohort without rereading signals.

    The existing content-bound cohort hash is reused. Consequently different
    selected populations cannot masquerade as matched scaling samples. A future
    representative-population benchmark must supply its own verified design.
    """
    from .storage import content_hash
    if stage_inputs.get("locked_test_only") is not False:
        raise ValueError("Development timing contracts require an explicit unlocked scope")
    for value in (scientific_config_sha256, scheduler_sha256, stage_inputs.get("feature_code_hash"),
                  stage_inputs.get("panel_hash"), stage_inputs.get("cohort_selection")):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Incomplete feature timing identity")
    contract = {
        "schema_version": "1.0.0", "stage": "extract-features",
        "implementation_sha256": content_hash({"features": stage_inputs["feature_code_hash"],
                                              "scheduler": scheduler_sha256}),
        "scientific_config_sha256": scientific_config_sha256,
        "panel_sha256": stage_inputs["panel_hash"],
        "input_population_sha256": stage_inputs["cohort_selection"],
        "sampling_design_sha256": content_hash({key: stage_inputs.get(key) for key in
            ("max_records", "capped_sampling_version", "locked_test_only", "freeze_hash")}),
        "resources_sha256": content_hash({"execution": dict(resource_payload),
            "partition_size": stage_inputs.get("partition_size"),
            "sensitivity_vcg_methods": stage_inputs.get("sensitivity_vcg_methods")}),
        "population_scope": "exact_selected_record_payloads_not_unverified_parent_population",
        "resources_scope": "configured_policy_and_machine_not_guaranteed_future_contention",
    }
    if not feature_timing_contracts_match(contract, contract):
        raise ValueError("Incomplete feature timing identity")
    return contract


def feature_timing_contracts_match(reference: object, target: object) -> bool:
    """Fail closed until both prospective and observed workloads are identified.

    Record count is intentionally outside the identity: this permits linear
    planning extrapolation, not a calibrated interval or a bound. Population,
    selection design, methods and execution resources must still match.
    """
    if not isinstance(reference, Mapping) or not isinstance(target, Mapping):
        return False
    for contract in (reference, target):
        if contract.get("schema_version") != "1.0.0" or contract.get("stage") != "extract-features":
            return False
        for field in FEATURE_TIMING_HASH_FIELDS:
            value = contract.get(field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                return False
    return all(reference[field] == target[field] for field in FEATURE_TIMING_HASH_FIELDS)


def nonreuse_attempt_medians(runtime: Mapping[str, Any]) -> dict[str, float]:
    """Use successful attempts explicitly logged as not reusing artifacts.

    Missing cache evidence, failed work, summed effort and cache hits never
    become compute-time priors. This still does not establish that source,
    panel, sampling, hardware or internal checkpoint state match a future run.
    """
    attempts = runtime.get("stage_nonreuse_reported_attempt_seconds")
    if not isinstance(attempts, Mapping):
        return {}
    result: dict[str, float] = {}
    for stage, values in attempts.items():
        if not isinstance(values, list):
            continue
        valid = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(value) and value > 0:
                valid.append(float(value))
        if valid:
            result[str(stage)] = statistics.median(valid)
    return result
