"""Post-outcome mechanism screening: actual common-cohort sensitivity fits.

These models do not discharge the registry's anatomy/rhythm/interaction gates.
No fitting on enriched pilots and no clinical promotion or marker selection.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .lead_endpoint_analysis import LEAD_COVARIATES, prepare_lead_frame
from .mechanism_models import compare_mechanism_adjustment

VERSION = "1.3.0"
INPUT_COLUMNS = ("context_heart_rate_bpm", "signal_quality_score",
                 "extraction_status", "qrs_duration_ms")


def analyze_mechanism_endpoint(universe, *, hypotheses, preview):
    """Consume the unchanged, reference-blind endpoint index, never reselect it."""
    if not hypotheses:
        return []
    required = {"subject_id", "split", "eligibility_state", "label_state", "label_id"}
    if not required <= set(universe):
        raise ValueError("Mechanism universe lacks patient/split/reference contract")
    endpoints = {row["endpoint_id"] for row in hypotheses}
    identities = {(row["marker_id"], row["endpoint_id"]) for row in hypotheses}
    if len(endpoints) != 1 or len(identities) != len(hypotheses):
        raise ValueError("Mechanism hypotheses must be unique and belong to one endpoint")
    if not universe.label_id.isin(endpoints).all():
        raise ValueError("Mechanism hypothesis and reference endpoint mismatch")
    if universe.subject_id.isna().any() or universe.subject_id.duplicated().any():
        raise ValueError("Mechanism universe requires unique index patients")
    if not universe.split.isin(["train", "validation"]).all():
        raise ValueError("Mechanism analysis forbids locked or unknown splits")
    frame = prepare_lead_frame(universe)
    eligible = (frame.eligibility_state.eq("eligible")
                & frame.label_state.isin(["positive", "negative"])
                & frame.extraction_status.eq("ok"))
    frame["target_value"] = frame.label_state.map({"positive": 1., "negative": 0.})
    output = []
    for hypothesis in hypotheses:
        marker = hypothesis["marker_id"]
        for split in ("train", "validation"):
            split_mask = frame.split.eq(split)
            selected = frame.loc[split_mask & eligible].copy()
            selected["marker_value"] = (pd.to_numeric(selected[marker], errors="coerce")
                                        if marker in selected else np.nan)
            result = ({"status": "non_estimable", "reason": "pilot_no_confirmation",
                       "n_index": len(selected), "n_common": 0, "models": [],
                       "coefficient_change": None}
                      if preview else compare_mechanism_adjustment(
                          selected, target_kind="binary",
                          baseline_covariates=LEAD_COVARIATES,
                          additional_covariates=("qrs_duration_ms",),
                          numeric_covariates=(*LEAD_COVARIATES, "qrs_duration_ms")))
            models = result["models"] or [
                {"model_id": name, "status": "non_estimable", "reason": result["reason"],
                 "p_value": np.nan, "n": result["n_common"]}
                for name in ("baseline_common", "mechanism_adjusted_common")]
            for fit in models:
                output.append({
                    **fit, "feature_id": marker, "endpoint_id": hypothesis["endpoint_id"],
                    "split": split, "n_index_universe": int(split_mask.sum()),
                    "n_reference_and_extraction_eligible": len(selected),
                    "n_common": result["n_common"],
                    "comparison_status": result["status"],
                    "comparison_reason": result["reason"],
                    "coefficient_change": result["coefficient_change"],
                    "comparison_id": "primary_vs_qrs_duration_common_cohort",
                    "adapter_version": VERSION,
                    "hypothesis_version": hypothesis["hypothesis_version"],
                    "research_origins_json": json.dumps(hypothesis.get("research_origins",
                        [hypothesis["origin"]] if hypothesis.get("origin") else []), sort_keys=True),
                    "reference_target": hypothesis["reference_target"],
                    "reference_limitations_json": json.dumps(hypothesis.get("reference_limitations", {}), sort_keys=True),
                    "measured_near_alias_json": json.dumps(hypothesis.get("measured_near_alias", {}), sort_keys=True),
                    "analysis_temporality": "cross_sectional_not_ecg_time_prediction",
                    "evidence_tier": "exploratory", "post_outcome_hypothesis": True,
                    "clinical_claim_allowed": False,
                    "mechanism_requirements_fulfilled": False,
                    "remaining_requirements_json": json.dumps({
                        "comparisons": hypothesis.get("required_comparisons", []),
                        "measurement_context": hypothesis.get("required_measurement_context", [])},
                        sort_keys=True),
                    "interpretation": "conditional_association_not_causal_or_incremental_prediction",
                    "patient_index_policy": "unchanged_reference_blind_endpoint_index",
                    "sampling_scope": "pilot_enriched_descriptive" if preview else "development_index_cohort",
                    "care_setting_reference": "unlinked_not_proven_outpatient",
                })
    return output


def adjust_mechanism_family(frame, *, planned_hypotheses):
    """One declared family: all hypotheses x two splits x two fitted models.

Uncomputed tests retain their places (p=1 internally), never reduce multiplicity.
Reported q remains missing for abstentions; no replication claim is inferred.
"""
    result = frame.copy()
    result["q_value"] = np.nan
    family_size = 4 * planned_hypotheses
    result["fdr_family"] = "post_outcome_mechanism_primary_and_qrs_sensitivity"
    result["fdr_planned_tests"] = family_size
    if result.empty:
        return result
    keys = ["feature_id", "endpoint_id", "split", "model_id"]
    if result.duplicated(keys).any() or len(result) > family_size:
        raise ValueError("Duplicate or undeclared mechanism tests")
    p = pd.to_numeric(result.p_value, errors="coerce").to_numpy(dtype=float)
    valid = result.status.eq("estimable").to_numpy() & np.isfinite(p) & (p >= 0) & (p <= 1)
    from statsmodels.stats.multitest import multipletests
    padded = np.ones(family_size)
    padded[:len(p)] = np.where(valid, p, 1.)
    q = multipletests(padded, method="fdr_bh")[1][:len(p)]
    result["q_value"] = np.where(valid, q, np.nan)
    return result
