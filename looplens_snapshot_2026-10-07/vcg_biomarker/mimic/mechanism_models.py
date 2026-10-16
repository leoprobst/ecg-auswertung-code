"""Common-patient adjusted association comparisons, not clinical validation."""

import numpy as np
import pandas as pd

from .broad_axis_adjustment import fit_adjusted_association

VERSION = "1.0.0"


def compare_mechanism_adjustment(frame, *, target_kind, baseline_covariates,
                                 additional_covariates, numeric_covariates=()):
    """Consume an already reference-blind index cohort from one development split.

    No patient selection, external joins, hypothesis selection or clinical
    qualification occurs here. Caller must supply a qualified cohort.
    """
    covariates = tuple(dict.fromkeys((*baseline_covariates, *additional_covariates)))
    if set(numeric_covariates) - set(covariates):
        raise ValueError("Numeric covariates must belong to the comparison design")
    required = {"subject_id", "split", "marker_value", "target_value", *covariates}
    missing = sorted(required - set(frame.columns))
    result = {"version": VERSION, "status": "non_estimable", "reason": "",
              "n_index": len(frame), "n_common": 0, "models": [],
              "coefficient_change": None, "clinical_claim_allowed": False,
              "interpretation": "conditional_association_not_causal_or_incremental_prediction"}
    if missing:
        return {**result, "reason": "required_column_missing:" + "|".join(missing)}
    if frame.subject_id.isna().any() or frame.subject_id.duplicated().any():
        raise ValueError("Requires unique nonmissing reference-blind index patients")
    if len(frame) and (frame.split.nunique(dropna=False) != 1
                       or not frame.split.isin(["train", "validation"]).all()):
        raise ValueError("Requires one development split; locked or mixed cohorts forbidden")
    numeric = {"marker_value", "target_value", "age", "heart_rate_bpm",
               "signal_quality_score", "qrs_duration_ms", *numeric_covariates}
    working = frame.copy()
    complete = pd.Series(True, index=working.index)
    for name in ("marker_value", "target_value", *covariates):
        if name in numeric:
            working[name] = pd.to_numeric(working[name], errors="coerce")
            complete &= np.isfinite(working[name])
        else:
            working[name] = working[name].astype("string").str.strip().replace("", pd.NA)
            complete &= working[name].notna()
    common = working.loc[complete].reset_index(drop=True)
    result["n_common"] = len(common)
    for model_id, selected in (("baseline_common", baseline_covariates),
                               ("mechanism_adjusted_common", covariates)):
        fit = fit_adjusted_association(
            common, target_kind=target_kind, covariates=selected, model_id=model_id,
            numeric_covariates=[name for name in numeric_covariates if name in selected],
        )
        fit["numeric_covariates_requested"] = [name for name in numeric_covariates if name in selected]
        result["models"].append(fit)
    first, second = result["models"]
    if first["status"] == second["status"] == "estimable":
        result["status"] = "estimable"
        result["coefficient_change"] = second["estimate"] - first["estimate"]
    else:
        result["reason"] = "comparison_model_non_estimable"
    return result
