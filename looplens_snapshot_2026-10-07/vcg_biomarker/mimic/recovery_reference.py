"""Observed laboratory normalization; no inferred healthy state or causal claim."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .patient_identity import known_patient_mask

RECOVERY_REFERENCE_VERSION = "1.1.0"


def attach_observed_normalization(episodes: pd.DataFrame, labs: pd.DataFrame, *,
                                  lab_group: str, baseline_window_hours: float = 6.) -> pd.DataFrame:
    """Use the evidence labs.parquet contract and its assay-specific intervals.

    Initial specimen is the latest at/before the independent event landmark.
    Do not skip a missing/invalid initial specimen to find an abnormal one.
    Follow-up must have the same itemid, unit and reference bounds. Different
    assays/specimens are not silently pooled. A first observed in-range value
    does not establish sustained normalization between or after measurements.
    """
    er = {"subject_id", "hadm_id", "episode_id", "landmark_time", "episode_end"}
    lr = {"subject_id", "hadm_id", "labevent_id", "itemid", "lab_group", "charttime",
          "storetime", "value_numeric", "unit", "reference_lower", "reference_upper", "value_status"}
    if not er <= set(episodes) or not lr <= set(labs):
        raise ValueError("Missing episode or standardized laboratory contract")
    if not np.isfinite(baseline_window_hours) or baseline_window_hours <= 0:
        raise ValueError("Positive finite baseline laboratory window required")
    if any(not known_patient_mask(episodes[c]).all() for c in ("subject_id", "hadm_id", "episode_id")):
        raise ValueError("Known episode/admission identities required")
    if episodes.duplicated(["subject_id", "episode_id"]).any():
        raise ValueError("Duplicate episode identity")
    working = labs.loc[labs.lab_group.eq(lab_group), sorted(lr)].copy()
    # Unlinkable labs remain outside the admission-scoped design, never linked
    # by patient identity alone across hospital stays.
    working = working.loc[known_patient_mask(working.subject_id) & known_patient_mask(working.hadm_id)]
    if not known_patient_mask(working.labevent_id).all() or working.labevent_id.duplicated().any():
        raise ValueError("Unknown or duplicate laboratory event identity")
    for c in ("charttime", "storetime"):
        working[c] = pd.to_datetime(working[c], errors="coerce")
    if working.charttime.isna().any():
        raise ValueError("Unresolved laboratory chart time")
    for c in ("value_numeric", "reference_lower", "reference_upper"):
        working[c] = pd.to_numeric(working[c], errors="coerce")
    grouped = {key: group.sort_values(["charttime", "labevent_id"], kind="stable")
               for key, group in working.groupby(["subject_id", "hadm_id"], sort=False)}
    output = []
    for row in episodes.to_dict("records"):
        result = dict(row, normalization_time=pd.NaT,
                      reference_normalization_status="not_evaluable",
                      reference_normalization_reason="no_baseline_lab_in_window",
                      reference_lab_group=lab_group, reference_version=RECOVERY_REFERENCE_VERSION,
                      reference_design="first_observed_in_range_same_assay_not_sustained_recovery",
                      reference_kind="independent_laboratory_measurement",
                      reference_observed_course="unknown",
                      reference_course_reason="baseline_not_verified",
                      reference_course_interpretation="retrospective_observations_not_continuous_stability_or_causal_controls",
                      clinical_claim_allowed=False, available_at_landmark=False)
        start, end = pd.to_datetime(row["landmark_time"], errors="coerce"), pd.to_datetime(row["episode_end"], errors="coerce")
        source = grouped.get((row["subject_id"], row["hadm_id"]))
        if pd.isna(start) or pd.isna(end) or start >= end:
            result["reference_normalization_reason"] = "invalid_episode_boundaries"
        elif source is not None:
            stamps = source.charttime.to_numpy()
            left = stamps.searchsorted((start-pd.Timedelta(hours=baseline_window_hours)).to_datetime64())
            right = stamps.searchsorted(start.to_datetime64(), side="right")
            initial = source.iloc[left:right]
            if not initial.empty:
                latest = initial.loc[initial.charttime.eq(initial.charttime.max())]
                # Two assays or disagreeing repeated results at the same time
                # do not license choosing whichever supports normalization.
                signature = ["itemid", "unit", "value_numeric", "reference_lower", "reference_upper", "value_status"]
                result["reference_normalization_reason"] = "ambiguous_baseline_lab"
                if len(latest[signature].drop_duplicates()) == 1:
                    baseline = latest.iloc[0]
                    result.update(reference_baseline_labevent_id=baseline.labevent_id,
                                  reference_baseline_time=baseline.charttime.isoformat())
                    values = baseline[["value_numeric", "reference_lower", "reference_upper"]].to_numpy(dtype=float, na_value=np.nan)
                    result["reference_normalization_reason"] = "invalid_baseline_value_unit_or_interval"
                    if (np.isfinite(values).all() and baseline.value_status == "usable"
                            and isinstance(baseline.unit, str) and baseline.unit.strip()
                            and baseline.reference_lower < baseline.reference_upper):
                        low, high = baseline.reference_lower, baseline.reference_upper
                        result.update(reference_itemid=baseline.itemid, reference_unit=baseline.unit,
                                      reference_lower=low, reference_upper=high,
                                      reference_baseline_value=baseline.value_numeric)
                        result["reference_normalization_reason"] = "baseline_already_in_range"
                        stop = stamps.searchsorted(end.to_datetime64())
                        post = source.iloc[right:stop]
                        compatible = (post.itemid.eq(baseline.itemid) & post.unit.eq(baseline.unit)
                            & post.reference_lower.eq(low) & post.reference_upper.eq(high)
                            & post.value_status.eq("usable") & np.isfinite(post.value_numeric))
                        result["reference_n_followup"] = len(post)
                        result["reference_n_compatible_followup"] = int(compatible.sum())
                        result.update(_observed_course(post, compatible, low=low, high=high,
                            baseline_value=float(baseline.value_numeric)))
                        if not low <= baseline.value_numeric <= high:
                            normal = post.loc[compatible & post.value_numeric.between(low, high)]
                            result["reference_normalization_reason"] = "no_observed_compatible_normalization"
                            if not normal.empty:
                                hit = normal.iloc[0]
                                at_time = post.loc[post.charttime.eq(hit.charttime) & compatible]
                                if at_time.value_numeric.nunique() > 1:
                                    result["reference_normalization_reason"] = "conflicting_normalization_measurements"
                                else:
                                    result.update(normalization_time=hit.charttime,
                                        reference_normalization_status="observed", reference_normalization_reason="",
                                        reference_normalization_labevent_id=hit.labevent_id,
                                        reference_normalization_value=hit.value_numeric,
                                        reference_result_available_at=(max(hit.charttime, hit.storetime).isoformat()
                                            if pd.notna(hit.storetime) else None))
        output.append(result)
    return pd.DataFrame(output)


def _observed_course(post: pd.DataFrame, compatible: pd.Series, *,
                     low: float, high: float, baseline_value: float) -> dict:
    """Describe available measurements, never infer unobserved stable controls.

    Require at least two distinct follow-up times and no incompatible or
    conflicting observations before classifying the full observed course.
    This is a prespecified descriptive QC rule, not a clinical stability rule.
    """
    result = dict(reference_observed_course="unknown", reference_course_reason="insufficient_followup_times",
                  reference_n_followup_times=int(post.charttime.nunique()))
    if post.empty:
        return result
    result.update(reference_followup_first_time=post.charttime.min().isoformat(),
                  reference_followup_last_time=post.charttime.max().isoformat(),
                  reference_followup_span_hours=(post.charttime.max()-post.charttime.min()).total_seconds()/3600)
    if not compatible.all():
        result["reference_course_reason"] = "incompatible_or_unusable_followup"
        return result
    if post.groupby("charttime").value_numeric.nunique().gt(1).any():
        result["reference_course_reason"] = "conflicting_followup_values"
        return result
    if post.charttime.nunique() < 2:
        return result
    in_range = post.value_numeric.between(low, high)
    result["reference_course_reason"] = ""
    if low <= baseline_value <= high:
        result["reference_observed_course"] = (
            "baseline_and_all_observed_followups_in_range" if in_range.all()
            else "baseline_in_range_followup_excursion_observed")
    elif not in_range.any():
        result["reference_observed_course"] = "baseline_abnormal_no_in_range_followup_observed"
    else:
        first_normal = post.loc[in_range, "charttime"].min()
        recurrence = (~in_range & post.charttime.gt(first_normal)).any()
        result["reference_observed_course"] = (
            "baseline_abnormal_in_range_then_excursion_observed" if recurrence
            else "baseline_abnormal_in_range_without_observed_recurrence")
    return result
