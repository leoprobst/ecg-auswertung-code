"""Prespecified mortality test families, including unavailable hypotheses."""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def adjust_population_survival(frame: pd.DataFrame, *, marker_ids, incremental_marker_ids) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    if result.duplicated(["cohort", "marker_id"]).any():
        raise ValueError("Duplicate cohort mortality hypothesis")
    markers = tuple(dict.fromkeys((*marker_ids, "ecg_age_gap_years")))
    incremental = tuple(dict.fromkeys(incremental_marker_ids))
    if not set(result.marker_id) <= set(markers):
        raise ValueError("Unplanned mortality marker")
    families = (
        (markers, "status", "p_value", "q_value_cohort"),
        (incremental, "incremental_comparison_status", "incremental_p_value", "incremental_q_value_cohort"),
        (tuple(x for x in incremental if x != "ecg_age_gap_years"),
         "ecg_age_comparison_status", "ecg_age_incremental_p_value", "ecg_age_incremental_q_value_cohort"),
    )
    for planned, status, p_column, q_column in families:
        result[q_column] = np.nan
        result[q_column + "_planned_hypotheses"] = len(planned)
        p = pd.to_numeric(result.get(p_column, pd.Series(np.nan, index=result.index)), errors="coerce")
        estimable = result.get(status, pd.Series("not_requested", index=result.index)).eq("estimable")
        if (estimable & ~result.marker_id.isin(planned)).any():
            raise ValueError("Estimated mortality comparison outside its planned family")
        valid = estimable & result.marker_id.isin(planned) & np.isfinite(p) & p.between(0, 1)
        for _, index in result.loc[valid].groupby("cohort").groups.items():
            # Padding contributes to multiplicity only. It is never exported as
            # an invented p/q value for an unavailable model.
            padded = np.r_[p.loc[index].to_numpy(), np.ones(len(planned)-len(index))]
            result.loc[index, q_column] = multipletests(padded, method="fdr_bh")[1][:len(index)]
    result["mortality_multiplicity_scope"] = "prespecified_markers_within_cohort_and_model_family_not_projectwide"
    return result
