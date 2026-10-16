"""Multiplicity for a fixed ratio-association family, never AUC-gain claims."""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .pathology_groups import pathology_group_for_endpoint
from .voltage_mass import VOLTAGE_MARKERS
from .voltage_mass_endpoint import VOLTAGE_MASS_ENDPOINTS


def adjust_voltage_mass_confirmation(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    columns = ("confirmation_q_global_bh", "confirmation_q_domain_bh", "confirmation_q_global_by")
    for column in columns:
        result[column] = np.nan
    if result.empty:
        return result
    required = {"endpoint_id", "voltage_marker", "model", "confirmation_status", "confirmation_p_value"}
    if not required <= set(result):
        raise ValueError("Missing ratio confirmation contract fields")
    if result.duplicated(["endpoint_id", "voltage_marker", "model"]).any():
        raise ValueError("Duplicate ratio hypotheses")
    if not set(result.endpoint_id) <= set(VOLTAGE_MASS_ENDPOINTS) or not set(result.voltage_marker) <= set(VOLTAGE_MARKERS):
        raise ValueError("Unplanned ratio hypothesis")
    target = result.model.eq("joint_quadratic_plus_ratio")
    if (result.confirmation_status.eq("estimable") & ~target).any():
        raise ValueError("Association test not planned for this model")
    p = pd.to_numeric(result.confirmation_p_value, errors="coerce")
    valid = target & result.confirmation_status.eq("estimable") & np.isfinite(p) & p.between(0, 1)
    result["confirmation_domain"] = [pathology_group_for_endpoint(x).group_id for x in result.endpoint_id]
    planned = len(VOLTAGE_MASS_ENDPOINTS) * len(VOLTAGE_MARKERS)

    def assign(indices, count, method, column):
        indices = list(indices)
        if len(indices) > count:
            raise ValueError("More tests than prespecified family")
        if indices:
            corrected = multipletests(np.r_[p.loc[indices], np.ones(count-len(indices))], method=method)[1]
            result.loc[indices, column] = corrected[:len(indices)]

    assign(result.index[valid], planned, "fdr_bh", columns[0])
    assign(result.index[valid], planned, "fdr_by", columns[2])
    for domain, group in result.groupby("confirmation_domain"):
        count = len(VOLTAGE_MARKERS) * sum(pathology_group_for_endpoint(x).group_id == domain for x in VOLTAGE_MASS_ENDPOINTS)
        assign(group.index[valid.loc[group.index]], count, "fdr_bh", columns[1])
    result["confirmation_planned_hypotheses_global"] = planned
    result["confirmation_multiplicity_scope"] = "voltage_mass_conditional_association_only_not_project_wide_or_auc_gain"
    return result
