"""Held-out conditional association, distinct from incremental discrimination."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, PerfectSeparationError

from .pathology_groups import pathology_group_for_endpoint
from .design_diagnostics import design_rank_condition
from ..core.data_loader import LEAD_NAMES_12

LEAD_CONFIRMATION_VERSION = "1.0.0"


def confirm_component(model: dict, train: np.ndarray, validation: np.ndarray,
                      labels: np.ndarray, n_covariates: int) -> dict:
    result = {"confirmation_status": "non_estimable", "confirmation_p_value": np.nan,
              "confirmation_reason": "", "confirmation_test": "validation_conditional_logit_HC0_Wald",
              "confirmation_design": "train_fixed_lead_component_validation_refitted_nuisance_not_prediction_refit",
              "confirmation_null": "zero_conditional_association_of_train_fixed_lead_component_not_zero_auc_gain"}
    try:
        active = np.asarray(model["active_columns"], dtype=int)
        coefficients = np.asarray(model["coefficients"], dtype=float)[1:]
        selected = active >= n_covariates
        indices = active[selected]
        means, scales = np.asarray(model["mean"])[indices], np.asarray(model["scale"])[indices]
        train_component = ((train[:, indices]-means)/scales) @ coefficients[selected]
        valid_component = ((validation[:, indices]-means)/scales) @ coefficients[selected]
        component_scale = float(train_component.std())
        if component_scale < 1e-10:
            raise ValueError("no_nonconstant_train_lead_component")
        # Clinical covariates are nuisance parameters, not reselected markers.
        clinical = validation[:, :n_covariates]
        clinical_scale = clinical.std(axis=0)
        varying = clinical_scale > 1e-10
        clinical = (clinical[:, varying]-clinical[:, varying].mean(axis=0))/clinical_scale[varying]
        design = np.column_stack([np.ones(len(validation)), clinical,
                                  (valid_component-train_component.mean())/component_scale])
        if min(np.sum(labels == 1), np.sum(labels == 0)) < 10*design.shape[1]:
            raise ValueError("insufficient_validation_events_per_parameter")
        rank, condition = design_rank_condition(design)
        if rank != design.shape[1] or condition > 1e8:
            raise ValueError("collinear_validation_design")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("error", RuntimeWarning)
            fitted = sm.GLM(labels, design, family=sm.families.Binomial()).fit(cov_type="HC0", maxiter=100)
        if (not fitted.converged or not np.isfinite(fitted.params).all()
                or not np.isfinite(fitted.bse).all() or fitted.bse[-1] <= 0
                or np.max(np.abs(fitted.params)) > 30 or not np.isfinite(fitted.pvalues[-1])):
            raise ValueError("unstable_validation_confirmation")
        low, high = np.asarray(fitted.conf_int())[-1]
        result.update(confirmation_status="estimable", confirmation_p_value=float(fitted.pvalues[-1]),
            confirmation_log_or_per_train_component_sd=float(fitted.params[-1]),
            confirmation_log_or_ci_low=float(low), confirmation_log_or_ci_high=float(high),
            confirmation_direction_preserved=bool(fitted.params[-1] > 0),
            confirmation_training_component_sd=component_scale)
    except (ValueError, RuntimeWarning, PerfectSeparationWarning, PerfectSeparationError, np.linalg.LinAlgError) as exc:
        result["confirmation_reason"] = str(exc) or type(exc).__name__
    return result


def adjust_lead_confirmation(frame: pd.DataFrame, *, planned_endpoints: tuple[str, ...]) -> pd.DataFrame:
    """BH global/domain plus global BY sensitivity, preserving planned failures.

    Non-estimable or absent tests count conservatively in the multiplicity
    denominator via internal p=1 padding; their reported p/q stay missing.
    This procedure never tests, ranks or licenses an AUC-gain claim.
    """
    result = frame.copy().reset_index(drop=True)
    for name in ("confirmation_q_global_bh", "confirmation_q_domain_bh", "confirmation_q_global_by"):
        result[name] = np.nan
    if result.empty:
        return result
    if result.duplicated(["endpoint_id", "family", "lead"]).any():
        raise ValueError("Duplicate lead confirmation hypotheses")
    if not result.family.isin(["p", "qrs", "st", "t"]).all() or not result.lead.isin(LEAD_NAMES_12).all():
        raise ValueError("Unknown prespecified lead or segment family")
    planned = tuple(dict.fromkeys(planned_endpoints))
    if not set(result.endpoint_id) <= set(planned):
        raise ValueError("Lead confirmation contains unplanned endpoints")
    result["confirmation_domain"] = [pathology_group_for_endpoint(x).group_id for x in result.endpoint_id]
    p = pd.to_numeric(result.get("confirmation_p_value", pd.Series(np.nan, index=result.index)), errors="coerce")
    estimable = result.get("confirmation_status", pd.Series("not_run", index=result.index)).eq("estimable")
    valid = estimable & np.isfinite(p) & p.between(0, 1)
    planned_count = 48*len(planned)  # 12 leads x four prespecified segment families

    def assign(indices, count, method, column):
        indices = list(indices)
        if len(indices) > count:
            raise ValueError("More hypotheses than the prespecified multiplicity family")
        if indices:
            adjusted = multipletests(np.r_[p.loc[indices].to_numpy(), np.ones(count-len(indices))], method=method)[1]
            result.loc[indices, column] = adjusted[:len(indices)]

    assign(result.index[valid], planned_count, "fdr_bh", "confirmation_q_global_bh")
    assign(result.index[valid], planned_count, "fdr_by", "confirmation_q_global_by")
    for domain, group in result.groupby("confirmation_domain"):
        count = 48*sum(pathology_group_for_endpoint(endpoint).group_id == domain for endpoint in planned)
        assign(group.index[valid.loc[group.index]], count, "fdr_bh", "confirmation_q_domain_bh")
    result["confirmation_planned_hypotheses_global"] = planned_count
    result["confirmation_multiplicity_scope"] = "conditional_association_not_discrimination_or_monitoring_utility"
    result["lead_confirmation_version"] = LEAD_CONFIRMATION_VERSION
    return result
