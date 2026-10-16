"""Exploratory rank-slope sensitivity, not causal or complete confounder control."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t

from .patient_identity import known_patient_mask

PAIRED_ADJUSTMENT_VERSION = "1.2.0"
TIMING_QUALITY_COVARIATES = ("hours_between", "signal_quality_first", "signal_quality_second")
BASELINE_CLINICAL_COVARIATES = (*TIMING_QUALITY_COVARIATES,
    "baseline_adjust_age", "baseline_adjust_sex_male", "baseline_adjust_heart_rate")


def adjusted_paired_rank_association(x, y, covariates, subjects, *, minimum_patients=30):
    """Rank x/y; linearly adjust standardized, explicitly supplied covariates.

    Patient-cluster CR1 covariance, t(G-1). CI is conditional on empirical ranks,
    not an exact Spearman interval. Constant/collinear controls are not dropped.
    """
    if type(minimum_patients) is not int or minimum_patients < 3:
        raise ValueError("At least three independent patient clusters required")
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(covariates, float)
    if x.ndim != 1 or y.shape != x.shape or z.ndim != 2 or len(z) != len(x) or z.shape[1] == 0:
        raise ValueError("Aligned one-dimensional outcomes and nonempty covariate matrix required")
    result = dict(adjusted_status="non_estimable", adjusted_abstention_reason="unknown_patient_identity",
        adjusted_rank_slope=np.nan, adjusted_ci_low=np.nan, adjusted_ci_high=np.nan,
        adjusted_se=np.nan, adjusted_p_value=np.nan, unadjusted_same_sample_rank_slope=np.nan,
        adjusted_n_input=len(x), adjusted_n_complete=0, adjusted_n_patients=0,
        adjusted_design_condition=np.nan,
        adjusted_method="standardized_empirical_rank_OLS_patient_cluster_CR1_t_Gminus1",
        adjusted_inference_limit="conditional_on_empirical_ranks_not_exact_spearman_not_causal")
    usable = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
    x, y, z = x[usable], y[usable], z[usable]
    n, p = len(x), z.shape[1]+2
    result["adjusted_n_complete"] = n
    if n <= p or n < 3:
        result["adjusted_abstention_reason"] = "insufficient_complete_pairs"
        return result
    if subjects is None:
        return result
    subjects = pd.Series(subjects).reset_index(drop=True)
    if len(subjects) != len(usable):
        raise ValueError("Patient vector must match pair count")
    subjects = subjects[usable]
    if not known_patient_mask(subjects).all():
        return result
    codes, patients = pd.factorize(subjects, sort=False)
    g = len(patients)
    result["adjusted_n_patients"] = g
    if g < max(minimum_patients, p+1):
        result["adjusted_abstention_reason"] = "too_few_independent_patients"
        return result
    if x.min() == x.max() or y.min() == y.max() or np.any(z.min(axis=0) == z.max(axis=0)):
        result["adjusted_abstention_reason"] = "constant_exposure_outcome_or_covariate"
        return result
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = (rx-rx.mean())/rx.std(), (ry-ry.mean())/ry.std()
    z = z/np.max(np.abs(z), axis=0)
    spread = z.std(axis=0)
    if np.any(spread == 0):
        result["adjusted_abstention_reason"] = "numerically_constant_covariate"
        return result
    z = (z-z.mean(axis=0))/spread
    design = np.column_stack((np.ones(n), rx, z))
    try:
        u, s, vt = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError:
        result["adjusted_abstention_reason"] = "svd_did_not_converge"
        return result
    condition = float(s[0]/s[-1]) if s[-1] > 0 else float("inf")
    result["adjusted_design_condition"] = condition
    if np.count_nonzero(s > s[0]*max(design.shape)*np.finfo(float).eps) != p or condition > 1e8:
        result["adjusted_abstention_reason"] = "collinear_or_ill_conditioned_design"
        return result
    coefficients = vt.T @ ((u.T @ ry)/s)
    residual = ry-design @ coefficients
    # Exposure row of the pseudoinverse, using the SAME SVD, no Gram inversion.
    influence = (u @ (vt[:, 1]/s))*residual
    scores = np.bincount(codes, weights=influence, minlength=g)
    variance = (g/(g-1))*((n-1)/(n-p))*float(scores @ scores)
    if not np.isfinite(variance) or variance <= np.finfo(float).eps**2:
        result["adjusted_abstention_reason"] = "degenerate_cluster_uncertainty"
        return result
    slope, se = float(coefficients[1]), float(np.sqrt(variance))
    critical = float(t.ppf(.975, g-1))
    result.update(adjusted_status="estimable", adjusted_abstention_reason="",
        adjusted_rank_slope=slope, adjusted_se=se, adjusted_ci_low=slope-critical*se,
        adjusted_ci_high=slope+critical*se, adjusted_p_value=float(2*t.sf(abs(slope/se), g-1)),
        unadjusted_same_sample_rank_slope=float(rx @ ry/(rx @ rx)))
    return result


def paired_timing_quality_sensitivity(changes, marker_ids, *, exposure_column):
    """A named partial sensitivity; never described as full clinical adjustment."""
    return _paired_sensitivity(changes, marker_ids, exposure_column=exposure_column,
        covariates=TIMING_QUALITY_COVARIATES, model="timing_quality_sensitivity_only")


def _paired_sensitivity(changes, marker_ids, *, exposure_column, covariates, model):
    rows = []
    absent = [name for name in (exposure_column, *covariates) if name not in changes]
    if not absent:
        x = pd.to_numeric(changes[exposure_column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        z = changes[list(covariates)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    for marker in marker_ids:
        column = f"delta_{marker}"
        if column not in changes:
            continue
        if absent:
            result = dict(adjusted_status="non_estimable", adjusted_abstention_reason="missing_covariates:"+",".join(absent),
                          adjusted_p_value=np.nan, adjusted_n_input=len(changes), adjusted_n_complete=0)
        else:
            result = adjusted_paired_rank_association(x,
                pd.to_numeric(changes[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan), z,
                changes.subject_id if "subject_id" in changes else None)
        rows.append(dict(feature_id=str(marker), **result,
            adjusted_model=model,
            adjusted_covariates="|".join(covariates),
            complete_time_varying_confounder_control=False))
    return pd.DataFrame(rows)


def paired_sensitivities(changes, marker_ids, *, exposure_column):
    """Two prespecified partial models, separate contracts and complete cases.

    First-ECG age/sex/HR complement timing/quality; no delta-HR/QRS mediator
    control. Same-admission interventions do not need a constant inpatient
    dummy. Acuity, indication, concurrent therapies and setting subtype remain
    uncontrolled: neither model qualifies as full clinical/causal adjustment.
    """
    markers = tuple(marker_ids)
    timing = paired_timing_quality_sensitivity(changes, markers, exposure_column=exposure_column)
    clinical = _paired_sensitivity(changes, markers, exposure_column=exposure_column,
        covariates=BASELINE_CLINICAL_COVARIATES, model="baseline_clinical_timing_quality_sensitivity")
    if timing.empty:
        return timing
    clinical = clinical.rename(columns={name: f"clinical_{name}" for name in clinical if name != "feature_id"})
    return timing.merge(clinical, on="feature_id", validate="one_to_one")


def add_all_paired_adjustment_fdr(frame):
    """BH within each prespecified sensitivity; not a best-of-models claim."""
    output = add_paired_adjustment_fdr(frame)
    for prefix in ("clinical_", "joint_"):
        names = [name for name in output if name.startswith(prefix)]
        if not names:
            continue
        sensitivity = output[["design", *names]].rename(columns={name: name.removeprefix(prefix) for name in names})
        sensitivity = add_paired_adjustment_fdr(sensitivity)
        for name in ("adjusted_design_q_value", "adjusted_global_q_value"):
            output[f"{prefix}{name}"] = sensitivity[name]
    return output


def add_paired_adjustment_fdr(frame):
    """Separate named sensitivity family, never replace primary association q."""
    from statsmodels.stats.multitest import multipletests
    output = frame.copy()
    output["adjusted_design_q_value"] = np.nan
    output["adjusted_global_q_value"] = np.nan
    if output.empty or not {"adjusted_p_value", "adjusted_status", "design"} <= set(output):
        return output
    p = pd.to_numeric(output.adjusted_p_value, errors="coerce")
    valid = output.adjusted_status.eq("estimable") & p.between(0, 1)
    for _, indices in output.loc[valid].groupby("design", sort=False).groups.items():
        output.loc[indices, "adjusted_design_q_value"] = multipletests(p.loc[indices], method="fdr_bh")[1]
    if valid.any():
        output.loc[valid, "adjusted_global_q_value"] = multipletests(p.loc[valid], method="fdr_bh")[1]
    return output
