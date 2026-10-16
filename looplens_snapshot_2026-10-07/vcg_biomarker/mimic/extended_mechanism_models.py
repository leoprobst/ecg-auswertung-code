"""Exploratory models with explicit measurement and development-role contracts."""
import numpy as np
import pandas as pd

VERSION = "1.0.0"


def abstain(reason, **metadata):
    return {"status": "non_estimable", "reason": reason, "version": VERSION,
            "clinical_claim_allowed": False, **metadata}


def pr_geometry_coupling(pr_ms, geometry, *, pr_error_variance_ms2=None,
                        error_covariance=None):
    """Corrected covariance/variance slope; requires externally quantified error.

    Independent, zero-mean additive errors and known PR/geometry error covariance
    are assumptions, not facts inferred from observed scatter. No p-value or CI
    is invented for uncertain error estimates. At least 20 qualified paired beats.
    Caller must qualify rhythm, timing and comparable measurement definitions.
    """
    x, y = np.asarray(pr_ms, float), np.asarray(geometry, float)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 1 or y.ndim != 2 or len(x) != len(y) or y.shape[1] == 0:
        raise ValueError("Paired one-dimensional PR and geometry matrix required")
    if len(x) < 20 or not np.isfinite(x).all() or not np.isfinite(y).all() or (x <= 0).any():
        return abstain("insufficient_or_invalid_qualified_pairs")
    if pr_error_variance_ms2 is None or error_covariance is None:
        return abstain("independent_measurement_error_contract_missing", n=len(x))
    ev = np.asarray(pr_error_variance_ms2, float)
    ec = np.asarray(error_covariance, float)
    if ev.ndim != 0 or not np.isfinite(ev) or ev < 0 or ec.shape != (y.shape[1],) or not np.isfinite(ec).all():
        return abstain("invalid_measurement_error_contract")
    xc, yc = x - x.mean(), y - y.mean(axis=0)
    observed = float(xc @ xc / (len(x) - 1))
    corrected = observed - float(ev)
    if corrected <= max(1e-8, observed * 1e-6):
        return abstain("PR_variation_not_resolved_above_measurement_error", n=len(x))
    cov = xc @ yc / (len(x) - 1)
    return {"status": "descriptive", "reason": "", "version": VERSION,
            "n": len(x), "slope_per_ms": ((cov - ec) / corrected).tolist(),
            "naive_slope_per_ms": (cov / observed).tolist(),
            "corrected_PR_variance_ms2": corrected,
            "clinical_claim_allowed": False, "error_model_validated": False,
            "interpretation": "conditional_error_corrected_association_not_pathway_diagnosis"}


def electromechanical_residuals(frame, *, predictors, mechanical):
    """Fit on train patients, whiten on disjoint train calibration, apply frozen.

    One preselected record per patient; validation never determines coefficients,
    scaling or residual covariance. No disease labels, cutoffs or p-values.
    Returns signed whitened components as well as their norm.
    """
    required = {"subject_id", "split", *predictors, *mechanical}
    if not predictors or not mechanical or set(predictors) & set(mechanical):
        raise ValueError("Nonempty disjoint electrical/context and mechanical columns required")
    if len(set(predictors)) != len(predictors) or len(set(mechanical)) != len(mechanical):
        raise ValueError("Duplicate model columns")
    if not required <= set(frame):
        return pd.DataFrame(), abstain("required_measurements_missing")
    if frame.subject_id.isna().any() or frame.subject_id.duplicated().any():
        raise ValueError("Residual model requires one reference-blind record per patient")
    if not frame.split.isin(["train", "validation"]).all():
        raise ValueError("Residual model forbids locked or unknown splits")
    numeric = frame[[*predictors, *mechanical]].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric.to_numpy()).all(axis=1)
    # Stable patient hash, not input order or observed outcome, selects roles.
    hashes = pd.util.hash_pandas_object(frame.subject_id.astype(str), index=False).to_numpy()
    fit_mask = frame.split.eq("train").to_numpy() & (hashes % 2 == 0) & complete
    cal_mask = frame.split.eq("train").to_numpy() & (hashes % 2 == 1) & complete
    minimum = max(30, 5 * (len(predictors) + 1), 5 * len(mechanical))
    if min(fit_mask.sum(), cal_mask.sum()) < minimum:
        return pd.DataFrame(), abstain("insufficient_independent_fit_or_calibration_patients")
    x = numeric[list(predictors)].to_numpy(float)
    y = numeric[list(mechanical)].to_numpy(float)
    mean, scale = x[fit_mask].mean(axis=0), x[fit_mask].std(axis=0)
    if (scale <= 1e-12).any():
        return pd.DataFrame(), abstain("constant_predictor")
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    if np.linalg.cond(design[fit_mask]) > 1e8:
        return pd.DataFrame(), abstain("ill_conditioned_electrical_design")
    beta = np.linalg.lstsq(design[fit_mask], y[fit_mask], rcond=None)[0]
    residual = y[complete] - design[complete] @ beta
    calibration = y[cal_mask] - design[cal_mask] @ beta
    covariance = np.atleast_2d(np.cov(calibration, rowvar=False, ddof=1))
    values, vectors = np.linalg.eigh(covariance)
    if values.min() <= max(1e-12, values.max() * 1e-8):
        return pd.DataFrame(), abstain("singular_residual_covariance")
    whiten = (vectors * (1 / np.sqrt(values))) @ vectors.T
    transformed = residual @ whiten
    out = frame[["subject_id", "split"]].copy()
    out["role"] = np.where(fit_mask, "model_fit", np.where(cal_mask, "covariance_calibration", "validation"))
    out.loc[~complete, "role"] = "measurement_missing"
    out["status"] = np.where(complete, "descriptive", "non_estimable")
    for i, name in enumerate(mechanical):
        out[name + "_residual"] = np.nan
        out[name + "_whitened_component"] = np.nan
        out.loc[complete, name + "_residual"] = residual[:, i]
        out.loc[complete, name + "_whitened_component"] = transformed[:, i]
    out["residual_norm"] = np.nan
    out.loc[complete, "residual_norm"] = np.linalg.norm(transformed, axis=1)
    out["clinical_claim_allowed"] = False
    return out, {"status": "descriptive", "version": VERSION, "clinical_claim_allowed": False,
        "predictors": list(predictors), "mechanical": list(mechanical), "beta": beta.tolist(),
        "predictor_mean": mean.tolist(), "predictor_scale": scale.tolist(),
        "residual_covariance": covariance.tolist(), "whitening_matrix": whiten.tolist(),
        "n_fit": int(fit_mask.sum()), "n_calibration": int(cal_mask.sum()),
        "interpretation": "cross_modal_discordance_not_independent_echo_diagnosis",
        "role_policy": "pandas_patient_string_hash_parity_train_only"}


def within_between_echo_model(frame):
    """Mundlak-style descriptive decomposition, CR1 patient-clustered OLS.

    Input: one qualified measurement per patient/ECG, marker_value and echo_value.
    No time-varying confounding correction or causal interpretation is implied.
    """
    import statsmodels.api as sm
    frame = frame.reset_index(drop=True)
    needed = {"subject_id", "study_id", "split", "marker_value", "echo_value"}
    if not needed <= set(frame):
        return abstain("required_columns_missing")
    if frame.subject_id.isna().any() or frame.duplicated(["subject_id", "study_id"]).any():
        raise ValueError("Unique known patient/ECG identities required")
    if len(frame) and (frame.split.nunique() != 1 or not frame.split.isin(["train", "validation"]).all()):
        raise ValueError("One development split required")
    values = frame[["marker_value", "echo_value"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values).all(axis=1)
    data = frame.loc[valid, ["subject_id"]].join(values.loc[valid])
    data = data.loc[data.groupby("subject_id").subject_id.transform("size").ge(2)]
    groups = data.subject_id.nunique()
    if groups < 30:
        return abstain("insufficient_repeated_patients", n_patients=groups)
    mean = data.groupby("subject_id").marker_value.transform("mean")
    design = np.column_stack([np.ones(len(data)), data.marker_value - mean, mean])
    if np.linalg.cond(design) > 1e8:
        return abstain("within_between_design_not_estimable")
    fit = sm.OLS(data.echo_value.to_numpy(), design).fit(cov_type="cluster",
        cov_kwds={"groups": data.subject_id.to_numpy(), "use_correction": True}, use_t=True)
    if not np.isfinite(np.r_[fit.params, fit.bse, fit.pvalues]).all():
        return abstain("nonfinite_clustered_estimate")
    ci = fit.conf_int()
    return {"status": "estimable", "version": VERSION, "clinical_claim_allowed": False,
        "n": len(data), "n_patients": groups, "terms": [{"term": name,
        "estimate": float(fit.params[i]), "ci_low": float(ci[i, 0]), "ci_high": float(ci[i, 1]),
        "p_value": float(fit.pvalues[i])} for i, name in ((1, "within_patient"), (2, "between_patient"))],
        "interpretation": "unadjusted_within_between_association_not_causal"}
