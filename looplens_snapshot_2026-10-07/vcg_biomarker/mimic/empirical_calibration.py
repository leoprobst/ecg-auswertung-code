"""Empirical calibration of screen p-values with negative controls (audit 5.3.3).

Schuemie et al., Stat Med 2014 (PMID 23900808); PNAS 2018 (PMID 29531023).
Only p-values are calibrated here: interval calibration (2018) needs
synthetic positive controls injected into the data, which this screen does
not construct.
A multiplicity correction bounds chance; it does not touch systematic error.
In an intensive-care cohort the dominant systematic error is confounding by
acuity: a marker and a phenotype both track how ill someone is. The negative
control phenotypes (no path from cardiac electrophysiology) carry that error
and nothing else. Their estimates are modelled as

    estimate_i ~ Normal(mu, sigma^2 + se_i^2)

by maximum likelihood; a calibrated p-value then asks how surprising an
estimate is under *that* null instead of under exact zero.

Effect scale: the signed AUC distance from 0.5 (raw AUC - 0.5, direction
kept), standard error from Hanley & McNeil 1982. The calibration is reported
beside the unchanged screen; it decides nothing.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

EMPIRICAL_CALIBRATION_VERSION = "1.0.0"
MINIMUM_NEGATIVE_CONTROLS_PER_FEATURE = 5


def auc_standard_error(auc: np.ndarray, n_positive: np.ndarray, n_negative: np.ndarray) -> np.ndarray:
    """Hanley & McNeil (1982) standard error of an AUC."""

    auc = np.asarray(auc, dtype=float)
    n1 = np.asarray(n_positive, dtype=float)
    n2 = np.asarray(n_negative, dtype=float)
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = (
            auc * (1.0 - auc)
            + (n1 - 1.0) * (q1 - auc * auc)
            + (n2 - 1.0) * (q2 - auc * auc)
        ) / (n1 * n2)
    variance = np.where((n1 > 0) & (n2 > 0), variance, np.nan)
    return np.sqrt(np.clip(variance, 0.0, None))


def fit_systematic_error(
    estimates: Iterable[float], standard_errors: Iterable[float]
) -> tuple[float, float]:
    """Maximum-likelihood mean and SD of the systematic error distribution."""

    from scipy import optimize, stats

    theta = np.asarray(list(estimates), dtype=float)
    se = np.asarray(list(standard_errors), dtype=float)
    usable = np.isfinite(theta) & np.isfinite(se) & (se > 0)
    theta, se = theta[usable], se[usable]
    if len(theta) < 2:
        return float("nan"), float("nan")

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        mu, log_sigma = parameters
        scale = np.sqrt(np.exp(2.0 * log_sigma) + se * se)
        return -float(np.sum(stats.norm.logpdf(theta, loc=mu, scale=scale)))

    start = np.array([float(np.mean(theta)), np.log(max(float(np.std(theta)), 1e-4))])
    fit = optimize.minimize(negative_log_likelihood, start, method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    mu, log_sigma = fit.x
    return float(mu), float(np.exp(log_sigma))


def calibrated_p_values(
    estimates: np.ndarray, standard_errors: np.ndarray, mu: float, sigma: float
) -> np.ndarray:
    """Two-sided p-values under the empirical null Normal(mu, sigma^2 + se^2)."""

    from scipy import stats

    estimates = np.asarray(estimates, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    if not (np.isfinite(mu) and np.isfinite(sigma)):
        return np.full(len(estimates), np.nan)
    scale = np.sqrt(sigma * sigma + standard_errors * standard_errors)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (estimates - mu) / scale
    return 2.0 * stats.norm.sf(np.abs(z))


def calibrate_screen(
    results: pd.DataFrame,
    negative_control_names: Iterable[str],
    *,
    name_column: str = "phenotype_name",
    feature_column: str = "feature_id",
    auc_column: str = "raw_auc",
    positive_column: str = "n_case",
    negative_column: str = "n_control",
    minimum_per_feature: int = MINIMUM_NEGATIVE_CONTROLS_PER_FEATURE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add empirically calibrated p-values to a screen table.

    The null is fitted per marker from its negative-control rows when at
    least ``minimum_per_feature`` exist, otherwise from all negative-control
    rows of all markers pooled. Returns the table
    with the new columns, the per-marker null table and a summary.
    """

    output = results.copy()
    for column in ("empirical_null_mean", "empirical_null_sd",
                   "empirical_calibrated_p_value"):
        output[column] = np.nan
    output["empirical_null_scope"] = "not_estimable"
    if output.empty or auc_column not in output:
        return output, pd.DataFrame(), {"version": EMPIRICAL_CALIBRATION_VERSION,
                                        "status": "no_screen_rows"}
    estimate = pd.to_numeric(output[auc_column], errors="coerce").to_numpy() - 0.5
    se = auc_standard_error(
        estimate + 0.5,
        pd.to_numeric(output[positive_column], errors="coerce").to_numpy(),
        pd.to_numeric(output[negative_column], errors="coerce").to_numpy(),
    )
    names = set(str(item) for item in negative_control_names)
    is_control = output[name_column].astype(str).isin(names).to_numpy()
    pooled_mu, pooled_sigma = fit_systematic_error(estimate[is_control], se[is_control])
    null_rows: list[dict] = []
    codes, uniques = pd.factorize(output[feature_column].astype(str), sort=False)
    order = np.argsort(codes, kind="mergesort")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    null_mean = np.full(len(output), np.nan)
    null_sd = np.full(len(output), np.nan)
    calibrated = np.full(len(output), np.nan)
    scopes = np.full(len(output), "not_estimable", dtype=object)
    usable = np.isfinite(estimate) & np.isfinite(se)
    for rows in np.split(order, boundaries):
        if len(rows) == 0:
            continue
        control_rows = rows[is_control[rows] & usable[rows]]
        if len(control_rows) >= int(minimum_per_feature):
            mu, sigma = fit_systematic_error(estimate[control_rows], se[control_rows])
            scope = "per_marker"
        else:
            mu, sigma = pooled_mu, pooled_sigma
            scope = "pooled_all_markers" if np.isfinite(pooled_mu) else "not_estimable"
        null_mean[rows] = mu
        null_sd[rows] = sigma
        calibrated[rows] = calibrated_p_values(estimate[rows], se[rows], mu, sigma)
        scopes[rows] = scope
        null_rows.append({
            "feature_id": str(uniques[codes[rows[0]]]),
            "null_mean_auc_distance": mu, "null_sd": sigma,
            "n_negative_controls": int(len(control_rows)), "scope": scope,
        })
    output["empirical_null_mean"] = null_mean
    output["empirical_null_sd"] = null_sd
    output["empirical_calibrated_p_value"] = calibrated
    output["empirical_null_scope"] = scopes
    nulls = pd.DataFrame(null_rows)
    raw_p = pd.to_numeric(output.get("p_value"), errors="coerce") if "p_value" in output else None
    summary = {
        "version": EMPIRICAL_CALIBRATION_VERSION,
        "effect_scale": "raw_auc_minus_0.5_hanley_mcneil_se",
        "negative_control_rows": int(np.sum(is_control)),
        "negative_control_names_found": sorted(set(output.loc[is_control, name_column].astype(str))),
        "pooled_null_mean": pooled_mu,
        "pooled_null_sd": pooled_sigma,
        "markers_with_own_null": int((nulls["scope"] == "per_marker").sum()) if len(nulls) else 0,
        "share_raw_p_below_0.05": (
            float((raw_p < 0.05).mean()) if raw_p is not None and raw_p.notna().any() else None
        ),
        "share_calibrated_p_below_0.05": float(
            (output["empirical_calibrated_p_value"] < 0.05).mean()
        ),
        "claim_scope": "reported_beside_the_screen_decides_no_hit",
        "reference": "Schuemie et al. Stat Med 2014, PMID 23900808",
    }
    return output, nulls, summary


def negative_control_names(control_sets: Optional[Iterable] = None) -> tuple[str, ...]:
    from .prespecified_controls import NEGATIVE_CONTROLS

    return tuple(
        name for control_set in (control_sets or NEGATIVE_CONTROLS)
        for name in control_set.target_ids
    )
