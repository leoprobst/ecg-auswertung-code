"""Prespecified adjusted confirmation for replicated broad-axis findings."""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BROAD_AXIS_ADJUSTMENT_VERSION = "1.3.0"
PRIMARY_COVARIATES: tuple[str, ...] = (
    "age",
    "sex",
    "heart_rate_bpm",
    "signal_quality_score",
    "care_setting",
)
SENSITIVITY_COVARIATES: tuple[str, ...] = (
    *PRIMARY_COVARIATES,
    "rhythm_context",
    "qrs_duration_ms",
)
MINIMUM_ADJUSTED_N = 60
MINIMUM_BINARY_CLASS = 30
MAXIMUM_DESIGN_CONDITION = 1e8


def _empty_result(
    *,
    target_kind: str,
    model_id: str,
    covariates: Sequence[str],
    status: str,
    reason: str,
    n: int = 0,
    n_events: int = 0,
    covariates_used: Sequence[str] = (),
    covariates_invariant: Sequence[str] = (),
) -> dict:
    return {
        "target_kind": target_kind,
        "model_id": model_id,
        "covariates": "|".join(covariates),
        "covariates_requested": "|".join(covariates),
        "covariates_used": "|".join(covariates_used),
        "covariates_invariant": "|".join(covariates_invariant),
        "n": int(n),
        "n_events": int(n_events),
        "estimate": np.nan,
        "effect": np.nan,
        "standard_error": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "effect_scale": (
            "odds_ratio_per_marker_sd"
            if target_kind == "binary"
            else "standardized_beta"
        ),
        "status": status,
        "reason": reason,
        "adjustment_version": BROAD_AXIS_ADJUSTMENT_VERSION,
    }


def _design_matrix(
    frame: pd.DataFrame,
    *,
    marker_column: str,
    covariates: Sequence[str],
    numeric_covariates: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str], list[str], list[str], pd.Series]:
    numeric = {
        "age",
        "heart_rate_bpm",
        "signal_quality_score",
        "qrs_duration_ms",
    }
    numeric.update(numeric_covariates)
    design = pd.DataFrame(index=frame.index)
    used: list[str] = []
    invariant: list[str] = []
    unavailable: list[str] = []
    complete = pd.Series(True, index=frame.index, dtype=bool)
    marker = pd.to_numeric(frame[marker_column], errors="coerce")
    complete &= marker.notna()
    marker_sd = float(marker.std(ddof=0))
    if not np.isfinite(marker_sd) or marker_sd <= 0:
        return pd.DataFrame(index=frame.index), used, invariant, unavailable, complete
    design["marker_z"] = (marker - float(marker.mean())) / marker_sd
    for name in covariates:
        if name not in frame.columns:
            unavailable.append(name)
            continue
        if name in numeric:
            values = pd.to_numeric(frame[name], errors="coerce")
            complete &= values.notna()
            if not values.notna().any():
                unavailable.append(name)
                continue
            sd = float(values.std(ddof=0))
            if not np.isfinite(sd) or sd <= 0:
                invariant.append(name)
                continue
            design[name] = (values - float(values.mean())) / sd
            used.append(name)
            continue
        values = frame[name].astype("string").str.strip().replace("", pd.NA)
        complete &= values.notna()
        if not values.notna().any():
            unavailable.append(name)
            continue
        dummies = pd.get_dummies(
            values,
            prefix=name,
            drop_first=True,
            dtype=float,
        )
        if dummies.shape[1]:
            design = pd.concat([design, dummies], axis=1)
            used.append(name)
        else:
            invariant.append(name)
    return design, used, invariant, unavailable, complete


def fit_adjusted_association(
    frame: pd.DataFrame,
    *,
    marker_column: str = "marker_value",
    target_column: str = "target_value",
    target_kind: str,
    covariates: Sequence[str] = PRIMARY_COVARIATES,
    model_id: str = "primary",
    numeric_covariates: Sequence[str] = (),
) -> dict:
    """Fit one robust adjusted association or return a typed abstention."""

    if target_kind not in {"binary", "continuous"}:
        raise ValueError(f"Unsupported target kind: {target_kind}")
    if set(numeric_covariates) - set(covariates):
        raise ValueError("Numeric covariates must be declared in covariates")
    required = [marker_column, target_column]
    if any(name not in frame.columns for name in required):
        return _empty_result(
            target_kind=target_kind,
            model_id=model_id,
            covariates=covariates,
            status="non_estimable",
            reason="required_column_missing",
        )
    working = frame.copy()
    # One deterministic cross-sectional index ECG per subject. Repeated ECGs
    # belong to the dedicated longitudinal design, not this confirmation.
    if "subject_id" in working.columns:
        order = [name for name in ("subject_id", "ecg_time", "study_id") if name in working]
        working = working.sort_values(order, kind="mergesort").drop_duplicates(
            "subject_id", keep="first"
        )
    target = pd.to_numeric(working[target_column], errors="coerce")
    design, used, invariant, unavailable, covariate_complete = _design_matrix(
        working,
        marker_column=marker_column,
        covariates=covariates,
        numeric_covariates=numeric_covariates,
    )
    if "marker_z" not in design:
        return _empty_result(
            target_kind=target_kind,
            model_id=model_id,
            covariates=covariates,
            covariates_used=used,
            covariates_invariant=invariant,
            status="non_estimable",
            reason="marker_has_no_variation",
        )
    if unavailable:
        return _empty_result(
            target_kind=target_kind,
            model_id=model_id,
            covariates=covariates,
            covariates_used=used,
            covariates_invariant=invariant,
            status="non_estimable",
            reason="required_covariate_unavailable:" + "|".join(unavailable),
        )
    complete = (
        target.notna()
        & covariate_complete
        & np.isfinite(design).all(axis=1)
    )
    target = target.loc[complete].astype(float)
    design = design.loc[complete].astype(float)
    n = len(target)
    n_events = int(target.sum()) if target_kind == "binary" else n
    if n < MINIMUM_ADJUSTED_N:
        return _empty_result(
            target_kind=target_kind,
            model_id=model_id,
            covariates=covariates,
            covariates_used=used,
            covariates_invariant=invariant,
            status="non_estimable",
            reason="insufficient_complete_cases",
            n=n,
            n_events=n_events,
        )
    # Build coding/scaling on the actual complete-case fit population. A
    # category present only among excluded outcomes otherwise leaves an empty
    # dummy column (or a missing reference level), falsely implying collinearity.
    design, used, invariant, unavailable, _ = _design_matrix(
        working.loc[complete], marker_column=marker_column, covariates=covariates,
        numeric_covariates=numeric_covariates,
    )
    if "marker_z" not in design:
        return _empty_result(
            target_kind=target_kind, model_id=model_id, covariates=covariates,
            covariates_used=used, covariates_invariant=invariant,
            status="non_estimable", reason="marker_has_no_complete_case_variation",
            n=n, n_events=n_events,
        )
    if target_kind == "binary":
        classes = set(target.unique())
        if not classes.issubset({0.0, 1.0}) or len(classes) != 2:
            return _empty_result(
                target_kind=target_kind,
                model_id=model_id,
                covariates=covariates,
                covariates_used=used,
                covariates_invariant=invariant,
                status="non_estimable",
                reason="binary_target_requires_two_classes",
                n=n,
                n_events=n_events,
            )
        if min(n_events, n - n_events) < MINIMUM_BINARY_CLASS:
            return _empty_result(
                target_kind=target_kind,
                model_id=model_id,
                covariates=covariates,
                covariates_used=used,
                covariates_invariant=invariant,
                status="non_estimable",
                reason="insufficient_binary_class",
                n=n,
                n_events=n_events,
            )
    else:
        target_sd = float(target.std(ddof=0))
        if not np.isfinite(target_sd) or target_sd <= 0:
            return _empty_result(
                target_kind=target_kind,
                model_id=model_id,
                covariates=covariates,
                covariates_used=used,
                covariates_invariant=invariant,
                status="non_estimable",
                reason="continuous_target_has_no_variation",
                n=n,
                n_events=n_events,
            )
        target = (target - float(target.mean())) / target_sd

    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    exog = sm.add_constant(design, has_constant="add")
    matrix = exog.to_numpy(dtype=float)
    from .design_diagnostics import design_rank_condition

    rank, condition = design_rank_condition(matrix)
    if rank < matrix.shape[1]:
        return _empty_result(
            target_kind=target_kind,
            model_id=model_id,
            covariates=covariates,
            covariates_used=used,
            covariates_invariant=invariant,
            status="non_estimable",
            reason="rank_deficient_design",
            n=n,
            n_events=n_events,
        )
    if not np.isfinite(condition) or condition > MAXIMUM_DESIGN_CONDITION:
        return _empty_result(
            target_kind=target_kind,
            model_id=model_id,
            covariates=covariates,
            covariates_used=used,
            covariates_invariant=invariant,
            status="non_estimable",
            reason="ill_conditioned_design",
            n=n,
            n_events=n_events,
        )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if target_kind == "binary":
                fitted = sm.GLM(
                    target,
                    exog,
                    family=sm.families.Binomial(),
                ).fit(cov_type="HC3")
            else:
                fitted = sm.OLS(target, exog).fit(cov_type="HC3")
        if any("separation" in str(item.message).lower() for item in caught):
            return _empty_result(
                target_kind=target_kind,
                model_id=model_id,
                covariates=covariates,
                covariates_used=used,
                covariates_invariant=invariant,
                status="non_estimable",
                reason="perfect_or_quasi_separation",
                n=n,
                n_events=n_events,
            )
        estimate = float(fitted.params["marker_z"])
        standard_error = float(fitted.bse["marker_z"])
        p_value = float(fitted.pvalues["marker_z"])
        low, high = (float(value) for value in fitted.conf_int().loc["marker_z"])
    except (
        PerfectSeparationError,
        ValueError,
        np.linalg.LinAlgError,
        FloatingPointError,
    ) as error:
        reason = (
            "perfect_or_quasi_separation"
            if isinstance(error, PerfectSeparationError)
            else f"fit_failed:{type(error).__name__}"
        )
        return _empty_result(
            target_kind=target_kind,
            model_id=model_id,
            covariates=covariates,
            covariates_used=used,
            covariates_invariant=invariant,
            status="non_estimable",
            reason=reason,
            n=n,
            n_events=n_events,
        )
    if not all(np.isfinite(value) for value in (estimate, standard_error, p_value, low, high)):
        return _empty_result(
            target_kind=target_kind,
            model_id=model_id,
            covariates=covariates,
            covariates_used=used,
            covariates_invariant=invariant,
            status="non_estimable",
            reason="non_finite_fit",
            n=n,
            n_events=n_events,
        )
    if target_kind == "binary":
        # A log-odds so large that its odds ratio overflows is quasi-separation
        # the fitter did not raise on. Reported as such rather than as an
        # infinite effect: seen on the CCSR axis of the production-path
        # preview, where the overflow warning ended the stage.
        with np.errstate(over="ignore"):
            effect, low, high = (float(np.exp(value)) for value in (estimate, low, high))
        if not all(np.isfinite(value) for value in (effect, low, high)):
            return _empty_result(
                target_kind=target_kind,
                model_id=model_id,
                covariates=covariates,
                covariates_used=used,
                covariates_invariant=invariant,
                status="non_estimable",
                reason="perfect_or_quasi_separation",
                n=n,
                n_events=n_events,
            )
    else:
        effect = estimate
    result = _empty_result(
        target_kind=target_kind,
        model_id=model_id,
        covariates=covariates,
        covariates_used=used,
        covariates_invariant=invariant,
        status="estimable",
        reason="",
        n=n,
        n_events=n_events,
    )
    result.update(
        {
            "estimate": estimate,
            "effect": effect,
            "standard_error": standard_error,
            "ci_low": low,
            "ci_high": high,
            "p_value": p_value,
            "design_condition": condition,
        }
    )
    return result


def add_adjusted_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    """Add domain-family and global FDR without treating abstentions as tests."""

    output = frame.copy()
    output["domain_q_value"] = np.nan
    output["global_q_value"] = np.nan
    valid = output["status"].eq("estimable") & pd.to_numeric(
        output["p_value"], errors="coerce"
    ).notna()
    if not valid.any():
        return output
    from statsmodels.stats.multitest import multipletests

    for _, index in output.loc[valid].groupby(
        ["axis", "model_id", "evaluation_split"], sort=True
    ).groups.items():
        p_values = output.loc[index, "p_value"].astype(float).to_numpy()
        output.loc[index, "domain_q_value"] = multipletests(
            p_values, method="fdr_bh"
        )[1]
    for _, index in output.loc[valid].groupby(
        ["model_id", "evaluation_split"], sort=True
    ).groups.items():
        p_values = output.loc[index, "p_value"].astype(float).to_numpy()
        output.loc[index, "global_q_value"] = multipletests(
            p_values, method="fdr_bh"
        )[1]
    return output


def add_adjustment_effect_change(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare adjusted and unadjusted coefficients on the same scale.

    Binary models are compared on their log-odds coefficient, not by
    subtracting odds ratios; continuous models already use standardized beta.
    Non-estimable reference models deliberately yield missing change values.
    """

    output = frame.copy()
    for name in (
        "unadjusted_estimate_reference",
        "estimate_change_from_unadjusted",
        "absolute_estimate_attenuation",
        "relative_absolute_estimate_attenuation",
    ):
        output[name] = np.nan
    if output.empty or "model_id" not in output:
        return output
    keys = [
        name
        for name in (
            "axis",
            "target_id",
            "temporal_relation",
            "feature_id",
            "evaluation_split",
        )
        if name in output.columns
    ]
    reference = output.loc[
        output["model_id"].eq("unadjusted_regression")
        & output["status"].eq("estimable")
    ]
    if reference.empty:
        return output
    reference_map = {
        tuple(row[name] for name in keys): float(row["estimate"])
        for _, row in reference.iterrows()
    }
    for index, row in output.iterrows():
        value = reference_map.get(tuple(row[name] for name in keys))
        estimate = pd.to_numeric(pd.Series([row.get("estimate")]), errors="coerce").iloc[0]
        if value is None or not np.isfinite(value) or not np.isfinite(estimate):
            continue
        change = float(estimate) - value
        attenuation = abs(value) - abs(float(estimate))
        output.at[index, "unadjusted_estimate_reference"] = value
        output.at[index, "estimate_change_from_unadjusted"] = change
        output.at[index, "absolute_estimate_attenuation"] = attenuation
        if abs(value) > np.finfo(float).eps:
            output.at[index, "relative_absolute_estimate_attenuation"] = (
                attenuation / abs(value)
            )
    return output


__all__ = [
    "BROAD_AXIS_ADJUSTMENT_VERSION",
    "PRIMARY_COVARIATES",
    "SENSITIVITY_COVARIATES",
    "add_adjustment_effect_change",
    "add_adjusted_fdr",
    "fit_adjusted_association",
]
