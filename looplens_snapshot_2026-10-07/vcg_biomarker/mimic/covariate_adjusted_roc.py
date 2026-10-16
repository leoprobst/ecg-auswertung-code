"""Covariate-adjusted AUC (audit 5.3.6).

Janes & Pepe, Biometrika 2009 (PMID 22822245), semiparametric estimator.
The question is not "does the marker separate cases from controls" but "does
it separate them among people of the same age, sex and heart rate". A marker
that only tracks rate (and rate tracks illness) keeps a good crude AUC and
loses it here.

* Reference distribution: the marker among **training controls** is
  modelled as m(X) + sigma * e, with m linear in the covariates and e from
  the empirical distribution of the standardised training residuals.
* Placement value of a **validation case**: F(y | x) = P(e < (y - m(x)) /
  sigma), ties counting one half.
* The covariate-adjusted AUC is the mean placement value over validation
  cases. It equals the ordinary AUC when the covariates carry nothing.

The marker is aligned with the training direction first, so 0.5 means no
covariate-adjusted discrimination in the direction the marker was chosen.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

COVARIATE_ADJUSTED_ROC_VERSION = "1.0.0"
HEART_RATE_COLUMNS = ("context_heart_rate_bpm", "machine_heart_rate_bpm", "heart_rate_bpm")


def covariate_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    parts = []
    for column in columns:
        if column == "female":
            sex = frame.get("sex", pd.Series(pd.NA, index=frame.index)).astype("string").str.upper()
            parts.append(np.where(sex.eq("F").fillna(False), 1.0,
                                  np.where(sex.eq("M").fillna(False), 0.0, np.nan)))
        else:
            parts.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
    return np.column_stack(parts) if parts else np.zeros((len(frame), 0))


def covariate_adjusted_auc(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    train_covariates: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    test_covariates: np.ndarray,
    *,
    direction: int = 1,
    minimum_controls: int = 50,
) -> dict:
    """Janes-Pepe semiparametric AAUC; reference from train controls."""

    result = {"covariate_adjusted_auc": np.nan, "unadjusted_auc_same_rows": np.nan,
              "n_reference_controls": 0, "n_cases": 0, "n_controls": 0}
    tv = direction * np.asarray(train_values, dtype=float)
    tl = np.asarray(train_labels)
    tx = np.asarray(train_covariates, dtype=float)
    reference = (tl == 0) & np.isfinite(tv) & np.all(np.isfinite(tx), axis=1)
    if int(np.sum(reference)) < int(minimum_controls):
        return result
    design = np.column_stack([np.ones(int(np.sum(reference))), tx[reference]])
    coefficients, *_ = np.linalg.lstsq(design, tv[reference], rcond=None)
    residuals = tv[reference] - design @ coefficients
    scale = float(np.std(residuals, ddof=design.shape[1]))
    if not np.isfinite(scale) or scale <= 0:
        return result
    standardized = np.sort(residuals / scale)

    vv = direction * np.asarray(test_values, dtype=float)
    vl = np.asarray(test_labels)
    vx = np.asarray(test_covariates, dtype=float)
    usable = np.isfinite(vv) & np.all(np.isfinite(vx), axis=1)
    cases = usable & (vl == 1)
    controls = usable & (vl == 0)
    if not np.any(cases) or not np.any(controls):
        return {**result, "n_reference_controls": int(np.sum(reference))}
    z = (vv[cases] - np.column_stack([np.ones(int(np.sum(cases))), vx[cases]]) @ coefficients) / scale
    lower = np.searchsorted(standardized, z, side="left")
    tied = np.searchsorted(standardized, z, side="right") - lower
    placement = (lower + 0.5 * tied) / len(standardized)

    control_values = np.sort(vv[controls])
    below = np.searchsorted(control_values, vv[cases], side="left")
    equal = np.searchsorted(control_values, vv[cases], side="right") - below
    crude = float(np.mean((below + 0.5 * equal) / len(control_values)))
    return {
        "covariate_adjusted_auc": float(np.mean(placement)),
        "unadjusted_auc_same_rows": crude,
        "n_reference_controls": int(np.sum(reference)),
        "n_cases": int(np.sum(cases)),
        "n_controls": int(np.sum(controls)),
    }


def heart_rate_column(frame: pd.DataFrame) -> str | None:
    for column in HEART_RATE_COLUMNS:
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    return None
