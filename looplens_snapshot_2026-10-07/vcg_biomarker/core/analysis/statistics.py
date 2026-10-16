from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from .threshold_sweep import SweepResult


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    significant = [p < alpha for p in corrected]
    return corrected, significant


def fdr_bh_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    p_arr = np.array(p_values)
    try:
        rejected, corrected, _, _ = multipletests(p_arr, alpha=alpha, method="fdr_bh")
        return corrected.tolist(), rejected.tolist()
    except Exception:
        corrected = [p for p in p_values]
        significant = [p < alpha for p in corrected]
        return corrected, significant


def apply_multiple_testing_correction(
    results: list[SweepResult],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> list[SweepResult]:
    p_values = [r.p_value for r in results]

    if method == "bonferroni":
        corrected, significant = bonferroni_correction(p_values, alpha)
    elif method == "fdr_bh":
        corrected, significant = fdr_bh_correction(p_values, alpha)
    else:
        corrected = p_values
        significant = [p < alpha for p in corrected]

    for i, result in enumerate(results):
        result.p_value_corrected = corrected[i]
        result.correction_method = method

    return results


def compute_effect_size(biomarker_values_pos: np.ndarray, biomarker_values_neg: np.ndarray) -> float:
    n1 = len(biomarker_values_pos)
    n2 = len(biomarker_values_neg)

    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = np.mean(biomarker_values_pos)
    mean2 = np.mean(biomarker_values_neg)
    var1 = np.var(biomarker_values_pos, ddof=1)
    var2 = np.var(biomarker_values_neg, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return float(d)


def compute_descriptive_stats(values: np.ndarray) -> dict:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}

    return {
        "n": len(valid),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=min(1, len(valid) - 1))),
        "median": float(np.median(valid)),
        "q25": float(np.percentile(valid, 25)),
        "q75": float(np.percentile(valid, 75)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def filter_significant_results(
    results: list[SweepResult],
    min_auc: float = 0.6,
    min_sensitivity: float = 0.5,
    min_specificity: float = 0.5,
) -> list[SweepResult]:
    filtered = []
    for r in results:
        if r.p_value_corrected is not None and not r.is_significant:
            continue
        if r.auc < min_auc:
            continue
        if r.optimal_sensitivity < min_sensitivity:
            continue
        if r.optimal_specificity < min_specificity:
            continue
        filtered.append(r)
    return filtered
