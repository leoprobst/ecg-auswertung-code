from __future__ import annotations

import numpy as np
import pandas as pd


COMPARISON_EFFECTS = (
    (
        "sensitivity_delta_vs_fixed_rule",
        "sensitivity_delta_ci_low",
        "sensitivity_delta_ci_high",
        "sensitivity_at_baseline_specificity",
    ),
    (
        "specificity_delta_vs_fixed_rule",
        "specificity_delta_ci_low",
        "specificity_delta_ci_high",
        "specificity_at_baseline_sensitivity",
    ),
    (
        "delta_auc",
        "delta_auc_ci_low",
        "delta_auc_ci_high",
        "roc_auc",
    ),
    (
        "delta_average_precision",
        "delta_average_precision_ci_low",
        "delta_average_precision_ci_high",
        "average_precision",
    ),
)


def comparison_effect_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize fixed-rule and continuous-baseline comparisons for display."""
    if frame.empty:
        return pd.DataFrame()
    effects: list[pd.DataFrame] = []
    for value_column, low_column, high_column, metric in COMPARISON_EFFECTS:
        if value_column not in frame:
            continue
        values = pd.to_numeric(frame[value_column], errors="coerce")
        selected = frame.loc[values.notna()].copy()
        if selected.empty:
            continue
        selected["effect_metric"] = metric
        selected["effect_value"] = values.loc[selected.index].to_numpy()
        selected["effect_ci_low"] = (
            pd.to_numeric(selected[low_column], errors="coerce")
            if low_column in selected
            else np.nan
        )
        selected["effect_ci_high"] = (
            pd.to_numeric(selected[high_column], errors="coerce")
            if high_column in selected
            else np.nan
        )
        effects.append(selected)
    if not effects:
        return pd.DataFrame()
    return pd.concat(effects, ignore_index=True, sort=False)


def comparison_effect_labels(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)
    endpoint = _column_text(frame, "endpoint_id", "endpoint")
    candidate = _column_text(
        frame,
        "vcg_feature_id",
        fallback_column="feature_id",
        fallback="candidate",
    )
    metric = _column_text(frame, "effect_metric", "effect")
    return endpoint + " | " + candidate + " | " + metric


def _column_text(
    frame: pd.DataFrame,
    column: str,
    fallback: str = "",
    *,
    fallback_column: str = "",
) -> pd.Series:
    if column in frame:
        return frame[column].fillna(fallback).astype(str)
    if fallback_column and fallback_column in frame:
        return frame[fallback_column].fillna(fallback).astype(str)
    return pd.Series([fallback] * len(frame), index=frame.index, dtype=str)
