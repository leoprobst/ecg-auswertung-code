from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .statistics import marker_bootstrap_intervals


DISCOVERY_STABILITY_GATE_VERSION = "1.0.0"
DISCOVERY_MIN_ALIGNED_AUC = 0.60
DISCOVERY_MIN_ABSOLUTE_EFFECT = 0.20
DISCOVERY_MAX_EFFECT_GAP = 0.10
DISCOVERY_MIN_EFFECT_RETENTION = 0.50


def add_auc_replication_status(
    frame: pd.DataFrame,
    *,
    validation_fdr_alpha: float,
    q_column: str = "q_value",
    min_aligned_auc: float = DISCOVERY_MIN_ALIGNED_AUC,
    max_auc_gap: float = DISCOVERY_MAX_EFFECT_GAP,
    min_effect_retention: float = DISCOVERY_MIN_EFFECT_RETENTION,
) -> pd.DataFrame:
    output = frame.copy()
    train_auc = pd.to_numeric(
        output["train_aligned_auc"],
        errors="coerce",
    ).to_numpy(dtype=float)
    validation_auc = pd.to_numeric(
        output["roc_auc"],
        errors="coerce",
    ).to_numpy(dtype=float)
    train_effect = np.maximum(2.0 * train_auc - 1.0, 0.0)
    validation_effect = np.maximum(
        2.0 * validation_auc - 1.0,
        0.0,
    )
    retention = np.full(len(output), np.nan, dtype=float)
    np.divide(
        validation_effect,
        train_effect,
        out=retention,
        where=train_effect > 0,
    )
    auc_gap = np.abs(train_auc - validation_auc)
    q_values = pd.to_numeric(
        output[q_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    direction_consistent = output["direction_consistent"].fillna(
        False
    ).to_numpy(dtype=bool)
    formal = output["inference_status"].eq(
        "formal_inference"
    ).to_numpy(dtype=bool)
    fdr_direction_replicated = (
        np.isfinite(q_values)
        & (q_values <= float(validation_fdr_alpha))
        & direction_consistent
        & formal
    )
    validation_stable = (
        np.isfinite(validation_auc)
        & np.isfinite(auc_gap)
        & np.isfinite(retention)
        & (validation_auc >= float(min_aligned_auc))
        & (auc_gap <= float(max_auc_gap))
        & (retention >= float(min_effect_retention))
    )
    output["validation_aligned_auc"] = validation_auc
    output["train_validation_auc_gap"] = auc_gap
    output["train_aligned_rank_biserial"] = train_effect
    output["validation_aligned_rank_biserial"] = validation_effect
    output["effect_retention_ratio"] = retention
    output["fdr_direction_replicated"] = fdr_direction_replicated
    output["validation_stable"] = validation_stable
    output["replicated"] = (
        fdr_direction_replicated & validation_stable
    )
    output["replication_gate_version"] = (
        DISCOVERY_STABILITY_GATE_VERSION
    )
    return output


def add_effect_replication_status(
    frame: pd.DataFrame,
    *,
    validation_fdr_alpha: float,
    q_column: str = "validation_q_value",
    min_absolute_effect: float = DISCOVERY_MIN_ABSOLUTE_EFFECT,
    max_effect_gap: float = DISCOVERY_MAX_EFFECT_GAP,
    min_effect_retention: float = DISCOVERY_MIN_EFFECT_RETENTION,
) -> pd.DataFrame:
    output = frame.copy()
    train_effect = pd.to_numeric(
        output["train_effect"],
        errors="coerce",
    ).to_numpy(dtype=float)
    validation_effect = pd.to_numeric(
        output["validation_effect"],
        errors="coerce",
    ).to_numpy(dtype=float)
    absolute_train = np.abs(train_effect)
    absolute_validation = np.abs(validation_effect)
    retention = np.full(len(output), np.nan, dtype=float)
    np.divide(
        absolute_validation,
        absolute_train,
        out=retention,
        where=absolute_train > 0,
    )
    effect_gap = np.abs(absolute_train - absolute_validation)
    q_values = pd.to_numeric(
        output[q_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    direction_consistent = output["direction_consistent"].fillna(
        False
    ).to_numpy(dtype=bool)
    formal = output["inference_status"].eq(
        "formal_inference"
    ).to_numpy(dtype=bool)
    fdr_direction_replicated = (
        np.isfinite(q_values)
        & (q_values <= float(validation_fdr_alpha))
        & direction_consistent
        & formal
    )
    validation_stable = (
        np.isfinite(absolute_validation)
        & np.isfinite(effect_gap)
        & np.isfinite(retention)
        & (absolute_validation >= float(min_absolute_effect))
        & (effect_gap <= float(max_effect_gap))
        & (retention >= float(min_effect_retention))
    )
    output["train_validation_effect_gap"] = effect_gap
    output["effect_retention_ratio"] = retention
    output["fdr_direction_replicated"] = fdr_direction_replicated
    output["validation_stable"] = validation_stable
    output["replicated"] = (
        fdr_direction_replicated & validation_stable
    )
    output["replication_gate_version"] = (
        DISCOVERY_STABILITY_GATE_VERSION
    )
    return output


def discovery_marker_bootstrap_intervals(
    subject_ids: Iterable,
    values: Iterable[float],
    labels: Iterable[int],
    *,
    direction: int,
    thresholds: Optional[Mapping[str, float]] = None,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    output = marker_bootstrap_intervals(
        subject_ids,
        values,
        labels,
        direction=direction,
        thresholds=thresholds,
        iterations=0,
        seed=seed,
    )
    output.update(
        {
            "aligned_rank_biserial_ci_low": float("nan"),
            "aligned_rank_biserial_ci_high": float("nan"),
            "bootstrap_metric_scope": (
                "roc_auc_and_aligned_rank_biserial"
            ),
        }
    )
    if iterations <= 0:
        return output

    raw_values = np.asarray(values, dtype=float)
    raw_labels = np.asarray(labels, dtype=float)
    raw_subjects = np.asarray(subject_ids)
    if (
        raw_values.shape != raw_labels.shape
        or raw_subjects.shape != raw_values.shape
    ):
        raise ValueError(
            "subject_ids, values and labels must have matching lengths"
        )
    clean = (
        np.isfinite(raw_values)
        & np.isfinite(raw_labels)
        & np.isin(raw_labels, (0, 1))
    )
    scores = int(direction) * raw_values[clean]
    binary = raw_labels[clean].astype(np.int8)
    subjects = raw_subjects[clean]
    if len(scores) == 0 or np.unique(binary).size < 2:
        return output

    subject_codes, unique_subjects = pd.factorize(
        subjects,
        sort=False,
        use_na_sentinel=False,
    )
    n_subjects = len(unique_subjects)
    if n_subjects < 2:
        return output
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    positive = binary == 1
    sorted_positive = positive[order]
    group_starts = np.flatnonzero(
        np.r_[
            True,
            sorted_scores[1:] != sorted_scores[:-1],
        ]
    )
    probabilities = np.full(
        n_subjects,
        1.0 / n_subjects,
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    batch_size = min(
        64,
        max(
            1,
            2_000_000 // max(len(scores), n_subjects),
        ),
    )
    estimates: list[np.ndarray] = []
    successful = 0
    remaining = int(iterations)
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        subject_weights = rng.multinomial(
            n_subjects,
            probabilities,
            size=current_batch,
        )
        row_weights = subject_weights[:, subject_codes]
        total_positive = row_weights[:, positive].sum(
            axis=1,
            dtype=np.int64,
        )
        total_negative = row_weights[:, ~positive].sum(
            axis=1,
            dtype=np.int64,
        )
        valid = (total_positive > 0) & (total_negative > 0)
        sorted_weights = row_weights[:, order]
        positive_by_group = np.add.reduceat(
            sorted_weights * sorted_positive[np.newaxis, :],
            group_starts,
            axis=1,
        ).astype(float)
        negative_by_group = np.add.reduceat(
            sorted_weights * (~sorted_positive)[np.newaxis, :],
            group_starts,
            axis=1,
        ).astype(float)
        cumulative_negative = np.cumsum(
            negative_by_group,
            axis=1,
        )
        negative_below = (
            total_negative.astype(float)[:, np.newaxis]
            - cumulative_negative
        )
        numerator = np.sum(
            positive_by_group
            * (negative_below + 0.5 * negative_by_group),
            axis=1,
        )
        denominator = (
            total_positive.astype(float)
            * total_negative.astype(float)
        )
        auc = np.full(current_batch, np.nan, dtype=float)
        np.divide(
            numerator,
            denominator,
            out=auc,
            where=valid,
        )
        finite = np.isfinite(auc)
        if finite.any():
            estimates.append(auc[finite])
        successful += int(np.sum(valid))
        remaining -= current_batch

    if estimates:
        all_estimates = np.concatenate(estimates)
        low, high = np.quantile(
            all_estimates,
            (0.025, 0.975),
        )
        output["roc_auc_ci_low"] = float(low)
        output["roc_auc_ci_high"] = float(high)
        output["aligned_rank_biserial_ci_low"] = float(
            2.0 * low - 1.0
        )
        output["aligned_rank_biserial_ci_high"] = float(
            2.0 * high - 1.0
        )
    output["bootstrap_iterations"] = int(iterations)
    output["bootstrap_successful_iterations"] = int(successful)
    output["bootstrap_unit"] = "subject_id"
    return output
