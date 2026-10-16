"""Swapping train and validation: recovering the data a split costs.

The question this answers: a 60/20/20 split spends 40 % of the cohort on
holding out. Does that not waste training data, and should the roles not be
swapped and the two runs compared?

The answer differs for the two boundaries, and conflating them is the trap:

**Train and validation may be swapped, and should be.** Their only job is to
separate selection from confirmation. Running the screen twice with the roles
exchanged -- fit on validation, confirm on train -- yields a second confirmation
of the same hypotheses on data the first run selected on. Averaging the two is
cross-fitting; every subject then contributes to both fitting and confirming,
without any single estimate ever being confirmed on the data that produced it.
That is the standard two-fold cross-fitting argument and it genuinely recovers
the lost efficiency.

**The locked test must never be swapped.** Its value is that it has been read
exactly zero times. Swapping it in would mean fitting on it; there is then no
untouched set left, and no later analysis can restore one. A sealed set is the
only thing in this pipeline that cannot be recreated by recomputation, which
is exactly why it is the one thing that must not be spent early.

So: cross-fit across train and validation, keep the locked test sealed. That
gives 80 % of the cohort working as both training and confirmation data, and
still leaves an untouched 20 % for the single confirmatory read.

Two conditions the implementation enforces, because cross-fitting is easy to
do in a way that quietly destroys the guarantee:

* **The folds must be disjoint at the patient level.** A subject appearing in
  both folds turns the swap into training on the test set.
* **The multiplicity family is the union, not each fold separately.** Two
  swapped runs are two looks at the same hypotheses; correcting each fold on
  its own and then reporting whichever looks better is selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from .run_log import profiled_phase


CROSS_FITTING_VERSION = "2.0.0"
NORMAL_95 = 1.959963984540054

FOLD_A = "train_fits_validation_confirms"
FOLD_B = "validation_fits_train_confirms"
CROSS_FIT_FOLDS: tuple[str, ...] = (FOLD_A, FOLD_B)

# Splits that may take part in a swap. The locked test is deliberately absent.
SWAPPABLE_SPLITS: frozenset[str] = frozenset({"train", "validation"})
SEALED_SPLITS: frozenset[str] = frozenset({"locked_test"})


@dataclass(frozen=True)
class CrossFitAssignment:
    fold: str
    fitting_split: str
    confirming_split: str

    def to_dict(self) -> dict[str, str]:
        return {
            "fold": self.fold,
            "fitting_split": self.fitting_split,
            "confirming_split": self.confirming_split,
            "cross_fitting_version": CROSS_FITTING_VERSION,
        }


CROSS_FIT_ASSIGNMENTS: tuple[CrossFitAssignment, ...] = (
    CrossFitAssignment(FOLD_A, "train", "validation"),
    CrossFitAssignment(FOLD_B, "validation", "train"),
)


def assert_locked_test_excluded(splits: Iterable[str]) -> None:
    """Refuse any cross-fit that would touch the sealed split.

    This is the one guarantee in the pipeline that cannot be restored by
    recomputation: once the locked test has been fitted on, no untouched set
    exists any more.
    """

    offending = sorted(set(str(item) for item in splits) & SEALED_SPLITS)
    if offending:
        raise PermissionError(
            "Cross-fitting must never include the sealed split: "
            f"{offending}. Swapping it in would mean fitting on the only data "
            "that has never been read, and nothing can restore that."
        )


def assert_folds_are_subject_disjoint(
    frame: pd.DataFrame,
    *,
    subject_column: str = "subject_id",
    split_column: str = "split",
) -> None:
    """Fail when a subject appears in more than one swappable split."""

    if frame.empty:
        return
    scoped = frame.loc[frame[split_column].astype(str).isin(SWAPPABLE_SPLITS)]
    if scoped.empty:
        return
    per_subject = scoped.groupby(subject_column)[split_column].nunique()
    leaking = per_subject.loc[per_subject > 1]
    if not leaking.empty:
        raise RuntimeError(
            f"{len(leaking)} subject(s) appear in both swappable splits; "
            "a swap would then fit and confirm on the same patients."
        )


def cross_fit_plan(
    frame: pd.DataFrame,
    *,
    subject_column: str = "subject_id",
    split_column: str = "split",
) -> pd.DataFrame:
    """Describe both folds and the data each of them uses."""

    assert_folds_are_subject_disjoint(
        frame,
        subject_column=subject_column,
        split_column=split_column,
    )
    rows: list[dict[str, object]] = []
    for assignment in CROSS_FIT_ASSIGNMENTS:
        fitting = frame.loc[frame[split_column].eq(assignment.fitting_split)]
        confirming = frame.loc[
            frame[split_column].eq(assignment.confirming_split)
        ]
        rows.append(
            {
                **assignment.to_dict(),
                "n_fitting_rows": int(len(fitting)),
                "n_confirming_rows": int(len(confirming)),
                "n_fitting_subjects": int(fitting[subject_column].nunique()),
                "n_confirming_subjects": int(
                    confirming[subject_column].nunique()
                ),
                "locked_test_read": False,
            }
        )
    return pd.DataFrame(rows)


@profiled_phase("postprocess.cross_fit_pooling")
def combine_folds(
    fold_results: Mapping[str, pd.DataFrame],
    *,
    key_columns: Sequence[str] = ("endpoint_id", "feature_id"),
    effect_column: str = "roc_auc",
    direction_column: str = "direction",
    n_column: str = "n_total",
    positive_column: str = "n_positive",
    negative_column: str = "n_negative",
    p_column: str = "p_value",
    variance_column: str = "roc_auc_variance",
    se_column: str = "roc_auc_se",
    ci_low_column: str = "roc_auc_ci_low",
    ci_high_column: str = "roc_auc_ci_high",
) -> pd.DataFrame:
    """Pool the two swapped confirmations into one estimate per hypothesis.

    Pooling uses an actual AUC variance, never ``n`` while calling it an
    inverse-variance estimate.  Variance is taken, in order, from an explicit
    variance/SE, a two-sided 95 % CI, or the Hanley--McNeil large-sample AUC
    variance using the confirming positive and negative class counts.  Only
    legacy rows lacking every uncertainty input use ``1/n`` and are labelled
    as such.

    The confirmatory p-value is an intersection-union test: both independent
    confirmations must reject in the same learned direction, hence
    ``max(p_A, p_B)``.  This avoids the invalid practice of averaging away a
    direction reversal or selecting the better of two looks.  Multiplicity
    across marker hypotheses is applied by the caller to this single IUT
    p-value.
    """

    frames: list[pd.DataFrame] = []
    for fold, frame in fold_results.items():
        if frame is None or frame.empty:
            continue
        scoped = frame.copy()
        scoped["cross_fit_fold"] = str(fold)
        frames.append(scoped)
    if not frames:
        return pd.DataFrame(
            columns=[*key_columns, "pooled_effect", "folds_available"]
        )
    combined = pd.concat(frames, ignore_index=True)
    keys = [column for column in key_columns if column in combined.columns]
    if not keys:
        raise ValueError("None of the key columns are present in the results")
    # Variance resolution is row-local, not hypothesis-local. Resolve immutable
    # numeric inputs once rather than constructing seven numeric Series per group.
    resolved_effect_column = effect_column
    if resolved_effect_column not in combined and "aligned_auc" in combined:
        resolved_effect_column = "aligned_auc"
    all_effects = pd.to_numeric(
        combined.get(resolved_effect_column, pd.Series(dtype=float)), errors="coerce",
    )
    all_variances, all_methods = _auc_variances(
        combined, all_effects, n_column=n_column, positive_column=positive_column,
        negative_column=negative_column, variance_column=variance_column,
        se_column=se_column, ci_low_column=ci_low_column, ci_high_column=ci_high_column,
    )
    all_weights = pd.Series(np.divide(
        1.0, all_variances, out=np.zeros(len(all_variances), dtype=float),
        where=np.isfinite(all_variances) & (all_variances > 0.0),
    ), index=combined.index)
    all_sample_sizes = pd.to_numeric(
        combined.get(n_column, pd.Series(0, index=combined.index)), errors="coerce",
    ).fillna(0.0)
    all_p_values = pd.to_numeric(
        combined.get(p_column, pd.Series(np.nan, index=combined.index)), errors="coerce",
    )
    records: list[dict[str, object]] = []
    for key_values, group in combined.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        effects = all_effects.loc[group.index]
        methods = all_methods.loc[group.index]
        weights = all_weights.loc[group.index]
        usable = effects.notna() & np.isfinite(weights) & (weights > 0)
        record: dict[str, object] = dict(zip(keys, key_values))
        record["folds_available"] = int(group["cross_fit_fold"].nunique())
        record["cross_fitting_version"] = CROSS_FITTING_VERSION
        if not usable.any():
            record["pooled_effect"] = float("nan")
            record["fold_agreement"] = "not_evaluable"
            records.append(record)
            continue
        pooled = float(np.average(effects[usable], weights=weights[usable]))
        record["pooled_effect"] = pooled
        sample_sizes = all_sample_sizes.loc[group.index]
        record["n_confirming_total"] = int(sample_sizes[usable].sum())
        pooled_se = float(np.sqrt(1.0 / weights[usable].sum()))
        record["pooled_effect_se"] = pooled_se
        record["pooled_effect_ci_low"] = float(
            np.clip(pooled - NORMAL_95 * pooled_se, 0.0, 1.0)
        )
        record["pooled_effect_ci_high"] = float(
            np.clip(pooled + NORMAL_95 * pooled_se, 0.0, 1.0)
        )
        used_methods = sorted(set(methods.loc[usable].astype(str)))
        record["pooling_weight_method"] = "|".join(used_methods)
        record["effect_spread"] = float(
            effects[usable].max() - effects[usable].min()
        )
        if direction_column in group.columns:
            directions = pd.to_numeric(
                group.loc[usable, direction_column],
                errors="coerce",
            ).dropna()
            consistent = bool(
                len(directions) >= 2 and directions.nunique() == 1
            )
        else:
            # Without an explicit direction, use which side of 0.5 the
            # discrimination falls on.
            consistent = bool(
                len(effects[usable]) >= 2
                and ((effects[usable] > 0.5).nunique() == 1)
            )
        record["direction_consistent_across_folds"] = consistent
        record["fold_agreement"] = (
            "consistent"
            if consistent and record["folds_available"] >= 2
            else "single_fold"
            if record["folds_available"] < 2
            else "direction_reversed"
        )
        p_values = all_p_values.loc[group.index]
        confirmatory_p = (
            float(np.max(p_values[usable]))
            if consistent
            and record["folds_available"] >= 2
            and int(np.sum(np.isfinite(p_values[usable]))) >= 2
            else 1.0
        )
        record["p_value"] = confirmatory_p
        record["cross_fit_confirmed_raw"] = bool(
            consistent
            and record["folds_available"] >= 2
            and confirmatory_p <= 0.05
            and np.all(effects[usable] > 0.5)
        )
        record["evaluation_split"] = "cross_fit_out_of_fold"
        for column in (
            "analysis_type",
            "canonical_feature_id",
            "conceptual_family",
            "feature_modality",
            "feature_role",
            "scientific_tier",
        ):
            if column in group:
                record[column] = group.iloc[0].get(column)
        fold_formal = (
            group.loc[usable, "inference_status"].astype(str).eq(
                "formal_inference"
            )
            if "inference_status" in group
            else pd.Series(True, index=group.loc[usable].index)
        )
        record["inference_status"] = (
            "formal_inference"
            if len(fold_formal) >= 2 and bool(fold_formal.all())
            else "descriptive_only"
        )
        for column in (
            "assessability_gate_passed",
            "missingness_sensitivity_gate_passed",
            "confirmatory_assessability_gate_passed",
        ):
            if column in group:
                record[column] = bool(
                    len(group.loc[usable, column]) >= 2
                    and group.loc[usable, column].fillna(False).astype(bool).all()
                )
        if len(effects[usable]) >= 2:
            q_statistic = float(
                np.sum(weights[usable] * (effects[usable] - pooled) ** 2)
            )
            record["heterogeneity_q"] = q_statistic
            record["heterogeneity_i2"] = float(
                max(0.0, (q_statistic - 1.0) / q_statistic)
                if q_statistic > 0.0
                else 0.0
            )
        for fold, fold_row in group.loc[usable].groupby(
            "cross_fit_fold", sort=True
        ):
            row = fold_row.iloc[0]
            prefix = "fold_a" if str(fold) == FOLD_A else "fold_b"
            record[f"{prefix}_effect"] = float(
                pd.to_numeric(row.get(resolved_effect_column), errors="coerce")
            )
            record[f"{prefix}_p_value"] = _finite_float_or_nan(
                row.get(p_column)
            )
            record[f"{prefix}_direction"] = _finite_float_or_nan(
                row.get(direction_column)
            )
            record[f"{prefix}_n_confirming"] = int(
                max(0.0, _finite_float_or_nan(row.get(n_column), default=0.0))
            )
        records.append(record)
    output = pd.DataFrame(records)
    return output.sort_values(
        "pooled_effect",
        ascending=False,
        kind="mergesort",
        ignore_index=True,
    )


def _finite_float_or_nan(value: object, *, default: float = np.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)


def _auc_variances(
    frame: pd.DataFrame,
    effects: pd.Series,
    *,
    n_column: str,
    positive_column: str,
    negative_column: str,
    variance_column: str,
    se_column: str,
    ci_low_column: str,
    ci_high_column: str,
) -> tuple[np.ndarray, pd.Series]:
    """Resolve one defensible variance per confirming-fold AUC."""

    variances = np.full(len(frame), np.nan, dtype=float)
    methods = pd.Series("unavailable", index=frame.index, dtype=object)

    def values(column: str) -> np.ndarray:
        return pd.to_numeric(
            frame.get(column, pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        ).to_numpy(dtype=float)

    explicit = values(variance_column)
    use = np.isfinite(explicit) & (explicit > 0.0)
    variances[use] = explicit[use]
    methods.iloc[np.flatnonzero(use)] = "explicit_auc_variance"

    standard_errors = values(se_column)
    use = (
        ~np.isfinite(variances)
        & np.isfinite(standard_errors)
        & (standard_errors > 0.0)
    )
    variances[use] = standard_errors[use] ** 2
    methods.iloc[np.flatnonzero(use)] = "explicit_auc_se"

    ci_low = values(ci_low_column)
    ci_high = values(ci_high_column)
    ci_se = (ci_high - ci_low) / (2.0 * NORMAL_95)
    use = ~np.isfinite(variances) & np.isfinite(ci_se) & (ci_se > 0.0)
    variances[use] = ci_se[use] ** 2
    methods.iloc[np.flatnonzero(use)] = "bootstrap_ci_derived_se"

    positives = values(positive_column)
    negatives = values(negative_column)
    aucs = effects.to_numpy(dtype=float)
    analytic = _hanley_mcneil_auc_variance(aucs, positives, negatives)
    use = ~np.isfinite(variances) & np.isfinite(analytic) & (analytic > 0.0)
    variances[use] = analytic[use]
    methods.iloc[np.flatnonzero(use)] = "hanley_mcneil_auc_variance"

    # Compatibility for old result tables that contain only n.  This is
    # deliberately explicit in provenance and must not be described as an
    # AUC variance estimate.
    sample_sizes = values(n_column)
    use = ~np.isfinite(variances) & np.isfinite(sample_sizes) & (sample_sizes > 0)
    variances[use] = 1.0 / sample_sizes[use]
    methods.iloc[np.flatnonzero(use)] = "sample_size_fallback_no_uncertainty"
    return variances, methods


def _hanley_mcneil_auc_variance(
    auc: np.ndarray,
    n_positive: np.ndarray,
    n_negative: np.ndarray,
) -> np.ndarray:
    auc = np.asarray(auc, dtype=float)
    n_positive = np.asarray(n_positive, dtype=float)
    n_negative = np.asarray(n_negative, dtype=float)
    output = np.full(np.broadcast_shapes(auc.shape, n_positive.shape, n_negative.shape), np.nan)
    valid = (
        np.isfinite(auc)
        & (auc >= 0.0)
        & (auc <= 1.0)
        & np.isfinite(n_positive)
        & np.isfinite(n_negative)
        & (n_positive > 0.0)
        & (n_negative > 0.0)
    )
    if not np.any(valid):
        return output
    a = auc[valid]
    q1 = np.divide(a, 2.0 - a, out=np.zeros_like(a), where=(2.0 - a) != 0)
    q2 = np.divide(2.0 * a * a, 1.0 + a, out=np.zeros_like(a), where=(1.0 + a) != 0)
    numerator = (
        a * (1.0 - a)
        + (n_positive[valid] - 1.0) * (q1 - a * a)
        + (n_negative[valid] - 1.0) * (q2 - a * a)
    )
    output[valid] = numerator / (n_positive[valid] * n_negative[valid])
    return output


def cross_fit_multiplicity_family(pooled: pd.DataFrame) -> Mapping[str, object]:
    """State the family the correction must span.

    Two swapped runs look at one set of hypotheses twice. Correcting each
    fold separately and then reporting the better one would be selection
    dressed as replication.
    """

    return {
        "hypotheses": int(len(pooled)),
        "folds": len(CROSS_FIT_FOLDS),
        "family_definition": "union_over_folds_not_per_fold",
        "note_en": (
            "The two folds are two looks at the same hypotheses. Multiplicity "
            "control applies to the union; per-fold correction followed by "
            "picking the better fold is selection, not replication."
        ),
        "cross_fitting_version": CROSS_FITTING_VERSION,
    }


def cross_fit_summary(pooled: pd.DataFrame) -> Mapping[str, object]:
    """How much the swap actually bought, and where the folds disagreed."""

    if pooled.empty:
        return {
            "hypotheses": 0,
            "confirmed_in_both_folds": 0,
            "cross_fitting_version": CROSS_FITTING_VERSION,
        }
    both = pooled.loc[pooled["folds_available"] >= 2]
    consistent = both.loc[
        both.get(
            "direction_consistent_across_folds",
            pd.Series(False, index=both.index),
        ).astype(bool)
    ]
    reversed_direction = both.loc[
        ~both.get(
            "direction_consistent_across_folds",
            pd.Series(False, index=both.index),
        ).astype(bool)
    ]
    return {
        "hypotheses": int(len(pooled)),
        "evaluated_in_both_folds": int(len(both)),
        "direction_consistent": int(len(consistent)),
        "direction_reversed": int(len(reversed_direction)),
        "median_effect_spread": (
            float(both["effect_spread"].median()) if len(both) else float("nan")
        ),
        "interpretation_en": (
            "A hypothesis whose direction reverses between folds did not have "
            "a smaller effect; it had none that survived being confirmed on "
            "different patients."
        ),
        "locked_test_read": False,
        "cross_fitting_version": CROSS_FITTING_VERSION,
    }


def efficiency_gain(
    n_train_subjects: int,
    n_validation_subjects: int,
    n_locked_subjects: int,
) -> Mapping[str, object]:
    """How much of the cohort each strategy actually uses for fitting."""

    total = max(1, n_train_subjects + n_validation_subjects + n_locked_subjects)
    single = n_train_subjects / total
    crossed = (n_train_subjects + n_validation_subjects) / total
    return {
        "subjects_total": total,
        "fitting_share_single_direction": single,
        "fitting_share_cross_fitted": crossed,
        "relative_gain": (crossed / single) if single > 0 else float("nan"),
        "sealed_share_retained": n_locked_subjects / total,
        "note_en": (
            "Cross-fitting lets every non-sealed subject contribute to both "
            "fitting and confirmation, while no estimate is ever confirmed on "
            "the data that produced it. The sealed share is untouched."
        ),
        "cross_fitting_version": CROSS_FITTING_VERSION,
    }


__all__ = [
    "CROSS_FITTING_VERSION",
    "CROSS_FIT_ASSIGNMENTS",
    "CROSS_FIT_FOLDS",
    "CrossFitAssignment",
    "FOLD_A",
    "FOLD_B",
    "SEALED_SPLITS",
    "SWAPPABLE_SPLITS",
    "assert_folds_are_subject_disjoint",
    "assert_locked_test_excluded",
    "combine_folds",
    "cross_fit_multiplicity_family",
    "cross_fit_plan",
    "cross_fit_summary",
    "efficiency_gain",
]
