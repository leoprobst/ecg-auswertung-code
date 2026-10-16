from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    auc as sklearn_auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.multitest import multipletests

from .models import FrozenScoreSpec
from .storage import content_hash
from .run_log import profiled_phase


ASSESSABILITY_GATE_VERSION = "1.0.0"
MINIMUM_CONFIRMATORY_TEST_YIELD = 0.80
MAXIMUM_CONFIRMATORY_CLASS_YIELD_GAP = 0.10
MISSINGNESS_SENSITIVITY_NULL_AUC = 0.50


@dataclass(frozen=True)
class MarkerModel:
    feature_id: str
    direction: int
    mean: float
    standard_deviation: float
    thresholds: Mapping[str, float]
    n_train: int
    n_positive_train: int
    n_negative_train: int
    # Reported, never acted on. See ``fit_marker_model`` for what they mean
    # and why they are only a diagnostic.
    folded_auc: float = float("nan")
    folded_effect_gain: float = float("nan")

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("Marker direction must be +1 or -1")
        if not np.isfinite(self.standard_deviation) or self.standard_deviation <= 0:
            raise ValueError("Marker standard deviation must be positive")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["thresholds"] = dict(self.thresholds)
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "MarkerModel":
        return cls(
            feature_id=str(data["feature_id"]),
            direction=int(data["direction"]),
            mean=float(data["mean"]),
            standard_deviation=float(data["standard_deviation"]),
            thresholds={
                str(key): float(value)
                for key, value in dict(data["thresholds"]).items()
            },
            n_train=int(data["n_train"]),
            n_positive_train=int(data["n_positive_train"]),
            n_negative_train=int(data["n_negative_train"]),
        )


@dataclass(frozen=True)
class SortedControlDistribution:
    sorted_values: np.ndarray
    tie_term: float


def prepare_sorted_control_distribution(
    values: Iterable[float],
) -> SortedControlDistribution:
    control_values = np.asarray(values, dtype=float)
    sorted_values = np.sort(control_values[np.isfinite(control_values)])
    _, counts = np.unique(sorted_values, return_counts=True)
    counts_float = counts.astype(np.float64)
    return SortedControlDistribution(
        sorted_values=sorted_values,
        tie_term=float(np.sum(counts_float**3 - counts_float)),
    )


def fast_mann_whitney_auc(
    case_values: Iterable[float],
    controls: Iterable[float] | SortedControlDistribution,
) -> tuple[float, float]:
    """Return AUC and tie-corrected asymptotic Mann-Whitney p-value."""
    cases = np.asarray(case_values, dtype=float)
    cases = cases[np.isfinite(cases)]
    distribution = (
        controls
        if isinstance(controls, SortedControlDistribution)
        else prepare_sorted_control_distribution(controls)
    )
    sorted_controls = distribution.sorted_values
    n_case = len(cases)
    n_control = len(sorted_controls)
    if n_case == 0 or n_control == 0:
        return float("nan"), 1.0

    less = np.searchsorted(sorted_controls, cases, side="left")
    less_or_equal = np.searchsorted(
        sorted_controls,
        cases,
        side="right",
    )
    u_value = float(np.sum((less + less_or_equal) / 2.0))
    auc = u_value / (n_case * n_control)

    case_unique, case_counts = np.unique(cases, return_counts=True)
    control_left = np.searchsorted(
        sorted_controls,
        case_unique,
        side="left",
    )
    control_right = np.searchsorted(
        sorted_controls,
        case_unique,
        side="right",
    )
    control_counts = (control_right - control_left).astype(np.float64)
    case_counts_float = case_counts.astype(np.float64)
    combined_counts = control_counts + case_counts_float
    combined_tie_term = distribution.tie_term + float(
        np.sum(
            combined_counts**3
            - combined_counts
            - (control_counts**3 - control_counts)
        )
    )
    total = n_case + n_control
    tie_correction = combined_tie_term / (total * (total - 1.0))
    variance = (
        n_case
        * n_control
        / 12.0
        * (total + 1.0 - tie_correction)
    )
    if variance <= 0:
        return float(auc), 1.0
    centered = abs(u_value - n_case * n_control / 2.0)
    z_value = max(0.0, centered - 0.5) / np.sqrt(variance)
    p_value = float(2.0 * scipy_stats.norm.sf(z_value))
    return float(auc), max(
        np.finfo(float).tiny,
        min(1.0, p_value),
    )


def excluded_mann_whitney_auc(
    case_values: Iterable[float],
    distribution: SortedControlDistribution,
    excluded_values: Iterable[float],
) -> tuple[float, float]:
    """AUC and p-value against a control set with some members removed.

    The phenotype scan ranks every marker against the controls of every
    phenotype, and those control sets differ only by the rows belonging to that
    phenotype's own cases -- typically well under one percent of them.  Sorting
    the whole control column again for each phenotype therefore repeats almost
    the same sort 2,885 times per marker.

    The Mann-Whitney statistic is a function of how many controls fall below
    and at each case value, so removing controls is a subtraction:

        below_reduced   = below_full   - below_excluded
        atOrBelow_reduced = atOrBelow_full - atOrBelow_excluded

    and the tie term corrects only at the values that were actually removed.
    The result is the same number the direct computation gives -- not an
    approximation -- at the cost of sorting the handful of excluded values
    instead of the whole control column.
    """

    cases = np.asarray(case_values, dtype=float)
    cases = cases[np.isfinite(cases)]
    sorted_controls = distribution.sorted_values
    excluded = np.asarray(list(excluded_values), dtype=float)
    excluded = np.sort(excluded[np.isfinite(excluded)])

    n_case = len(cases)
    n_control = len(sorted_controls) - len(excluded)
    if n_case == 0 or n_control <= 0:
        return float("nan"), 1.0

    below = np.searchsorted(sorted_controls, cases, side="left") - np.searchsorted(
        excluded, cases, side="left"
    )
    at_or_below = np.searchsorted(
        sorted_controls, cases, side="right"
    ) - np.searchsorted(excluded, cases, side="right")
    u_value = float(np.sum((below + at_or_below) / 2.0))
    auc = u_value / (n_case * n_control)

    control_tie_term = _reduced_control_tie_term(
        sorted_controls, excluded, distribution.tie_term
    )
    case_unique, case_counts = np.unique(cases, return_counts=True)
    control_counts = (
        np.searchsorted(sorted_controls, case_unique, side="right")
        - np.searchsorted(sorted_controls, case_unique, side="left")
        - (
            np.searchsorted(excluded, case_unique, side="right")
            - np.searchsorted(excluded, case_unique, side="left")
        )
    ).astype(np.float64)
    case_counts_float = case_counts.astype(np.float64)
    combined_counts = control_counts + case_counts_float
    combined_tie_term = control_tie_term + float(
        np.sum(
            combined_counts**3
            - combined_counts
            - (control_counts**3 - control_counts)
        )
    )
    total = n_case + n_control
    tie_correction = combined_tie_term / (total * (total - 1.0))
    variance = n_case * n_control / 12.0 * (total + 1.0 - tie_correction)
    if variance <= 0:
        return float(auc), 1.0
    centered = abs(u_value - n_case * n_control / 2.0)
    z_value = max(0.0, centered - 0.5) / np.sqrt(variance)
    p_value = float(2.0 * scipy_stats.norm.sf(z_value))
    return float(auc), max(np.finfo(float).tiny, min(1.0, p_value))


def _reduced_control_tie_term(
    sorted_controls: np.ndarray,
    excluded_sorted: np.ndarray,
    full_tie_term: float,
) -> float:
    """``sum(t^3 - t)`` after removing ``excluded_sorted`` from the controls.

    Only the values that were removed can change a tie group, so the correction
    touches at most as many groups as there are distinct excluded values.
    """

    if len(excluded_sorted) == 0:
        return float(full_tie_term)
    unique_excluded, removed_counts = np.unique(excluded_sorted, return_counts=True)
    full_counts = (
        np.searchsorted(sorted_controls, unique_excluded, side="right")
        - np.searchsorted(sorted_controls, unique_excluded, side="left")
    ).astype(np.float64)
    remaining = np.maximum(full_counts - removed_counts.astype(np.float64), 0.0)
    before = float(np.sum(full_counts**3 - full_counts))
    after = float(np.sum(remaining**3 - remaining))
    return float(full_tie_term) - before + after


@profiled_phase("marker_fit")
def fit_marker_model(
    values: Iterable[float],
    labels: Iterable[int],
    feature_id: str,
) -> MarkerModel:
    clean_values, clean_labels, _ = _clean_binary(values, labels)
    _require_two_classes(clean_labels)
    raw_auc = float(roc_auc_score(clean_labels, clean_values))
    direction = 1 if raw_auc >= 0.5 else -1
    aligned = direction * clean_values
    mean = float(np.mean(clean_values))
    standard_deviation = float(np.std(clean_values, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError(f"Feature {feature_id!r} is constant in the training data")
    thresholds = learn_operating_thresholds(aligned, clean_labels)
    folded_auc, folded_effect_gain = _folded_association(
        clean_values,
        clean_labels,
        raw_auc,
        feature_id,
    )
    return MarkerModel(
        feature_id=feature_id,
        direction=direction,
        mean=mean,
        standard_deviation=standard_deviation,
        thresholds=thresholds,
        n_train=len(clean_values),
        n_positive_train=int(np.sum(clean_labels == 1)),
        n_negative_train=int(np.sum(clean_labels == 0)),
        folded_auc=folded_auc,
        folded_effect_gain=folded_effect_gain,
    )


def _is_circular_feature(feature_id: object) -> bool:
    """Is this marker an angular coordinate on a full turn?

    Asked of the projection module, which owns the list, so the two cannot
    disagree about which markers wrap. An import failure means "treat it as an
    ordinary scale", which is the behaviour before this existed.
    """

    try:
        from ..core.biomarkers.circular_projection_biomarkers import (
            CIRCULAR_SOURCE_MARKERS,
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not break the fit
        return False
    return str(feature_id) in CIRCULAR_SOURCE_MARKERS


def _circular_fold(values: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Angular distance from the controls' mean direction, in degrees.

    The reference is the resultant mean direction of the controls rather than
    their median, because a median of angles is not defined without first
    choosing a branch -- which is the very thing that goes wrong here. The
    result lies in [0, 180] and is continuous across the seam.
    """

    radians = np.radians(values.astype(float))
    control = radians[~positive]
    reference = np.arctan2(
        float(np.mean(np.sin(control))), float(np.mean(np.cos(control)))
    )
    delta = np.abs(np.degrees(radians - reference)) % 360.0
    return np.minimum(delta, 360.0 - delta)


def _folded_association(
    values: np.ndarray,
    labels: np.ndarray,
    raw_auc: float,
    feature_id: object = "",
) -> tuple[float, float]:
    """How much a U-shaped reading would add, as a diagnostic only.

    Every effect measure this project screens with -- ROC AUC and the aligned
    rank-biserial correlation -- is rank-based, so all of them are blind to an
    association where both tails mark the positive class. The direction flip
    above rescues an *inverted* monotone marker; nothing rescues a folded one,
    and its AUC sits at 0,5 while the marker is discarded.

    Platonov (J Electrocardiol 2026) names exactly this as a methodological
    weakness of large-scale automated ECG analysis, and P-wave duration is the
    textbook case: very short and very long are both abnormal.

    Measured before this was written, over 17.290 marker-endpoint pairs from
    the existing feature partitions: **461 pairs (2,67 %)** separate better
    after folding by more than 0,05 in AUC, and **271 of those are invisible
    to the current screen** (raw |AUC - 0,5| below 0,02). Among the largest
    were ``qrs_loop_sagittal_yz_winding_number`` against three separate
    infarct endpoints (raw 0,49 against folded 0,65-0,67) and ``svg_azimuth``
    against strict echocardiographic LVH (0,52 against 0,66, 307 positives).

    **Angles are folded on the circle, not on the line.** For the twenty-four
    markers that are angular coordinates, ``|value - median|`` is arithmetic
    on a scale that does not exist: it makes -179 and +179 degrees look 358
    degrees apart. Those markers are folded by angular distance from the
    controls' mean direction instead. The ``svg_azimuth`` figure above was
    obtained with the linear fold and is kept as the record of what prompted
    the check; the circular fold is what this function now computes, and the
    wrap those markers suffer from is addressed at its source by the
    projection markers in ``circular_projection_biomarkers``.

    **This changes no decision.** It is not a gate, it enters no FDR family
    and no claim level. It is a number that says "this marker may carry a
    two-tailed association the screen cannot see", so that a reader can look.
    Turning it into a second inferential arm would need its own multiplicity
    accounting and its own pre-registration, which is a larger step than a
    diagnostic and not one to take by implication.
    """

    positive = labels == 1
    if _is_circular_feature(feature_id):
        folded = _circular_fold(values, positive)
    else:
        reference = float(np.median(values[~positive]))
        if not np.isfinite(reference):
            return float("nan"), float("nan")
        folded = np.abs(values - reference)
    if not np.all(np.isfinite(folded)) or float(np.std(folded)) <= 0:
        return float("nan"), float("nan")
    folded_auc = float(roc_auc_score(labels, folded))
    gain = abs(folded_auc - 0.5) - abs(float(raw_auc) - 0.5)
    return folded_auc, float(gain)


def learn_operating_thresholds(
    aligned_values: Iterable[float],
    labels: Iterable[int],
) -> dict[str, float]:
    values, binary, _ = _clean_binary(aligned_values, labels)
    _require_two_classes(binary)
    fpr, tpr, thresholds = roc_curve(
        binary,
        values,
        drop_intermediate=False,
    )
    specificity = 1.0 - fpr
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite candidate threshold is available")
    fpr = fpr[finite]
    tpr = tpr[finite]
    specificity = specificity[finite]
    thresholds = thresholds[finite]

    return {
        "specificity_90": _threshold_for_minimum_specificity(
            thresholds, tpr, specificity, 0.90
        ),
        "specificity_95": _threshold_for_minimum_specificity(
            thresholds, tpr, specificity, 0.95
        ),
        "sensitivity_90": _threshold_for_minimum_sensitivity(
            thresholds, tpr, specificity, 0.90
        ),
        "youden_exploratory": float(
            thresholds[int(np.argmax(tpr + specificity - 1.0))]
        ),
    }


def learn_threshold_for_target(
    aligned_values: Iterable[float],
    labels: Iterable[int],
    *,
    target_specificity: Optional[float] = None,
    target_sensitivity: Optional[float] = None,
) -> float:
    if (target_specificity is None) == (target_sensitivity is None):
        raise ValueError(
            "Specify exactly one of target_specificity or target_sensitivity"
        )
    values, binary, _ = _clean_binary(aligned_values, labels)
    _require_two_classes(binary)
    fpr, tpr, thresholds = roc_curve(
        binary,
        values,
        drop_intermediate=False,
    )
    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    tpr = tpr[finite]
    specificity = (1.0 - fpr)[finite]
    if target_specificity is not None:
        return _threshold_for_minimum_specificity(
            thresholds,
            tpr,
            specificity,
            float(np.clip(target_specificity, 0.0, 1.0)),
        )
    return _threshold_for_minimum_sensitivity(
        thresholds,
        tpr,
        specificity,
        float(np.clip(target_sensitivity, 0.0, 1.0)),
    )


def bootstrap_operating_threshold_bands(
    subject_ids: Iterable,
    values: Iterable[float],
    labels: Iterable[int],
    *,
    direction: int,
    point_thresholds: Optional[Mapping[str, float]] = None,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int | str]]:
    """Estimate train-threshold uncertainty with a subject-cluster bootstrap.

    Thresholds are always fitted inside each resampled training set. Returned
    limits are available in both aligned and original marker coordinates, so
    callers never need to infer how a negatively directed marker is applied.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if iterations < 0:
        raise ValueError("iterations cannot be negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    raw_subjects = np.asarray(subject_ids)
    raw_values = np.asarray(values)
    raw_labels = np.asarray(labels)
    if not (
        len(raw_subjects) == len(raw_values) == len(raw_labels)
    ):
        raise ValueError(
            "subject_ids, values and labels must have matching lengths"
        )
    clean_values, clean_labels, mask = _clean_binary(raw_values, raw_labels)
    subjects = raw_subjects[mask]
    _require_two_classes(clean_labels)
    aligned = int(direction) * clean_values
    fitted = (
        {str(key): float(value) for key, value in point_thresholds.items()}
        if point_thresholds is not None
        else learn_operating_thresholds(aligned, clean_labels)
    )
    threshold_names = tuple(fitted)
    estimates: dict[str, list[float]] = {
        name: [] for name in threshold_names
    }
    successful = 0

    unique_subjects = pd.unique(subjects)
    if iterations > 0 and len(unique_subjects) >= 2:
        rows_by_subject = {
            subject: np.flatnonzero(subjects == subject)
            for subject in unique_subjects
        }
        rng = np.random.default_rng(seed)
        for _ in range(int(iterations)):
            sampled = rng.choice(
                unique_subjects,
                size=len(unique_subjects),
                replace=True,
            )
            indices = np.concatenate(
                [rows_by_subject[subject] for subject in sampled]
            )
            sampled_labels = clean_labels[indices]
            if np.unique(sampled_labels).size < 2:
                continue
            try:
                bootstrap_thresholds = learn_operating_thresholds(
                    aligned[indices],
                    sampled_labels,
                )
            except (ValueError, FloatingPointError):
                continue
            if not all(
                np.isfinite(bootstrap_thresholds.get(name, np.nan))
                for name in threshold_names
            ):
                continue
            successful += 1
            for name in threshold_names:
                estimates[name].append(float(bootstrap_thresholds[name]))

    alpha = (1.0 - confidence) / 2.0
    output: dict[str, dict[str, float | int | str]] = {}
    for name, point_aligned in fitted.items():
        samples = estimates[name]
        if samples:
            low_aligned, high_aligned = np.quantile(
                samples,
                (alpha, 1.0 - alpha),
            )
            method = "train_subject_cluster_bootstrap"
        else:
            low_aligned = high_aligned = point_aligned
            method = "point_only_not_estimated"
        raw_limits = sorted(
            (
                float(direction * low_aligned),
                float(direction * high_aligned),
            )
        )
        output[name] = {
            "threshold_aligned": float(point_aligned),
            "threshold_original": float(direction * point_aligned),
            "gray_zone_aligned_low": float(low_aligned),
            "gray_zone_aligned_high": float(high_aligned),
            "gray_zone_original_low": raw_limits[0],
            "gray_zone_original_high": raw_limits[1],
            "direction": int(direction),
            "uncertainty_method": method,
            "bootstrap_iterations": int(iterations),
            "successful_bootstrap_iterations": int(successful),
        }
    return output


@profiled_phase("marker_metrics")
def evaluate_marker(
    values: Iterable[float],
    labels: Iterable[int],
    *,
    direction: int = 1,
    thresholds: Optional[Mapping[str, float]] = None,
    observed_prevalence: Optional[float] = None,
    minimum_test_yield: float = MINIMUM_CONFIRMATORY_TEST_YIELD,
    maximum_class_yield_gap: float = MAXIMUM_CONFIRMATORY_CLASS_YIELD_GAP,
) -> dict[str, float | int | bool | str | None]:
    assessability = marker_assessability(
        values,
        labels,
        minimum_test_yield=minimum_test_yield,
        maximum_class_yield_gap=maximum_class_yield_gap,
    )
    clean_values, clean_labels, _ = _clean_binary(values, labels)
    result = _empty_metric_result()
    result.update(
        {
            "n_total": int(len(clean_values)),
            "n_positive": int(np.sum(clean_labels == 1)),
            "n_negative": int(np.sum(clean_labels == 0)),
            **assessability,
        }
    )
    sensitivity = _missingness_auc_sensitivity(
        values,
        labels,
        direction=direction,
        assessability_gate_passed=bool(
            assessability["assessability_gate_passed"]
        ),
    )
    # Carried out of band: it belongs to the caller, not to the result table.
    observed_aligned_auc = float(sensitivity.pop("observed_aligned_auc"))
    result.update(sensitivity)
    result["confirmatory_assessability_gate_passed"] = bool(
        result["assessability_gate_passed"]
        and result["missingness_sensitivity_gate_passed"]
    )
    if len(clean_values) == 0 or len(np.unique(clean_labels)) < 2:
        return result

    aligned = int(direction) * clean_values
    # ``roc_auc_score`` was called three times on what is, at most, two distinct
    # rankings.  The aligned AUC is reused rather than recomputed, and for
    # direction ``+1`` the aligned scores are the clean values multiplied by 1.0
    # -- the same array, bit for bit -- so the raw AUC is the same number and is
    # reused too.  Only a flipped marker still needs the second ranking.
    aligned_auc = (
        observed_aligned_auc
        if math.isfinite(observed_aligned_auc)
        else float(roc_auc_score(clean_labels, aligned))
    )
    raw_auc = (
        aligned_auc
        if int(direction) > 0
        else float(roc_auc_score(clean_labels, clean_values))
    )
    result.update(
        {
            "roc_auc": aligned_auc,
            "average_precision": float(
                average_precision_score(clean_labels, aligned)
            ),
            "pr_auc": exact_pr_auc(clean_labels, aligned),
            "p_value": mann_whitney_p_value(clean_values, clean_labels),
            "rank_biserial": float(2.0 * raw_auc - 1.0),
            "aligned_rank_biserial": float(2.0 * aligned_auc - 1.0),
        }
    )
    prevalence = (
        float(np.mean(clean_labels))
        if observed_prevalence is None
        else float(observed_prevalence)
    )
    for name, threshold in (thresholds or {}).items():
        result.update(
            {
                f"{name}_{key}": value
                for key, value in classification_metrics(
                    clean_labels,
                    aligned >= float(threshold),
                    prevalence=prevalence,
                ).items()
            }
        )
        result[f"{name}_threshold_aligned"] = float(threshold)
        result[f"{name}_threshold_original"] = float(direction * threshold)
        result[f"{name}_threshold_direction"] = (
            "above" if direction > 0 else "below"
        )
        target = OPERATING_POINT_TARGETS.get(str(name))
        if target is not None:
            metric, minimum = target
            achieved = result.get(f"{name}_{metric}")
            result[f"{name}_target_attainable"] = bool(
                achieved is not None
                and np.isfinite(float(achieved))
                and float(achieved) >= minimum - OPERATING_POINT_TARGET_TOLERANCE
            )
            # Compatibility flag above refers only to this evaluation sample.
            # It must not be interpreted as training-threshold attainability.
            result[f"{name}_observed_target_met"] = result[f"{name}_target_attainable"]
            result[f"{name}_target_value"] = minimum
            result[f"{name}_target_metric"] = metric
            result[f"{name}_target_evaluation_scope"] = "current_evaluation_sample"
    return result


def marker_assessability(
    values: Iterable[float],
    labels: Iterable[int],
    *,
    minimum_test_yield: float = MINIMUM_CONFIRMATORY_TEST_YIELD,
    maximum_class_yield_gap: float = MAXIMUM_CONFIRMATORY_CLASS_YIELD_GAP,
) -> dict[str, float | int | bool | str]:
    """Describe the complete-case estimand and its differential test yield.

    ``n_input`` contains rows with an assessable binary reference label, while
    ``n_evaluable`` additionally requires a finite marker value.  The gate is a
    conservative confirmatory guard, not evidence that missingness is random.
    Failure therefore triggers (rather than substitutes for) a dedicated
    missing-data sensitivity analysis.
    """
    if not 0.0 <= float(minimum_test_yield) <= 1.0:
        raise ValueError("minimum_test_yield must be between zero and one")
    if not 0.0 <= float(maximum_class_yield_gap) <= 1.0:
        raise ValueError("maximum_class_yield_gap must be between zero and one")
    values_array = np.asarray(values, dtype=float)
    labels_array = np.asarray(labels, dtype=float)
    if values_array.shape != labels_array.shape:
        raise ValueError("Values and labels must have matching shapes")

    label_assessable = np.isfinite(labels_array) & np.isin(labels_array, (0, 1))
    value_assessable = np.isfinite(values_array)
    evaluable = label_assessable & value_assessable
    positive_input = label_assessable & (labels_array == 1)
    negative_input = label_assessable & (labels_array == 0)
    positive_evaluable = positive_input & value_assessable
    negative_evaluable = negative_input & value_assessable

    return _marker_assessability_from_counts(
        n_positive_input=int(np.sum(positive_input)),
        n_negative_input=int(np.sum(negative_input)),
        n_positive_evaluable=int(np.sum(positive_evaluable)),
        n_negative_evaluable=int(np.sum(negative_evaluable)),
        minimum_test_yield=minimum_test_yield,
        maximum_class_yield_gap=maximum_class_yield_gap,
    )


def _marker_assessability_from_counts(
    *,
    n_positive_input: int,
    n_negative_input: int,
    n_positive_evaluable: int,
    n_negative_evaluable: int,
    minimum_test_yield: float = MINIMUM_CONFIRMATORY_TEST_YIELD,
    maximum_class_yield_gap: float = MAXIMUM_CONFIRMATORY_CLASS_YIELD_GAP,
) -> dict[str, float | int | bool | str]:
    """Exact assessability contract without materialising row-sized arrays."""
    counts = (
        n_positive_input,
        n_negative_input,
        n_positive_evaluable,
        n_negative_evaluable,
    )
    if any(int(value) != value or value < 0 for value in counts):
        raise ValueError("Marker assessability counts must be non-negative integers")
    if n_positive_evaluable > n_positive_input:
        raise ValueError("Evaluable positive count exceeds the input count")
    if n_negative_evaluable > n_negative_input:
        raise ValueError("Evaluable negative count exceeds the input count")

    n_input = int(n_positive_input + n_negative_input)
    n_evaluable = int(n_positive_evaluable + n_negative_evaluable)
    test_yield = _safe_ratio(n_evaluable, n_input)
    positive_test_yield = _safe_ratio(n_positive_evaluable, n_positive_input)
    negative_test_yield = _safe_ratio(n_negative_evaluable, n_negative_input)
    class_yields = np.asarray(
        (positive_test_yield, negative_test_yield),
        dtype=float,
    )
    test_yield_gap = (
        float(abs(positive_test_yield - negative_test_yield))
        if np.isfinite(class_yields).all()
        else float("nan")
    )

    reasons: list[str] = []
    if n_input == 0:
        reasons.append("no_assessable_reference_labels")
    if n_positive_input == 0 or n_negative_input == 0:
        reasons.append("reference_class_unavailable")
    if not np.isfinite(test_yield) or test_yield < float(minimum_test_yield):
        reasons.append("low_overall_test_yield")
    if (
        not np.isfinite(class_yields).all()
        or float(np.min(class_yields)) < float(minimum_test_yield)
    ):
        reasons.append("low_class_specific_test_yield")
    if (
        np.isfinite(test_yield_gap)
        and test_yield_gap > float(maximum_class_yield_gap)
    ):
        reasons.append("differential_test_yield")
    gate_passed = not reasons
    return {
        "n_input": n_input,
        "n_evaluable": n_evaluable,
        "n_positive_input": n_positive_input,
        "n_negative_input": n_negative_input,
        "n_positive_evaluable": n_positive_evaluable,
        "n_negative_evaluable": n_negative_evaluable,
        "test_yield": test_yield,
        "positive_test_yield": positive_test_yield,
        "negative_test_yield": negative_test_yield,
        "test_yield_gap": test_yield_gap,
        "overall_missingness": (
            float(1.0 - test_yield) if np.isfinite(test_yield) else float("nan")
        ),
        "complete_case_analysis": True,
        "complete_case_estimand": "marker_observed_and_binary_reference_assessable",
        "assessability_gate_version": ASSESSABILITY_GATE_VERSION,
        "assessability_minimum_test_yield": float(minimum_test_yield),
        "assessability_maximum_class_yield_gap": float(
            maximum_class_yield_gap
        ),
        "assessability_status": "acceptable" if gate_passed else "|".join(reasons),
        "assessability_gate_passed": gate_passed,
        "missingness_sensitivity_status": "pending_auc_bounds",
        "missingness_sensitivity_gate_passed": False,
    }


def screen_marker_assessability_from_counts(
    *,
    n_positive_input: int,
    n_negative_input: int,
    n_positive_evaluable: int,
    n_negative_evaluable: int,
    raw_auc: float,
    direction: int,
    minimum_test_yield: float = MINIMUM_CONFIRMATORY_TEST_YIELD,
    maximum_class_yield_gap: float = MAXIMUM_CONFIRMATORY_CLASS_YIELD_GAP,
) -> dict[str, float | int | bool | str]:
    """Assess a binary screen from sufficient statistics only.

    This is mathematically identical to the assessability and adversarial AUC
    bound subset of :func:`evaluate_marker`.  It exists for high-dimensional
    screens where copying and ranking the full case/control vector again for
    millions of marker-outcome pairs would dominate the entire run.
    """
    if int(direction) not in (-1, 1):
        raise ValueError("Marker direction must be +1 or -1")
    result = _marker_assessability_from_counts(
        n_positive_input=n_positive_input,
        n_negative_input=n_negative_input,
        n_positive_evaluable=n_positive_evaluable,
        n_negative_evaluable=n_negative_evaluable,
        minimum_test_yield=minimum_test_yield,
        maximum_class_yield_gap=maximum_class_yield_gap,
    )
    if n_positive_input == 0 or n_negative_input == 0:
        result.update(
            {
                "missingness_auc_lower_bound": float("nan"),
                "missingness_auc_upper_bound": float("nan"),
                "missingness_sensitivity_null_auc": (
                    MISSINGNESS_SENSITIVITY_NULL_AUC
                ),
                "missingness_sensitivity_status": "not_assessable",
                "missingness_sensitivity_gate_passed": False,
                "confirmatory_assessability_gate_passed": False,
            }
        )
        return result

    observed_pairs = n_positive_evaluable * n_negative_evaluable
    total_pairs = n_positive_input * n_negative_input
    if observed_pairs == 0 or not np.isfinite(raw_auc):
        lower_bound = 0.0
        upper_bound = 1.0
    else:
        aligned_auc = float(raw_auc) if direction > 0 else 1.0 - float(raw_auc)
        concordant = aligned_auc * observed_pairs
        lower_bound = float(concordant / total_pairs)
        upper_bound = float(
            (concordant + total_pairs - observed_pairs) / total_pairs
        )
    missing = (
        n_positive_evaluable != n_positive_input
        or n_negative_evaluable != n_negative_input
    )
    sensitivity_gate = bool(
        result["assessability_gate_passed"]
        and lower_bound > MISSINGNESS_SENSITIVITY_NULL_AUC
    )
    result.update(
        {
            "missingness_auc_lower_bound": lower_bound,
            "missingness_auc_upper_bound": upper_bound,
            "missingness_sensitivity_null_auc": (
                MISSINGNESS_SENSITIVITY_NULL_AUC
            ),
            "missingness_sensitivity_status": (
                "worst_case_auc_bounds_computed"
                if missing
                else "complete_marker_observation"
            ),
            "missingness_sensitivity_gate_passed": sensitivity_gate,
            "confirmatory_assessability_gate_passed": bool(
                result["assessability_gate_passed"] and sensitivity_gate
            ),
        }
    )
    return result


def _missingness_auc_sensitivity(
    values: Iterable[float],
    labels: Iterable[int],
    *,
    direction: int,
    assessability_gate_passed: bool,
) -> dict[str, float | bool | str]:
    """Sharp worst/best AUC bounds under arbitrary missing marker values."""
    values_array = np.asarray(values, dtype=float)
    labels_array = np.asarray(labels, dtype=float)
    label_mask = np.isfinite(labels_array) & np.isin(labels_array, (0, 1))
    scoped_labels = labels_array[label_mask].astype(np.int8)
    scoped_values = int(direction) * values_array[label_mask]
    missing = ~np.isfinite(scoped_values)
    if len(scoped_labels) == 0 or np.unique(scoped_labels).size < 2:
        return {
            "observed_aligned_auc": float("nan"),
            "missingness_auc_lower_bound": float("nan"),
            "missingness_auc_upper_bound": float("nan"),
            "missingness_sensitivity_null_auc": MISSINGNESS_SENSITIVITY_NULL_AUC,
            "missingness_sensitivity_status": "not_assessable",
            "missingness_sensitivity_gate_passed": False,
        }
    observed = scoped_values[~missing]
    if len(observed):
        low_extreme = float(np.min(observed) - max(np.ptp(observed), 1.0))
        high_extreme = float(np.max(observed) + max(np.ptp(observed), 1.0))
    else:
        low_extreme, high_extreme = -1.0, 1.0
    # The bounds are the observed AUC rescaled, not two more rankings.
    #
    # Every missing positive is placed strictly below every observed value and
    # every missing negative strictly above it (and vice versa for the upper
    # bound), so each pair involving a missing value contributes a known 0 or 1
    # and the observed pairs keep their contribution unchanged:
    #
    #     lower = AUC_obs * (n+_obs * n-_obs) / (n+ * n-)
    #     upper = [AUC_obs * (n+_obs * n-_obs) + (n+ * n- - n+_obs * n-_obs)]
    #             / (n+ * n-)
    #
    # That replaces two O(n log n) rankings per marker and endpoint with one.
    # At roughly five million marker-by-phenotype pairs per split it is the
    # difference between ranking the cohort three times and ranking it once.
    n_positive = int(np.sum(scoped_labels == 1))
    n_negative = int(np.sum(scoped_labels == 0))
    observed_positive = int(np.sum((scoped_labels == 1) & ~missing))
    observed_negative = int(np.sum((scoped_labels == 0) & ~missing))
    observed_pairs = observed_positive * observed_negative
    total_pairs = n_positive * n_negative
    if observed_pairs == 0 or total_pairs == 0:
        observed_auc = float("nan")
        lower_bound = 0.0
        upper_bound = 1.0
    else:
        observed_auc = float(
            roc_auc_score(scoped_labels[~missing], scoped_values[~missing])
        )
        concordant = observed_auc * observed_pairs
        lower_bound = float(concordant / total_pairs)
        upper_bound = float(
            (concordant + (total_pairs - observed_pairs)) / total_pairs
        )
    gate_passed = (
        assessability_gate_passed
        and lower_bound > MISSINGNESS_SENSITIVITY_NULL_AUC
    )
    return {
        "observed_aligned_auc": observed_auc,
        "missingness_auc_lower_bound": lower_bound,
        "missingness_auc_upper_bound": upper_bound,
        "missingness_sensitivity_null_auc": MISSINGNESS_SENSITIVITY_NULL_AUC,
        "missingness_sensitivity_status": (
            "complete_marker_observation"
            if not missing.any()
            else "worst_case_auc_bounds_computed"
        ),
        "missingness_sensitivity_gate_passed": gate_passed,
    }


def classification_metrics(
    labels: Iterable[int],
    predictions: Iterable[bool],
    *,
    prevalence: Optional[float] = None,
) -> dict[str, float | int]:
    binary = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(predictions, dtype=bool)
    if binary.shape != predicted.shape:
        raise ValueError("Labels and predictions must have matching shapes")
    tp = int(np.sum((binary == 1) & predicted))
    fp = int(np.sum((binary == 0) & predicted))
    tn = int(np.sum((binary == 0) & ~predicted))
    fn = int(np.sum((binary == 1) & ~predicted))
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    observed_ppv = _safe_ratio(tp, tp + fp)
    observed_npv = _safe_ratio(tn, tn + fn)
    target_prevalence = (
        _safe_ratio(tp + fn, len(binary))
        if prevalence is None
        else float(np.clip(prevalence, 0.0, 1.0))
    )
    ppv, npv = predictive_values(
        sensitivity,
        specificity,
        target_prevalence,
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "observed_ppv": observed_ppv,
        "observed_npv": observed_npv,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def predictive_values(
    sensitivity: float,
    specificity: float,
    prevalence: float,
) -> tuple[float, float]:
    prevalence = float(np.clip(prevalence, 0.0, 1.0))
    ppv_denominator = (
        sensitivity * prevalence
        + (1.0 - specificity) * (1.0 - prevalence)
    )
    npv_denominator = (
        specificity * (1.0 - prevalence)
        + (1.0 - sensitivity) * prevalence
    )
    ppv = (
        sensitivity * prevalence / ppv_denominator
        if ppv_denominator > 0
        else float("nan")
    )
    npv = (
        specificity * (1.0 - prevalence) / npv_denominator
        if npv_denominator > 0
        else float("nan")
    )
    return float(ppv), float(npv)


def exact_pr_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    binary = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=float)
    precision, recall, _ = precision_recall_curve(binary, values)
    return float(sklearn_auc(recall[::-1], precision[::-1]))


def mann_whitney_p_value(
    values: Iterable[float],
    labels: Iterable[int],
) -> float:
    clean_values, clean_labels, _ = _clean_binary(values, labels)
    positive = clean_values[clean_labels == 1]
    negative = clean_values[clean_labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        return 1.0
    if np.unique(clean_values).size == 1:
        return 1.0
    result = scipy_stats.mannwhitneyu(
        positive,
        negative,
        alternative="two-sided",
        method="asymptotic",
    )
    return float(result.pvalue) if np.isfinite(result.pvalue) else 1.0


def subject_cluster_bootstrap(
    subject_ids: Iterable,
    values: Iterable[float],
    labels: Iterable[int],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if iterations <= 0:
        return float("nan"), float("nan")
    clean_values, clean_labels, mask = _clean_binary(values, labels)
    subjects = np.asarray(subject_ids)[mask]
    if len(subjects) != len(clean_values):
        raise ValueError("subject_ids, values and labels must have matching lengths")
    unique_subjects = pd.unique(subjects)
    if len(unique_subjects) < 2:
        return float("nan"), float("nan")
    rows_by_subject = {
        subject: np.flatnonzero(subjects == subject)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(int(iterations)):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([rows_by_subject[subject] for subject in sampled])
        sampled_labels = clean_labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        try:
            estimate = float(metric(clean_values[indices], sampled_labels))
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def paired_auc_difference_bootstrap(
    subject_ids: Iterable,
    labels: Iterable[int],
    candidate_scores: Iterable[float],
    baseline_scores: Iterable[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    subjects = np.asarray(subject_ids)
    binary = np.asarray(labels, dtype=float)
    candidate = np.asarray(candidate_scores, dtype=float)
    baseline = np.asarray(baseline_scores, dtype=float)
    if not (
        subjects.shape
        == binary.shape
        == candidate.shape
        == baseline.shape
    ):
        raise ValueError("Paired AUC inputs must have matching shapes")
    valid = (
        np.isfinite(binary)
        & np.isin(binary, (0, 1))
        & np.isfinite(candidate)
        & np.isfinite(baseline)
    )
    subjects = subjects[valid]
    binary = binary[valid].astype(np.int8)
    candidate = candidate[valid]
    baseline = baseline[valid]
    result: dict[str, float | int] = {
        "paired_n": int(len(binary)),
        "paired_n_positive": int(np.sum(binary == 1)),
        "paired_n_negative": int(np.sum(binary == 0)),
        "candidate_auc_paired": float("nan"),
        "baseline_auc_paired": float("nan"),
        "incremental_auc": float("nan"),
        "incremental_auc_ci_low": float("nan"),
        "incremental_auc_ci_high": float("nan"),
        "incremental_auc_bootstrap_iterations": 0,
    }
    if len(binary) == 0 or np.unique(binary).size < 2:
        return result
    candidate_auc = float(roc_auc_score(binary, candidate))
    baseline_auc = float(roc_auc_score(binary, baseline))
    result.update(
        {
            "candidate_auc_paired": candidate_auc,
            "baseline_auc_paired": baseline_auc,
            "incremental_auc": candidate_auc - baseline_auc,
        }
    )
    if iterations <= 0:
        return result
    unique_subjects = pd.unique(subjects)
    rows_by_subject = {
        subject: np.flatnonzero(subjects == subject)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(int(iterations)):
        sampled = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        indices = np.concatenate(
            [rows_by_subject[subject] for subject in sampled]
        )
        sampled_labels = binary[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        differences.append(
            float(
                roc_auc_score(sampled_labels, candidate[indices])
                - roc_auc_score(sampled_labels, baseline[indices])
            )
        )
    if differences:
        low, high = np.quantile(differences, (0.025, 0.975))
        result["incremental_auc_ci_low"] = float(low)
        result["incremental_auc_ci_high"] = float(high)
    result["incremental_auc_bootstrap_iterations"] = int(iterations)
    return result


def paired_discrimination_difference_bootstrap(
    subject_ids: Iterable,
    labels: Iterable[int],
    candidate_scores: Iterable[float],
    baseline_scores: Iterable[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    subjects = np.asarray(subject_ids)
    binary = np.asarray(labels, dtype=float)
    candidate = np.asarray(candidate_scores, dtype=float)
    baseline = np.asarray(baseline_scores, dtype=float)
    if not (
        subjects.shape
        == binary.shape
        == candidate.shape
        == baseline.shape
    ):
        raise ValueError("Paired discrimination inputs must have matching shapes")
    valid = (
        np.isfinite(binary)
        & np.isin(binary, (0, 1))
        & np.isfinite(candidate)
        & np.isfinite(baseline)
    )
    subjects = subjects[valid]
    binary = binary[valid].astype(np.int8)
    candidate = candidate[valid]
    baseline = baseline[valid]
    result: dict[str, float | int] = {
        "paired_n": int(len(binary)),
        "paired_n_positive": int(np.sum(binary == 1)),
        "paired_n_negative": int(np.sum(binary == 0)),
        "candidate_auc_paired": float("nan"),
        "baseline_auc_paired": float("nan"),
        "delta_auc": float("nan"),
        "delta_auc_ci_low": float("nan"),
        "delta_auc_ci_high": float("nan"),
        "candidate_average_precision_paired": float("nan"),
        "baseline_average_precision_paired": float("nan"),
        "delta_average_precision": float("nan"),
        "delta_average_precision_ci_low": float("nan"),
        "delta_average_precision_ci_high": float("nan"),
        "bootstrap_iterations": int(max(0, iterations)),
        "bootstrap_successful_iterations": 0,
    }
    if len(binary) == 0 or np.unique(binary).size < 2:
        return result
    candidate_auc = float(roc_auc_score(binary, candidate))
    baseline_auc = float(roc_auc_score(binary, baseline))
    candidate_ap = float(average_precision_score(binary, candidate))
    baseline_ap = float(average_precision_score(binary, baseline))
    result.update(
        {
            "candidate_auc_paired": candidate_auc,
            "baseline_auc_paired": baseline_auc,
            "delta_auc": candidate_auc - baseline_auc,
            "candidate_average_precision_paired": candidate_ap,
            "baseline_average_precision_paired": baseline_ap,
            "delta_average_precision": candidate_ap - baseline_ap,
        }
    )
    if iterations <= 0:
        return result
    unique_subjects = pd.unique(subjects)
    rows_by_subject = {
        subject: np.flatnonzero(subjects == subject)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    auc_differences: list[float] = []
    ap_differences: list[float] = []
    for _ in range(int(iterations)):
        sampled = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        indices = np.concatenate(
            [rows_by_subject[subject] for subject in sampled]
        )
        sampled_labels = binary[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        auc_differences.append(
            float(
                roc_auc_score(sampled_labels, candidate[indices])
                - roc_auc_score(sampled_labels, baseline[indices])
            )
        )
        ap_differences.append(
            float(
                average_precision_score(
                    sampled_labels,
                    candidate[indices],
                )
                - average_precision_score(
                    sampled_labels,
                    baseline[indices],
                )
            )
        )
    if auc_differences:
        auc_low, auc_high = np.quantile(
            auc_differences,
            (0.025, 0.975),
        )
        ap_low, ap_high = np.quantile(
            ap_differences,
            (0.025, 0.975),
        )
        result.update(
            {
                "delta_auc_ci_low": float(auc_low),
                "delta_auc_ci_high": float(auc_high),
                "delta_average_precision_ci_low": float(ap_low),
                "delta_average_precision_ci_high": float(ap_high),
                "bootstrap_successful_iterations": len(auc_differences),
            }
        )
    return result


def paired_operating_point_bootstrap(
    subject_ids: Iterable,
    labels: Iterable[int],
    candidate_predictions: Iterable[bool],
    baseline_predictions: Iterable[bool],
    *,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    if metric not in ("sensitivity", "specificity"):
        raise ValueError("metric must be sensitivity or specificity")
    subjects = np.asarray(subject_ids)
    binary = np.asarray(labels, dtype=float)
    candidate_raw = np.asarray(candidate_predictions)
    baseline_raw = np.asarray(baseline_predictions)
    if not (
        subjects.shape
        == binary.shape
        == candidate_raw.shape
        == baseline_raw.shape
    ):
        raise ValueError("Paired operating-point inputs must have matching shapes")
    valid = np.isfinite(binary) & np.isin(binary, (0, 1))
    if np.issubdtype(candidate_raw.dtype, np.number):
        valid &= np.isfinite(candidate_raw.astype(float))
    if np.issubdtype(baseline_raw.dtype, np.number):
        valid &= np.isfinite(baseline_raw.astype(float))
    subjects = subjects[valid]
    binary = binary[valid].astype(np.int8)
    candidate = candidate_raw[valid].astype(bool)
    baseline = baseline_raw[valid].astype(bool)
    candidate_metrics = classification_metrics(binary, candidate)
    baseline_metrics = classification_metrics(binary, baseline)
    candidate_value = float(candidate_metrics[metric])
    baseline_value = float(baseline_metrics[metric])
    result: dict[str, float | int] = {
        "paired_n": int(len(binary)),
        "paired_n_positive": int(np.sum(binary == 1)),
        "paired_n_negative": int(np.sum(binary == 0)),
        f"candidate_{metric}": candidate_value,
        f"baseline_{metric}": baseline_value,
        f"delta_{metric}": candidate_value - baseline_value,
        f"delta_{metric}_ci_low": float("nan"),
        f"delta_{metric}_ci_high": float("nan"),
        "bootstrap_iterations": int(max(0, iterations)),
        "bootstrap_successful_iterations": 0,
    }
    if (
        iterations <= 0
        or len(binary) == 0
        or np.unique(binary).size < 2
    ):
        return result
    unique_subjects = pd.unique(subjects)
    rows_by_subject = {
        subject: np.flatnonzero(subjects == subject)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(int(iterations)):
        sampled = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        indices = np.concatenate(
            [rows_by_subject[subject] for subject in sampled]
        )
        sampled_labels = binary[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        candidate_metric = classification_metrics(
            sampled_labels,
            candidate[indices],
        )[metric]
        baseline_metric = classification_metrics(
            sampled_labels,
            baseline[indices],
        )[metric]
        difference = float(candidate_metric) - float(baseline_metric)
        if np.isfinite(difference):
            differences.append(difference)
    if differences:
        low, high = np.quantile(differences, (0.025, 0.975))
        result[f"delta_{metric}_ci_low"] = float(low)
        result[f"delta_{metric}_ci_high"] = float(high)
    result["bootstrap_successful_iterations"] = len(differences)
    return result


# Screening a marker out is much cheaper than measuring it precisely.
#
# A subject-cluster bootstrap over 640k rows costs about 2.6 s per marker and
# scales linearly with rows; at 1,753 markers and 72 endpoints that is the
# dominant cost of the whole validation. Almost all of it is spent on markers
# that will never be reported, because their discrimination is indistinguishable
# from chance.
#
# Those can be identified without resampling. Hanley-McNeil gives a closed-form
# standard error for the AUC, and dividing the sample size by the design effect
# DEFF = 1 + (m - 1) * ICC accounts for the clustering the raw formula ignores.
#
# Measured against the cluster bootstrap on simulated five-ECG-per-subject data
# (2026-08-19, see DELINEATION_ABSTENTION_2026-08-18.md sibling notes):
#
#     ICC 0.74, DEFF 3.95   bootstrap 0.0580   raw HM 0.0339   corrected 0.0674
#     ICC 0.74, DEFF 3.97   bootstrap 0.0267   raw HM 0.0145   corrected 0.0289
#     ICC 0.31, DEFF 2.23   bootstrap 0.0185   raw HM 0.0142   corrected 0.0212
#
# The raw formula understates the interval by up to 46 % -- it must never be
# used here. The corrected one runs 8-16 % *wider* than the bootstrap, so it
# errs towards keeping a marker rather than discarding it. That is the only
# direction in which this screen is safe, and it is why the analytic interval
# is allowed to skip work but never to replace a reported interval.
SCREENING_DESIGN_EFFECT_MINIMUM = 1.0
# There is deliberately no default for the negligible-AUC margin. It states
# what discrimination the study considers too small to report, which is a
# scientific commitment, not an implementation detail -- and the screen does
# nothing at all unless a caller supplies it.


def clustered_auc_screening_interval(
    subject_codes: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
) -> Optional[tuple[float, float, float, float]]:
    """Conservative (auc, low, high, design_effect) without resampling.

    Returns None when the closed form does not apply -- too few subjects, one
    class absent -- so the caller falls back to the bootstrap rather than to a
    guess.
    """

    positive = values[labels == 1]
    negative = values[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    sorted_negative = np.sort(negative)
    below = np.searchsorted(sorted_negative, positive, side="left")
    equal = np.searchsorted(sorted_negative, positive, side="right") - below
    auc = float(
        (below + 0.5 * equal).sum() / (len(positive) * len(negative))
    )

    # One-way random-effects ICC on the marker values, which is what carries
    # the within-subject dependence the bootstrap resamples around.
    order = np.argsort(subject_codes, kind="stable")
    ordered_codes = subject_codes[order]
    ordered_values = values[order]
    boundaries = np.flatnonzero(np.diff(ordered_codes)) + 1
    groups = np.split(ordered_values, boundaries)
    sizes = np.array([len(group) for group in groups], dtype=float)
    if len(groups) < 2:
        return None
    if sizes.max() < 2:
        # Exactly one observation per patient is already independent at the
        # bootstrap unit.  An ICC is neither needed nor estimable; DEFF=1 is
        # the exact clustered-design correction.  Returning ``None`` here
        # used to disable the safe null-marker screen for the common
        # deterministic index-ECG design and forced a full bootstrap for every
        # clearly uninformative marker.
        design_effect = SCREENING_DESIGN_EFFECT_MINIMUM
    else:
        means = np.array([group.mean() for group in groups])
        within = float(
            np.mean(
                [
                    group.var(ddof=1)
                    for group in groups
                    if len(group) > 1
                ]
            )
        )
        mean_size = float(sizes.mean())
        between = float(means.var(ddof=1)) - within / max(
            mean_size, 1.0
        )
        between = max(between, 0.0)
        total = between + within
        icc = (between / total) if total > 0 else 0.0
        design_effect = max(
            SCREENING_DESIGN_EFFECT_MINIMUM,
            1.0 + (mean_size - 1.0) * icc,
        )

    effective_positive = max(1.0, len(positive) / design_effect)
    effective_negative = max(1.0, len(negative) / design_effect)
    q1 = auc / (2.0 - auc) if (2.0 - auc) != 0 else 0.0
    q2 = (2.0 * auc * auc) / (1.0 + auc) if (1.0 + auc) != 0 else 0.0
    variance = (
        auc * (1.0 - auc)
        + (effective_positive - 1.0) * (q1 - auc * auc)
        + (effective_negative - 1.0) * (q2 - auc * auc)
    ) / (effective_positive * effective_negative)
    if not np.isfinite(variance) or variance < 0:
        return None
    half = 1.959963984540054 * float(np.sqrt(variance))
    return auc, auc - half, auc + half, design_effect


# --- Refinable bootstrap -------------------------------------------------
#
# Resamples are exchangeable, so they accumulate: 500 draws followed by 1,500
# more are the same evidence as 2,000 drawn at once. Verified 2026-08-19 to be
# bit-identical when the draws come from a continuing stream.
#
# That makes an analysis resumable. A first pass can run coarse, and a later
# pass can add draws without discarding anything -- as long as each batch uses
# a stream nobody has used before. Carrying a live RNG between processes is
# fragile, so batches are addressed instead: batch k of a marker always draws
# from ``default_rng([seed, k])``. Resuming needs only the batch counter, and
# the same batch always yields the same draws no matter when it runs.
#
# The honest finding that came with it: the interval saturates. Measured on
# clustered data, the bound moved 0.00037 between 200 and 500 draws and 0.00037
# between 4,000 and 8,000 -- past a few hundred draws the interval wanders
# inside its own Monte-Carlo noise rather than converging on something better.
# So the refinement report is not a promise that more compute buys precision.
# It is the instrument that shows when it stops doing so.
BOOTSTRAP_BATCH_SIZE = 250


@dataclass
class RefinableBootstrapState:
    """Accumulated bootstrap evidence for one marker, resumable across runs."""

    seed: int
    completed_batches: int
    estimates: dict[str, list[float]]

    @property
    def iterations(self) -> int:
        return self.completed_batches * BOOTSTRAP_BATCH_SIZE

    def interval(self, metric: str = "roc_auc") -> tuple[float, float]:
        values = self.estimates.get(metric) or []
        if len(values) < 2:
            return (float("nan"), float("nan"))
        low, high = np.quantile(values, (0.025, 0.975))
        return (float(low), float(high))

    def to_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "completed_batches": int(self.completed_batches),
            "batch_size": BOOTSTRAP_BATCH_SIZE,
            "iterations": int(self.iterations),
            "estimates": {
                name: [float(value) for value in values]
                for name, values in self.estimates.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "RefinableBootstrapState":
        stored = int(payload.get("batch_size", BOOTSTRAP_BATCH_SIZE))
        if stored != BOOTSTRAP_BATCH_SIZE:
            raise ValueError(
                "Refinable bootstrap state was written with batch size "
                f"{stored}; batches address the random stream, so a different "
                "size would silently redraw the same numbers."
            )
        return cls(
            seed=int(payload["seed"]),
            completed_batches=int(payload["completed_batches"]),
            estimates={
                str(name): [float(value) for value in values]
                for name, values in (payload.get("estimates") or {}).items()
            },
        )


def refinement_report(
    before: Optional[RefinableBootstrapState],
    after: RefinableBootstrapState,
    metric: str = "roc_auc",
) -> dict[str, float]:
    """How much the extra draws actually moved the interval.

    This is what tells a user whether to keep spending time. A shift that has
    stopped shrinking means the remaining uncertainty is in the data, not in
    the number of resamples, and no amount of further computation will help.
    """

    low_after, high_after = after.interval(metric)
    if before is None:
        return {
            "iterations_before": 0,
            "iterations_after": after.iterations,
            "ci_low": low_after,
            "ci_high": high_after,
            "largest_bound_shift": float("nan"),
            "width_change": float("nan"),
        }
    low_before, high_before = before.interval(metric)
    return {
        "iterations_before": before.iterations,
        "iterations_after": after.iterations,
        "ci_low": low_after,
        "ci_high": high_after,
        "largest_bound_shift": float(
            max(abs(low_after - low_before), abs(high_after - high_before))
        ),
        "width_change": float(
            (high_after - low_after) - (high_before - low_before)
        ),
    }


@profiled_phase("marker_bootstrap")
def marker_bootstrap_intervals(
    subject_ids: Iterable,
    values: Iterable[float],
    labels: Iterable[int],
    *,
    direction: int,
    thresholds: Optional[Mapping[str, float]] = None,
    iterations: int,
    seed: int,
    screen_uninformative: bool = False,
    negligible_auc_margin: Optional[float] = None,
) -> dict[str, float]:
    clean_values, clean_labels, mask = _clean_binary(values, labels)
    subjects = np.asarray(subject_ids)[mask]
    if len(subjects) != len(clean_values):
        raise ValueError(
            "subject_ids, values and labels must have matching lengths"
        )
    aligned = int(direction) * clean_values
    metric_names = ["roc_auc", "average_precision", "pr_auc"]
    classification_names = (
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "balanced_accuracy",
    )
    for threshold_name in (thresholds or {}):
        metric_names.extend(
            f"{threshold_name}_{name}" for name in classification_names
        )
    estimates = {name: [] for name in metric_names}
    output = {
        f"{name}_ci_{bound}": float("nan")
        for name in metric_names
        for bound in ("low", "high")
    }
    output.update(
        {
            "bootstrap_iterations": 0,
            "bootstrap_successful_iterations": 0,
            "bootstrap_unit": "subject_id",
            "interval_method": "subject_cluster_bootstrap",
            "screened_out_as_uninformative": False,
        }
    )
    if iterations <= 0 or len(clean_values) == 0:
        return output

    unique_subjects = pd.unique(subjects)
    if len(unique_subjects) < 2:
        return output
    subject_codes, unique_subjects = pd.factorize(
        subjects,
        sort=False,
        use_na_sentinel=False,
    )
    n_subjects = len(unique_subjects)

    if screen_uninformative and negligible_auc_margin is not None:
        screen = clustered_auc_screening_interval(
            subject_codes, aligned, clean_labels
        )
        if screen is not None:
            auc, low, high, design_effect = screen
            # The criterion is effect size, not significance. On 640k rows an
            # AUC of 0.502 is "significant" and means nothing, so a
            # significance-based screen catches nothing exactly where the cost
            # is highest -- measured 2026-08-19: at 128k subjects the analytic
            # half-width is 0.0039 and the screening band closes to zero.
            #
            # A marker is skipped only when its *conservative* interval lies
            # entirely inside the band around 0.5 that the study has declared
            # too small to report. Even the most favourable end of the
            # interval then fails to reach relevance, so no resampling can
            # change the outcome. The interval is 8-16 % wider than the
            # bootstrap, so the judgement is made on weaker evidence than the
            # truth -- erring towards keeping the marker.
            floor = float(negligible_auc_margin)
            if 0.5 - floor < low and high < 0.5 + floor:
                output.update(
                    {
                        "roc_auc_ci_low": float(low),
                        "roc_auc_ci_high": float(high),
                        "bootstrap_iterations": 0,
                        "bootstrap_successful_iterations": 0,
                        "bootstrap_unit": "subject_id",
                        "interval_method": (
                            "hanley_mcneil_design_effect_screen"
                        ),
                        "screening_negligible_auc_margin": floor,
                        "screening_design_effect": float(design_effect),
                        "screened_out_as_uninformative": True,
                    }
                )
                return output
    prevalence = float(np.mean(clean_labels))
    rng = np.random.default_rng(seed)
    successful = 0
    batch_size = min(
        64,
        max(
            1,
            2_000_000 // max(len(clean_values), n_subjects),
        ),
    )
    probabilities = np.full(n_subjects, 1.0 / n_subjects, dtype=float)
    # A bootstrap changes only the subject weights; the marker scores and
    # their tie groups never change.  Sorting them inside every batch made a
    # 1,000-iteration interval repeat the exact same O(n log n) operation up
    # to hundreds of times on large cohorts.  Prepare the immutable order
    # once and reuse it for every weight batch.
    score_order, score_group_starts, sorted_positive = (
        _prepare_weighted_bootstrap_score_order(clean_labels, aligned)
    )
    remaining = int(iterations)
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        subject_weights = rng.multinomial(
            n_subjects,
            probabilities,
            size=current_batch,
        )
        row_weights = subject_weights[:, subject_codes]
        batch = _weighted_bootstrap_metric_batch(
            clean_labels,
            aligned,
            row_weights,
            thresholds=thresholds or {},
            prevalence=prevalence,
            score_order=score_order,
            score_group_starts=score_group_starts,
            sorted_positive=sorted_positive,
        )
        valid = batch.pop("_valid")
        successful += int(np.sum(valid))
        for name, values_for_metric in batch.items():
            finite = np.isfinite(values_for_metric)
            if finite.any():
                estimates[name].extend(
                    values_for_metric[finite].astype(float).tolist()
                )
        remaining -= current_batch

    for name, values_for_metric in estimates.items():
        if not values_for_metric:
            continue
        low, high = np.quantile(values_for_metric, (0.025, 0.975))
        output[f"{name}_ci_low"] = float(low)
        output[f"{name}_ci_high"] = float(high)
    output["bootstrap_iterations"] = int(iterations)
    output["bootstrap_successful_iterations"] = int(successful)
    return output


def _prepare_weighted_bootstrap_score_order(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the immutable score order and tie groups for a bootstrap.

    Subject-cluster resampling changes row weights, not labels or scores, so
    this work belongs to the interval rather than to each weight batch.
    Keeping it separate also lets the batch helper remain independently
    testable and backwards compatible for callers that do not precompute it.
    """

    binary = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=float)
    if binary.ndim != 1 or values.shape != binary.shape:
        raise ValueError("Bootstrap labels and scores have incompatible shapes")
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    group_starts = np.flatnonzero(
        np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    )
    sorted_positive = (binary == 1)[order]
    return order, group_starts, sorted_positive


def _weighted_bootstrap_metric_batch(
    labels: np.ndarray,
    scores: np.ndarray,
    row_weights: np.ndarray,
    *,
    thresholds: Mapping[str, float],
    prevalence: float,
    score_order: Optional[np.ndarray] = None,
    score_group_starts: Optional[np.ndarray] = None,
    sorted_positive: Optional[np.ndarray] = None,
    auc_only: bool = False,
) -> dict[str, np.ndarray]:
    """Weighted metrics; AUC-only callers can omit unused PR curve arrays."""
    if auc_only and thresholds:
        raise ValueError("AUC-only bootstrap cannot request threshold metrics")
    binary = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=float)
    weights = np.asarray(row_weights, dtype=np.int64)
    if (
        binary.ndim != 1
        or values.shape != binary.shape
        or weights.ndim != 2
        or weights.shape[1] != len(binary)
    ):
        raise ValueError("Weighted bootstrap arrays have incompatible shapes")

    positive = binary == 1
    negative = ~positive
    total_positive = weights[:, positive].sum(axis=1, dtype=np.int64)
    total_negative = weights[:, negative].sum(axis=1, dtype=np.int64)
    valid = (total_positive > 0) & (total_negative > 0)
    batch_size = weights.shape[0]
    result: dict[str, np.ndarray] = {
        "_valid": valid,
        "roc_auc": np.full(batch_size, np.nan, dtype=float),
        "average_precision": np.full(batch_size, np.nan, dtype=float),
        "pr_auc": np.full(batch_size, np.nan, dtype=float),
    }
    if valid.any():
        if (
            score_order is None
            or score_group_starts is None
            or sorted_positive is None
        ):
            order, group_starts, positive_in_order = (
                _prepare_weighted_bootstrap_score_order(binary, values)
            )
        else:
            order = np.asarray(score_order, dtype=np.int64)
            group_starts = np.asarray(score_group_starts, dtype=np.int64)
            positive_in_order = np.asarray(sorted_positive, dtype=bool)
            if (
                order.shape != binary.shape
                or positive_in_order.shape != binary.shape
                or group_starts.ndim != 1
                or len(group_starts) == 0
            ):
                raise ValueError(
                    "Prepared bootstrap score order has incompatible shapes"
                )
        sorted_weights = weights[:, order]
        positive_by_group = np.add.reduceat(
            sorted_weights * positive_in_order[np.newaxis, :],
            group_starts,
            axis=1,
        ).astype(float)
        negative_by_group = np.add.reduceat(
            sorted_weights * (~positive_in_order)[np.newaxis, :],
            group_starts,
            axis=1,
        ).astype(float)
        positive_total = total_positive.astype(float)
        negative_total = total_negative.astype(float)

        cumulative_negative = np.cumsum(negative_by_group, axis=1)
        negative_below = (
            negative_total[:, np.newaxis] - cumulative_negative
        )
        auc_numerator = np.sum(
            positive_by_group
            * (negative_below + 0.5 * negative_by_group),
            axis=1,
        )
        denominator = positive_total * negative_total
        np.divide(
            auc_numerator,
            denominator,
            out=result["roc_auc"],
            where=valid,
        )

        if auc_only:
            return {"_valid": valid, "roc_auc": result["roc_auc"]}

        cumulative_positive = np.cumsum(positive_by_group, axis=1)
        cumulative_total = cumulative_positive + cumulative_negative

        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        recall = np.divide(
            cumulative_positive,
            positive_total[:, np.newaxis],
            out=np.zeros_like(cumulative_positive),
            where=positive_total[:, np.newaxis] > 0,
        )
        recall_increment = np.diff(
            np.concatenate(
                (np.zeros((batch_size, 1)), recall),
                axis=1,
            ),
            axis=1,
        )
        result["average_precision"] = np.sum(
            recall_increment * precision,
            axis=1,
        )
        previous_precision = np.concatenate(
            (np.ones((batch_size, 1)), precision[:, :-1]),
            axis=1,
        )
        result["pr_auc"] = np.sum(
            recall_increment * (precision + previous_precision) * 0.5,
            axis=1,
        )
        for name in ("average_precision", "pr_auc"):
            result[name][~valid] = np.nan

    if auc_only:
        return {"_valid": valid, "roc_auc": result["roc_auc"]}

    positive_float = positive.astype(np.int8)
    negative_float = negative.astype(np.int8)
    for threshold_name, threshold in thresholds.items():
        predicted = values >= float(threshold)
        predicted_positive = predicted.astype(np.int8)
        predicted_negative = (~predicted).astype(np.int8)
        tp = weights @ (positive_float * predicted_positive)
        fn = weights @ (positive_float * predicted_negative)
        tn = weights @ (negative_float * predicted_negative)
        fp = weights @ (negative_float * predicted_positive)
        sensitivity = _vector_ratio(tp, tp + fn)
        specificity = _vector_ratio(tn, tn + fp)
        ppv, npv = _vector_predictive_values(
            sensitivity,
            specificity,
            prevalence,
        )
        result[f"{threshold_name}_sensitivity"] = sensitivity
        result[f"{threshold_name}_specificity"] = specificity
        result[f"{threshold_name}_ppv"] = ppv
        result[f"{threshold_name}_npv"] = npv
        result[f"{threshold_name}_balanced_accuracy"] = (
            sensitivity + specificity
        ) / 2.0
    return result


def _vector_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    result = np.full(np.asarray(numerator).shape, np.nan, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=result,
        where=np.asarray(denominator) > 0,
    )


def _vector_predictive_values(
    sensitivity: np.ndarray,
    specificity: np.ndarray,
    prevalence: float,
) -> tuple[np.ndarray, np.ndarray]:
    prevalence = float(np.clip(prevalence, 0.0, 1.0))
    ppv_denominator = (
        sensitivity * prevalence
        + (1.0 - specificity) * (1.0 - prevalence)
    )
    npv_denominator = (
        specificity * (1.0 - prevalence)
        + (1.0 - sensitivity) * prevalence
    )
    ppv = _vector_ratio(
        sensitivity * prevalence,
        ppv_denominator,
    )
    npv = _vector_ratio(
        specificity * (1.0 - prevalence),
        npv_denominator,
    )
    return ppv, npv


@profiled_phase("postprocess.fdr")
def apply_fdr_by_family(
    frame: pd.DataFrame,
    *,
    family_columns: tuple[str, ...] = ("endpoint_id", "panel_id"),
    p_column: str = "p_value",
    alpha_by_panel: Optional[Mapping[str, float]] = None,
    multiplicity_sensitivity: bool = True,
    sensitivity_strata: Optional[tuple[str, ...]] = None,
    sensitivity_alpha: float = 0.05,
) -> pd.DataFrame:
    """Correct the complete hypothesis family using only narrow metadata.

    No per-partition BH and no truncation of the hypothesis family. Wide effect,
    threshold and curve columns never enter the repeated grouping/copy passes.
    Preserve the caller's independent-output contract with one final wide copy.
    """
    required = tuple(dict.fromkeys((*family_columns, p_column, "panel_id",
        "inference_status", "confirmatory_assessability_gate_passed",
        *(sensitivity_strata if sensitivity_strata is not None else ("evaluation_split",)))))
    selected = [name for name in required if name in frame.columns]
    narrow = _apply_fdr_by_family_narrow(
        frame.loc[:, selected], family_columns=family_columns, p_column=p_column,
        alpha_by_panel=alpha_by_panel, multiplicity_sensitivity=multiplicity_sensitivity,
        sensitivity_strata=sensitivity_strata, sensitivity_alpha=sensitivity_alpha)
    output = frame.copy()
    for column in narrow.columns:
        if column not in selected:
            output[column] = narrow[column]
    return output


def _apply_fdr_by_family_narrow(
    frame: pd.DataFrame,
    *,
    family_columns: tuple[str, ...] = ("endpoint_id", "panel_id"),
    p_column: str = "p_value",
    alpha_by_panel: Optional[Mapping[str, float]] = None,
    multiplicity_sensitivity: bool = True,
    sensitivity_strata: Optional[tuple[str, ...]] = None,
    sensitivity_alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply the prespecified family-wise BH procedure and broad sensitivities.

    ``q_value`` retains the project's primary within-family interpretation.
    The additional global and hierarchical columns are deliberately labelled
    as sensitivity analyses.  Global BH spans every formal test in a stratum.
    The hierarchy first screens families with Simes/BH and then uses the
    Benjamini--Bogomolov selective within-family level ``alpha * R / M``.
    """
    output = frame.copy()
    output["q_value"] = np.nan
    output["fdr_significant"] = False
    if output.empty:
        return _empty_multiplicity_sensitivity_columns(output)
    for _, indices in output.groupby(list(family_columns), dropna=False).groups.items():
        index = list(indices)
        p_values = pd.to_numeric(
            output.loc[index, p_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = np.isfinite(p_values)
        if "inference_status" in output.columns:
            formal = output.loc[index, "inference_status"].eq(
                "formal_inference"
            ).to_numpy(dtype=bool)
            valid &= formal
        if not valid.any():
            continue
        panel_id = (
            str(output.loc[index[0], "panel_id"])
            if "panel_id" in output
            else ""
        )
        alpha = float((alpha_by_panel or {}).get(panel_id, 0.05))
        rejected, corrected, _, _ = multipletests(
            p_values[valid],
            alpha=alpha,
            method="fdr_bh",
        )
        valid_indices = np.asarray(index, dtype=object)[valid]
        output.loc[valid_indices, "q_value"] = corrected
        output.loc[valid_indices, "fdr_significant"] = rejected
    if multiplicity_sensitivity:
        output = _apply_multiplicity_sensitivity(
            output,
            family_columns=family_columns,
            p_column=p_column,
            strata=sensitivity_strata,
            alpha=float(sensitivity_alpha),
        )
    else:
        output = _empty_multiplicity_sensitivity_columns(output)
    confirmatory_gate = (
        output["confirmatory_assessability_gate_passed"].astype("boolean").fillna(False).astype(bool)
        if "confirmatory_assessability_gate_passed" in output.columns
        else pd.Series(True, index=output.index, dtype=bool)
    )
    output["confirmatory_fdr_significant"] = (
        output["fdr_significant"].astype(bool) & confirmatory_gate
    )
    output["confirmatory_global_fdr_significant"] = (
        output["global_fdr_significant"].astype(bool) & confirmatory_gate
    )
    output["confirmatory_hierarchical_fdr_significant"] = (
        output["hierarchical_fdr_significant"].astype(bool) & confirmatory_gate
    )
    return output


def _empty_multiplicity_sensitivity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    numeric_columns = (
        "global_q_value",
        "hierarchical_family_p_value",
        "hierarchical_family_q_value",
        "hierarchical_within_family_q_value",
        "hierarchical_selective_alpha",
        "hierarchical_q_value",
    )
    boolean_columns = (
        "global_fdr_significant",
        "hierarchical_family_selected",
        "hierarchical_fdr_significant",
        "confirmatory_fdr_significant",
        "confirmatory_global_fdr_significant",
        "confirmatory_hierarchical_fdr_significant",
    )
    for column in numeric_columns:
        output[column] = np.nan
    for column in boolean_columns:
        output[column] = False
    output["multiplicity_sensitivity_method"] = (
        "global_bh_and_simes_bh_selective_within_family"
    )
    return output


def _apply_multiplicity_sensitivity(
    frame: pd.DataFrame,
    *,
    family_columns: tuple[str, ...],
    p_column: str,
    strata: Optional[tuple[str, ...]],
    alpha: float,
) -> pd.DataFrame:
    if not 0.0 < alpha < 1.0:
        raise ValueError("sensitivity_alpha must be strictly between zero and one")
    output = _empty_multiplicity_sensitivity_columns(frame)
    if output.empty:
        return output
    if strata is None:
        strata = tuple(
            column
            for column in ("evaluation_split",)
            if column in output.columns
        )
    missing_strata = [column for column in strata if column not in output.columns]
    if missing_strata:
        raise KeyError(f"Multiplicity sensitivity strata are missing: {missing_strata}")
    hierarchical_family_columns = tuple(
        column for column in family_columns if column not in strata
    )
    if not hierarchical_family_columns:
        hierarchical_family_columns = ("__single_family__",)

    p_values = pd.to_numeric(output[p_column], errors="coerce").to_numpy(dtype=float)
    formal = np.ones(len(output), dtype=bool)
    if "inference_status" in output.columns:
        formal = output["inference_status"].eq("formal_inference").to_numpy(
            dtype=bool
        )
    valid = np.isfinite(p_values) & formal
    working = output.reset_index(drop=True).copy()
    working["__row_position__"] = np.arange(len(working), dtype=int)
    if hierarchical_family_columns == ("__single_family__",):
        working["__single_family__"] = "all"
    stratum_groups = (
        working.groupby(list(strata), dropna=False, sort=False).indices.values()
        if strata
        else (np.arange(len(working), dtype=int),)
    )

    global_q = np.full(len(output), np.nan, dtype=float)
    global_rejected = np.zeros(len(output), dtype=bool)
    family_p = np.full(len(output), np.nan, dtype=float)
    family_q = np.full(len(output), np.nan, dtype=float)
    within_q = np.full(len(output), np.nan, dtype=float)
    selective_alpha = np.full(len(output), np.nan, dtype=float)
    hierarchy_q = np.full(len(output), np.nan, dtype=float)
    family_selected = np.zeros(len(output), dtype=bool)
    hierarchy_rejected = np.zeros(len(output), dtype=bool)

    for stratum_positions in stratum_groups:
        positions = np.asarray(stratum_positions, dtype=int)
        valid_positions = positions[valid[positions]]
        if len(valid_positions) == 0:
            continue
        rejected, corrected, _, _ = multipletests(
            p_values[valid_positions],
            alpha=alpha,
            method="fdr_bh",
        )
        global_q[valid_positions] = corrected
        global_rejected[valid_positions] = rejected

        scoped = working.iloc[positions]
        relative_groups = scoped.groupby(
            list(hierarchical_family_columns),
            dropna=False,
            sort=False,
        ).indices.values()
        families: list[tuple[np.ndarray, np.ndarray, float]] = []
        for relative_positions in relative_groups:
            member_positions = positions[np.asarray(relative_positions, dtype=int)]
            valid_members = member_positions[valid[member_positions]]
            if len(valid_members) == 0:
                continue
            ordered = np.sort(p_values[valid_members])
            ranks = np.arange(1, len(ordered) + 1, dtype=float)
            simes_p = float(
                np.clip(np.min(ordered * len(ordered) / ranks), 0.0, 1.0)
            )
            families.append((member_positions, valid_members, simes_p))
        if not families:
            continue
        family_p_values = np.asarray([item[2] for item in families], dtype=float)
        selected, corrected_family, _, _ = multipletests(
            family_p_values,
            alpha=alpha,
            method="fdr_bh",
        )
        family_count = len(families)
        selected_count = int(np.sum(selected))
        selected_alpha = (
            float(alpha * selected_count / family_count)
            if selected_count
            else 0.0
        )
        for family_index, (members, valid_members, simes_p) in enumerate(families):
            family_p[members] = simes_p
            family_q[members] = corrected_family[family_index]
            family_selected[members] = bool(selected[family_index])
            selective_alpha[members] = selected_alpha
            _, corrected_within, _, _ = multipletests(
                p_values[valid_members],
                alpha=alpha,
                method="fdr_bh",
            )
            within_q[valid_members] = corrected_within
            if selected_count:
                scaled_within = np.clip(
                    corrected_within * family_count / selected_count,
                    0.0,
                    1.0,
                )
                combined = np.maximum(
                    float(corrected_family[family_index]),
                    scaled_within,
                )
                hierarchy_q[valid_members] = combined
                hierarchy_rejected[valid_members] = (
                    bool(selected[family_index])
                    & (corrected_within <= selected_alpha)
                )
            else:
                hierarchy_q[valid_members] = 1.0

    output["global_q_value"] = global_q
    output["global_fdr_significant"] = global_rejected
    output["hierarchical_family_p_value"] = family_p
    output["hierarchical_family_q_value"] = family_q
    output["hierarchical_within_family_q_value"] = within_q
    output["hierarchical_selective_alpha"] = selective_alpha
    output["hierarchical_q_value"] = hierarchy_q
    output["hierarchical_family_selected"] = family_selected
    output["hierarchical_fdr_significant"] = hierarchy_rejected
    return output


def fit_frozen_score(
    frame: pd.DataFrame,
    labels: Iterable[int],
    *,
    score_id: str,
    endpoint_id: str,
    components: tuple[str, ...],
    origin: str,
    version: str,
    subject_ids: Optional[Iterable] = None,
) -> FrozenScoreSpec:
    if not 1 <= len(components) <= 4:
        raise ValueError("A frozen score needs one to four complete components")
    missing = sorted(set(components) - set(frame.columns))
    if missing:
        raise KeyError(f"Score components are missing: {missing}")
    binary = np.asarray(labels, dtype=float)
    if len(binary) != len(frame):
        raise ValueError("Labels and score frame must have matching lengths")
    component_matrix = np.column_stack(
        [
            pd.to_numeric(frame[component], errors="coerce").to_numpy(
                dtype=float
            )
            for component in components
        ]
    )
    complete = (
        np.isfinite(binary)
        & np.isin(binary, (0, 1))
        & np.all(np.isfinite(component_matrix), axis=1)
    )
    clean_labels = binary[complete].astype(np.int8)
    _require_two_classes(clean_labels)
    directions: list[int] = []
    means: list[float] = []
    deviations: list[float] = []
    for index, component in enumerate(components):
        component_values = component_matrix[complete, index]
        raw_auc = roc_auc_score(clean_labels, component_values)
        directions.append(1 if raw_auc >= 0.5 else -1)
        means.append(float(np.mean(component_values)))
        deviation = float(np.std(component_values, ddof=1))
        if not np.isfinite(deviation) or deviation <= 0:
            raise ValueError(f"Score component {component!r} is constant")
        deviations.append(deviation)
    if subject_ids is not None:
        subject_array = np.asarray(subject_ids)
        if len(subject_array) != len(frame):
            raise ValueError(
                "subject_ids and score frame must have matching lengths"
            )
        split_payload = sorted(
            {str(value) for value in subject_array[complete]}
        )
    else:
        split_payload = {"complete_case_row_count": int(np.sum(complete))}
    return FrozenScoreSpec(
        score_id=score_id,
        version=version,
        origin=origin,
        endpoint_id=endpoint_id,
        components=components,
        directions=tuple(directions),
        means=tuple(means),
        standard_deviations=tuple(deviations),
        training_split_hash=content_hash(split_payload),
        aggregation="unweighted_mean",
        exploratory_score=True,
    )


def apply_frozen_score(
    frame: pd.DataFrame,
    specification: FrozenScoreSpec,
) -> np.ndarray:
    missing = sorted(set(specification.components) - set(frame.columns))
    if missing:
        raise KeyError(f"Score components are missing: {missing}")
    standardized = []
    for component, direction, mean, deviation in zip(
        specification.components,
        specification.directions,
        specification.means,
        specification.standard_deviations,
    ):
        values = pd.to_numeric(frame[component], errors="coerce").to_numpy(dtype=float)
        standardized.append(direction * (values - mean) / deviation)
    matrix = np.column_stack(standardized)
    score = np.full(len(frame), np.nan, dtype=float)
    complete = np.all(np.isfinite(matrix), axis=1)
    score[complete] = np.mean(matrix[complete], axis=1)
    return score


def inference_status(
    n_positive: int,
    n_negative: int,
    *,
    minimum_inference: int = 30,
    minimum_primary: int = 100,
) -> str:
    smallest = min(int(n_positive), int(n_negative))
    if smallest < minimum_inference:
        return "descriptive_only"
    if smallest < minimum_primary:
        return "underpowered"
    return "formal_inference"


def enforce_inference_policy(
    metrics: Mapping[str, float | int | None],
    status: str,
) -> dict[str, float | int | None]:
    output = dict(metrics)
    if status == "descriptive_only":
        output["p_value"] = None
    return output


# Operating points are named by the target they were fitted to. The achieved
# value on the fitting data can still miss it: a discrete marker has only a
# handful of thresholds, and none of them may sit near 95 % specificity.
OPERATING_POINT_TARGETS: dict[str, tuple[str, float]] = {
    "specificity_90": ("specificity", 0.90),
    "specificity_95": ("specificity", 0.95),
    "sensitivity_90": ("sensitivity", 0.90),
}
OPERATING_POINT_TARGET_TOLERANCE = 1e-12


def _threshold_for_minimum_specificity(
    thresholds: np.ndarray,
    sensitivity: np.ndarray,
    specificity: np.ndarray,
    target: float,
) -> float:
    candidates = np.flatnonzero(specificity >= target)
    if len(candidates) == 0:
        # No finite threshold reaches the target. The old fallback took the
        # most specific one available, which for a flag carried by 90 % of
        # the controls is a specificity of 0.09 -- reported afterwards as
        # "sensitivity 0.96 at 95 % specificity". The only operating point
        # that honours the constraint is the one above every observed value:
        # nobody positive, specificity 1, sensitivity 0. That is the honest
        # answer, and the explorer ranks it as such.
        return float(np.nextafter(np.max(thresholds), np.inf))
    best_sensitivity = np.max(sensitivity[candidates])
    best = candidates[np.flatnonzero(sensitivity[candidates] == best_sensitivity)]
    return float(np.min(thresholds[best]))


def _threshold_for_minimum_sensitivity(
    thresholds: np.ndarray,
    sensitivity: np.ndarray,
    specificity: np.ndarray,
    target: float,
) -> float:
    candidates = np.flatnonzero(sensitivity >= target)
    if len(candidates) == 0:
        return float(thresholds[int(np.argmax(sensitivity))])
    best_specificity = np.max(specificity[candidates])
    best = candidates[np.flatnonzero(specificity[candidates] == best_specificity)]
    return float(np.max(thresholds[best]))


def _clean_binary(
    values: Iterable[float],
    labels: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_array = np.asarray(values, dtype=float)
    labels_array = np.asarray(labels, dtype=float)
    if values_array.shape != labels_array.shape:
        raise ValueError("Values and labels must have matching shapes")
    mask = (
        np.isfinite(values_array)
        & np.isfinite(labels_array)
        & np.isin(labels_array, (0, 1))
    )
    return (
        values_array[mask],
        labels_array[mask].astype(np.int8),
        mask,
    )


def _require_two_classes(labels: np.ndarray) -> None:
    if len(labels) == 0 or np.unique(labels).size < 2:
        raise ValueError("Both positive and negative training examples are required")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _empty_metric_result() -> dict[str, float | int | bool | str | None]:
    return {
        "n_total": 0,
        "n_positive": 0,
        "n_negative": 0,
        "roc_auc": None,
        "average_precision": None,
        "pr_auc": None,
        "p_value": None,
        "rank_biserial": None,
        "aligned_rank_biserial": None,
    }
