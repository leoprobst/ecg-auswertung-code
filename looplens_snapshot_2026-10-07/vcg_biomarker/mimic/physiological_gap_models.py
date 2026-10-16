"""Trained "physiological gap" markers (age-gap, BMI-gap).

Unlike every other marker in this project, a gap marker is not computed
per-ECG from the waveform alone: it is the residual of a regression model
that predicts a physiological quantity (age, BMI) from the existing
curated Core biomarker panel. The literature calls the age variant
"AI-ECG age gap" — the difference between predicted and chronological age
is itself a marker, independently associated with mortality and
cardiovascular risk in several published cohorts.

Scientific discipline mirrored from the rest of this project:

- The regression model (which features, which hyperparameters, which
  coefficients) is fitted using Train only.
- Hyperparameters are chosen inside a nested patient-grouped cross-validation
  within Train. Imputation and scaling are fitted anew inside every inner and
  outer fold, so neither held-out values nor hyperparameter selection leak
  into the reported out-of-fold residuals.
- The "Train gap" reported for every Train ECG is an out-of-fold
  prediction (predicted by a model that never saw that patient), not an
  in-sample residual — otherwise Train gaps would be optimistically small
  by construction and not comparable to Validation gaps.
- Validation gaps come from a single model refit on the *entire* Train
  split with the winning hyperparameters; Validation is never used to
  select features or hyperparameters.
- Locked Test is never read here.
- Missing panel features are imputed with the Train median (raw space),
  identically at fit and apply time. An ECG with too many missing panel
  features gets ``evaluable=False`` instead of an imputed gap.
"""

from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import MimicProjectConfig
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    feature_parquet_clause,
    atomic_write_parquet,
    content_hash,
    file_fingerprint,
    operational_config_hash,
    parquet_columns,
    scientific_config_hash,
)


GAP_MODEL_VERSION = "1.2.0"  # Bias-Korrektur, Zielgrenzen, Brauchbarkeitsschwelle
GAP_STAGE_VERSION = "1.2.1"

# Small, fixed grouped-CV grid. sklearn's ElasticNetCV only supports plain
# (non-grouped) internal CV, which would leak same-subject ECGs across
# folds here, so the grid is searched manually with GroupKFold.
_ALPHA_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)
_L1_RATIO_GRID: tuple[float, ...] = (0.1, 0.5, 0.9)

# An ECG needs at least this fraction of the candidate panel present to
# receive a gap value at all; otherwise the model input is too sparse to
# trust and the row is reported as not evaluable rather than imputed.
MINIMUM_FEATURE_COVERAGE = 0.7

_TARGET_COLUMNS = {
    "age_gap": "age",
    "bmi_gap": "bmi_before_ecg",
}


def target_column_for(target_id: str) -> str:
    return _TARGET_COLUMNS.get(target_id, target_id)


@dataclass(frozen=True)
class GapMarkerModel:
    target_id: str
    feature_names: tuple[str, ...]
    feature_medians: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float
    l1_ratio: float
    training_subject_hash: str
    n_train_subjects: int
    train_out_of_fold_r2: float
    train_out_of_fold_mae: float
    oof_method: str = "nested_grouped_cv"
    validation_r2: float = float("nan")
    validation_mae: float = float("nan")
    # Bias correction, fitted on the training gaps (see _fit_gap_bias).
    bias_slope: float = 0.0
    bias_intercept: float = 0.0
    gap_actual_correlation: float = float("nan")
    corrected_gap_actual_correlation: float = float("nan")
    target_lower_bound: float = float("-inf")
    target_upper_bound: float = float("inf")
    n_target_out_of_bounds: int = 0
    usable: bool = True
    usability_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GapMarkerModel":
        payload = dict(data)
        for key in (
            "feature_names",
            "feature_medians",
            "feature_means",
            "feature_scales",
            "coefficients",
        ):
            payload[key] = tuple(payload[key])
        return cls(**payload)

    @property
    def nonzero_features(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (name, coefficient)
            for name, coefficient in zip(self.feature_names, self.coefficients)
            if coefficient != 0.0
        )


def _design_matrix(
    frame: pd.DataFrame,
    feature_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_values_with_nan, coverage_fraction_per_row)."""
    matrix = np.column_stack(
        [
            pd.to_numeric(frame.get(feature_id), errors="coerce").to_numpy(
                dtype=float
            )
            if feature_id in frame
            else np.full(len(frame), np.nan)
            for feature_id in feature_ids
        ]
    )
    coverage = np.mean(np.isfinite(matrix), axis=1)
    return matrix, coverage


def _impute_and_standardize(
    matrix: np.ndarray,
    *,
    medians: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    filled = np.where(np.isfinite(matrix), matrix, medians[np.newaxis, :])
    return (filled - means[np.newaxis, :]) / scales[np.newaxis, :]


def target_bounds_for(target_id: str) -> tuple[float, float]:
    return GAP_TARGET_BOUNDS.get(str(target_id), (float("-inf"), float("inf")))


def _within_target_bounds(values: np.ndarray, target_id: str) -> np.ndarray:
    lower, upper = target_bounds_for(target_id)
    finite = np.isfinite(values)
    return finite & (values >= lower) & (values <= upper)


def _fit_gap_bias(gap: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Least-squares line of the gap on the true value.

    A regression that is not perfect is biased towards the mean of its target,
    so the gap it produces is itself a function of that target. Measured on the
    2026-08-28 run: age_gap correlated with age at **r = -0.872**, and the mean
    gap ran from +24.4 years under 40 to -15.4 years over 75. Seventy-six per
    cent of the "cardiac ageing" signal was age.

    This is the standard correction in the brain-age literature (de Lange &
    Cole): fit gap = a * actual + b on the training data, then subtract that
    line. What remains is the part of the gap the true value does not explain.

    It uses the label, so the corrected gap is an *interpretive* residual, not
    something a model could produce for an unlabelled patient. The artifact
    says so rather than leaving a reader to discover it.
    """

    usable = np.isfinite(gap) & np.isfinite(actual)
    if int(usable.sum()) < 3:
        return 0.0, 0.0
    x, y = actual[usable], gap[usable]
    variance = float(np.var(x, ddof=1))
    if not np.isfinite(variance) or variance <= 1e-12:
        return 0.0, 0.0
    slope = float(np.cov(x, y, ddof=1)[0, 1] / variance)
    intercept = float(np.mean(y) - slope * np.mean(x))
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return 0.0, 0.0
    return slope, intercept


def _corrected_gap(
    gap: np.ndarray, actual: np.ndarray, *, slope: float, intercept: float
) -> np.ndarray:
    return gap - (slope * actual + intercept)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    usable = np.isfinite(left) & np.isfinite(right)
    if int(usable.sum()) < 3:
        return float("nan")
    x, y = left[usable], right[usable]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_gap_marker_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    target_id: str,
    feature_ids: tuple[str, ...],
    n_folds: int = 5,
    minimum_train_subjects: int = 200,
    random_seed: int = 20260725,
) -> tuple[GapMarkerModel, pd.DataFrame, pd.DataFrame]:
    """Fit a physiological gap marker (e.g. age-gap, BMI-gap).

    Returns ``(model, train_rows, validation_rows)``. Both row frames have
    columns ``subject_id, study_id, predicted_value, actual_value,
    gap_value, evaluable``. Train rows are out-of-fold predictions;
    Validation rows come from a single model refit on all of Train.
    """
    target_column = target_column_for(target_id)
    train_target = pd.to_numeric(train[target_column], errors="coerce")
    train_has_target = train_target.notna().to_numpy()
    train_matrix, train_coverage = _design_matrix(train, feature_ids)
    train_in_bounds = _within_target_bounds(
        train_target.to_numpy(dtype=float), target_id
    )
    n_target_out_of_bounds = int((train_has_target & ~train_in_bounds).sum())
    train_usable = (
        train_has_target
        & train_in_bounds
        & (train_coverage >= MINIMUM_FEATURE_COVERAGE)
    )
    usable_subjects = pd.Series(
        train.loc[train_usable, "subject_id"]
    ).nunique()
    if usable_subjects < minimum_train_subjects:
        raise ValueError(
            f"Gap marker for {target_id!r} needs at least "
            f"{minimum_train_subjects} usable train subjects; got "
            f"{usable_subjects}"
        )

    fit_matrix = train_matrix[train_usable]
    fit_target = train_target.to_numpy(dtype=float)[train_usable]
    fit_subjects = train.loc[train_usable, "subject_id"].to_numpy()

    medians = np.nanmedian(fit_matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(fit_matrix), fit_matrix, medians[np.newaxis, :])
    means = np.mean(filled, axis=0)
    scales = np.std(filled, axis=0, ddof=1)
    scales = np.where(scales > 1e-9, scales, 1.0)

    best = _select_hyperparameters_by_grouped_cv(
        fit_matrix,
        fit_target,
        fit_subjects,
        n_folds=n_folds,
        random_seed=random_seed,
    )

    standardized_fit = _impute_and_standardize(
        fit_matrix,
        medians=medians,
        means=means,
        scales=scales,
    )
    final_model = ElasticNet(
        alpha=best["alpha"],
        l1_ratio=best["l1_ratio"],
        max_iter=20000,
        random_state=random_seed,
    )
    final_model.fit(standardized_fit, fit_target)

    model = GapMarkerModel(
        target_id=target_id,
        feature_names=tuple(feature_ids),
        feature_medians=tuple(float(value) for value in medians),
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in final_model.coef_),
        intercept=float(final_model.intercept_),
        alpha=float(best["alpha"]),
        l1_ratio=float(best["l1_ratio"]),
        training_subject_hash=content_hash(
            sorted(str(value) for value in np.unique(fit_subjects))
        ),
        n_train_subjects=int(usable_subjects),
        train_out_of_fold_r2=float(best["r2"]),
        train_out_of_fold_mae=float(best["mae"]),
        oof_method=str(best["oof_method"]),
    )

    train_rows = pd.DataFrame(
        {
            "subject_id": train.loc[train_usable, "subject_id"].to_numpy(),
            "study_id": _study_ids(train, train_usable),
            "predicted_value": best["out_of_fold_predictions"],
            "actual_value": fit_target,
            "evaluable": True,
        }
    )
    train_rows["gap_value"] = (
        train_rows["predicted_value"] - train_rows["actual_value"]
    )
    not_evaluable_mask = ~train_usable
    not_evaluable_train = pd.DataFrame(
        {
            "subject_id": train.loc[not_evaluable_mask, "subject_id"].to_numpy(),
            "study_id": _study_ids(train, not_evaluable_mask),
            "predicted_value": np.nan,
            "actual_value": train_target.to_numpy(dtype=float)[
                not_evaluable_mask
            ],
            "evaluable": False,
        }
    )
    not_evaluable_train["gap_value"] = np.nan
    train_output = pd.concat(
        (train_rows, not_evaluable_train),
        ignore_index=True,
    )

    validation_output = apply_gap_model(validation, model)
    validation_evaluable = validation_output.loc[
        validation_output["evaluable"]
    ]
    if len(validation_evaluable) >= 2:
        residuals = (
            validation_evaluable["predicted_value"]
            - validation_evaluable["actual_value"]
        )
        total_variance = np.var(
            validation_evaluable["actual_value"].to_numpy(),
            ddof=1,
        )
        r2 = (
            float(
                1.0
                - np.sum(residuals.to_numpy() ** 2)
                / (total_variance * (len(validation_evaluable) - 1))
            )
            if total_variance > 1e-12
            else float("nan")
        )
        model = _with_validation_metrics(
            model,
            r2=r2,
            mae=float(np.mean(np.abs(residuals))),
        )

    # The bias line is fitted on the out-of-fold *training* gaps only, so the
    # correction applied to validation is not informed by validation.
    slope, intercept = _fit_gap_bias(
        train_rows["gap_value"].to_numpy(dtype=float),
        train_rows["actual_value"].to_numpy(dtype=float),
    )
    for frame in (train_output, validation_output):
        frame["gap_corrected"] = _corrected_gap(
            frame["gap_value"].to_numpy(dtype=float),
            frame["actual_value"].to_numpy(dtype=float),
            slope=slope,
            intercept=intercept,
        )
        frame.loc[~frame["evaluable"].astype(bool), "gap_corrected"] = np.nan

    reference = (
        validation_output.loc[validation_output["evaluable"].astype(bool)]
        if bool(validation_output["evaluable"].astype(bool).any())
        else train_rows
    )
    raw_correlation = _safe_correlation(
        reference["gap_value"].to_numpy(dtype=float),
        reference["actual_value"].to_numpy(dtype=float),
    )
    corrected_correlation = _safe_correlation(
        reference["gap_corrected"].to_numpy(dtype=float),
        reference["actual_value"].to_numpy(dtype=float),
    )

    lower, upper = target_bounds_for(target_id)
    usable = True
    reason = "held-out R^2 over the prespecified floor"
    if not np.isfinite(model.validation_r2):
        usable, reason = False, "held-out R^2 not computable"
    elif model.validation_r2 < GAP_MINIMUM_VALIDATION_R2:
        usable = False
        reason = (
            f"held-out R^2 {model.validation_r2:.4f} below the prespecified "
            f"floor {GAP_MINIMUM_VALIDATION_R2:.2f}: the model predicts the "
            "population mean, so the gap is the negated target, not an "
            "ECG-derived quantity"
        )
    model = _with_gap_diagnostics(
        model,
        bias_slope=slope,
        bias_intercept=intercept,
        gap_actual_correlation=raw_correlation,
        corrected_gap_actual_correlation=corrected_correlation,
        target_lower_bound=lower,
        target_upper_bound=upper,
        n_target_out_of_bounds=n_target_out_of_bounds,
        usable=usable,
        usability_reason=reason,
    )
    return model, train_output, validation_output


def _study_ids(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    if "study_id" in frame:
        return frame.loc[mask, "study_id"].to_numpy()
    return np.arange(int(np.sum(mask)))


def _with_gap_diagnostics(model: "GapMarkerModel", **fields) -> "GapMarkerModel":
    payload = model.to_dict()
    payload.update(fields)
    return GapMarkerModel.from_dict(payload)


def _with_validation_metrics(
    model: GapMarkerModel,
    *,
    r2: float,
    mae: float,
) -> GapMarkerModel:
    payload = model.to_dict()
    payload["validation_r2"] = r2
    payload["validation_mae"] = mae
    return GapMarkerModel.from_dict(payload)


def _select_hyperparameters_by_grouped_cv(
    matrix: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    *,
    n_folds: int,
    random_seed: int,
) -> dict:
    unique_subjects = np.unique(subjects)
    effective_folds = max(2, min(n_folds, len(unique_subjects)))
    outer = GroupKFold(n_splits=effective_folds)
    honest_oof = np.full(len(target), np.nan, dtype=float)
    outer_parameters: list[tuple[float, float]] = []
    for outer_train, outer_test in outer.split(matrix, target, groups=subjects):
        selected = _best_gap_parameters(
            matrix[outer_train],
            target[outer_train],
            subjects[outer_train],
            n_folds=min(n_folds, len(np.unique(subjects[outer_train]))),
            random_seed=random_seed,
        )
        outer_parameters.append((selected["alpha"], selected["l1_ratio"]))
        medians, means, scales = _fold_preprocessing(matrix[outer_train])
        estimator = ElasticNet(
            alpha=selected["alpha"],
            l1_ratio=selected["l1_ratio"],
            max_iter=20000,
            random_state=random_seed,
        )
        estimator.fit(
            _impute_and_standardize(
                matrix[outer_train],
                medians=medians,
                means=means,
                scales=scales,
            ),
            target[outer_train],
        )
        honest_oof[outer_test] = estimator.predict(
            _impute_and_standardize(
                matrix[outer_test],
                medians=medians,
                means=means,
                scales=scales,
            )
        )
    final = _best_gap_parameters(
        matrix,
        target,
        subjects,
        n_folds=effective_folds,
        random_seed=random_seed,
    )
    residuals = honest_oof - target
    valid = np.isfinite(residuals)
    mae = float(np.mean(np.abs(residuals[valid])))
    total_variance = float(np.var(target[valid], ddof=1))
    r2 = (
        1.0
        - float(np.sum(residuals[valid] ** 2))
        / (total_variance * (int(np.sum(valid)) - 1))
        if int(np.sum(valid)) >= 2 and total_variance > 1e-12
        else float("nan")
    )
    return {
        **final,
        "mae": mae,
        "r2": r2,
        "out_of_fold_predictions": honest_oof,
        "oof_method": "nested_grouped_cv",
        "outer_selected_parameters": outer_parameters,
    }


def _fold_preprocessing(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(matrix), matrix, medians[np.newaxis, :])
    means = np.mean(filled, axis=0)
    scales = np.std(filled, axis=0, ddof=1)
    scales = np.where(np.isfinite(scales) & (scales > 1e-9), scales, 1.0)
    return medians, means, scales


# Below this many rows a grid point takes milliseconds and the worker start
# (the package import) dominates: measured 2.0 s serial against 18.8 s on
# eight processes at 3,000 rows.
GRID_PARALLEL_MINIMUM_ROWS = 20_000


def _grid_workers(task_count: int) -> int:
    from .grant_parallel import grant_workers

    return grant_workers(task_count)


def _run_grid(function, argument_list: list[tuple], *, rows: int) -> list:
    """The hyperparameter grid, in grid order (audit R15).

    Each grid point is a deterministic fit; the selection loop reads the
    scores in grid order, so the chosen parameters equal the serial search.
    """
    from .grant_parallel import grant_parallel_map

    return grant_parallel_map(
        function,
        argument_list,
        parallel=int(rows) >= GRID_PARALLEL_MINIMUM_ROWS,
    )


def _gap_grid_point(
    matrix: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    alpha: float,
    l1_ratio: float,
    effective_folds: int,
    random_seed: int,
) -> float:
    """Grouped out-of-fold R2 of one (alpha, l1_ratio) point."""
    splitter = GroupKFold(n_splits=effective_folds)
    predictions = np.full(len(target), np.nan, dtype=float)
    for train_index, test_index in splitter.split(
        matrix, target, groups=subjects
    ):
        medians, means, scales = _fold_preprocessing(
            matrix[train_index]
        )
        estimator = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20000,
            random_state=random_seed,
        )
        estimator.fit(
            _impute_and_standardize(
                matrix[train_index],
                medians=medians,
                means=means,
                scales=scales,
            ),
            target[train_index],
        )
        predictions[test_index] = estimator.predict(
            _impute_and_standardize(
                matrix[test_index],
                medians=medians,
                means=means,
                scales=scales,
            )
        )
    residuals = predictions - target
    valid = np.isfinite(residuals)
    total_variance = float(np.var(target[valid], ddof=1))
    return (
        1.0
        - float(np.sum(residuals[valid] ** 2))
        / (total_variance * (int(np.sum(valid)) - 1))
        if int(np.sum(valid)) >= 2 and total_variance > 1e-12
        else float("nan")
    )


def _best_gap_parameters(
    matrix: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    *,
    n_folds: int,
    random_seed: int,
) -> dict:
    effective_folds = max(2, min(int(n_folds), len(np.unique(subjects))))
    grid = [
        (float(alpha), float(l1_ratio))
        for alpha in _ALPHA_GRID
        for l1_ratio in _L1_RATIO_GRID
    ]
    scores = _run_grid(
        _gap_grid_point,
        [
            (matrix, target, subjects, alpha, l1_ratio, effective_folds, random_seed)
            for alpha, l1_ratio in grid
        ],
        rows=len(target),
    )
    best: Optional[dict] = None
    for (alpha, l1_ratio), r2 in zip(grid, scores):
        candidate = {
            "alpha": float(alpha),
            "l1_ratio": float(l1_ratio),
            "selection_r2": r2,
        }
        if best is None or (
            np.isfinite(r2)
            and (
                not np.isfinite(best["selection_r2"])
                or r2 > best["selection_r2"]
            )
        ):
            best = candidate
    assert best is not None
    return best


def apply_gap_model(
    frame: pd.DataFrame,
    model: GapMarkerModel,
) -> pd.DataFrame:
    """Score new rows (Validation or a single ECG-Labor case) with an
    already-fitted, frozen gap model. Never refits anything."""
    matrix, coverage = _design_matrix(frame, model.feature_names)
    standardized = _impute_and_standardize(
        matrix,
        medians=np.asarray(model.feature_medians, dtype=float),
        means=np.asarray(model.feature_means, dtype=float),
        scales=np.asarray(model.feature_scales, dtype=float),
    )
    predicted = (
        model.intercept
        + standardized @ np.asarray(model.coefficients, dtype=float)
    )
    actual = pd.to_numeric(
        frame.get(
            target_column_for(model.target_id),
            pd.Series(np.nan, index=frame.index),
        ),
        errors="coerce",
    ).to_numpy(dtype=float)
    evaluable = (coverage >= MINIMUM_FEATURE_COVERAGE) & np.isfinite(actual)
    output = pd.DataFrame(
        {
            "subject_id": frame["subject_id"].to_numpy(),
            "study_id": (
                frame["study_id"].to_numpy()
                if "study_id" in frame
                else np.arange(len(frame))
            ),
            "predicted_value": np.where(evaluable, predicted, np.nan),
            "actual_value": np.where(evaluable, actual, np.nan),
            "evaluable": evaluable,
        }
    )
    output["gap_value"] = output["predicted_value"] - output["actual_value"]
    return output


GAP_TARGETS: tuple[str, ...] = ("age_gap", "bmi_gap")

# Plausibility bounds for the *target*, declared before the model sees it.
#
# The marker side has had signal QC since the beginning; the outcome side had
# nothing. Measured on the 2026-08-28 run: BMI ranged to **107,840** -- 149
# records above 200, plainly unit errors (grams for kilograms, centimetres for
# metres). Those 0.05 % of rows inflated the BMI-gap standard deviation from
# 7.06 to 298.89, a factor of 42. A dispersion statistic destroyed by 149 rows
# is not a finding about hearts.
#
# Rows outside the bounds are excluded from fitting and evaluation and counted
# in the model record, never silently clipped: clipping would invent a value,
# and the count is what tells a reader how much was dropped.
GAP_TARGET_BOUNDS: dict[str, tuple[float, float]] = {
    "age_gap": (18.0, 95.0),
    "bmi_gap": (12.0, 70.0),
}

# A gap is only a gap if the model predicts something. Prespecified floor on
# held-out R^2.
#
# Measured on the same run: age_gap reached 0.253, bmi_gap **-0.0000**. A model
# with R^2 = 0 predicts the population mean, so "gap" becomes the negated
# deviation from that mean -- and indeed the bmi_gap column correlated with
# BMI itself at r = -1.000, exactly. Reporting that as an ECG-derived marker
# would be reporting BMI with extra steps.
GAP_MINIMUM_VALIDATION_R2 = 0.05


def run_physiological_gap_discovery(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    feature_ids: tuple[str, ...],
    output_root: Path,
    random_seed: int = 20260725,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Fit and evaluate the physiological gap markers (age-gap, bmi-gap)
    and persist them as project artifacts.

    Reads only ``train``/``validation`` rows; Locked Test is never touched.
    Follows the same content-hashed, idempotent, atomically-written stage
    pattern as every other stage in this module (see
    ``association_domains.run_context_association_discovery`` for the
    closest sibling).
    """
    context_path = config.evidence_root / "clinical_context_features.parquet"
    if not context_path.is_file():
        raise FileNotFoundError(context_path)
    feature_paths = sorted(Path(path) for path in glob.glob(feature_glob))
    if not feature_paths:
        raise FileNotFoundError(
            f"No feature files matched {feature_glob!r}; run "
            "extract-features first."
        )
    input_hash = content_hash(
        {
            "context": file_fingerprint(context_path),
            "features": [file_fingerprint(path) for path in feature_paths],
            "feature_ids": list(feature_ids),
            "gap_model_version": GAP_MODEL_VERSION,
        }
    )
    config_hash = scientific_config_hash(config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "physiological_gap.manifest.json"
    models_path = output_root / "physiological_gap_models.json"
    markers_path = output_root / "physiological_gap_markers.parquet"
    sex_model_path = output_root / "sex_discordance_model.json"
    sex_markers_path = output_root / "sex_discordance_markers.parquet"
    if manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, GAP_STAGE_VERSION):
            return previous

    train, validation = _load_gap_population(
        config,
        feature_glob=feature_glob,
        context_path=context_path,
        feature_ids=feature_ids,
    )

    models: dict[str, GapMarkerModel] = {}
    long_rows: list[pd.DataFrame] = []
    for target_id in GAP_TARGETS:
        if progress:
            progress(f"Physiologischer Gap-Marker: {target_id} wird trainiert")
        try:
            model, train_rows, validation_rows = fit_gap_marker_model(
                train,
                validation,
                target_id=target_id,
                feature_ids=feature_ids,
                random_seed=random_seed,
            )
        except ValueError as exc:
            if progress:
                progress(f"Physiologischer Gap-Marker {target_id}: {exc}")
            continue
        models[target_id] = model
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            tagged = rows.copy()
            tagged["target_id"] = target_id
            tagged["split"] = split
            long_rows.append(tagged)

    markers = (
        pd.concat(long_rows, ignore_index=True)
        if long_rows
        else pd.DataFrame(
            columns=[
                "subject_id",
                "study_id",
                "predicted_value",
                "actual_value",
                "evaluable",
                "gap_value",
                "target_id",
                "split",
            ]
        )
    )
    atomic_write_parquet(markers, markers_path)
    atomic_write_json(
        models_path,
        {target_id: model.to_dict() for target_id, model in models.items()},
    )

    output_paths = [str(models_path.resolve()), str(markers_path.resolve())]
    sex_row_count = 0
    if "sex" in train.columns:
        if progress:
            progress(
                "Sex-Diskordanz-Marker (Plausibilitäts-/Fairness-Signal) "
                "wird trainiert"
            )
        try:
            sex_model, sex_train_rows, sex_validation_rows = (
                fit_sex_discordance_model(
                    train,
                    validation,
                    feature_ids=feature_ids,
                    random_seed=random_seed,
                )
            )
        except ValueError as exc:
            if progress:
                progress(f"Sex-Diskordanz-Marker: {exc}")
        else:
            sex_train_rows = sex_train_rows.copy()
            sex_train_rows["split"] = "train"
            sex_validation_rows = sex_validation_rows.copy()
            sex_validation_rows["split"] = "validation"
            sex_markers = pd.concat(
                (sex_train_rows, sex_validation_rows),
                ignore_index=True,
            )
            atomic_write_parquet(sex_markers, sex_markers_path)
            atomic_write_json(sex_model_path, sex_model.to_dict())
            output_paths.extend(
                (str(sex_model_path.resolve()), str(sex_markers_path.resolve()))
            )
            sex_row_count = len(sex_markers)

    manifest = StageManifest.complete(
        stage="physiological-gap-markers",
        stage_version=GAP_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(output_paths),
        row_counts={
            "physiological_gap_markers": len(markers),
            "sex_discordance_markers": sex_row_count,
        },
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            "Train gaps/probabilities are out-of-fold predictions from "
            "patient-grouped cross-validation; Validation values come from "
            "a single model refit on all of Train. Locked Test was not "
            "read. The sex-discordance model is a secondary plausibility/"
            "fairness signal, not a primary marker.",
        ),
    )
    manifest.save(manifest_path)
    return manifest


def load_sex_discordance_model(
    output_root: Path,
) -> Optional[SexDiscordanceModel]:
    model_path = output_root / "sex_discordance_model.json"
    if not model_path.is_file():
        return None
    return SexDiscordanceModel.from_dict(
        json.loads(model_path.read_text(encoding="utf-8"))
    )


def load_physiological_gap_models(
    output_root: Path,
) -> dict[str, GapMarkerModel]:
    models_path = output_root / "physiological_gap_models.json"
    if not models_path.is_file():
        return {}
    payload = json.loads(models_path.read_text(encoding="utf-8"))
    return {
        target_id: GapMarkerModel.from_dict(data)
        for target_id, data in payload.items()
    }


GAP_COLUMN_BLOCK_SIZE = 200


def _fetch_gap_population_in_blocks(
    connection,
    *,
    feature_sql: str,
    context_sql: str,
    selected: tuple[str, ...],
    block_size: int = GAP_COLUMN_BLOCK_SIZE,
) -> pd.DataFrame:
    """Read the cohort in column blocks instead of one wide result.

    **Measured size, after an estimate that was wrong.** This was built on the
    assumption that the stage materialises every marker in the feature file --
    640,104 x 1,753 doubles, about 9 GB. It does not. ``feature_ids`` comes
    from the ``primary_core`` panel alone: 80 markers, 640,104 rows, **0.41
    GB**, roughly 0.82 GB while ``_load_gap_population`` slices train and
    validation out of it. At that size one block covers the whole selection
    and blocking changes nothing.

    Blocking is kept because it bounds the cost if the panel ever grows.  For
    the current, single-block panel the keys and marker columns are fetched in
    one query; this avoids an otherwise redundant second scan of every feature
    partition. The real memory risk for this
    stage was on the *read*: it opens the full 1,889-column glob, where
    ``union_by_name`` cost 19.36 GB of per-file schema state outside DuckDB's
    limit. See ``storage.feature_parquet_clause``.

    Blocking follows the pattern ``_SharedValidationFeatureMatrix`` already
    uses in validation.py: decode in column blocks, keep the row identity, and
    let the frame grow one block at a time.

    Dtypes are deliberately untouched. Narrowing to float32 would halve the
    payload and change model fits, and a memory fix has no business moving
    numbers.
    """

    keys = ["subject_id", "study_id", "split", "age", "bmi_before_ecg", "sex"]
    # One read clause for the key frame and every block, so they cannot
    # disagree about which partitions they see. Whether the schemas have to
    # be unified is decided from the partition footers -- on this project
    # that removes 17 GB of per-file schema state per read; see
    # ``storage.feature_parquet_clause``.
    read_features = feature_parquet_clause(sql_path=feature_sql)
    normalized_block_size = max(1, int(block_size))
    if selected and len(selected) <= normalized_block_size:
        columns = ", ".join(
            f"f.{_quote_identifier(item)}" for item in selected
        )
        return connection.execute(
            f"""
            SELECT
                f.subject_id,
                f.study_id,
                f.split,
                c.age,
                c.bmi_before_ecg,
                c.sex,
                {columns}
            FROM {read_features} f
            JOIN read_parquet('{context_sql}') c
              USING (subject_id, study_id, split)
            WHERE f.split IN ('train', 'validation')
              AND f.extraction_status = 'ok'
            ORDER BY f.subject_id, f.study_id, f.split
            """
        ).fetchdf()[keys + list(selected)]

    frame = connection.execute(
        f"""
        SELECT
            f.subject_id,
            f.study_id,
            f.split,
            c.age,
            c.bmi_before_ecg,
            c.sex
        FROM {read_features} f
        JOIN read_parquet('{context_sql}') c
          USING (subject_id, study_id, split)
        WHERE f.split IN ('train', 'validation')
          AND f.extraction_status = 'ok'
        ORDER BY f.subject_id, f.study_id, f.split
        """
    ).fetchdf()
    if not selected:
        return frame

    for start in range(0, len(selected), normalized_block_size):
        block = selected[start : start + normalized_block_size]
        columns = ", ".join(f"f.{_quote_identifier(item)}" for item in block)
        # Same filter, same ORDER BY, so the block lines up row for row with
        # the key frame. Joining on the keys again would cost another hash
        # table per block for no gain.
        part = connection.execute(
            f"""
            SELECT {columns}
            FROM {read_features} f
            JOIN read_parquet('{context_sql}') c
              USING (subject_id, study_id, split)
            WHERE f.split IN ('train', 'validation')
              AND f.extraction_status = 'ok'
            ORDER BY f.subject_id, f.study_id, f.split
            """
        ).fetchdf()
        if len(part) != len(frame):
            raise RuntimeError(
                "Gap-Population: Spaltenblock hat "
                f"{len(part):,} Zeilen, Schluesselrahmen {len(frame):,} -- "
                "die Blockreihenfolge stimmt nicht ueberein"
            )
        for column in part.columns:
            frame[column] = part[column].to_numpy(copy=False)
        del part
    return frame[keys + [item for item in selected]]


def _load_gap_population(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    context_path: Path,
    feature_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    context_sql = _sql_path(context_path)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        available = parquet_columns(connection, feature_sql)
        selected = tuple(item for item in feature_ids if item in available)
        frame = _fetch_gap_population_in_blocks(
            connection,
            feature_sql=feature_sql,
            context_sql=context_sql,
            selected=selected,
        )
    for feature_id in feature_ids:
        if feature_id not in frame:
            frame[feature_id] = np.nan
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(
        drop=True
    )
    return train, validation


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


# ---------------------------------------------------------------------------
# Sex-discordance plausibility marker (smaller companion to the age/BMI gap
# markers): a classification analogue. Sex is categorical, so there is no
# continuous "gap" — instead, low-confidence or misclassified cases are
# flagged as a research/fairness signal, not as a primary target in its own
# right. Same architecture and discipline as the gap models above: Train-only
# fit, patient-grouped CV, Validation-only evaluation, Locked Test untouched.
# ---------------------------------------------------------------------------

_SEX_C_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)


@dataclass(frozen=True)
class SexDiscordanceModel:
    feature_names: tuple[str, ...]
    feature_medians: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    inverse_regularization: float
    training_subject_hash: str
    n_train_subjects: int
    train_out_of_fold_auc: float
    oof_method: str = "nested_grouped_cv"
    validation_auc: float = float("nan")
    validation_accuracy: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SexDiscordanceModel":
        payload = dict(data)
        for key in (
            "feature_names",
            "feature_medians",
            "feature_means",
            "feature_scales",
            "coefficients",
        ):
            payload[key] = tuple(payload[key])
        return cls(**payload)


def fit_sex_discordance_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_ids: tuple[str, ...],
    n_folds: int = 5,
    minimum_train_subjects: int = 200,
    random_seed: int = 20260725,
) -> tuple[SexDiscordanceModel, pd.DataFrame, pd.DataFrame]:
    """Fit the sex-discordance plausibility marker.

    Returns ``(model, train_rows, validation_rows)`` with columns
    ``subject_id, study_id, predicted_probability_male, actual_sex_male,
    evaluable, discordant`` (``discordant`` = predicted class != actual
    class at a 0.5 threshold). Train rows are out-of-fold predictions.
    """
    normalized_sex = train["sex"].astype("string").str.upper()
    train_target = pd.Series(np.nan, index=train.index, dtype=float)
    train_target.loc[normalized_sex.eq("M").fillna(False)] = 1.0
    train_target.loc[normalized_sex.eq("F").fillna(False)] = 0.0
    train_has_target = train_target.notna().to_numpy()
    train_matrix, train_coverage = _design_matrix(train, feature_ids)
    train_usable = train_has_target & (
        train_coverage >= MINIMUM_FEATURE_COVERAGE
    )
    usable_subjects = pd.Series(
        train.loc[train_usable, "subject_id"]
    ).nunique()
    if usable_subjects < minimum_train_subjects:
        raise ValueError(
            "Sex-discordance model needs at least "
            f"{minimum_train_subjects} usable train subjects; got "
            f"{usable_subjects}"
        )

    fit_matrix = train_matrix[train_usable]
    fit_target = train_target.to_numpy(dtype=float)[train_usable].astype(
        np.int8
    )
    fit_subjects = train.loc[train_usable, "subject_id"].to_numpy()

    medians = np.nanmedian(fit_matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(fit_matrix), fit_matrix, medians[np.newaxis, :])
    means = np.mean(filled, axis=0)
    scales = np.std(filled, axis=0, ddof=1)
    scales = np.where(scales > 1e-9, scales, 1.0)

    best = _select_sex_hyperparameters_by_grouped_cv(
        fit_matrix,
        fit_target,
        fit_subjects,
        n_folds=n_folds,
        random_seed=random_seed,
    )

    standardized_fit = _impute_and_standardize(
        fit_matrix,
        medians=medians,
        means=means,
        scales=scales,
    )
    final_model = LogisticRegression(
        C=best["c"],
        max_iter=5000,
        random_state=random_seed,
    )
    final_model.fit(standardized_fit, fit_target)

    model = SexDiscordanceModel(
        feature_names=tuple(feature_ids),
        feature_medians=tuple(float(value) for value in medians),
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in final_model.coef_[0]),
        intercept=float(final_model.intercept_[0]),
        inverse_regularization=float(best["c"]),
        training_subject_hash=content_hash(
            sorted(str(value) for value in np.unique(fit_subjects))
        ),
        n_train_subjects=int(usable_subjects),
        train_out_of_fold_auc=float(best["auc"]),
        oof_method=str(best["oof_method"]),
    )

    train_rows = pd.DataFrame(
        {
            "subject_id": train.loc[train_usable, "subject_id"].to_numpy(),
            "study_id": _study_ids(train, train_usable),
            "predicted_probability_male": best["out_of_fold_probabilities"],
            "actual_sex_male": fit_target.astype(float),
            "evaluable": True,
        }
    )
    train_rows["discordant"] = (
        train_rows["predicted_probability_male"] >= 0.5
    ) != (train_rows["actual_sex_male"] >= 0.5)
    not_evaluable_mask = ~train_usable
    not_evaluable_train = pd.DataFrame(
        {
            "subject_id": train.loc[not_evaluable_mask, "subject_id"].to_numpy(),
            "study_id": _study_ids(train, not_evaluable_mask),
            "predicted_probability_male": np.nan,
            "actual_sex_male": train_target.to_numpy(dtype=float)[
                not_evaluable_mask
            ],
            "evaluable": False,
        }
    )
    not_evaluable_train["discordant"] = pd.NA
    train_output = pd.concat(
        (train_rows, not_evaluable_train),
        ignore_index=True,
    )

    validation_output = apply_sex_discordance_model(validation, model)
    validation_evaluable = validation_output.loc[
        validation_output["evaluable"]
    ]
    if (
        len(validation_evaluable) >= 2
        and validation_evaluable["actual_sex_male"].nunique() == 2
    ):
        model = _with_sex_validation_metrics(
            model,
            auc=float(
                roc_auc_score(
                    validation_evaluable["actual_sex_male"],
                    validation_evaluable["predicted_probability_male"],
                )
            ),
            accuracy=float(
                (
                    (
                        validation_evaluable["predicted_probability_male"]
                        >= 0.5
                    )
                    == (validation_evaluable["actual_sex_male"] >= 0.5)
                ).mean()
            ),
        )
    return model, train_output, validation_output


def _with_sex_validation_metrics(
    model: SexDiscordanceModel,
    *,
    auc: float,
    accuracy: float,
) -> SexDiscordanceModel:
    payload = model.to_dict()
    payload["validation_auc"] = auc
    payload["validation_accuracy"] = accuracy
    return SexDiscordanceModel.from_dict(payload)


def _select_sex_hyperparameters_by_grouped_cv(
    matrix: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    *,
    n_folds: int,
    random_seed: int,
) -> dict:
    unique_subjects = np.unique(subjects)
    effective_folds = max(2, min(n_folds, len(unique_subjects)))
    outer = GroupKFold(n_splits=effective_folds)
    honest_oof = np.full(len(target), np.nan, dtype=float)
    outer_parameters: list[float] = []
    for outer_train, outer_test in outer.split(matrix, target, groups=subjects):
        selected = _best_sex_parameter(
            matrix[outer_train],
            target[outer_train],
            subjects[outer_train],
            n_folds=min(n_folds, len(np.unique(subjects[outer_train]))),
            random_seed=random_seed,
        )
        outer_parameters.append(float(selected["c"]))
        if np.unique(target[outer_train]).size < 2:
            continue
        medians, means, scales = _fold_preprocessing(matrix[outer_train])
        estimator = LogisticRegression(
            C=selected["c"],
            max_iter=5000,
            random_state=random_seed,
        )
        estimator.fit(
            _impute_and_standardize(
                matrix[outer_train],
                medians=medians,
                means=means,
                scales=scales,
            ),
            target[outer_train],
        )
        honest_oof[outer_test] = estimator.predict_proba(
            _impute_and_standardize(
                matrix[outer_test],
                medians=medians,
                means=means,
                scales=scales,
            )
        )[:, 1]
    final = _best_sex_parameter(
        matrix,
        target,
        subjects,
        n_folds=effective_folds,
        random_seed=random_seed,
    )
    valid = np.isfinite(honest_oof)
    auc = (
        float(roc_auc_score(target[valid], honest_oof[valid]))
        if valid.any() and np.unique(target[valid]).size == 2
        else float("nan")
    )
    return {
        **final,
        "auc": auc,
        "out_of_fold_probabilities": honest_oof,
        "oof_method": "nested_grouped_cv",
        "outer_selected_parameters": outer_parameters,
    }


def _sex_grid_point(
    matrix: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    c_value: float,
    effective_folds: int,
    random_seed: int,
) -> float:
    """Grouped out-of-fold AUC of one C value."""
    splitter = GroupKFold(n_splits=effective_folds)
    probabilities = np.full(len(target), np.nan, dtype=float)
    for train_index, test_index in splitter.split(
        matrix, target, groups=subjects
    ):
        if np.unique(target[train_index]).size < 2:
            continue
        medians, means, scales = _fold_preprocessing(matrix[train_index])
        estimator = LogisticRegression(
            C=c_value,
            max_iter=5000,
            random_state=random_seed,
        )
        estimator.fit(
            _impute_and_standardize(
                matrix[train_index],
                medians=medians,
                means=means,
                scales=scales,
            ),
            target[train_index],
        )
        probabilities[test_index] = estimator.predict_proba(
            _impute_and_standardize(
                matrix[test_index],
                medians=medians,
                means=means,
                scales=scales,
            )
        )[:, 1]
    valid = np.isfinite(probabilities)
    return (
        float(roc_auc_score(target[valid], probabilities[valid]))
        if valid.any() and np.unique(target[valid]).size == 2
        else float("nan")
    )


def _best_sex_parameter(
    matrix: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    *,
    n_folds: int,
    random_seed: int,
) -> dict:
    effective_folds = max(2, min(int(n_folds), len(np.unique(subjects))))
    grid = [float(value) for value in _SEX_C_GRID]
    scores = _run_grid(
        _sex_grid_point,
        [
            (matrix, target, subjects, c_value, effective_folds, random_seed)
            for c_value in grid
        ],
        rows=len(target),
    )
    best: Optional[dict] = None
    for c_value, auc in zip(grid, scores):
        candidate = {"c": float(c_value), "selection_auc": auc}
        if best is None or (
            np.isfinite(auc)
            and (
                not np.isfinite(best["selection_auc"])
                or auc > best["selection_auc"]
            )
        ):
            best = candidate
    assert best is not None
    return best


def apply_sex_discordance_model(
    frame: pd.DataFrame,
    model: SexDiscordanceModel,
) -> pd.DataFrame:
    """Score new rows with an already-fitted, frozen sex-discordance
    model. Never refits anything."""
    matrix, coverage = _design_matrix(frame, model.feature_names)
    standardized = _impute_and_standardize(
        matrix,
        medians=np.asarray(model.feature_medians, dtype=float),
        means=np.asarray(model.feature_means, dtype=float),
        scales=np.asarray(model.feature_scales, dtype=float),
    )
    linear = model.intercept + standardized @ np.asarray(
        model.coefficients, dtype=float
    )
    probability_male = 1.0 / (1.0 + np.exp(-np.clip(linear, -40, 40)))
    normalized_sex = frame["sex"].astype("string").str.upper()
    actual = pd.Series(np.nan, index=frame.index, dtype=float)
    actual.loc[normalized_sex.eq("M").fillna(False)] = 1.0
    actual.loc[normalized_sex.eq("F").fillna(False)] = 0.0
    actual_values = actual.to_numpy(dtype=float)
    evaluable = (coverage >= MINIMUM_FEATURE_COVERAGE) & np.isfinite(
        actual_values
    )
    output = pd.DataFrame(
        {
            "subject_id": frame["subject_id"].to_numpy(),
            "study_id": (
                frame["study_id"].to_numpy()
                if "study_id" in frame
                else np.arange(len(frame))
            ),
            "predicted_probability_male": np.where(
                evaluable, probability_male, np.nan
            ),
            "actual_sex_male": np.where(evaluable, actual_values, np.nan),
            "evaluable": evaluable,
        }
    )
    output["discordant"] = np.where(
        evaluable,
        (output["predicted_probability_male"] >= 0.5)
        != (output["actual_sex_male"] >= 0.5),
        pd.NA,
    )
    return output
