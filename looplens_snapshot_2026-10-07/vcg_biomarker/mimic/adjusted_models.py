from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .statistics import (
    evaluate_marker,
    inference_status,
    marker_bootstrap_intervals,
    paired_auc_difference_bootstrap,
)
from .storage import content_hash


_ANCHOR_YEAR_GROUP_RE = re.compile(r"(\d{4})")


def anchor_year_group_start(values: Iterable[object]) -> pd.Series:
    """Map MIMIC-IV ``anchor_year_group`` strings to their true start year.

    MIMIC-IV shifts every patient's timestamps by a random per-patient
    offset, so the year inside ``ecg_time`` says nothing about real calendar
    time across patients: measured on this project's own data, ``anchor_year``
    ranges from 2110 to well past 2200 inside *every* true three-year group.
    ``anchor_year_group`` (e.g. ``"2014 - 2016"``) is the only field that
    carries the real era, so it -- and not the shifted year -- is what a
    secular-trend covariate or an era subgroup may be built from.

    Returns a float Series of start years, NaN where the group is missing or
    unparseable.
    """

    series = pd.Series(list(values), dtype="object")
    parsed = series.astype("string").str.extract(
        _ANCHOR_YEAR_GROUP_RE,
        expand=False,
    )
    return pd.to_numeric(parsed, errors="coerce").astype(float)


@dataclass(frozen=True)
class FrozenLogisticModel:
    model_id: str
    endpoint_id: str
    predictor_names: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    standardize: tuple[bool, ...]
    coefficients: tuple[float, ...]
    baseline_model_id: Optional[str]
    training_subject_hash: str
    events_per_parameter: float
    model_role: str = "secondary_adjusted"

    def __post_init__(self) -> None:
        size = len(self.predictor_names)
        if size == 0:
            raise ValueError("An adjusted model needs at least one predictor")
        if not (
            len(self.means)
            == len(self.standard_deviations)
            == len(self.standardize)
            == size
            and len(self.coefficients) == size + 1
        ):
            raise ValueError("Frozen logistic model arrays have inconsistent lengths")
        if any(
            flag and (not np.isfinite(sd) or sd <= 0)
            for flag, sd in zip(self.standardize, self.standard_deviations)
        ):
            raise ValueError("Standardized predictors need positive training SDs")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "FrozenLogisticModel":
        payload = dict(data)
        for key in (
            "predictor_names",
            "means",
            "standard_deviations",
            "standardize",
            "coefficients",
        ):
            payload[key] = tuple(payload[key])
        return cls(**payload)


@dataclass(frozen=True)
class AdjustedModelDefinition:
    model_id: str
    predictor_names: tuple[str, ...]
    standardize: tuple[bool, ...]
    baseline_model_id: Optional[str] = None


def fit_repeated_ecg_clustered_marker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    endpoint_id: str,
    feature_id: str,
    direction: int,
    minimum_events_per_parameter: int = 20,
    bootstrap_iterations: int = 0,
    random_seed: int = 20260725,
) -> tuple[pd.DataFrame, dict]:
    train_design = _repeated_marker_design(train, feature_id, direction)
    validation_design = _repeated_marker_design(
        validation,
        feature_id,
        direction,
    )
    valid_train = train_design.dropna(
        subset=["subject_id", "label", "vcg_marker"]
    )
    subject_class_counts = (
        valid_train.groupby("label")["subject_id"].nunique().to_dict()
    )
    required_subjects = minimum_events_per_parameter * 2
    if (
        subject_class_counts.get(0, 0) < required_subjects
        or subject_class_counts.get(1, 0) < required_subjects
    ):
        raise ValueError(
            "Repeated-ECG clustered model needs at least "
            f"{required_subjects} unique subjects in each class; got "
            f"{subject_class_counts.get(1, 0)}/"
            f"{subject_class_counts.get(0, 0)}"
        )
    definition = AdjustedModelDefinition(
        model_id=f"{endpoint_id}__all_ecg_clustered_vcg",
        predictor_names=("vcg_marker",),
        standardize=(True,),
    )
    frozen, coefficients = fit_logistic_model(
        train_design,
        endpoint_id=endpoint_id,
        definition=definition,
        minimum_events_per_parameter=minimum_events_per_parameter,
    )
    frozen = replace(
        frozen,
        model_role="secondary_all_ecg_patient_clustered",
        training_subject_hash=content_hash(
            sorted(
                str(value)
                for value in valid_train["subject_id"].drop_duplicates()
            )
        ),
    )
    coefficient = coefficients.loc[
        coefficients["predictor"] == "vcg_marker"
    ].iloc[0]
    rows = []
    for split_index, (split, design) in enumerate(
        (("train", train_design), ("validation", validation_design))
    ):
        probabilities, labels, valid = predict_logistic_model(design, frozen)
        subjects = design.loc[valid, "subject_id"].to_numpy()
        metrics = evaluate_logistic_model(design, frozen)
        marker_metrics = evaluate_marker(probabilities, labels)
        unique_positive_subjects = int(
            pd.Series(subjects[labels == 1]).nunique()
        )
        unique_negative_subjects = int(
            pd.Series(subjects[labels == 0]).nunique()
        )
        status = inference_status(
            unique_positive_subjects,
            unique_negative_subjects,
        )
        uncertainty = marker_bootstrap_intervals(
            subjects,
            probabilities,
            labels,
            direction=1,
            thresholds={},
            iterations=(
                int(bootstrap_iterations)
                if status != "descriptive_only" and split == "validation"
                else 0
            ),
            seed=(
                int(random_seed)
                + int(content_hash(f"{endpoint_id}:all-ecg:{split}")[:8], 16)
                + split_index
            ),
        )
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "model_id": frozen.model_id,
                "model_role": frozen.model_role,
                "evaluation_split": split,
                "metric_scope": (
                    "all_ecg_train_in_sample"
                    if split == "train"
                    else "all_ecg_validation_holdout"
                ),
                "inference_method": (
                    "binomial_glm_with_subject_cluster_robust_covariance"
                ),
                "vcg_feature_id": feature_id,
                "vcg_direction": int(direction),
                "n_ecgs": int(len(labels)),
                "n_subjects": int(pd.Series(subjects).nunique()),
                "positive_subjects": unique_positive_subjects,
                "negative_subjects": unique_negative_subjects,
                "inference_status": status,
                "bootstrap_scope": (
                    "validation_subject_cluster"
                    if split == "validation"
                    else "not_run_for_train_in_sample"
                ),
                "vcg_or_per_sd": float(coefficient["odds_ratio"]),
                "vcg_or_ci_low": float(coefficient["odds_ratio_ci_low"]),
                "vcg_or_ci_high": float(coefficient["odds_ratio_ci_high"]),
                "vcg_p_value_clustered": float(coefficient["p_value"]),
                "p_value": float(coefficient["p_value"]),
                "roc_auc": marker_metrics["roc_auc"],
                "average_precision": marker_metrics["average_precision"],
                "pr_auc": marker_metrics["pr_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "calibration_intercept": metrics["calibration_intercept"],
                "calibration_slope": metrics["calibration_slope"],
                "decision_curve_eligible": False,
                **uncertainty,
            }
        )
    return (
        pd.DataFrame(rows),
        {
            "model": frozen.to_dict(),
            "coefficients": coefficients.to_dict(orient="records"),
            "vcg_feature_id": feature_id,
            "vcg_direction": int(direction),
            "analysis_population": "all_eligible_ecgs",
            "inference_method": (
                "binomial_glm_with_subject_cluster_robust_covariance"
            ),
        },
    )


def fit_endpoint_adjusted_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    endpoint_id: str,
    vcg_feature_id: str,
    vcg_direction: int,
    classical_feature_id: Optional[str],
    minimum_events_per_parameter: int = 20,
    bootstrap_iterations: int = 0,
    random_seed: int = 20260725,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, pd.DataFrame, list[dict]]:
    train_design = _endpoint_design_frame(
        train,
        endpoint_id=endpoint_id,
        vcg_feature_id=vcg_feature_id,
        vcg_direction=vcg_direction,
        classical_feature_id=classical_feature_id,
    )
    validation_design = _endpoint_design_frame(
        validation,
        endpoint_id=endpoint_id,
        vcg_feature_id=vcg_feature_id,
        vcg_direction=vcg_direction,
        classical_feature_id=classical_feature_id,
    )
    covariates = [
        "age",
        "sex_male",
        "heart_rate",
        "qrs_duration",
        "anchor_era_start_year",
        "care_setting_ed",
        # Acquisition hour as a circular pair. Recording volume swings 6.6-fold
        # across the day in this cohort, so the hour is a proxy for why the
        # ECG was ordered -- confounding by indication that a linear 0..23
        # covariate could not capture either (23:00 and 00:00 are adjacent).
        "acquisition_hour_sin",
        "acquisition_hour_cos",
    ]
    if _uses_bmi_covariate(endpoint_id):
        covariates.append("bmi")
    definitions = _model_definitions(
        covariates=tuple(covariates),
        classical_available=classical_feature_id is not None,
        report_available=(
            "report_positive" in train_design
            and train_design["report_positive"].notna().any()
        ),
    )
    metric_rows: list[dict] = []
    model_payloads: list[dict] = []
    decision_rows: list[dict] = []
    calibration_rows: list[dict] = []
    failures: list[dict] = []
    fitted_models: dict[str, FrozenLogisticModel] = {}
    for definition in definitions:
        try:
            frozen, coefficient_frame = fit_logistic_model(
                train_design,
                endpoint_id=endpoint_id,
                definition=definition,
                minimum_events_per_parameter=minimum_events_per_parameter,
            )
            train_metrics = evaluate_logistic_model(train_design, frozen)
            validation_metrics = evaluate_logistic_model(
                validation_design,
                frozen,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "model_id": definition.model_id,
                    "stage": "adjusted_model",
                    "reason": str(exc),
                }
            )
            continue
        model_payloads.append(
            {
                "model": frozen.to_dict(),
                "coefficients": coefficient_frame.to_dict(orient="records"),
                "vcg_feature_id": vcg_feature_id,
                "vcg_direction": vcg_direction,
                "classical_feature_id": classical_feature_id,
            }
        )
        fitted_models[frozen.model_id] = frozen
        for split, metrics in (
            ("train", train_metrics),
            ("validation", validation_metrics),
        ):
            split_frame = train_design if split == "train" else validation_design
            row = {
                "endpoint_id": endpoint_id,
                "model_id": frozen.model_id,
                "model_role": frozen.model_role,
                "evaluation_split": split,
                "metric_scope": (
                    "train_in_sample"
                    if split == "train"
                    else "validation_holdout"
                ),
                "predictors": "|".join(frozen.predictor_names),
                "parameter_count": len(frozen.coefficients),
                "events_per_parameter_train": frozen.events_per_parameter,
                "baseline_model_id": frozen.baseline_model_id,
                "vcg_feature_id": (
                    vcg_feature_id
                    if "vcg_marker" in frozen.predictor_names
                    else None
                ),
                "automatic_feature_selection": False,
                **metrics,
            }
            vcg_coefficient = coefficient_frame.loc[
                coefficient_frame["predictor"] == "vcg_marker"
            ]
            if not vcg_coefficient.empty:
                coefficient = vcg_coefficient.iloc[0]
                row.update(
                    {
                        "vcg_or_per_sd": coefficient["odds_ratio"],
                        "vcg_or_ci_low": coefficient["odds_ratio_ci_low"],
                        "vcg_or_ci_high": coefficient["odds_ratio_ci_high"],
                        "vcg_p_value": coefficient["p_value"],
                    }
                )
            metric_rows.append(row)
            calibration_rows.extend(
                calibration_curve_points(
                    split_frame,
                    frozen,
                    evaluation_split=split,
                )
            )
        if validation_metrics.get("decision_curve_eligible", False):
            decision_rows.extend(
                decision_curve(
                    validation_design,
                    frozen,
                    thresholds=np.arange(0.05, 0.51, 0.05),
                )
            )

    validation_labels = pd.to_numeric(
        validation_design["label"],
        errors="coerce",
    ).to_numpy(dtype=float)
    for row in metric_rows:
        if row["evaluation_split"] != "validation":
            continue
        baseline_id = row.get("baseline_model_id")
        candidate_model = fitted_models.get(str(row["model_id"]))
        baseline_model = fitted_models.get(str(baseline_id))
        if baseline_id and candidate_model is not None and baseline_model is not None:
            candidate_scores = _full_model_probabilities(
                validation_design,
                candidate_model,
            )
            baseline_scores = _full_model_probabilities(
                validation_design,
                baseline_model,
            )
            paired_status = inference_status(
                int(
                    np.sum(
                        np.isfinite(candidate_scores)
                        & np.isfinite(baseline_scores)
                        & (validation_labels == 1)
                    )
                ),
                int(
                    np.sum(
                        np.isfinite(candidate_scores)
                        & np.isfinite(baseline_scores)
                        & (validation_labels == 0)
                    )
                ),
            )
            row.update(
                paired_auc_difference_bootstrap(
                    validation_design["subject_id"].to_numpy(),
                    validation_labels,
                    candidate_scores,
                    baseline_scores,
                    iterations=(
                        int(bootstrap_iterations)
                        if paired_status != "descriptive_only"
                        else 0
                    ),
                    seed=(
                        int(random_seed)
                        + int(content_hash(row["model_id"])[:8], 16)
                    ),
                )
            )
            row["incremental_comparison_status"] = paired_status
        else:
            row["incremental_auc"] = np.nan
    return (
        pd.DataFrame(metric_rows),
        model_payloads,
        pd.DataFrame(decision_rows),
        pd.DataFrame(calibration_rows),
        failures,
    )


def fit_logistic_model(
    train: pd.DataFrame,
    *,
    endpoint_id: str,
    definition: AdjustedModelDefinition,
    minimum_events_per_parameter: int = 20,
) -> tuple[FrozenLogisticModel, pd.DataFrame]:
    missing = sorted(set(definition.predictor_names) - set(train.columns))
    if missing:
        raise ValueError(f"Predictors are unavailable: {missing}")
    labels = pd.to_numeric(train["label"], errors="coerce").to_numpy(dtype=float)
    predictors = np.column_stack(
        [
            pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=float)
            for name in definition.predictor_names
        ]
    )
    valid = np.isfinite(labels) & np.all(np.isfinite(predictors), axis=1)
    labels = labels[valid].astype(np.int8)
    predictors = predictors[valid]
    subjects = train.loc[valid, "subject_id"].to_numpy()
    if np.unique(labels).size < 2:
        raise ValueError("Both classes are required for adjusted modeling")
    parameter_count = predictors.shape[1] + 1
    positive = int(pd.Series(subjects[labels == 1]).nunique())
    negative = int(pd.Series(subjects[labels == 0]).nunique())
    required = int(minimum_events_per_parameter) * parameter_count
    if positive < required or negative < required:
        raise ValueError(
            f"Need at least {required} positive and negative training subjects "
            f"for {parameter_count} parameters; got {positive}/{negative}"
        )

    means = np.mean(predictors, axis=0)
    deviations = np.std(predictors, axis=0, ddof=1)
    transformed = predictors.copy()
    for index, should_standardize in enumerate(definition.standardize):
        if not should_standardize:
            means[index] = 0.0
            deviations[index] = 1.0
            continue
        if not np.isfinite(deviations[index]) or deviations[index] <= 0:
            raise ValueError(
                f"Predictor {definition.predictor_names[index]!r} is constant"
            )
        transformed[:, index] = (
            transformed[:, index] - means[index]
        ) / deviations[index]

    design = sm.add_constant(transformed, prepend=True, has_constant="add")
    fitted = sm.GLM(labels, design, family=sm.families.Binomial()).fit()
    try:
        robust = sm.GLM(
            labels,
            design,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": subjects})
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "Subject-clustered covariance estimation failed"
        ) from exc
    parameters = np.asarray(fitted.params, dtype=float)
    robust_se = np.asarray(robust.bse, dtype=float)
    p_values = np.asarray(robust.pvalues, dtype=float)
    coefficient_names = ("intercept",) + definition.predictor_names
    coefficient_frame = pd.DataFrame(
        {
            "predictor": coefficient_names,
            "coefficient": parameters,
            "standard_error_clustered": robust_se,
            "p_value": p_values,
            "odds_ratio": np.exp(parameters),
            "odds_ratio_ci_low": np.exp(parameters - 1.96 * robust_se),
            "odds_ratio_ci_high": np.exp(parameters + 1.96 * robust_se),
        }
    )
    frozen = FrozenLogisticModel(
        model_id=definition.model_id,
        endpoint_id=endpoint_id,
        predictor_names=definition.predictor_names,
        means=tuple(float(value) for value in means),
        standard_deviations=tuple(float(value) for value in deviations),
        standardize=definition.standardize,
        coefficients=tuple(float(value) for value in parameters),
        baseline_model_id=definition.baseline_model_id,
        training_subject_hash=content_hash(
            sorted(str(value) for value in subjects)
        ),
        events_per_parameter=float(min(positive, negative) / parameter_count),
    )
    return frozen, coefficient_frame


def evaluate_logistic_model(
    frame: pd.DataFrame,
    model: FrozenLogisticModel,
) -> dict:
    probabilities, labels, _ = predict_logistic_model(frame, model)
    result: dict[str, object] = {
        "n_total": len(labels),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "missingness": float(1.0 - len(labels) / max(len(frame), 1)),
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "brier_score": np.nan,
        "log_loss": np.nan,
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
        "decision_curve_eligible": False,
    }
    if len(labels) == 0 or np.unique(labels).size < 2:
        result["inference_status"] = "descriptive_only"
        return result
    result.update(
        {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(
                average_precision_score(labels, probabilities)
            ),
            "brier_score": float(brier_score_loss(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities)),
        }
    )
    intercept, slope = calibration_intercept_slope(labels, probabilities)
    result["calibration_intercept"] = intercept
    result["calibration_slope"] = slope
    result["inference_status"] = inference_status(
        result["n_positive"],
        result["n_negative"],
    )
    result["decision_curve_eligible"] = bool(
        result["inference_status"] == "formal_inference"
        and np.isfinite(intercept)
        and np.isfinite(slope)
        and abs(intercept) <= 0.2
        and 0.8 <= slope <= 1.2
    )
    return result


def predict_logistic_model(
    frame: pd.DataFrame,
    model: FrozenLogisticModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = sorted(set(model.predictor_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Predictors are unavailable: {missing}")
    labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=float)
    predictors = np.column_stack(
        [
            pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            for name in model.predictor_names
        ]
    )
    valid = np.isfinite(labels) & np.all(np.isfinite(predictors), axis=1)
    transformed = predictors[valid].copy()
    for index, should_standardize in enumerate(model.standardize):
        if should_standardize:
            transformed[:, index] = (
                transformed[:, index] - model.means[index]
            ) / model.standard_deviations[index]
    linear = model.coefficients[0] + transformed @ np.asarray(
        model.coefficients[1:],
        dtype=float,
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(linear, -40, 40)))
    return probabilities, labels[valid].astype(np.int8), valid


def _full_model_probabilities(
    frame: pd.DataFrame,
    model: FrozenLogisticModel,
) -> np.ndarray:
    probabilities, _, valid = predict_logistic_model(frame, model)
    output: np.ndarray = np.full(len(frame), np.nan, dtype=float)
    output[valid] = probabilities
    return output


def _repeated_marker_design(
    frame: pd.DataFrame,
    feature_id: str,
    direction: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": pd.to_numeric(
                frame["subject_id"],
                errors="coerce",
            ),
            "label": pd.to_numeric(frame["label"], errors="coerce"),
            "vcg_marker": int(direction)
            * pd.to_numeric(frame[feature_id], errors="coerce"),
        }
    )


def calibration_intercept_slope(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    if len(labels) < 20 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    design = sm.add_constant(logits, prepend=True, has_constant="add")
    try:
        fitted = sm.GLM(
            labels,
            design,
            family=sm.families.Binomial(),
        ).fit()
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), float("nan")
    return float(fitted.params[0]), float(fitted.params[1])


def calibration_curve_points(
    frame: pd.DataFrame,
    model: FrozenLogisticModel,
    *,
    evaluation_split: str,
    maximum_bins: int = 10,
) -> list[dict]:
    """Return observed-versus-predicted calibration bins with Wilson CIs.

    Bins are equal-frequency and derived independently within the requested
    evaluation split.  This artifact is descriptive; calibration intercept
    and slope remain the formal model-level summaries.
    """

    probabilities, labels, _ = predict_logistic_model(frame, model)
    if len(labels) == 0:
        return []
    bin_count = min(int(maximum_bins), max(2, len(labels) // 25))
    if np.unique(probabilities).size < 2:
        bin_ids: np.ndarray = np.zeros(len(labels), dtype=int)
    else:
        bin_ids = np.asarray(
            pd.qcut(
                probabilities,
                q=min(bin_count, int(np.unique(probabilities).size)),
                labels=False,
                duplicates="drop",
            ),
            dtype=int,
        )
    rows: list[dict] = []
    for bin_id in sorted(np.unique(bin_ids)):
        selected = bin_ids == bin_id
        count = int(np.sum(selected))
        events = int(np.sum(labels[selected] == 1))
        ci_low, ci_high = proportion_confint(
            events,
            count,
            alpha=0.05,
            method="wilson",
        )
        rows.append(
            {
                "endpoint_id": model.endpoint_id,
                "model_id": model.model_id,
                "evaluation_split": str(evaluation_split),
                "calibration_bin": int(bin_id) + 1,
                "n": count,
                "events": events,
                "mean_predicted": float(np.mean(probabilities[selected])),
                "observed_rate": float(events / count),
                "observed_ci_low": float(ci_low),
                "observed_ci_high": float(ci_high),
            }
        )
    return rows


def decision_curve(
    frame: pd.DataFrame,
    model: FrozenLogisticModel,
    *,
    thresholds: Iterable[float],
    evaluation_split: str = "validation",
) -> list[dict]:
    probabilities, labels, _ = predict_logistic_model(frame, model)
    if len(labels) == 0:
        return []
    prevalence = float(np.mean(labels))
    rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0 < threshold < 1:
            continue
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        n = len(labels)
        odds = threshold / (1.0 - threshold)
        net_benefit = tp / n - fp / n * odds
        treat_all = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "endpoint_id": model.endpoint_id,
                "model_id": model.model_id,
                "evaluation_split": evaluation_split,
                "threshold_probability": threshold,
                "net_benefit": float(net_benefit),
                "treat_all_net_benefit": float(treat_all),
                "treat_none_net_benefit": 0.0,
                "eligibility": "validated_calibrated_model",
                # This is an internal, model-by-model development curve.  It
                # has neither an external standard-care comparison nor paired
                # uncertainty and must therefore never be promoted to AHA
                # clinical-utility evidence.
                "evidence_scope": "internal_development_descriptive",
                "clinical_utility_evidence_eligible": False,
                "incremental_standard_care_comparator": False,
                "paired_net_benefit_uncertainty_reported": False,
            }
        )
    return rows


def _endpoint_design_frame(
    frame: pd.DataFrame,
    *,
    endpoint_id: str,
    vcg_feature_id: str,
    vcg_direction: int,
    classical_feature_id: Optional[str],
) -> pd.DataFrame:
    normalized_sex = frame["sex"].astype("string").str.upper()
    sex_male = pd.Series(np.nan, index=frame.index, dtype=float)
    sex_male.loc[normalized_sex.eq("M").fillna(False)] = 1.0
    sex_male.loc[normalized_sex.eq("F").fillna(False)] = 0.0
    derived_qrs = pd.to_numeric(
        frame.get(
            "qrs_duration_ms",
            pd.Series(np.nan, index=frame.index),
        ),
        errors="coerce",
    )
    machine_qrs = pd.to_numeric(
        frame.get(
            "machine_qrs_duration_ms",
            pd.Series(np.nan, index=frame.index),
        ),
        errors="coerce",
    )
    ecg_time = pd.to_datetime(
        (
            frame["ecg_time"]
            if "ecg_time" in frame
            else pd.Series(pd.NaT, index=frame.index)
        ),
        errors="coerce",
    )
    # Heart rate is both a strong non-specific outcome predictor and a
    # mechanical driver of most repolarization/loop-geometry markers. Leaving
    # it out made every hit ambiguous between "marker signal" and "this
    # patient was tachycardic".
    derived_rate = pd.to_numeric(
        frame.get("heart_rate_bpm", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    machine_rate = pd.to_numeric(
        frame.get(
            "machine_heart_rate_bpm",
            pd.Series(np.nan, index=frame.index),
        ),
        errors="coerce",
    )
    heart_rate = derived_rate.combine_first(machine_rate)
    output = pd.DataFrame(
        {
            "subject_id": frame["subject_id"],
            "label": frame["label"],
            "age": pd.to_numeric(frame["age"], errors="coerce"),
            "sex_male": sex_male,
            "qrs_duration": derived_qrs.combine_first(machine_qrs),
            "heart_rate": heart_rate,
            # Real calendar era, not the de-identification-shifted year; see
            # anchor_year_group_start().
            "acquisition_hour_sin": np.sin(
                2 * np.pi * _acquisition_hour(ecg_time) / 24.0
            ),
            "acquisition_hour_cos": np.cos(
                2 * np.pi * _acquisition_hour(ecg_time) / 24.0
            ),
            "anchor_era_start_year": anchor_year_group_start(
                frame.get(
                    "anchor_year_group",
                    pd.Series(pd.NA, index=frame.index),
                )
            ).to_numpy(),
            "care_setting_ed": frame.get(
                "ed_stay_id",
                pd.Series(pd.NA, index=frame.index),
            ).notna().astype(float),
            "vcg_marker": int(vcg_direction)
            * pd.to_numeric(frame[vcg_feature_id], errors="coerce"),
            "bmi": pd.to_numeric(
                (
                    frame["bmi_before_ecg"]
                    if "bmi_before_ecg" in frame
                    else pd.Series(np.nan, index=frame.index)
                ),
                errors="coerce",
            ),
        }
    )
    if classical_feature_id is not None:
        column = f"{classical_feature_id}_positive"
        output["classical_positive"] = (
            frame[column].astype("boolean").astype(float)
            if column in frame
            else np.nan
        )
    states = frame.get("report_reference_state")
    if states is not None:
        normalized_states = states.astype("string").str.casefold()
        report_positive = pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )
        report_positive.loc[
            normalized_states.eq("positive").fillna(False)
        ] = 1.0
        report_positive.loc[
            normalized_states.isin(("negative", "not_mentioned")).fillna(
                False
            )
        ] = 0.0
        output["report_positive"] = report_positive
    return output



def _acquisition_hour(ecg_time: pd.Series) -> np.ndarray:
    """Clock hour of acquisition, which MIMIC's date shifting preserves."""

    stamps = pd.to_datetime(ecg_time, errors="coerce")
    return (
        stamps.dt.hour.astype(float) + stamps.dt.minute.astype(float) / 60.0
    ).to_numpy(dtype=float)


def _model_definitions(
    *,
    covariates: tuple[str, ...],
    classical_available: bool,
    report_available: bool,
) -> tuple[AdjustedModelDefinition, ...]:
    covariate_standardization = tuple(
        name not in {"sex_male", "care_setting_ed"}
        for name in covariates
    )
    definitions = [
        AdjustedModelDefinition(
            "covariates",
            covariates,
            covariate_standardization,
        ),
        AdjustedModelDefinition(
            "vcg_adjusted",
            covariates + ("vcg_marker",),
            covariate_standardization + (True,),
            baseline_model_id="covariates",
        ),
    ]
    if classical_available:
        definitions.extend(
            (
                AdjustedModelDefinition(
                    "classical_adjusted",
                    covariates + ("classical_positive",),
                    covariate_standardization + (False,),
                    baseline_model_id="covariates",
                ),
                AdjustedModelDefinition(
                    "classical_plus_vcg",
                    covariates + ("classical_positive", "vcg_marker"),
                    covariate_standardization + (False, True),
                    baseline_model_id="classical_adjusted",
                ),
            )
        )
    if report_available:
        definitions.extend(
            (
                AdjustedModelDefinition(
                    "report_adjusted",
                    covariates + ("report_positive",),
                    covariate_standardization + (False,),
                    baseline_model_id="covariates",
                ),
                AdjustedModelDefinition(
                    "report_plus_vcg",
                    covariates + ("report_positive", "vcg_marker"),
                    covariate_standardization + (False, True),
                    baseline_model_id="report_adjusted",
                ),
            )
        )
    return tuple(definitions)


def _uses_bmi_covariate(endpoint_id: str) -> bool:
    return endpoint_id in {
        "concentric_remodeling_rwt",
        "la_dilatation_linear",
        "ra_dilatation_linear",
        "aortic_root_dilatation",
        "echo_lvh_strict",
        "echo_lvh_sensitive",
        "lv_wall_thickening",
        "lv_dilatation_proxy",
        "concentric_remodeling_proxy",
    }
