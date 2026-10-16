from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .adjusted_models import AdjustedModelDefinition, fit_logistic_model
from .frame_checkpoints import (
    checkpointed_frame_chunk_path,
    materialize_frame_chunks,
)
from .statistics import apply_fdr_by_family, evaluate_marker, marker_assessability


PHENOME_ADJUSTMENT_VERSION = "1.5.0"
PHENOME_ADJUSTMENT_CHUNK_SIZE = 16
# MIMIC-IV is an emergency-department and intensive-care population. Illness
# acuity there correlates with nearly every ECG abnormality and nearly every
# phenotype simultaneously, so it -- not age and sex -- is the dominant
# confounder of the broad scan. Heart rate is added for the same reason on the
# marker side: it drives most rate-dependent morphology markers mechanically.
#
# The set is deliberately kept to five: every added parameter raises the
# events-per-parameter floor by 20 minority-class events, and a phenotype that
# cannot support the prespecified set is reported as not evaluable rather than
# silently adjusted with a smaller model. Length of stay and admission type are
# not added because they are largely collinear with the ICU indicator; the
# acquisition era is not adjusted here because the index is a single
# cross-sectional ECG per patient.
PHENOME_ADJUSTMENT_COVARIATES = (
    "age",
    "sex_male",
    "heart_rate",
    "signal_quality_score",
    "index_icu_stay",
)
_BINARY_ADJUSTMENT_COVARIATES = frozenset({"sex_male", "index_icu_stay"})
# Audit X5, prespecified sensitivity model: age and heart rate as restricted
# cubic splines (four knots at the 5th/35th/65th/95th percentiles, Harrell)
# and emergency versus planned admission. It adds five parameters, so it runs
# only where the events allow and never replaces the primary five-covariate
# model; its role is to show whether a confirmation survives a flexible
# adjustment.
SPLINE_KNOT_QUANTILES = (0.05, 0.35, 0.65, 0.95)
_EMERGENCY_ADMISSION_PATTERN = r"EMER|URGENT|^EU "
PHENOME_ADJUSTMENT_MIN_EFFECT_RETENTION = 0.50


@dataclass(frozen=True)
class PhenomeAdjustmentPolicy:
    minimum_events_per_parameter: int = 20
    fdr_alpha: float = 0.05
    minimum_effect_retention: float = (
        PHENOME_ADJUSTMENT_MIN_EFFECT_RETENTION
    )

    def __post_init__(self) -> None:
        if self.minimum_events_per_parameter < 1:
            raise ValueError(
                "minimum_events_per_parameter must be positive"
            )
        if not 0.0 < self.fdr_alpha < 1.0:
            raise ValueError("fdr_alpha must be between zero and one")
        if not 0.0 <= self.minimum_effect_retention <= 1.0:
            raise ValueError(
                "minimum_effect_retention must be between zero and one"
            )


def confirm_replicated_phenome_hits(
    replicated: pd.DataFrame,
    features: pd.DataFrame,
    current_phenotypes: pd.DataFrame,
    history_body_systems: pd.DataFrame,
    *,
    policy: PhenomeAdjustmentPolicy = PhenomeAdjustmentPolicy(),
    checkpoint_root: Optional[Path] = None,
    checkpoint_input_hash: str = "",
    progress: Optional[Callable[[str], None]] = None,
    finalize: bool = True,
) -> pd.DataFrame:
    """Run a pre-specified, validation-only confounding confirmation.

    The complete candidate family must be selected in Train before this
    function is called. The marker direction is inherited from Train. This is
    an association confirmation, not a fitted clinical probability model and
    not a reason to access the locked test.
    """
    required = {
        "phenotype_id",
        "body_system",
        "panel_id",
        "feature_id",
        "train_direction",
    }
    missing = sorted(required - set(replicated.columns))
    if missing:
        raise ValueError(
            f"Replicated phenome hits are missing columns: {missing}"
        )
    if replicated.empty:
        return _empty_adjustment_frame()
    if features.get("split", pd.Series(dtype=str)).eq("locked_test").any():
        raise RuntimeError(
            "Locked-test rows are forbidden in phenome adjustment"
        )

    validation = features.loc[
        features["split"].astype(str).eq("validation")
    ].reset_index(drop=True)
    lookup = _validation_row_lookup(
        validation,
        current_phenotypes,
        history_body_systems,
        replicated,
    )
    records: list[dict] = []
    paths: list[Path] = []
    for chunk_index, start in enumerate(
        range(0, len(replicated), PHENOME_ADJUSTMENT_CHUNK_SIZE)
    ):
        chunk = replicated.iloc[
            start : start + PHENOME_ADJUSTMENT_CHUNK_SIZE
        ].copy()
        item_keys = tuple(
            (
                str(row.phenotype_id),
                str(row.feature_id),
                str(row.panel_id),
            )
            for row in chunk.itertuples(index=False)
        )

        def compute(selected: pd.DataFrame = chunk) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    _adjust_one_hit(row, validation, lookup, policy)
                    for row in selected.itertuples(index=False)
                ]
            )

        if checkpoint_root is None:
            frame = compute()
            resumed = False
            records.extend(frame.to_dict(orient="records"))
        else:
            path, resumed = checkpointed_frame_chunk_path(
                checkpoint_root,
                namespace="pathology_adjustment",
                chunk_index=chunk_index,
                input_hash=checkpoint_input_hash,
                item_keys=item_keys,
                compute=compute,
            )
            paths.append(path)
        if progress is not None:
            progress(
                "PheWAS Adjustierung "
                f"{min(start + len(chunk), len(replicated)):,}/"
                f"{len(replicated):,}: "
                f"{'Checkpoint fortgesetzt' if resumed else 'Checkpoint geschrieben'}"
            )

    output = (
        materialize_frame_chunks(paths)
        if checkpoint_root is not None
        else pd.DataFrame(records)
    )
    if not finalize:
        return output
    return finalize_phenome_adjustment_output(output, policy)


def finalize_phenome_adjustment_output(
    output: pd.DataFrame,
    policy: PhenomeAdjustmentPolicy,
) -> pd.DataFrame:
    """Apply the prespecified multiplicity and robustness gates once."""
    if output.empty:
        return output.copy()
    output = output.copy()
    corrected = apply_fdr_by_family(
        output,
        family_columns=("panel_id",),
        p_column="adjusted_p_value",
        alpha_by_panel={
            str(panel): policy.fdr_alpha
            for panel in output["panel_id"].dropna().unique()
        },
    )
    output["adjusted_q_value"] = corrected["q_value"]
    output["adjusted_fdr_significant"] = corrected[
        "fdr_significant"
    ].fillna(False)
    output["adjusted_global_q_value"] = corrected["global_q_value"]
    output["adjusted_hierarchical_q_value"] = corrected[
        "hierarchical_q_value"
    ]
    output["adjusted_global_fdr_significant"] = corrected[
        "global_fdr_significant"
    ].fillna(False)
    output["adjusted_hierarchical_fdr_significant"] = corrected[
        "hierarchical_fdr_significant"
    ].fillna(False)
    output["adjustment_robust"] = (
        output["adjustment_status"].eq("evaluable")
        & output["adjusted_fdr_significant"]
        & output["adjusted_global_fdr_significant"]
        & output["adjusted_hierarchical_fdr_significant"]
        & output["adjusted_direction_consistent"].fillna(False)
        & output["marker_missingness_gate_passed"].fillna(False)
        & output["model_complete_case_gate_passed"].fillna(False)
        & pd.to_numeric(
            output["effect_retention_ratio"], errors="coerce"
        ).ge(policy.minimum_effect_retention)
        & pd.to_numeric(
            output["adjusted_or_ci_low"], errors="coerce"
        ).gt(1.0)
    )
    output["likely_confounded"] = (
        output["adjustment_status"].eq("evaluable")
        & ~output["adjustment_robust"]
    )
    output["confounding_control"] = (
        "validation_glm_age_sex_signal_quality_train_direction"
    )
    output["locked_test_scanned"] = False
    output["adjustment_version"] = PHENOME_ADJUSTMENT_VERSION
    return output


def build_phenome_discovery_status(
    train_hits: pd.DataFrame,
    validation_results: pd.DataFrame,
    adjusted_confirmation: pd.DataFrame,
) -> pd.DataFrame:
    keys = ("phenotype_id", "feature_id", "panel_id")
    validation_lookup = _record_lookup(validation_results, keys)
    adjustment_lookup = _record_lookup(adjusted_confirmation, keys)
    rows: list[dict] = []
    for train in train_hits.itertuples(index=False):
        key = tuple(str(getattr(train, item)) for item in keys)
        validation = validation_lookup.get(key)
        adjustment = adjustment_lookup.get(key)
        status_id, status_de, rank, rationale = _discovery_status(
            validation,
            adjustment,
        )
        validation_raw_auc = _record_number(
            validation,
            "validation_raw_auc",
            "raw_auc",
            "roc_auc",
        )
        validation_aligned_auc = _record_number(
            validation,
            "validation_aligned_auc",
            "roc_auc",
        )
        validation_effect = (
            2.0 * validation_raw_auc - 1.0
            if validation_raw_auc is not None
            else None
        )
        rows.append(
            {
                "phenotype_id": str(train.phenotype_id),
                "phenotype_name": str(
                    getattr(train, "phenotype_name", train.phenotype_id)
                ),
                "body_system": str(getattr(train, "body_system", "")),
                "feature_id": str(train.feature_id),
                "panel_id": str(train.panel_id),
                "train_q_value": _number(getattr(train, "q_value", None)),
                "train_aligned_auc": _number(
                    getattr(train, "aligned_auc", None)
                ),
                "validation_q_value": _record_number(
                    validation, "q_value"
                ),
                "validation_raw_auc": validation_raw_auc,
                "validation_aligned_auc": validation_aligned_auc,
                "validation_effect": validation_effect,
                "association_strength": (
                    abs(validation_effect)
                    if validation_effect is not None
                    else (
                        abs(2.0 * validation_aligned_auc - 1.0)
                        if validation_aligned_auc is not None
                        else None
                    )
                ),
                "n_positive": _record_number(
                    validation,
                    "n_positive",
                ),
                "n_negative": _record_number(
                    validation,
                    "n_negative",
                ),
                "inference_status": (
                    str(validation.get("inference_status", ""))
                    if validation
                    else "not_evaluated"
                ),
                "validation_stable": bool(
                    validation
                    and validation.get("validation_stable") is True
                ),
                "direction_consistent": bool(
                    validation and validation.get("direction_consistent") is True
                ),
                "unadjusted_replicated": bool(
                    validation and validation.get("replicated") is True
                ),
                "adjusted_q_value": _record_number(
                    adjustment, "adjusted_q_value"
                ),
                "adjusted_or_per_sd": _record_number(
                    adjustment, "adjusted_or_per_sd"
                ),
                "effect_retention_ratio": _record_number(
                    adjustment, "effect_retention_ratio"
                ),
                "adjustment_status": (
                    str(adjustment.get("adjustment_status"))
                    if adjustment
                    else "not_requested_not_replicated"
                ),
                "adjustment_robust": bool(
                    adjustment and adjustment.get("adjustment_robust") is True
                ),
                "status_id": status_id,
                "status_de": status_de,
                "evidence_rank": rank,
                "status_reason_de": rationale,
                "locked_test_scanned": False,
                "automatic_promotion": False,
                "adjustment_version": PHENOME_ADJUSTMENT_VERSION,
            }
        )
    return pd.DataFrame(rows)


def restricted_cubic_spline_basis(
    values: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    """Harrell's restricted cubic spline: linear term plus k-2 columns.

    Linear beyond the outer knots. Scaled by (t_k - t_1)^2 so the columns
    stay on the scale of the input.
    """
    x = np.asarray(values, dtype=float)
    t = np.asarray(knots, dtype=float)
    k = len(t)
    if k < 3:
        raise ValueError("A restricted cubic spline needs at least three knots")
    scale = (t[-1] - t[0]) ** 2
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Spline knots are not distinct")

    def plus(value):
        return np.where(value > 0, value, 0.0) ** 3

    columns = [x]
    for j in range(k - 2):
        term = (
            plus(x - t[j])
            - plus(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + plus(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        )
        columns.append(term / scale)
    return np.column_stack(columns)


def _emergency_admission(frame: pd.DataFrame) -> np.ndarray:
    admission = frame.get("admission_type", pd.Series(np.nan, index=frame.index))
    text = admission.astype("string").str.strip().str.upper()
    ed_only = pd.to_numeric(
        frame.get("index_ed_encounter", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).fillna(0).astype(bool)
    emergency = text.str.contains(_EMERGENCY_ADMISSION_PATTERN, regex=True, na=False)
    return np.where(
        text.notna() & emergency.to_numpy(bool),
        1.0,
        np.where(
            text.notna().to_numpy(bool),
            0.0,
            np.where(ed_only.to_numpy(bool), 1.0, np.nan),
        ),
    )


def _spline_sensitivity(
    design: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    phenotype_id: str,
    feature_id: str,
    policy: "PhenomeAdjustmentPolicy",
    unadjusted_coefficient: float,
) -> dict:
    """Refit with splines and admission urgency (audit X5); never raises."""
    output = {
        "spline_adjustment_status": "not_evaluable",
        "spline_adjustment_failure_reason": "",
        "spline_adjusted_or_per_sd": np.nan,
        "spline_adjusted_p_value": np.nan,
        "spline_effect_retention_ratio": np.nan,
        "spline_n_complete": 0,
    }
    sensitivity = design.copy()
    sensitivity["admission_emergency"] = _emergency_admission(frame.loc[design.index])
    sensitivity = sensitivity.loc[sensitivity.notna().all(axis=1)]
    output["spline_n_complete"] = int(len(sensitivity))
    if sensitivity["label"].nunique() < 2 or len(sensitivity) < 20:
        output["spline_adjustment_failure_reason"] = (
            "too_few_complete_rows_with_admission_type"
        )
        return output
    extra: list[str] = []
    try:
        for column in ("age", "heart_rate"):
            if column not in sensitivity:
                continue
            values = sensitivity[column].to_numpy(dtype=float)
            knots = np.quantile(values, SPLINE_KNOT_QUANTILES)
            basis = restricted_cubic_spline_basis(values, knots)
            for index in range(1, basis.shape[1]):
                name = f"{column}_spline_{index}"
                sensitivity[name] = basis[:, index]
                extra.append(name)
        if sensitivity["admission_emergency"].nunique() > 1:
            extra.append("admission_emergency")
        predictors = tuple(
            name for name in sensitivity.columns
            if name not in ("subject_id", "label", "admission_emergency")
            and not name.endswith(tuple(f"_spline_{i}" for i in range(1, 10)))
        ) + tuple(extra)
        output["spline_n_complete"] = int(len(sensitivity))
        _, coefficients = fit_logistic_model(
            sensitivity,
            endpoint_id=f"phewas:{phenotype_id}",
            definition=AdjustedModelDefinition(
                model_id=f"phewas:{phenotype_id}:{feature_id}:spline_sensitivity",
                predictor_names=predictors,
                standardize=tuple(
                    name not in _BINARY_ADJUSTMENT_COVARIATES
                    and name != "admission_emergency"
                    and "_spline_" not in name
                    for name in predictors
                ),
            ),
            minimum_events_per_parameter=policy.minimum_events_per_parameter,
        )
    except (
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
        PerfectSeparationError,
    ) as exc:
        output["spline_adjustment_failure_reason"] = f"{type(exc).__name__}: {exc}"
        return output
    marker = coefficients.loc[coefficients["predictor"].eq("aligned_marker")].iloc[0]
    coefficient = float(marker["coefficient"])
    output.update(
        {
            "spline_adjustment_status": "evaluable",
            "spline_adjusted_or_per_sd": float(marker["odds_ratio"]),
            "spline_adjusted_p_value": float(marker["p_value"]),
            "spline_effect_retention_ratio": (
                coefficient / unadjusted_coefficient
                if unadjusted_coefficient > 0
                else np.nan
            ),
        }
    )
    return output


def _adjust_one_hit(
    hit,
    validation: pd.DataFrame,
    lookup: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    policy: PhenomeAdjustmentPolicy,
) -> dict:
    phenotype_id = str(hit.phenotype_id)
    body_system = str(hit.body_system)
    feature_id = str(hit.feature_id)
    panel_id = str(hit.panel_id)
    base = {
        "phenotype_id": phenotype_id,
        "phenotype_name": str(
            getattr(hit, "phenotype_name", phenotype_id)
        ),
        "body_system": body_system,
        "feature_id": feature_id,
        "panel_id": panel_id,
        "train_direction": int(hit.train_direction),
        "candidate_selection_split": "train",
        "candidate_family_policy": "all_train_selected_hits",
        "fit_split": "validation",
        "inference_status": "descriptive_only",
        "adjustment_status": "not_evaluable",
        "adjustment_failure_reason": "",
        "n_complete": 0,
        "n_positive_complete": 0,
        "n_negative_complete": 0,
        "unadjusted_coefficient": np.nan,
        "unadjusted_or_per_sd": np.nan,
        "adjusted_coefficient": np.nan,
        "adjusted_standard_error": np.nan,
        "adjusted_p_value": np.nan,
        "adjusted_or_per_sd": np.nan,
        "adjusted_or_ci_low": np.nan,
        "adjusted_or_ci_high": np.nan,
        "adjusted_direction_consistent": False,
        "effect_retention_ratio": np.nan,
        "marker_test_yield": np.nan,
        "marker_positive_test_yield": np.nan,
        "marker_negative_test_yield": np.nan,
        "marker_missingness_auc_lower_bound": np.nan,
        "marker_missingness_auc_upper_bound": np.nan,
        "marker_missingness_gate_passed": False,
        "model_complete_case_test_yield": np.nan,
        "model_complete_case_test_yield_gap": np.nan,
        "model_complete_case_gate_passed": False,
        "missing_data_method": (
            "complete_case_plus_adversarial_marker_auc_bounds;mi_ipcw_not_run"
        ),
        "claim_scope": "complete_case_adjusted_association_only",
        "covariates": "|".join(PHENOME_ADJUSTMENT_COVARIATES),
        "covariates_applied": "",
        "covariates_unavailable": "|".join(PHENOME_ADJUSTMENT_COVARIATES),
        "covariate_set_complete": False,
        "fit_split": "validation",
        "marker_direction_source": "train",
        "model_role": "association_confirmation_not_probability_model",
    }
    if feature_id not in validation.columns:
        base["adjustment_failure_reason"] = "feature_unavailable"
        return base
    rows = lookup.get((phenotype_id, body_system))
    if rows is None:
        base["adjustment_failure_reason"] = "case_control_rows_unavailable"
        return base
    case_rows, control_rows = rows
    selected = np.concatenate((case_rows, control_rows))
    labels = np.concatenate(
        (
            np.ones(len(case_rows), dtype=np.int8),
            np.zeros(len(control_rows), dtype=np.int8),
        )
    )
    if len(selected) == 0:
        base["adjustment_failure_reason"] = "empty_case_control_sample"
        return base
    frame = validation.iloc[selected]
    sex = frame["sex"].astype("string").str.strip().str.upper()
    sex_male = np.where(
        sex.isin(("M", "MALE")),
        1.0,
        np.where(sex.isin(("F", "FEMALE")), 0.0, np.nan),
    )
    design_source = pd.DataFrame(
        {
            "subject_id": frame["subject_id"].to_numpy(),
            "label": labels,
            "aligned_marker": (
                int(hit.train_direction)
                * pd.to_numeric(
                    frame[feature_id], errors="coerce"
                ).to_numpy(dtype=float)
            ),
            "age": pd.to_numeric(
                frame["age"], errors="coerce"
            ).to_numpy(dtype=float),
            "sex_male": sex_male,
            "heart_rate": pd.to_numeric(
                frame.get(
                    "heart_rate_bpm",
                    pd.Series(np.nan, index=frame.index),
                ),
                errors="coerce",
            ).combine_first(
                pd.to_numeric(
                    frame.get(
                        "machine_heart_rate_bpm",
                        pd.Series(np.nan, index=frame.index),
                    ),
                    errors="coerce",
                )
            ).to_numpy(dtype=float),
            "signal_quality_score": pd.to_numeric(
                frame["signal_quality_score"], errors="coerce"
            ).to_numpy(dtype=float),
            # NaN (not False) when the column is absent: a fabricated
            # constant would look like "no ICU anywhere" and silently enter
            # the model as a zero-variance predictor.
            "index_icu_stay": pd.to_numeric(
                frame.get(
                    "index_icu_stay",
                    pd.Series(np.nan, index=frame.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=float),
        }
    )
    marker_assessment = evaluate_marker(
        design_source["aligned_marker"].to_numpy(dtype=float),
        labels,
        direction=1,
    )
    # A covariate that is entirely absent in this sample (an older artifact
    # that never carried the column, or a source that was not ingested) must
    # not silently void every hit, nor silently pretend it was adjusted for.
    # It is dropped from the model and named in the output instead.
    applied_covariates = tuple(
        name
        for name in PHENOME_ADJUSTMENT_COVARIATES
        if name in design_source
        and design_source[name].notna().any()
        and design_source[name].nunique(dropna=True) > 1
    )
    unavailable_covariates = tuple(
        name
        for name in PHENOME_ADJUSTMENT_COVARIATES
        if name not in applied_covariates
    )
    base["covariates_applied"] = "|".join(applied_covariates)
    base["covariates_unavailable"] = "|".join(unavailable_covariates)
    base["covariate_set_complete"] = not unavailable_covariates
    model_columns = ["subject_id", "label", "aligned_marker"]
    model_columns.extend(applied_covariates)
    design_source = design_source[model_columns]
    complete_mask = design_source.notna().all(axis=1).to_numpy(dtype=bool)
    complete_case_assessment = marker_assessability(
        np.where(complete_mask, 1.0, np.nan),
        labels,
    )
    base.update(
        {
            "marker_test_yield": marker_assessment["test_yield"],
            "marker_positive_test_yield": marker_assessment[
                "positive_test_yield"
            ],
            "marker_negative_test_yield": marker_assessment[
                "negative_test_yield"
            ],
            "marker_missingness_auc_lower_bound": marker_assessment[
                "missingness_auc_lower_bound"
            ],
            "marker_missingness_auc_upper_bound": marker_assessment[
                "missingness_auc_upper_bound"
            ],
            "marker_missingness_gate_passed": marker_assessment[
                "confirmatory_assessability_gate_passed"
            ],
            "model_complete_case_test_yield": complete_case_assessment[
                "test_yield"
            ],
            "model_complete_case_test_yield_gap": complete_case_assessment[
                "test_yield_gap"
            ],
            "model_complete_case_gate_passed": complete_case_assessment[
                "assessability_gate_passed"
            ],
        }
    )
    design = design_source.loc[complete_mask].copy()
    base["n_complete"] = len(design)
    base["n_positive_complete"] = int(design["label"].eq(1).sum())
    base["n_negative_complete"] = int(design["label"].eq(0).sum())
    if design["subject_id"].duplicated().any():
        base["adjustment_failure_reason"] = "duplicate_subject_rows"
        return base
    try:
        _, unadjusted_coefficients = fit_logistic_model(
            design,
            endpoint_id=f"phewas:{phenotype_id}",
            definition=AdjustedModelDefinition(
                model_id=f"phewas:{phenotype_id}:{feature_id}:unadjusted",
                predictor_names=("aligned_marker",),
                standardize=(True,),
            ),
            minimum_events_per_parameter=(
                policy.minimum_events_per_parameter
            ),
        )
        _, adjusted_coefficients = fit_logistic_model(
            design,
            endpoint_id=f"phewas:{phenotype_id}",
            definition=AdjustedModelDefinition(
                model_id=f"phewas:{phenotype_id}:{feature_id}:adjusted",
                predictor_names=("aligned_marker",) + applied_covariates,
                standardize=(True,)
                + tuple(
                    name not in _BINARY_ADJUSTMENT_COVARIATES
                    for name in applied_covariates
                ),
            ),
            minimum_events_per_parameter=(
                policy.minimum_events_per_parameter
            ),
        )
    except (
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
        PerfectSeparationError,
    ) as exc:
        base["adjustment_failure_reason"] = f"{type(exc).__name__}: {exc}"
        return base
    unadjusted = unadjusted_coefficients.loc[
        unadjusted_coefficients["predictor"].eq("aligned_marker")
    ].iloc[0]
    adjusted = adjusted_coefficients.loc[
        adjusted_coefficients["predictor"].eq("aligned_marker")
    ].iloc[0]
    unadjusted_coefficient = float(unadjusted["coefficient"])
    adjusted_coefficient = float(adjusted["coefficient"])
    retention = (
        adjusted_coefficient / unadjusted_coefficient
        if unadjusted_coefficient > 0
        else np.nan
    )
    base.update(
        {
            "inference_status": "formal_inference",
            "adjustment_status": "evaluable",
            "unadjusted_coefficient": unadjusted_coefficient,
            "unadjusted_or_per_sd": float(unadjusted["odds_ratio"]),
            "adjusted_coefficient": adjusted_coefficient,
            "adjusted_standard_error": float(
                adjusted["standard_error_clustered"]
            ),
            "adjusted_p_value": float(adjusted["p_value"]),
            "adjusted_or_per_sd": float(adjusted["odds_ratio"]),
            "adjusted_or_ci_low": float(adjusted["odds_ratio_ci_low"]),
            "adjusted_or_ci_high": float(adjusted["odds_ratio_ci_high"]),
            "adjusted_direction_consistent": adjusted_coefficient > 0,
            "effect_retention_ratio": float(retention),
        }
    )
    base.update(
        _spline_sensitivity(
            design,
            frame.reset_index(drop=True).loc[complete_mask],
            phenotype_id=phenotype_id,
            feature_id=feature_id,
            policy=policy,
            unadjusted_coefficient=unadjusted_coefficient,
        )
    )
    return base


def _validation_row_lookup(
    validation: pd.DataFrame,
    current: pd.DataFrame,
    history: pd.DataFrame,
    hits: pd.DataFrame,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    required_features = {
        "subject_id",
        "study_id",
        "is_control_index",
        "age",
        "sex",
        "signal_quality_score",
    }
    missing = sorted(required_features - set(validation.columns))
    if missing:
        raise ValueError(
            f"Phenome adjustment feature frame is missing columns: {missing}"
        )
    current_validation = current.loc[
        current["split"].astype(str).eq("validation")
    ]
    study_to_row = {
        (int(row.subject_id), int(row.study_id)): index
        for index, row in enumerate(
            validation[["subject_id", "study_id"]].itertuples(index=False)
        )
    }
    subjects = validation["subject_id"].astype(int).to_numpy()
    control_base = np.flatnonzero(
        validation["is_control_index"].fillna(False).to_numpy(dtype=bool)
    )
    # Retained as a provenance/API input. Diagnoses after the index admission
    # must never make a subject ineligible as a control.
    del history
    output: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    from .phenome import _normalized_sex, _related_phenotype_ids

    sex = _normalized_sex(validation)
    known = tuple(str(item) for item in current["phenotype_id"].astype(str).unique())
    subjects_by_phenotype = {
        str(key): frame["subject_id"].astype(int).to_numpy()
        for key, frame in current_validation[["phenotype_id", "subject_id"]].groupby(
            current_validation["phenotype_id"].astype(str)
        )
    }
    restriction_by_phenotype = {
        str(row.phenotype_id): str(getattr(row, "sex_restriction", "none") or "none")
        for row in hits.itertuples(index=False)
    }
    for phenotype_id, body_system in sorted(
        {
            (str(row.phenotype_id), str(row.body_system))
            for row in hits.itertuples(index=False)
        }
    ):
        cases = current_validation.loc[
            current_validation["phenotype_id"].astype(str).eq(phenotype_id),
            ["subject_id", "study_id"],
        ].drop_duplicates()
        case_rows = np.asarray(
            [
                study_to_row[key]
                for key in (
                    (int(row.subject_id), int(row.study_id))
                    for row in cases.itertuples(index=False)
                )
                if key in study_to_row
            ],
            dtype=int,
        )
        # Audits X3 and X4, as in the train screen and the replication.
        restriction = restriction_by_phenotype.get(phenotype_id, "none")
        if restriction in ("M", "F"):
            case_rows = case_rows[sex[case_rows] == restriction]
        excluded = [subjects[case_rows]]
        for related in _related_phenotype_ids(phenotype_id, known):
            if related in subjects_by_phenotype:
                excluded.append(subjects_by_phenotype[related])
        control_rows = control_base[
            ~np.isin(subjects[control_base], np.concatenate(excluded))
        ]
        if restriction in ("M", "F"):
            control_rows = control_rows[sex[control_rows] == restriction]
        output[(phenotype_id, body_system)] = (case_rows, control_rows)
    return output


def _discovery_status(
    validation: Optional[dict],
    adjustment: Optional[dict],
) -> tuple[str, str, int, str]:
    if validation is None:
        return (
            "new",
            "neu",
            1,
            "Im Training auffaellig; eine Validation-Auswertung fehlt.",
        )
    if validation.get("replicated") is True:
        if adjustment and adjustment.get("adjustment_robust") is True:
            return (
                "replicated",
                "repliziert",
                4,
                "In Validation stabil und nach Alter, Geschlecht und Signalqualitaet bestaetigt.",
            )
        if adjustment and adjustment.get("adjustment_status") == "evaluable":
            return (
                "likely_confounded",
                "wahrscheinlich confounded",
                2,
                "Unadjustiert repliziert, nach der vorgegebenen Adjustierung jedoch nicht robust.",
            )
        return (
            "promising",
            "vielversprechend",
            3,
            "Unadjustiert repliziert; die adjustierte Stufe war noch nicht belastbar auswertbar.",
        )
    same_direction = validation.get("direction_consistent") is True
    auc = _record_number(validation, "validation_aligned_auc", "roc_auc")
    inference = str(validation.get("inference_status", ""))
    if same_direction and auc is not None and auc >= 0.55 and inference != "formal_inference":
        return (
            "promising",
            "vielversprechend",
            2,
            "Gleiche Richtung in Validation, aber fuer eine formale Aussage unterpowert.",
        )
    return (
        "rejected",
        "verworfen",
        0,
        "Die vorab definierte Validation-Replikation wurde nicht erreicht.",
    )


def _record_lookup(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], dict]:
    if frame.empty or not set(keys).issubset(frame.columns):
        return {}
    return {
        tuple(str(row[key]) for key in keys): row
        for row in frame.to_dict(orient="records")
    }


def _record_number(
    record: Optional[Mapping],
    *names: str,
) -> Optional[float]:
    if record is None:
        return None
    for name in names:
        value = _number(record.get(name))
        if value is not None:
            return value
    return None


def _number(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _empty_adjustment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "phenotype_id",
            "phenotype_name",
            "body_system",
            "feature_id",
            "panel_id",
            "train_direction",
            "inference_status",
            "adjustment_status",
            "adjusted_p_value",
            "adjusted_q_value",
            "adjusted_global_q_value",
            "adjusted_hierarchical_q_value",
            "adjustment_robust",
            "marker_missingness_gate_passed",
            "model_complete_case_gate_passed",
            "likely_confounded",
            "locked_test_scanned",
            "adjustment_version",
        )
    )


__all__ = [
    "PHENOME_ADJUSTMENT_COVARIATES",
    "PHENOME_ADJUSTMENT_MIN_EFFECT_RETENTION",
    "PHENOME_ADJUSTMENT_VERSION",
    "PhenomeAdjustmentPolicy",
    "build_phenome_discovery_status",
    "confirm_replicated_phenome_hits",
]
