"""Development-only paired comparisons, not a clinical validation claim."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

from .lead_comparison import _fit, paired_auc_samples
from .patient_identity import known_patient_mask
from .lead_confirmation import confirm_component

VOLTAGE_MASS_MODEL_VERSION = "1.2.0"


def compare_voltage_mass_models(frame: pd.DataFrame, *, voltage: str,
                                covariates: tuple[str, ...], iterations: int = 200,
                                seed: int = 20260911,
                                confirm_association: bool = True) -> tuple[pd.DataFrame, dict]:
    """All models use the same patients and training-only preprocessing.

    Caller supplies a reference-blind index cohort and explicit reference
    eligibility; absent clinical labels are never assumed negative. A quadratic
    components baseline tests whether a ratio helps beyond a simple linear
    parameterisation. It is not an unrestricted information-theoretic test.
    """
    required = {"subject_id", "split", "label_state", "reference_eligible",
                "extraction_status", "lv_mass_g", voltage, *covariates}
    if not covariates or not required <= set(frame):
        raise ValueError("Missing prespecified adjustment or reference columns")
    if iterations < 0:
        raise ValueError("Bootstrap iterations must be nonnegative")
    if set(covariates) & {voltage, "lv_mass_g"}:
        raise ValueError("Components cannot also be adjustment covariates")
    if not frame.split.isin(["train", "validation"]).all():
        raise PermissionError("Only development train/validation allowed")
    if not known_patient_mask(frame.subject_id).all() or frame.subject_id.duplicated().any():
        raise ValueError("One known patient per endpoint across splits required")
    numeric = frame[[*covariates, voltage, "lv_mass_g"]].apply(pd.to_numeric, errors="coerce")
    reference = frame.reference_eligible.map(lambda x: isinstance(x, (bool, np.bool_)) and bool(x))
    reference &= frame.label_state.isin(["positive", "negative"])
    quality = frame.extraction_status.eq("ok")
    adjustment = np.isfinite(numeric[list(covariates)].to_numpy(dtype=float)).all(axis=1)
    anatomy = np.isfinite(numeric.lv_mass_g) & numeric.lv_mass_g.gt(0)
    voltage_available = np.isfinite(numeric[voltage])
    after_quality = reference & quality
    after_adjustment = after_quality & adjustment
    after_anatomy = after_adjustment & anatomy
    usable = after_anatomy & voltage_available
    # Ordered exclusions are disjoint. A patient missing both anatomy and
    # voltage is counted once at the first failed gate, not twice.
    funnel = dict(
        n_excluded_reference=int((~reference).sum()),
        n_excluded_quality_after_reference=int((reference & ~quality).sum()),
        n_excluded_covariates_after_quality=int((after_quality & ~adjustment).sum()),
        n_excluded_anatomy_after_covariates=int((after_adjustment & ~anatomy).sum()),
        n_excluded_voltage_after_anatomy=int((after_anatomy & ~voltage_available).sum()),
        n_after_quality=int(after_quality.sum()),
        n_after_covariates=int(after_adjustment.sum()),
        n_after_anatomy=int(after_anatomy.sum()),
        missingness_counting_design="sequential_reference_quality_covariates_anatomy_voltage",
    )
    scoped = numeric.loc[usable]
    y = frame.loc[usable, "label_state"].eq("positive").to_numpy(dtype=int)
    train = frame.loc[usable, "split"].eq("train").to_numpy()
    valid = ~train
    names = ("clinical", "voltage", "mass", "ratio", "joint_linear",
             "joint_quadratic", "joint_quadratic_plus_ratio")
    common = dict(**funnel, n_index=len(frame), n_reference=int(reference.sum()), n_complete=int(usable.sum()),
                  n_train=int(train.sum()), n_validation=int(valid.sum()),
                  n_validation_positive=int(y[valid].sum()), evidence_stage="exploratory",
                  clinical_claim_allowed=False, scientific_tier="post_outcome_discovery",
                  model_version=VOLTAGE_MASS_MODEL_VERSION,
                  comparison_scope="paired_complete_cases_train_fitted_validation_scored",
                  interval_multiplicity="unadjusted_not_confirmatory")
    rows = [dict(common, model=name, status="non_estimable", abstention_reason="") for name in names]
    for row in rows:
        planned = row["model"] == "joint_quadratic_plus_ratio"
        row["confirmation_status"] = ("non_estimable" if confirm_association else "descriptive_only") if planned else "not_planned"
        row["confirmation_p_value"] = np.nan
        row["confirmation_reason"] = "prediction_model_not_estimable" if planned else "not_planned"
    if train.sum() == 0 or min(np.sum(y[valid] == 0), np.sum(y[valid] == 1)) < 20:
        for row in rows:
            row["abstention_reason"] = "insufficient_common_patients_or_events"
        return pd.DataFrame(rows), {}
    clinical = scoped[list(covariates)].to_numpy(dtype=float)
    components = scoped[[voltage, "lv_mass_g"]].to_numpy(dtype=float)
    center, scale = components[train].mean(axis=0), components[train].std(axis=0)
    z = (components - center) / np.where(scale > 1e-10, scale, 1.)
    ratio = (components[:, 0] / components[:, 1])[:, None]
    quadratic = np.column_stack([z, z * z, z[:, 0] * z[:, 1]])
    blocks = [np.empty((len(y), 0)), components[:, :1], components[:, 1:], ratio,
              z, quadratic, np.column_stack([quadratic, ratio])]
    scores, models = {}, {}
    failures = (ValueError, RuntimeWarning, PerfectSeparationError,
                PerfectSeparationWarning, np.linalg.LinAlgError)
    for row, block in zip(rows, blocks, strict=True):
        design = np.column_stack([clinical, block])
        try:
            _, prediction, fitted = _fit(design[train], y[train], design[valid])
            if not np.isfinite(prediction).all():
                raise ValueError("nonfinite_predictions")
            scores[row["model"]] = prediction
            models[row["model"]] = dict(fitted, component_center=center.tolist(),
                component_scale=scale.tolist(), voltage=voltage, covariates=list(covariates))
            row.update(status="estimable", validation_auc=float(roc_auc_score(y[valid], prediction)),
                       validation_brier=float(brier_score_loss(y[valid], prediction)))
            if row["model"] == "joint_quadratic_plus_ratio" and confirm_association:
                row.update(confirm_component(fitted, design[train], design[valid], y[valid],
                                             len(covariates) + quadratic.shape[1]))
                row["confirmation_design"] = "validation_ratio_association_adjusted_for_clinical_and_quadratic_components"
                row["confirmation_null"] = "zero_conditional_ratio_association_not_zero_auc_gain"
                row["confirmation_effect_scale"] = "per_training_sd_of_training_oriented_ratio_component"
        except failures as exc:
            row["abstention_reason"] = str(exc) or type(exc).__name__
    if scores:
        fitted_names = list(scores)
        draws = paired_auc_samples(y[valid], np.column_stack(list(scores.values())),
                                   iterations=iterations, seed=seed)
        by_name = {row["model"]: row for row in rows}
        for i, name in enumerate(fitted_names):
            row = by_name[name]
            comparisons = [("validation_auc", draws[:, i])]
            for baseline in ("clinical", "joint_linear", "joint_quadratic"):
                if baseline in scores:
                    j = fitted_names.index(baseline)
                    field = "delta_auc_vs_" + baseline
                    row[field] = row["validation_auc"] - by_name[baseline]["validation_auc"]
                    comparisons.append((field, draws[:, i] - draws[:, j]))
            for field, samples in comparisons:
                finite = samples[np.isfinite(samples)]
                low, high = np.quantile(finite, [.025, .975]) if len(finite) >= 100 else (np.nan, np.nan)
                row[field + "_ci_low"], row[field + "_ci_high"] = low, high
            row["bootstrap_successful"] = int(np.isfinite(draws[:, i]).sum())
    return pd.DataFrame(rows), models
