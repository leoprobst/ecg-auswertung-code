"""Patient-separated, paired single-lead models; not monitoring recommendations."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, PerfectSeparationError

from ..core.data_loader import LEAD_NAMES_12
from .lead_measurements import LEAD_MEASUREMENT_SPEC, measurement_column
from .lead_confirmation import confirm_component
from .design_diagnostics import design_rank_condition
from .patient_identity import known_patient_mask
from .statistics import _prepare_weighted_bootstrap_score_order, _weighted_bootstrap_metric_batch

LEAD_COMPARISON_VERSION = "1.3.0"


@dataclass
class LeadComparison:
    results: pd.DataFrame
    predictions: pd.DataFrame
    models: dict


def _fit(train: np.ndarray, labels: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean, scale = train.mean(axis=0), train.std(axis=0)
    active = scale > 1e-10
    design = np.column_stack([np.ones(len(train)), (train[:, active]-mean[active])/scale[active]])
    if min(np.sum(labels == 1), np.sum(labels == 0)) < 10*design.shape[1]:
        raise ValueError("insufficient_events_per_parameter")
    rank, condition = design_rank_condition(design)
    if rank != design.shape[1] or condition > 1e8:
        raise ValueError("collinear_design")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerfectSeparationWarning)
        warnings.simplefilter("error", RuntimeWarning)
        fitted = sm.GLM(labels, design, family=sm.families.Binomial()).fit(cov_type="HC3", maxiter=100)
    if not fitted.converged or not np.isfinite(fitted.params).all() or np.max(np.abs(fitted.params)) > 30:
        raise ValueError("unstable_fit")
    check = np.column_stack([np.ones(len(validation)), (validation[:, active]-mean[active])/scale[active]])
    return fitted.predict(design), fitted.predict(check), {
        "mean": mean.tolist(), "scale": scale.tolist(), "active_columns": np.flatnonzero(active).tolist(),
        "coefficients": fitted.params.tolist(), "training_only_preprocessing": True}


def paired_auc_samples(labels: np.ndarray, scores: np.ndarray, *, iterations: int, seed: int) -> np.ndarray:
    """One common patient draw per replicate; sort each model only once."""
    if iterations < 0:
        raise ValueError("Bootstrap iterations must be nonnegative")
    if iterations == 0:
        # The 10k pipeline-QC pilot deliberately disables uncertainty
        # resampling.  Sorting every score vector in every endpoint/family in
        # that mode is pure overhead and previously made a nominally
        # no-bootstrap lead audit look CPU-bound.
        return np.empty((0, scores.shape[1]), dtype=float)
    orderings = [_prepare_weighted_bootstrap_score_order(labels, scores[:, i]) for i in range(scores.shape[1])]
    output = np.full((iterations, scores.shape[1]), np.nan)
    rng = np.random.default_rng(seed)
    batch_size = max(1, min(32, 250_000 // max(1, len(labels))))
    probabilities = np.full(len(labels), 1/len(labels))
    prevalence = float(labels.mean())
    for start in range(0, iterations, batch_size):
        count = min(batch_size, iterations-start)
        weights = rng.multinomial(len(labels), probabilities, size=count)
        for column, (order, groups, positive) in enumerate(orderings):
            output[start:start+count, column] = _weighted_bootstrap_metric_batch(labels, scores[:, column], weights,
                thresholds={}, prevalence=prevalence, score_order=order,
                score_group_starts=groups, sorted_positive=positive, auc_only=True)["roc_auc"]
    return output


def compare_leads(frame: pd.DataFrame, *, endpoint_id: str, cohort_id: str, reference_kind: str,
                  family: str, covariates: tuple[str, ...], iterations: int = 200, seed: int = 20260908,
                  confirm_association: bool = True) -> LeadComparison:
    """Input must already be a reference-blind index cohort for one endpoint.

    `reference_eligible` is an actual boolean. Only train/validation may enter.
    Covariates must be prespecified numeric encodings (including setting/sex),
    never inferred from outcomes. Missingness is not filled using validation.
    Family-specific complete cases avoid making P observability a prerequisite
    for a QRS or T comparison. All twelve leads share the same fitted cohort.
    """
    if family not in {"qrs", "p", "st", "t"} or not covariates:
        raise ValueError("A prespecified family and covariates are required")
    required = {"subject_id", "split", "label_state", "reference_eligible", "extraction_status", *covariates}
    if not required <= set(frame):
        raise ValueError("Missing index-cohort or adjustment columns")
    if not frame.split.isin(["train", "validation"]).all():
        raise PermissionError("Only development train/validation are allowed")
    if not known_patient_mask(frame.subject_id).all() or frame.subject_id.duplicated().any():
        raise ValueError("Exactly one known patient per endpoint, separated across splits, is required")
    specs = [spec for spec in LEAD_MEASUREMENT_SPEC if spec[0] == family]
    columns = {lead: [measurement_column(lead, *spec) for spec in specs] for lead in LEAD_NAMES_12}
    all_columns = [name for names in columns.values() for name in names]
    if set(covariates) & set(all_columns):
        raise ValueError("A tested lead measurement cannot be its own adjustment covariate")
    values = frame.reindex(columns=[*covariates, *all_columns]).apply(pd.to_numeric, errors="coerce")
    eligible = frame.reference_eligible.map(lambda x: isinstance(x, (bool, np.bool_)) and bool(x)) & frame.label_state.isin(["positive", "negative"])
    qc = frame.extraction_status.eq("ok")
    covariates_complete = np.isfinite(values[list(covariates)].to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    leads_complete = np.isfinite(values[all_columns].to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    complete = eligible & qc & covariates_complete & leads_complete
    # Sequential, disjoint exclusions: no patient is counted twice.
    funnel = {
        "n_excluded_reference": int((~eligible).sum()),
        "n_excluded_extraction_after_reference": int((eligible & ~qc).sum()),
        "n_excluded_covariates_after_extraction": int((eligible & qc & ~covariates_complete).sum()),
        "n_excluded_leads_after_covariates": int((eligible & qc & covariates_complete & ~leads_complete).sum()),
        "n_common_complete": int(complete.sum()),
        "missingness_counting_design": "sequential_reference_extraction_covariates_family_leads",
    }
    train, valid = complete & frame.split.eq("train"), complete & frame.split.eq("validation")
    labels = frame.label_state.eq("positive").astype(int)
    rows = [{"endpoint_id": endpoint_id, "cohort_id": cohort_id, "reference_kind": reference_kind,
        "family": family, "lead": lead, "status": "non_estimable", "abstention_reason": "",
        **funnel, "n_index": len(frame), "n_reference_eligible": int(eligible.sum()),
        "n_lead_measured": int((qc & np.isfinite(values[names].to_numpy(dtype=float, na_value=np.nan)).all(axis=1)).sum()),
        "n_train_common": int(train.sum()), "n_validation_common": int(valid.sum()),
        "n_validation_positive": int(labels[valid].sum()), "n_validation_negative": int((1-labels[valid]).sum()),
        "analysis_design": "paired_family_complete_case_train_selected_anchor",
        "evidence_stage": "exploratory", "multiplicity_status": "intervals_not_multiplicity_adjusted",
        "confirmation_status": "non_estimable" if confirm_association else "descriptive_only",
        "confirmation_p_value": np.nan,
        "monitoring_recommendation_allowed": False, "clinical_claim_allowed": False,
        "lead_comparison_version": LEAD_COMPARISON_VERSION} for lead, names in columns.items()]
    if min(labels[valid].sum(), (1-labels[valid]).sum()) < 20 or train.sum() == 0:
        for row in rows: row["abstention_reason"] = "insufficient_common_patients_or_events"
        return LeadComparison(pd.DataFrame(rows), pd.DataFrame(), {})
    fitted_models, train_scores, valid_scores, confirmations = {}, {}, {}, {}
    # Materialise each patient split once, not once for every lead/model.
    # Keep column order identical to the prespecified model definitions.
    # Explicit float: a nullable (extension-dtype) column in the frame would
    # otherwise make this an object matrix, on which ``std`` cannot take a
    # square root even though every retained row is complete.
    train_matrix = values.loc[train].to_numpy(dtype=float, na_value=np.nan)
    validation_matrix = values.loc[valid].to_numpy(dtype=float, na_value=np.nan)
    column_indices = {name: index for index, name in enumerate(values.columns)}
    train_labels = labels[train].to_numpy()
    validation_labels = labels[valid].to_numpy()
    failed = (ValueError, PerfectSeparationWarning, PerfectSeparationError, RuntimeWarning, np.linalg.LinAlgError)
    for name, names in {"clinical_baseline": list(covariates), **{lead: [*covariates, *cols] for lead, cols in columns.items()}}.items():
        try:
            indices = [column_indices[column] for column in names]
            train_values, validation_values = train_matrix[:, indices], validation_matrix[:, indices]
            tr, va, model = _fit(train_values, train_labels, validation_values)
            fitted_models[name] = {**model, "columns": names}
            train_scores[name], valid_scores[name] = tr, va
            if name != "clinical_baseline" and confirm_association:
                confirmations[name] = confirm_component(model, train_values, validation_values,
                    validation_labels, len(covariates))
        except failed as exc:
            fitted_models[name] = {"status": "non_estimable", "reason": str(exc) or type(exc).__name__}
    if "clinical_baseline" not in valid_scores:
        for row in rows: row["abstention_reason"] = "clinical_baseline_non_estimable"
        return LeadComparison(pd.DataFrame(rows), pd.DataFrame(), fitted_models)
    usable = [lead for lead in LEAD_NAMES_12 if lead in valid_scores]
    if not usable:
        for row in rows: row["abstention_reason"] = fitted_models[str(row["lead"])]["reason"]
        return LeadComparison(pd.DataFrame(rows), pd.DataFrame(), fitted_models)
    train_auc = {lead: float(roc_auc_score(train_labels, train_scores[lead])) for lead in usable}
    anchor = sorted(usable, key=lambda lead: (-train_auc[lead], lead))[0]
    names = ["clinical_baseline", *usable]
    scores = np.column_stack([valid_scores[name] for name in names])
    bootstrap = paired_auc_samples(validation_labels, scores, iterations=iterations, seed=seed)
    observed = np.array([roc_auc_score(validation_labels, scores[:, i]) for i in range(len(names))])
    anchor_index = names.index(anchor)
    for row in rows:
        lead = str(row["lead"])
        if lead not in usable:
            row["abstention_reason"] = fitted_models[lead]["reason"]
            continue
        index = names.index(lead)
        row.update(confirmations.get(lead, {}))
        row.update(status="estimable", train_auc=train_auc[lead], validation_auc=observed[index],
            delta_auc_clinical_baseline=observed[index]-observed[0], anchor_selected_in_train=anchor,
            delta_auc_train_anchor=observed[index]-observed[anchor_index], bootstrap_requested=iterations,
            bootstrap_successful=int(np.isfinite(bootstrap[:, index]).sum()))
        for metric, draws in (("validation_auc", bootstrap[:, index]),
                              ("delta_auc_clinical_baseline", bootstrap[:, index]-bootstrap[:, 0]),
                              ("delta_auc_train_anchor", bootstrap[:, index]-bootstrap[:, anchor_index])):
            draws = draws[np.isfinite(draws)]
            lo, hi = np.quantile(draws, [.025, .975]) if len(draws) >= 100 else (np.nan, np.nan)
            row[metric+"_ci_low"], row[metric+"_ci_high"] = lo, hi
    predictions = frame.loc[valid, ["subject_id", "split"]].copy()
    predictions["label"] = labels[valid]
    for name in names: predictions[name] = valid_scores[name]
    return LeadComparison(pd.DataFrame(rows), predictions, fitted_models)
