"""Prespecified exploratory follow-ups on immutable development measurements.

No new diagnostic thresholds, source pooling, or causal treatment claims.
These derived outputs are not automatically admitted to score discovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "1.0.0"


@dataclass(frozen=True)
class ExtensionPolicy:
    seed: int = 20260914
    folds: int = 5
    minimum_history: int = 5
    maximum_history_days: int = 365
    minimum_model_patients: int = 50
    window_seconds: int = 10
    minimum_windows: int = 6
    minimum_quality: float = 0.7


def research_contract():
    return {
        "version": VERSION,
        "policy": asdict(ExtensionPolicy()),
        "evidence_stage": "exploratory",
        "clinical_claim_allowed": False,
        "locked_test_allowed": False,
        "automatic_score_predictor_admission": False,
        "analyses": {
            "personal_deviation": "strictly_earlier_same_source_patient_split_rhythm_transform",
            "activation_conditioned_repolarization": "group_cross_fitted_training_residuals_frozen_validation",
            "measurement_sensitivity": "perturbation_range_not_confidence_interval",
            "clinical_specificity": "competing_verified_diagnoses_not_unlabelled_controls",
            "continuous_reference": "adjusted_continuous_association_not_threshold_optimization",
            "reference_timing": "signed_time_strata_and_verification_availability",
            "long_recording_windows": "nonoverlapping_windows_patient_level_summary",
            "cohort_value": "independent_patients_events_availability_not_automatic_model_improvement",
        },
    }


def development_frame(frame, required=()):
    keys = ["source", "subject_id", "study_id", "split"]
    missing = set(keys).union(required) - set(frame)
    if missing:
        raise ValueError("missing_columns:" + ",".join(sorted(missing)))
    if (
        frame[keys].isna().any().any()
        or not frame["split"].isin(["train", "validation"]).all()
    ):
        raise ValueError("unknown_identity_or_nondevelopment_split")
    if frame.duplicated(["source", "study_id"]).any():
        raise ValueError("duplicate_record_identity")
    if frame.groupby(["source", "subject_id"])["split"].nunique().gt(1).any():
        raise ValueError("patient_split_leakage")
    return frame.copy()


def personal_deviations(frame, markers, policy=ExtensionPolicy()):
    """Previous value and expanding median/MAD; no future or same-time history.

    Invalid earlier measurements remain counted, never relabelled as healthy.
    Per-marker histories use past available measurements; current missingness
    does not change the reference-blind selected clock. No cross-rhythm baseline.
    """
    if policy.minimum_history < 3 or policy.maximum_history_days <= 0:
        raise ValueError("invalid_history_policy")
    data = development_frame(
        frame, ["ecg_time", "rhythm_context", "transform", *markers]
    )
    data["ecg_time"] = pd.to_datetime(data.ecg_time, utc=True, errors="raise")
    if data.ecg_time.isna().any():
        raise ValueError("missing_ecg_time")
    data = data.sort_values(["ecg_time", "study_id"], kind="stable")
    # One deterministic observation per clock, chosen before measurement availability.
    data = data.drop_duplicates(["source", "subject_id", "ecg_time"], keep="first")
    rows = []
    for _, block in data.groupby(
        ["source", "subject_id", "split", "rhythm_context", "transform"], dropna=False
    ):
        values = (
            block[list(markers)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
        times = block.ecg_time.to_numpy()
        lower = 0
        for i, (_, record) in enumerate(block.iterrows()):
            while (
                lower < i
                and (times[i] - times[lower]) / np.timedelta64(1, "D")
                > policy.maximum_history_days
            ):
                lower += 1
            unknown = any(
                pd.isna(record[k]) or str(record[k]).lower() in {"unknown", ""}
                for k in ("rhythm_context", "transform")
            )
            for j, marker in enumerate(markers):
                earlier = values[lower:i, j]
                usable = earlier[np.isfinite(earlier)]
                current = values[i, j]
                status, reason = "available", ""
                if unknown:
                    status, reason = "non_estimable", "unknown_rhythm_or_transform"
                elif not np.isfinite(current) or not len(usable):
                    status, reason = (
                        "non_estimable",
                        "current_or_prior_measurement_missing",
                    )
                baseline = (
                    float(np.median(usable)) if len(usable) and not unknown else None
                )
                scale = (
                    float(1.4826 * np.median(np.abs(usable - baseline)))
                    if baseline is not None
                    else None
                )
                standardizable = (
                    status == "available"
                    and len(usable) >= policy.minimum_history
                    and scale is not None
                    and scale > 1e-12
                )
                previous = (
                    earlier[-1] if len(earlier) and np.isfinite(earlier[-1]) else None
                )
                rows.append(
                    {
                        "source": record.source,
                        "subject_id": record.subject_id,
                        "study_id": record.study_id,
                        "split": record.split,
                        "marker": marker,
                        "history_n": len(usable),
                        "history_total_n": len(earlier),
                        "delta_previous": float(current - previous)
                        if previous is not None and status == "available"
                        else None,
                        "delta_historical_median": float(current - baseline)
                        if status == "available"
                        else None,
                        "historical_mad_scale": scale,
                        "standardized_deviation": float((current - baseline) / scale)
                        if standardizable
                        else None,
                        "standardization_reason": ""
                        if standardizable
                        else "insufficient_history_or_zero_scale_or_unavailable",
                        "status": status,
                        "reason": reason,
                        "clinical_claim_allowed": False,
                    }
                )
    return pd.DataFrame(rows)


def activation_residuals(frame, predictors, targets, policy=ExtensionPolicy()):
    """Continuous residuals, source-specific models and patient-grouped OOF.

    Complete-case only; unknown inputs abstain. Predictors must be frozen ECG
    activation/rhythm/rate fields, never the target or later clinical evidence.
    The caller passes numeric rhythm indicators with explicit knownness.
    """
    if set(predictors) & set(targets) or not predictors or not targets:
        raise ValueError("invalid_predictor_target_overlap")
    if policy.folds < 2 or policy.minimum_model_patients < policy.folds:
        raise ValueError("invalid_model_policy")
    data = development_frame(frame, [*predictors, *targets])
    outputs, models = [], []
    for source, block in data.groupby("source", sort=True):
        x = (
            block[list(predictors)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        y = block[list(targets)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        complete = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        train = np.flatnonzero(complete & block.split.eq("train").to_numpy())
        validation = np.flatnonzero(complete & block.split.eq("validation").to_numpy())
        groups = block.subject_id.to_numpy()
        prediction = np.full_like(y, np.nan)
        reason = "insufficient_training_patients"
        if len(np.unique(groups[train])) >= policy.minimum_model_patients:
            reason = "missing_numeric_input_or_target"

            def fit_model(indices):
                counts = pd.Series(groups[indices]).value_counts()
                weights = 1.0 / pd.Series(groups[indices]).map(counts).to_numpy(float)
                weights /= weights.mean()
                fitted = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                fitted.fit(
                    x[indices],
                    y[indices],
                    standardscaler__sample_weight=weights,
                    ridge__sample_weight=weights,
                )
                return fitted

            for fit, test in GroupKFold(policy.folds).split(
                x[train], groups=groups[train]
            ):
                model = fit_model(train[fit])
                prediction[train[test]] = np.asarray(
                    model.predict(x[train[test]])
                ).reshape(-1, len(targets))
            model = fit_model(train)
            if len(validation):
                prediction[validation] = np.asarray(
                    model.predict(x[validation])
                ).reshape(-1, len(targets))
            models.append(
                {
                    "source": source,
                    "status": "fitted",
                    "predictors": list(predictors),
                    "targets": list(targets),
                    "training_patients": len(np.unique(groups[train])),
                    "scaler_mean": model[0].mean_.tolist(),
                    "scaler_scale": model[0].scale_.tolist(),
                    "coefficient": model[1].coef_.tolist(),
                    "intercept": np.asarray(model[1].intercept_).tolist(),
                    "alpha": 1.0,
                    "weighting": "equal_patient_total_weight_within_fit",
                    "interpretation": "conditional_association_not_primary_repolarization_or_causality",
                }
            )
        else:
            models.append(
                {"source": source, "status": "non_estimable", "reason": reason}
            )
        for i, record in enumerate(block.itertuples()):
            for j, target in enumerate(targets):
                ok = np.isfinite(prediction[i, j])
                outputs.append(
                    {
                        "source": source,
                        "subject_id": record.subject_id,
                        "study_id": record.study_id,
                        "split": record.split,
                        "target": target,
                        "predicted": prediction[i, j] if ok else None,
                        "residual": y[i, j] - prediction[i, j] if ok else None,
                        "evaluation_role": "out_of_fold"
                        if record.split == "train"
                        else "frozen_validation",
                        "status": "available" if ok else "non_estimable",
                        "reason": "" if ok else reason,
                        "clinical_claim_allowed": False,
                    }
                )
    return pd.DataFrame(outputs), models


def sensitivity_summary(baseline, variants, *, threshold=None, threshold_frozen=False):
    """Keep variant families separate; finite deterministic sensitivity, not CI."""
    if threshold is not None and (not threshold_frozen or not np.isfinite(threshold)):
        raise ValueError("threshold_must_be_frozen_and_finite")
    rows = []
    for family, raw in sorted(variants.items()):
        values = np.asarray(raw, dtype=float).ravel()
        finite = values[np.isfinite(values)]
        ok = np.isfinite(baseline) and len(finite) > 0
        rows.append(
            {
                "family": family,
                "n_attempted": len(values),
                "n_valid": len(finite),
                "status": "available" if ok else "non_estimable",
                "range_low": float(finite.min()) if ok else None,
                "range_high": float(finite.max()) if ok else None,
                "max_absolute_change": float(np.max(np.abs(finite - baseline)))
                if ok
                else None,
                "classification_flip_fraction": float(
                    np.mean((finite >= threshold) != (baseline >= threshold))
                )
                if ok and threshold is not None
                else None,
                "interval_kind": "prespecified_sensitivity_range_not_confidence_interval",
                "clinical_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def reference_timing_summary(frame, *, marker, endpoint):
    """Availability denominators precede filtering; signed reference timing.

    Actual modality/reference provenance must be provided by endpoint adapters.
    Missing reference is unknown, never negative. No verification-bias correction
    is implied by this descriptive stratification.
    """
    data = development_frame(
        frame,
        [
            marker,
            "reference_kind",
            "reference_verified",
            "reference_value",
            "reference_delta_hours",
            "setting",
        ],
    )
    delta = pd.to_numeric(data.reference_delta_hours, errors="coerce")
    delta = delta.where(np.isfinite(delta))
    data["time_stratum"] = (
        pd.cut(
            delta,
            [-np.inf, -24, -6, 0, 6, 24, np.inf],
            labels=[
                "before_over_24h",
                "before_6_24h",
                "before_0_6h",
                "after_0_6h",
                "after_6_24h",
                "after_over_24h",
            ],
        )
        .astype("string")
        .fillna("unknown")
    )
    data["marker_available"] = np.isfinite(pd.to_numeric(data[marker], errors="coerce"))
    data["reference_available"] = (
        data.reference_verified.map(lambda x: type(x) in (bool, np.bool_) and bool(x))
        & data.reference_value.notna()
    )
    rows = []
    for keys, block in data.groupby(
        ["source", "split", "setting", "reference_kind", "time_stratum"],
        dropna=False,
        observed=True,
    ):
        rows.append(
            dict(
                zip(
                    ["source", "split", "setting", "reference_kind", "time_stratum"],
                    keys,
                )
            )
            | {
                "endpoint": endpoint,
                "marker": marker,
                "n_records": len(block),
                "n_patients": block.subject_id.nunique(),
                "marker_available_n": int(block.marker_available.sum()),
                "verified_reference_n": int(block.reference_available.sum()),
                "both_available_n": int(
                    (block.marker_available & block.reference_available).sum()
                ),
                "clinical_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def long_window_summary(frame, markers, policy=ExtensionPolicy()):
    """Summarize disjoint fixed windows without treating them as patients.

    Input windows have already been extracted with the same frozen measurement
    pipeline. Short-window variability is not automatically long-term PRD/HRV.
    """
    required = [
        "source",
        "subject_id",
        "record_id",
        "split",
        "start_s",
        "duration_s",
        "quality",
        *markers,
    ]
    if set(required) - set(frame):
        raise ValueError("missing_window_columns")
    data = frame.copy()
    if (
        not data.split.isin(["train", "validation"]).all()
        or data[required[:4]].isna().any().any()
    ):
        raise ValueError("nondevelopment_or_missing_window_identity")
    if data.groupby(["source", "subject_id"]).split.nunique().gt(1).any():
        raise ValueError("patient_split_leakage")
    rows = []
    for (source, subject, record), block in data.groupby(
        ["source", "subject_id", "record_id"]
    ):
        block = block.sort_values("start_s")
        start = pd.to_numeric(block.start_s, errors="raise").to_numpy(float)
        duration = pd.to_numeric(block.duration_s, errors="raise").to_numpy(float)
        if (
            not np.isfinite(start).all()
            or (start < 0).any()
            or not np.all(duration == policy.window_seconds)
            or (start[1:] < start[:-1] + duration[:-1]).any()
        ):
            raise ValueError("invalid_or_overlapping_windows")
        quality = (
            pd.to_numeric(block.quality, errors="coerce")
            .between(policy.minimum_quality, 1)
            .to_numpy()
        )
        for marker in markers:
            values = pd.to_numeric(block[marker], errors="coerce").to_numpy(float)
            valid = np.isfinite(values) & quality
            usable = values[valid]
            ok = len(usable) >= policy.minimum_windows
            rows.append(
                {
                    "source": source,
                    "subject_id": subject,
                    "record_id": record,
                    "marker": marker,
                    "n_windows_total": len(block),
                    "n_windows_usable": len(usable),
                    "status": "available" if ok else "non_estimable",
                    "median": float(np.median(usable)) if ok else None,
                    "p05": float(np.quantile(usable, 0.05)) if ok else None,
                    "p95": float(np.quantile(usable, 0.95)) if ok else None,
                    "mad": float(np.median(np.abs(usable - np.median(usable))))
                    if ok
                    else None,
                    "independent_patients": 1,
                    "clinical_claim_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def clinical_association(
    frame,
    *,
    marker,
    outcome,
    covariates=("age", "sex", "heart_rate_bpm", "signal_quality_score", "care_setting"),
    binary=False,
    numeric_covariates=(),
):
    """One index per patient, source-specific adjusted HC3/robust-GLM effects.

    Selection of a reference-blind index is upstream; duplicates are rejected
    rather than selecting a better-labelled record here. Unknown labels remain
    missing. This is association, not validated prediction or treatment benefit.
    """
    from .broad_axis_adjustment import fit_adjusted_association

    if marker == outcome or outcome in covariates or marker in covariates:
        raise ValueError("target_predictor_overlap")
    data = development_frame(
        frame, [marker, outcome, "reference_verified", *covariates]
    )
    if data.duplicated(["source", "subject_id"]).any():
        raise ValueError("requires_reference_blind_patient_index")
    rows = []
    for (source, split), block in data.groupby(["source", "split"]):
        values = block[[marker, *covariates, outcome]].copy()
        for column in (marker, outcome, *numeric_covariates):
            values[column] = pd.to_numeric(values[column], errors="coerce")
        verified = block.reference_verified.map(
            lambda x: type(x) in (bool, np.bool_) and bool(x)
        )
        complete = (
            np.isfinite(values[[marker, outcome]].to_numpy()).all(axis=1)
            & verified.to_numpy()
        )
        complete &= values[list(covariates)].notna().all(axis=1).to_numpy()
        usable = values.loc[complete]
        result = dict(
            source=source,
            split=split,
            marker=marker,
            outcome=outcome,
            model="logit_robust" if binary else "OLS_HC3",
            n_index=len(block),
            n_complete=len(usable),
            n_marker_available=int(np.isfinite(values[marker]).sum()),
            clinical_claim_allowed=False,
            status="non_estimable",
            reason="insufficient_complete_patients",
        )
        fitted = fit_adjusted_association(
            usable,
            marker_column=marker,
            target_column=outcome,
            target_kind="binary" if binary else "continuous",
            covariates=covariates,
            numeric_covariates=numeric_covariates,
            model_id="prespecified_extension",
        )
        result.update(fitted)
        result["p"] = fitted["p_value"]
        result["coefficient"] = fitted["estimate"]
        result["status"] = (
            "estimated" if fitted["status"] == "estimable" else "non_estimable"
        )
        rows.append(result)
    return pd.DataFrame(rows)


def competing_diagnosis_frame(frame, *, target, competitor):
    """Known exclusive diagnostic groups only, with explicit overlap counts.

    Both label definitions must come from independently verified, compatible
    endpoints. Unknown competitor status cannot establish a pure target group.
    """
    data = development_frame(
        frame, [target, competitor, target + "_verified", competitor + "_verified"]
    )
    if data.duplicated(["source", "subject_id"]).any():
        raise ValueError("requires_reference_blind_patient_index")
    known = data[[target, competitor]].isin([0, 1]).all(axis=1)
    for name in (target, competitor):
        known &= data[name + "_verified"].map(
            lambda x: type(x) in (bool, np.bool_) and bool(x)
        )
    exclusive = known & data[target].ne(data[competitor])
    result = data.loc[exclusive].copy()
    result["comparison_outcome"] = result[target].astype(int)
    result["reference_verified"] = True
    return result, {
        "n_index": len(data),
        "n_known_both": int(known.sum()),
        "n_overlap": int((known & data[target].eq(1) & data[competitor].eq(1)).sum()),
        "n_neither": int((known & data[target].eq(0) & data[competitor].eq(0)).sum()),
        "n_target": int(result.comparison_outcome.eq(1).sum()),
        "n_competitor": int(result.comparison_outcome.eq(0).sum()),
        "clinical_claim_allowed": False,
    }


def cohort_information(frame, markers, *, event=None):
    """Measured source-wise sample/availability contribution, no pooling gain claim."""
    data = development_frame(frame, [*markers, *([event] if event else [])])
    rows = []
    for (source, split), block in data.groupby(["source", "split"]):
        for marker in markers:
            valid = np.isfinite(pd.to_numeric(block[marker], errors="coerce"))
            rows.append(
                {
                    "source": source,
                    "split": split,
                    "marker": marker,
                    "n_records": len(block),
                    "n_patients": block.subject_id.nunique(),
                    "available_records": int(valid.sum()),
                    "available_patients": block.loc[valid, "subject_id"].nunique(),
                    "event_patients": block.loc[
                        block[event].eq(1), "subject_id"
                    ].nunique()
                    if event
                    else None,
                    "unknown_event_records": int((~block[event].isin([0, 1])).sum())
                    if event
                    else None,
                    "cross_source_identity_status": "requires_external_copy_audit_no_global_unique_count_claim",
                    "model_improvement_proven": False,
                    "clinical_claim_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def window_rank_stability(frame, markers):
    """Descriptive first-window rank versus other-window median, one patient once.

    No p-values or gold-standard claims. First record is chosen by identity,
    not successful measurement; the comparator excludes the first window.
    """
    from scipy.stats import spearmanr

    long_window_summary(frame, markers)
    data = frame.sort_values(
        ["source", "subject_id", "record_id", "start_s"], kind="stable"
    )
    first = data.drop_duplicates(["source", "subject_id"])[
        ["source", "subject_id", "record_id"]
    ]
    data = data.merge(
        first, on=["source", "subject_id", "record_id"], validate="many_to_one"
    )
    pairs = []
    for (source, split, patient), block in data.groupby(
        ["source", "split", "subject_id"]
    ):
        for marker in markers:
            values = pd.to_numeric(block[marker], errors="coerce")
            quality = pd.to_numeric(block.quality, errors="coerce").between(0.7, 1)
            index = values.loc[block.start_s.eq(0) & quality]
            reference = values.loc[block.start_s.gt(0) & quality]
            reference = reference[np.isfinite(reference)]
            pairs.append(
                {
                    "source": source,
                    "split": split,
                    "subject_id": patient,
                    "marker": marker,
                    "first": float(index.iloc[0]) if len(index) else np.nan,
                    "other_median": float(reference.median())
                    if len(reference) >= 6
                    else np.nan,
                }
            )
    rows = []
    if not pairs:
        return pd.DataFrame()
    for (source, split, marker), block in pd.DataFrame(pairs).groupby(
        ["source", "split", "marker"]
    ):
        x = block[["first", "other_median"]].to_numpy(float)
        x = x[np.isfinite(x).all(axis=1)]
        ok = len(x) >= 30 and (np.ptp(x, axis=0) > 0).all()
        rows.append(
            {
                "source": source,
                "split": split,
                "marker": marker,
                "n_patients_total": len(block),
                "n_patients_complete": len(x),
                "status": "available" if ok else "non_estimable",
                "spearman_rho": float(spearmanr(x[:, 0], x[:, 1]).statistic)
                if ok
                else None,
                "interpretation": "descriptive_rank_stability_not_gold_standard_or_clinical_validation",
                "clinical_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def clinical_time_sensitivity(
    frame,
    *,
    marker,
    outcome,
    covariates=("age", "sex", "heart_rate_bpm", "signal_quality_score", "care_setting"),
    binary=False,
):
    """Prespecified signed time strata, same upstream index, no better-reference search."""
    data = development_frame(
        frame,
        ["reference_delta_hours", marker, outcome, "reference_verified", *covariates],
    )
    delta = pd.to_numeric(data.reference_delta_hours, errors="coerce")
    delta = delta.where(np.isfinite(delta))
    edges = [-np.inf, -24.0, -6.0, 0.0, 6.0, 24.0, np.inf]
    labels = [
        "before_over_24h",
        "before_6_24h",
        "before_0_6h",
        "after_0_6h",
        "after_6_24h",
        "after_over_24h",
    ]
    strata = pd.cut(delta, edges, labels=labels)
    outputs = []
    for label in labels:
        selected = data.loc[strata.eq(label)]
        if selected.empty:
            outputs.append(
                pd.DataFrame(
                    [
                        {
                            "time_stratum": label,
                            "status": "non_estimable",
                            "reason": "no_index_records_in_prespecified_stratum",
                            "n_index": 0,
                            "clinical_claim_allowed": False,
                        }
                    ]
                )
            )
        else:
            result = clinical_association(
                selected,
                marker=marker,
                outcome=outcome,
                covariates=covariates,
                binary=binary,
            )
            result["time_stratum"] = label
            outputs.append(result)
    outputs.append(
        pd.DataFrame(
            [
                {
                    "time_stratum": "unknown",
                    "status": "non_estimable",
                    "reason": "reference_timing_unknown",
                    "n_index": int(delta.isna().sum()),
                    "clinical_claim_allowed": False,
                }
            ]
        )
    )
    return pd.concat(outputs, ignore_index=True)
