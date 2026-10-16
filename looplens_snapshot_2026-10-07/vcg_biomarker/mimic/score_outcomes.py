"""Typed continuous/survival score adapters, sharing formula and evaluation contracts.

These APIs require verified outcomes. Administrative absence is never converted
to censoring, and a Cox linear predictor is never reported as absolute risk.
"""

from dataclasses import asdict
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg
from threadpoolctl import threadpool_limits

from .feature_roles import canonical_feature_id
from .score_research import NotEstimable, ScoreSpec, SearchPolicy, ScoreEvaluation
from .survival import concordance_index


def _loss(y, prediction, time=None):
    valid = np.isfinite(prediction)
    if not valid.all():
        return float("inf")
    if time is None:
        return float(mean_squared_error(y, prediction))
    c, pairs = concordance_index(time, y, prediction)
    return 1 - c if pairs else float("inf")


def _fit(frame, y, selected, endpoint, group, provenance, time=None):
    x = frame[list(selected)].to_numpy(float)
    valid = np.isfinite(x).any(axis=1)
    if valid.sum() < 10 or (
        time is not None and y[valid].sum() < max(20, 10 * len(selected))
    ):
        raise NotEstimable("insufficient_outcomes_for_complexity")
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(x[valid]))
    if np.linalg.matrix_rank(z) < len(selected):
        raise NotEstimable("collinear_score_components")
    if time is None:
        fit = Ridge(alpha=1.0).fit(z, y[valid])
        coefficients, intercept, link = fit.coef_, float(fit.intercept_), "identity"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.simplefilter("error", RuntimeWarning)
            try:
                fit = PHReg(
                    time[valid], z, status=y[valid], ties="efron"
                ).fit_regularized(alpha=0.01, L1_wt=0, maxiter=200)
            except (UserWarning, RuntimeWarning, np.linalg.LinAlgError) as exc:
                raise NotEstimable("unstable_regularized_cox") from exc
        coefficients, intercept, link = (
            np.asarray(fit.params),
            0.0,
            "log_relative_hazard",
        )
    return ScoreSpec(
        endpoint,
        group,
        tuple(selected),
        tuple(imputer.statistics_),
        tuple(scaler.mean_),
        tuple(scaler.scale_),
        tuple(coefficients),
        intercept,
        provenance,
        purpose="prognosis_research"
        if time is not None
        else "continuous_outcome_research",
        link=link,
    )


def _select(
    frame, y, groups, family, policy, depth, endpoint, group, provenance, time=None
):
    # Bounded train-only univariate ranking, followed by family/diversity guards.
    ranked = []
    for name in frame:
        if canonical_feature_id(name) != name:
            continue
        x = frame[name].to_numpy(float)
        valid = np.isfinite(x)
        if valid.mean() < policy.minimum_available or np.std(x[valid]) < 1e-12:
            continue
        if time is None:
            strength = abs(np.corrcoef(x[valid], y[valid])[0, 1])
        else:
            c, pairs = concordance_index(time[valid], y[valid], x[valid])
            strength = abs(c - 0.5) if pairs else np.nan
        if np.isfinite(strength):
            ranked.append((strength, name))
    candidates = [
        name
        for _, name in sorted(ranked, key=lambda a: (-a[0], a[1]))[
            : policy.max_features
        ]
    ]
    chosen: list[str] = []
    for _ in range(depth):
        results = []
        for name in candidates:
            if name in chosen or sum(family(n) == family(name) for n in chosen) >= 2:
                continue
            try:
                spec = _fit(
                    frame, y, chosen + [name], endpoint, group, provenance, time
                )
                loss = _loss(y, spec.predict(frame), time)
                if np.isfinite(loss):
                    results.append((loss, name))
            except NotEstimable:
                continue
        if not results:
            break
        chosen.append(min(results)[1])
    if not chosen:
        raise NotEstimable("no_stable_outcome_score")
    return tuple(chosen)


def discover_outcome_score(
    frame,
    outcome,
    patient_ids,
    *,
    endpoint_id,
    model_group,
    family,
    provenance,
    time=None,
    policy=SearchPolicy(),
):
    """Nested additive regression. `time` selects right-censored Cox prognosis."""
    if provenance.get("split") != "train" or not provenance.get("input_hash"):
        raise ValueError("explicit_train_provenance_required")
    if provenance.get("outcome_reference_verified") is not True:
        raise ValueError("verified_outcome_adapter_required")
    if time is not None and (
        provenance.get("followup_verified") is not True
        or not provenance.get("time_zero")
    ):
        raise ValueError("verified_followup_and_time_zero_required")
    y, groups = np.asarray(outcome, float), np.asarray(patient_ids)
    frame = frame.astype(float).replace([np.inf, -np.inf], np.nan)
    if (
        len(frame) != len(y)
        or len(y) != len(groups)
        or not np.isfinite(y).all()
        or pd.isna(groups).any()
    ):
        raise ValueError("unknown_or_unaligned_outcomes")
    # Primary cross-sectional / prognosis index analyses contain one row per
    # patient. Within-subject response must use a separately prepared episode model.
    if len(np.unique(groups)) != len(groups):
        raise ValueError("one_reference_blind_index_per_patient_required")
    if len(groups) < max(30, policy.outer_folds * policy.inner_folds):
        raise NotEstimable("insufficient_patients")
    if time is not None:
        time = np.asarray(time, float)
        if (
            len(time) != len(y)
            or not np.isfinite(time).all()
            or (time <= 0).any()
            or not np.isin(y, [0, 1]).all()
        ):
            raise ValueError("invalid_survival_outcome")
        if y.sum() < policy.minimum_class_patients:
            raise NotEstimable("insufficient_events")
    elif np.std(y) < 1e-12:
        raise NotEstimable("constant_outcome")

    def select_depth(x, yy, gg, tt):
        candidates = []
        for depth in range(1, policy.max_components + 1):
            losses = []
            for fit, test in GroupKFold(policy.inner_folds).split(x, yy, gg):
                try:
                    selected = _select(
                        x.iloc[fit],
                        yy[fit],
                        gg[fit],
                        family,
                        policy,
                        depth,
                        endpoint_id,
                        model_group,
                        provenance,
                        tt[fit] if tt is not None else None,
                    )
                    spec = _fit(
                        x.iloc[fit],
                        yy[fit],
                        selected,
                        endpoint_id,
                        model_group,
                        provenance,
                        tt[fit] if tt is not None else None,
                    )
                    losses.append(
                        _loss(
                            yy[test],
                            spec.predict(x.iloc[test]),
                            tt[test] if tt is not None else None,
                        )
                    )
                except NotEstimable:
                    losses.append(float("inf"))
            candidates.append((np.mean(losses), depth))
        loss, depth = min(candidates)
        if not np.isfinite(loss):
            raise NotEstimable("no_estimable_inner_models")
        return depth

    prediction = np.full(len(y), np.nan)
    fold_info = []
    with threadpool_limits(limits=1):
        for fit, test in GroupKFold(policy.outer_folds).split(frame, y, groups):
            tt = time[fit] if time is not None else None
            depth = select_depth(frame.iloc[fit], y[fit], groups[fit], tt)
            selected = _select(
                frame.iloc[fit],
                y[fit],
                groups[fit],
                family,
                policy,
                depth,
                endpoint_id,
                model_group,
                provenance,
                tt,
            )
            spec = _fit(
                frame.iloc[fit],
                y[fit],
                selected,
                endpoint_id,
                model_group,
                provenance,
                tt,
            )
            prediction[test] = spec.predict(frame.iloc[test])
            fold_info.append(
                {"components": selected, "n_fit": len(fit), "n_test": len(test)}
            )
            if time is not None:
                c, pairs = concordance_index(time[test], y[test], prediction[test])
                fold_info[-1].update(harrell_c=c, comparable_pairs=pairs)
        depth = select_depth(frame, y, groups, time)
        selected = _select(
            frame,
            y,
            groups,
            family,
            policy,
            depth,
            endpoint_id,
            model_group,
            provenance,
            time,
        )
        spec = _fit(frame, y, selected, endpoint_id, model_group, provenance, time)
    metrics: dict[str, Any]
    if time is None:
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
            "r2": float(r2_score(y, prediction)),
        }
    else:
        # Cox predictors from different folds have arbitrary baseline offsets.
        # Compare only within-fold pairs, not uncalibrated cross-fold risks.
        pairs = sum(row["comparable_pairs"] for row in fold_info)
        if not pairs:
            raise NotEstimable("no_comparable_survival_pairs")
        c = (
            sum(
                row["harrell_c"] * row["comparable_pairs"]
                for row in fold_info
                if row["comparable_pairs"]
            )
            / pairs
        )
        metrics = {
            "harrell_c": c,
            "comparable_pairs": pairs,
            "events": int(y.sum()),
            "absolute_risk": "not_estimated",
            "competing_risks": "separate_endpoint_adapter_required",
            "proportional_hazards": "not_yet_externally_verified",
        }
    metrics.update(
        n_patients=len(groups),
        prediction_availability=float(np.isfinite(prediction).mean()),
    )
    return {
        "score_spec": spec.to_dict(),
        "evaluation": asdict(
            ScoreEvaluation(
                endpoint_id,
                model_group,
                "evaluated_search_procedure",
                metrics,
                provenance,
            )
        ),
        "oof_predictions": prediction,
        "folds": fold_info,
        "policy": asdict(policy),
        "clinical_claim_allowed": False,
    }
