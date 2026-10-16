"""Protocol-bound, cross-fitted AIPW research; not a treatment recommender."""

from dataclasses import dataclass, asdict
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .decision_research import TreatmentProfile


@dataclass(frozen=True)
class CausalProtocol:
    profile: TreatmentProfile
    confounders: tuple[str, ...]
    outcome_kind: str = "binary"
    folds: int = 5
    seed: int = 20260914
    minimum_arm: int = 30
    propensity_bound: float = 0.01
    minimum_ess: float = 20
    horizon_seconds: float = 30 * 86400
    maximum_weighted_smd: float = 0.2


def estimate_treatment_effects(
    frame,
    protocol,
    *,
    verified_fields,
    record_level_verified=False,
    confounding_plan_verified=False,
    identity_audit_verified=False,
):
    """One index/patient; time-zero strategies, complete fixed-horizon outcomes.

    Required adapter times are relative seconds: treatment_time=0,
    covariate_available_time<=0, followup_complete=True. No grace periods,
    unknown censoring, post-treatment confounders or automatic imputation.
    ATE sign is treated minus comparator; benefit/harm are never combined.
    """
    result = {
        "version": "1.0.0",
        "status": "non_estimable",
        "clinical_claim_allowed": False,
        "reasons": [],
        "identifying_assumptions": [
            "consistency",
            "no_unmeasured_confounding",
            "positivity",
            "no_interference",
        ],
        "assumptions_verified_by_model": False,
    }
    p = protocol
    if not isinstance(p, CausalProtocol):
        raise TypeError("CausalProtocol required")
    readiness = p.profile.readiness(
        verified_fields=verified_fields,
        record_level_verified=record_level_verified,
        overlap_verified=True,
        confounding_plan_verified=confounding_plan_verified,
    )
    result["protocol"] = asdict(p)
    result["protocol_content_hash"] = hashlib.sha256(
        json.dumps(asdict(p), sort_keys=True).encode()
    ).hexdigest()
    reasons = list(readiness["reasons"])
    if p.profile.design != "treatment_benefit_research":
        reasons.append("benefit_research_protocol_required")
    if identity_audit_verified is not True:
        reasons.append("cross_source_identity_audit_required")
    if (
        p.outcome_kind not in {"binary", "continuous"}
        or p.folds < 2
        or p.minimum_arm < 10
        or p.minimum_ess < 10
        or not 0 < p.propensity_bound < 0.5
        or not np.isfinite(p.horizon_seconds)
        or p.horizon_seconds <= 0
        or not 0 < p.maximum_weighted_smd <= 1
    ):
        reasons.append("invalid_analysis_policy")
    outcomes = [p.profile.benefit_endpoint, p.profile.harm_endpoint]
    metadata = {
        "cohort_id",
        "patient_id",
        "split",
        "treatment",
        "treatment_time",
        "covariate_available_time",
        "followup_complete",
        "outcome_horizon_seconds",
    }
    if (
        not p.confounders
        or len(set(p.confounders)) != len(p.confounders)
        or set(p.confounders) & (metadata | set(outcomes))
        or len(set(outcomes)) != 2
        or set(outcomes) & metadata
    ):
        reasons.append("invalid_predictor_or_endpoint_roles")
    required = metadata | set(outcomes) | set(p.confounders)
    if required - set(frame):
        reasons.append("missing_columns:" + ",".join(sorted(required - set(frame))))
    if reasons:
        result["reasons"] = reasons
        return result
    if frame.empty or frame[list(required)].isna().any().any():
        reasons.append("missing_data_requires_explicit_protocol")
    if not frame.split.eq("train").all():
        reasons.append("development_only_locked_or_validation_forbidden")
    if frame.duplicated(["cohort_id", "patient_id"]).any():
        reasons.append("one_index_per_patient_required")
    if not frame.followup_complete.map(
        lambda v: isinstance(v, (bool, np.bool_)) and bool(v)
    ).all():
        reasons.append("censoring_or_unknown_followup_unsupported")
    try:
        x = frame[list(p.confounders)].to_numpy(dtype=float)
        a = frame.treatment.to_numpy(dtype=float)
        ys = frame[outcomes].to_numpy(dtype=float)
        times = frame[["treatment_time", "covariate_available_time"]].to_numpy(
            dtype=float
        )
        if not all(np.isfinite(v).all() for v in (x, a, ys, times)):
            reasons.append("nonfinite_data")
        if not np.isin(a, [0, 1]).all():
            reasons.append("binary_strategy_required")
        if (times[:, 0] != 0).any() or (times[:, 1] > 0).any():
            reasons.append("time_zero_or_future_covariate_violation")
        if not frame.outcome_horizon_seconds.eq(p.horizon_seconds).all():
            reasons.append("endpoint_horizon_does_not_match_protocol")
        if p.outcome_kind == "binary" and not np.isin(ys, [0, 1]).all():
            reasons.append("binary_outcomes_required")
    except (ValueError, TypeError):
        reasons.append("invalid_numeric_data")
    if reasons:
        result["reasons"] = reasons
        return result
    if min(np.sum(a == 0), np.sum(a == 1)) < max(p.minimum_arm, p.folds):
        result["reasons"] = ["insufficient_treatment_patients"]
        return result
    e = np.full(len(a), np.nan)
    mu = np.full((len(a), 2, 2), np.nan)
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            for train, test in StratifiedKFold(
                p.folds, shuffle=True, random_state=p.seed
            ).split(x, a):
                ps = make_pipeline(
                    StandardScaler(), LogisticRegression(C=1, max_iter=1000)
                )
                ps.fit(x[train], a[train])
                e[test] = ps.predict_proba(x[test])[:, 1]
                for arm in (0, 1):
                    rows = train[a[train] == arm]
                    for j in range(2):
                        if p.outcome_kind == "binary":
                            if (
                                min(np.sum(ys[rows, j] == 0), np.sum(ys[rows, j] == 1))
                                < 5
                            ):
                                raise ValueError("insufficient_outcome_events_in_fold")
                            model = make_pipeline(
                                StandardScaler(), LogisticRegression(C=1, max_iter=1000)
                            )
                            model.fit(x[rows], ys[rows, j])
                            mu[test, arm, j] = model.predict_proba(x[test])[:, 1]
                        else:
                            model = make_pipeline(StandardScaler(), Ridge(alpha=1))
                            model.fit(x[rows], ys[rows, j])
                            mu[test, arm, j] = model.predict(x[test])
    except (ValueError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
        result["reasons"] = ["nuisance_model_failure:" + str(exc)]
        return result
    if (e <= p.propensity_bound).any() or (e >= 1 - p.propensity_bound).any():
        result["reasons"] = ["insufficient_overlap_no_silent_trimming"]
        return result
    w = np.where(a == 1, 1 / e, 1 / (1 - e))
    ess = [
        float(w[a == arm].sum() ** 2 / np.square(w[a == arm]).sum()) for arm in (0, 1)
    ]
    result["overlap"] = {
        "propensity_range": [float(e.min()), float(e.max())],
        "arm_ess": ess,
    }
    if min(ess) < p.minimum_ess:
        result["reasons"] = ["insufficient_weighted_effective_sample_size"]
        return result
    balance = {}
    for j, name in enumerate(p.confounders):
        values = x[:, j]
        denominator = np.sqrt(
            (values[a == 0].var(ddof=1) + values[a == 1].var(ddof=1)) / 2
        )
        difference = float(values[a == 1].mean() - values[a == 0].mean())
        weighted_difference = float(
            np.average(values[a == 1], weights=w[a == 1])
            - np.average(values[a == 0], weights=w[a == 0])
        )
        if denominator <= 1e-12 and abs(difference) > 1e-12:
            result["reasons"] = ["deterministic_confounder_treatment_separation"]
            return result
        balance[name] = {
            "unweighted_smd": difference / denominator if denominator > 1e-12 else 0.0,
            "weighted_smd": weighted_difference / denominator
            if denominator > 1e-12
            else 0.0,
        }
    result["confounder_balance"] = balance
    if any(abs(v["weighted_smd"]) > p.maximum_weighted_smd for v in balance.values()):
        result["reasons"] = ["residual_weighted_confounder_imbalance"]
        return result
    effects = {}
    for j, name in enumerate(outcomes):
        pseudo = (
            mu[:, 1, j]
            - mu[:, 0, j]
            + a * (ys[:, j] - mu[:, 1, j]) / e
            - (1 - a) * (ys[:, j] - mu[:, 0, j]) / (1 - e)
        )
        ate = float(pseudo.mean())
        se = float(pseudo.std(ddof=1) / np.sqrt(len(a)))
        effects[name] = {
            "ate": ate,
            "ci95": [ate - 1.96 * se, ate + 1.96 * se],
            "standard_error": se,
            "contrast": "treatment_minus_comparator",
            "scale": "risk_difference"
            if p.outcome_kind == "binary"
            else "outcome_units",
        }
    result.update(
        status="research_estimate",
        n_patients=len(a),
        effects=effects,
        uncertainty="cross_fitted_influence_function_asymptotic",
        input_hash=hashlib.sha256(
            pd.util.hash_pandas_object(
                frame[sorted(required)], index=False
            ).values.tobytes()
        ).hexdigest(),
    )
    return result
