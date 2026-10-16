"""Train-only, nested evaluation of the existing small-combination search.

Outer predictions evaluate a search PROCEDURE, not a formula selected using
those predictions. Final formulas are fitted on development training data only.
No function in this module reads a cohort, changes labels, or releases a score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping
import warnings
from joblib import Parallel, delayed, parallel_config

import numpy as np
import pandas as pd
import psutil
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .feature_roles import canonical_feature_id
from .optimization import grouped_score_search

VERSION = "1.1.0"


class NotEstimable(ValueError):
    """A scientific abstention, not an unexpected programming error."""


@dataclass(frozen=True)
class SearchPolicy:
    outer_folds: int = 5
    inner_folds: int = 3
    max_components: int = 3
    max_features: int = 40
    minimum_class_patients: int = 30
    minimum_available: float = 0.80
    bootstrap_iterations: int = 200
    seed: int = 20260914
    benchmark: bool = True
    workers: int = 1

    def __post_init__(self):
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 1
        ):
            raise ValueError("positive_integer_workers_required")
        if min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("at_least_two_folds_required")
        if not 1 <= self.max_components <= 3 or self.max_features < self.max_components:
            raise ValueError("invalid_search_budget")
        if (
            self.minimum_class_patients < self.outer_folds
            or self.bootstrap_iterations < 0
        ):
            raise ValueError("invalid_inference_budget")
        if not 0 < self.minimum_available <= 1:
            raise ValueError("invalid_availability")


@dataclass(frozen=True)
class ScoreSpec:
    """Weighted extension contract; legacy FrozenScoreSpec remains readable."""

    endpoint_id: str
    model_group: str
    components: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    provenance: Mapping[str, object]
    purpose: str = "diagnosis_research"
    version: str = VERSION
    evidence_stage: str = "exploratory"
    clinical_claim_allowed: bool = False
    missingness_rule: str = "training_median; abstain_if_all_components_missing"
    units: Mapping[str, str] = field(default_factory=dict)
    link: str = "logit"
    prespecified_thresholds: tuple[float, ...] = ()

    def __post_init__(self):
        # JSON arrays arrive as lists; do not leave a supposedly frozen formula
        # mutable through coefficients[0] or a caller-owned input list.
        for name in (
            "components",
            "medians",
            "means",
            "scales",
            "coefficients",
            "prespecified_thresholds",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.components)
        if not n or len(set(self.components)) != n:
            raise ValueError("empty_or_duplicate_components")
        for values in (self.medians, self.means, self.scales, self.coefficients):
            if len(values) != n or not np.isfinite(values).all():
                raise ValueError("invalid_formula_arrays")
        if min(self.scales) <= 0 or not np.isfinite(self.intercept):
            raise ValueError("invalid_scale_or_intercept")
        if self.clinical_claim_allowed or self.evidence_stage != "exploratory":
            raise ValueError("research_contract_cannot_release_clinical_score")
        if self.link not in {"logit", "identity", "log_relative_hazard"}:
            raise ValueError("unsupported_score_link")
        if any(
            not np.isfinite(t) or not 0 < t < 1 for t in self.prespecified_thresholds
        ):
            raise ValueError("invalid_prespecified_probability_threshold")

    def predict(self, frame: pd.DataFrame, *, point_step: float | None = None):
        x = frame.reindex(columns=self.components).to_numpy(dtype=float)
        valid = np.isfinite(x)
        x = np.where(valid, x, self.medians)
        contributions = (x - self.means) / self.scales * self.coefficients
        if point_step is not None:
            if not np.isfinite(point_step) or point_step <= 0:
                raise ValueError("invalid_point_step")
            contributions = np.rint(contributions / point_step) * point_step
        linear = self.intercept + contributions.sum(axis=1)
        result = expit(linear) if self.link == "logit" else linear
        result[~valid.any(axis=1)] = np.nan
        return result

    def to_dict(self):
        linear = "intercept + sum(coef * (impute(x)-mean)/scale)"
        return {
            **asdict(self),
            "formula": f"expit({linear})" if self.link == "logit" else linear,
        }


@dataclass(frozen=True)
class ScoreEvaluation:
    endpoint_id: str
    model_group: str
    status: str
    metrics: Mapping[str, object]
    provenance: Mapping[str, object]
    abstention_reason: str = ""
    evidence_stage: str = "exploratory"
    clinical_claim_allowed: bool = False


def _splits(y, groups, folds, seed, minimum=2):
    if np.unique(y).tolist() != [0, 1]:
        raise NotEstimable("binary_known_labels_required")
    if any(len(np.unique(groups[y == c])) < max(folds, minimum) for c in (0, 1)):
        raise NotEstimable("insufficient_class_patients")
    pairs = list(
        StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(y, y, groups)
    )
    for a, b in pairs:
        if len(np.unique(y[a])) != 2 or len(np.unique(y[b])) != 2:
            raise NotEstimable("single_class_fold")
        if set(groups[a]) & set(groups[b]):
            raise AssertionError("patient_leakage")
    return pairs


def _pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=500),
    )


def _fit(frame, y, components):
    model = _pipeline()
    measurable = np.isfinite(frame[list(components)].to_numpy(float)).any(axis=1)
    if np.unique(y[measurable]).size != 2:
        raise NotEstimable("insufficient_measurable_classes")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(frame.loc[measurable, list(components)], y[measurable])
        except ConvergenceWarning as exc:
            raise NotEstimable("non_convergent_model") from exc
    return model


def _predict(model, frame, components):
    x = frame[list(components)]
    p = model.predict_proba(x)[:, 1]
    p[~np.isfinite(x.to_numpy(float)).any(axis=1)] = np.nan
    return p


def _representatives(frame, y, family, policy):
    """All outcome-dependent screening is called on fitting rows only."""
    ranked = []
    for name in frame:
        if canonical_feature_id(name) != name:
            continue
        x = frame[name].to_numpy(float)
        valid = np.isfinite(x)
        if valid.mean() < policy.minimum_available or np.unique(y[valid]).size != 2:
            continue
        if np.std(x[valid]) <= 1e-12:
            continue
        ranked.append((abs(roc_auc_score(y[valid], x[valid]) - 0.5), name))
    counts: dict[str, int] = {}
    chosen: list[str] = []
    for _, name in sorted(ranked, key=lambda row: (-row[0], row[1])):
        group = family(name)
        if counts.get(group, 0) >= 2:
            continue
        # Pairwise-complete redundancy, evaluated on these fitting rows only.
        redundant = False
        for other in chosen:
            pair = frame[[name, other]].dropna()
            if len(pair) > 2 and abs(pair.corr().iloc[0, 1]) >= 0.95:
                redundant = True
                break
        if not redundant:
            chosen.append(name)
            counts[group] = counts.get(group, 0) + 1
        if len(chosen) >= policy.max_features:
            break
    if not chosen:
        raise NotEstimable("no_available_nonredundant_features")
    return tuple(chosen)


def _selection_choices(frame, y, groups, family, policy):
    representatives = _representatives(frame, y, family, policy)
    choices = {
        depth: representatives[:1] for depth in range(1, policy.max_components + 1)
    }
    if policy.max_components == 1:
        return choices
    candidates = grouped_score_search(
        frame,
        labels=y,
        subject_ids=groups,
        feature_ids=representatives,
        family_for_feature=family,
        folds=policy.inner_folds,
        max_components=policy.max_components,
        representatives_per_family=2,
        selected_pairs=2,
        selected_triples=2,
        triple_seed_pairs=8,
        minimum_complete_fraction=policy.minimum_available,
        maximum_pairwise_abs_correlation=0.95,
        random_seed=policy.seed,
        eligibility_predicate=lambda name: name in representatives,
    )
    for depth in range(2, policy.max_components + 1):
        eligible = [c for c in candidates if c.stable and len(c.components) == depth]
        if eligible:
            choices[depth] = max(
                eligible, key=lambda c: (c.mean_auc, c.components)
            ).components
    return choices


def _select(frame, y, groups, family, policy, depth):
    return _selection_choices(frame, y, groups, family, policy)[depth]


def _choose_depth(frame, y, groups, family, policy, progress=None, fixed_components=()):
    pairs = _splits(y, groups, policy.inner_folds, policy.seed + 1)
    losses_by_depth: dict[int, list[float]] = {
        depth: [] for depth in range(1, policy.max_components + 1)
    }
    for fold_number, (fit, test) in enumerate(pairs):
        training, testing = frame.iloc[fit], frame.iloc[test]
        choices = _selection_choices(
            training.drop(columns=list(fixed_components)),
            y[fit],
            groups[fit],
            family,
            policy,
        )
        # The same fallback singleton may occur at several depths. Fit/evaluate
        # each distinct formula only once on this exact fitting fold.
        cached_loss = {}
        for depth, selected in choices.items():
            selected = tuple(fixed_components) + selected
            if progress:
                progress(
                    "inner_selection",
                    fold_number * policy.max_components + depth - 1,
                    policy.max_components * len(pairs),
                )
            if selected not in cached_loss:
                model = _fit(training, y[fit], selected)
                p = _predict(model, testing, selected)
                # Selection only: reported predictions retain their abstentions.
                selection_p = np.where(np.isfinite(p), p, float(y[fit].mean()))
                cached_loss[selected] = log_loss(y[test], selection_p, labels=[0, 1])
            losses_by_depth[depth].append(cached_loss[selected])
    losses = [
        (float(np.mean(values)), depth) for depth, values in losses_by_depth.items()
    ]
    loss, depth = min(losses)
    if not np.isfinite(loss):
        raise NotEstimable("incomplete_inner_predictions")
    return depth


def probability_metrics(y, p, groups, *, iterations=200, seed=20260914):
    y, p, groups = map(np.asarray, (y, p, groups))
    if not len(y):
        return {
            "n_eligible": 0,
            "n_predicted": 0,
            "status": "non_estimable",
            "reason": "empty_cohort",
        }
    if len(y) != len(p) or len(y) != len(groups) or not np.isin(y, [0, 1]).all():
        raise ValueError("unaligned_or_unknown_binary_outcomes")
    valid = np.isfinite(p)
    result: dict[str, Any] = {
        "confidence_interval_scope": "patient_cluster_bootstrap_conditional_on_supplied_predictions; no_search_refit",
        "n_eligible": len(y),
        "n_predicted": int(valid.sum()),
        "prediction_availability": float(valid.mean()),
        "n_patients": len(np.unique(groups)),
        "n_events": int(np.sum(y == 1)),
    }
    if valid.sum() == 0 or np.unique(y[valid]).size < 2:
        return {**result, "status": "non_estimable"}
    yy, pp, gg = y[valid], p[valid], groups[valid]
    result.update(
        auc=float(roc_auc_score(yy, pp)),
        average_precision=float(average_precision_score(yy, pp)),
        brier=float(brier_score_loss(yy, pp)),
        log_loss=float(log_loss(yy, pp, labels=[0, 1])),
    )
    bins = np.minimum((pp * 10).astype(int), 9)
    result["calibration_bins"] = [
        {
            "bin": b,
            "n": int(np.sum(bins == b)),
            "predicted": float(pp[bins == b].mean()),
            "observed": float(yy[bins == b].mean()),
        }
        for b in range(10)
        if np.any(bins == b)
    ]
    # Diagnostics only: this model is never used to recalibrate these predictions.
    from statsmodels.api import GLM, families, add_constant
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    logits = np.log(np.clip(pp, 1e-7, 1 - 1e-7) / np.clip(1 - pp, 1e-7, 1))
    try:
        if np.std(logits) < 1e-10:
            raise NotEstimable("constant_predictions")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            fit = GLM(yy, add_constant(logits), family=families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": gg}
            )
        if not fit.converged or not np.isfinite(fit.params).all():
            raise NotEstimable("unstable_calibration")
        result["calibration_intercept"], result["calibration_slope"] = map(
            float, fit.params
        )
        result["calibration_coefficient_ci95"] = fit.conf_int().tolist()
        result["calibration_status"] = "estimated"
    except (
        PerfectSeparationWarning,
        PerfectSeparationError,
        NotEstimable,
        np.linalg.LinAlgError,
    ):
        result["calibration_status"] = "non_estimable"
    rng = np.random.default_rng(seed)
    ids, inverse = np.unique(gg, return_inverse=True)
    # Cluster multinomial weights avoid O(patients * rows) index construction
    # and avoid expanding the cohort once per bootstrap replicate.
    aucs, briers = [], []
    observed: dict[int, list[float]] = {b: [] for b in range(10)}
    for _ in range(iterations):
        weights = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[
            inverse
        ]
        if all(weights[yy == label].sum() > 0 for label in (0, 1)):
            aucs.append(roc_auc_score(yy, pp, sample_weight=weights))
            briers.append(np.average((yy - pp) ** 2, weights=weights))
        denominator = np.bincount(bins, weights=weights, minlength=10)
        numerator = np.bincount(bins, weights=weights * yy, minlength=10)
        for b in range(10):
            if denominator[b] > 0:
                observed[b].append(numerator[b] / denominator[b])
    result["auc_ci95"] = (
        list(map(float, np.quantile(aucs, [0.025, 0.975]))) if aucs else None
    )
    result["bootstrap_evaluable"] = len(aucs)
    result["brier_ci95"] = (
        list(map(float, np.quantile(briers, [0.025, 0.975]))) if briers else None
    )
    for row in result["calibration_bins"]:
        values = observed[row["bin"]]
        row["observed_ci95"] = (
            list(map(float, np.quantile(values, [0.025, 0.975]))) if values else None
        )
    return result


def paired_probability_comparison(
    y, first, second, patient_ids, *, iterations=200, seed=20260914
):
    y, first, second, groups = map(np.asarray, (y, first, second, patient_ids))
    valid = np.isfinite(first) & np.isfinite(second)
    if np.unique(y[valid]).size < 2:
        return {"status": "non_estimable", "reason": "insufficient_common_predictions"}
    y, first, second, groups = y[valid], first[valid], second[valid], groups[valid]
    ids, inverse = np.unique(groups, return_inverse=True)
    delta = float(roc_auc_score(y, second) - roc_auc_score(y, first))
    rng, replicates = np.random.default_rng(seed), []
    for _ in range(iterations):
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[
            inverse
        ]
        if all(w[y == c].sum() > 0 for c in (0, 1)):
            replicates.append(
                roc_auc_score(y, second, sample_weight=w)
                - roc_auc_score(y, first, sample_weight=w)
            )
    return {
        "status": "research_comparison",
        "confidence_interval_scope": "paired_patient_cluster_bootstrap_conditional_on_supplied_predictions; no_search_refit",
        "n_common": len(y),
        "n_patients": len(ids),
        "delta_auc_second_minus_first": delta,
        "delta_auc_ci95": np.quantile(replicates, [0.025, 0.975]).tolist()
        if replicates
        else None,
        "delta_brier_second_minus_first": float(
            np.mean((y - second) ** 2 - (y - first) ** 2)
        ),
        "clinical_claim_allowed": False,
    }


def operating_points(y, probabilities, thresholds=()):
    """Research-only decision curves at thresholds frozen BEFORE evaluation."""
    if not thresholds:
        return {"status": "not_computed", "reason": "no_prespecified_thresholds"}
    if any(not np.isfinite(t) or not 0 < t < 1 for t in thresholds):
        raise ValueError("invalid_prespecified_probability_threshold")
    y, p = np.asarray(y), np.asarray(probabilities, float)
    if len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid_binary_operating_point_inputs")
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    if np.unique(y).size != 2:
        return {"status": "non_estimable", "reason": "insufficient_evaluable_classes"}
    rows = []
    for threshold in thresholds:
        selected = p >= threshold
        tp, fp = int(np.sum(selected & (y == 1))), int(np.sum(selected & (y == 0)))
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "n": len(y),
                "sensitivity": tp / int(np.sum(y == 1)),
                "specificity": 1 - fp / int(np.sum(y == 0)),
                "net_benefit": (tp - fp * odds) / len(y),
                "treat_all_net_benefit": float(y.mean() - (1 - y.mean()) * odds),
                "treat_none_net_benefit": 0.0,
            }
        )
    return {
        "status": "research_estimates",
        "operating_points": rows,
        "clinical_claim_allowed": False,
        "scope": "evaluated_reference_subset; assumes threshold odds encode the relevant harm-benefit tradeoff",
    }


def _spec(model, selected, endpoint, group, provenance):
    imputer, scaler, regression = (
        model.steps[0][1],
        model.steps[1][1],
        model.steps[2][1],
    )
    return ScoreSpec(
        endpoint,
        group,
        tuple(selected),
        tuple(imputer.statistics_),
        tuple(scaler.mean_),
        tuple(scaler.scale_),
        tuple(regression.coef_[0]),
        float(regression.intercept_[0]),
        provenance,
        units={
            name: provenance.get("feature_units", {}).get(name, "unverified")
            for name in selected
        },
        prespecified_thresholds=tuple(provenance.get("prespecified_thresholds", ())),
    )


def _evaluate_outer_fold(
    frame,
    y,
    groups,
    fit,
    test,
    number,
    family,
    policy,
    fixed_components,
    endpoint_id,
    model_group,
    provenance,
):
    """Independent fold; no GUI callback, shared mutable state or nested executor."""
    with threadpool_limits(limits=1):
        train = frame.iloc[fit]
        candidates = train.drop(columns=list(fixed_components))
        depth = (
            _choose_depth(
                train,
                y[fit],
                groups[fit],
                family,
                policy,
                fixed_components=fixed_components,
            )
            if len(candidates.columns)
            else 0
        )
        selected = fixed_components + (
            _select(candidates, y[fit], groups[fit], family, policy, depth)
            if depth
            else ()
        )
        model = _fit(train, y[fit], selected)
        spec = _spec(model, selected, endpoint_id, model_group, provenance)
        prediction = spec.predict(frame.iloc[test])
        points = spec.predict(frame.iloc[test], point_step=0.25)
        benchmark = np.full(len(test), np.nan)
        if policy.benchmark:
            names = fixed_components + (
                _representatives(candidates, y[fit], family, policy)
                if len(candidates.columns)
                else ()
            )
            blackbox = HistGradientBoostingClassifier(
                max_iter=50,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=policy.seed,
            )
            blackbox.fit(train[list(names)], y[fit])
            benchmark = blackbox.predict_proba(frame.iloc[test][list(names)])[:, 1]
            benchmark[
                ~np.isfinite(frame.iloc[test][list(names)].to_numpy()).any(axis=1)
            ] = np.nan
        return (
            test,
            prediction,
            points,
            benchmark,
            {
                "fold": number,
                "components": selected,
                "depth": depth,
                "n_fit_patients": len(np.unique(groups[fit])),
                "n_test_patients": len(np.unique(groups[test])),
            },
        )


def discover_binary_score(
    frame: pd.DataFrame,
    y,
    patient_ids,
    *,
    endpoint_id: str,
    model_group: str,
    family: Callable[[str], str],
    policy: SearchPolicy = SearchPolicy(),
    provenance=None,
    fixed_components: tuple[str, ...] = (),
    progress=None,
):
    """Return an independently evaluated search procedure and train-frozen formula.

    Caller MUST pass only train rows, prediction-time eligible inputs, and the
    prespecified reference-blind index cohort. Validation is a separate function.
    """
    y, groups = np.asarray(y), np.asarray(patient_ids)
    if len(frame) != len(y) or len(y) != len(groups):
        raise ValueError("unaligned_inputs")
    if pd.isna(groups).any() or frame.columns.duplicated().any():
        raise ValueError("missing_patient_or_duplicate_feature")
    frame = frame.astype(float, copy=False).replace([np.inf, -np.inf], np.nan)
    provenance = dict(provenance or {})
    if provenance.get("split") != "train" or not provenance.get("input_hash"):
        raise ValueError("explicit_train_provenance_required")
    if set(fixed_components) - set(frame) or len(set(fixed_components)) != len(
        fixed_components
    ):
        raise ValueError("invalid_fixed_components")
    if (
        fixed_components
        and provenance.get("context_available_at_decision_verified") is not True
    ):
        raise ValueError("prediction_time_context_verification_required")
    pairs = _splits(
        y, groups, policy.outer_folds, policy.seed, policy.minimum_class_patients
    )
    prediction = np.full(len(y), np.nan)
    points = prediction.copy()
    benchmark = prediction.copy()
    folds = []
    per_worker_bytes = max(256 * 1024**2, int(frame.memory_usage(deep=True).sum()) * 8)
    import os
    from .resources import recommend_dynamic_workers
    reserve_gb = float(os.environ.get("VCG_JOB_OS_RESERVE_GB", "6"))
    memory_cap = max(1, int(max(0., psutil.virtual_memory().available - reserve_gb * 1024**3) / per_worker_bytes))
    workers = min(policy.workers, len(pairs), memory_cap)
    if progress:
        progress("outer_evaluation", 0, len(pairs))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        jobs = [
            delayed(_evaluate_outer_fold)(
                frame,
                y,
                groups,
                fit,
                test,
                number,
                family,
                policy,
                fixed_components,
                endpoint_id,
                model_group,
                provenance,
            )
            for number, (fit, test) in enumerate(pairs)
        ]
        completed = 0
        worker_history = []
        while completed < len(jobs):
            # A fold is indivisible; re-admit only after the previous wave has
            # joined. Do not let a queued fold start under stale RAM capacity.
            current_cap = max(1, int(max(0., psutil.virtual_memory().available - reserve_gb * 1024**3) / per_worker_bytes))
            admitted = min(policy.workers, current_cap, len(jobs) - completed,
                recommend_dynamic_workers(policy.workers, sample_interval=.05))
            worker_history.append(admitted)
            with Parallel(n_jobs=admitted, pre_dispatch=admitted, batch_size=1,
                          max_nbytes="1M", mmap_mode="r",
                          return_as="generator_unordered") as pool:
                wave = jobs[completed:completed + admitted]
                for test, pred, pts, bench, detail in pool(wave):
                    prediction[test], points[test], benchmark[test] = pred, pts, bench
                    folds.append(detail)
                    completed += 1
                    if progress:
                        progress("outer_evaluation", completed, len(pairs))
        workers = max(worker_history)
    folds.sort(key=lambda item: item["fold"])
    with threadpool_limits(limits=1):
        candidates = frame.drop(columns=list(fixed_components))
        if progress:
            progress("train_only_final_fit", 0, 1)
        depth = (
            _choose_depth(frame, y, groups, family, policy, progress, fixed_components)
            if len(candidates.columns)
            else 0
        )
        selected = fixed_components + (
            _select(candidates, y, groups, family, policy, depth) if depth else ()
        )
        model = _fit(frame, y, selected)
    final_spec = _spec(model, selected, endpoint_id, model_group, provenance)
    kwargs = dict(iterations=policy.bootstrap_iterations, seed=policy.seed)
    if progress:
        progress("bootstrap_and_calibration", 0, 1)
    evaluation = ScoreEvaluation(
        endpoint_id,
        model_group,
        "evaluated_search_procedure",
        probability_metrics(y, prediction, groups, **kwargs),
        provenance,
    )
    return {
        "score_spec": final_spec.to_dict(),
        "evaluation": asdict(evaluation),
        "point_score": {
            "log_odds_step": 0.25,
            "metrics": probability_metrics(y, points, groups, **kwargs),
        },
        "benchmark": probability_metrics(y, benchmark, groups, **kwargs)
        if policy.benchmark
        else None,
        "folds": folds,
        "policy": asdict(policy),
        "execution": {
            "requested_workers": policy.workers,
            "effective_workers": workers,
            "admitted_workers_by_wave": worker_history,
            "os_memory_reserve_gb": reserve_gb,
            "backend": "loky" if workers > 1 else "serial",
            "native_threads_per_worker": 1,
            "estimated_bytes_per_worker": per_worker_bytes,
        },
        "oof_predictions": prediction,
        "selection_rule": "minimum_inner_log_loss_then_smallest_depth",
        "clinical_claim_allowed": False,
        "operating_points": operating_points(
            y, prediction, final_spec.prespecified_thresholds
        ),
    }


def evaluate_frozen_score(
    spec: ScoreSpec,
    validation: pd.DataFrame,
    y,
    patient_ids,
    *,
    training_patient_ids,
    provenance,
    iterations=200,
):
    if spec.link != "logit":
        raise ValueError("binary_probability_score_required")
    if provenance.get("split") != "validation" or not provenance.get("input_hash"):
        raise ValueError("explicit_validation_provenance_required")
    if set(patient_ids) & set(training_patient_ids):
        raise ValueError("patient_leakage")
    p = spec.predict(validation)
    return {
        **probability_metrics(
            np.asarray(y), p, np.asarray(patient_ids), iterations=iterations
        ),
        "operating_points": operating_points(y, p, spec.prespecified_thresholds),
    }
