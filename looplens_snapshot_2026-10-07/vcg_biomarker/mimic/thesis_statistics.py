"""Statistics for the fixed thesis versions (``THESEN_FASSUNGEN_2026-09-24.md``).

Everything here is deliberately small and explicit: a subject-cluster
bootstrap, a handful of estimators, and the verdict rule of section 0.4 of
the versions document. The thesis functions in ``thesis_tests`` combine them;
none of them chooses anything after seeing a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260924
MINIMUM_UNITS = 30

CONFIRMED = "bestaetigt"
REFUTED = "widerlegt"
UNDECIDED = "unentschieden"
NOT_EVALUABLE = "nicht_auswertbar"


@dataclass
class ThesisResult:
    thesis_id: str
    component: str
    prediction: str
    estimate: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    train_estimate: float = float("nan")
    validation_estimate: float = float("nan")
    n_units: int = 0
    n_subjects: int = 0
    verdict: str = NOT_EVALUABLE
    reason: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        record = {
            "thesis_id": self.thesis_id,
            "component": self.component,
            "prediction": self.prediction,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "train_estimate": self.train_estimate,
            "validation_estimate": self.validation_estimate,
            "n_units": int(self.n_units),
            "n_subjects": int(self.n_subjects),
            "verdict": self.verdict,
            "reason": self.reason,
        }
        for key, value in self.extra.items():
            record[f"extra_{key}"] = value
        return record


def not_evaluable(thesis_id: str, component: str, prediction: str, reason: str, **extra) -> ThesisResult:
    return ThesisResult(thesis_id, component, prediction, verdict=NOT_EVALUABLE, reason=reason, extra=extra)


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster: str = "subject_id",
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, np.ndarray]:
    """Percentile interval of ``statistic`` under resampling of clusters."""

    if frame.empty or iterations <= 0:
        return float("nan"), float("nan"), np.empty(0)
    codes, _ = pd.factorize(frame[cluster], sort=False)
    order = np.argsort(codes, kind="mergesort")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, boundaries)
    n_groups = len(groups)
    if n_groups < 2:
        return float("nan"), float("nan"), np.empty(0)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(iterations)):
        chosen = rng.integers(0, n_groups, size=n_groups)
        index = np.concatenate([groups[item] for item in chosen])
        try:
            value = float(statistic(frame.iloc[index]))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            continue
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < max(10, iterations // 2):
        return float("nan"), float("nan"), np.asarray(draws)
    values = np.asarray(draws)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), values


def ols_coefficient(
    frame: pd.DataFrame, outcome: str, exposure: str, covariates: Sequence[str] = ()
) -> float:
    """Coefficient of ``exposure`` in an OLS model with intercept."""

    columns = [exposure, *covariates]
    data = frame[[outcome, *columns]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < len(columns) + 2:
        return float("nan")
    design = np.column_stack([np.ones(len(data)), data[columns].to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, data[outcome].to_numpy(dtype=float), rcond=None)
    return float(coefficients[1])


def spearman(frame: pd.DataFrame, a: str, b: str) -> float:
    data = frame[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3 or data[a].nunique() < 2 or data[b].nunique() < 2:
        return float("nan")
    return float(data[a].rank().corr(data[b].rank()))


def mean_difference(frame: pd.DataFrame, value: str, group: str, first, second) -> float:
    numeric = pd.to_numeric(frame[value], errors="coerce")
    a = numeric[frame[group].eq(first)].dropna()
    b = numeric[frame[group].eq(second)].dropna()
    if a.empty or b.empty:
        return float("nan")
    return float(a.mean() - b.mean())


def median_difference(frame: pd.DataFrame, value: str, group: str, first, second) -> float:
    numeric = pd.to_numeric(frame[value], errors="coerce")
    a = numeric[frame[group].eq(first)].dropna()
    b = numeric[frame[group].eq(second)].dropna()
    if a.empty or b.empty:
        return float("nan")
    return float(a.median() - b.median())


def auc(values: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC of ``values`` for label 1 over label 0."""

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    mask = np.isfinite(values)
    values, labels = values[mask], labels[mask]
    positive, negative = values[labels == 1], values[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([positive, negative])).rank().to_numpy()
    return float((ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def deming_slope(x: np.ndarray, y: np.ndarray, variance_ratio: float = 1.0) -> float:
    """Deming regression slope; ``variance_ratio`` = var(error y) / var(error x)."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")
    sxx = float(np.var(x, ddof=1))
    syy = float(np.var(y, ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    if abs(sxy) < 1e-15:
        return float("nan")
    delta = float(variance_ratio)
    return float((syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy))


def split_estimates(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float]) -> tuple[float, float]:
    def safe(part: pd.DataFrame) -> float:
        if part.empty:
            return float("nan")
        try:
            return float(statistic(part))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return float("nan")

    split = frame["split"].astype(str) if "split" in frame else pd.Series("", index=frame.index)
    return safe(frame.loc[split.eq("train")]), safe(frame.loc[split.eq("validation")])


def directional_result(
    thesis_id: str,
    component: str,
    prediction: str,
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    direction: int,
    null: float = 0.0,
    refuted: Optional[Callable[[float, float, float], bool]] = None,
    minimum_units: int = MINIMUM_UNITS,
    unit_count: Optional[int] = None,
    iterations: int = BOOTSTRAP_ITERATIONS,
    extra: Optional[dict] = None,
) -> ThesisResult:
    """Estimate, interval, split replication and the verdict of section 0.4.

    ``refuted(estimate, low, high)`` implements the version's refutation
    criterion; without it the refutation is "interval entirely on the wrong
    side of the null".
    """

    units = int(unit_count if unit_count is not None else len(frame))
    subjects = int(frame["subject_id"].nunique()) if "subject_id" in frame and len(frame) else 0
    if units < minimum_units:
        return not_evaluable(thesis_id, component, prediction,
                             f"{units} units, minimum {minimum_units}", **(extra or {}))
    estimate = float(statistic(frame))
    low, high, _ = cluster_bootstrap(frame, statistic, iterations=iterations)
    train, validation = split_estimates(frame, statistic)
    result = ThesisResult(thesis_id, component, prediction, estimate, low, high, train,
                          validation, units, subjects, extra=dict(extra or {}))
    if not np.isfinite(estimate):
        result.verdict, result.reason = NOT_EVALUABLE, "estimate not finite"
        return result
    wrong_side = (
        np.isfinite(low) and np.isfinite(high)
        and (high < null if direction > 0 else low > null)
    )
    if (refuted(estimate, low, high) if refuted is not None else wrong_side):
        result.verdict = REFUTED
        return result
    excludes_null = (
        np.isfinite(low) and np.isfinite(high)
        and (low > null if direction > 0 else high < null)
    )
    replicated = all(
        np.isfinite(value) and np.sign(value - null) == np.sign(direction)
        for value in (train, validation)
    )
    if excludes_null and replicated:
        result.verdict = CONFIRMED
    else:
        result.verdict = UNDECIDED
        if excludes_null and not replicated:
            result.reason = "interval excludes the null but the splits disagree"
    return result


def threshold_at_specificity(control_values: np.ndarray, specificity: float = 0.95) -> float:
    """Cut-off above which a value is positive, at the given specificity."""

    values = np.asarray(control_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, specificity))


def paired_sensitivity_difference(
    frame: pd.DataFrame, test_flag: str, standard_flag: str
) -> float:
    """Sensitivity(test) - sensitivity(standard) among cases (label == 1)."""

    cases = frame.loc[frame["label"].eq(1)]
    both = cases[[test_flag, standard_flag]].dropna()
    if both.empty:
        return float("nan")
    return float(both[test_flag].astype(float).mean() - both[standard_flag].astype(float).mean())


def mcnemar_p(frame: pd.DataFrame, test_flag: str, standard_flag: str) -> float:
    from scipy import stats

    cases = frame.loc[frame["label"].eq(1), [test_flag, standard_flag]].dropna().astype(bool)
    b = int((cases[test_flag] & ~cases[standard_flag]).sum())
    c = int((~cases[test_flag] & cases[standard_flag]).sum())
    if b + c == 0:
        return 1.0
    return float(stats.binomtest(b, b + c, 0.5).pvalue)


def logistic_auc_increment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    outcome: str,
    base: Sequence[str],
    added: Sequence[str],
) -> float:
    """Validation AUC(base + added) - AUC(base), models fitted on train only."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    columns = [*base, *added]

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[[outcome, *columns]].apply(pd.to_numeric, errors="coerce").dropna()

    fit_rows, test_rows = prepare(train), prepare(validation)
    if fit_rows[outcome].nunique() < 2 or test_rows[outcome].nunique() < 2:
        return float("nan")

    def score(features: Sequence[str]) -> float:
        scaler = StandardScaler().fit(fit_rows[list(features)])
        model = LogisticRegression(max_iter=1000).fit(
            scaler.transform(fit_rows[list(features)]), fit_rows[outcome].astype(int)
        )
        predicted = model.predict_proba(scaler.transform(test_rows[list(features)]))[:, 1]
        return auc(predicted, test_rows[outcome].astype(int).to_numpy())

    return score(columns) - score(base)


def fold_angle(values: pd.Series) -> pd.Series:
    """Wrap a difference of angles to -180..180 degrees."""

    return ((pd.to_numeric(values, errors="coerce") + 180.0) % 360.0) - 180.0


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise angle in degrees between vectors of shape (n, 3)."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.sum(a * b, axis=1)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.clip(dot / norm, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    angle[~np.isfinite(norm) | (norm <= 0)] = np.nan
    return angle
