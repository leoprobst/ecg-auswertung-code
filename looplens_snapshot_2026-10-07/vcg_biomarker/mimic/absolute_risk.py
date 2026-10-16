"""Absolute risk under competing events, and whether it is calibrated.

The survival arm estimates cause-specific hazards and says plainly that they
are not cumulative incidence. That is correct — and it leaves a hole. The
project's own evidence ladder requires a paired decision curve for any claim
above "clinically associated", and net benefit is a function of *absolute*
risk at a threshold. Until now nothing in the pipeline could produce one:
``clinical_utility.py`` has an ``outputs_absolute_risk`` flag and no estimator
behind it.

Two things are needed and both are here:

* **Cumulative incidence in the presence of competing events.** The naive
  ``1 - Kaplan-Meier`` treats a competing death as censoring, i.e. as if the
  patient could still have the event afterwards. In an elderly, acutely ill
  cohort that overestimates risk, sometimes badly. The Aalen-Johansen
  estimator weights each cause-specific hazard by the probability of still
  being event-free, and is what "absolute risk" has to mean here.

* **Calibration of predicted against observed risk.** A discriminating model
  can be systematically wrong about level, and net benefit depends on level,
  not on ranking. Calibration is therefore a precondition for a decision
  curve, not an optional extra.

Everything is estimated on development splits only. Nothing here authorises a
clinical claim; it makes one *assessable*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


ABSOLUTE_RISK_VERSION = "1.1.0"

# Event codes for the competing-risk representation.
EVENT_CENSORED = 0
EVENT_OF_INTEREST = 1
EVENT_COMPETING = 2

# A calibration slope this far from 1.0, or an intercept this far from 0,
# means the predicted level cannot carry a decision-curve claim.
CALIBRATION_SLOPE_TOLERANCE = 0.25
CALIBRATION_INTERCEPT_TOLERANCE = 0.25


@dataclass(frozen=True)
class CumulativeIncidenceCurve:
    times: np.ndarray
    cause_incidence: np.ndarray
    competing_incidence: np.ndarray
    event_free: np.ndarray
    n_at_risk: np.ndarray
    variance: np.ndarray

    def _index_at(self, horizon: float) -> int:
        """Index of the last event time at or before ``horizon``."""

        if len(self.times) == 0:
            return -1
        return int(np.searchsorted(self.times, float(horizon), side="right") - 1)

    def risk_at(self, horizon: float) -> float:
        """Absolute risk of the event of interest by ``horizon``."""

        if len(self.times) == 0:
            return float("nan")
        index = self._index_at(horizon)
        if index < 0:
            return 0.0
        return float(self.cause_incidence[index])

    def competing_risk_at(self, horizon: float) -> float:
        if len(self.times) == 0:
            return float("nan")
        index = self._index_at(horizon)
        if index < 0:
            return 0.0
        return float(self.competing_incidence[index])

    def confidence_interval_at(
        self,
        horizon: float,
        *,
        z: float = 1.959963984540054,
    ) -> tuple[float, float]:
        """Pointwise interval on the complementary log-log scale.

        The plain Wald interval on the probability scale routinely leaves
        [0, 1] for the small risks this cohort produces.
        """

        if len(self.times) == 0:
            return (float("nan"), float("nan"))
        index = self._index_at(horizon)
        if index < 0:
            return (0.0, 0.0)
        risk = float(self.cause_incidence[index])
        variance = float(self.variance[index])
        if risk <= 0.0 or risk >= 1.0 or not np.isfinite(variance) or variance <= 0:
            return (risk, risk)
        log_risk = np.log(risk)
        se_cloglog = np.sqrt(variance) / (risk * abs(log_risk))
        half = z * se_cloglog
        lower = risk ** np.exp(half)
        upper = risk ** np.exp(-half)
        return (float(min(lower, upper)), float(max(lower, upper)))


def aalen_johansen(
    duration: Iterable[float],
    event: Iterable[int],
    *,
    cause: int = EVENT_OF_INTEREST,
) -> CumulativeIncidenceCurve:
    """Non-parametric cumulative incidence with competing events.

    ``event`` uses 0 for censored, 1 for the event of interest and 2 for a
    competing event. Variance follows the Aalen delta-method approximation,
    which is adequate for the interval widths reported here.
    """

    durations = np.asarray(list(duration), dtype=float)
    events = np.asarray(list(event), dtype=int)
    if durations.shape != events.shape:
        raise ValueError("Duration and event arrays must have equal length")
    valid = np.isfinite(durations) & (durations >= 0)
    durations = durations[valid]
    events = events[valid]
    if len(durations) == 0:
        empty: np.ndarray = np.empty(0, dtype=float)
        return CumulativeIncidenceCurve(
            empty, empty, empty, empty, empty.astype(int), empty
        )
    order = np.argsort(durations, kind="mergesort")
    durations = durations[order]
    events = events[order]
    event_rows = events != EVENT_CENSORED
    event_durations = durations[event_rows]
    event_codes = events[event_rows]
    unique_times, event_time_index = np.unique(
        event_durations,
        return_inverse=True,
    )
    if len(unique_times) == 0:
        empty = np.empty(0, dtype=float)
        return CumulativeIncidenceCurve(
            empty, empty, empty, empty, empty.astype(int), empty
        )
    event_group_count = len(unique_times)
    d_cause = np.bincount(
        event_time_index,
        weights=(event_codes == cause),
        minlength=event_group_count,
    ).astype(float)
    d_competing = np.bincount(
        event_time_index,
        weights=(event_codes != cause),
        minlength=event_group_count,
    ).astype(float)
    # The stable sort above makes this vectorized search exact even with
    # censoring tied to an event time: all people at that time are at risk
    # immediately before its events, matching the original definition.
    at_risk: np.ndarray = np.asarray(
        len(durations)
        - np.searchsorted(
            durations,
            unique_times,
            side="left",
        ),
        dtype=np.int64,
    )
    if np.any(at_risk <= 0):  # pragma: no cover - defensive invariant
        raise RuntimeError("Aalen-Johansen produced a non-positive risk set")
    d_total = d_cause + d_competing
    hazard_total = d_total / at_risk
    survival_after = np.cumprod(1.0 - hazard_total)
    survival_before = np.concatenate(
        (np.ones(1, dtype=float), survival_after[:-1])
    )
    # Increment uses the probability of still being event-free *before* this
    # time, which is precisely what 1 - Kaplan-Meier gets wrong.
    cause_values = np.cumsum(survival_before * d_cause / at_risk)
    competing_values = np.cumsum(
        survival_before * d_competing / at_risk
    )
    variance_increment: np.ndarray = np.zeros(
        event_group_count,
        dtype=float,
    )
    variance_valid = (d_cause > 0) & (at_risk > d_cause)
    variance_increment[variance_valid] = (
        (survival_before[variance_valid] ** 2)
        * d_cause[variance_valid]
        * (at_risk[variance_valid] - d_cause[variance_valid])
        / (at_risk[variance_valid] ** 3)
    )
    variances = np.cumsum(variance_increment)
    return CumulativeIncidenceCurve(
        times=np.asarray(unique_times, dtype=float),
        cause_incidence=np.asarray(cause_values, dtype=float),
        competing_incidence=np.asarray(competing_values, dtype=float),
        event_free=np.asarray(survival_after, dtype=float),
        n_at_risk=np.asarray(at_risk, dtype=int),
        variance=np.asarray(variances, dtype=float),
    )


def kaplan_meier_complement(
    duration: Iterable[float],
    event: Iterable[int],
    *,
    cause: int = EVENT_OF_INTEREST,
) -> CumulativeIncidenceCurve:
    """``1 - KM`` treating competing events as censoring — the biased version.

    Provided so the bias can be *shown* rather than asserted: reporting both
    next to each other makes the size of the competing-risk distortion visible
    in the cohort at hand.
    """

    durations = np.asarray(list(duration), dtype=float)
    events = np.asarray(list(event), dtype=int)
    censored = np.where(events == cause, EVENT_OF_INTEREST, EVENT_CENSORED)
    curve = aalen_johansen(durations, censored, cause=EVENT_OF_INTEREST)
    # With no competing events the Aalen-Johansen estimator reduces exactly to
    # 1 - KM, so this is the same routine with the competing events removed.
    return curve


def stratified_absolute_risk(
    frame: pd.DataFrame,
    *,
    duration_column: str,
    event_column: str,
    stratum_column: str,
    horizons: Sequence[float],
) -> pd.DataFrame:
    """Absolute risk by marker stratum, with the naive comparison alongside."""

    required = {duration_column, event_column, stratum_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Absolute-risk frame is missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for stratum, group in frame.groupby(stratum_column, sort=True):
        durations = pd.to_numeric(group[duration_column], errors="coerce")
        events = pd.to_numeric(group[event_column], errors="coerce")
        curve = aalen_johansen(durations, events.fillna(0).astype(int))
        naive = kaplan_meier_complement(
            durations,
            events.fillna(0).astype(int),
        )
        for horizon in horizons:
            risk = curve.risk_at(horizon)
            low, high = curve.confidence_interval_at(horizon)
            naive_risk = naive.risk_at(horizon)
            rows.append(
                {
                    "stratum": str(stratum),
                    "horizon": float(horizon),
                    "n": int(len(group)),
                    "n_events": int((events == EVENT_OF_INTEREST).sum()),
                    "n_competing": int((events == EVENT_COMPETING).sum()),
                    "absolute_risk": risk,
                    "absolute_risk_ci_low": low,
                    "absolute_risk_ci_high": high,
                    "competing_risk": curve.competing_risk_at(horizon),
                    "naive_one_minus_km": naive_risk,
                    "competing_risk_bias": (
                        float(naive_risk - risk)
                        if np.isfinite(naive_risk) and np.isfinite(risk)
                        else float("nan")
                    ),
                    "estimator": "aalen_johansen",
                    "absolute_risk_version": ABSOLUTE_RISK_VERSION,
                }
            )
    return pd.DataFrame(rows)


def calibration_by_risk_group(
    predicted_risk: Iterable[float],
    duration: Iterable[float],
    event: Iterable[int],
    *,
    horizon: float,
    n_groups: int = 10,
) -> pd.DataFrame:
    """Observed versus predicted risk per predicted-risk decile.

    Observed risk is the Aalen-Johansen estimate inside the group, not the
    crude event proportion: censoring before the horizon would otherwise make
    every group look safer than it is.
    """

    predicted = np.asarray(list(predicted_risk), dtype=float)
    durations = np.asarray(list(duration), dtype=float)
    events = np.asarray(list(event), dtype=int)
    valid = np.isfinite(predicted) & np.isfinite(durations)
    predicted = predicted[valid]
    durations = durations[valid]
    events = events[valid]
    if len(predicted) < n_groups * 5:
        return pd.DataFrame(
            columns=[
                "risk_group",
                "n",
                "mean_predicted_risk",
                "observed_risk",
                "observed_ci_low",
                "observed_ci_high",
            ]
        )
    quantiles = np.quantile(predicted, np.linspace(0, 1, int(n_groups) + 1))
    quantiles[0] -= 1e-9
    groups = np.digitize(predicted, quantiles[1:-1], right=True)
    rows: list[dict[str, object]] = []
    for group_index in range(int(n_groups)):
        mask = groups == group_index
        if mask.sum() < 5:
            continue
        curve = aalen_johansen(durations[mask], events[mask])
        low, high = curve.confidence_interval_at(horizon)
        rows.append(
            {
                "risk_group": int(group_index) + 1,
                "n": int(mask.sum()),
                "mean_predicted_risk": float(np.mean(predicted[mask])),
                "observed_risk": curve.risk_at(horizon),
                "observed_ci_low": low,
                "observed_ci_high": high,
                "horizon": float(horizon),
                "absolute_risk_version": ABSOLUTE_RISK_VERSION,
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(table: pd.DataFrame) -> dict[str, object]:
    """Calibration slope, intercept and the gate a decision curve depends on.

    Fitted on the complementary log-log scale, which is the natural scale for
    a risk and avoids the boundary problems of a plain linear fit near 0.
    """

    if table.empty or len(table) < 3:
        return {
            "status": "not_evaluable",
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_adequate_for_decision_curve": False,
        }
    predicted = pd.to_numeric(
        table["mean_predicted_risk"],
        errors="coerce",
    ).to_numpy(dtype=float)
    observed = pd.to_numeric(
        table["observed_risk"],
        errors="coerce",
    ).to_numpy(dtype=float)
    usable = (
        np.isfinite(predicted)
        & np.isfinite(observed)
        & (predicted > 0)
        & (predicted < 1)
        & (observed > 0)
        & (observed < 1)
    )
    if usable.sum() < 3:
        return {
            "status": "not_evaluable",
            "failure_reason": "degenerate_risk_values",
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_adequate_for_decision_curve": False,
        }

    def cloglog(values: np.ndarray) -> np.ndarray:
        return np.log(-np.log(1.0 - values))

    x = cloglog(predicted[usable])
    y = cloglog(observed[usable])
    slope, intercept = np.polyfit(x, y, 1)
    mean_predicted = float(np.mean(predicted[usable]))
    mean_observed = float(np.mean(observed[usable]))
    adequate = bool(
        abs(slope - 1.0) <= CALIBRATION_SLOPE_TOLERANCE
        and abs(intercept) <= CALIBRATION_INTERCEPT_TOLERANCE
    )
    return {
        "status": "evaluable",
        "n_groups": int(usable.sum()),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_in_the_large": float(mean_observed - mean_predicted),
        "mean_predicted_risk": mean_predicted,
        "mean_observed_risk": mean_observed,
        "max_absolute_deviation": float(
            np.max(np.abs(observed[usable] - predicted[usable]))
        ),
        "scale": "complementary_log_log",
        "calibration_adequate_for_decision_curve": adequate,
        "gate_reason": (
            ""
            if adequate
            else "predicted level is off; net benefit at a threshold would be "
            "computed from a miscalibrated risk"
        ),
        "absolute_risk_version": ABSOLUTE_RISK_VERSION,
    }


def net_benefit_from_absolute_risk(
    predicted_risk: Iterable[float],
    duration: Iterable[float],
    event: Iterable[int],
    *,
    horizon: float,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Decision-curve net benefit for time-to-event data with competing risks.

    Uses the Aalen-Johansen risk inside the treated group rather than a crude
    proportion, so censoring before the horizon does not inflate net benefit.
    ``treat all`` is likewise computed from the cohort's own absolute risk.
    """

    predicted = np.asarray(list(predicted_risk), dtype=float)
    durations = np.asarray(list(duration), dtype=float)
    events = np.asarray(list(event), dtype=int)
    valid = np.isfinite(predicted) & np.isfinite(durations)
    predicted = predicted[valid]
    durations = durations[valid]
    events = events[valid]
    total = len(predicted)
    if total == 0:
        return pd.DataFrame(
            columns=[
                "threshold",
                "net_benefit_model",
                "net_benefit_treat_all",
                "net_benefit_treat_none",
            ]
        )
    overall = aalen_johansen(durations, events).risk_at(horizon)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0.0 < threshold < 1.0:
            continue
        treated = predicted >= threshold
        n_treated = int(treated.sum())
        if n_treated == 0:
            model_benefit = 0.0
        else:
            risk_treated = aalen_johansen(
                durations[treated],
                events[treated],
            ).risk_at(horizon)
            true_positive = risk_treated * n_treated / total
            false_positive = (1.0 - risk_treated) * n_treated / total
            model_benefit = float(
                true_positive
                - false_positive * threshold / (1.0 - threshold)
            )
        treat_all = float(
            overall - (1.0 - overall) * threshold / (1.0 - threshold)
        )
        rows.append(
            {
                "threshold": threshold,
                "n_treated": n_treated,
                "net_benefit_model": model_benefit,
                "net_benefit_treat_all": treat_all,
                "net_benefit_treat_none": 0.0,
                "advantage_over_treat_all": model_benefit - treat_all,
                "advantage_over_treat_none": model_benefit,
                "horizon": float(horizon),
                "risk_estimator": "aalen_johansen",
                "absolute_risk_version": ABSOLUTE_RISK_VERSION,
            }
        )
    return pd.DataFrame(rows)


def absolute_risk_report(
    frame: pd.DataFrame,
    *,
    predicted_risk_column: str,
    duration_column: str,
    event_column: str,
    horizon: float,
    thresholds: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30),
) -> Mapping[str, object]:
    """Everything a decision-curve claim needs, in one record."""

    predicted = frame[predicted_risk_column]
    durations = frame[duration_column]
    events = frame[event_column]
    calibration_table = calibration_by_risk_group(
        predicted,
        durations,
        events,
        horizon=horizon,
    )
    calibration = calibration_metrics(calibration_table)
    benefit = net_benefit_from_absolute_risk(
        predicted,
        durations,
        events,
        horizon=horizon,
        thresholds=thresholds,
    )
    curve = aalen_johansen(
        pd.to_numeric(durations, errors="coerce"),
        pd.to_numeric(events, errors="coerce").fillna(0).astype(int),
    )
    return {
        "horizon": float(horizon),
        "cohort_absolute_risk": curve.risk_at(horizon),
        "cohort_competing_risk": curve.competing_risk_at(horizon),
        "calibration": calibration,
        "calibration_table": calibration_table,
        "net_benefit": benefit,
        # The gate the evidence ladder actually depends on.
        "decision_curve_claim_permitted": bool(
            calibration.get("calibration_adequate_for_decision_curve", False)
        ),
        "clinical_use_authorized": False,
        "absolute_risk_version": ABSOLUTE_RISK_VERSION,
    }


__all__ = [
    "ABSOLUTE_RISK_VERSION",
    "CALIBRATION_INTERCEPT_TOLERANCE",
    "CALIBRATION_SLOPE_TOLERANCE",
    "CumulativeIncidenceCurve",
    "EVENT_CENSORED",
    "EVENT_COMPETING",
    "EVENT_OF_INTEREST",
    "aalen_johansen",
    "absolute_risk_report",
    "calibration_by_risk_group",
    "calibration_metrics",
    "kaplan_meier_complement",
    "net_benefit_from_absolute_risk",
    "stratified_absolute_risk",
]
