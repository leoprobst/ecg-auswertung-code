"""Development survival analyses with the ECG as time origin.

The primary analysis uses an outcome-blind index ECG per patient. Repeated
ECGs are a separately labelled sensitivity analysis: cluster-robust variance
does not eliminate informative recording frequency, time-varying confounding,
or repeated attribution of the same death. Agreement between the analyses is
supportive, not proof that these biases are absent.

Cox models use Efron handling of ties. Proportional hazards over time and a
linear marker term on the log-hazard scale are distinct assumptions. Neither
is established by monotonic tertile curves. Group boundaries are fixed in
training; curves are evaluated only in validation, without fallback to pooled
or validation-derived boundaries. Unknown/locked splits are rejected.

Censoring and death ascertainment must be justified by the actual cohort and
linkage coverage. Incomplete ascertainment does not necessarily bias an effect
toward the null. A low raw-marker concordance can reflect inverse orientation,
noise, bias, or instability; it does not by itself establish protection.

Historical numerical examples were removed from this module documentation:
without a run identifier, feature generation, input provenance and independent
validation they must not be presented as current replicated results. Use the
versioned run artifacts and their explicit limitations for numerical claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from .panels import STUDY_PANEL_IDS


SURVIVAL_VERSION = "1.8.0"  # audit 5.3.2 Uno C, time-dependent AUC, landmark
# Audit T1: MIMIC-IV links state death records up to exactly one year after a
# patient's last hospital discharge.
REGISTRY_WINDOW_DAYS = 365.0
CENSORING_REGISTRY_WINDOW = "registry_window"
CENSORING_LAST_CONTACT = "last_contact"

# Follow-up beyond which the linkage stops being informative. MIMIC's date of
# death comes from state records with a limited window; a patient still alive
# at the end of it is censored, not immortal.
DEFAULT_HORIZON_DAYS = 365.0

# Newton-Raphson is being asked to maximise a concave function, so it converges
# in a handful of steps or something is wrong with the design matrix.
MAX_NEWTON_STEPS = 40
CONVERGENCE_TOLERANCE = 1e-9


class ConvergenceFailure(RuntimeError):
    """Raised when the partial likelihood did not converge.

    Loud rather than silent: a Cox fit that stopped early still returns
    plausible-looking coefficients, and nothing downstream would notice.
    """


@dataclass(frozen=True)
class CoxFit:
    """A fitted proportional-hazards model and how far it can be trusted."""

    covariates: tuple[str, ...]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    variance: np.ndarray
    log_likelihood: float
    n_observations: int
    n_events: int
    n_clusters: int
    steps: int
    robust: bool
    ties: str = "efron"

    @property
    def hazard_ratios(self) -> np.ndarray:
        return np.exp(self.coefficients)

    @property
    def z_scores(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.coefficients / self.standard_errors

    @property
    def p_values(self) -> np.ndarray:
        from scipy import stats

        return 2.0 * stats.norm.sf(np.abs(self.z_scores))

    def confidence_interval(self, level: float = 0.95) -> np.ndarray:
        from scipy import stats

        z = float(stats.norm.ppf(0.5 + level / 2.0))
        lower = self.coefficients - z * self.standard_errors
        upper = self.coefficients + z * self.standard_errors
        return np.exp(np.column_stack([lower, upper]))

    def to_frame(self, level: float = 0.95) -> pd.DataFrame:
        interval = self.confidence_interval(level)
        return pd.DataFrame(
            {
                "covariate": list(self.covariates),
                "coefficient": self.coefficients,
                "hazard_ratio": self.hazard_ratios,
                "hr_lower": interval[:, 0],
                "hr_upper": interval[:, 1],
                "standard_error": self.standard_errors,
                "z": self.z_scores,
                "p_value": self.p_values,
            }
        )

    def to_dict(self) -> dict:
        return {
            "version": SURVIVAL_VERSION,
            "n_observations": int(self.n_observations),
            "n_events": int(self.n_events),
            "n_clusters": int(self.n_clusters),
            "robust_variance": bool(self.robust),
            "ties": str(self.ties),
            "steps": int(self.steps),
            "log_likelihood": float(self.log_likelihood),
        }


def survival_frame_query(
    master_sql: str,
    patients_sql: str,
    admissions_sql: str,
    feature_clause: str,
    *,
    marker_columns: str,
    death_column: str = "date_of_death",
    splits: Sequence[str] = ("train", "validation"),
) -> str:
    """One row per ECG, with the clock starting at the recording.

    ``death_column`` is a parameter because the standardized copy of
    ``patients`` renames MIMIC's ``dod`` to ``date_of_death``. Hard-coding
    either name would tie this arm to one of the two copies, and the whole
    point of reading the standardized layer is that every stage sees the same
    table.

    Censoring is the part that has to be right. A patient without a date of
    death is not immortal -- they simply stop being observed -- so follow-up
    ends at their last recorded hospital discharge. Treating them as alive to
    the horizon would count every lost patient as a survivor and pull every
    hazard ratio towards one.

    ``ecg_rank`` is carried so the caller can take the first ECG per patient
    without a second pass.
    """

    split_list = ", ".join(f"'{value}'" for value in splits)
    return f"""
        WITH patient AS (
            SELECT subject_id, {death_column} AS dod
            FROM read_parquet('{patients_sql}')
        ),
        contact AS (
            SELECT subject_id, max(dischtime) AS last_seen
            FROM read_parquet('{admissions_sql}')
            GROUP BY subject_id
        ),
        last_ecg AS (
            SELECT subject_id, max(ecg_time) AS last_ecg_time
            FROM read_parquet('{master_sql}')
            GROUP BY subject_id
        ),
        base AS (
            SELECT
                m.subject_id, m.study_id, m.split, m.ecg_time, m.age,
                CASE
                    WHEN lower(m.sex) LIKE 'f%' THEN 1
                    WHEN lower(m.sex) LIKE 'm%' THEN 0
                    ELSE NULL
                END AS female,
                p.dod, c.last_seen, e.last_ecg_time
            FROM read_parquet('{master_sql}') m
            LEFT JOIN patient p ON p.subject_id = m.subject_id
            LEFT JOIN contact c ON c.subject_id = m.subject_id
            LEFT JOIN last_ecg e ON e.subject_id = m.subject_id
            WHERE m.ecg_time IS NOT NULL AND m.split IN ({split_list})
        ),
        -- Audit 5.3.2: discharge of the ECG's own admission, for the
        -- landmark population that starts the clock at discharge.
        index_discharge AS (
            SELECT
                b.subject_id, b.study_id,
                DATE_DIFF(
                    'day', CAST(b.ecg_time AS DATE), CAST(ia.dischtime AS DATE)
                ) AS days_to_index_discharge
            FROM base b
            JOIN read_parquet('{master_sql}') mi
              ON mi.subject_id = b.subject_id AND mi.study_id = b.study_id
            JOIN read_parquet('{admissions_sql}') ia
              ON ia.hadm_id = mi.hadm_id AND ia.subject_id = mi.subject_id
        )
        SELECT
            b.subject_id, b.study_id, b.split, b.age, b.female, b.ecg_time,
            CASE WHEN b.dod IS NOT NULL THEN 1 ELSE 0 END AS died,
            DATE_DIFF('day', CAST(b.ecg_time AS DATE), CAST(b.dod AS DATE))
                AS days_to_death,
            DATE_DIFF('day', CAST(b.ecg_time AS DATE), CAST(b.last_seen AS DATE))
                AS days_to_last_seen,
            CASE WHEN b.last_seen IS NULL THEN 0 ELSE 1 END AS ever_admitted,
            DATE_DIFF(
                'day',
                CAST(b.ecg_time AS DATE),
                CAST(GREATEST(b.last_seen, b.last_ecg_time) AS DATE)
            ) AS days_to_last_contact,
            row_number() OVER (
                PARTITION BY b.subject_id ORDER BY b.ecg_time, b.study_id
            ) AS ecg_rank,
            idx.days_to_index_discharge,
            {marker_columns}
        FROM base b
        LEFT JOIN index_discharge idx
          ON idx.subject_id = b.subject_id AND idx.study_id = b.study_id
        JOIN {feature_clause} f
          ON f.subject_id = b.subject_id AND f.study_id = b.study_id
        WHERE f.extraction_status = 'ok'
    """


def follow_up(
    frame: pd.DataFrame,
    *,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    death_column: str = "days_to_death",
    contact_column: str = "days_to_last_seen",
    died_column: str = "died",
    censoring: str = CENSORING_REGISTRY_WINDOW,
    last_contact_column: str = "days_to_last_contact",
    admitted_column: str = "ever_admitted",
) -> pd.DataFrame:
    """Turn the raw dates into a (time, event) pair, censored at the horizon.

    Audit T1. Deaths come from hospital and state registry records; the
    registry covers a patient up to exactly one year after the last hospital
    discharge. The earlier rule censored survivors at that discharge but
    counted every later registry death as an event -- 67 % of the deaths lay
    after the last discharge, so the risk set filled with patients whose
    death would be registered, and 29,069 survivors without an admission or
    with the admission before the ECG dropped out while their deceased
    counterparts stayed. Two consistent rules replace it:

    ``registry_window`` (primary): ascertainment is the registry window for
    both outcomes. Survivors are censored at last discharge + 365 days (or
    the horizon); patients never admitted have no documented registry
    coverage and are excluded whether they died or not.

    ``last_contact`` (sensitivity): follow-up ends at the last hospital
    contact (discharge or ECG) for everyone; a death after it is not an
    event.

    A death recorded *before* the ECG is a linkage error and the patient is
    dropped entirely, not clamped and not censored.

    The first version said that and did something else. ``observed`` required
    ``death >= 0``, so a negative death offset fell through to the censoring
    branch and the patient survived as censored at their last contact --
    silently contributing person-time on the strength of a date known to be
    wrong. The test written for it only covered the case where the contact was
    negative too, so it passed. ``n_dropped_negative_followup`` in the stage
    summary now says how many rows this removes, because a linkage error that
    disappears without a count is the same problem one level down.
    """

    result = frame.copy()
    death = pd.to_numeric(result.get(death_column), errors="coerce")
    contact = pd.to_numeric(result.get(contact_column), errors="coerce")
    died = result.get(died_column, pd.Series(0, index=result.index)).fillna(0)
    horizon = float(horizon_days)

    # A death date before the recording makes the whole row unusable: it is
    # not evidence of survival either.
    corrupt = died.astype(bool) & death.notna() & death.lt(0)
    result = result.loc[~corrupt]
    death = death.loc[~corrupt]
    contact = contact.loc[~corrupt]
    died = died.loc[~corrupt]

    observed = died.astype(bool) & death.notna()
    if censoring == CENSORING_REGISTRY_WINDOW:
        admitted = (
            pd.to_numeric(result[admitted_column], errors="coerce").fillna(0).astype(bool)
            if admitted_column in result
            else contact.notna()
        )
        window_end = contact + REGISTRY_WINDOW_DAYS
        # DuckDB returns nullable integer day offsets. A missing coverage end
        # makes comparison NA, whereas the equivalent float/NaN comparison
        # is False. Treat it as outside the window; missing follow-up time is
        # excluded below, preserving the existing censoring rule.
        in_window = (observed & death.le(window_end)).fillna(False)
        end = death.where(in_window, window_end)
        result["event"] = (in_window & death.le(horizon)).astype(int)
        result["time_days"] = np.minimum(end, horizon)
        keep = admitted & result["time_days"].notna() & result["time_days"].gt(0)
        return result.loc[keep]
    if censoring == CENSORING_LAST_CONTACT:
        last_contact = (
            pd.to_numeric(result[last_contact_column], errors="coerce")
            if last_contact_column in result
            else contact
        )
        in_contact = (observed & death.le(last_contact)).fillna(False)
        end = death.where(in_contact, last_contact)
        result["event"] = (in_contact & death.le(horizon)).astype(int)
        result["time_days"] = np.minimum(end, horizon)
        return result.loc[result["time_days"].notna() & result["time_days"].gt(0)]
    raise ValueError(f"Unknown censoring rule: {censoring}")


def _risk_set_blocks(time: np.ndarray, event: np.ndarray):
    """Distinct event times, descending, with the rows entering at each.

    Sorted ascending, the risk set at the j-th distinct time is everything
    from its first index onwards, so walking the times backwards lets one
    running total serve every risk set -- which is what keeps this O(n) rather
    than O(n * distinct times).
    """

    order = np.argsort(time, kind="mergesort")
    time = time[order]
    event = event[order]
    distinct = np.unique(time[event.astype(bool)])
    # Where each distinct event time starts in the ascending array.
    starts = np.searchsorted(time, distinct, side="left")
    return order, time, event, distinct, starts


def fit_cox(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    *,
    time_column: str = "time_days",
    event_column: str = "event",
    cluster_column: Optional[str] = "subject_id",
    robust: Optional[bool] = None,
    max_steps: int = MAX_NEWTON_STEPS,
    tolerance: float = CONVERGENCE_TOLERANCE,
) -> CoxFit:
    """Cox proportional hazards with Efron's correction for tied times.

    Ties matter here and are not a formality: follow-up is recorded in days and
    the death date has no time of day, so a single day can carry hundreds of
    deaths. Breslow's approximation biases coefficients towards zero as ties
    accumulate; Efron's costs a few more operations and does not.

    ``robust`` defaults to true whenever a cluster column is present and has
    fewer clusters than rows -- i.e. whenever the same patient appears twice.
    """

    columns = list(covariates)
    if not columns:
        raise ValueError("Ohne Kovariaten gibt es nichts zu schaetzen")
    # Validate before complete-case filtering: a malformed event must not
    # disappear merely because its marker or another covariate is missing.
    raw_event = frame[event_column]
    if (raw_event.notna() & ~raw_event.isin([0, 1])).any():
        raise ValueError("Event indicator must be binary 0/1 or missing")
    needed = columns + [time_column, event_column]
    if cluster_column and cluster_column in frame.columns:
        needed.append(cluster_column)
    usable = frame[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if usable.empty:
        raise ValueError("Keine vollstaendigen Zeilen fuer die Cox-Schaetzung")

    time = usable[time_column].to_numpy(dtype=float)
    event = usable[event_column].to_numpy(dtype=float)
    if not np.any(event > 0):
        raise ValueError("Keine Ereignisse -- eine Cox-Schaetzung ist sinnlos")
    if np.any(time <= 0):
        raise ValueError(
            "Nicht-positive Nachbeobachtungszeit; ein Ereignis vor dem "
            "Zeitnullpunkt ist ein Kopplungsfehler, kein Ergebnis"
        )
    design = usable[columns].to_numpy(dtype=float)

    clusters = None
    if cluster_column and cluster_column in usable.columns:
        clusters = usable[cluster_column].to_numpy()
    n_clusters = int(len(np.unique(clusters))) if clusters is not None else len(usable)
    if robust is None:
        robust = clusters is not None and n_clusters < len(usable)

    order, sorted_time, sorted_event, distinct, starts = _risk_set_blocks(
        time, event
    )
    X = design[order]
    beta = np.zeros(X.shape[1])
    log_likelihood = -np.inf
    steps = 0
    information = np.eye(X.shape[1])

    for steps in range(1, int(max_steps) + 1):
        log_likelihood, gradient, information = _partial_likelihood(
            X, sorted_time, sorted_event, distinct, starts, beta
        )
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as error:
            raise ConvergenceFailure(
                "Die Informationsmatrix ist singulaer -- vermutlich sind zwei "
                "Kovariaten linear abhaengig"
            ) from error
        beta = beta + step
        if np.max(np.abs(step)) < float(tolerance):
            break
    else:
        raise ConvergenceFailure(
            f"Die partielle Likelihood konvergierte nicht in {max_steps} Schritten"
        )

    log_likelihood, gradient, information = _partial_likelihood(
        X, sorted_time, sorted_event, distinct, starts, beta
    )
    covariance = np.linalg.inv(information)

    if robust and clusters is not None:
        residuals = _score_residuals(
            X, sorted_time, sorted_event, distinct, starts, beta
        )
        grouped = pd.DataFrame(residuals).groupby(clusters[order]).sum().to_numpy()
        meat = grouped.T @ grouped
        covariance = covariance @ meat @ covariance

    return CoxFit(
        covariates=tuple(columns),
        coefficients=beta,
        standard_errors=np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        variance=covariance,
        log_likelihood=float(log_likelihood),
        n_observations=int(len(usable)),
        n_events=int(np.sum(event > 0)),
        n_clusters=n_clusters,
        steps=int(steps),
        robust=bool(robust and clusters is not None),
    )


def _partial_likelihood(X, time, event, distinct, starts, beta):
    """Efron log partial likelihood, its gradient and its information.

    Walks the distinct event times from the last backwards, so the risk-set
    totals accumulate once instead of being rebuilt at every time.
    """

    n, p = X.shape
    eta = X @ beta
    eta -= eta.max()  # exp() overflows on a wide linear predictor otherwise
    weight = np.exp(eta)

    s0 = 0.0
    s1 = np.zeros(p)
    s2 = np.zeros((p, p))
    log_likelihood = 0.0
    gradient = np.zeros(p)
    information = np.zeros((p, p))
    upper = n

    for index in range(len(distinct) - 1, -1, -1):
        start = starts[index]
        block = slice(start, upper)
        if upper > start:
            weight_block = weight[block]
            x_block = X[block]
            weighted = x_block * weight_block[:, None]
            s0 += float(weight_block.sum())
            s1 += weighted.sum(axis=0)
            s2 += x_block.T @ weighted
        upper = start

        # Deaths at exactly this time. They sit at the front of the risk set
        # because the array is sorted by time.
        end = start + int(np.searchsorted(time[start:], distinct[index], side="right"))
        tied = np.flatnonzero(event[start:end] > 0) + start
        d = len(tied)
        if d == 0:
            continue
        x_tied = X[tied]
        weight_tied = weight[tied]
        weighted_tied = x_tied * weight_tied[:, None]
        d0 = float(weight_tied.sum())
        d1 = weighted_tied.sum(axis=0)
        d2 = x_tied.T @ weighted_tied

        log_likelihood += float(eta[tied].sum())
        gradient += x_tied.sum(axis=0)
        for k in range(d):
            fraction = k / d
            denominator = s0 - fraction * d0
            numerator1 = s1 - fraction * d1
            numerator2 = s2 - fraction * d2
            mean = numerator1 / denominator
            log_likelihood -= float(np.log(denominator))
            gradient -= mean
            information += numerator2 / denominator - np.outer(mean, mean)
    return log_likelihood, gradient, information


def _score_residuals(X, time, event, distinct, starts, beta):
    """Per-observation score residuals, for the cluster-robust variance.

    ``U_i = d_i (x_i - xbar(t_i)) - sum_{t_j <= t_i} w_i (x_i - xbar(t_j)) / S0(t_j)``

    The risk-set means are computed once, backwards, and then applied forwards
    as a cumulative sum -- the same trick as the likelihood, in the other
    direction. Breslow risk sets are used here rather than Efron: the
    correction affects the point estimate, and the residuals only have to
    define a consistent sandwich.
    """

    n, p = X.shape
    eta = X @ beta
    eta -= eta.max()
    weight = np.exp(eta)

    s0 = 0.0
    s1 = np.zeros(p)
    upper = n
    means = np.zeros((len(distinct), p))
    inverse_s0 = np.zeros(len(distinct))
    for index in range(len(distinct) - 1, -1, -1):
        start = starts[index]
        if upper > start:
            weight_block = weight[start:upper]
            s0 += float(weight_block.sum())
            s1 += (X[start:upper] * weight_block[:, None]).sum(axis=0)
        upper = start
        means[index] = s1 / s0
        inverse_s0[index] = 1.0 / s0

    # Forward cumulative sums over the event times an observation survived.
    cumulative_inverse = np.cumsum(inverse_s0)
    cumulative_mean = np.cumsum(means * inverse_s0[:, None], axis=0)

    # How many distinct event times each observation was still at risk for.
    seen = np.searchsorted(distinct, time, side="right")
    residuals = np.zeros((n, p))
    has_history = seen > 0
    index = seen[has_history] - 1
    residuals[has_history] = weight[has_history, None] * (
        cumulative_mean[index] - X[has_history] * cumulative_inverse[index, None]
    )

    died = event > 0
    at = np.searchsorted(distinct, time[died], side="left")
    residuals[died] += X[died] - means[at]
    return residuals


def concordance_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> tuple[float, int]:
    """Harrell's C over comparable pairs, in O(n log n).

    The textbook definition compares every pair, which is 320 billion
    comparisons on 800.035 ECGs. Sorting by risk and counting with a Fenwick
    tree gives the identical number in seconds, so the whole cohort can be
    scored rather than a sample of it.
    """

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    usable = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time, event, risk = time[usable], event[usable], risk[usable]
    if len(time) == 0 or not np.any(event > 0):
        return float("nan"), 0

    ranks = pd.Series(risk).rank(method="dense").to_numpy().astype(int)
    size = int(ranks.max())
    counts = _Fenwick(size)

    # Descending, so that by the time a death is scored the tree holds exactly
    # the subjects who outlived it -- its comparable partners. Ascending would
    # fill the tree with people who died earlier, which is the other half of
    # every pair and gives 1 - C.
    order = np.argsort(-time, kind="mergesort")
    time, event, ranks = time[order], event[order], ranks[order]

    concordant = 0.0
    comparable = 0
    index = 0
    n = len(time)
    while index < n:
        stop = index
        while stop < n and time[stop] == time[index]:
            stop += 1
        for position in range(index, stop):
            if event[position] <= 0:
                continue
            rank = ranks[position]
            later = counts.total()
            if later == 0:
                continue
            # A pair is concordant when the one who died first carried the
            # higher risk, so the partners that count are the lower ranks.
            lower = counts.query(rank - 1)
            tied_risk = counts.query(rank) - lower
            concordant += lower + 0.5 * tied_risk
            comparable += later
        # The block joins only after it is scored: two subjects sharing a time
        # are not comparable with each other in either direction.
        for position in range(index, stop):
            counts.add(ranks[position], 1)
        index = stop
    if comparable == 0:
        return float("nan"), 0
    return float(concordant / comparable), int(comparable)


def _censoring_survival_before(
    time: np.ndarray, event: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Kaplan-Meier of the censoring distribution, G(t-), at ``query``.

    Censorings are the rows with ``event == 0``. At a tied time the at-risk
    set is everyone with T >= t, the usual convention; G(t-) multiplies only
    the factors of times strictly before t.
    """

    order = np.argsort(time, kind="mergesort")
    sorted_time = time[order]
    sorted_event = event[order]
    unique, first = np.unique(sorted_time, return_index=True)
    at_risk = len(sorted_time) - first
    censored = np.bincount(
        np.searchsorted(unique, sorted_time[sorted_event <= 0]),
        minlength=len(unique),
    )
    survival = np.cumprod(1.0 - censored / at_risk)
    position = np.searchsorted(unique, query, side="left")
    result = np.ones(len(query), dtype=float)
    later = position > 0
    result[later] = survival[position[later] - 1]
    return result


def uno_concordance(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    tau: float,
) -> tuple[float, int]:
    """Uno's IPCW C-statistic truncated at ``tau`` (audit 5.3.2).

    Uno et al., Stat Med 2011: each comparable pair (i died first, before
    tau) is weighted by G(T_i-)^-2, which removes the dependence of Harrell's
    C on the censoring distribution. Assumes censoring independent of the
    marker. Ties in the marker count one half. Returns the statistic and the
    number of events that entered it.
    """

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    usable = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time, event, risk = time[usable], event[usable], risk[usable]
    if len(time) == 0 or not np.any(event > 0):
        return float("nan"), 0
    weights = _censoring_survival_before(time, event, time)
    ranks = pd.Series(risk).rank(method="dense").to_numpy().astype(int)
    counts = _Fenwick(int(ranks.max()))
    order = np.argsort(-time, kind="mergesort")
    time, event, ranks, weights = time[order], event[order], ranks[order], weights[order]
    numerator = 0.0
    denominator = 0.0
    events_used = 0
    index, n = 0, len(time)
    while index < n:
        stop = index
        while stop < n and time[stop] == time[index]:
            stop += 1
        for position in range(index, stop):
            if event[position] <= 0 or time[position] >= tau:
                continue
            g = weights[position]
            later = counts.total()
            if later == 0 or g <= 0:
                continue
            weight = 1.0 / (g * g)
            lower = counts.query(ranks[position] - 1)
            tied = counts.query(ranks[position]) - lower
            numerator += weight * (lower + 0.5 * tied)
            denominator += weight * later
            events_used += 1
        for position in range(index, stop):
            counts.add(ranks[position], 1)
        index = stop
    if denominator <= 0:
        return float("nan"), 0
    return float(numerator / denominator), int(events_used)


def cumulative_dynamic_auc(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    horizon: float,
) -> float:
    """Cumulative/dynamic time-dependent AUC at ``horizon`` with IPCW.

    Cases died by the horizon (weight 1/G(T_i-)), controls were still at risk
    after it (constant weight, cancels). Uno et al., JASA 2007; Hung &
    Chiang 2010. Ties in the marker count one half.
    """

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    usable = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time, event, risk = time[usable], event[usable], risk[usable]
    cases = (event > 0) & (time <= horizon)
    controls = time > horizon
    if not np.any(cases) or not np.any(controls):
        return float("nan")
    g = _censoring_survival_before(time, event, time[cases])
    valid = g > 0
    case_risk = risk[cases][valid]
    case_weight = 1.0 / g[valid]
    control_risk = np.sort(risk[controls])
    lower = np.searchsorted(control_risk, case_risk, side="left")
    tied = np.searchsorted(control_risk, case_risk, side="right") - lower
    numerator = float(np.sum(case_weight * (lower + 0.5 * tied)))
    denominator = float(np.sum(case_weight) * len(control_risk))
    return numerator / denominator if denominator > 0 else float("nan")


def landmark_from_index_discharge(frame: pd.DataFrame) -> pd.DataFrame:
    """Restart the clock at the discharge of the ECG's own admission.

    Audit 5.3.2. Keeps ECGs recorded during an admission whose patient was
    alive and under observation at discharge; follow-up is shortened by the
    days to discharge (the horizon still counts from the ECG). In-hospital
    deaths, which dominate the early hazard and reflect the acute illness
    around the ECG, leave the analysis.
    """

    if frame is None or len(frame) == 0 or "days_to_index_discharge" not in frame:
        return frame.iloc[0:0].copy() if frame is not None else frame
    offset = pd.to_numeric(frame["days_to_index_discharge"], errors="coerce")
    keep = offset.notna() & offset.ge(0) & frame["time_days"].gt(offset)
    result = frame.loc[keep].copy()
    result["time_days"] = result["time_days"] - offset.loc[keep]
    return result


class _Fenwick:
    """Prefix sums with point updates; the counting structure behind Harrell's C."""

    __slots__ = ("_tree", "_size", "_total")

    def __init__(self, size: int) -> None:
        self._size = int(size)
        self._tree = np.zeros(self._size + 1, dtype=np.int64)
        self._total = 0

    def add(self, index: int, value: int) -> None:
        self._total += int(value)
        position = int(index)
        while position <= self._size:
            self._tree[position] += value
            position += position & (-position)

    def query(self, index: int) -> int:
        position = int(index)
        result = 0
        while position > 0:
            result += int(self._tree[position])
            position -= position & (-position)
        return result

    def total(self) -> int:
        return int(self._total)


# The markers carried into the survival arm. Kept short on purpose: this is a
# confirmatory design with a prespecified covariate set, not a second screen.
# ``spatial_qrs_t_angle_deg`` leads because it is the marker the adjusted
# cross-sectional models already put first for 30-day mortality, so the two
# designs can be compared on the same quantity.
SURVIVAL_MARKERS: tuple[str, ...] = (
    "spatial_qrs_t_angle_deg",
    "p_loop_template_instability",
    "qrs_t_hausdorff_distance",
    "svg_magnitude",
)

# Age and sex, and nothing else. A survival model adjusted for heart rate and
# QRS duration would be adjusting for the marker's own mechanism; the
# cross-sectional arm does that deliberately and this one deliberately does
# not, so the two answer different questions rather than the same one twice.
SURVIVAL_COVARIATES: tuple[str, ...] = ("age", "female")


# How the curves are cut. Tertiles rather than quartiles or a median split:
# three groups are enough to show whether risk rises monotonically -- which is
# the property a reader is checking -- while keeping each curve large enough
# that its tail is not a staircase of single events.
SURVIVAL_GROUP_COUNT = 3

# Boundaries are fixed on the training split and applied unchanged to
# validation. Cutting each split at its own tertiles would let the boundaries
# move to wherever the events happen to be, which pulls the curves apart on
# any marker at all.
SURVIVAL_GROUP_SPLIT = "train"


def _patient_index_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one outcome-blind, deterministic ECG per patient.

    ``survival_frame_query`` assigns ``ecg_rank`` before marker-specific
    missingness is inspected, so rank one is the prespecified index ECG. The
    fallbacks keep the public curve helper safe for synthetic/ad-hoc frames
    without selecting on marker value or outcome.
    """

    if frame is None or len(frame) == 0 or "subject_id" not in frame:
        return frame
    if "ecg_rank" in frame:
        rank = pd.to_numeric(frame["ecg_rank"], errors="coerce")
        return frame.loc[rank.eq(1)].copy()
    if "ecg_time" in frame:
        columns = ["subject_id", "ecg_time"]
        if "study_id" in frame:
            columns.append("study_id")
        return (
            frame.sort_values(columns, kind="stable")
            .drop_duplicates("subject_id", keep="first")
            .copy()
        )
    return frame.drop_duplicates("subject_id", keep="first").copy()


def tertile_curves(
    frame: pd.DataFrame,
    marker: str,
    *,
    groups: int = SURVIVAL_GROUP_COUNT,
    boundary_split: str = SURVIVAL_GROUP_SPLIT,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    minimum_per_group: int = 200,
) -> pd.DataFrame:
    """Cumulative incidence per marker group, on the held-out split.

    Uses the project's own estimator (:func:`absolute_risk.aalen_johansen`
    through ``kaplan_meier_complement``) rather than a second implementation,
    so a curve here and a risk figure elsewhere cannot disagree.
    """

    from .absolute_risk import kaplan_meier_complement

    columns = [
        "feature_id", "group", "group_label", "n", "n_events",
        "time_days", "cumulative_incidence", "lower_bound", "upper_bound",
    ]
    if frame is None or len(frame) == 0 or marker not in frame.columns:
        return pd.DataFrame(columns=columns)
    if boundary_split != "train":
        raise ValueError("Survival group boundaries must be fixed on training")
    if "split" not in frame:
        raise ValueError("Survival curves require explicit development splits")
    if not frame["split"].isin(["train", "validation"]).all():
        raise PermissionError("Only development train/validation may enter survival curves")
    if "subject_id" in frame:
        from .patient_identity import known_patient_mask
        if not known_patient_mask(frame.subject_id).all():
            raise ValueError("Survival curves require known patient identities")
        if frame.groupby("subject_id")["split"].nunique().gt(1).any():
            raise ValueError("Patient overlap between survival splits")

    # Kaplan-Meier requires independent observational units. Repeated ECGs
    # belong only in the explicitly cluster-robust Cox sensitivity analysis;
    # otherwise both person-time and the eventual death are duplicated.
    frame = _patient_index_frame(frame)
    if "subject_id" in frame and frame.subject_id.duplicated().any():
        raise ValueError("Ambiguous survival index: multiple index ECGs per patient")
    values = pd.to_numeric(frame[marker], errors="coerce")
    finite = np.isfinite(values)
    usable = frame.loc[finite].copy()
    usable["_value"] = values[finite]
    if len(usable) < groups * minimum_per_group:
        return pd.DataFrame(columns=columns)

    reference = usable.loc[usable["split"].eq(boundary_split)]
    display = usable.loc[usable["split"].eq("validation")]
    if reference.empty or display.empty:
        return pd.DataFrame(columns=columns)

    quantiles = [i / groups for i in range(1, groups)]
    edges = list(reference["_value"].quantile(quantiles))
    if len(set(edges)) != len(edges):
        # A marker with a mass point can produce identical boundaries; three
        # groups would then not be three groups.
        return pd.DataFrame(columns=columns)

    assignment = np.digitize(display["_value"].to_numpy(), edges, right=False)
    records: list[dict] = []
    for index in range(groups):
        member = display.loc[assignment == index]
        if len(member) < minimum_per_group:
            continue
        curve = kaplan_meier_complement(
            member["time_days"].to_numpy(),
            member["event"].astype(int).to_numpy(),
        )
        label = (
            f"Terzil {index + 1} von {groups}"
            if groups == 3
            else f"Gruppe {index + 1} von {groups}"
        )
        for time, incidence, variance in zip(
            curve.times, curve.cause_incidence, curve.variance
        ):
            if float(time) > float(horizon_days):
                break
            deviation = float(np.sqrt(max(float(variance), 0.0)))
            records.append(
                {
                    "feature_id": str(marker),
                    "group": int(index + 1),
                    "group_label": label,
                    "n": int(len(member)),
                    "n_events": int(member["event"].sum()),
                    "time_days": float(time),
                    "cumulative_incidence": float(incidence),
                    "lower_bound": float(max(incidence - 1.96 * deviation, 0.0)),
                    "upper_bound": float(min(incidence + 1.96 * deviation, 1.0)),
                }
            )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)


def run_time_to_event(
    config: "MimicProjectConfig",
    *,
    run_id: str,
    panel_ids: Sequence[str] = STUDY_PANEL_IDS,
    markers: Sequence[str] = SURVIVAL_MARKERS,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    feature_glob: Optional[str] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """Fit the survival arm: the ECG as a clock, once per marker.

    Every marker is fitted twice -- on the first ECG per patient, and on all
    ECGs with a cluster-robust variance. The two are written side by side
    because their agreement is the evidence that neither design is carrying
    the result alone.
    """

    import duckdb

    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        capture_output_integrity,
        content_hash,
        development_feature_glob,
        feature_manifest_path,
        feature_parquet_clause,
        input_artifact_fingerprints,
        scientific_config_hash,
    )

    output_root = config.runs_root / run_id / "survival"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "time_to_event.manifest.json"

    marker_ids = tuple(dict.fromkeys(str(item) for item in markers))
    if not marker_ids:
        raise ValueError("Ohne Marker gibt es keine Ueberlebenszeitanalyse")

    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    feature_manifest = feature_manifest_path(resolved_glob)
    upstream = input_artifact_fingerprints(
        {
            "feature_manifest": feature_manifest,
            "feature_partition_index": feature_manifest.parent
            / "partition_index.json",
            "master_index_manifest": config.master_root
            / "master_index.manifest.json",
            "ingest_manifest": config.standardized_root / "ingest.manifest.json",
        }
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "markers": list(marker_ids),
            "covariates": list(SURVIVAL_COVARIATES),
            "horizon_days": float(horizon_days),
            "upstream": upstream,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, SURVIVAL_VERSION):
            return previous

    # The standardized parquet copies, not the raw csv.gz: they are what every
    # other stage reads, so a discrepancy between this arm and the rest can
    # never come from two different copies of the same table.
    patients = config.standardized_root / "patients.parquet"
    admissions = config.standardized_root / "admissions.parquet"
    for path in (patients, admissions):
        if not path.exists():
            raise FileNotFoundError(
                f"Fuer die Ueberlebenszeitanalyse fehlt {path}"
            )

    def sql_path(value) -> str:
        return str(value).replace(chr(92), "/")

    if progress:
        progress("Zeit-bis-Ereignis: Kohorte mit Zeitachse bilden")

    from .storage import isolated_analysis_connection

    with isolated_analysis_connection(config) as connection:
        clause = feature_parquet_clause(connection, sql_path(resolved_glob))
        query = survival_frame_query(
            sql_path(config.master_root / "ecg_master_index.parquet"),
            sql_path(patients),
            sql_path(admissions),
            clause,
            marker_columns=", ".join(f"f.{name}" for name in marker_ids),
        )
        raw = connection.execute(query).df()

    frame = follow_up(raw, horizon_days=float(horizon_days))
    sensitivity_frame = follow_up(
        raw,
        horizon_days=float(horizon_days),
        censoring=CENSORING_LAST_CONTACT,
    )
    excluded_never_admitted = int(
        (pd.to_numeric(raw.get("ever_admitted"), errors="coerce").fillna(0) == 0).sum()
    )
    # Rows whose death date precedes the recording. They are dropped, and the
    # count is reported rather than left to be inferred from a shrinking n.
    death_offset = pd.to_numeric(raw.get("days_to_death"), errors="coerce")
    dropped_negative = int(
        (
            raw.get("died", pd.Series(0, index=raw.index)).fillna(0).astype(bool)
            & death_offset.notna()
            & death_offset.lt(0)
        ).sum()
    )
    first = _patient_index_frame(frame)
    if progress:
        progress(
            f"Zeit-bis-Ereignis: {len(frame):,} EKGs, "
            f"{int(frame['event'].sum()):,} Ereignisse"
        )

    records: list[dict] = []
    for marker in marker_ids:
        if marker not in frame.columns:
            continue
        column = f"z_{marker}"
        # Fix one common unit on the prespecified training index ECGs. Using
        # every repeated ECG weighted frequently recorded patients multiple
        # times and let validation observations determine the HR-per-SD unit.
        scaling_frame = (
            first.loc[first["split"].astype(str).eq("train")]
            if "split" in first
            else first
        )
        scaling_values = pd.to_numeric(
            scaling_frame[marker], errors="coerce"
        )
        spread = float(scaling_values.std())
        if not np.isfinite(spread) or spread <= 0:
            continue
        centre = float(scaling_values.mean())
        for label, data, robust in (
            ("first_ecg_per_patient", first, False),
            ("all_ecgs_cluster_robust", frame, True),
            (
                "first_ecg_per_patient_last_contact_censoring",
                _patient_index_frame(sensitivity_frame),
                False,
            ),
            (
                "first_ecg_per_patient_landmark_index_discharge",
                landmark_from_index_discharge(first),
                False,
            ),
        ):
            block = data.copy()
            block[column] = (
                pd.to_numeric(block[marker], errors="coerce") - centre
            ) / spread
            usable = block[
                [column, *SURVIVAL_COVARIATES, "time_days", "event", "subject_id"]
            ].dropna()
            if len(usable) < 100 or int(usable["event"].sum()) < 20:
                continue
            try:
                fit = fit_cox(
                    usable,
                    [column, *SURVIVAL_COVARIATES],
                    robust=robust,
                )
            except (ValueError, np.linalg.LinAlgError, ConvergenceFailure) as exc:
                records.append(
                    {
                        "feature_id": str(marker),
                        "population": label,
                        "status": "non_estimable",
                        "abstention_reason": str(exc),
                        "n_observations": int(len(usable)),
                        "n_events": int(usable["event"].sum()),
                        "n_patients": int(usable["subject_id"].nunique()),
                        "scaling_center_train_index": centre,
                        "scaling_sd_train_index": spread,
                        "scaling_split": "train_patient_index_ecg",
                    }
                )
                continue
            interval = fit.confidence_interval()
            concordance, pairs = concordance_index(
                usable["time_days"].to_numpy(),
                usable["event"].to_numpy(),
                usable[column].to_numpy(),
            )
            uno_c, uno_events = uno_concordance(
                usable["time_days"].to_numpy(),
                usable["event"].to_numpy(),
                usable[column].to_numpy(),
                tau=float(horizon_days),
            )
            time_dependent_auc = {
                f"td_auc_{days}d": cumulative_dynamic_auc(
                    usable["time_days"].to_numpy(),
                    usable["event"].to_numpy(),
                    usable[column].to_numpy(),
                    horizon=float(days),
                )
                for days in (90, 365)
                if days <= float(horizon_days)
            }
            records.append(
                {
                    "feature_id": str(marker),
                    "population": label,
                    "status": "estimable",
                    "abstention_reason": "",
                    "n_observations": fit.n_observations,
                    "n_events": fit.n_events,
                    "n_patients": fit.n_clusters,
                    "hazard_ratio_per_sd": float(fit.hazard_ratios[0]),
                    "hr_lower": float(interval[0, 0]),
                    "hr_upper": float(interval[0, 1]),
                    "p_value": float(fit.p_values[0]),
                    "robust_variance": bool(fit.robust),
                    "concordance": float(concordance),
                    "concordance_pairs": int(pairs),
                    "uno_c_index": float(uno_c),
                    "uno_c_events": int(uno_events),
                    **time_dependent_auc,
                    "discrimination_note": (
                        "raw marker as risk score; Uno C and cumulative/dynamic "
                        "AUC with Kaplan-Meier censoring weights, censoring "
                        "assumed independent of the marker"
                    ),
                    "horizon_days": float(horizon_days),
                    "scaling_center_train_index": centre,
                    "scaling_sd_train_index": spread,
                    "scaling_split": "train_patient_index_ecg",
                }
            )

    table = pd.DataFrame(records)
    if len(table):
        from statsmodels.stats.multitest import multipletests

        table["q_value_global"] = np.nan
        table["q_value_population"] = np.nan
        p_values = pd.to_numeric(
            table.get("p_value", pd.Series(np.nan, index=table.index)),
            errors="coerce",
        )
        estimable = table["status"].eq("estimable") & p_values.notna()
        if estimable.any():
            table.loc[estimable, "q_value_global"] = multipletests(
                pd.to_numeric(table.loc[estimable, "p_value"]),
                method="fdr_bh",
            )[1]
            groups = table.loc[estimable].groupby("population").groups
            for _, index in groups.items():
                table.loc[index, "q_value_population"] = multipletests(
                    pd.to_numeric(table.loc[index, "p_value"]),
                    method="fdr_bh",
                )[1]

    curve_frames = [
        tertile_curves(first, marker, horizon_days=float(horizon_days))
        for marker in marker_ids
        if marker in frame.columns
    ]
    curves = pd.concat(
        [item for item in curve_frames if not item.empty], ignore_index=True
    ) if any(not item.empty for item in curve_frames) else pd.DataFrame()

    summary = {
        "version": SURVIVAL_VERSION,
        "run_id": str(run_id),
        "horizon_days": float(horizon_days),
        "covariates": list(SURVIVAL_COVARIATES),
        "n_ecgs": int(len(frame)),
        "n_patients": int(frame["subject_id"].nunique()) if len(frame) else 0,
        "n_events": int(frame["event"].sum()) if len(frame) else 0,
        "n_first_ecgs": int(len(first)),
        "n_dropped_negative_followup": dropped_negative,
        "censoring_primary": CENSORING_REGISTRY_WINDOW,
        "censoring_sensitivity": CENSORING_LAST_CONTACT,
        "registry_window_days": REGISTRY_WINDOW_DAYS,
        "n_ecgs_excluded_never_admitted": excluded_never_admitted,
        "n_events_last_contact_censoring": int(sensitivity_frame["event"].sum())
        if len(sensitivity_frame) else 0,
        "n_fits": int(len(table)),
        "ties": "efron",
        "primary_observational_unit": "first_ecg_per_patient",
        "repeated_ecg_role": "cluster_robust_sensitivity_only",
        "marker_scaling": "mean_and_sd_from_train_patient_index_ecgs_only",
    }
    if not table.empty and "hazard_ratio_per_sd" in table:
        # Whether the two designs agree is the property worth recording, not
        # a detail: a disagreement would mean the clustering carried the
        # result, and nothing downstream would otherwise notice.
        pivot = table.pivot_table(
            index="feature_id", columns="population",
            values="hazard_ratio_per_sd", aggfunc="first",
        )
        if {"first_ecg_per_patient", "all_ecgs_cluster_robust"} <= set(pivot.columns):
            difference = (
                pivot["first_ecg_per_patient"] - pivot["all_ecgs_cluster_robust"]
            ).abs()
            summary["max_hr_disagreement_between_designs"] = float(
                difference.max()
            )

    summary["n_curve_points"] = int(len(curves))
    summary["group_count"] = int(SURVIVAL_GROUP_COUNT)
    summary["group_boundary_split"] = str(SURVIVAL_GROUP_SPLIT)

    table_path = output_root / "time_to_event.parquet"
    curves_path = output_root / "time_to_event_curves.parquet"
    summary_path = output_root / "time_to_event_summary.json"
    atomic_write_parquet(table, table_path)
    atomic_write_parquet(curves, curves_path)
    atomic_write_json(summary_path, summary)

    manifest = StageManifest.complete(
        stage="time-to-event",
        stage_version=SURVIVAL_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(table_path.resolve()),
            str(curves_path.resolve()),
            str(summary_path.resolve()),
        ),
        row_counts={
            "fits": int(len(table)),
            "events": summary["n_events"],
            "curve_points": int(len(curves)),
        },
        output_integrity=capture_output_integrity(
            (table_path, curves_path, summary_path)
        ),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "CONVERGENCE_TOLERANCE",
    "ConvergenceFailure",
    "CoxFit",
    "DEFAULT_HORIZON_DAYS",
    "MAX_NEWTON_STEPS",
    "SURVIVAL_VERSION",
    "concordance_index",
    "cumulative_dynamic_auc",
    "landmark_from_index_discharge",
    "uno_concordance",
    "SURVIVAL_COVARIATES",
    "SURVIVAL_GROUP_COUNT",
    "SURVIVAL_GROUP_SPLIT",
    "SURVIVAL_MARKERS",
    "fit_cox",
    "follow_up",
    "run_time_to_event",
    "tertile_curves",
    "survival_frame_query",
]
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import MimicProjectConfig
