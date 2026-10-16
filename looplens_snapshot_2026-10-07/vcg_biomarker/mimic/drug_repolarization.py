"""Repolarisation change after starting a QT-prolonging drug (audit 5.2.6).

A self-controlled before/after design with untreated comparison pairs:

* **Start**: the first eMAR/Pyxis administration of a drug from
  ``MEDICATION_PATTERNS["qt_prolonging_drug"]`` with no administration of any
  of them in the seven days before (a new start, not a continued therapy).
* **Baseline ECG**: the last ECG within 48 hours before the start.
* **On-drug ECG**: the last ECG within 2 to 72 hours after the start.
  Repolarisation effects build up over hours to days; the ±2 h window of the
  AV-nodal therapy-response pairs is too short for them.
* **Untreated pairs**: consecutive ECGs of one admission 2 to 120 hours apart
  with no QT-prolonging administration from seven days before the first ECG
  to the second.

The treated change minus the untreated change is estimated per marker after
reweighting the untreated pairs to the treated distribution over gap bin x
baseline tertile of the marker. The baseline stratum matters: drugs are
withheld when the QTc is already long, and regression to the mean alone moves
an extreme baseline. Subjects are resampled for the interval with the weights
held fixed. It is an association in observational data, not a drug effect:
the indication (arrhythmia, infection, delirium, nausea) changes with the
start, and nothing here separates it from the drug.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

QT_DRUG_REPOLARIZATION_VERSION = "1.0.0"
WASHOUT_DAYS = 7
BASELINE_WINDOW_HOURS = 48
ON_DRUG_MIN_HOURS = 2
ON_DRUG_MAX_HOURS = 72
CONTROL_MIN_GAP_HOURS = 2
CONTROL_MAX_GAP_HOURS = 120
GAP_BIN_EDGES_HOURS = (2.0, 12.0, 24.0, 48.0, 120.0)
QT_DRUG_MARKERS: tuple[str, ...] = (
    "qtc_fridericia_ms",
    "jt_interval_ms",
    "tpeak_tend_ms",
    "spatial_qrs_t_angle_deg",
    "svg_magnitude",
)


def _drug_case(column: str) -> str:
    from .mappings import MEDICATION_PATTERNS

    branches = " ".join(
        f"WHEN LOWER({column}) LIKE '%{name}%' THEN '{name}'"
        for name in MEDICATION_PATTERNS["qt_prolonging_drug"]
    )
    return f"CASE {branches} END"


def qt_drug_treated_pair_query(master_sql: str, medications_sql: str) -> str:
    """Baseline and on-drug ECG around the start of a QT-prolonging drug."""

    return f"""
        WITH administrations AS (
            SELECT DISTINCT
                subject_id,
                event_time,
                {_drug_case("medication_raw")} AS drug
            FROM read_parquet('{medications_sql}')
            WHERE source_table IN ('emar', 'pyxis')
              AND medication_class = 'qt_prolonging_drug'
              AND event_time IS NOT NULL
        ),
        ordered AS (
            SELECT
                *,
                LAG(event_time) OVER (
                    PARTITION BY subject_id ORDER BY event_time, drug
                ) AS previous_any_time
            FROM administrations
            WHERE drug IS NOT NULL
        ),
        starts AS (
            SELECT subject_id, drug, event_time AS start_time
            FROM ordered
            WHERE previous_any_time IS NULL
               OR previous_any_time < event_time - INTERVAL {WASHOUT_DAYS} DAY
        ),
        before_ecg AS (
            SELECT
                s.subject_id, s.drug, s.start_time,
                m.study_id AS study_id_before, m.ecg_time AS ecg_time_before,
                ROW_NUMBER() OVER (
                    PARTITION BY s.subject_id, s.drug, s.start_time
                    ORDER BY m.ecg_time DESC, m.study_id DESC
                ) AS rn
            FROM starts s
            JOIN read_parquet('{master_sql}') m
              ON m.subject_id = s.subject_id
             AND m.ecg_time < s.start_time
             AND m.ecg_time >= s.start_time - INTERVAL {BASELINE_WINDOW_HOURS} HOUR
        ),
        after_ecg AS (
            SELECT
                s.subject_id, s.drug, s.start_time,
                m.study_id AS study_id_after, m.ecg_time AS ecg_time_after,
                ROW_NUMBER() OVER (
                    PARTITION BY s.subject_id, s.drug, s.start_time
                    ORDER BY m.ecg_time DESC, m.study_id DESC
                ) AS rn
            FROM starts s
            JOIN read_parquet('{master_sql}') m
              ON m.subject_id = s.subject_id
             AND m.ecg_time >= s.start_time + INTERVAL {ON_DRUG_MIN_HOURS} HOUR
             AND m.ecg_time <= s.start_time + INTERVAL {ON_DRUG_MAX_HOURS} HOUR
        )
        SELECT
            b.subject_id,
            'treated' AS pair_kind,
            b.drug,
            b.start_time AS exposure_event_time,
            b.study_id_before,
            a.study_id_after,
            DATE_DIFF('second', b.ecg_time_before, a.ecg_time_after) / 3600.0
                AS ecg_gap_hours
        FROM (SELECT * FROM before_ecg WHERE rn = 1) b
        JOIN (SELECT * FROM after_ecg WHERE rn = 1) a
          USING (subject_id, drug, start_time)
        WHERE b.study_id_before != a.study_id_after
    """


def qt_drug_control_pair_query(master_sql: str, medications_sql: str) -> str:
    """Consecutive same-admission ECG pairs without QT-prolonging drugs."""

    return f"""
        WITH ordered AS (
            SELECT
                subject_id, study_id, hadm_id, ecg_time,
                LEAD(study_id) OVER w AS next_study_id,
                LEAD(ecg_time) OVER w AS next_ecg_time,
                LEAD(hadm_id) OVER w AS next_hadm_id
            FROM read_parquet('{master_sql}')
            WHERE ecg_time IS NOT NULL
            WINDOW w AS (PARTITION BY subject_id ORDER BY ecg_time, study_id)
        ),
        candidates AS (
            SELECT
                subject_id,
                study_id AS study_id_before,
                ecg_time AS ecg_time_before,
                next_study_id AS study_id_after,
                next_ecg_time AS ecg_time_after
            FROM ordered
            WHERE next_study_id IS NOT NULL
              AND hadm_id IS NOT NULL
              AND next_hadm_id = hadm_id
              AND next_ecg_time >= ecg_time + INTERVAL {CONTROL_MIN_GAP_HOURS} HOUR
              AND next_ecg_time <= ecg_time + INTERVAL {CONTROL_MAX_GAP_HOURS} HOUR
        ),
        exposures AS (
            SELECT DISTINCT subject_id, event_time
            FROM read_parquet('{medications_sql}')
            WHERE source_table IN ('emar', 'pyxis')
              AND medication_class = 'qt_prolonging_drug'
        )
        SELECT
            c.subject_id,
            'untreated' AS pair_kind,
            CAST(NULL AS VARCHAR) AS drug,
            CAST(NULL AS TIMESTAMP) AS exposure_event_time,
            c.study_id_before,
            c.study_id_after,
            DATE_DIFF('second', c.ecg_time_before, c.ecg_time_after) / 3600.0
                AS ecg_gap_hours
        FROM candidates c
        WHERE NOT EXISTS (
            SELECT 1 FROM exposures e
            WHERE e.subject_id = c.subject_id
              AND e.event_time >= c.ecg_time_before - INTERVAL {WASHOUT_DAYS} DAY
              AND e.event_time <= c.ecg_time_after
        )
    """


def attach_marker_values_query(
    pairs_sql: str, feature_clause: str, markers: Sequence[str]
) -> str:
    """Before/after values of each marker for every pair."""

    columns = ", ".join(f"f.{name}" for name in markers)
    selected = ", ".join(
        f"fb.{name} AS {name}_before, fa.{name} AS {name}_after" for name in markers
    )
    return f"""
        WITH features AS (
            SELECT f.subject_id, f.study_id, f.split, {columns}
            FROM {feature_clause} f
            WHERE f.extraction_status = 'ok'
              AND f.split IN ('train', 'validation')
        )
        SELECT p.*, fb.split AS split_before, {selected}
        FROM read_parquet('{pairs_sql}') p
        JOIN features fb
          ON fb.subject_id = p.subject_id AND fb.study_id = p.study_id_before
        JOIN features fa
          ON fa.subject_id = p.subject_id AND fa.study_id = p.study_id_after
    """


def repolarization_comparison(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    marker: str,
    *,
    iterations: int = 2000,
    seed: int = 20260924,
    confidence: float = 0.95,
) -> dict:
    """Treated change minus standardised untreated change for one marker."""

    def prepared(frame: pd.DataFrame) -> pd.DataFrame:
        before = pd.to_numeric(frame.get(f"{marker}_before"), errors="coerce")
        after = pd.to_numeric(frame.get(f"{marker}_after"), errors="coerce")
        gap = pd.to_numeric(frame.get("ecg_gap_hours"), errors="coerce")
        result = pd.DataFrame(
            {
                "subject_id": frame["subject_id"].to_numpy(),
                "before": before.to_numpy(dtype=float),
                "delta": (after - before).to_numpy(dtype=float),
                "gap": gap.to_numpy(dtype=float),
            }
        )
        return result.loc[
            np.isfinite(result["before"]) & np.isfinite(result["delta"])
            & np.isfinite(result["gap"])
        ].copy()

    empty = {
        "marker": marker, "n_treated_pairs": 0, "n_control_pairs": 0,
        "n_subjects": 0, "treated_mean_change": np.nan,
        "standardized_control_mean_change": np.nan, "difference": np.nan,
        "ci_low": np.nan, "ci_high": np.nan, "n_strata_matched": 0,
    }
    treated_rows = prepared(treated)
    control_rows = prepared(controls)
    if len(treated_rows) < 3 or control_rows.empty:
        return {**empty, "n_treated_pairs": int(len(treated_rows)),
                "n_control_pairs": int(len(control_rows))}
    # Tertile cut points from the treated baselines; the untreated pairs are
    # placed on the same scale.
    cuts = np.quantile(treated_rows["before"], [1 / 3, 2 / 3])

    def stratum(frame: pd.DataFrame) -> pd.Series:
        gap_bin = pd.cut(frame["gap"], bins=list(GAP_BIN_EDGES_HOURS), include_lowest=True)
        tertile = np.searchsorted(cuts, frame["before"].to_numpy(), side="right")
        return gap_bin.astype(str) + "|" + pd.Series(tertile, index=frame.index).astype(str)

    treated_rows["stratum"] = stratum(treated_rows)
    control_rows["stratum"] = stratum(control_rows)
    treated_rows = treated_rows.loc[~treated_rows["stratum"].str.startswith("nan")]
    control_rows = control_rows.loc[~control_rows["stratum"].str.startswith("nan")]
    matched = pd.Index(treated_rows["stratum"].unique()).intersection(
        pd.Index(control_rows["stratum"].unique())
    )
    if matched.empty:
        return {**empty, "n_treated_pairs": int(len(treated_rows)),
                "n_control_pairs": int(len(control_rows))}
    treated_rows = treated_rows.loc[treated_rows["stratum"].isin(matched)]
    control_rows = control_rows.loc[control_rows["stratum"].isin(matched)]
    target = treated_rows["stratum"].value_counts(normalize=True)
    source = control_rows["stratum"].value_counts(normalize=True)
    control_rows = control_rows.assign(
        weight=control_rows["stratum"].map(target / source).astype(float)
    )

    def estimate(t_values, c_values, c_weights) -> float:
        if len(t_values) == 0 or np.sum(c_weights) <= 0:
            return float("nan")
        return float(np.mean(t_values) - np.average(c_values, weights=c_weights))

    point = estimate(
        treated_rows["delta"].to_numpy(),
        control_rows["delta"].to_numpy(),
        control_rows["weight"].to_numpy(),
    )
    subjects = pd.unique(pd.concat([treated_rows["subject_id"], control_rows["subject_id"]]))
    low = high = float("nan")
    if iterations > 0 and len(subjects) >= 2:
        t_by = {s: g["delta"].to_numpy() for s, g in treated_rows.groupby("subject_id")}
        c_by = {
            s: (g["delta"].to_numpy(), g["weight"].to_numpy())
            for s, g in control_rows.groupby("subject_id")
        }
        rng = np.random.default_rng(seed)
        nothing = np.empty(0)
        draws = []
        for _ in range(int(iterations)):
            sample = rng.choice(subjects, size=len(subjects), replace=True)
            t = np.concatenate([t_by.get(s, nothing) for s in sample])
            cv = np.concatenate([c_by[s][0] if s in c_by else nothing for s in sample])
            cw = np.concatenate([c_by[s][1] if s in c_by else nothing for s in sample])
            value = estimate(t, cv, cw)
            if np.isfinite(value):
                draws.append(value)
        if draws:
            alpha = (1.0 - confidence) / 2.0
            low, high = (float(item) for item in np.quantile(draws, [alpha, 1.0 - alpha]))
    return {
        "marker": marker,
        "n_treated_pairs": int(len(treated_rows)),
        "n_control_pairs": int(len(control_rows)),
        "n_subjects": int(len(subjects)),
        "treated_mean_change": float(treated_rows["delta"].mean()),
        "standardized_control_mean_change": float(
            np.average(control_rows["delta"], weights=control_rows["weight"])
        ),
        "difference": point,
        "ci_low": low,
        "ci_high": high,
        "n_strata_matched": int(len(matched)),
    }


def qt_drug_results(
    pairs: pd.DataFrame,
    markers: Sequence[str],
    *,
    iterations: int,
    seed: int,
    confidence: float,
    minimum_drug_pairs: int = 30,
) -> pd.DataFrame:
    """One row per marker for all drugs, and per drug with enough pairs."""

    treated = pairs.loc[pairs["pair_kind"].eq("treated")]
    controls = pairs.loc[pairs["pair_kind"].eq("untreated")]
    scopes: list[tuple[str, pd.DataFrame]] = [("all_qt_prolonging_drugs", treated)]
    for drug, group in treated.groupby("drug", sort=True):
        if len(group) >= int(minimum_drug_pairs):
            scopes.append((str(drug), group))
    rows: list[dict] = []
    for scope, group in scopes:
        for marker in markers:
            result = repolarization_comparison(
                group, controls, marker,
                iterations=iterations, seed=seed, confidence=confidence,
            )
            estimable = np.isfinite(result["difference"])
            rows.append(
                {
                    "scope": scope,
                    **result,
                    "status": "estimable" if estimable else "non_estimable",
                    "effect_measure": (
                        "mean_change_treated_minus_standardized_untreated"
                    ),
                    "uncertainty_method": (
                        "non_estimable" if not estimable
                        else "not_computed_scaling_pilot" if int(iterations) <= 0
                        else "subject_cluster_bootstrap_fixed_stratum_weights"
                    ),
                    "stratification": "gap_bin_x_treated_baseline_tertile",
                    "claim_scope": (
                        "association_after_new_drug_start_against_untreated_"
                        "same_admission_pairs_not_a_causal_drug_effect"
                    ),
                    "stage_version": QT_DRUG_REPOLARIZATION_VERSION,
                }
            )
    return pd.DataFrame(rows)


def available_markers(columns: set[str], markers: Optional[Sequence[str]] = None) -> tuple[str, ...]:
    return tuple(name for name in (markers or QT_DRUG_MARKERS) if name in columns)


__all__ = [
    "QT_DRUG_MARKERS",
    "QT_DRUG_REPOLARIZATION_VERSION",
    "attach_marker_values_query",
    "available_markers",
    "qt_drug_control_pair_query",
    "qt_drug_results",
    "qt_drug_treated_pair_query",
    "repolarization_comparison",
]
