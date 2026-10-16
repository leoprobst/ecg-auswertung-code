"""Serial ECG / medication pairing for tachycardia therapy-response markers.

For each discrete, timestamp-anchored AV-nodal-blocking medication
exposure (adenosine, class I/III antiarrhythmics, calcium-channel
blockers, beta-blockers, digoxin -- see
``mappings.AV_NODAL_BLOCKING_MEDICATION_CLASSES``), the nearest ECG within
2h before the exposure is paired with the nearest ECG within 2h after it
(the pairing itself happens in
``evidence._ecg_medication_exposure_pair_query``, part of the
build-evidence stage). This module joins in each paired ECG's own
already-computed rhythm-classification features and derives before/after
delta markers.

Honesty caveat: this compares discrete ECG snapshots, potentially
minutes to hours apart, not continuous telemetry. It cannot capture an
"abrupt termination within seconds" in the classic electrophysiology
sense. It answers the weaker but still clinically relevant question of
whether the rhythm classification normalized between the nearest ECG
before and the nearest ECG after a rhythm-active medication exposure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .evidence import EvidenceArtifact
from .run_log import profile_query
from .statistics import subject_cluster_bootstrap
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    feature_manifest_path,
    file_sha256,
    feature_parquet_clause,
    input_artifact_fingerprints,
    parquet_columns,
    scientific_config_hash,
)

_NORMALIZED_RHYTHM_CONTEXT = "regular"
THERAPY_RESPONSE_VERSION = "1.4.0"  # audit 5.2.6 QT-drug repolarisation
# Audit X7: untreated comparison pairs. Consecutive ECGs of one admission at
# most four hours apart (the longest treated pair), with no AV-nodal exposure
# from 24 h before the first ECG to the second.
UNTREATED_CONTROL_MAXIMUM_GAP_HOURS = 4.0
UNTREATED_CONTROL_WASHOUT_HOURS = 24.0
UNTREATED_CONTROL_CLASS = "untreated_control"
GAP_BIN_EDGES_HOURS = (0.0, 1.0, 2.0, 3.0, 4.0)


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def _copy_query_atomic(connection, query: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    if temporary.exists():
        temporary.unlink()
    temporary_sql = _sql_path(temporary)
    with profile_query(f"therapy-response:{output_path.name}") as profile:
        copy_result = connection.execute(
            f"COPY ({query}) TO '{temporary_sql}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
        ).fetchone()
        if copy_result is None:
            raise RuntimeError("DuckDB COPY returned no row count")
        count = int(copy_result[0])
        atomic_replace(temporary, output_path)
        profile["rows"] = count
        return count


HEART_RATE_SOURCE_OWN = "own_delineation"
HEART_RATE_SOURCE_RECORDER = "recorder_reported"


def _feature_columns(connection, feature_sql: str) -> set[str]:
    """Which columns the extracted features actually carry.

    Deliberately without ``union_by_name``.  Unifying 1,889 columns across
    1,527 partitions to answer one yes/no question about a column name was
    measured at 42 s and 3.6 GB -- against a stage profile of 2 GB, and it
    paused the run seconds after every resume.  Reading the schema plainly
    costs 0.07 s and no measurable memory, because the partitions of one
    extraction share a schema.

    If they ever do not, DuckDB raises rather than guessing, and the slow
    unifying form answers instead.  The fast path is the normal case; the
    correct path is still there for the case that would otherwise be wrong.
    """

    return parquet_columns(connection, feature_sql)


def _therapy_response_pair_query(
    pairs_sql: str,
    feature_sql: str,
    *,
    master_sql: Optional[str] = None,
    feature_clause: Optional[str] = None,
    temporal_available: bool = False,
) -> str:
    """Join the before/after pairs to each ECG's rhythm context and rate.

    The heart rate has two possible sources and the query adapts to what is
    on disk rather than assuming one.

    ``heart_rate_bpm`` is a catalog marker, but its role is
    ``ecg_context_comparator`` -- a covariate, not a candidate biomarker --
    so the panels exclude it and the extraction does not write it, unlike
    ``rhythm_context`` which is carried as context.  Requiring it made this
    stage fail on a completed extraction, and re-extracting 800,000 records
    to add one column is not a proportionate answer.

    When the column is absent the recorder's own ``machine_heart_rate_bpm``
    is used instead (97.6 % populated). The extraction does carry the own
    rate as the context column ``context_heart_rate_bpm`` (mean RR of the
    delineation); it is deliberately not used here, because the recorder's
    value is independent of this project's delineation -- a decision, not a
    gap.  For a before/after comparison that
    is defensible and arguably cleaner: it is independent of this project's
    delineation, and a systematic device offset cancels in the difference.
    It is not the same measurement though, so every row records which source
    it used in ``heart_rate_source`` -- a substitution that does not announce
    itself is worse than the failure it replaced.

    Preferring the own measurement when it exists means the next extraction
    that carries the column upgrades this stage without another edit.
    """

    # ``union_by_name`` over 1,527 wide partitions held 19.36 GB resident
    # against an 8 GB DuckDB limit and took 109 s; the plain read returned the
    # identical 3,357 rows in 9.5 s at 2.11 GB. The clause is therefore chosen
    # from what is on disk -- see ``storage.feature_parquet_clause``.
    read_features = feature_clause or (
        f"read_parquet('{feature_sql}', union_by_name=true)"
    )
    if master_sql is None:
        rate_select = "heart_rate_bpm"
        rate_source = f"'{HEART_RATE_SOURCE_OWN}'"
        rate_from = f"{read_features} f"
    else:
        rate_select = "m.machine_heart_rate_bpm AS heart_rate_bpm"
        rate_source = f"'{HEART_RATE_SOURCE_RECORDER}'"
        rate_from = (
            f"{read_features} f "
            f"LEFT JOIN read_parquet('{master_sql}') m "
            "ON m.study_id = f.study_id"
        )

    temporal_select = "f.context_temporal_research_json" if temporal_available else "CAST(NULL AS VARCHAR)"
    return f"""
        WITH pairs AS (
            SELECT * FROM read_parquet('{pairs_sql}')
        ),
        features AS (
            SELECT
                f.subject_id,
                f.study_id,
                f.split,
                f.rhythm_context,
                {temporal_select} AS temporal_descriptor,
                {rate_select},
                {rate_source} AS heart_rate_source,
                (COALESCE(f.narrow_complex_tachycardia_pattern_flag, 0) = 1
                 OR COALESCE(f.wide_complex_tachycardia_pattern_flag, 0) = 1)
                    AS tachycardia_pattern
            FROM {rate_from}
            WHERE f.extraction_status = 'ok'
              AND f.split IN ('train', 'validation')
        )
        SELECT
            p.subject_id,
            p.medication_class,
            p.exposure_event_time,
            p.study_id_before,
            p.study_id_after,
            p.hours_before_to_exposure,
            p.hours_exposure_to_after,
            fb.temporal_descriptor AS temporal_before,
            fa.temporal_descriptor AS temporal_after,
            fb.split AS split_before,
            fb.rhythm_context AS rhythm_class_before,
            fb.heart_rate_source AS heart_rate_source,
            fb.heart_rate_bpm AS heart_rate_bpm_before,
            fb.tachycardia_pattern AS tachycardia_pattern_before,
            fa.split AS split_after,
            fa.rhythm_context AS rhythm_class_after,
            fa.heart_rate_bpm AS heart_rate_bpm_after,
            fa.tachycardia_pattern AS tachycardia_pattern_after,
            fa.heart_rate_bpm - fb.heart_rate_bpm AS heart_rate_delta_bpm,
            CASE
                WHEN fb.tachycardia_pattern IS NULL
                     OR fa.tachycardia_pattern IS NULL THEN NULL
                WHEN fb.tachycardia_pattern AND NOT fa.tachycardia_pattern
                     AND fa.rhythm_context = '{_NORMALIZED_RHYTHM_CONTEXT}'
                THEN TRUE
                ELSE FALSE
            END AS rhythm_normalized_flag
        FROM pairs p
        JOIN features fb
          ON fb.subject_id = p.subject_id AND fb.study_id = p.study_id_before
        JOIN features fa
          ON fa.subject_id = p.subject_id AND fa.study_id = p.study_id_after
    """


def build_therapy_response_pairs(
    connection,
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    output_path: Path,
    progress: Optional[Callable[[str], None]] = None,
) -> EvidenceArtifact:
    """Join before/after ECG exposure pairs with each ECG's own rhythm
    features and compute delta markers.

    Requires build-evidence (for ecg_medication_exposure_pairs.parquet)
    and extract-features (for `feature_glob`, whose columns must include
    rhythm_context, narrow_complex_tachycardia_pattern_flag and
    wide_complex_tachycardia_pattern_flag) to have already run.

    ``heart_rate_bpm`` is used when the extraction carries it and the
    recorder's ``machine_heart_rate_bpm`` otherwise; the chosen source is
    recorded per row.
    """
    if progress:
        progress("Verknuepfe EKG-Vorher/Nachher-Paare mit Rhythmus-Features")
    pairs_path = config.evidence_root / "ecg_medication_exposure_pairs.parquet"
    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"{pairs_path} fehlt -- zuerst die build-evidence-Stage ausfuehren"
        )
    feature_sql = str(feature_glob).replace("\\", "/").replace("'", "''")
    master_sql: Optional[str] = None
    available_columns = _feature_columns(connection, feature_sql)
    if "heart_rate_bpm" not in available_columns:
        master_path = config.master_root / "ecg_master_index.parquet"
        if not master_path.is_file():
            raise FileNotFoundError(
                f"{master_path} fehlt -- die Merkmale enthalten keine "
                "heart_rate_bpm-Spalte, und ohne den Master-Index gibt es "
                "keine Ersatzquelle fuer die Herzfrequenz"
            )
        master_sql = _sql_path(master_path)
        if progress:
            progress(
                "Herzfrequenz aus dem Master-Index (Geraetemessung); die "
                "Merkmale fuehren keine eigene heart_rate_bpm-Spalte"
            )
    query = _therapy_response_pair_query(
        _sql_path(pairs_path),
        feature_sql,
        master_sql=master_sql,
        feature_clause=feature_parquet_clause(connection, feature_sql),
        temporal_available="context_temporal_research_json" in available_columns,
    )
    count = _copy_query_atomic(connection, query, output_path)
    return EvidenceArtifact("therapy_response_pairs", output_path, count)


def _untreated_control_pair_query(
    master_sql: str,
    medications_sql: str,
) -> str:
    """Consecutive same-admission ECG pairs without AV-nodal exposure (X7).

    Shaped like the treated pairs so the same feature join applies:
    ``medication_class`` is ``untreated_control`` and the exposure columns
    are NULL. The gap between the two ECGs is kept for matching.
    """
    from .mappings import AV_NODAL_BLOCKING_MEDICATION_CLASSES

    class_list = ", ".join(
        f"'{name}'" for name in sorted(AV_NODAL_BLOCKING_MEDICATION_CLASSES)
    )
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
              AND next_ecg_time > ecg_time
              AND next_ecg_time <= ecg_time
                  + INTERVAL {int(UNTREATED_CONTROL_MAXIMUM_GAP_HOURS * 60)} MINUTE
        ),
        exposures AS (
            SELECT DISTINCT subject_id, event_time
            FROM read_parquet('{medications_sql}')
            WHERE source_table IN ('emar', 'pyxis')
              AND medication_class IN ({class_list})
        )
        SELECT
            c.subject_id,
            '{UNTREATED_CONTROL_CLASS}' AS medication_class,
            CAST(NULL AS TIMESTAMP) AS exposure_event_time,
            c.study_id_before,
            c.study_id_after,
            CAST(NULL AS DOUBLE) AS hours_before_to_exposure,
            CAST(NULL AS DOUBLE) AS hours_exposure_to_after
        FROM candidates c
        WHERE NOT EXISTS (
            SELECT 1 FROM exposures e
            WHERE e.subject_id = c.subject_id
              AND e.event_time >= c.ecg_time_before
                  - INTERVAL {int(UNTREATED_CONTROL_WASHOUT_HOURS)} HOUR
              AND e.event_time <= c.ecg_time_after
        )
    """


def _pair_gap_hours(pairs: pd.DataFrame) -> pd.Series:
    def column(name: str) -> pd.Series:
        return pd.to_numeric(
            pairs.get(name, pd.Series(np.nan, index=pairs.index)), errors="coerce"
        )

    treated_gap = column("hours_before_to_exposure") + column("hours_exposure_to_after")
    return column("ecg_gap_hours").combine_first(treated_gap)


def untreated_comparison(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 20260809,
    confidence: float = 0.95,
) -> dict:
    """Normalisation among pre-tachycardic pairs, treated minus untreated.

    Audit X7: spontaneous conversion and regression to the mean (the more
    extreme a tachycardia, the likelier it is treated) look exactly like a
    drug effect in a before/after design. Untreated pairs are reweighted to
    the treated pairs' distribution over gap bin x baseline rhythm class, so
    the difference compares like with like on both. The weights are fixed on
    the full sample; subjects are resampled for the interval.
    """
    def pre_tachycardic(frame: pd.DataFrame) -> pd.DataFrame:
        usable = frame.loc[
            frame.get("tachycardia_pattern_before", pd.Series(False, index=frame.index))
            .fillna(False).astype(bool)
        ].copy()
        usable = usable.loc[pd.to_numeric(usable["rhythm_normalized_flag"], errors="coerce").notna()]
        usable["gap_bin"] = pd.cut(
            _pair_gap_hours(usable), bins=list(GAP_BIN_EDGES_HOURS), include_lowest=True
        ).astype(str)
        usable["stratum"] = usable["gap_bin"] + "|" + usable["rhythm_class_before"].astype(str)
        usable["normalized"] = pd.to_numeric(usable["rhythm_normalized_flag"], errors="coerce").astype(float)
        return usable

    empty = {
        "n_treated_pairs": 0, "n_control_pairs": 0, "n_subjects": 0,
        "treated_normalization_rate": np.nan,
        "standardized_control_normalization_rate": np.nan,
        "difference": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "n_strata_matched": 0,
    }
    treated_rows = pre_tachycardic(treated)
    control_rows = pre_tachycardic(controls)
    if treated_rows.empty or control_rows.empty:
        return {**empty, "n_treated_pairs": int(len(treated_rows)),
                "n_control_pairs": int(len(control_rows))}
    treated_share = treated_rows["stratum"].value_counts(normalize=True)
    control_share = control_rows["stratum"].value_counts(normalize=True)
    matched = treated_share.index.intersection(control_share.index)
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
        if len(t_values) == 0 or c_weights.sum() <= 0:
            return float("nan")
        return float(t_values.mean() - np.average(c_values, weights=c_weights))

    point = estimate(
        treated_rows["normalized"].to_numpy(),
        control_rows["normalized"].to_numpy(),
        control_rows["weight"].to_numpy(),
    )
    subjects = pd.unique(pd.concat([treated_rows["subject_id"], control_rows["subject_id"]]))
    low = high = float("nan")
    if iterations > 0 and len(subjects) >= 2:
        t_by = {s: g["normalized"].to_numpy() for s, g in treated_rows.groupby("subject_id")}
        c_by = {s: (g["normalized"].to_numpy(), g["weight"].to_numpy())
                for s, g in control_rows.groupby("subject_id")}
        rng = np.random.default_rng(seed)
        draws = []
        empty_array = np.empty(0)
        for _ in range(int(iterations)):
            sample = rng.choice(subjects, size=len(subjects), replace=True)
            t = np.concatenate([t_by.get(s, empty_array) for s in sample])
            cv = np.concatenate([c_by[s][0] if s in c_by else empty_array for s in sample])
            cw = np.concatenate([c_by[s][1] if s in c_by else empty_array for s in sample])
            value = estimate(t, cv, cw)
            if np.isfinite(value):
                draws.append(value)
        if draws:
            alpha = (1.0 - confidence) / 2.0
            low, high = (float(item) for item in np.quantile(draws, [alpha, 1.0 - alpha]))
    return {
        "n_treated_pairs": int(len(treated_rows)),
        "n_control_pairs": int(len(control_rows)),
        "n_subjects": int(len(subjects)),
        "treated_normalization_rate": float(treated_rows["normalized"].mean()),
        "standardized_control_normalization_rate": float(
            np.average(control_rows["normalized"], weights=control_rows["weight"])
        ),
        "difference": point,
        "ci_low": low,
        "ci_high": high,
        "n_strata_matched": int(len(matched)),
    }


def therapy_response_tachycardia_rate_change(
    pairs: pd.DataFrame,
    *,
    iterations: int = 2000,
    seed: int = 20260809,
    confidence: float = 0.95,
) -> dict:
    """Subject-cluster-bootstrapped estimate of how much the
    tachycardia-pattern rate changes from the before-ECG to the
    after-ECG.

    Reuses the platform's existing paired subject-cluster bootstrap
    (``statistics.subject_cluster_bootstrap``) instead of adding a new
    statistical method: each pair contributes two rows to the resampling
    frame (before, label 0; after, label 1) sharing the same
    ``subject_id``, so resampling subjects with replacement keeps each
    pair's before/after values coupled together rather than treating
    them as independent observations.
    """
    usable = pairs.dropna(
        subset=["tachycardia_pattern_before", "tachycardia_pattern_after"]
    )
    n_pairs = int(len(usable))
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "n_subjects": 0,
            "rate_before": float("nan"),
            "rate_after": float("nan"),
            "rate_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    before = usable["tachycardia_pattern_before"].to_numpy(dtype=float)
    after = usable["tachycardia_pattern_after"].to_numpy(dtype=float)
    subject_ids = np.concatenate(
        [usable["subject_id"].to_numpy(), usable["subject_id"].to_numpy()]
    )
    values = np.concatenate([before, after])
    labels = np.concatenate(
        [np.zeros(n_pairs, dtype=int), np.ones(n_pairs, dtype=int)]
    )
    rate_before = float(before.mean())
    rate_after = float(after.mean())
    ci_low, ci_high = subject_cluster_bootstrap(
        subject_ids,
        values,
        labels,
        lambda v, l: float(v[l == 1].mean() - v[l == 0].mean()),
        iterations=iterations,
        seed=seed,
        confidence=confidence,
    )
    return {
        "n_pairs": n_pairs,
        "n_subjects": int(usable["subject_id"].nunique()),
        "rate_before": rate_before,
        "rate_after": rate_after,
        "rate_difference": rate_after - rate_before,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_therapy_response_analysis(
    config: MimicProjectConfig,
    *,
    run_id: str,
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel"),
    feature_glob: Optional[str] = None,
    iterations: int = 2000,
    seed: int = 20260809,
    confidence: float = 0.95,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Entry point for the "therapy-response" pipeline stage: build the
    before/after pair table, then summarize the tachycardia-pattern rate
    change per medication class (and pooled across all AV-nodal-blocking
    classes) with a subject-cluster-bootstrapped confidence interval.
    """
    resolved_feature_glob = feature_glob or str(
        config.features_root
        / "_".join(panel_ids)
        / "development"
        / "split=*"
        / "part-*.parquet"
    )
    output_root = config.runs_root / str(run_id) / "therapy_response"
    output_root.mkdir(parents=True, exist_ok=True)
    pairs_path = output_root / "therapy_response_pairs.parquet"
    results_path = output_root / "therapy_response_results.parquet"
    summary_path = output_root / "therapy_response_summary.json"
    manifest_path = output_root / "therapy_response.manifest.json"

    upstream = input_artifact_fingerprints(
        {
            "feature_manifest": feature_manifest_path(resolved_feature_glob),
            "evidence_manifest": config.evidence_root / "evidence.manifest.json",
            "medication_exposure_pairs": (
                config.evidence_root / "ecg_medication_exposure_pairs.parquet"
            ),
            "master_index_manifest": (
                config.master_root / "master_index.manifest.json"
            ),
        }
    )
    input_hash = content_hash(
        {
            "version": THERAPY_RESPONSE_VERSION,
            "implementation_sources": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in (
                    "therapy_response.py",
                    "temporal_response.py",
                    "decision_research.py",
                    "drug_repolarization.py",
                )
            },
            "temporal_core_source": file_sha256(Path(__file__).parents[1] / "core" / "temporal_research.py"),
            "run_id": str(run_id),
            "panel_ids": list(panel_ids),
            "feature_glob": str(resolved_feature_glob),
            "iterations": int(iterations),
            "seed": int(seed),
            "confidence": float(confidence),
            "upstream": upstream,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.is_file():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, THERAPY_RESPONSE_VERSION):
            return previous

    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    control_pairs_path = output_root / "untreated_control_pairs.parquet"
    with catalog.connect(isolated=True) as connection:
        artifact = build_therapy_response_pairs(
            connection,
            config,
            feature_glob=resolved_feature_glob,
            output_path=pairs_path,
            progress=progress,
        )
        master_file = config.master_root / "ecg_master_index.parquet"
        medication_file = config.evidence_root / "medication_evidence.parquet"
        control_sources_available = master_file.is_file() and medication_file.is_file()
    if not control_sources_available:
        # Reported, not silently empty: the comparison is non-estimable.
        atomic_write_parquet(pd.DataFrame(columns=[
            "subject_id", "medication_class", "study_id_before", "study_id_after",
            "tachycardia_pattern_before", "rhythm_normalized_flag", "rhythm_class_before",
        ]), control_pairs_path)
        control_artifact_count = 0
    else:
        with catalog.connect(isolated=True) as connection:
            if progress:
                progress("Unbehandelte Vergleichspaare derselben Aufnahme suchen")
            raw_controls_path = output_root / "untreated_control_candidates.parquet"
            _copy_query_atomic(
                connection,
                _untreated_control_pair_query(
                    _sql_path(config.master_root / "ecg_master_index.parquet"),
                    _sql_path(config.evidence_root / "medication_evidence.parquet"),
                ),
                raw_controls_path,
            )
            feature_sql = str(resolved_feature_glob).replace("\\", "/").replace("'", "''")
            available_columns = _feature_columns(connection, feature_sql)
            master_sql = (
                None
                if "heart_rate_bpm" in available_columns
                else _sql_path(config.master_root / "ecg_master_index.parquet")
            )
            control_artifact_count = _copy_query_atomic(
                connection,
                _therapy_response_pair_query(
                    _sql_path(raw_controls_path),
                    feature_sql,
                    master_sql=master_sql,
                    feature_clause=feature_parquet_clause(connection, feature_sql),
                    temporal_available="context_temporal_research_json" in available_columns,
                ),
                control_pairs_path,
            )
            raw_controls_path.unlink(missing_ok=True)
    # Audit 5.2.6: repolarisation change after starting a QT-prolonging drug,
    # against untreated same-admission pairs.
    from .drug_repolarization import (
        QT_DRUG_REPOLARIZATION_VERSION,
        attach_marker_values_query,
        available_markers,
        qt_drug_control_pair_query,
        qt_drug_results,
        qt_drug_treated_pair_query,
    )

    qt_pairs_path = output_root / "qt_drug_repolarization_pairs.parquet"
    qt_results_path = output_root / "qt_drug_repolarization_results.parquet"
    qt_markers: tuple[str, ...] = ()
    if control_sources_available:
        with catalog.connect(isolated=True) as connection:
            if progress:
                progress("Neubeginn QT-verlaengernder Medikamente mit EKG-Paaren verbinden")
            master_sql_path = _sql_path(config.master_root / "ecg_master_index.parquet")
            medication_sql_path = _sql_path(config.evidence_root / "medication_evidence.parquet")
            qt_candidates = output_root / "qt_drug_pair_candidates.parquet"
            _copy_query_atomic(
                connection,
                "SELECT * FROM ("
                + qt_drug_treated_pair_query(master_sql_path, medication_sql_path)
                + ") UNION ALL SELECT * FROM ("
                + qt_drug_control_pair_query(master_sql_path, medication_sql_path)
                + ")",
                qt_candidates,
            )
            feature_sql = str(resolved_feature_glob).replace("\\", "/").replace("'", "''")
            qt_markers = available_markers(_feature_columns(connection, feature_sql))
            if qt_markers:
                _copy_query_atomic(
                    connection,
                    attach_marker_values_query(
                        _sql_path(qt_candidates),
                        feature_parquet_clause(connection, feature_sql),
                        qt_markers,
                    ),
                    qt_pairs_path,
                )
            qt_candidates.unlink(missing_ok=True)
    if qt_markers and qt_pairs_path.is_file():
        qt_pairs = pd.read_parquet(qt_pairs_path)
        qt_results = qt_drug_results(
            qt_pairs, qt_markers, iterations=iterations, seed=seed, confidence=confidence
        )
    else:
        qt_pairs = pd.DataFrame(columns=["subject_id", "pair_kind", "drug"])
        atomic_write_parquet(qt_pairs, qt_pairs_path)
        qt_results = pd.DataFrame(
            [{
                "scope": "all_qt_prolonging_drugs",
                "status": "non_estimable",
                "abstention_reason": (
                    "medication_or_master_source_unavailable"
                    if not control_sources_available
                    else "no_repolarization_marker_in_features"
                ),
                "stage_version": QT_DRUG_REPOLARIZATION_VERSION,
            }]
        )
    atomic_write_parquet(qt_results, qt_results_path)

    if progress:
        progress("Bootstrappe Tachykardie-Normalisierungsrate je Medikamentenklasse")
    pairs = pd.read_parquet(artifact.path)
    from .temporal_response import temporal_response_table
    from .decision_research import tracking_requirements
    temporal_results = temporal_response_table(pairs)
    temporal_path = output_root / "temporal_response_pairs.parquet"
    per_class = {
        str(medication_class): therapy_response_tachycardia_rate_change(
            group,
            iterations=iterations,
            seed=seed,
            confidence=confidence,
        )
        for medication_class, group in pairs.groupby("medication_class")
    }
    overall = therapy_response_tachycardia_rate_change(
        pairs,
        iterations=iterations,
        seed=seed,
        confidence=confidence,
    )

    def result_row(scope: str, medication_class: str, frame: pd.DataFrame, result: dict):
        n_pairs = int(result.get("n_pairs", 0))
        ci_low = float(result.get("ci_low", np.nan))
        ci_high = float(result.get("ci_high", np.nan))
        if n_pairs == 0:
            status = "non_estimable"
            abstention = "no_pairs_with_before_and_after_tachycardia_state"
            uncertainty = "non_estimable"
        elif int(iterations) <= 0:
            status = "estimable"
            abstention = ""
            uncertainty = "not_computed_scaling_pilot"
        elif not np.isfinite(ci_low) or not np.isfinite(ci_high):
            status = "estimable_point_only"
            abstention = "cluster_bootstrap_interval_not_estimable"
            uncertainty = "non_estimable"
        else:
            status = "estimable"
            abstention = ""
            uncertainty = "subject_cluster_bootstrap"
        heart_rate = pd.to_numeric(
            frame.get("heart_rate_delta_bpm", pd.Series(dtype=float)),
            errors="coerce",
        )
        pre_tachycardic = frame.loc[
            frame.get(
                "tachycardia_pattern_before",
                pd.Series(False, index=frame.index),
            ).fillna(False).astype(bool)
        ]
        normalized = pd.to_numeric(
            pre_tachycardic.get(
                "rhythm_normalized_flag",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        return {
            "scope": scope,
            "medication_class": medication_class,
            "status": status,
            "abstention_reason": abstention,
            "effect_measure": "paired_tachycardia_pattern_rate_difference",
            "effect": result.get("rate_difference"),
            "ci_low": result.get("ci_low"),
            "ci_high": result.get("ci_high"),
            "p_value": np.nan,
            "q_value": np.nan,
            "n_pairs": n_pairs,
            "n_subjects": int(result.get("n_subjects", 0)),
            "rate_before": result.get("rate_before"),
            "rate_after": result.get("rate_after"),
            "median_heart_rate_delta_bpm": (
                float(heart_rate.median()) if heart_rate.notna().any() else np.nan
            ),
            "n_pre_tachycardic": int(len(pre_tachycardic)),
            "normalization_rate_among_pre_tachycardic": (
                float(normalized.mean()) if normalized.notna().any() else np.nan
            ),
            "uncertainty_method": uncertainty,
            "bootstrap_iterations_requested": int(iterations),
            "confidence": float(confidence),
            "observational_unit": "medication_exposure_ecg_pair",
            "cluster_unit": "subject_id",
            "evidence_tier": "exploratory",
            "claim_scope": "physiological_response_association_not_causal_effect",
            "run_id": str(run_id),
            "stage_version": THERAPY_RESPONSE_VERSION,
        }

    controls = pd.read_parquet(control_pairs_path)
    master_times = None
    if len(controls):
        # The treated gap is before + after around the exposure; the control
        # gap is read from the ECG times.
        import duckdb

        with duckdb.connect() as timing:
            master_times = timing.execute(
                "SELECT study_id, ecg_time FROM read_parquet(?)",
                [str(config.master_root / "ecg_master_index.parquet")],
            ).df().set_index("study_id")["ecg_time"]
        controls["ecg_gap_hours"] = (
            (
                controls["study_id_after"].map(master_times)
                - controls["study_id_before"].map(master_times)
            ).dt.total_seconds() / 3600.0
        )
    untreated = {
        "all": untreated_comparison(
            pairs, controls, iterations=iterations, seed=seed, confidence=confidence
        )
    }
    for medication_class, group in pairs.groupby("medication_class", sort=True):
        untreated[str(medication_class)] = untreated_comparison(
            group, controls, iterations=iterations, seed=seed, confidence=confidence
        )
    rows = [result_row("all_classes", "all", pairs, overall)]
    for medication_class, group in pairs.groupby("medication_class", sort=True):
        rows.append(
            result_row(
                "medication_class",
                str(medication_class),
                group,
                per_class[str(medication_class)],
            )
        )
    for medication_class, comparison in untreated.items():
        rows.append(
            {
                "scope": "untreated_comparison",
                "medication_class": medication_class,
                "status": (
                    "estimable" if np.isfinite(comparison["difference"]) else "non_estimable"
                ),
                "abstention_reason": (
                    ""
                    if np.isfinite(comparison["difference"])
                    else "untreated_control_sources_unavailable"
                    if not control_sources_available
                    else "no_matched_pre_tachycardic_treated_and_untreated_pairs"
                ),
                "effect_measure": (
                    "normalization_rate_treated_minus_standardized_untreated"
                ),
                "effect": comparison["difference"],
                "ci_low": comparison["ci_low"],
                "ci_high": comparison["ci_high"],
                "p_value": np.nan,
                "q_value": np.nan,
                "n_pairs": comparison["n_treated_pairs"],
                "n_control_pairs": comparison["n_control_pairs"],
                "n_subjects": comparison["n_subjects"],
                "rate_before": np.nan,
                "rate_after": comparison["treated_normalization_rate"],
                "standardized_control_normalization_rate": comparison[
                    "standardized_control_normalization_rate"
                ],
                "n_strata_matched": comparison["n_strata_matched"],
                "uncertainty_method": (
                    "non_estimable"
                    if not np.isfinite(comparison["difference"])
                    else "not_computed_scaling_pilot"
                    if int(iterations) <= 0
                    else "subject_cluster_bootstrap_fixed_stratum_weights"
                ),
                "bootstrap_iterations_requested": int(iterations),
                "confidence": float(confidence),
                "observational_unit": "pre_tachycardic_ecg_pair",
                "cluster_unit": "subject_id",
                "evidence_tier": "exploratory",
                "claim_scope": (
                    "difference_against_untreated_same_admission_pairs_"
                    "matched_on_gap_and_baseline_rhythm_not_randomized"
                ),
                "run_id": str(run_id),
                "stage_version": THERAPY_RESPONSE_VERSION,
            }
        )
    results = pd.DataFrame(rows)
    n_subjects = int(pairs["subject_id"].nunique()) if len(pairs) else 0
    summary = {
        "decision_research_tracking_contract": tracking_requirements(),
        "decision_research_tracking_scope": "semantic_requirements_not_a_record_level_availability_claim",
        "version": THERAPY_RESPONSE_VERSION,
        "run_id": str(run_id),
        "n_total_pairs": int(len(pairs)),
        "temporal_response": {
            "n_pairs_before_missingness_filter": int(len(temporal_results)),
            "status_counts": temporal_results["status"].value_counts().to_dict(),
            "claim_scope": "exploratory_physiological_response_not_causal_treatment_effect",
        },
        "n_subjects": n_subjects,
        "bootstrap_iterations_requested": int(iterations),
        "confidence": float(confidence),
        "observational_unit": "medication_exposure_ecg_pair_clustered_by_subject",
        "evidence_tier": "exploratory",
        "interpretation_limit": (
            "Nearest discrete ECGs within two hours of a documented administration; "
            "this is a physiological response association, not a randomized or "
            "continuous-telemetry treatment-effect estimate."
        ),
        "upstream_fingerprints": upstream,
        "overall": {key: _json_safe(value) for key, value in overall.items()},
        "untreated_comparison": {
            name: {key: _json_safe(value) for key, value in result.items()}
            for name, result in untreated.items()
        },
        "untreated_control_definition": (
            "consecutive same-admission ECGs <= "
            f"{UNTREATED_CONTROL_MAXIMUM_GAP_HOURS:g} h apart, no AV-nodal "
            f"exposure from {UNTREATED_CONTROL_WASHOUT_HOURS:g} h before the "
            "first ECG to the second; reweighted to the treated gap bin x "
            "baseline rhythm distribution"
        ),
        "n_untreated_control_pairs": int(control_artifact_count),
        "qt_drug_repolarization": {
            "version": QT_DRUG_REPOLARIZATION_VERSION,
            "markers": list(qt_markers),
            "n_treated_pairs": int((qt_pairs.get("pair_kind", pd.Series(dtype=str)) == "treated").sum()),
            "n_untreated_pairs": int((qt_pairs.get("pair_kind", pd.Series(dtype=str)) == "untreated").sum()),
            "definition": (
                "new start of a QT-prolonging drug (no administration of any "
                "in the 7 days before); last ECG <=48 h before and last ECG "
                "2-72 h after the start; untreated consecutive same-admission "
                "pairs 2-120 h apart without exposure from 7 days before; "
                "reweighted on gap bin x treated baseline tertile"
            ),
            "claim_scope": "association_not_causal_drug_effect",
        },
        "by_medication_class": {
            medication_class: {key: _json_safe(value) for key, value in result.items()}
            for medication_class, result in per_class.items()
        },
    }
    atomic_write_parquet(results, results_path)
    atomic_write_parquet(temporal_results, temporal_path)
    atomic_write_json(summary_path, summary)
    output_paths = (
        artifact.path, results_path, summary_path, temporal_path, control_pairs_path,
        qt_pairs_path, qt_results_path,
    )
    manifest = StageManifest.complete(
        stage="therapy-response",
        stage_version=THERAPY_RESPONSE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(str(path.resolve()) for path in output_paths),
        row_counts={
            "pairs": int(artifact.row_count),
            "result_rows": int(len(results)),
            "subjects": n_subjects,
        },
        notes=(
            "run_local_outputs=True",
            "physiological_response_association_not_causal=True",
        ),
        output_integrity=capture_output_integrity(output_paths),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "HEART_RATE_SOURCE_OWN",
    "HEART_RATE_SOURCE_RECORDER",
    "THERAPY_RESPONSE_VERSION",
    "build_therapy_response_pairs",
    "run_therapy_response_analysis",
    "therapy_response_tachycardia_rate_change",
]
