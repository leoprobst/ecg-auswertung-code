from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import MimicProjectConfig
from .splits import assert_subject_split_integrity, build_subject_splits, split_spec_hash
from .run_log import profile_query
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_content_fingerprint,
)


MASTER_STAGE_VERSION = "1.6.0"


def build_ecg_master_index(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    required = [
        config.standardized_root / name
        for name in (
            "ecg_records.parquet",
            "machine_measurements.parquet",
            "patients.parquet",
            "admissions.parquet",
            "edstays.parquet",
            "triage.parquet",
            "echo_study_index.parquet",
        )
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Standardized inputs are missing. Run 'ingest' first:\n" + "\n".join(missing)
        )

    input_hash = content_hash(
        [file_content_fingerprint(path) for path in required]
    )
    config_hash = scientific_config_hash(config)
    manifest_path = config.master_root / "master_index.manifest.json"
    master_path = config.master_root / "ecg_master_index.parquet"
    split_path = config.master_root / "subject_splits.parquet"
    summary_path = config.master_root / "ecg_master_index_summary.json"

    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, MASTER_STAGE_VERSION):
            return previous

    if progress:
        progress("Erzeuge stabile Patientensplits")
    subjects = pd.read_parquet(
        config.standardized_root / "ecg_records.parquet",
        columns=["subject_id"],
    )["subject_id"]
    split_frame = build_subject_splits(subjects, config, split_path)
    assert_subject_split_integrity(split_frame)

    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect() as connection:
        if progress:
            progress("Verknuepfe ECGs mit Aufenthalten, ED und Echo")
        query = _master_query(config)
        temporary = master_path.parent / f".{master_path.name}.{os.getpid()}.tmp"
        temporary_sql = _sql_path(temporary)
        with profile_query(f"master-index:{master_path.name}") as profile:
            copy_result = connection.execute(
                f"COPY ({query}) TO '{temporary_sql}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
            ).fetchone()
            if copy_result is None:
                raise RuntimeError("DuckDB COPY returned no row count")
            row_count = int(copy_result[0])
            profile["rows"] = row_count
        duplicate_studies = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT study_id
                    FROM read_parquet('{temporary_sql}')
                    GROUP BY study_id
                    HAVING count(*) > 1
                )
                """
            ).fetchone()[0]
        )
        if duplicate_studies:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Master index would contain {duplicate_studies} duplicate study IDs"
            )
        atomic_replace(temporary, master_path)

        summary = _build_summary(connection, master_path, config)

    atomic_write_json(summary_path, summary)
    summary_table = _summary_frame(summary)
    summary_parquet = config.master_root / "ecg_master_index_summary.parquet"
    atomic_write_parquet(summary_table, summary_parquet)

    manifest = StageManifest.complete(
        stage="build-index",
        stage_version=MASTER_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(master_path.resolve()),
            str(split_path.resolve()),
            str(summary_path.resolve()),
            str(summary_parquet.resolve()),
        ),
        row_counts={
            "ecg_master_index": row_count,
            "subjects": int(split_frame["subject_id"].nunique()),
        },
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            f"split_spec_hash={split_spec_hash(config)}",
            (
                "ehr_time_alignment=device_clock_not_externally_"
                "synchronized; clinical time windows require sensitivity "
                "analysis and cannot alone support a clinical claim"
            ),
        ),
        output_integrity=capture_output_integrity(
            (master_path, split_path, summary_path, summary_parquet)
        ),
    )
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    catalog.create_parquet_view("ecg_master_index", master_path)
    catalog.create_parquet_view("subject_splits", split_path)
    return manifest


def _master_query(config: MimicProjectConfig) -> str:
    std = config.standardized_root
    ecg = _sql_path(std / "ecg_records.parquet")
    machine = _sql_path(std / "machine_measurements.parquet")
    patients = _sql_path(std / "patients.parquet")
    admissions = _sql_path(std / "admissions.parquet")
    edstays = _sql_path(std / "edstays.parquet")
    triage = _sql_path(std / "triage.parquet")
    echo = _sql_path(std / "echo_study_index.parquet")
    splits = _sql_path(config.master_root / "subject_splits.parquet")
    ecg_root = str(config.mimic_ecg_root.resolve()).replace("\\", "/").replace("'", "''")

    icustays_path = std / "icustays.parquet"
    if icustays_path.exists():
        icustays = _sql_path(icustays_path)
        icustay_with_block = f"""
    ,
    icustay_candidates AS (
        SELECT
            e.study_id,
            i.icustay_id,
            i.intime AS icu_intime,
            i.outtime AS icu_outtime,
            i.first_careunit,
            i.los_days AS icu_los_days,
            COUNT(*) OVER (PARTITION BY e.study_id) AS candidate_count,
            ROW_NUMBER() OVER (
                PARTITION BY e.study_id
                ORDER BY i.intime DESC, i.icustay_id
            ) AS candidate_rank
        FROM ecg e
        JOIN read_parquet('{icustays}') i
          ON e.subject_id = i.subject_id
         AND e.ecg_time BETWEEN i.intime AND i.outtime
    ),
    icustay_best AS (
        SELECT * EXCLUDE(candidate_rank)
        FROM icustay_candidates
        WHERE candidate_rank = 1
    )"""
    else:
        icustay_with_block = """
    ,
    icustay_best AS (
        SELECT
            NULL::BIGINT AS study_id,
            NULL::BIGINT AS icustay_id,
            NULL::TIMESTAMP AS icu_intime,
            NULL::TIMESTAMP AS icu_outtime,
            NULL::VARCHAR AS first_careunit,
            NULL::DOUBLE AS icu_los_days,
            NULL::BIGINT AS candidate_count
        WHERE FALSE
    )"""

    return f"""
    WITH
    ecg AS (
        SELECT * FROM read_parquet('{ecg}')
    ),
    admission_candidates AS (
        SELECT
            e.study_id,
            a.hadm_id,
            a.admittime,
            a.dischtime,
            a.admission_type,
            a.admission_location,
            a.discharge_location,
            a.race,
            a.insurance,
            a.marital_status,
            a.language,
            a.hospital_expire_flag,
            COUNT(*) OVER (PARTITION BY e.study_id) AS candidate_count,
            ROW_NUMBER() OVER (
                PARTITION BY e.study_id
                ORDER BY a.admittime DESC, a.hadm_id
            ) AS candidate_rank
        FROM ecg e
        JOIN read_parquet('{admissions}') a
          ON e.subject_id = a.subject_id
         AND e.ecg_time BETWEEN a.admittime AND a.dischtime
    ),
    admission_best AS (
        SELECT * EXCLUDE(candidate_rank)
        FROM admission_candidates
        WHERE candidate_rank = 1
    ),
    ed_candidates AS (
        SELECT
            e.study_id,
            d.ed_stay_id,
            d.hadm_id AS ed_hadm_id,
            d.intime AS ed_intime,
            d.outtime AS ed_outtime,
            d.race AS ed_race,
            d.arrival_transport,
            d.disposition,
            COUNT(*) OVER (PARTITION BY e.study_id) AS candidate_count,
            ROW_NUMBER() OVER (
                PARTITION BY e.study_id
                ORDER BY d.intime DESC, d.ed_stay_id
            ) AS candidate_rank
        FROM ecg e
        JOIN read_parquet('{edstays}') d
          ON e.subject_id = d.subject_id
         AND e.ecg_time BETWEEN d.intime AND d.outtime
    ),
    ed_best AS (
        SELECT * EXCLUDE(candidate_rank)
        FROM ed_candidates
        WHERE candidate_rank = 1
    ),
    echo_candidates AS (
        SELECT
            e.study_id,
            h.measurement_id AS nearest_echo_measurement_id,
            h.echo_time AS nearest_echo_time,
            h.test_type AS nearest_echo_test_type,
            h.identity_conflict AS nearest_echo_identity_conflict,
            DATE_DIFF('second', e.ecg_time, h.echo_time) / 3600.0
                AS nearest_echo_delta_hours,
            COUNT(*) OVER (PARTITION BY e.study_id) AS candidate_count,
            ROW_NUMBER() OVER (
                PARTITION BY e.study_id
                ORDER BY ABS(DATE_DIFF('second', e.ecg_time, h.echo_time)),
                         h.echo_time,
                         h.measurement_id
            ) AS candidate_rank
        FROM ecg e
        JOIN read_parquet('{echo}') h
          ON e.subject_id = h.subject_id
         AND h.echo_time BETWEEN e.ecg_time - INTERVAL 90 DAY
                             AND e.ecg_time + INTERVAL 90 DAY
         AND NOT h.identity_conflict
    ),
    echo_best AS (
        SELECT * EXCLUDE(candidate_rank)
        FROM echo_candidates
        WHERE candidate_rank = 1
    ){icustay_with_block}
    SELECT
        e.subject_id,
        e.study_id,
        e.ecg_time,
        '{ecg_root}/' || REPLACE(e.relative_waveform_path, '\\\\', '/')
            AS waveform_path,
        e.relative_waveform_path,
        COALESCE(a.hadm_id, d.ed_hadm_id) AS hadm_id,
        d.ed_stay_id,
        p.sex,
        p.anchor_age + EXTRACT(year FROM e.ecg_time) - p.anchor_year AS age,
        p.anchor_age >= 91 AS age_deidentified_over_89,
        -- MIMIC-IV shifts every patient's dates by a random per-patient
        -- offset, so EXTRACT(year FROM ecg_time) carries no real calendar
        -- information across patients (anchor_year spans 2110..2208 inside
        -- every true 3-year group). anchor_year_group is the only field that
        -- states the real era, and the offset within it stays meaningful
        -- because it is constant per patient.
        p.anchor_year_group,
        EXTRACT(year FROM e.ecg_time) - p.anchor_year
            AS ecg_years_after_anchor,
        COALESCE(a.race, d.ed_race) AS race,
        a.admission_type,
        a.admission_location,
        a.discharge_location,
        a.insurance,
        a.marital_status,
        a.language,
        a.hospital_expire_flag,
        a.admittime,
        a.dischtime,
        a.candidate_count AS admission_candidate_count,
        COALESCE(a.candidate_count, 0) > 1 AS admission_match_ambiguous,
        d.ed_intime,
        d.ed_outtime,
        d.arrival_transport,
        d.disposition AS ed_disposition,
        d.candidate_count AS ed_candidate_count,
        COALESCE(d.candidate_count, 0) > 1 AS ed_match_ambiguous,
        t.chief_complaint,
        t.acuity AS triage_acuity,
        t.heart_rate AS triage_heart_rate,
        t.systolic_bp AS triage_systolic_bp,
        t.diastolic_bp AS triage_diastolic_bp,
        t.oxygen_saturation AS triage_oxygen_saturation,
        t.temperature AS triage_temperature,
        t.respiratory_rate AS triage_respiratory_rate,
        TRY_CAST(t.pain AS DOUBLE) AS triage_pain_score,
        m.cart_id AS machine_cart_id,
        m.machine_ecg_time,
        ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time))
            AS machine_time_mismatch_seconds,
        ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) > 60
            AS machine_time_mismatch,
        CASE
            WHEN m.study_id IS NULL THEN 'machine_record_unavailable'
            WHEN m.machine_ecg_time IS NULL THEN 'machine_time_unavailable'
            WHEN ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60
                THEN 'time_consistent_le_60s'
            ELSE 'time_mismatch_gt_60s'
        END AS machine_metadata_temporal_status,
        COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) AS machine_metadata_usable_for_primary,
        m.machine_report,
        m.bandwidth AS machine_bandwidth,
        m.filtering AS machine_filtering,
        m.machine_heart_rate_bpm AS raw_machine_heart_rate_bpm,
        m.machine_pr_interval_ms AS raw_machine_pr_interval_ms,
        m.machine_qrs_duration_ms AS raw_machine_qrs_duration_ms,
        m.machine_qt_interval_ms AS raw_machine_qt_interval_ms,
        m.p_axis_deg AS raw_machine_p_axis_deg,
        m.qrs_axis_deg AS raw_machine_qrs_axis_deg,
        m.t_axis_deg AS raw_machine_t_axis_deg,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.machine_heart_rate_bpm END AS machine_heart_rate_bpm,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.machine_pr_interval_ms END AS machine_pr_interval_ms,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.machine_qrs_duration_ms END AS machine_qrs_duration_ms,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.machine_qt_interval_ms END AS machine_qt_interval_ms,
        -- The recorder's own global fiducials for the identical waveform.
        -- They make the delineation bias measurable inside MIMIC itself and
        -- let the interval-threshold comparator criteria be recomputed on an
        -- independent boundary estimate.
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.p_onset_ms END AS machine_p_onset_ms,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.p_end_ms END AS machine_p_end_ms,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.qrs_onset_ms END AS machine_qrs_onset_ms,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.qrs_end_ms END AS machine_qrs_end_ms,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.t_end_ms END AS machine_t_end_ms,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) AND m.p_end_ms IS NOT NULL AND m.p_onset_ms IS NOT NULL
            THEN m.p_end_ms - m.p_onset_ms
        END AS machine_p_duration_ms,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.p_axis_deg END AS machine_p_axis_deg,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.qrs_axis_deg END AS machine_qrs_axis_deg,
        CASE WHEN COALESCE(
            ABS(DATE_DIFF('second', e.ecg_time, m.machine_ecg_time)) <= 60,
            FALSE
        ) THEN m.t_axis_deg END AS machine_t_axis_deg,
        h.nearest_echo_measurement_id,
        h.nearest_echo_time,
        h.nearest_echo_test_type,
        h.nearest_echo_delta_hours,
        h.candidate_count AS echo_candidate_count_90d,
        icu.icustay_id,
        icu.icu_intime,
        icu.icu_outtime,
        icu.first_careunit AS icu_first_careunit,
        icu.icu_los_days,
        icu.candidate_count AS icustay_candidate_count,
        COALESCE(icu.candidate_count, 0) > 1 AS icustay_match_ambiguous,
        s.split,
        s.subject_split_hash,
        s.split_spec_hash,
        'not_checked' AS waveform_status,
        'not_extracted' AS feature_extraction_status,
        NULL::DOUBLE AS signal_quality_score,
        NULL::VARCHAR AS signal_quality_tier,
        NULL::DOUBLE AS fiducial_confidence,
        'record_list.ecg_time' AS ecg_time_source,
        'device_clock_not_externally_synchronized' AS ehr_time_alignment_status,
        FALSE AS ehr_time_alignment_confirmatory_eligible,
        CASE
            WHEN d.ed_stay_id IS NOT NULL THEN 'overlaps_ed_stay'
            WHEN a.hadm_id IS NOT NULL THEN 'overlaps_hospital_admission'
            ELSE 'no_ehr_encounter_overlap'
        END AS ehr_encounter_overlap_status
    FROM ecg e
    LEFT JOIN read_parquet('{patients}') p USING (subject_id)
    LEFT JOIN read_parquet('{machine}') m USING (subject_id, study_id)
    LEFT JOIN admission_best a USING (study_id)
    LEFT JOIN ed_best d USING (study_id)
    LEFT JOIN read_parquet('{triage}') t
      ON d.ed_stay_id = t.ed_stay_id
     AND e.subject_id = t.subject_id
    LEFT JOIN echo_best h USING (study_id)
    LEFT JOIN icustay_best icu USING (study_id)
    LEFT JOIN read_parquet('{splits}') s USING (subject_id)
    """


def _build_summary(connection, master_path: Path, config: MimicProjectConfig) -> dict:
    path = _sql_path(master_path)
    base = connection.execute(
        f"""
        SELECT
            COUNT(*) AS ecg_count,
            COUNT(DISTINCT subject_id) AS subject_count,
            COUNT(*) FILTER (WHERE admittime IS NOT NULL) AS hospital_match_count,
            COUNT(*) FILTER (WHERE ed_stay_id IS NOT NULL) AS ed_match_count,
            COUNT(*) FILTER (WHERE nearest_echo_measurement_id IS NOT NULL)
                AS echo_match_90d_count,
            COUNT(*) FILTER (
                WHERE ABS(nearest_echo_delta_hours) <= 24 * 30
            ) AS echo_match_30d_count,
            COUNT(*) FILTER (
                WHERE ABS(nearest_echo_delta_hours) <= 24 * 7
            ) AS echo_match_7d_count,
            COUNT(*) FILTER (WHERE machine_time_mismatch) AS machine_time_mismatch_count,
            COUNT(*) FILTER (WHERE admission_match_ambiguous)
                AS admission_ambiguous_count,
            COUNT(*) FILTER (WHERE ed_match_ambiguous) AS ed_ambiguous_count,
            COUNT(*) FILTER (WHERE subject_id IS NULL) AS missing_subject_count,
            COUNT(*) FILTER (WHERE ecg_time IS NULL) AS invalid_ecg_time_count
        FROM read_parquet('{path}')
        """
    ).fetchdf().iloc[0].to_dict()
    split_rows = connection.execute(
        f"""
        SELECT split, COUNT(*) AS ecg_count, COUNT(DISTINCT subject_id) AS subject_count
        FROM read_parquet('{path}')
        GROUP BY split
        ORDER BY split
        """
    ).fetchdf().to_dict(orient="records")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "MIMIC-IV-ECG clinical validation",
        "releases": {
            "mimic_iv": config.mimic_iv_release,
            "mimic_ed": config.mimic_ed_release,
            "mimic_ecg": config.mimic_ecg_release,
            "mimic_echo": config.mimic_echo_release,
        },
        "counts": {key: int(value) for key, value in base.items()},
        "splits": split_rows,
        "split_spec_hash": split_spec_hash(config),
        "canonical_ecg_time": (
            "record_list.ecg_time; invalid source rows are quarantined"
        ),
        "ehr_time_alignment": {
            "status": "device_clock_not_externally_synchronized",
            "confirmatory_eligible": False,
            "interpretation": (
                "MIMIC-IV-ECG v1.0 documents that ECG machine clocks were "
                "often not synchronized to external clinical systems. "
                "Encounter overlap and narrow event windows are therefore "
                "retrospective association anchors requiring sensitivity "
                "analysis, not minute-resolved clinical ground truth."
            ),
            "source": "https://physionet.org/content/mimic-iv-ecg/1.0/",
        },
    }


def _summary_frame(summary: dict) -> pd.DataFrame:
    rows = [
        {"metric": key, "value": value}
        for key, value in summary["counts"].items()
    ]
    for split in summary["splits"]:
        rows.append(
            {
                "metric": f"split_{split['split']}_ecgs",
                "value": split["ecg_count"],
            }
        )
        rows.append(
            {
                "metric": f"split_{split['split']}_subjects",
                "value": split["subject_count"],
            }
        )
    return pd.DataFrame(rows)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")
