from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import MimicProjectConfig
from .inventory import build_inventory
from .run_log import profile_query
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    capture_output_integrity,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_fingerprint,
)


INGEST_STAGE_VERSION = "1.5.0"
NOTE_TABLE_NAMES = (
    "discharge",
    "discharge_detail",
    "radiology",
    "radiology_detail",
)


@dataclass(frozen=True)
class StandardizedTable:
    name: str
    source_path: Path
    output_path: Path
    select_sql: str
    invalid_condition: str = "FALSE"
    required: bool = True


def ingest_standardized_tables(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    config.ensure_directories()
    errors = config.validate_paths()
    if errors:
        raise FileNotFoundError("\n".join(errors))

    inventory_path = config.work_root / "data_inventory.json"
    if not inventory_path.exists():
        build_inventory(config, count_rows=False, progress=progress)

    tables = _table_specs(config)
    input_fingerprints = [
        {"name": table.name, **_ingest_source_fingerprint(table.source_path)}
        for table in tables
        if table.source_path.exists()
    ]
    input_hash = content_hash(input_fingerprints)
    config_hash = scientific_config_hash(config)
    manifest_path = config.standardized_root / "ingest.manifest.json"

    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, INGEST_STAGE_VERSION):
            return previous

    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    row_counts: dict[str, int] = {}
    output_paths: list[str] = []
    diagnostics: list[dict[str, object]] = []

    with catalog.connect() as connection:
        for table in tables:
            if progress:
                progress(f"Standardisiere {table.name}")
            if not table.source_path.exists():
                if table.required:
                    raise FileNotFoundError(table.source_path)
                diagnostics.append(
                    {
                        "table": table.name,
                        "status": "optional_missing",
                        "path": str(table.source_path),
                    }
                )
                continue

            raw_relation = _raw_csv_relation(table.source_path)
            query = table.select_sql.format(raw=raw_relation)
            row_count = _copy_query_atomic(connection, query, table.output_path)
            row_counts[table.name] = row_count
            output_paths.append(str(table.output_path.resolve()))

            quarantine_path = config.quarantine_root / f"{table.name}.parquet"
            if table.invalid_condition != "FALSE":
                invalid_query = (
                    "SELECT *, 'invalid required key or timestamp' AS _diagnostic "
                    f"FROM {raw_relation} WHERE {table.invalid_condition}"
                )
                invalid_count = _copy_query_atomic(
                    connection,
                    invalid_query,
                    quarantine_path,
                )
                row_counts[f"{table.name}_quarantine"] = invalid_count
                output_paths.append(str(quarantine_path.resolve()))
            diagnostics.append(
                {
                    "table": table.name,
                    "status": "ok",
                    "rows": row_count,
                    "source": str(table.source_path),
                    "output": str(table.output_path),
                }
            )

        _derive_echo_tables(connection, config, row_counts, output_paths, progress)

    notes_coverage = _notes_coverage(config, tables)
    diagnostics_path = config.standardized_root / "ingest_diagnostics.json"
    atomic_write_json(
        diagnostics_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tables": diagnostics,
            "row_counts": row_counts,
            "notes_coverage": notes_coverage,
        },
    )
    output_paths.append(str(diagnostics_path.resolve()))
    output_integrity = capture_output_integrity(output_paths)
    fingerprint_mode = (
        "sha256"
        if any("sha256" in item for item in input_fingerprints)
        else "metadata"
    )

    manifest = StageManifest.complete(
        stage="ingest",
        stage_version=INGEST_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(output_paths),
        row_counts=row_counts,
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            f"source_fingerprint_mode={fingerprint_mode}",
            f"notes_coverage_status={notes_coverage['status']}",
            "notes_coverage_present="
            + ",".join(notes_coverage["present_tables"]),
            "notes_coverage_missing="
            + ",".join(notes_coverage["missing_tables"]),
        ),
        output_integrity=output_integrity,
    )
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    _register_standardized_views(catalog, config)
    return manifest


def _table_specs(config: MimicProjectConfig) -> list[StandardizedTable]:
    hosp = config.mimic_iv_root / "hosp"
    icu = config.mimic_iv_root / "icu"
    ed = config.mimic_ed_root / "ed"
    ecg = config.mimic_ecg_root
    out = config.standardized_root
    notes = (
        config.mimic_note_root
        if config.mimic_note_root is not None
        else Path("__mimic_note_root_not_configured__")
    )

    integer_key_invalid = (
        "TRY_CAST(subject_id AS BIGINT) IS NULL "
        "AND NULLIF(TRIM(subject_id), '') IS NOT NULL"
    )
    return [
        StandardizedTable(
            "patients",
            hosp / "patients.csv.gz",
            out / "patients.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                NULLIF(TRIM(gender), '') AS sex,
                TRY_CAST(anchor_age AS INTEGER) AS anchor_age,
                TRY_CAST(anchor_year AS INTEGER) AS anchor_year,
                NULLIF(TRIM(anchor_year_group), '') AS anchor_year_group,
                TRY_CAST(dod AS DATE) AS date_of_death
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
            """,
            integer_key_invalid,
        ),
        StandardizedTable(
            "admissions",
            hosp / "admissions.csv.gz",
            out / "admissions.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(admittime AS TIMESTAMP) AS admittime,
                TRY_CAST(dischtime AS TIMESTAMP) AS dischtime,
                TRY_CAST(deathtime AS TIMESTAMP) AS deathtime,
                NULLIF(TRIM(admission_type), '') AS admission_type,
                NULLIF(TRIM(admission_location), '') AS admission_location,
                NULLIF(TRIM(discharge_location), '') AS discharge_location,
                NULLIF(TRIM(race), '') AS race,
                NULLIF(TRIM(insurance), '') AS insurance,
                NULLIF(TRIM(marital_status), '') AS marital_status,
                NULLIF(TRIM(language), '') AS language,
                TRY_CAST(edregtime AS TIMESTAMP) AS edregtime,
                TRY_CAST(edouttime AS TIMESTAMP) AS edouttime,
                TRY_CAST(hospital_expire_flag AS TINYINT) AS hospital_expire_flag
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(hadm_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(admittime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(hadm_id AS BIGINT) IS NULL "
                "OR TRY_CAST(admittime AS TIMESTAMP) IS NULL"
            ),
        ),
        StandardizedTable(
            "transfers",
            hosp / "transfers.csv.gz",
            out / "transfers.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(transfer_id AS BIGINT) AS transfer_id,
                NULLIF(TRIM(eventtype), '') AS event_type,
                NULLIF(TRIM(careunit), '') AS care_unit,
                TRY_CAST(intime AS TIMESTAMP) AS intime,
                TRY_CAST(outtime AS TIMESTAMP) AS outtime
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(transfer_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(intime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(transfer_id AS BIGINT) IS NULL "
                "OR TRY_CAST(intime AS TIMESTAMP) IS NULL"
            ),
        ),
        StandardizedTable(
            "services",
            hosp / "services.csv.gz",
            out / "services.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(transfertime AS TIMESTAMP) AS transfer_time,
                NULLIF(TRIM(prev_service), '') AS previous_service,
                NULLIF(TRIM(curr_service), '') AS current_service
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(hadm_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(transfertime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(hadm_id AS BIGINT) IS NULL "
                "OR TRY_CAST(transfertime AS TIMESTAMP) IS NULL"
            ),
            required=False,
        ),
        StandardizedTable(
            "diagnoses_icd",
            hosp / "diagnoses_icd.csv.gz",
            out / "diagnoses_icd.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(seq_num AS INTEGER) AS seq_num,
                UPPER(REPLACE(TRIM(icd_code), '.', '')) AS icd_code,
                TRY_CAST(icd_version AS TINYINT) AS icd_version
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(hadm_id AS BIGINT) IS NOT NULL
              AND NULLIF(TRIM(icd_code), '') IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(hadm_id AS BIGINT) IS NULL "
                "OR NULLIF(TRIM(icd_code), '') IS NULL"
            ),
        ),
        StandardizedTable(
            "procedures_icd",
            hosp / "procedures_icd.csv.gz",
            out / "procedures_icd.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(seq_num AS INTEGER) AS seq_num,
                TRY_CAST(chartdate AS DATE) AS chartdate,
                UPPER(REPLACE(TRIM(icd_code), '.', '')) AS icd_code,
                TRY_CAST(icd_version AS TINYINT) AS icd_version,
                'date' AS time_precision
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(hadm_id AS BIGINT) IS NOT NULL
              AND NULLIF(TRIM(icd_code), '') IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(hadm_id AS BIGINT) IS NULL "
                "OR NULLIF(TRIM(icd_code), '') IS NULL"
            ),
        ),
        StandardizedTable(
            "d_labitems",
            hosp / "d_labitems.csv.gz",
            out / "d_labitems.parquet",
            """
            SELECT
                TRY_CAST(itemid AS INTEGER) AS itemid,
                NULLIF(TRIM(label), '') AS label,
                NULLIF(TRIM(fluid), '') AS fluid,
                NULLIF(TRIM(category), '') AS category
            FROM {raw}
            WHERE TRY_CAST(itemid AS INTEGER) IS NOT NULL
            """,
            "TRY_CAST(itemid AS INTEGER) IS NULL",
        ),
        StandardizedTable(
            "omr",
            hosp / "omr.csv.gz",
            out / "omr.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(chartdate AS DATE) AS chartdate,
                TRY_CAST(seq_num AS INTEGER) AS seq_num,
                NULLIF(TRIM(result_name), '') AS result_name,
                NULLIF(TRIM(result_value), '') AS result_value
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
            """,
            integer_key_invalid,
            required=False,
        ),
        StandardizedTable(
            "microbiologyevents",
            hosp / "microbiologyevents.csv.gz",
            out / "microbiologyevents.parquet",
            """
            SELECT
                TRY_CAST(microevent_id AS BIGINT) AS microevent_id,
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(chartdate AS DATE) AS chartdate,
                TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                NULLIF(TRIM(spec_type_desc), '') AS specimen_type,
                NULLIF(TRIM(org_name), '') AS organism,
                NULLIF(TRIM(ab_name), '') AS antibiotic,
                NULLIF(TRIM(interpretation), '') AS interpretation
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
            """,
            integer_key_invalid,
            required=False,
        ),
        StandardizedTable(
            "edstays",
            ed / "edstays.csv.gz",
            out / "edstays.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(stay_id AS BIGINT) AS ed_stay_id,
                TRY_CAST(intime AS TIMESTAMP) AS intime,
                TRY_CAST(outtime AS TIMESTAMP) AS outtime,
                NULLIF(TRIM(gender), '') AS sex,
                NULLIF(TRIM(race), '') AS race,
                NULLIF(TRIM(arrival_transport), '') AS arrival_transport,
                NULLIF(TRIM(disposition), '') AS disposition
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(intime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR TRY_CAST(intime AS TIMESTAMP) IS NULL"
            ),
        ),
        StandardizedTable(
            "triage",
            ed / "triage.csv.gz",
            out / "triage.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(stay_id AS BIGINT) AS ed_stay_id,
                TRY_CAST(temperature AS DOUBLE) AS temperature,
                TRY_CAST(heartrate AS DOUBLE) AS heart_rate,
                TRY_CAST(resprate AS DOUBLE) AS respiratory_rate,
                TRY_CAST(o2sat AS DOUBLE) AS oxygen_saturation,
                TRY_CAST(sbp AS DOUBLE) AS systolic_bp,
                TRY_CAST(dbp AS DOUBLE) AS diastolic_bp,
                NULLIF(TRIM(pain), '') AS pain,
                TRY_CAST(acuity AS DOUBLE) AS acuity,
                NULLIF(TRIM(chiefcomplaint), '') AS chief_complaint
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL"
            ),
        ),
        StandardizedTable(
            "vitalsign",
            ed / "vitalsign.csv.gz",
            out / "vitalsign.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(stay_id AS BIGINT) AS ed_stay_id,
                TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                TRY_CAST(temperature AS DOUBLE) AS temperature,
                TRY_CAST(heartrate AS DOUBLE) AS heart_rate,
                TRY_CAST(resprate AS DOUBLE) AS respiratory_rate,
                TRY_CAST(o2sat AS DOUBLE) AS oxygen_saturation,
                TRY_CAST(sbp AS DOUBLE) AS systolic_bp,
                TRY_CAST(dbp AS DOUBLE) AS diastolic_bp,
                NULLIF(TRIM(rhythm), '') AS rhythm,
                NULLIF(TRIM(pain), '') AS pain
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL"
            ),
        ),
        StandardizedTable(
            "ed_diagnosis",
            ed / "diagnosis.csv.gz",
            out / "ed_diagnosis.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(stay_id AS BIGINT) AS ed_stay_id,
                TRY_CAST(seq_num AS INTEGER) AS seq_num,
                UPPER(REPLACE(TRIM(icd_code), '.', '')) AS icd_code,
                TRY_CAST(icd_version AS TINYINT) AS icd_version,
                NULLIF(TRIM(icd_title), '') AS icd_title
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
              AND NULLIF(TRIM(icd_code), '') IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR NULLIF(TRIM(icd_code), '') IS NULL"
            ),
        ),
        StandardizedTable(
            "ecg_records",
            ecg / "record_list.csv",
            out / "ecg_records.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(study_id AS BIGINT) AS study_id,
                NULLIF(TRIM(file_name), '') AS file_name,
                TRY_CAST(ecg_time AS TIMESTAMP) AS ecg_time,
                NULLIF(TRIM(path), '') AS relative_waveform_path
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(study_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(ecg_time AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(study_id AS BIGINT) IS NULL "
                "OR TRY_CAST(ecg_time AS TIMESTAMP) IS NULL"
            ),
        ),
        StandardizedTable(
            "machine_measurements",
            ecg / "machine_measurements.csv",
            out / "machine_measurements.parquet",
            _machine_measurements_query(),
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(study_id AS BIGINT) IS NULL"
            ),
        ),
        StandardizedTable(
            "echo_study_links",
            config.echo_study_list_path,
            out / "echo_study_links.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(study_id AS BIGINT) AS echo_study_id,
                TRY_CAST(study_datetime AS TIMESTAMP) AS study_datetime,
                TRY_CAST(measurement_id AS BIGINT) AS measurement_id,
                TRY_CAST(measurement_datetime AS TIMESTAMP) AS measurement_datetime
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(measurement_id AS BIGINT) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(measurement_id AS BIGINT) IS NULL"
            ),
        ),
        StandardizedTable(
            "echo_measurements_all",
            config.echo_structured_path,
            out / "echo_measurements_all.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(measurement_id AS BIGINT) AS measurement_id,
                TRY_CAST(measurement_datetime AS TIMESTAMP) AS measurement_datetime,
                LOWER(NULLIF(TRIM(test_type), '')) AS test_type,
                LOWER(NULLIF(TRIM(measurement), '')) AS measurement,
                NULLIF(TRIM(measurement_description), '') AS measurement_description,
                NULLIF(TRIM(result), '') AS result_raw,
                TRY_CAST(NULLIF(TRIM(result), '') AS DOUBLE) AS result_numeric,
                NULLIF(TRIM(unit), '') AS unit,
                CASE
                    WHEN NULLIF(TRIM(result), '') IS NULL THEN 'missing'
                    WHEN TRY_CAST(NULLIF(TRIM(result), '') AS DOUBLE) IS NOT NULL THEN 'numeric'
                    ELSE 'categorical'
                END AS parse_status
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(measurement_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(measurement_datetime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(measurement_id AS BIGINT) IS NULL "
                "OR TRY_CAST(measurement_datetime AS TIMESTAMP) IS NULL"
            ),
        ),
        StandardizedTable(
            "icustays",
            icu / "icustays.csv.gz",
            out / "icustays.parquet",
            """
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(stay_id AS BIGINT) AS icustay_id,
                TRY_CAST(intime AS TIMESTAMP) AS intime,
                TRY_CAST(outtime AS TIMESTAMP) AS outtime,
                NULLIF(TRIM(first_careunit), '') AS first_careunit,
                NULLIF(TRIM(last_careunit), '') AS last_careunit,
                TRY_CAST(los AS DOUBLE) AS los_days
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(intime AS TIMESTAMP) IS NOT NULL
              AND TRY_CAST(outtime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR TRY_CAST(intime AS TIMESTAMP) IS NULL "
                "OR TRY_CAST(outtime AS TIMESTAMP) IS NULL"
            ),
            required=False,
        ),
        StandardizedTable(
            "d_items",
            icu / "d_items.csv.gz",
            out / "d_items.parquet",
            """
            SELECT
                TRY_CAST(itemid AS INTEGER) AS itemid,
                NULLIF(TRIM(label), '') AS label,
                NULLIF(TRIM(category), '') AS category,
                NULLIF(TRIM(unitname), '') AS unitname,
                NULLIF(TRIM(param_type), '') AS param_type,
                NULLIF(TRIM(linksto), '') AS linksto
            FROM {raw}
            WHERE TRY_CAST(itemid AS INTEGER) IS NOT NULL
            """,
            "TRY_CAST(itemid AS INTEGER) IS NULL",
            required=False,
        ),
        StandardizedTable(
            "discharge",
            _resolve_note_csv(notes, "discharge.csv"),
            out / "discharge.parquet",
            """
            SELECT
                NULLIF(TRIM(note_id), '') AS note_id,
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                NULLIF(TRIM(note_type), '') AS note_type,
                TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                text AS text
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND NULLIF(TRIM(note_id), '') IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR NULLIF(TRIM(note_id), '') IS NULL"
            ),
            required=False,
        ),
        StandardizedTable(
            "discharge_detail",
            _resolve_note_csv(notes, "discharge_detail.csv"),
            out / "discharge_detail.parquet",
            """
            SELECT
                NULLIF(TRIM(note_id), '') AS note_id,
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                NULLIF(TRIM(field_name), '') AS field_name,
                NULLIF(TRIM(field_value), '') AS field_value,
                TRY_CAST(field_ordinal AS INTEGER) AS field_ordinal
            FROM {raw}
            WHERE NULLIF(TRIM(note_id), '') IS NOT NULL
            """,
            "NULLIF(TRIM(note_id), '') IS NULL",
            required=False,
        ),
        StandardizedTable(
            "radiology",
            _resolve_note_csv(notes, "radiology.csv"),
            out / "radiology.parquet",
            """
            SELECT
                NULLIF(TRIM(note_id), '') AS note_id,
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                NULLIF(TRIM(note_type), '') AS note_type,
                TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                text AS text
            FROM {raw}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND NULLIF(TRIM(note_id), '') IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR NULLIF(TRIM(note_id), '') IS NULL"
            ),
            required=False,
        ),
        StandardizedTable(
            "radiology_detail",
            _resolve_note_csv(notes, "radiology_detail.csv"),
            out / "radiology_detail.parquet",
            """
            SELECT
                NULLIF(TRIM(note_id), '') AS note_id,
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                NULLIF(TRIM(field_name), '') AS field_name,
                NULLIF(TRIM(field_value), '') AS field_value,
                TRY_CAST(field_ordinal AS INTEGER) AS field_ordinal
            FROM {raw}
            WHERE NULLIF(TRIM(note_id), '') IS NOT NULL
            """,
            "NULLIF(TRIM(note_id), '') IS NULL",
            required=False,
        ),
    ]


def _resolve_note_csv(root: Path, filename: str) -> Path:
    """Resolve official gzip notes while retaining legacy plain-CSV support."""

    candidates = (
        root / f"{filename}.gz",
        root / filename,
        root / "note" / f"{filename}.gz",
        root / "note" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Prefer the official compressed layout in missing-input diagnostics.
    return candidates[0]


def _ingest_source_fingerprint(path: Path) -> dict[str, object]:
    mode = os.environ.get(
        "VCG_INGEST_SOURCE_FINGERPRINT_MODE",
        "sha256",
    )
    return file_fingerprint(path, mode=mode)


def _notes_coverage(
    config: MimicProjectConfig,
    tables: list[StandardizedTable],
) -> dict[str, object]:
    note_tables = {
        table.name: table
        for table in tables
        if table.name in NOTE_TABLE_NAMES
    }
    present = tuple(
        name
        for name in NOTE_TABLE_NAMES
        if name in note_tables and note_tables[name].source_path.is_file()
    )
    missing = tuple(name for name in NOTE_TABLE_NAMES if name not in present)
    if config.mimic_note_root is None:
        status = "not_configured"
    elif not present:
        status = "configured_but_missing"
    elif missing:
        status = "partial"
    else:
        status = "complete"
    return {
        "status": status,
        "configured": config.mimic_note_root is not None,
        "configured_root": (
            str(config.mimic_note_root.resolve())
            if config.mimic_note_root is not None
            else None
        ),
        "present_tables": list(present),
        "missing_tables": list(missing),
        "resolved_sources": {
            name: str(note_tables[name].source_path.resolve())
            for name in NOTE_TABLE_NAMES
            if name in note_tables
        },
        "supported_layouts": [
            "<root>/<table>.csv.gz",
            "<root>/<table>.csv",
            "<root>/note/<table>.csv.gz",
            "<root>/note/<table>.csv",
        ],
    }


def _axis_sql(column: str) -> str:
    """Machine axis in -180..180 degrees (audit D1).

    MIMIC-IV-ECG stores part of the axes as -90..+270. Values in (180, 270]
    are real extreme axes (continuous from 181 to 269) and become x - 360;
    anything above 270 (999, 29999, 32767, four-digit values) is a sentinel.
    """
    value = f"TRY_CAST({column} AS DOUBLE)"
    return (
        f"CASE WHEN {value} BETWEEN -180 AND 180 THEN {value} "
        f"WHEN {value} > 180 AND {value} <= 270 THEN {value} - 360 END"
    )


def _machine_measurements_query() -> str:
    reports = ", ".join(
        f"NULLIF(TRIM(report_{index}), '')" for index in range(18)
    )
    return f"""
        SELECT
            TRY_CAST(subject_id AS BIGINT) AS subject_id,
            TRY_CAST(study_id AS BIGINT) AS study_id,
            TRY_CAST(cart_id AS BIGINT) AS cart_id,
            TRY_CAST(ecg_time AS TIMESTAMP) AS machine_ecg_time,
            CONCAT_WS(' | ', {reports}) AS machine_report,
            NULLIF(TRIM(bandwidth), '') AS bandwidth,
            NULLIF(TRIM(filtering), '') AS filtering,
            CASE WHEN TRY_CAST(rr_interval AS DOUBLE) BETWEEN 200 AND 3000
                 THEN TRY_CAST(rr_interval AS DOUBLE) END AS rr_interval_ms,
            CASE WHEN TRY_CAST(p_onset AS DOUBLE) BETWEEN 0 AND 10000
                 THEN TRY_CAST(p_onset AS DOUBLE) END AS p_onset_ms,
            CASE WHEN TRY_CAST(p_end AS DOUBLE) BETWEEN 0 AND 10000
                 THEN TRY_CAST(p_end AS DOUBLE) END AS p_end_ms,
            CASE WHEN TRY_CAST(qrs_onset AS DOUBLE) BETWEEN 0 AND 10000
                 THEN TRY_CAST(qrs_onset AS DOUBLE) END AS qrs_onset_ms,
            CASE WHEN TRY_CAST(qrs_end AS DOUBLE) BETWEEN 0 AND 10000
                 THEN TRY_CAST(qrs_end AS DOUBLE) END AS qrs_end_ms,
            CASE WHEN TRY_CAST(t_end AS DOUBLE) BETWEEN 0 AND 10000
                 THEN TRY_CAST(t_end AS DOUBLE) END AS t_end_ms,
            {_axis_sql('p_axis')} AS p_axis_deg,
            {_axis_sql('qrs_axis')} AS qrs_axis_deg,
            {_axis_sql('t_axis')} AS t_axis_deg,
            CASE WHEN TRY_CAST(rr_interval AS DOUBLE) BETWEEN 200 AND 3000
                 THEN 60000.0 / TRY_CAST(rr_interval AS DOUBLE) END AS machine_heart_rate_bpm,
            CASE
                WHEN TRY_CAST(qrs_end AS DOUBLE) - TRY_CAST(qrs_onset AS DOUBLE)
                     BETWEEN 40 AND 300
                THEN TRY_CAST(qrs_end AS DOUBLE) - TRY_CAST(qrs_onset AS DOUBLE)
            END AS machine_qrs_duration_ms,
            CASE
                WHEN TRY_CAST(qrs_onset AS DOUBLE) - TRY_CAST(p_onset AS DOUBLE)
                     BETWEEN 60 AND 400
                THEN TRY_CAST(qrs_onset AS DOUBLE) - TRY_CAST(p_onset AS DOUBLE)
            END AS machine_pr_interval_ms,
            CASE
                WHEN TRY_CAST(t_end AS DOUBLE) - TRY_CAST(qrs_onset AS DOUBLE)
                     BETWEEN 150 AND 800
                THEN TRY_CAST(t_end AS DOUBLE) - TRY_CAST(qrs_onset AS DOUBLE)
            END AS machine_qt_interval_ms
        FROM {{raw}}
        WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
          AND TRY_CAST(study_id AS BIGINT) IS NOT NULL
    """


def _derive_echo_tables(
    connection,
    config: MimicProjectConfig,
    row_counts: dict[str, int],
    output_paths: list[str],
    progress: Optional[Callable[[str], None]],
) -> None:
    all_path = config.standardized_root / "echo_measurements_all.parquet"
    normalized = _sql_path(all_path)

    if progress:
        progress("Erzeuge Echo-Studienindex")
    index_path = config.standardized_root / "echo_study_index.parquet"
    index_query = f"""
        SELECT
            measurement_id,
            MIN(subject_id) AS subject_id,
            MIN(measurement_datetime) AS echo_time,
            MIN(test_type) AS test_type,
            COUNT(DISTINCT subject_id) > 1
                OR COUNT(DISTINCT measurement_datetime) > 1
                OR COUNT(DISTINCT test_type) > 1 AS identity_conflict,
            COUNT(*) AS source_row_count,
            COUNT(result_raw) AS populated_measurement_count
        FROM read_parquet('{normalized}')
        GROUP BY measurement_id
    """
    count = _copy_query_atomic(connection, index_query, index_path)
    row_counts["echo_study_index"] = count
    output_paths.append(str(index_path.resolve()))

    if progress:
        progress("Filtere vorhandene Echo-Messwerte")
    long_path = config.standardized_root / "echo_measurements_long.parquet"
    long_query = f"""
        SELECT *
        FROM read_parquet('{normalized}')
        WHERE result_raw IS NOT NULL
    """
    count = _copy_query_atomic(connection, long_query, long_path)
    row_counts["echo_measurements_long"] = count
    output_paths.append(str(long_path.resolve()))


def _register_standardized_views(
    catalog: DuckDBCatalog,
    config: MimicProjectConfig,
) -> None:
    for path in config.standardized_root.glob("*.parquet"):
        catalog.create_parquet_view(path.stem, path)


def _raw_csv_relation(path: Path) -> str:
    normalized = _sql_path(path)
    return (
        f"read_csv('{normalized}', header=true, all_varchar=true, "
        "auto_detect=true, ignore_errors=false, null_padding=false)"
    )


def _copy_query_atomic(connection, query: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    if temporary.exists():
        temporary.unlink()
    temporary_sql = _sql_path(temporary)
    with profile_query(f"ingest:{output_path.name}") as profile:
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


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")
