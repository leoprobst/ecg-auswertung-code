from __future__ import annotations

import json
import csv
import gzip
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet

from .config import MimicProjectConfig
from .schema import read_csv_header
from .run_log import profile_query
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    capture_output_integrity,
    content_hash,
    file_content_fingerprint,
    file_fingerprint,
    operational_config_hash,
    parquet_file_integrity,
    verify_output_integrity,
)


EVENT_INGEST_STAGE_VERSION = "1.5.0"
EVENT_SCHEMA_VERSION = "1.2.0"
EVENT_REGISTRY_INTEGRITY_VERSION = "1.0.0"
EVENT_PARTITION_COUNT = 128
EVENT_BATCH_BLOCK_BYTES = 64 * 1024 * 1024
EVENT_CHECKPOINT_VERSION = "2.0.0"
EVENT_RESUME_STAGING_VERSION = "1.0.0"
EVENT_RESUME_STAGING_COMPRESSION = "zstd"
EVENT_DISK_RESERVE_BYTES = 5 * 1024**3
_VERIFIED_EVENT_REGISTRY_SIGNATURES: set[str] = set()


def _require_event_disk_space(path: Path, additional_bytes: int = 0) -> None:
    """Operational checkpoint guard, not a bound on final compression/spill size.

    Fail before another write; leave completed checkpoints and the active old
    registry intact. Concurrent writers can still consume space after checking.
    """
    if type(additional_bytes) is not int or additional_bytes < 0:
        raise ValueError("Nonnegative integer disk allowance required")
    existing = Path(path).resolve()
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    required = EVENT_DISK_RESERVE_BYTES + additional_bytes
    if free < required:
        raise OSError(28, f"Clinical event ingest disk reserve: {free} bytes free, "
            f"{required} required; completed checkpoints retained, free space before retry", str(path))

# Curated chartevents itemids (of ~4,095 total in d_items) -- chartevents
# is ~2.7x labevents' row count (433M vs 158M rows), so unlike every other
# table here it is filtered at ingest time to keep output volume bounded.
# Verified against the actual local d_items.csv.gz label text, not just
# published MIMIC-IV documentation. Vasopressor/fluid-balance signals come
# from inputevents/outputevents instead, which are ingested unfiltered.
# Chart items kept regardless of type, because they are categorical and the
# numeric rule below would drop them.
CHARTEVENTS_CURATED_ITEMIDS: tuple[int, ...] = (
    220045,  # Heart Rate
    220050,  # Arterial Blood Pressure systolic
    220051,  # Arterial Blood Pressure diastolic
    220052,  # Arterial Blood Pressure mean
    220179,  # Non Invasive Blood Pressure systolic
    220180,  # Non Invasive Blood Pressure diastolic
    220181,  # Non Invasive Blood Pressure mean
    220210,  # Respiratory Rate
    220277,  # O2 saturation pulseoxymetry
    223761,  # Temperature Fahrenheit
    223762,  # Temperature Celsius
    223849,  # Ventilator Mode
    220339,  # PEEP set
    223835,  # Inspired O2 Fraction
    220739,  # GCS - Eye Opening
    223900,  # GCS - Verbal Response
    223901,  # GCS - Motor Response
)


@dataclass(frozen=True)
class ClinicalEventTableSpec:
    table_id: str
    source_path: Path
    required_columns: tuple[str, ...]
    select_sql: str
    invalid_condition: str
    required: bool = False


def ingest_clinical_event_tables(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    specs = _clinical_event_specs(config)
    available = [spec for spec in specs if spec.source_path.is_file()]
    missing_required = [
        str(spec.source_path)
        for spec in specs
        if spec.required and not spec.source_path.is_file()
    ]
    if missing_required:
        raise FileNotFoundError(
            "Required clinical event tables are missing: "
            + ", ".join(missing_required)
        )
    schema_diagnostics = _validate_event_headers(specs)
    invalid_headers = [
        item
        for item in schema_diagnostics
        if item["status"] == "invalid_schema"
    ]
    if invalid_headers:
        raise RuntimeError(
            "Clinical event schema validation failed: "
            + "; ".join(
                f"{item['table_id']}: {item['missing_columns']}"
                for item in invalid_headers
            )
        )

    input_payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "ingest_stage_version": EVENT_INGEST_STAGE_VERSION,
        "registry_integrity_version": EVENT_REGISTRY_INTEGRITY_VERSION,
        "partition_count": EVENT_PARTITION_COUNT,
        "batch_block_bytes": EVENT_BATCH_BLOCK_BYTES,
        "resume_staging_version": EVENT_RESUME_STAGING_VERSION,
        "sources": {
            spec.table_id: _event_source_fingerprint(spec.source_path)
            for spec in available
        },
        "schemas": {
            spec.table_id: {
                "required_columns": spec.required_columns,
                "select_sql": spec.select_sql,
                "invalid_condition": spec.invalid_condition,
            }
            for spec in specs
        },
    }
    input_hash = content_hash(input_payload)
    source_fingerprint_mode = (
        "sha256"
        if any(
            "sha256" in fingerprint
            for fingerprint in input_payload["sources"].values()
        )
        else "metadata"
    )
    config_hash = content_hash(
        {
            "mimic_iv_root": str(config.mimic_iv_root.resolve()),
            "mimic_ed_root": str(config.mimic_ed_root.resolve()),
            "standardized_root": str(config.standardized_root.resolve()),
            "partition_count": EVENT_PARTITION_COUNT,
        }
    )
    base_root = config.standardized_root / "clinical_events"
    version_root = (
        base_root
        / "versions"
        / f"schema-{EVENT_SCHEMA_VERSION}-{input_hash[:16]}"
    )
    manifest_path = base_root / "clinical_events.manifest.json"
    registry_path = version_root / "clinical_event_registry.json"
    if not force and manifest_path.is_file():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash,
            config_hash,
            EVENT_INGEST_STAGE_VERSION,
        ):
            _verify_event_registry(registry_path)
            _register_event_views(config, registry_path)
            return previous
    if version_root.exists():
        _verify_event_registry(registry_path)
        manifest = _manifest_from_existing_registry(
            registry_path,
            input_hash=input_hash,
            config_hash=config_hash,
            operational_hash=operational_config_hash(config),
        )
        manifest.save(manifest_path)
        _register_event_views(config, registry_path)
        return manifest

    base_root.mkdir(parents=True, exist_ok=True)
    _require_event_disk_space(base_root)
    work_root = (
        base_root
        / "versions"
        / f".work-schema-{EVENT_SCHEMA_VERSION}-{input_hash[:16]}"
    )
    checkpoint_path = work_root / "ingest_checkpoint.json"
    checkpoint = _load_or_initialize_event_checkpoint(
        work_root,
        checkpoint_path,
        input_hash=input_hash,
        config_hash=config_hash,
    )
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        temp_root = work_root / "_duckdb_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"SET temp_directory='{_sql_path(temp_root)}'"
        )
        connection.execute("SET preserve_insertion_order=FALSE")
        for index, spec in enumerate(specs, start=1):
            if not spec.source_path.is_file():
                checkpoint["table_records"][spec.table_id] = {
                    "table_id": spec.table_id,
                    "status": "optional_missing",
                    "source_path": str(spec.source_path),
                }
                _save_event_checkpoint(checkpoint_path, checkpoint)
                continue
            if progress:
                progress(
                    "Klinische Ereignisse "
                    f"{index:,}/{len(specs):,}: {spec.table_id}"
                )
            record = _ingest_event_table_resumable(
                connection,
                spec,
                work_root=work_root,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                source_fingerprint=input_payload["sources"][spec.table_id],
                table_index=index,
                table_total=len(specs),
                progress=progress,
            )
            checkpoint["table_records"][spec.table_id] = record
            _save_event_checkpoint(checkpoint_path, checkpoint)

    table_records = [
        checkpoint["table_records"][spec.table_id]
        for spec in specs
        if spec.table_id in checkpoint["table_records"]
    ]
    row_counts: dict[str, int] = {}
    for record in table_records:
        if record.get("status") != "ok":
            continue
        table_id = str(record["table_id"])
        row_counts[table_id] = int(record["row_count"])
        row_counts[f"{table_id}_quarantine"] = int(
            record["quarantine_row_count"]
        )
        row_counts[f"{table_id}_source"] = int(
            record.get(
                "source_row_count",
                int(record["row_count"])
                + int(record["quarantine_row_count"]),
            )
        )
        row_counts[f"{table_id}_filtered"] = int(
            record.get("filtered_source_row_count", 0)
        )
    registry = {
        "registry_version": EVENT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_hash": input_hash,
        "partition_count": EVENT_PARTITION_COUNT,
        "batch_block_bytes": EVENT_BATCH_BLOCK_BYTES,
        "checkpoint_granularity": (
            "seekable_zstd_parquet_source_batch_then_transformed_batch"
        ),
        "resume_strategy": "temporary_seekable_source_batch_staging",
        "resume_staging_version": EVENT_RESUME_STAGING_VERSION,
        "resume_staging_compression": EVENT_RESUME_STAGING_COMPRESSION,
        "immutable": True,
        "integrity_version": EVENT_REGISTRY_INTEGRITY_VERSION,
        "integrity_policy": "all_partitions_sha256_parquet_footer_row_count",
        "tables": table_records,
        "schema_diagnostics": schema_diagnostics,
        "raw_data_modified": False,
    }
    atomic_write_json(
        work_root / "clinical_event_registry.json",
        registry,
    )
    checkpoint["status"] = "complete"
    checkpoint["completed_at"] = datetime.now(timezone.utc).isoformat()
    _save_event_checkpoint(checkpoint_path, checkpoint)
    _safe_remove_internal_path(
        work_root / "_duckdb_tmp",
        root=work_root,
        allow_missing=True,
    )
    version_root.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace(work_root, version_root)

    manifest = StageManifest.complete(
        stage="ingest-clinical-events",
        stage_version=EVENT_INGEST_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(version_root.resolve()),
            str(registry_path.resolve()),
        ),
        row_counts=row_counts,
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            f"schema_version={EVENT_SCHEMA_VERSION}",
            f"partition_count={EVENT_PARTITION_COUNT}",
            "immutable_content_addressed=True",
            "raw_data_modified=False",
            f"source_fingerprint_mode={source_fingerprint_mode}",
            f"registry_integrity_version={EVENT_REGISTRY_INTEGRITY_VERSION}",
            "resume_strategy=temporary_seekable_source_batch_staging",
            f"resume_staging_version={EVENT_RESUME_STAGING_VERSION}",
        ),
        output_integrity=capture_output_integrity((registry_path,)),
    )
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    _register_event_views(config, registry_path)
    return manifest


def load_clinical_event_registry(
    config: MimicProjectConfig,
) -> Optional[dict]:
    manifest_path = (
        config.standardized_root
        / "clinical_events"
        / "clinical_events.manifest.json"
    )
    if not manifest_path.is_file():
        return None
    manifest = StageManifest.load(manifest_path)
    if manifest.status != "complete":
        return None
    registry_paths = [
        Path(path)
        for path in manifest.output_paths
        if Path(path).name == "clinical_event_registry.json"
    ]
    if len(registry_paths) != 1:
        raise RuntimeError(
            "Clinical event manifest has no unique registry"
        )
    return _verify_event_registry(registry_paths[0])


def clinical_event_relation(
    config: MimicProjectConfig,
    table_id: str,
) -> Optional[str]:
    registry = load_clinical_event_registry(config)
    if registry is None:
        return None
    registry_path = Path(str(registry["_registry_path"]))
    root = registry_path.parent
    records = {
        str(item.get("table_id")): item
        for item in registry.get("tables", ())
        if item.get("status") == "ok"
    }
    record = records.get(table_id)
    if record is None:
        return None
    source_glob = root / str(record["relative_glob"])
    return (
        "read_parquet("
        f"'{_sql_path(source_glob)}', "
        "hive_partitioning=true, union_by_name=true)"
    )


def clinical_event_quarantine_relation(
    config: MimicProjectConfig,
    table_id: str,
) -> Optional[str]:
    registry = load_clinical_event_registry(config)
    if registry is None:
        return None
    registry_path = Path(str(registry["_registry_path"]))
    root = registry_path.parent
    records = {
        str(item.get("table_id")): item
        for item in registry.get("tables", ())
        if item.get("status") == "ok"
    }
    record = records.get(table_id)
    if record is None:
        return None
    relative_glob = record.get("relative_quarantine_glob")
    if not relative_glob:
        return None
    source_glob = root / str(relative_glob)
    return (
        "read_parquet("
        f"'{_sql_path(source_glob)}', "
        "hive_partitioning=true, union_by_name=true)"
    )


def numeric_chartevent_itemids(config: MimicProjectConfig) -> tuple[int, ...]:
    """Every numeric ICU chart item, resolved from ``d_items`` at ingest time.

    The filter used to be seventeen hand-picked itemids out of 4,095 -- vital
    signs, ventilator mode and GCS. Everything else the ICU charts
    (haemodynamics, respiratory mechanics, dialysis, ECMO, balances, scores)
    was discarded at ingest, and because the filter sits in the *ingest* it
    could only be widened by reading the source again and rebuilding every
    downstream artifact.

    Reading it from ``d_items`` instead costs almost nothing: the ingest
    decompresses and scans the whole 3.3 GB source either way, and the filter
    only decides what is written. A data-driven rule also survives a MIMIC
    version bump that a hand-maintained list would silently miss.

    Falls back to the curated list when ``d_items`` is unreadable, so an
    incomplete download degrades to the old behaviour rather than to nothing.
    """

    path = config.mimic_iv_root / "icu" / "d_items.csv.gz"
    if not path.is_file():
        return tuple(CHARTEVENTS_CURATED_ITEMIDS)
    resolved = set()
    try:
        # This small dictionary needs no database, spill lease or work root.
        # A resource-configuration error must not silently discard ICU items.
        with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'itemid', 'linksto', 'param_type'}.issubset(reader.fieldnames or ()):
                raise ValueError(f'Invalid ICU item dictionary schema: {path}')
            for row in reader:
                if row['linksto'] == 'chartevents' and 'numeric' in (row['param_type'] or '').lower():
                    try:
                        resolved.add(int(row['itemid']))
                    except (ValueError, TypeError):
                        continue
    except (OSError, EOFError, UnicodeError) as exc:
        logging.getLogger(__name__).warning('ICU dictionary unreadable; curated fallback: %s', exc)
        return tuple(CHARTEVENTS_CURATED_ITEMIDS)
    resolved.update(CHARTEVENTS_CURATED_ITEMIDS)
    return tuple(sorted(resolved))


def chartevents_itemid_filter_sql(config: MimicProjectConfig) -> str:
    return ", ".join(str(itemid) for itemid in numeric_chartevent_itemids(config))


def _clinical_event_specs(
    config: MimicProjectConfig,
) -> tuple[ClinicalEventTableSpec, ...]:
    hosp = config.mimic_iv_root / "hosp"
    icu = config.mimic_iv_root / "icu"
    ed = config.mimic_ed_root / "ed"
    chartevents_itemid_filter = chartevents_itemid_filter_sql(config)
    return (
        ClinicalEventTableSpec(
            "labevents",
            hosp / "labevents.csv.gz",
            (
                "labevent_id",
                "subject_id",
                "hadm_id",
                "specimen_id",
                "itemid",
                "order_provider_id",
                "charttime",
                "storetime",
                "value",
                "valuenum",
                "valueuom",
                "ref_range_lower",
                "ref_range_upper",
                "flag",
                "priority",
                "comments",
            ),
            """
                SELECT
                    TRY_CAST(labevent_id AS BIGINT) AS labevent_id,
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    TRY_CAST(specimen_id AS BIGINT) AS specimen_id,
                    TRY_CAST(itemid AS INTEGER) AS itemid,
                    NULLIF(TRIM(order_provider_id), '') AS order_provider_id,
                    TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                    TRY_CAST(storetime AS TIMESTAMP) AS storetime,
                    NULLIF(TRIM(value), '') AS value,
                    TRY_CAST(valuenum AS DOUBLE) AS valuenum,
                    NULLIF(TRIM(valueuom), '') AS valueuom,
                    TRY_CAST(ref_range_lower AS DOUBLE) AS ref_range_lower,
                    TRY_CAST(ref_range_upper AS DOUBLE) AS ref_range_upper,
                    NULLIF(TRIM(flag), '') AS flag,
                    NULLIF(TRIM(priority), '') AS priority,
                    NULLIF(TRIM(comments), '') AS comments
                FROM {raw}
                WHERE TRY_CAST(labevent_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(itemid AS INTEGER) IS NOT NULL
                  AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(labevent_id AS BIGINT) IS NULL "
                "OR TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(itemid AS INTEGER) IS NULL "
                "OR TRY_CAST(charttime AS TIMESTAMP) IS NULL"
            ),
            required=True,
        ),
        ClinicalEventTableSpec(
            "prescriptions",
            hosp / "prescriptions.csv.gz",
            (
                "subject_id",
                "hadm_id",
                "pharmacy_id",
                "poe_id",
                "poe_seq",
                "order_provider_id",
                "starttime",
                "stoptime",
                "drug_type",
                "drug",
                "formulary_drug_cd",
                "gsn",
                "ndc",
                "prod_strength",
                "dose_val_rx",
                "dose_unit_rx",
                "doses_per_24_hrs",
                "route",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    NULLIF(TRIM(pharmacy_id), '') AS pharmacy_id,
                    NULLIF(TRIM(poe_id), '') AS poe_id,
                    TRY_CAST(poe_seq AS INTEGER) AS poe_seq,
                    NULLIF(TRIM(order_provider_id), '') AS order_provider_id,
                    TRY_CAST(starttime AS TIMESTAMP) AS starttime,
                    TRY_CAST(stoptime AS TIMESTAMP) AS stoptime,
                    NULLIF(TRIM(drug_type), '') AS drug_type,
                    NULLIF(TRIM(drug), '') AS drug,
                    NULLIF(TRIM(formulary_drug_cd), '') AS formulary_drug_cd,
                    NULLIF(TRIM(gsn), '') AS gsn,
                    NULLIF(TRIM(ndc), '') AS ndc,
                    NULLIF(TRIM(prod_strength), '') AS prod_strength,
                    NULLIF(TRIM(dose_val_rx), '') AS dose_val_rx,
                    NULLIF(TRIM(dose_unit_rx), '') AS dose_unit_rx,
                    TRY_CAST(doses_per_24_hrs AS DOUBLE) AS doses_per_24_hrs,
                    NULLIF(TRIM(route), '') AS route
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(starttime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(starttime AS TIMESTAMP) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "pharmacy",
            hosp / "pharmacy.csv.gz",
            (
                "subject_id",
                "hadm_id",
                "pharmacy_id",
                "poe_id",
                "starttime",
                "stoptime",
                "medication",
                "proc_type",
                "status",
                "entertime",
                "verifiedtime",
                "route",
                "frequency",
                "infusion_type",
                "doses_per_24_hrs",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    NULLIF(TRIM(pharmacy_id), '') AS pharmacy_id,
                    NULLIF(TRIM(poe_id), '') AS poe_id,
                    TRY_CAST(starttime AS TIMESTAMP) AS starttime,
                    TRY_CAST(stoptime AS TIMESTAMP) AS stoptime,
                    NULLIF(TRIM(medication), '') AS medication,
                    NULLIF(TRIM(proc_type), '') AS proc_type,
                    NULLIF(TRIM(status), '') AS status,
                    TRY_CAST(entertime AS TIMESTAMP) AS entertime,
                    TRY_CAST(verifiedtime AS TIMESTAMP) AS verifiedtime,
                    NULLIF(TRIM(route), '') AS route,
                    NULLIF(TRIM(frequency), '') AS frequency,
                    NULLIF(TRIM(infusion_type), '') AS infusion_type,
                    TRY_CAST(doses_per_24_hrs AS DOUBLE) AS doses_per_24_hrs
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(starttime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(starttime AS TIMESTAMP) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "emar",
            hosp / "emar.csv.gz",
            (
                "subject_id",
                "hadm_id",
                "emar_id",
                "emar_seq",
                "poe_id",
                "pharmacy_id",
                "charttime",
                "medication",
                "event_txt",
                "scheduletime",
                "storetime",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    NULLIF(TRIM(emar_id), '') AS emar_id,
                    TRY_CAST(emar_seq AS INTEGER) AS emar_seq,
                    NULLIF(TRIM(poe_id), '') AS poe_id,
                    NULLIF(TRIM(pharmacy_id), '') AS pharmacy_id,
                    TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                    NULLIF(TRIM(medication), '') AS medication,
                    NULLIF(TRIM(event_txt), '') AS event_txt,
                    TRY_CAST(scheduletime AS TIMESTAMP) AS scheduletime,
                    TRY_CAST(storetime AS TIMESTAMP) AS storetime
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND NULLIF(TRIM(emar_id), '') IS NOT NULL
                  AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR NULLIF(TRIM(emar_id), '') IS NULL "
                "OR TRY_CAST(charttime AS TIMESTAMP) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "emar_detail",
            hosp / "emar_detail.csv.gz",
            (
                "subject_id",
                "emar_id",
                "emar_seq",
                "parent_field_ordinal",
                "administration_type",
                "pharmacy_id",
                "complete_dose_not_given",
                "dose_due",
                "dose_due_unit",
                "dose_given",
                "dose_given_unit",
                "product_amount_given",
                "product_unit",
                "product_code",
                "product_description",
                "infusion_rate",
                "infusion_rate_unit",
                "route",
                "site",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    NULLIF(TRIM(emar_id), '') AS emar_id,
                    TRY_CAST(emar_seq AS INTEGER) AS emar_seq,
                    TRY_CAST(parent_field_ordinal AS INTEGER)
                        AS parent_field_ordinal,
                    NULLIF(TRIM(administration_type), '')
                        AS administration_type,
                    NULLIF(TRIM(pharmacy_id), '') AS pharmacy_id,
                    NULLIF(TRIM(complete_dose_not_given), '')
                        AS complete_dose_not_given,
                    NULLIF(TRIM(dose_due), '') AS dose_due,
                    NULLIF(TRIM(dose_due_unit), '') AS dose_due_unit,
                    NULLIF(TRIM(dose_given), '') AS dose_given,
                    NULLIF(TRIM(dose_given_unit), '') AS dose_given_unit,
                    NULLIF(TRIM(product_amount_given), '')
                        AS product_amount_given,
                    NULLIF(TRIM(product_unit), '') AS product_unit,
                    NULLIF(TRIM(product_code), '') AS product_code,
                    NULLIF(TRIM(product_description), '')
                        AS product_description,
                    NULLIF(TRIM(infusion_rate), '') AS infusion_rate,
                    NULLIF(TRIM(infusion_rate_unit), '')
                        AS infusion_rate_unit,
                    NULLIF(TRIM(route), '') AS route,
                    NULLIF(TRIM(site), '') AS site
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND NULLIF(TRIM(emar_id), '') IS NOT NULL
                  AND TRY_CAST(emar_seq AS INTEGER) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR NULLIF(TRIM(emar_id), '') IS NULL "
                "OR TRY_CAST(emar_seq AS INTEGER) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "poe",
            hosp / "poe.csv.gz",
            (
                "poe_id",
                "poe_seq",
                "subject_id",
                "hadm_id",
                "ordertime",
                "order_type",
                "order_subtype",
                "transaction_type",
                "discontinue_of_poe_id",
                "discontinued_by_poe_id",
                "order_provider_id",
                "order_status",
            ),
            """
                SELECT
                    NULLIF(TRIM(poe_id), '') AS poe_id,
                    TRY_CAST(poe_seq AS INTEGER) AS poe_seq,
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    TRY_CAST(ordertime AS TIMESTAMP) AS ordertime,
                    NULLIF(TRIM(order_type), '') AS order_type,
                    NULLIF(TRIM(order_subtype), '') AS order_subtype,
                    NULLIF(TRIM(transaction_type), '') AS transaction_type,
                    NULLIF(TRIM(discontinue_of_poe_id), '')
                        AS discontinue_of_poe_id,
                    NULLIF(TRIM(discontinued_by_poe_id), '')
                        AS discontinued_by_poe_id,
                    NULLIF(TRIM(order_provider_id), '') AS order_provider_id,
                    NULLIF(TRIM(order_status), '') AS order_status
                FROM {raw}
                WHERE NULLIF(TRIM(poe_id), '') IS NOT NULL
                  AND TRY_CAST(poe_seq AS INTEGER) IS NOT NULL
                  AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(ordertime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "NULLIF(TRIM(poe_id), '') IS NULL "
                "OR TRY_CAST(poe_seq AS INTEGER) IS NULL "
                "OR TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(ordertime AS TIMESTAMP) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "poe_detail",
            hosp / "poe_detail.csv.gz",
            (
                "poe_id",
                "poe_seq",
                "subject_id",
                "field_name",
                "field_value",
            ),
            """
                SELECT
                    NULLIF(TRIM(poe_id), '') AS poe_id,
                    TRY_CAST(poe_seq AS INTEGER) AS poe_seq,
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    NULLIF(TRIM(field_name), '') AS field_name,
                    NULLIF(TRIM(field_value), '') AS field_value
                FROM {raw}
                WHERE NULLIF(TRIM(poe_id), '') IS NOT NULL
                  AND TRY_CAST(poe_seq AS INTEGER) IS NOT NULL
                  AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND NULLIF(TRIM(field_name), '') IS NOT NULL
            """,
            (
                "NULLIF(TRIM(poe_id), '') IS NULL "
                "OR TRY_CAST(poe_seq AS INTEGER) IS NULL "
                "OR TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR NULLIF(TRIM(field_name), '') IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "medrecon",
            ed / "medrecon.csv.gz",
            (
                "subject_id",
                "stay_id",
                "charttime",
                "name",
                "gsn",
                "ndc",
                "etc_rn",
                "etccode",
                "etcdescription",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(stay_id AS BIGINT) AS stay_id,
                    TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                    NULLIF(TRIM(name), '') AS name,
                    NULLIF(TRIM(gsn), '') AS gsn,
                    NULLIF(TRIM(ndc), '') AS ndc,
                    TRY_CAST(etc_rn AS INTEGER) AS etc_rn,
                    NULLIF(TRIM(etccode), '') AS etccode,
                    NULLIF(TRIM(etcdescription), '') AS etcdescription
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR TRY_CAST(charttime AS TIMESTAMP) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "pyxis",
            ed / "pyxis.csv.gz",
            (
                "subject_id",
                "stay_id",
                "charttime",
                "med_rn",
                "name",
                "gsn_rn",
                "gsn",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(stay_id AS BIGINT) AS stay_id,
                    TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                    TRY_CAST(med_rn AS INTEGER) AS med_rn,
                    NULLIF(TRIM(name), '') AS name,
                    TRY_CAST(gsn_rn AS INTEGER) AS gsn_rn,
                    NULLIF(TRIM(gsn), '') AS gsn
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR TRY_CAST(charttime AS TIMESTAMP) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "chartevents",
            icu / "chartevents.csv.gz",
            (
                "subject_id",
                "hadm_id",
                "stay_id",
                "charttime",
                "itemid",
                "value",
                "valuenum",
                "valueuom",
            ),
            f"""
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    TRY_CAST(stay_id AS BIGINT) AS icustay_id,
                    TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                    TRY_CAST(itemid AS INTEGER) AS itemid,
                    NULLIF(TRIM(value), '') AS value,
                    TRY_CAST(valuenum AS DOUBLE) AS valuenum,
                    NULLIF(TRIM(valueuom), '') AS valueuom
                FROM {{raw}}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
                  AND TRY_CAST(itemid AS INTEGER) IN ({chartevents_itemid_filter})
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR TRY_CAST(charttime AS TIMESTAMP) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "inputevents",
            icu / "inputevents.csv.gz",
            (
                "subject_id",
                "hadm_id",
                "stay_id",
                "starttime",
                "endtime",
                "itemid",
                "amount",
                "amountuom",
                "rate",
                "rateuom",
                "storetime",
                "orderid",
                "linkorderid",
                "statusdescription",
                "ordercategoryname",
                "ordercategorydescription",
                "ordercomponenttypedescription",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    TRY_CAST(stay_id AS BIGINT) AS icustay_id,
                    TRY_CAST(starttime AS TIMESTAMP) AS starttime,
                    TRY_CAST(endtime AS TIMESTAMP) AS endtime,
                    TRY_CAST(itemid AS INTEGER) AS itemid,
                    TRY_CAST(amount AS DOUBLE) AS amount,
                    NULLIF(TRIM(amountuom), '') AS amountuom,
                    TRY_CAST(rate AS DOUBLE) AS rate,
                    NULLIF(TRIM(rateuom), '') AS rateuom,
                    TRY_CAST(storetime AS TIMESTAMP) AS storetime,
                    TRY_CAST(orderid AS BIGINT) AS orderid,
                    TRY_CAST(linkorderid AS BIGINT) AS linkorderid,
                    NULLIF(TRIM(statusdescription), '') AS statusdescription,
                    NULLIF(TRIM(ordercategoryname), '') AS ordercategoryname,
                    NULLIF(TRIM(ordercategorydescription), '') AS ordercategorydescription,
                    NULLIF(TRIM(ordercomponenttypedescription), '') AS ordercomponenttypedescription
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(starttime AS TIMESTAMP) IS NOT NULL
                  AND TRY_CAST(itemid AS INTEGER) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR TRY_CAST(starttime AS TIMESTAMP) IS NULL "
                "OR TRY_CAST(itemid AS INTEGER) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "outputevents",
            icu / "outputevents.csv.gz",
            (
                "subject_id",
                "hadm_id",
                "stay_id",
                "charttime",
                "itemid",
                "value",
                "valueuom",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    TRY_CAST(stay_id AS BIGINT) AS icustay_id,
                    TRY_CAST(charttime AS TIMESTAMP) AS charttime,
                    TRY_CAST(itemid AS INTEGER) AS itemid,
                    TRY_CAST(value AS DOUBLE) AS value,
                    NULLIF(TRIM(valueuom), '') AS valueuom
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
                  AND TRY_CAST(itemid AS INTEGER) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR TRY_CAST(charttime AS TIMESTAMP) IS NULL "
                "OR TRY_CAST(itemid AS INTEGER) IS NULL"
            ),
        ),
        ClinicalEventTableSpec(
            "procedureevents",
            icu / "procedureevents.csv.gz",
            (
                "subject_id",
                "hadm_id",
                "stay_id",
                "starttime",
                "endtime",
                "itemid",
                "value",
                "valueuom",
            ),
            """
                SELECT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    TRY_CAST(stay_id AS BIGINT) AS icustay_id,
                    TRY_CAST(starttime AS TIMESTAMP) AS starttime,
                    TRY_CAST(endtime AS TIMESTAMP) AS endtime,
                    TRY_CAST(itemid AS INTEGER) AS itemid,
                    TRY_CAST(value AS DOUBLE) AS value,
                    NULLIF(TRIM(valueuom), '') AS valueuom
                FROM {raw}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(starttime AS TIMESTAMP) IS NOT NULL
                  AND TRY_CAST(itemid AS INTEGER) IS NOT NULL
            """,
            (
                "TRY_CAST(subject_id AS BIGINT) IS NULL "
                "OR TRY_CAST(stay_id AS BIGINT) IS NULL "
                "OR TRY_CAST(starttime AS TIMESTAMP) IS NULL "
                "OR TRY_CAST(itemid AS INTEGER) IS NULL"
            ),
        ),
    )


def _load_or_initialize_event_checkpoint(
    work_root: Path,
    checkpoint_path: Path,
    *,
    input_hash: str,
    config_hash: str,
) -> dict:
    if checkpoint_path.is_file():
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if (
            checkpoint.get("checkpoint_version")
            != EVENT_CHECKPOINT_VERSION
            or checkpoint.get("input_hash") != input_hash
            or checkpoint.get("config_hash") != config_hash
        ):
            raise RuntimeError(
                "Clinical event checkpoint does not match the current "
                "inputs or configuration"
            )
        checkpoint.setdefault("tables", {})
        checkpoint.setdefault("table_records", {})
        return checkpoint

    if work_root.exists() and any(work_root.iterdir()):
        raise RuntimeError(
            "Clinical event work directory exists without a valid "
            f"checkpoint: {work_root}"
        )
    work_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": EVENT_CHECKPOINT_VERSION,
        "stage_version": EVENT_INGEST_STAGE_VERSION,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "tables": {},
        "table_records": {},
    }
    _save_event_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def _save_event_checkpoint(path: Path, checkpoint: dict) -> None:
    checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, checkpoint)


def _ingest_event_table_resumable(
    connection,
    spec: ClinicalEventTableSpec,
    *,
    work_root: Path,
    checkpoint_path: Path,
    checkpoint: dict,
    source_fingerprint: dict,
    table_index: int,
    table_total: int,
    progress: Optional[Callable[[str], None]],
) -> dict:
    table_state = checkpoint["tables"].setdefault(
        spec.table_id,
        {
            "source_fingerprint": dict(source_fingerprint),
            "source_staging": {
                "version": EVENT_RESUME_STAGING_VERSION,
                "compression": EVENT_RESUME_STAGING_COMPRESSION,
                "completed_batches": {},
                "stream_complete": False,
            },
            "completed_batches": {},
            "stream_complete": False,
            "compacted_buckets": {},
            "table_complete": False,
        },
    )
    if table_state.get("source_fingerprint") != source_fingerprint:
        raise RuntimeError(
            f"Source changed while ingesting {spec.table_id}"
        )
    if table_state.get("table_complete"):
        record = checkpoint["table_records"].get(spec.table_id)
        if record is None:
            raise RuntimeError(
                f"Completed table {spec.table_id} has no registry record"
            )
        _verify_completed_event_table(work_root, record)
        return record

    columns = read_csv_header(spec.source_path)
    table_root = work_root / spec.table_id
    batch_root = table_root / "_batches"
    quarantine_root = work_root / "quarantine" / spec.table_id
    completed_batches = table_state["completed_batches"]

    if not table_state.get("stream_complete"):
        staged_batches = _stage_event_source_batches(
            spec,
            columns,
            work_root=work_root,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            table_state=table_state,
            table_index=table_index,
            table_total=table_total,
            progress=progress,
        )
        observed_batches = len(staged_batches)
        for batch_index, staged_record in enumerate(staged_batches):
            batch_key = str(batch_index)
            if batch_key in completed_batches:
                _verify_event_batch_checkpoint(
                    work_root,
                    spec.table_id,
                    batch_index,
                    completed_batches[batch_key],
                )
                continue
            batch = _read_staged_event_batch(
                work_root,
                staged_record,
                expected_batch_index=batch_index,
                integrity_preverified=True,
            )
            batch_record = _write_event_batch(
                connection,
                spec,
                batch,
                work_root=work_root,
                batch_index=batch_index,
            )
            completed_batches[batch_key] = batch_record
            table_state["source_rows_seen"] = sum(
                int(item["source_rows"])
                for item in completed_batches.values()
            )
            _save_event_checkpoint(checkpoint_path, checkpoint)
            _remove_staged_event_batch(work_root, staged_record)
            if progress:
                progress(
                    "Klinische Ereignisse "
                    f"{table_index:,}/{table_total:,}: {spec.table_id}; "
                    f"Batch {batch_index + 1:,} atomar abgeschlossen"
                )
        expected_indices = set(range(observed_batches))
        completed_indices = {int(key) for key in completed_batches}
        if completed_indices != expected_indices:
            raise RuntimeError(
                f"Non-contiguous event batches for {spec.table_id}: "
                f"expected {observed_batches}, got {len(completed_indices)}"
            )
        table_state["stream_complete"] = True
        table_state["batch_count"] = observed_batches
        _save_event_checkpoint(checkpoint_path, checkpoint)

    valid_count = sum(
        int(item["valid_rows"])
        for item in completed_batches.values()
    )
    invalid_count = sum(
        int(item["invalid_rows"])
        for item in completed_batches.values()
    )
    if valid_count:
        for bucket in range(EVENT_PARTITION_COUNT):
            bucket_key = str(bucket)
            if bucket_key in table_state["compacted_buckets"]:
                _verify_compacted_bucket(
                    work_root,
                    spec.table_id,
                    bucket,
                    table_state["compacted_buckets"][bucket_key],
                )
                continue
            source_glob = (
                batch_root
                / "batch-*"
                / f"subject_bucket={bucket}"
                / "*.parquet"
            )
            source_files = list(
                batch_root.glob(
                    f"batch-*/subject_bucket={bucket}/*.parquet"
                )
            )
            if source_files:
                target_path = (
                    table_root
                    / f"subject_bucket={bucket}"
                    / "part-00000.parquet"
                )
                if target_path.exists():
                    _safe_remove_internal_path(
                        target_path,
                        root=work_root,
                    )
                query = (
                    "SELECT * EXCLUDE (subject_bucket) FROM read_parquet("
                    f"'{_sql_path(source_glob)}', "
                    "hive_partitioning=true, union_by_name=true)"
                )
                bucket_count = _copy_query_atomic(
                    connection,
                    query,
                    target_path,
                )
            else:
                bucket_count = 0
            bucket_record = {
                "row_count": bucket_count,
                "relative_path": (
                    f"{spec.table_id}/subject_bucket={bucket}/"
                    "part-00000.parquet"
                    if bucket_count
                    else None
                ),
            }
            if bucket_count:
                bucket_record.update(
                    _event_partition_integrity(
                        target_path,
                        root=work_root,
                        expected_row_count=bucket_count,
                    )
                )
            table_state["compacted_buckets"][bucket_key] = bucket_record
            _save_event_checkpoint(checkpoint_path, checkpoint)
            if progress:
                progress(
                    "Klinische Ereignisse "
                    f"{table_index:,}/{table_total:,}: {spec.table_id}; "
                    f"Partition {bucket + 1:,}/{EVENT_PARTITION_COUNT:,} "
                    "kompaktiert"
                )
        compacted_count = sum(
            int(item["row_count"])
            for item in table_state["compacted_buckets"].values()
        )
        if compacted_count != valid_count:
            raise RuntimeError(
                f"Compaction row mismatch for {spec.table_id}: "
                f"{compacted_count:,} != {valid_count:,}"
            )
    else:
        _write_empty_event_table(
            connection,
            spec,
            columns,
            table_root=table_root,
        )

    _safe_remove_internal_path(
        batch_root,
        root=work_root,
        allow_missing=True,
    )
    _safe_remove_internal_path(
        table_root / "_source_batches",
        root=work_root,
        allow_missing=True,
    )
    if progress:
        progress(
            "Klinische Ereignisse "
            f"{table_index:,}/{table_total:,}: {spec.table_id}; "
            "pruefe Quellfingerprint"
        )
    checkpoint_source = table_state["source_fingerprint"]
    source_fingerprint = file_fingerprint(
        spec.source_path,
        include_sha256="sha256" in checkpoint_source,
    )
    for key in ("size", "mtime_ns", "sha256"):
        if (
            key in checkpoint_source
            and source_fingerprint.get(key) != checkpoint_source.get(key)
        ):
            raise RuntimeError(
                "Source changed while ingesting "
                f"{spec.table_id}: {key} mismatch"
            )
    source_row_count = sum(
        int(item["source_rows"])
        for item in completed_batches.values()
    )
    accounted_row_count = valid_count + invalid_count
    if accounted_row_count > source_row_count:
        raise RuntimeError(
            f"Event row accounting exceeds source rows for {spec.table_id}: "
            f"{accounted_row_count:,} > {source_row_count:,}"
        )
    filtered_row_count = source_row_count - accounted_row_count
    partitions = _finalized_compacted_partitions(
        work_root,
        spec.table_id,
        table_state=table_state,
        expected_row_count=valid_count,
    )
    quarantine_partitions = _capture_event_partitions(
        work_root,
        f"quarantine/{spec.table_id}/*.parquet",
        expected_row_count=invalid_count,
    )
    record = {
        "table_id": spec.table_id,
        "status": "ok",
        "source_path": str(spec.source_path.resolve()),
        "source_fingerprint": source_fingerprint,
        "relative_glob": (
            f"{spec.table_id}/subject_bucket=*/*.parquet"
        ),
        "relative_quarantine_glob": (
            f"quarantine/{spec.table_id}/*.parquet"
            if invalid_count
            else None
        ),
        "row_count": valid_count,
        "quarantine_row_count": invalid_count,
        "source_row_count": source_row_count,
        "filtered_source_row_count": filtered_row_count,
        "accounted_source_row_count": accounted_row_count,
        "partitions": partitions,
        "quarantine_partitions": quarantine_partitions,
        "batch_count": int(table_state.get("batch_count", 0)),
        "resume_strategy": "temporary_seekable_source_batch_staging",
        "source_staging_version": EVENT_RESUME_STAGING_VERSION,
        "source_staging_batch_count": int(
            table_state.get("source_staging", {}).get("batch_count", 0)
        ),
    }
    table_state["table_complete"] = True
    table_state["row_count"] = valid_count
    table_state["quarantine_row_count"] = invalid_count
    table_state["source_row_count"] = source_row_count
    table_state["filtered_source_row_count"] = filtered_row_count
    checkpoint["table_records"][spec.table_id] = record
    _save_event_checkpoint(checkpoint_path, checkpoint)
    _verify_completed_event_table(work_root, record)
    return record


def _open_event_batch_reader(
    path: Path,
    columns: list[str],
):
    return pa_csv.open_csv(
        path,
        read_options=pa_csv.ReadOptions(
            block_size=EVENT_BATCH_BLOCK_BYTES,
            use_threads=False,
            encoding="utf8",
        ),
        parse_options=pa_csv.ParseOptions(
            delimiter=",",
            quote_char='"',
            double_quote=True,
            newlines_in_values=True,
        ),
        convert_options=pa_csv.ConvertOptions(
            column_types={column: pa.string() for column in columns},
            strings_can_be_null=True,
            null_values=[""],
        ),
    )


def _stage_event_source_batches(
    spec: ClinicalEventTableSpec,
    columns: list[str],
    *,
    work_root: Path,
    checkpoint_path: Path,
    checkpoint: dict,
    table_state: dict,
    table_index: int,
    table_total: int,
    progress: Optional[Callable[[str], None]],
) -> tuple[dict, ...]:
    """Materialize a compressed CSV stream into seekable raw batches.

    ``.csv.gz`` has no portable, serializable random-access cursor.  Keeping
    the parsed source batches as temporary ZSTD Parquet files makes the
    expensive/interruptible DuckDB transformation independently resumable:
    once this staging pass is complete, a resumed run never opens the GZIP
    source and never reads already transformed raw batches.

    If the process is interrupted *during* this initial staging pass, GZIP
    decompression must replay up to the last staged batch.  Those batches are
    integrity-checked and are neither rewritten nor sent through DuckDB.
    """

    staging = table_state.setdefault(
        "source_staging",
        {
            "version": EVENT_RESUME_STAGING_VERSION,
            "compression": EVENT_RESUME_STAGING_COMPRESSION,
            "completed_batches": {},
            "stream_complete": False,
        },
    )
    if (
        staging.get("version") != EVENT_RESUME_STAGING_VERSION
        or staging.get("compression")
        != EVENT_RESUME_STAGING_COMPRESSION
    ):
        raise RuntimeError(
            f"Unsupported source staging checkpoint for {spec.table_id}"
        )
    staged = staging.setdefault("completed_batches", {})
    transformed = set(table_state.get("completed_batches", {}))
    verified_indices: set[int] = set()
    if not staging.get("stream_complete"):
        reader = _open_event_batch_reader(spec.source_path, columns)
        observed_batches = 0
        for batch_index, batch in enumerate(reader):
            observed_batches = batch_index + 1
            key = str(batch_index)
            if key in staged:
                if key not in transformed:
                    _verify_staged_event_batch(
                        work_root,
                        staged[key],
                        expected_batch_index=batch_index,
                    )
                    verified_indices.add(batch_index)
                continue
            staged[key] = _write_staged_event_batch(
                work_root,
                spec.table_id,
                batch_index,
                batch,
            )
            verified_indices.add(batch_index)
            staging["source_rows_staged"] = sum(
                int(item["row_count"]) for item in staged.values()
            )
            _save_event_checkpoint(checkpoint_path, checkpoint)
            if progress:
                progress(
                    "Klinische Ereignisse "
                    f"{table_index:,}/{table_total:,}: {spec.table_id}; "
                    f"Resume-Staging Batch {batch_index + 1:,} gesichert"
                )
        expected = set(range(observed_batches))
        observed = {int(key) for key in staged}
        if observed != expected:
            raise RuntimeError(
                f"Non-contiguous source staging for {spec.table_id}: "
                f"expected {observed_batches}, got {len(observed)}"
            )
        staging["stream_complete"] = True
        staging["batch_count"] = observed_batches
        staging["source_rows_staged"] = sum(
            int(item["row_count"]) for item in staged.values()
        )
        _save_event_checkpoint(checkpoint_path, checkpoint)

    batch_count = int(staging.get("batch_count", -1))
    if batch_count < 0 or len(staged) != batch_count:
        raise RuntimeError(
            f"Incomplete source staging checkpoint for {spec.table_id}"
        )
    records: list[dict] = []
    for batch_index in range(batch_count):
        record = staged.get(str(batch_index))
        if record is None:
            raise RuntimeError(
                f"Missing staged batch {batch_index + 1} for {spec.table_id}"
            )
        if str(batch_index) not in transformed and (
            batch_index not in verified_indices
        ):
            _verify_staged_event_batch(
                work_root,
                record,
                expected_batch_index=batch_index,
            )
        records.append(record)
    return tuple(records)


def _write_staged_event_batch(
    work_root: Path,
    table_id: str,
    batch_index: int,
    batch: pa.RecordBatch,
) -> dict:
    _require_event_disk_space(work_root, max(16 * 1024**2, 2 * batch.nbytes))
    target = (
        work_root
        / table_id
        / "_source_batches"
        / f"batch-{batch_index:08d}.parquet"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    _safe_remove_internal_path(
        temporary,
        root=work_root,
        allow_missing=True,
    )
    try:
        pa_parquet.write_table(
            pa.Table.from_batches([batch]),
            temporary,
            compression=EVENT_RESUME_STAGING_COMPRESSION,
            use_dictionary=True,
            write_statistics=False,
        )
        atomic_replace(temporary, target)
    finally:
        _safe_remove_internal_path(
            temporary,
            root=work_root,
            allow_missing=True,
        )
    record = {
        "batch_index": int(batch_index),
        "row_count": int(batch.num_rows),
    }
    record.update(
        _event_partition_integrity(
            target,
            root=work_root,
            expected_row_count=int(batch.num_rows),
        )
    )
    return record


def _verify_staged_event_batch(
    work_root: Path,
    record: dict,
    *,
    expected_batch_index: int,
) -> None:
    if int(record.get("batch_index", -1)) != int(expected_batch_index):
        raise RuntimeError(
            "Clinical event source staging batch index mismatch: "
            f"{record.get('batch_index')} != {expected_batch_index}"
        )
    relative_path = record.get("relative_path")
    if not relative_path:
        raise FileNotFoundError(
            f"Staged clinical event batch {expected_batch_index + 1} has no path"
        )
    _verify_event_partition_record(
        work_root / str(relative_path),
        record,
        expected_row_count=int(record.get("row_count", -1)),
        label=f"source staging batch {expected_batch_index + 1}",
    )


def _read_staged_event_batch(
    work_root: Path,
    record: dict,
    *,
    expected_batch_index: int,
    integrity_preverified: bool = False,
) -> pa.RecordBatch:
    if not integrity_preverified:
        _verify_staged_event_batch(
            work_root,
            record,
            expected_batch_index=expected_batch_index,
        )
    table = pa_parquet.read_table(
        work_root / str(record["relative_path"]),
        use_threads=False,
    ).combine_chunks()
    batches = table.to_batches(max_chunksize=max(1, int(table.num_rows)))
    if len(batches) != 1:
        raise RuntimeError(
            "Staged clinical event batch did not decode as one record batch: "
            f"{expected_batch_index + 1}"
        )
    return batches[0]


def _remove_staged_event_batch(work_root: Path, record: dict) -> None:
    relative_path = record.get("relative_path")
    if not relative_path:
        return
    _safe_remove_internal_path(
        work_root / str(relative_path),
        root=work_root,
        allow_missing=True,
    )


def _event_source_fingerprint(path: Path) -> dict[str, object]:
    mode = os.environ.get(
        "VCG_EVENT_SOURCE_FINGERPRINT_MODE",
        os.environ.get("VCG_INGEST_SOURCE_FINGERPRINT_MODE", "sha256"),
    )
    return file_fingerprint(path, mode=mode)


def _write_event_batch(
    connection,
    spec: ClinicalEventTableSpec,
    batch: pa.RecordBatch,
    *,
    work_root: Path,
    batch_index: int,
) -> dict:
    _require_event_disk_space(work_root, max(16 * 1024**2, 4 * batch.nbytes))
    table_root = work_root / spec.table_id
    batch_target = (
        table_root / "_batches" / f"batch-{batch_index:08d}"
    )
    quarantine_target = (
        work_root
        / "quarantine"
        / spec.table_id
        / f"batch-{batch_index:08d}.parquet"
    )
    if batch_target.exists():
        _safe_remove_internal_path(batch_target, root=work_root)
    if quarantine_target.exists():
        _safe_remove_internal_path(quarantine_target, root=work_root)

    connection.register(
        "raw_event_batch",
        pa.Table.from_batches([batch]),
    )
    try:
        selected = spec.select_sql.format(raw="raw_event_batch")
        valid_query = f"""
            SELECT
                selected.*,
                CAST(
                    MOD(ABS(subject_id), {EVENT_PARTITION_COUNT})
                    AS SMALLINT
                ) AS subject_bucket
            FROM ({selected}) selected
        """
        invalid_query = f"""
            SELECT
                *,
                'invalid required key or timestamp' AS _diagnostic
            FROM raw_event_batch
            WHERE {spec.invalid_condition}
        """
        valid_count = int(
            connection.execute(
                f"SELECT count(*) FROM ({valid_query})"
            ).fetchone()[0]
        )
        invalid_count = int(
            connection.execute(
                f"SELECT count(*) FROM ({invalid_query})"
            ).fetchone()[0]
        )
        if valid_count:
            batch_target.parent.mkdir(parents=True, exist_ok=True)
            temporary_root = (
                batch_target.parent
                / f".{batch_target.name}.{os.getpid()}.tmp"
            )
            _safe_remove_internal_path(
                temporary_root,
                root=work_root,
                allow_missing=True,
            )
            _copy_partitioned(
                connection,
                valid_query,
                temporary_root,
            )
            atomic_replace(temporary_root, batch_target)
        if invalid_count:
            _copy_query_atomic(
                connection,
                invalid_query,
                quarantine_target,
            )
    finally:
        connection.unregister("raw_event_batch")

    return {
        "source_rows": int(batch.num_rows),
        "valid_rows": valid_count,
        "invalid_rows": invalid_count,
        "relative_batch_root": (
            f"{spec.table_id}/_batches/batch-{batch_index:08d}"
            if valid_count
            else None
        ),
        "relative_quarantine_path": (
            f"quarantine/{spec.table_id}/"
            f"batch-{batch_index:08d}.parquet"
            if invalid_count
            else None
        ),
    }


def _write_empty_event_table(
    connection,
    spec: ClinicalEventTableSpec,
    columns: list[str],
    *,
    table_root: Path,
) -> None:
    target = table_root / "subject_bucket=0" / "part-00000.parquet"
    if target.exists():
        return
    empty = pa.table(
        {column: pa.array([], type=pa.string()) for column in columns}
    )
    connection.register("raw_event_batch", empty)
    try:
        selected = spec.select_sql.format(raw="raw_event_batch")
        _copy_query_atomic(
            connection,
            f"SELECT * FROM ({selected}) LIMIT 0",
            target,
        )
    finally:
        connection.unregister("raw_event_batch")


def _verify_event_batch_checkpoint(
    work_root: Path,
    table_id: str,
    batch_index: int,
    record: dict,
) -> None:
    relative_batch = record.get("relative_batch_root")
    if relative_batch:
        batch_root = work_root / str(relative_batch)
        if not list(batch_root.glob("subject_bucket=*/*.parquet")):
            raise FileNotFoundError(
                "Checkpointed clinical event batch is missing: "
                f"{table_id} batch {batch_index + 1}"
            )
    relative_quarantine = record.get("relative_quarantine_path")
    if relative_quarantine and not (
        work_root / str(relative_quarantine)
    ).is_file():
        raise FileNotFoundError(
            "Checkpointed clinical event quarantine batch is missing: "
            f"{table_id} batch {batch_index + 1}"
        )


def _verify_compacted_bucket(
    work_root: Path,
    table_id: str,
    bucket: int,
    record: dict,
) -> None:
    row_count = int(record.get("row_count", 0))
    relative_path = record.get("relative_path")
    if not row_count:
        return
    if not relative_path:
        raise FileNotFoundError(
            f"Compacted event partition is missing: "
            f"{table_id} bucket {bucket}"
        )
    path = work_root / str(relative_path)
    _verify_event_partition_record(
        path,
        record,
        expected_row_count=row_count,
        label=f"{table_id} bucket {bucket}",
    )


def _verify_completed_event_table(
    root: Path,
    record: dict,
) -> None:
    table_id = str(record["table_id"])
    table_glob = str(record["relative_glob"])
    _verify_event_partition_collection(
        root,
        table_glob,
        records=record.get("partitions"),
        expected_row_count=int(record.get("row_count", 0)),
        label=table_id,
        require_file=True,
    )
    quarantine_glob = record.get("relative_quarantine_glob")
    quarantine_count = int(record.get("quarantine_row_count", 0))
    if quarantine_glob or quarantine_count:
        if not quarantine_glob:
            raise RuntimeError(
                f"Clinical event quarantine glob is missing: {table_id}"
            )
        _verify_event_partition_collection(
            root,
            str(quarantine_glob),
            records=record.get("quarantine_partitions"),
            expected_row_count=quarantine_count,
            label=f"{table_id} quarantine",
            require_file=quarantine_count > 0,
        )


def _event_partition_integrity(
    path: Path,
    *,
    root: Path,
    expected_row_count: Optional[int] = None,
) -> dict:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RuntimeError(
            f"Clinical event partition escapes immutable root: {resolved}"
        )
    fingerprint = parquet_file_integrity(resolved, include_sha256=True)
    row_count = int(fingerprint["row_count"])
    if expected_row_count is not None and row_count != int(expected_row_count):
        raise RuntimeError(
            f"Clinical event partition row mismatch for {resolved}: "
            f"{row_count:,} != {int(expected_row_count):,}"
        )
    fingerprint.pop("path", None)
    fingerprint.pop("mtime_ns", None)
    fingerprint["relative_path"] = resolved.relative_to(
        resolved_root
    ).as_posix()
    return fingerprint


def _capture_event_partitions(
    root: Path,
    relative_glob: str,
    *,
    expected_row_count: int,
) -> list[dict]:
    paths = sorted(path for path in root.glob(relative_glob) if path.is_file())
    records = [
        _event_partition_integrity(path, root=root)
        for path in paths
    ]
    observed_row_count = sum(int(item["row_count"]) for item in records)
    if observed_row_count != int(expected_row_count):
        raise RuntimeError(
            "Clinical event partition row mismatch for "
            f"{relative_glob}: {observed_row_count:,} != "
            f"{int(expected_row_count):,}"
        )
    if not records and int(expected_row_count):
        raise FileNotFoundError(
            f"Clinical event partitions are missing: {root / relative_glob}"
        )
    return records


def _finalized_compacted_partitions(
    root: Path,
    table_id: str,
    *,
    table_state: dict,
    expected_row_count: int,
) -> list[dict]:
    relative_glob = f"{table_id}/subject_bucket=*/*.parquet"
    recorded = [
        dict(item)
        for _, item in sorted(
            table_state.get("compacted_buckets", {}).items(),
            key=lambda pair: int(pair[0]),
        )
        if item.get("relative_path")
    ]
    actual_paths = {
        path.resolve()
        for path in root.glob(relative_glob)
        if path.is_file()
    }
    recorded_paths = {
        (root / str(item["relative_path"])).resolve()
        for item in recorded
    }
    if (
        recorded
        and all("sha256" in item for item in recorded)
        and recorded_paths == actual_paths
        and sum(int(item["row_count"]) for item in recorded)
        == int(expected_row_count)
    ):
        return recorded
    return _capture_event_partitions(
        root,
        relative_glob,
        expected_row_count=expected_row_count,
    )


def _verify_event_partition_collection(
    root: Path,
    relative_glob: str,
    *,
    records: object,
    expected_row_count: int,
    label: str,
    require_file: bool,
) -> None:
    paths = sorted(path.resolve() for path in root.glob(relative_glob) if path.is_file())
    if require_file and not paths:
        raise FileNotFoundError(
            f"Clinical event partitions are missing: {label}"
        )
    resolved_root = root.resolve()
    if any(not path.is_relative_to(resolved_root) for path in paths):
        raise RuntimeError(
            f"Clinical event partition path escapes immutable root: {label}"
        )

    recorded = list(records) if isinstance(records, list) else []
    if recorded:
        expected_paths = {
            (resolved_root / str(item.get("relative_path", ""))).resolve()
            for item in recorded
        }
        if expected_paths != set(paths):
            missing = sorted(str(path) for path in expected_paths - set(paths))
            unexpected = sorted(str(path) for path in set(paths) - expected_paths)
            raise RuntimeError(
                f"Clinical event partition set mismatch for {label}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        observed = 0
        for item in recorded:
            path = (
                resolved_root / str(item["relative_path"])
            ).resolve()
            row_count = int(item.get("row_count", -1))
            _verify_event_partition_record(
                path,
                item,
                expected_row_count=row_count,
                label=label,
            )
            observed += row_count
    else:
        # Legacy registries have no per-partition hashes.  Verify every
        # discovered footer and aggregate row count so they remain safely
        # readable, while new registries receive full SHA-256 checks.
        observed = 0
        for path in paths:
            fingerprint = parquet_file_integrity(
                path,
                include_sha256=False,
            )
            observed += int(fingerprint["row_count"])
    if observed != int(expected_row_count):
        raise RuntimeError(
            f"Clinical event row mismatch for {label}: "
            f"{observed:,} != {int(expected_row_count):,}"
        )


def _verify_event_partition_record(
    path: Path,
    record: dict,
    *,
    expected_row_count: int,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Clinical event partition is missing: {label}: {path}"
        )
    structural = parquet_file_integrity(path, include_sha256=False)
    if int(structural["row_count"]) != int(expected_row_count):
        raise RuntimeError(
            f"Clinical event partition row mismatch for {label}: "
            f"{structural['row_count']} != {expected_row_count}"
        )
    if "sha256" not in record:
        return
    absolute_record = dict(record)
    absolute_record["path"] = str(path.resolve())
    if not verify_output_integrity((absolute_record,)):
        raise RuntimeError(
            f"Clinical event partition integrity failed: {label}: {path}"
        )
    if int(record.get("row_count", -1)) != int(expected_row_count):
        raise RuntimeError(
            f"Clinical event partition row mismatch for {label}: "
            f"{record.get('row_count')} != {expected_row_count}"
        )


def _safe_remove_internal_path(
    path: Path,
    *,
    root: Path,
    allow_missing: bool = False,
) -> None:
    if not path.exists():
        if allow_missing:
            return
        raise FileNotFoundError(path)
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise RuntimeError(
            f"Refusing to remove unsafe event work path: {resolved}"
        )
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def _validate_event_headers(
    specs: tuple[ClinicalEventTableSpec, ...],
) -> list[dict]:
    diagnostics: list[dict] = []
    for spec in specs:
        if not spec.source_path.is_file():
            diagnostics.append(
                {
                    "table_id": spec.table_id,
                    "status": (
                        "required_missing"
                        if spec.required
                        else "optional_missing"
                    ),
                    "missing_columns": list(spec.required_columns),
                }
            )
            continue
        columns = read_csv_header(spec.source_path)
        missing = [
            column
            for column in spec.required_columns
            if column not in set(columns)
        ]
        diagnostics.append(
            {
                "table_id": spec.table_id,
                "status": "invalid_schema" if missing else "ok",
                "column_count": len(columns),
                "missing_columns": missing,
            }
        )
    return diagnostics


def _copy_partitioned(connection, query: str, output_root: Path) -> None:
    _require_event_disk_space(output_root)
    connection.execute(
        f"COPY ({query}) TO '{_sql_path(output_root)}' ("
        "FORMAT PARQUET, COMPRESSION ZSTD, "
        "PARTITION_BY (subject_bucket), ROW_GROUP_SIZE 122880)"
    )


def _copy_query_atomic(connection, query: str, output_path: Path) -> int:
    _require_event_disk_space(output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    try:
        with profile_query(f"event-ingest:{output_path.name}") as profile:
            copy_result = connection.execute(
                f"COPY ({query}) TO '{_sql_path(temporary)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
            ).fetchone()
            if copy_result is None:
                raise RuntimeError("DuckDB COPY returned no row count")
            count = int(copy_result[0])
            atomic_replace(temporary, output_path)
            profile["rows"] = count
            return count
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_from_existing_registry(
    registry_path: Path,
    *,
    input_hash: str,
    config_hash: str,
    operational_hash: str,
) -> StageManifest:
    registry = _verify_event_registry(registry_path)
    row_counts: dict[str, int] = {}
    for item in registry.get("tables", ()):
        if item.get("status") != "ok":
            continue
        table_id = str(item["table_id"])
        row_counts[table_id] = int(item["row_count"])
        row_counts[f"{table_id}_quarantine"] = int(
            item.get("quarantine_row_count", 0)
        )
        row_counts[f"{table_id}_source"] = int(
            item.get(
                "source_row_count",
                int(item["row_count"])
                + int(item.get("quarantine_row_count", 0)),
            )
        )
        row_counts[f"{table_id}_filtered"] = int(
            item.get("filtered_source_row_count", 0)
        )
    fingerprint_mode = (
        "sha256"
        if any(
            "sha256" in item.get("source_fingerprint", {})
            for item in registry.get("tables", ())
            if item.get("status") == "ok"
        )
        else "metadata"
    )
    return StageManifest.complete(
        stage="ingest-clinical-events",
        stage_version=EVENT_INGEST_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(registry_path.parent.resolve()),
            str(registry_path.resolve()),
        ),
        row_counts=row_counts,
        notes=(
            f"operational_config_hash={operational_hash}",
            f"schema_version={EVENT_SCHEMA_VERSION}",
            f"partition_count={EVENT_PARTITION_COUNT}",
            "immutable_content_addressed=True",
            "raw_data_modified=False",
            f"source_fingerprint_mode={fingerprint_mode}",
            "registry_integrity="
            + str(registry.get("integrity_version", "legacy_structural")),
            "reused_existing_content_addressed_version=True",
        ),
        output_integrity=capture_output_integrity((registry_path,)),
    )


def _verify_event_registry(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Clinical event registry is missing: {path}"
        )
    registry = json.loads(path.read_text(encoding="utf-8"))
    if (
        registry.get("registry_version") != EVENT_SCHEMA_VERSION
        or registry.get("immutable") is not True
        or registry.get("raw_data_modified") is not False
    ):
        raise RuntimeError(
            "Clinical event registry is stale or not immutable"
        )
    root = path.parent.resolve()
    signature = _event_registry_signature(path, registry, root=root)
    if signature in _VERIFIED_EVENT_REGISTRY_SIGNATURES:
        registry["_registry_path"] = str(path.resolve())
        registry["_integrity_status"] = (
            "verified_strong_cached"
            if registry.get("integrity_version")
            else "verified_legacy_structural_cached"
        )
        return registry
    for item in registry.get("tables", ()):
        if item.get("status") != "ok":
            continue
        source_glob = (root / str(item["relative_glob"])).resolve()
        if not source_glob.is_relative_to(root):
            raise RuntimeError(
                "Clinical event registry path escapes its immutable root"
            )
        _verify_completed_event_table(root, item)
        source_row_count = int(
            item.get(
                "source_row_count",
                int(item.get("row_count", 0))
                + int(item.get("quarantine_row_count", 0)),
            )
        )
        accounted = int(item.get("row_count", 0)) + int(
            item.get("quarantine_row_count", 0)
        )
        filtered = int(item.get("filtered_source_row_count", 0))
        if source_row_count != accounted + filtered:
            raise RuntimeError(
                "Clinical event source row accounting mismatch for "
                f"{item.get('table_id')}: {source_row_count:,} != "
                f"{accounted:,} + {filtered:,}"
            )
    _VERIFIED_EVENT_REGISTRY_SIGNATURES.add(signature)
    registry["_registry_path"] = str(path.resolve())
    registry["_integrity_status"] = (
        "verified_strong"
        if registry.get("integrity_version")
        else "verified_legacy_structural"
    )
    return registry


def _event_registry_signature(
    path: Path,
    registry: dict,
    *,
    root: Path,
) -> str:
    files: list[dict[str, object]] = []
    for item in registry.get("tables", ()):
        if item.get("status") != "ok":
            continue
        globs = [str(item.get("relative_glob", ""))]
        quarantine_glob = item.get("relative_quarantine_glob")
        if quarantine_glob:
            globs.append(str(quarantine_glob))
        for relative_glob in globs:
            for candidate in sorted(root.glob(relative_glob)):
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if not resolved.is_relative_to(root):
                    raise RuntimeError(
                        "Clinical event registry path escapes its immutable root"
                    )
                stat = resolved.stat()
                files.append(
                    {
                        "relative_path": resolved.relative_to(root).as_posix(),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
    registry_stat = path.stat()
    return content_hash(
        {
            "registry_path": str(path.resolve()),
            "registry_size": int(registry_stat.st_size),
            "registry_mtime_ns": int(registry_stat.st_mtime_ns),
            "registry_sha256": file_content_fingerprint(path)["sha256"],
            "files": files,
        }
    )


def _register_event_views(
    config: MimicProjectConfig,
    registry_path: Path,
) -> None:
    registry = _verify_event_registry(registry_path)
    root = registry_path.parent
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    for item in registry.get("tables", ()):
        if item.get("status") != "ok":
            continue
        table_id = str(item["table_id"])
        catalog.create_parquet_view(
            f"standardized_{table_id}",
            root / str(item["relative_glob"]),
        )


def _raw_csv_relation(path: Path) -> str:
    return (
        "read_csv("
        f"'{_sql_path(path)}', "
        "header=true, all_varchar=true, ignore_errors=false, "
        "parallel=true, sample_size=200000)"
    )


def _sql_path(path: Path | str) -> str:
    return (
        str(Path(path).resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )
