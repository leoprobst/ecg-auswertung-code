from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, cast

from .config import MimicProjectConfig, scientific_config_payload
from .event_ingest import (
    clinical_event_quarantine_relation,
    clinical_event_relation,
    load_clinical_event_registry,
)
from .label_registry import LABEL_RULE_VERSION, default_label_definitions
from .echo_segments import resting_wall_motion_coverage_sql
from .lab_intervals import (
    censored_result_sql, upper_reference_state_sql,
    interval_window_aggregates, troponin_interval_pivot,
)
from .note_labels import read_note_coverage
from .jobs import JobCancellationRequested, JobPauseRequested
from .run_log import profile_query
from .mappings import (
    AV_NODAL_BLOCKING_MEDICATION_CLASSES,
    ICD_GROUPS,
    ICU_CHARTEVENTS_ITEMIDS,
    ICU_VASOPRESSOR_ITEMIDS,
    ICU_VENTILATION_ITEMIDS,
    LAB_ITEM_GROUPS,
    MAPPING_VERSION,
    MEDICATION_PATTERNS,
    PROCEDURE_GROUPS,
    TYPICAL_ENDOCARDITIS_ORGANISMS,
    lab_group_case_sql,
)
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    capture_output_integrity,
    content_hash,
    file_content_fingerprint,
    operational_config_hash,
)


EVIDENCE_STAGE_VERSION = "2.30.0"  # thesis T3: specimen_id in labs.parquet
EVIDENCE_CHECKPOINT_VERSION = "1.0.0"
# The medication-context join exceeded the default 16 GiB spill cap with
# a 4 GiB RAM budget. Reserve the entire allowance through DuckDBCatalog;
# never raise the SQL cap independently of the shared 96 GiB ledger.
# Operational only: query semantics and checkpoint identity are unchanged.
EVIDENCE_BUILD_SPILL_GIB = 48


@dataclass(frozen=True)
class EvidenceArtifact:
    name: str
    path: Path
    row_count: int


class _AttestationResourceConfig:
    """Constrain one-time shadow migration without changing run settings.

    Single-threaded and in low-memory mode, because this rebuild is a
    correctness check rather than production work and must not crowd out the
    run. It does get the project's own DuckDB budget, though: it executes the
    *same* queries as the real build, so a tighter cap does not make it
    cheaper -- it makes it impossible. The former six-gigabyte ceiling ran out
    inside the laboratory window query, and the caller reported that as
    "exact event-layer equivalence could not be proven", which reads as a
    finding about the data rather than about the memory limit.
    """

    def __init__(self, base: MimicProjectConfig) -> None:
        self._base = base

    @property
    def duckdb_memory_limit_gb(self) -> int:
        # Admission control and ``stage_worker_memory_scope`` already reserve
        # OS/Python headroom before this code runs.  Re-probing *available*
        # memory here is incorrect: it includes memory already owned by this
        # admitted worker and caused the 14 GB evidence job to shrink its
        # shadow rebuild to 2 GB mid-stage.  The identical laboratory query is
        # known to need more than 6 GB.  Use the deterministic evidence-stage
        # ceiling instead; a direct CLI invocation remains bounded at 10 GB.
        budget = max(2, int(self._base.duckdb_memory_limit_gb))
        active_limit = os.environ.get("VCG_JOB_MEMORY_GB")
        if active_limit:
            try:
                budget = min(budget, max(2, int(float(active_limit))))
            except ValueError:
                pass
        return max(2, min(budget, 10))

    @property
    def workers(self) -> int:
        return 1

    @property
    def evidence_low_memory_mode(self) -> bool:
        return True

    def to_dict(self) -> dict:
        return self._base.to_dict()

    def __getattr__(self, name: str):
        return getattr(self._base, name)


def build_clinical_evidence(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    master_path = config.master_root / "ecg_master_index.parquet"
    required = [
        master_path,
        config.evidence_root / "machine_report_labels.parquet",
        config.evidence_root / "note_encounter_flags.parquet",
        config.standardized_root / "diagnoses_icd.parquet",
        config.standardized_root / "procedures_icd.parquet",
        config.standardized_root / "echo_measurements_long.parquet",
        config.standardized_root / "echo_study_index.parquet",
        config.standardized_root / "admissions.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required master/standardized data are missing:\n" + "\n".join(missing)
        )

    raw_sources = [
        config.mimic_iv_root / "hosp" / "labevents.csv.gz",
        config.mimic_iv_root / "hosp" / "emar.csv.gz",
        config.mimic_iv_root / "hosp" / "pharmacy.csv.gz",
        config.mimic_iv_root / "hosp" / "prescriptions.csv.gz",
        config.mimic_iv_root / "hosp" / "emar_detail.csv.gz",
        config.mimic_iv_root / "hosp" / "poe.csv.gz",
        config.mimic_iv_root / "hosp" / "poe_detail.csv.gz",
        config.mimic_ed_root / "ed" / "pyxis.csv.gz",
        config.mimic_ed_root / "ed" / "medrecon.csv.gz",
    ]
    event_registry = load_clinical_event_registry(config)
    event_registry_input = (
        {
            "input_hash": event_registry.get("input_hash"),
            "registry_path": event_registry.get("_registry_path"),
            "registry_version": event_registry.get("registry_version"),
        }
        if event_registry is not None
        else None
    )
    input_hash = content_hash(
        {
            "required": [
                file_content_fingerprint(path)
                for path in required
                if path.exists()
            ],
            "raw_sources": [
                file_content_fingerprint(path)
                for path in raw_sources
                if path.exists()
            ] if event_registry is None else [],
            "clinical_event_registry": event_registry_input,
        }
    )
    config_hash = content_hash(
        {
            "project": scientific_config_payload(config),
            "mapping_version": MAPPING_VERSION,
            "label_rule_version": LABEL_RULE_VERSION,
            "labels": [item.to_dict() for item in default_label_definitions()],
        }
    )
    manifest_path = config.evidence_root / "evidence.manifest.json"
    previous: Optional[StageManifest] = None
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, EVIDENCE_STAGE_VERSION):
            return previous
    if (
        manifest_path.is_file()
        and _has_bound_development_features(config)
    ):
        previous = previous or StageManifest.load(manifest_path)
        return _reuse_bound_evidence_after_exact_attestation(
            config,
            previous,
            progress=progress,
        )
    # The stage version belongs in the checkpoint identity, not just in the
    # manifest. Without it a version bump rejects the manifest -- correctly,
    # the computation changed -- and then resumes from a checkpoint that still
    # reports every group finished, so the "rebuild" writes a fresh manifest
    # over untouched artifacts and the change never reaches the data.
    checkpoint_path = (
        config.evidence_root
        / ".checkpoints"
        / f"evidence-{EVIDENCE_STAGE_VERSION}-{input_hash[:20]}.json"
    )
    checkpoint = _load_evidence_checkpoint(
        checkpoint_path,
        input_hash=input_hash,
        config_hash=config_hash,
    )

    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
        spill_limit_gb=EVIDENCE_BUILD_SPILL_GIB,
    )
    artifacts: list[EvidenceArtifact] = []
    groups = (
        (
            "laboratory",
            lambda connection: _build_lab_evidence(
                connection,
                config,
                progress,
            ),
        ),
        (
            "codes",
            lambda connection: _build_code_evidence(
                connection,
                config,
                progress,
            ),
        ),
        (
            "medications",
            lambda connection: _build_medication_evidence(
                connection,
                config,
                progress,
                query_checkpoint=(checkpoint_path, checkpoint),
            ),
        ),
        (
            "echo",
            lambda connection: _build_echo_evidence(
                connection,
                config,
                progress,
            ),
        ),
        (
            "context",
            lambda connection: _build_context_features(
                connection,
                config,
                progress,
            ),
        ),
    )
    with catalog.connect() as connection:
        # Evidence joins can spill even during the ordinary canonical build,
        # not only during the constrained equivalence rebuild.  Keeping this
        # behind ``evidence_low_memory_mode`` made the six-thread production
        # path retain insertion-order state until it exhausted the 10 GiB
        # DuckDB budget, while the otherwise identical one-thread shadow
        # build completed successfully.  Row order is not part of any
        # evidence contract; every downstream relation joins by explicit
        # keys.  Give both paths the same bounded spill behaviour.
        spill_root = config.work_root / "tmp" / "evidence_duckdb"
        spill_root.mkdir(parents=True, exist_ok=True)
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            "SET temp_directory=?",
            [str(spill_root.resolve())],
        )
        for group_index, (group_id, builder) in enumerate(groups, start=1):
            resumed = _resume_evidence_group(
                checkpoint,
                group_id=group_id,
                progress=progress,
            )
            if resumed is None:
                resumed = builder(connection)
                _save_evidence_group(
                    checkpoint_path,
                    checkpoint,
                    group_id=group_id,
                    artifacts=resumed,
                )
                status = "atomar abgeschlossen"
            else:
                status = "vom Checkpoint fortgesetzt"
            artifacts.extend(resumed)
            if progress:
                progress(
                    f"Klinische Evidenz {group_index:,}/{len(groups):,}: "
                    f"{group_id} {status}"
                )

    # The evidence stage's own copy of the definitions it was built against,
    # kept under the evidence root. It used to be written to the labels root
    # under the same name build-labels uses for *its* definitions file; the
    # two stages then overwrote each other's recorded output, every later run
    # found one manifest's integrity broken, recomputed evidence (40 min),
    # which broke the labels manifest, which recomputed labels (30 min) --
    # the reason A091 to A093 each restarted from the evidence build.
    definitions_path = config.evidence_root / "label_definitions.json"
    atomic_write_json(
        definitions_path,
        {
            "rule_version": LABEL_RULE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "definitions": [
                definition.to_dict()
                for definition in default_label_definitions()
            ],
        },
    )
    provenance_path = config.evidence_root / "evidence_provenance.json"
    atomic_write_json(
        provenance_path,
        {
            "stage_version": EVIDENCE_STAGE_VERSION,
            "clinical_event_registry": event_registry_input,
            "clinical_event_source_policy": (
                "verified_partitioned_parquet"
                if event_registry is not None
                else "raw_csv_fallback"
            ),
            "clinical_event_tables": (
                sorted(
                    str(item.get("table_id"))
                    for item in event_registry.get("tables", ())
                    if item.get("status") == "ok"
                )
                if event_registry is not None
                else []
            ),
            "raw_sources_unchanged": True,
        },
    )
    output_paths = [str(artifact.path.resolve()) for artifact in artifacts]
    output_paths.append(str(definitions_path.resolve()))
    output_paths.append(str(provenance_path.resolve()))
    row_counts = {artifact.name: artifact.row_count for artifact in artifacts}
    # Which optional evidence layers were actually readable. An unreadable
    # layer produces empty columns downstream; recording the state here is
    # what keeps "empty" distinguishable from "assessed and negative".
    icu_coverage = icu_evidence_coverage(config)
    icu_present_tables = cast(list[str], icu_coverage["present_tables"])
    icu_missing_tables = cast(list[str], icu_coverage["missing_tables"])
    note_coverage = read_note_coverage(config)
    manifest = StageManifest.complete(
        stage="build-evidence",
        stage_version=EVIDENCE_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(output_paths),
        row_counts=row_counts,
        status=(
            "complete"
            if icu_coverage["status"] in ("complete", "no_icu_stays")
            else f"incomplete_icu_events_{icu_coverage['status']}"
        ),
        notes=(
            f"icu_coverage_status={icu_coverage['status']}",
            "icu_coverage_present="
            + ",".join(icu_present_tables),
            "icu_coverage_missing="
            + ",".join(icu_missing_tables),
            f"note_coverage_status={note_coverage['status']}",
            "zero_rows_are_not_evidence_of_assessed_absence=True",
            f"operational_config_hash={operational_config_hash(config)}",
        ),
        output_integrity=capture_output_integrity(
            tuple(Path(path) for path in output_paths)
        ),
    )
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    for artifact in artifacts:
        catalog.create_parquet_view(artifact.name, artifact.path)
    return manifest


def _read_checkpoint_payload(path: Path) -> Optional[dict]:
    """Checkpoint contents, or None if the file cannot be trusted.

    Truncation is the normal way a checkpoint dies: the process was killed
    mid-write, or the machine lost power. That must not be fatal either --
    an unreadable checkpoint carries exactly as much information as a
    missing one.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_evidence_checkpoint(
    path: Path,
    *,
    input_hash: str,
    config_hash: str,
) -> dict:
    if path.is_file():
        payload = _read_checkpoint_payload(path)
        if payload is not None and (
            payload.get("checkpoint_version")
            == EVIDENCE_CHECKPOINT_VERSION
            and payload.get("stage_version") == EVIDENCE_STAGE_VERSION
            and payload.get("input_hash") == input_hash
            and payload.get("config_hash") == config_hash
        ):
            payload.setdefault("groups", {})
            return payload
    # A checkpoint that describes different work is not an error, it is
    # simply unusable: discard it and start the stage fresh. Refusing to
    # continue would cost a multi-day run over a file whose only purpose is
    # to save time, and whose contents this stage can always produce again.
    payload = {
        "checkpoint_version": EVIDENCE_CHECKPOINT_VERSION,
        "stage_version": EVIDENCE_STAGE_VERSION,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "groups": {},
    }
    atomic_write_json(path, payload)
    return payload


def _resume_evidence_group(
    checkpoint: dict,
    *,
    group_id: str,
    progress: Optional[Callable[[str], None]] = None,
) -> Optional[list[EvidenceArtifact]]:
    """Artifacts of a finished group, or None if it has to be rebuilt.

    A checkpoint is a way to avoid repeating work, not a record anyone
    depends on. When its artifacts no longer match what it recorded, the
    honest response is to recompute the group -- the stage owns those files
    and can produce them again. Refusing to continue instead turns a
    recoverable inconsistency into a failed multi-day run, and the run has
    no way to recover on its own.

    What is *not* silent is the reason: it goes to the progress log, so a
    rebuild nobody expected can still be traced to the file that triggered
    it.
    """

    record = checkpoint.get("groups", {}).get(group_id)
    if record is None:
        return None

    def rebuild(reason: str) -> None:
        if progress:
            progress(
                f"Evidenzgruppe {group_id} wird neu berechnet: {reason}"
            )

    artifacts: list[EvidenceArtifact] = []
    for item in record.get("artifacts", ()):
        path = Path(str(item["path"]))
        if not path.is_file():
            rebuild(f"{path.name} fehlt")
            return None
        fingerprint = file_content_fingerprint(path)
        if (
            int(item.get("size", -1)) != int(fingerprint["size"])
            or item.get("sha256") != fingerprint["sha256"]
        ):
            rebuild(f"{path.name} wurde ausserhalb der Stufe veraendert")
            return None
        artifacts.append(
            EvidenceArtifact(
                name=str(item["name"]),
                path=path,
                row_count=int(item["row_count"]),
            )
        )
    if not artifacts:
        rebuild("der Eintrag enthaelt keine Artefakte")
        return None
    return artifacts


def _save_evidence_group(
    checkpoint_path: Path,
    checkpoint: dict,
    *,
    group_id: str,
    artifacts: list[EvidenceArtifact],
) -> None:
    if not artifacts:
        raise RuntimeError(
            f"Clinical evidence group produced no artifacts: {group_id}"
        )
    checkpoint.setdefault("groups", {})[group_id] = {
        "artifacts": [
            {
                "name": artifact.name,
                "path": str(artifact.path.resolve()),
                "row_count": int(artifact.row_count),
                **{
                    key: value
                    for key, value in file_content_fingerprint(
                        artifact.path
                    ).items()
                    if key in {"size", "sha256"}
                },
            }
            for artifact in artifacts
        ]
    }
    atomic_write_json(checkpoint_path, checkpoint)


def _has_bound_development_features(config: MimicProjectConfig) -> bool:
    if any(
        config.features_root.glob(
            "*/development/split=*/part-*.parquet"
        )
    ):
        return True
    if any(
        config.features_root.glob(
            "*/development/split=*/.part-*.resume.parquet"
        )
    ):
        return True
    for path in config.features_root.glob(
        "*/development/features.manifest.json"
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete":
            return True
    return False


def _reuse_bound_evidence_after_exact_attestation(
    config: MimicProjectConfig,
    previous: StageManifest,
    *,
    progress: Optional[Callable[[str], None]],
) -> StageManifest:
    """Keep immutable evidence only after reproducing it exactly.

    A guided workflow may refresh upstream manifests even though the clinical
    rows are unchanged. Rebuilding the canonical evidence in place would
    invalidate completed feature snapshots. The event-equivalence service
    instead creates a content-addressed shadow version and compares every
    label-relevant table in both directions before the old manifest is reused.
    """
    outputs_complete = bool(previous.output_paths) and all(
        Path(path).is_file() for path in previous.output_paths
    )
    if (
        previous.stage != "build-evidence"
        or previous.status != "complete"
        or not outputs_complete
    ):
        raise RuntimeError(
            "Refusing to replace clinical evidence that is already bound "
            "to a complete development feature artifact. The existing "
            "evidence manifest is incomplete or does not describe a "
            "build-evidence artifact; create a new versioned project root."
        )
    if progress:
        progress(
            "Gebundene klinische Evidenz gefunden; exakte "
            "Ereignis-Aequivalenz wird vor der Wiederverwendung geprueft"
        )
    try:
        # Lazy import avoids the evidence <-> attestation module cycle.
        from .event_equivalence import (
            attest_clinical_event_evidence_equivalence,
        )

        attestation = attest_clinical_event_evidence_equivalence(
            _AttestationResourceConfig(config),
            progress=progress,
        )
    except (JobPauseRequested, JobCancellationRequested):
        raise
    except Exception as exc:
        # Name the cause. "Equivalence could not be proven" reads as a finding
        # about the data, but the same sentence covered a TypeError in a
        # config wrapper and an out-of-memory in the shadow rebuild -- neither
        # of which says anything about whether the evidence changed. Twice now
        # that wording sent the investigation to the data instead of the code,
        # so the underlying error goes into the message.
        detail = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "Refusing to replace clinical evidence that is already bound to a "
            "complete development feature artifact. The equivalence check did "
            "not complete, so whether the evidence actually changed is still "
            f"unknown. Underlying cause: {detail}. If that is a resource "
            "limit or a defect, fix it and re-run the stage -- the check is "
            "repeatable. Only if the evidence genuinely differs does this "
            "need a new versioned project root."
        ) from exc
    if not str(attestation.get("status", "")).startswith("verified_exact"):
        raise RuntimeError(
            "Bound clinical evidence did not receive an exact equivalence "
            "attestation and cannot be reused."
        )
    if progress:
        progress(
            "Klinische Evidenz ist tabellengenau aequivalent und bleibt "
            "unveraendert gebunden"
        )
    return previous


def _lab_censoring_sql() -> tuple[str, str]:
    """Compatibility helper: bound plus qualifier, NOT an exact measurement."""
    return censored_result_sql()


def _build_lab_evidence(
    connection,
    config: MimicProjectConfig,
    progress: Optional[Callable[[str], None]],
    *,
    development_scope: Optional[Path] = None,
) -> list[EvidenceArtifact]:
    if progress:
        progress("Filtere und standardisiere klinisch relevante Laborwerte")
    source = config.mimic_iv_root / "hosp" / "labevents.csv.gz"
    source_relation = _event_or_raw_relation(
        config,
        "labevents",
        source,
    )
    if development_scope is not None:
        scope_sql = _sql_path(development_scope)
        invalid = connection.execute(
            f"SELECT COUNT(*) FROM read_parquet('{scope_sql}') "
            "WHERE subject_id IS NULL OR split IS NULL OR split NOT IN ('train','validation')"
        ).fetchone()[0]
        if invalid:
            raise ValueError("Laboratory delta scope must contain development patients only")
        source_relation = (
            f"(SELECT e.* FROM {source_relation} e SEMI JOIN "
            f"read_parquet('{scope_sql}') s ON TRY_CAST(e.subject_id AS BIGINT)=s.subject_id)"
        )
    output = config.evidence_root / "labs.parquet"
    lab_case = lab_group_case_sql("TRY_CAST(itemid AS INTEGER)")
    all_itemids = sorted(
        {itemid for itemids in LAB_ITEM_GROUPS.values() for itemid in itemids}
    )
    item_sql = ", ".join(str(itemid) for itemid in all_itemids)
    unit_status_case = _lab_unit_status_case_sql()
    bound_sql, censor_sql = censored_result_sql("value")
    effective_numeric, censoring = "parsed_bound", "parsed_censoring"
    reference_state = upper_reference_state_sql(
        effective_numeric, censoring, "TRY_CAST(ref_range_upper AS DOUBLE)"
    )
    from .lab_quality import hemolysis_invalidates_sql, hemolysis_state_sql

    hemolysis_state = hemolysis_state_sql("comments")
    hemolysis_invalidates = hemolysis_invalidates_sql("comments", "itemid")
    # Thesis T3 pairs ionised calcium and potassium from one blood sample.
    # The event registry carries ``specimen_id``; a raw or fixture source that
    # lacks it yields NULL instead of failing.
    source_columns = {
        str(row[0]).lower()
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM {source_relation} LIMIT 0"
        ).fetchall()
    }
    specimen_sql = (
        "TRY_CAST(specimen_id AS BIGINT)"
        if "specimen_id" in source_columns
        else "NULL::BIGINT"
    )
    query = f"""
        WITH parsed AS (
            SELECT *, {bound_sql} AS parsed_bound, {censor_sql} AS parsed_censoring
            FROM {source_relation}
        )
        SELECT
            TRY_CAST(labevent_id AS BIGINT) AS labevent_id,
            {specimen_sql} AS specimen_id,
            TRY_CAST(subject_id AS BIGINT) AS subject_id,
            TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
            TRY_CAST(itemid AS INTEGER) AS itemid,
            {lab_case} AS lab_group,
            TRY_CAST(charttime AS TIMESTAMP) AS charttime,
            TRY_CAST(storetime AS TIMESTAMP) AS storetime,
            NULLIF(TRIM(value), '') AS value_raw,
            CASE WHEN {censoring} IS NULL AND isfinite({effective_numeric})
                 THEN {effective_numeric} END AS value_numeric,
            {effective_numeric} AS value_bound,
            {censoring} AS censoring,
            CASE WHEN {censoring} IS NULL OR {censoring} IN ('above_limit', 'at_or_above_limit')
                 THEN {effective_numeric} END AS value_interval_lower,
            CASE WHEN {censoring} IS NULL OR {censoring} IN ('below_limit', 'at_or_below_limit')
                 THEN {effective_numeric} END AS value_interval_upper,
            COALESCE({censoring} = 'above_limit', FALSE) AS value_interval_lower_open,
            COALESCE({censoring} = 'below_limit', FALSE) AS value_interval_upper_open,
            {reference_state} AS upper_reference_state,
            NULLIF(TRIM(valueuom), '') AS unit,
            TRY_CAST(ref_range_lower AS DOUBLE) AS reference_lower,
            TRY_CAST(ref_range_upper AS DOUBLE) AS reference_upper,
            CASE
                WHEN {censoring} IS NULL AND isfinite({effective_numeric})
                 AND TRY_CAST(ref_range_upper AS DOUBLE) > 0
                THEN {effective_numeric}
                     / TRY_CAST(ref_range_upper AS DOUBLE)
            END AS upper_reference_ratio,
            NULLIF(TRIM(flag), '') AS abnormal_flag,
            {hemolysis_state} AS hemolysis_state,
            CASE
                WHEN NOT COALESCE(isfinite({effective_numeric}), FALSE) THEN 'non_numeric'
                -- Audit D2: a pseudo-elevation is not a measurement.
                WHEN {hemolysis_invalidates} THEN 'hemolysis_reported'
                {unit_status_case}
                WHEN NULLIF(TRIM(valueuom), '') IS NULL THEN 'unit_missing'
                ELSE 'unit_unrecognized'
            END AS value_status
        FROM parsed
        WHERE TRY_CAST(itemid AS INTEGER) IN ({item_sql})
          AND TRY_CAST(subject_id AS BIGINT) IS NOT NULL
          AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
    """
    lab_count = _copy_query_atomic(connection, query, output)
    lab_sql = _sql_path(output)
    lab_issues_path = config.quarantine_root / "lab_value_issues.parquet"
    lab_issue_count = _copy_query_atomic(
        connection,
        f"""
            SELECT *
            FROM read_parquet('{lab_sql}')
            WHERE value_status <> 'usable'
        """,
        lab_issues_path,
    )

    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    labs = _sql_path(output)
    windows_path = config.evidence_root / "ecg_lab_windows.parquet"

    def _windows_query(partition_index: int, partition_count: int) -> str:
        return _lab_windows_query(
            master,
            labs,
            partition_index=partition_index,
            partition_count=partition_count,
        )

    # 800k ECGs x 27 lab groups produce ~3.5M groups, and the window carries
    # 42 ARG_MIN/ARG_MAX aggregates whose ordering keys are STRUCTs.  Those
    # states cannot spill, so the unpartitioned query needs far more than the
    # 6 GiB this stage is allocated.  Bucket by subject and let the helper
    # refine the count if the machine still says no.
    window_count = _copy_partitioned_query_atomic(
        connection,
        _windows_query,
        windows_path,
        initial_partitions=max(16, min(64, int(config.workers) * 2)),
        progress=progress,
        label="Berechne ECG-zentrierte Laborfenster",
    )

    serial_path = config.evidence_root / "ecg_troponin_serial.parquet"

    def _serial_query(partition_index: int, partition_count: int) -> str:
        return _lab_troponin_serial_query(
            master,
            labs,
            partition_index=partition_index,
            partition_count=partition_count,
        )

    serial_count = _copy_partitioned_query_atomic(
        connection,
        _serial_query,
        serial_path,
        initial_partitions=max(8, min(32, int(config.workers))),
        progress=progress,
        label="Berechne serielle Troponin-Dynamik ohne Assay-Mischung",
    )

    return [
        EvidenceArtifact("labs", output, lab_count),
        EvidenceArtifact("lab_value_issues", lab_issues_path, lab_issue_count),
        EvidenceArtifact("ecg_lab_windows", windows_path, window_count),
        EvidenceArtifact("ecg_troponin_serial", serial_path, serial_count),
    ]


def _build_code_evidence(
    connection,
    config: MimicProjectConfig,
    progress: Optional[Callable[[str], None]],
) -> list[EvidenceArtifact]:
    if progress:
        progress("Mappe ICD-Diagnosen und Prozeduren mit Versionsbezug")
    diagnosis_source = config.standardized_root / "diagnoses_icd.parquet"
    diagnosis_output = config.evidence_root / "diagnosis_evidence.parquet"
    diagnosis_query = _group_union_query(
        _sql_path(diagnosis_source),
        ICD_GROUPS,
        select_columns=(
            "subject_id, hadm_id, seq_num, icd_code, icd_version"
        ),
    )
    diagnosis_count = _copy_query_atomic(
        connection,
        diagnosis_query,
        diagnosis_output,
    )

    procedure_source = config.standardized_root / "procedures_icd.parquet"
    procedure_output = config.evidence_root / "procedure_evidence.parquet"
    procedure_query = _group_union_query(
        _sql_path(procedure_source),
        PROCEDURE_GROUPS,
        select_columns=(
            "subject_id, hadm_id, seq_num, chartdate, icd_code, "
            "icd_version, time_precision"
        ),
    )
    procedure_count = _copy_query_atomic(
        connection,
        procedure_query,
        procedure_output,
    )
    return [
        EvidenceArtifact("diagnosis_evidence", diagnosis_output, diagnosis_count),
        EvidenceArtifact("procedure_evidence", procedure_output, procedure_count),
    ]


def _build_medication_evidence(
    connection,
    config: MimicProjectConfig,
    progress: Optional[Callable[[str], None]],
    *,
    query_checkpoint: Optional[tuple[Path, dict]] = None,
) -> list[EvidenceArtifact]:
    if progress:
        progress("Standardisiere relevante Medikamentenexpositionen")
    emar = config.mimic_iv_root / "hosp" / "emar.csv.gz"
    prescriptions = config.mimic_iv_root / "hosp" / "prescriptions.csv.gz"
    pharmacy = config.mimic_iv_root / "hosp" / "pharmacy.csv.gz"
    emar_detail = config.mimic_iv_root / "hosp" / "emar_detail.csv.gz"
    poe = config.mimic_iv_root / "hosp" / "poe.csv.gz"
    poe_detail = config.mimic_iv_root / "hosp" / "poe_detail.csv.gz"
    pyxis = config.mimic_ed_root / "ed" / "pyxis.csv.gz"
    medrecon = config.mimic_ed_root / "ed" / "medrecon.csv.gz"
    output = config.evidence_root / "medication_evidence.parquet"
    raw_paths = {
        "emar": emar,
        "emar_detail": emar_detail,
        "prescriptions": prescriptions,
        "pharmacy": pharmacy,
        "poe": poe,
        "poe_detail": poe_detail,
        "pyxis": pyxis,
        "medrecon": medrecon,
    }
    source_relations = {
        table_id: relation
        for table_id, path in raw_paths.items()
        if (
            relation := clinical_event_relation(config, table_id)
            or (_raw_csv(path) if path.is_file() else None)
        )
        is not None
    }
    quarantine_relations = {
        table_id: relation
        for table_id in raw_paths
        if (
            relation := clinical_event_quarantine_relation(
                config,
                table_id,
            )
        )
        is not None
    }

    source_unions: list[str] = []
    all_patterns = sorted(
        {pattern for patterns in MEDICATION_PATTERNS.values() for pattern in patterns}
    )
    mapping_rows = []
    for medication_class, patterns in MEDICATION_PATTERNS.items():
        for pattern in patterns:
            mapping_rows.append(
                f"('{medication_class}', '{pattern.replace(chr(39), chr(39) * 2)}')"
            )
    mapping_sql = ", ".join(mapping_rows)

    if "emar" in source_relations:
        source_unions.append(
            f"""
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                NULL::BIGINT AS ed_stay_id,
                TRY_CAST(charttime AS TIMESTAMP) AS event_time,
                TRY_CAST(charttime AS TIMESTAMP) AS start_time,
                TRY_CAST(charttime AS TIMESTAMP) AS end_time,
                NULLIF(TRIM(medication), '') AS medication_raw,
                'emar' AS source_table,
                'administered' AS exposure_type,
                CASE
                    WHEN LOWER(event_txt) LIKE '%administered%'
                      OR LOWER(event_txt) LIKE '%started%'
                      OR LOWER(event_txt) LIKE '%restarted%'
                      OR LOWER(event_txt) LIKE '%applied%'
                    THEN 'high'
                    ELSE 'medium'
                END AS confidence,
                NULLIF(TRIM(event_txt), '') AS source_status
            FROM {source_relations["emar"]}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
              AND ({_name_condition("medication", all_patterns)})
              AND (
                    LOWER(event_txt) LIKE '%administered%'
                 OR LOWER(event_txt) LIKE '%started%'
                 OR LOWER(event_txt) LIKE '%restarted%'
                 OR LOWER(event_txt) LIKE '%applied%'
                 OR LOWER(event_txt) = 'confirmed'
              )
              AND NOT REGEXP_MATCHES(
                    LOWER(event_txt),
                    '(^|[^a-z])not (administered|started|applied|restarted)'
              )
            """
        )
    if "prescriptions" in source_relations:
        source_unions.append(
            f"""
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                NULL::BIGINT AS ed_stay_id,
                TRY_CAST(starttime AS TIMESTAMP) AS event_time,
                TRY_CAST(starttime AS TIMESTAMP) AS start_time,
                TRY_CAST(stoptime AS TIMESTAMP) AS end_time,
                NULLIF(TRIM(drug), '') AS medication_raw,
                'prescriptions' AS source_table,
                'active_order' AS exposure_type,
                'medium' AS confidence,
                NULL::VARCHAR AS source_status
            FROM {source_relations["prescriptions"]}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(starttime AS TIMESTAMP) IS NOT NULL
              AND ({_name_condition("drug", all_patterns)})
            """
        )
    if "pharmacy" in source_relations:
        source_unions.append(
            f"""
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                NULL::BIGINT AS ed_stay_id,
                TRY_CAST(starttime AS TIMESTAMP) AS event_time,
                TRY_CAST(starttime AS TIMESTAMP) AS start_time,
                TRY_CAST(stoptime AS TIMESTAMP) AS end_time,
                NULLIF(TRIM(medication), '') AS medication_raw,
                'pharmacy' AS source_table,
                'active_order' AS exposure_type,
                'medium' AS confidence,
                NULLIF(TRIM(status), '') AS source_status
            FROM {source_relations["pharmacy"]}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(starttime AS TIMESTAMP) IS NOT NULL
              AND ({_name_condition("medication", all_patterns)})
            """
        )
    if "pyxis" in source_relations:
        source_unions.append(
            f"""
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                NULL::BIGINT AS hadm_id,
                TRY_CAST(stay_id AS BIGINT) AS ed_stay_id,
                TRY_CAST(charttime AS TIMESTAMP) AS event_time,
                TRY_CAST(charttime AS TIMESTAMP) AS start_time,
                TRY_CAST(charttime AS TIMESTAMP) AS end_time,
                NULLIF(TRIM(name), '') AS medication_raw,
                'pyxis' AS source_table,
                'dispensed' AS exposure_type,
                'medium' AS confidence,
                NULL::VARCHAR AS source_status
            FROM {source_relations["pyxis"]}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
              AND ({_name_condition("name", all_patterns)})
            """
        )
    if "medrecon" in source_relations:
        source_unions.append(
            f"""
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                NULL::BIGINT AS hadm_id,
                TRY_CAST(stay_id AS BIGINT) AS ed_stay_id,
                TRY_CAST(charttime AS TIMESTAMP) AS event_time,
                TRY_CAST(charttime AS TIMESTAMP) AS start_time,
                NULL::TIMESTAMP AS end_time,
                NULLIF(TRIM(name), '') AS medication_raw,
                'medrecon' AS source_table,
                'reported_home_medication' AS exposure_type,
                'low' AS confidence,
                NULL::VARCHAR AS source_status
            FROM {source_relations["medrecon"]}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(charttime AS TIMESTAMP) IS NOT NULL
              AND ({_name_condition("name", all_patterns)})
            """
        )
    if not source_unions:
        query = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS hadm_id,
                NULL::BIGINT AS ed_stay_id,
                NULL::TIMESTAMP AS event_time,
                NULL::TIMESTAMP AS start_time,
                NULL::TIMESTAMP AS end_time,
                NULL::VARCHAR AS medication_raw,
                NULL::VARCHAR AS medication_class,
                NULL::VARCHAR AS source_table,
                NULL::VARCHAR AS exposure_type,
                NULL::VARCHAR AS confidence,
                NULL::VARCHAR AS source_status
            WHERE FALSE
        """
    else:
        sources_sql = " UNION ALL ".join(source_unions)
        query = f"""
            WITH source_medications AS (
                {sources_sql}
            ),
            medication_mapping(medication_class, name_pattern) AS (
                VALUES {mapping_sql}
            )
            SELECT DISTINCT
                s.subject_id,
                s.hadm_id,
                s.ed_stay_id,
                s.event_time,
                s.start_time,
                s.end_time,
                s.medication_raw,
                m.medication_class,
                s.source_table,
                s.exposure_type,
                s.confidence,
                s.source_status
            FROM source_medications s
            JOIN medication_mapping m
              ON LOWER(s.medication_raw) LIKE '%' || m.name_pattern || '%'
        """
    medication_count = _copy_checkpointed_medication_query(
        connection, query, output, checkpoint_context=query_checkpoint,
    )
    medication_context_path = (
        config.evidence_root
        / "medication_order_administration_context.parquet"
    )
    # This join is memory-bound rather than CPU-bound.  On the complete
    # project it completed in 748 s with one DuckDB thread, whereas six
    # threads built six parallel hash states, reached the 10 GiB cap and
    # failed after 1,179 s.  Limit only this query; the setting is restored
    # before the ECG-window and remaining evidence work continues.
    medication_context_count = _copy_checkpointed_medication_query(
        connection,
        _medication_order_administration_context_query(
            source_relations,
            all_patterns,
        ),
        medication_context_path,
        threads=1,
        checkpoint_context=query_checkpoint,
    )

    if progress:
        progress("Verknuepfe Medikamentenfenster mit ECGs")
    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    medications = _sql_path(output)
    windows_path = config.evidence_root / "ecg_medication_windows.parquet"
    # The former OR predicate forced DuckDB to hash-join every medication row
    # of a patient to every ECG and only then apply the temporal filter.  Keep
    # the three temporal relations explicit and execute disjoint subject
    # buckets so peak memory is bounded.  UNION (not UNION ALL) preserves the
    # original OR semantics when an order is also inside the event window.
    # Eight buckets kept the real-data worker at ~3.7 GiB RSS under a 6 GiB
    # DuckDB cap; larger worker allocations get correspondingly finer buckets.
    # The measured eight buckets held on this machine, but a bucket count is
    # only ever a claim about data already seen; the shared helper doubles it
    # if a later admission mix says otherwise instead of ending the run.
    windows_count = _copy_partitioned_query_atomic(
        connection,
        lambda partition_index, partition_count: _medication_windows_query(
            master,
            medications,
            partition_index=partition_index,
            partition_count=partition_count,
        ),
        windows_path,
        initial_partitions=max(8, min(32, int(config.workers))),
        progress=progress,
        label="Verknuepfe Medikamentenfenster mit ECGs",
    )

    if progress:
        progress("Suche EKG-Paare vor/nach AV-Knoten-wirksamer Medikation")
    exposure_pairs_path = (
        config.evidence_root / "ecg_medication_exposure_pairs.parquet"
    )
    exposure_pairs_query = _ecg_medication_exposure_pair_query(master, medications)
    exposure_pairs_count = _copy_query_atomic(
        connection, exposure_pairs_query, exposure_pairs_path
    )

    if progress:
        progress("Ermittle encounter-spezifische Medikationsquellenabdeckung")
    coverage_path = config.evidence_root / "medication_source_coverage.parquet"
    coverage_unions: list[str] = []
    for source_name in (
        "emar",
        "prescriptions",
        "pharmacy",
    ):
        relations = tuple(
            relation
            for relation in (
                source_relations.get(source_name),
                quarantine_relations.get(source_name),
            )
            if relation is not None
        )
        for relation in relations:
            coverage_unions.append(
                f"""
                SELECT DISTINCT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                    NULL::BIGINT AS ed_stay_id,
                    '{source_name}' AS source_table
                FROM {relation}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(hadm_id AS BIGINT) IS NOT NULL
                """
            )
    for source_name in (
        "pyxis",
        "medrecon",
    ):
        relations = tuple(
            relation
            for relation in (
                source_relations.get(source_name),
                quarantine_relations.get(source_name),
            )
            if relation is not None
        )
        for relation in relations:
            coverage_unions.append(
                f"""
                SELECT DISTINCT
                    TRY_CAST(subject_id AS BIGINT) AS subject_id,
                    NULL::BIGINT AS hadm_id,
                    TRY_CAST(stay_id AS BIGINT) AS ed_stay_id,
                    '{source_name}' AS source_table
                FROM {relation}
                WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
                  AND TRY_CAST(stay_id AS BIGINT) IS NOT NULL
                """
            )
    coverage_query = (
        "SELECT DISTINCT * FROM ("
        + " UNION ALL ".join(coverage_unions)
        + ") medication_coverage"
        if coverage_unions
        else """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS hadm_id,
                NULL::BIGINT AS ed_stay_id,
                NULL::VARCHAR AS source_table
            WHERE FALSE
        """
    )
    coverage_count = _copy_query_atomic(
        connection,
        coverage_query,
        coverage_path,
    )
    ecg_coverage_path = (
        config.evidence_root / "ecg_medication_source_coverage.parquet"
    )
    ecg_coverage_query = f"""
        SELECT
            m.subject_id,
            m.study_id,
            TRUE AS medication_assessed,
            BOOL_OR(c.source_table = 'emar') AS has_emar_coverage,
            BOOL_OR(c.source_table = 'prescriptions')
                AS has_prescription_coverage,
            BOOL_OR(c.source_table = 'pharmacy') AS has_pharmacy_coverage,
            BOOL_OR(c.source_table = 'pyxis') AS has_pyxis_coverage,
            BOOL_OR(c.source_table = 'medrecon') AS has_medrecon_coverage
        FROM read_parquet('{master}') m
        JOIN read_parquet('{_sql_path(coverage_path)}') c
          ON m.subject_id = c.subject_id
         AND (
                (m.hadm_id IS NOT NULL AND m.hadm_id = c.hadm_id)
             OR (m.ed_stay_id IS NOT NULL AND m.ed_stay_id = c.ed_stay_id)
         )
        GROUP BY m.subject_id, m.study_id
    """
    ecg_coverage_count = _copy_query_atomic(
        connection,
        ecg_coverage_query,
        ecg_coverage_path,
    )
    return [
        EvidenceArtifact("medication_evidence", output, medication_count),
        EvidenceArtifact(
            "medication_order_administration_context",
            medication_context_path,
            medication_context_count,
        ),
        EvidenceArtifact("ecg_medication_windows", windows_path, windows_count),
        EvidenceArtifact(
            "ecg_medication_exposure_pairs",
            exposure_pairs_path,
            exposure_pairs_count,
        ),
        EvidenceArtifact(
            "medication_source_coverage",
            coverage_path,
            coverage_count,
        ),
        EvidenceArtifact(
            "ecg_medication_source_coverage",
            ecg_coverage_path,
            ecg_coverage_count,
        ),
    ]


def _lab_windows_query(
    master: str,
    labs: str,
    *,
    partition_index: int,
    partition_count: int,
) -> str:
    """Build one subject bucket of the ECG-centred laboratory window."""

    if partition_count < 1:
        raise ValueError("partition_count must be positive")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index is outside partition_count")
    nearest_order = """
        STRUCT_PACK(
            admission_match_rank := admission_match_rank,
            distance_hours := ABS(delta_hours),
            before_ecg_rank := CASE WHEN delta_hours <= 0 THEN 0 ELSE 1 END,
            charttime := charttime,
            labevent_id := labevent_id
        )
    """
    url_order = """
        STRUCT_PACK(
            inverse_admission_match_rank := -admission_match_rank,
            url_ratio := upper_reference_ratio,
            inverse_distance_hours := -ABS(delta_hours),
            before_ecg_rank := CASE WHEN delta_hours <= 0 THEN 1 ELSE 0 END,
            inverse_labevent_id := -labevent_id
        )
    """
    master_bucket = (
        "TRUE"
        if partition_count <= 1
        else f"hash(m.subject_id) % {partition_count} = {partition_index}"
    )
    lab_bucket = (
        "TRUE"
        if partition_count <= 1
        else f"hash(l.subject_id) % {partition_count} = {partition_index}"
    )
    return f"""
    WITH candidates AS (
        SELECT
            m.subject_id,
            m.study_id,
            m.ecg_time,
            m.hadm_id AS ecg_hadm_id,
            l.* EXCLUDE(subject_id),
            CASE
                WHEN m.hadm_id IS NOT NULL
                 AND l.hadm_id = m.hadm_id THEN 0
                ELSE 1
            END AS admission_match_rank,
            CASE
                WHEN m.hadm_id IS NOT NULL
                 AND l.hadm_id = m.hadm_id THEN 'same_admission'
                WHEN m.hadm_id IS NULL THEN 'ecg_admission_unavailable'
                ELSE 'lab_admission_unavailable'
            END AS admission_match_type,
            DATE_DIFF('second', m.ecg_time, l.charttime) / 3600.0
                AS delta_hours
        FROM read_parquet('{master}') m
        JOIN read_parquet('{labs}') l
          ON m.subject_id = l.subject_id
         AND {lab_bucket}
         AND l.charttime BETWEEN m.ecg_time - INTERVAL 24 HOUR
                             AND m.ecg_time + INTERVAL 24 HOUR
         -- Never silently borrow a lab from a different known admission.
         -- Missing admission identifiers remain an explicit fallback and
         -- rank behind a same-admission candidate.
         AND (
                m.hadm_id IS NULL
             OR l.hadm_id IS NULL
             OR m.hadm_id = l.hadm_id
         )
        WHERE {master_bucket}
    )
    SELECT
        subject_id,
        study_id,
        lab_group,
        {interval_window_aggregates(nearest_order)}
        ARG_MIN_NULL(value_numeric, {nearest_order})
            FILTER (WHERE value_status = 'usable') AS nearest_value,
        ARG_MIN(unit, {nearest_order})
            FILTER (WHERE value_status = 'usable') AS nearest_unit,
        ARG_MIN(labevent_id, {nearest_order})
            FILTER (WHERE value_status = 'usable') AS nearest_labevent_id,
        ARG_MIN(delta_hours, {nearest_order})
            FILTER (WHERE value_status = 'usable') AS nearest_delta_hours,
        ARG_MIN(hadm_id, {nearest_order})
            FILTER (WHERE value_status = 'usable') AS nearest_lab_hadm_id,
        ARG_MIN(admission_match_type, {nearest_order})
            FILTER (WHERE value_status = 'usable')
            AS nearest_admission_match_type,
        ARG_MIN_NULL(value_numeric, {nearest_order})
            FILTER (WHERE delta_hours <= 0 AND value_status = 'usable')
            AS nearest_before_value,
        ARG_MIN(labevent_id, {nearest_order})
            FILTER (WHERE delta_hours <= 0 AND value_status = 'usable')
            AS nearest_before_labevent_id,
        MAX(delta_hours)
            FILTER (WHERE delta_hours <= 0 AND value_status = 'usable')
            AS nearest_before_delta_hours,
        ARG_MIN_NULL(value_numeric, {nearest_order})
            FILTER (WHERE delta_hours >= 0 AND value_status = 'usable')
            AS nearest_after_value,
        ARG_MIN(labevent_id, {nearest_order})
            FILTER (WHERE delta_hours >= 0 AND value_status = 'usable')
            AS nearest_after_labevent_id,
        MIN(delta_hours)
            FILTER (WHERE delta_hours >= 0 AND value_status = 'usable')
            AS nearest_after_delta_hours,
        ARG_MIN_NULL(value_numeric, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 2 AND value_status = 'usable'
        ) AS nearest_value_2h,
        ARG_MIN(unit, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 2 AND value_status = 'usable'
        ) AS nearest_unit_2h,
        ARG_MIN(labevent_id, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 2 AND value_status = 'usable'
        ) AS nearest_labevent_id_2h,
        ARG_MIN_NULL(reference_lower, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 2 AND value_status = 'usable'
        ) AS nearest_reference_lower_2h,
        ARG_MIN_NULL(reference_upper, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 2 AND value_status = 'usable'
        ) AS nearest_reference_upper_2h,
        ARG_MIN(delta_hours, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 2 AND value_status = 'usable'
        ) AS nearest_delta_hours_2h,
        ARG_MIN(admission_match_type, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 2 AND value_status = 'usable'
        ) AS nearest_admission_match_type_2h,
        ARG_MIN_NULL(value_numeric, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_value_6h,
        ARG_MIN_NULL(value_bound, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_bound_6h,
        ARG_MIN_NULL(censoring, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_censoring_6h,
        ARG_MIN(unit, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_unit_6h,
        ARG_MIN(labevent_id, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_labevent_id_6h,
        ARG_MIN_NULL(reference_lower, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_reference_lower_6h,
        ARG_MIN_NULL(reference_upper, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_reference_upper_6h,
        ARG_MIN(delta_hours, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_delta_hours_6h,
        ARG_MIN(admission_match_type, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        ) AS nearest_admission_match_type_6h,
        ARG_MIN_NULL(value_numeric, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 12 AND value_status = 'usable'
        ) AS nearest_value_12h,
        ARG_MIN(labevent_id, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 12 AND value_status = 'usable'
        ) AS nearest_labevent_id_12h,
        ARG_MIN(delta_hours, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 12 AND value_status = 'usable'
        ) AS nearest_delta_hours_12h,
        ARG_MIN(admission_match_type, {nearest_order}) FILTER (
            WHERE ABS(delta_hours) <= 12 AND value_status = 'usable'
        ) AS nearest_admission_match_type_12h,
        ARG_MIN_NULL(value_numeric, {nearest_order}) FILTER (
            WHERE delta_hours BETWEEN -3 AND 6
              AND value_status = 'usable'
        ) AS nearest_value_strict,
        ARG_MIN(labevent_id, {nearest_order}) FILTER (
            WHERE delta_hours BETWEEN -3 AND 6
              AND value_status = 'usable'
        ) AS nearest_labevent_id_strict,
        ARG_MIN(delta_hours, {nearest_order}) FILTER (
            WHERE delta_hours BETWEEN -3 AND 6
              AND value_status = 'usable'
        ) AS nearest_delta_hours_strict,
        ARG_MIN(admission_match_type, {nearest_order}) FILTER (
            WHERE delta_hours BETWEEN -3 AND 6
              AND value_status = 'usable'
        ) AS nearest_admission_match_type_strict,
        ARG_MIN_NULL(value_numeric, {nearest_order}) FILTER (
            WHERE delta_hours BETWEEN -6 AND 12
              AND value_status = 'usable'
        ) AS nearest_value_standard,
        ARG_MIN(labevent_id, {nearest_order}) FILTER (
            WHERE delta_hours BETWEEN -6 AND 12
              AND value_status = 'usable'
        ) AS nearest_labevent_id_standard,
        ARG_MIN(delta_hours, {nearest_order}) FILTER (
            WHERE delta_hours BETWEEN -6 AND 12
              AND value_status = 'usable'
        ) AS nearest_delta_hours_standard,
        ARG_MIN(admission_match_type, {nearest_order}) FILTER (
            WHERE delta_hours BETWEEN -6 AND 12
              AND value_status = 'usable'
        ) AS nearest_admission_match_type_standard,
        MIN(value_numeric) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        )
            AS min_6h,
        MAX(value_numeric) FILTER (
            WHERE ABS(delta_hours) <= 6 AND value_status = 'usable'
        )
            AS max_6h,
        MIN(value_numeric) FILTER (
            WHERE ABS(delta_hours) <= 12 AND value_status = 'usable'
        )
            AS min_12h,
        MAX(value_numeric) FILTER (
            WHERE ABS(delta_hours) <= 12 AND value_status = 'usable'
        )
            AS max_12h,
        MIN(value_numeric) FILTER (WHERE value_status = 'usable') AS min_24h,
        MAX(value_numeric) FILTER (WHERE value_status = 'usable') AS max_24h,
        MAX(upper_reference_ratio)
            FILTER (
                WHERE delta_hours BETWEEN -3 AND 6
                  AND value_status = 'usable'
            ) AS max_upper_reference_ratio_strict,
        ARG_MAX(labevent_id, {url_order})
            FILTER (
                WHERE delta_hours BETWEEN -3 AND 6
                  AND value_status = 'usable'
                  AND upper_reference_ratio IS NOT NULL
            ) AS max_upper_reference_labevent_id_strict,
        ARG_MAX(delta_hours, {url_order})
            FILTER (
                WHERE delta_hours BETWEEN -3 AND 6
                  AND value_status = 'usable'
                  AND upper_reference_ratio IS NOT NULL
            ) AS max_upper_reference_delta_hours_strict,
        MAX(upper_reference_ratio)
            FILTER (
                WHERE delta_hours BETWEEN -6 AND 12
                  AND value_status = 'usable'
            )
            AS max_upper_reference_ratio_standard,
        ARG_MAX(labevent_id, {url_order})
            FILTER (
                WHERE delta_hours BETWEEN -6 AND 12
                  AND value_status = 'usable'
                  AND upper_reference_ratio IS NOT NULL
            ) AS max_upper_reference_labevent_id_standard,
        ARG_MAX(delta_hours, {url_order})
            FILTER (
                WHERE delta_hours BETWEEN -6 AND 12
                  AND value_status = 'usable'
                  AND upper_reference_ratio IS NOT NULL
            ) AS max_upper_reference_delta_hours_standard,
        MIN(upper_reference_ratio)
            FILTER (
                WHERE delta_hours BETWEEN -6 AND 12
                  AND value_status = 'usable'
            )
            AS min_upper_reference_ratio_standard,
        MAX(upper_reference_ratio)
            FILTER (WHERE value_status = 'usable')
            AS max_upper_reference_ratio_24h,
        ARG_MAX(labevent_id, {url_order})
            FILTER (
                WHERE value_status = 'usable'
                  AND upper_reference_ratio IS NOT NULL
            ) AS max_upper_reference_labevent_id_24h,
        ARG_MAX(delta_hours, {url_order})
            FILTER (
                WHERE value_status = 'usable'
                  AND upper_reference_ratio IS NOT NULL
            ) AS max_upper_reference_delta_hours_24h,
        COUNT(*) FILTER (
            WHERE ABS(delta_hours) <= 2
              AND value_status = 'usable'
        ) AS count_2h,
        COUNT(*) FILTER (
            WHERE ABS(delta_hours) <= 6
              AND value_status = 'usable'
        ) AS count_6h,
        COUNT(*) FILTER (
            WHERE ABS(delta_hours) <= 12
              AND value_status = 'usable'
        ) AS count_12h,
        COUNT(*) FILTER (
            WHERE delta_hours BETWEEN -3 AND 6
              AND value_status = 'usable'
        ) AS count_strict,
        COUNT(*) FILTER (
            WHERE delta_hours BETWEEN -6 AND 12
              AND value_status = 'usable'
        ) AS count_standard,
        COUNT(*) FILTER (
            WHERE value_status = 'usable'
        ) AS count_24h
    FROM candidates
    GROUP BY subject_id, study_id, lab_group
"""


def _lab_troponin_serial_query(
    master: str,
    labs: str,
    *,
    partition_index: int,
    partition_count: int,
) -> str:
    """Build one subject bucket of the serial troponin dynamics query."""

    if partition_count < 1:
        raise ValueError("partition_count must be positive")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index is outside partition_count")
    master_bucket = (
        "TRUE"
        if partition_count <= 1
        else f"hash(m.subject_id) % {partition_count} = {partition_index}"
    )
    lab_bucket = (
        "TRUE"
        if partition_count <= 1
        else f"hash(l.subject_id) % {partition_count} = {partition_index}"
    )
    # Audit D3: censored results ("<0.01") are intervals, not missing values.
    # The old rule required every value of a series to be exact; 61 % of the
    # series with two or more values were then unassessable, among them the
    # classic rise from "not detectable" and every consistently undetectable
    # (clearest negative) series. Each criterion is now decided over the whole
    # interval: positive when it holds for every value the bounds allow,
    # negative when it can hold for none, indeterminate otherwise. For series
    # of exact values this is the old rule. Troponin is non-negative, so a
    # below-limit result spans [0, limit].
    return f"""
    WITH candidates AS (
        SELECT
            m.subject_id,
            m.study_id,
            l.labevent_id,
            l.itemid,
            l.unit,
            l.charttime,
            l.value_numeric,
            l.upper_reference_ratio,
            COALESCE(l.value_interval_lower, 0.0) AS lo,
            l.value_interval_upper AS hi,
            COALESCE(l.value_interval_lower_open, FALSE) AS lo_open,
            CASE WHEN l.reference_upper > 0 THEN l.reference_upper END AS url,
            DATE_DIFF('second', m.ecg_time, l.charttime) / 3600.0
                AS delta_hours
        FROM read_parquet('{master}') m
        JOIN read_parquet('{labs}') l
          ON m.subject_id = l.subject_id
         AND {lab_bucket}
         AND {master_bucket}
         AND l.lab_group IN ('troponin_i', 'troponin_t')
         AND l.charttime BETWEEN m.ecg_time - INTERVAL 6 HOUR
                             AND m.ecg_time + INTERVAL 12 HOUR
         AND l.value_status = 'usable'
    ),
    bounded AS (
        SELECT
            *,
            url IS NOT NULL AND (
                lo / url > 1 OR (lo / url = 1 AND lo_open)
            ) AS certainly_above_url,
            url IS NOT NULL AND hi IS NOT NULL AND hi / url <= 1
                AS certainly_at_or_below_url,
            url IS NULL OR hi IS NULL OR hi / url > 1 AS possibly_above_url,
            url IS NULL OR lo / url < 1 OR (lo / url = 1 AND NOT lo_open)
                AS possibly_at_or_below_url
        FROM candidates
    ),
    ordered AS (
        SELECT
            *,
            LAG(value_numeric) OVER assay_order AS previous_value,
            LAG(lo) OVER assay_order AS previous_lo,
            LAG(hi) OVER assay_order AS previous_hi,
            ROW_NUMBER() OVER assay_order AS position
        FROM bounded
        WINDOW assay_order AS (
            PARTITION BY subject_id, study_id, itemid, unit
            ORDER BY charttime, labevent_id
        )
    ),
    paired AS (
        SELECT
            *,
            -- Smallest relative change the two intervals allow.
            CASE
                WHEN position = 1 THEN NULL
                WHEN previous_hi IS NOT NULL AND previous_hi > 0
                     AND lo > previous_hi
                THEN (lo - previous_hi) / previous_hi
                WHEN hi IS NOT NULL AND previous_lo > 0 AND hi < previous_lo
                THEN (previous_lo - hi) / previous_lo
                ELSE 0.0
            END AS certain_relative_delta,
            -- Largest relative change the two intervals allow.
            CASE
                WHEN position = 1 THEN NULL
                ELSE GREATEST(
                    CASE
                        WHEN hi IS NULL THEN 'infinity'::DOUBLE
                        WHEN previous_lo > 0 THEN (hi - previous_lo) / previous_lo
                        WHEN hi > 0 THEN 'infinity'::DOUBLE
                        ELSE 0.0
                    END,
                    CASE
                        WHEN previous_hi IS NULL THEN 1.0
                        WHEN previous_hi > 0 THEN (previous_hi - lo) / previous_hi
                        ELSE 0.0
                    END
                )
            END AS possible_relative_delta
        FROM ordered
    ),
    by_assay AS (
        SELECT
            subject_id,
            study_id,
            itemid,
            unit,
            COUNT(*) AS serial_count,
            COUNT(value_numeric) = COUNT(*) AS all_values_exact,
            ARG_MIN(
                value_numeric,
                STRUCT_PACK(charttime := charttime, labevent_id := labevent_id)
            ) AS first_value,
            ARG_MAX(
                value_numeric,
                STRUCT_PACK(charttime := charttime, labevent_id := labevent_id)
            ) AS last_value,
            ARG_MIN(
                labevent_id,
                STRUCT_PACK(charttime := charttime, labevent_id := labevent_id)
            ) AS first_labevent_id,
            ARG_MAX(
                labevent_id,
                STRUCT_PACK(charttime := charttime, labevent_id := labevent_id)
            ) AS last_labevent_id,
            MIN(upper_reference_ratio) AS min_url_ratio,
            MAX(upper_reference_ratio) AS max_url_ratio,
            MAX(
                ABS(value_numeric - previous_value)
                / NULLIF(ABS(previous_value), 0)
            ) AS relative_delta,
            BOOL_OR(certainly_above_url) AS any_certainly_above,
            BOOL_OR(certainly_at_or_below_url) AS any_certainly_at_or_below,
            COALESCE(MAX(certain_relative_delta) >= 0.20, FALSE)
                AS certain_delta,
            COALESCE(MAX(possible_relative_delta) >= 0.20, FALSE)
                AS possible_delta,
            COUNT(*) FILTER (WHERE possibly_above_url) AS possibly_above_count,
            COUNT(*) FILTER (WHERE possibly_at_or_below_url)
                AS possibly_at_or_below_count,
            COUNT(*) FILTER (
                WHERE possibly_above_url AND possibly_at_or_below_url
            ) AS possibly_both_count,
            -- Audit 5.2.3: a series that starts "not detectable" (a result
            -- reported only as below a limit that is itself at or below the
            -- URL) and later is certainly above the URL.
            ARG_MIN(
                value_numeric IS NULL AND hi IS NOT NULL AND lo = 0
                    AND certainly_at_or_below_url,
                STRUCT_PACK(charttime := charttime, labevent_id := labevent_id)
            ) AS first_undetectable,
            BOOL_OR(certainly_above_url AND position > 1)
                AS later_certainly_above,
            BOOL_OR(possibly_above_url AND position > 1)
                AS later_possibly_above,
            MIN(charttime) AS first_time,
            MAX(charttime) AS last_time
        FROM paired
        GROUP BY subject_id, study_id, itemid, unit
    ),
    assessed AS (
        SELECT
            *,
            CASE
                WHEN serial_count >= 2
                 AND any_certainly_above
                 AND (any_certainly_at_or_below OR certain_delta)
                THEN 'positive'
                WHEN serial_count >= 2
                 AND possibly_above_count >= 1
                 AND (
                     possible_delta
                     OR (
                         possibly_at_or_below_count >= 1
                         AND NOT (
                             possibly_above_count = 1
                             AND possibly_at_or_below_count = 1
                             AND possibly_both_count = 1
                         )
                     )
                 )
                THEN 'indeterminate'
                ELSE 'negative'
            END AS interval_state
        FROM by_assay
    ),
    flagged AS (
        SELECT *, interval_state = 'positive' AS qualifies_acute_change
        FROM assessed
    )
    SELECT
        subject_id,
        study_id,
        BOOL_OR(qualifies_acute_change) AS acute_change_proxy,
        BOOL_OR(serial_count >= 2) AND (
            BOOL_OR(interval_state = 'positive')
            OR BOOL_AND(interval_state = 'negative')
        ) AS serial_assessment_available,
        CASE
            WHEN BOOL_OR(interval_state = 'positive') THEN 'positive'
            WHEN NOT BOOL_OR(serial_count >= 2) THEN 'single_value'
            WHEN BOOL_AND(interval_state = 'negative') THEN 'negative'
            ELSE 'indeterminate'
        END AS serial_interval_state,
        CASE
            WHEN BOOL_OR(
                serial_count >= 2 AND first_undetectable AND later_certainly_above
            ) THEN 'rise_from_undetectable'
            WHEN BOOL_OR(serial_count >= 2 AND first_undetectable)
                 AND NOT BOOL_OR(
                     serial_count >= 2 AND first_undetectable AND later_possibly_above
                 )
                 AND NOT BOOL_OR(qualifies_acute_change)
            THEN 'undetectable_without_rise'
            WHEN BOOL_OR(serial_count >= 2 AND first_undetectable)
            THEN 'indeterminate'
            ELSE 'not_applicable'
        END AS undetectable_rise_state,
        BOOL_AND(all_values_exact) AS all_values_exact,
        MAX(serial_count) AS max_serial_count,
        MAX(max_url_ratio) AS max_url_ratio,
        MAX(relative_delta) FILTER (WHERE all_values_exact) AS max_relative_delta,
        MIN(first_time) AS first_troponin_time,
        MAX(last_time) AS last_troponin_time,
        ARG_MAX(
            last_time,
            STRUCT_PACK(
                max_url_ratio := max_url_ratio,
                relative_delta := COALESCE(relative_delta, -1),
                itemid := itemid
            )
        ) FILTER (
            WHERE qualifies_acute_change
        ) AS acute_change_event_time,
        ARG_MAX(
            first_labevent_id,
            STRUCT_PACK(
                max_url_ratio := max_url_ratio,
                relative_delta := COALESCE(relative_delta, -1),
                itemid := itemid
            )
        ) FILTER (
            WHERE qualifies_acute_change
        ) AS qualifying_first_labevent_id,
        ARG_MAX(
            last_labevent_id,
            STRUCT_PACK(
                max_url_ratio := max_url_ratio,
                relative_delta := COALESCE(relative_delta, -1),
                itemid := itemid
            )
        ) FILTER (
            WHERE qualifies_acute_change
        ) AS qualifying_last_labevent_id
    FROM flagged
    GROUP BY subject_id, study_id
    """


def _lab_feature_pivot_query(
    labs: str,
    generated_lab_window_pivot: str,
    *,
    partition_index: int,
    partition_count: int,
) -> str:
    """Build one subject bucket of the ECG laboratory feature pivot."""

    if partition_count < 1:
        raise ValueError("partition_count must be positive")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index is outside partition_count")
    bucket = (
        "TRUE"
        if partition_count <= 1
        else f"hash(subject_id) % {partition_count} = {partition_index}"
    )
    return f"""
    SELECT
        subject_id,
        study_id,
        {troponin_interval_pivot()}
        MAX(max_upper_reference_ratio_standard) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_max_url_ratio,
        ARG_MAX(
            max_upper_reference_labevent_id_standard,
            STRUCT_PACK(
                url_ratio := max_upper_reference_ratio_standard,
                inverse_distance_hours :=
                    -ABS(max_upper_reference_delta_hours_standard),
                lab_group := lab_group
            )
        ) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
              AND max_upper_reference_ratio_standard IS NOT NULL
        ) AS troponin_max_url_labevent_id,
        ARG_MAX(
            max_upper_reference_delta_hours_standard,
            STRUCT_PACK(
                url_ratio := max_upper_reference_ratio_standard,
                inverse_distance_hours :=
                    -ABS(max_upper_reference_delta_hours_standard),
                lab_group := lab_group
            )
        ) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
              AND max_upper_reference_ratio_standard IS NOT NULL
        ) AS troponin_max_url_delta_hours,
        MAX(max_upper_reference_ratio_strict) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_max_url_ratio_strict,
        ARG_MAX(
            max_upper_reference_labevent_id_strict,
            STRUCT_PACK(
                url_ratio := max_upper_reference_ratio_strict,
                inverse_distance_hours :=
                    -ABS(max_upper_reference_delta_hours_strict),
                lab_group := lab_group
            )
        ) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
              AND max_upper_reference_ratio_strict IS NOT NULL
        ) AS troponin_max_url_labevent_id_strict,
        ARG_MAX(
            max_upper_reference_delta_hours_strict,
            STRUCT_PACK(
                url_ratio := max_upper_reference_ratio_strict,
                inverse_distance_hours :=
                    -ABS(max_upper_reference_delta_hours_strict),
                lab_group := lab_group
            )
        ) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
              AND max_upper_reference_ratio_strict IS NOT NULL
        ) AS troponin_max_url_delta_hours_strict,
        MAX(max_upper_reference_ratio_24h) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_max_url_ratio_24h,
        ARG_MAX(
            max_upper_reference_labevent_id_24h,
            STRUCT_PACK(
                url_ratio := max_upper_reference_ratio_24h,
                inverse_distance_hours :=
                    -ABS(max_upper_reference_delta_hours_24h),
                lab_group := lab_group
            )
        ) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
              AND max_upper_reference_ratio_24h IS NOT NULL
        ) AS troponin_max_url_labevent_id_24h,
        ARG_MAX(
            max_upper_reference_delta_hours_24h,
            STRUCT_PACK(
                url_ratio := max_upper_reference_ratio_24h,
                inverse_distance_hours :=
                    -ABS(max_upper_reference_delta_hours_24h),
                lab_group := lab_group
            )
        ) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
              AND max_upper_reference_ratio_24h IS NOT NULL
        ) AS troponin_max_url_delta_hours_24h,
        ARG_MIN(nearest_delta_hours_standard,
                ABS(nearest_delta_hours_standard)) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_nearest_delta_hours,
        ARG_MIN(nearest_delta_hours_strict,
                ABS(nearest_delta_hours_strict)) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_nearest_delta_hours_strict,
        ARG_MIN(nearest_delta_hours,
                ABS(nearest_delta_hours)) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_nearest_delta_hours_24h,
        ARG_MIN(nearest_labevent_id_standard,
                ABS(nearest_delta_hours_standard)) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_nearest_labevent_id,
        ARG_MIN(nearest_labevent_id_strict,
                ABS(nearest_delta_hours_strict)) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_nearest_labevent_id_strict,
        ARG_MIN(nearest_labevent_id,
                ABS(nearest_delta_hours)) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_nearest_labevent_id_24h,
        SUM(count_strict) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_count_strict,
        SUM(count_standard) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_count_standard,
        SUM(count_24h) FILTER (
            WHERE lab_group IN ('troponin_i', 'troponin_t')
        ) AS troponin_count_24h,
        ARG_MIN(nearest_value_6h, ABS(nearest_delta_hours_6h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest,
        ARG_MIN_NULL(nearest_bound_6h, ABS(nearest_delta_hours_6h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_bound,
        ARG_MIN_NULL(nearest_censoring_6h, ABS(nearest_delta_hours_6h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_censoring,
        ARG_MIN(nearest_delta_hours_6h, ABS(nearest_delta_hours_6h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_delta_hours,
        ARG_MIN(nearest_labevent_id_6h,
                ABS(nearest_delta_hours_6h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_labevent_id,
        ARG_MIN(nearest_value_2h, ABS(nearest_delta_hours_2h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_2h,
        ARG_MIN(nearest_delta_hours_2h, ABS(nearest_delta_hours_2h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_delta_hours_2h,
        ARG_MIN(nearest_labevent_id_2h,
                ABS(nearest_delta_hours_2h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_labevent_id_2h,
        ARG_MIN(nearest_value_12h, ABS(nearest_delta_hours_12h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_12h,
        ARG_MIN(nearest_delta_hours_12h, ABS(nearest_delta_hours_12h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_delta_hours_12h,
        ARG_MIN(nearest_labevent_id_12h,
                ABS(nearest_delta_hours_12h)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_labevent_id_12h,
        ARG_MIN(nearest_value, ABS(nearest_delta_hours)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_24h,
        ARG_MIN(nearest_delta_hours, ABS(nearest_delta_hours)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_delta_hours_24h,
        ARG_MIN(nearest_labevent_id,
                ABS(nearest_delta_hours)) FILTER (
            WHERE lab_group = 'potassium'
        ) AS potassium_nearest_labevent_id_24h,
        MAX(count_2h) FILTER (WHERE lab_group = 'potassium')
            AS potassium_count_2h,
        MAX(count_6h) FILTER (WHERE lab_group = 'potassium')
            AS potassium_count_6h,
        MAX(count_12h) FILTER (WHERE lab_group = 'potassium')
            AS potassium_count_12h,
        MAX(count_24h) FILTER (WHERE lab_group = 'potassium')
            AS potassium_count_24h,
        {generated_lab_window_pivot}
        MAX(max_upper_reference_ratio_standard) FILTER (
            WHERE lab_group = 'ntprobnp'
        ) AS ntprobnp_max_url_ratio,
        ARG_MAX(
            max_upper_reference_labevent_id_standard,
            STRUCT_PACK(
                url_ratio := max_upper_reference_ratio_standard,
                inverse_distance_hours :=
                    -ABS(max_upper_reference_delta_hours_standard)
            )
        ) FILTER (
            WHERE lab_group = 'ntprobnp'
              AND max_upper_reference_ratio_standard IS NOT NULL
        ) AS ntprobnp_max_url_labevent_id,
        ARG_MAX(
            max_upper_reference_delta_hours_standard,
            STRUCT_PACK(
                url_ratio := max_upper_reference_ratio_standard,
                inverse_distance_hours :=
                    -ABS(max_upper_reference_delta_hours_standard)
            )
        ) FILTER (
            WHERE lab_group = 'ntprobnp'
              AND max_upper_reference_ratio_standard IS NOT NULL
        ) AS ntprobnp_max_url_delta_hours,
        ARG_MIN(nearest_value, ABS(nearest_delta_hours)) FILTER (
            WHERE lab_group = 'creatinine'
        ) AS creatinine_nearest,
        ARG_MIN(nearest_value, ABS(nearest_delta_hours)) FILTER (
            WHERE lab_group = 'hemoglobin'
        ) AS hemoglobin_nearest,
        ARG_MIN(nearest_value, ABS(nearest_delta_hours)) FILTER (
            WHERE lab_group = 'lactate'
        ) AS lactate_nearest
    FROM read_parquet('{labs}')
    WHERE {bucket}
    GROUP BY subject_id, study_id
    """


def _medication_windows_query(
    master: str,
    medications: str,
    *,
    partition_index: int,
    partition_count: int,
) -> str:
    """Build one bounded, OR-free medication/ECG range-join partition."""

    if partition_count < 1:
        raise ValueError("partition_count must be positive")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index is outside partition_count")
    candidate_columns = """
        m.subject_id,
        m.study_id,
        m.ecg_time,
        e.hadm_id,
        e.ed_stay_id,
        e.event_time,
        e.start_time,
        e.end_time,
        e.medication_raw,
        e.medication_class,
        e.source_table,
        e.exposure_type,
        e.confidence,
        e.source_status
    """
    return f"""
        WITH ecgs AS (
            SELECT subject_id, study_id, ecg_time
            FROM read_parquet('{master}')
            WHERE hash(subject_id) % {partition_count} = {partition_index}
        ),
        medications AS (
            SELECT * FROM read_parquet('{medications}')
            WHERE hash(subject_id) % {partition_count} = {partition_index}
        ),
        near_event_candidates AS (
            SELECT {candidate_columns}
            FROM ecgs m
            JOIN medications e
              ON m.subject_id = e.subject_id
             AND e.event_time >= m.ecg_time - INTERVAL 48 HOUR
             AND e.event_time <= m.ecg_time + INTERVAL 2 HOUR
        ),
        bounded_active_order_candidates AS (
            SELECT {candidate_columns}
            FROM ecgs m
            JOIN medications e
              ON m.subject_id = e.subject_id
             AND e.source_table IN ('prescriptions', 'pharmacy')
             AND e.start_time <= m.ecg_time
             AND e.end_time >= m.ecg_time
        ),
        open_active_order_candidates AS (
            SELECT {candidate_columns}
            FROM ecgs m
            JOIN medications e
              ON m.subject_id = e.subject_id
             AND e.source_table IN ('prescriptions', 'pharmacy')
             AND e.start_time <= m.ecg_time
             AND e.end_time IS NULL
        ),
        candidate_matches AS (
            SELECT * FROM near_event_candidates
            UNION
            SELECT * FROM bounded_active_order_candidates
            UNION
            SELECT * FROM open_active_order_candidates
        )
        SELECT
            subject_id,
            study_id,
            medication_class,
            BOOL_OR(
                source_table = 'emar'
                AND event_time BETWEEN
                    ecg_time
                    - CASE
                        WHEN medication_class IN (
                            'digoxin', 'antiarrhythmic_class_i_iii'
                        ) THEN INTERVAL 48 HOUR
                        ELSE INTERVAL 24 HOUR
                      END
                    AND ecg_time + INTERVAL 2 HOUR
            ) AS administered_near_ecg,
            BOOL_OR(
                source_table IN ('prescriptions', 'pharmacy')
                AND start_time <= ecg_time
                AND (end_time IS NULL OR end_time >= ecg_time)
            ) AS active_order_at_ecg,
            BOOL_OR(
                source_table = 'pyxis'
                AND event_time BETWEEN
                    ecg_time
                    - CASE
                        WHEN medication_class IN (
                            'digoxin', 'antiarrhythmic_class_i_iii'
                        ) THEN INTERVAL 48 HOUR
                        ELSE INTERVAL 24 HOUR
                      END
                    AND ecg_time + INTERVAL 2 HOUR
            ) AS ed_dispensed_near_ecg,
            BOOL_OR(
                source_table = 'medrecon'
                AND event_time BETWEEN
                    ecg_time
                    - CASE
                        WHEN medication_class IN (
                            'digoxin', 'antiarrhythmic_class_i_iii'
                        ) THEN INTERVAL 48 HOUR
                        ELSE INTERVAL 24 HOUR
                      END
                    AND ecg_time + INTERVAL 2 HOUR
            ) AS home_medication_reported_near_ecg,
            COUNT(*) AS evidence_count,
            ARG_MIN(
                medication_raw,
                STRUCT_PACK(
                    distance_seconds := ABS(
                        DATE_DIFF('second', ecg_time, event_time)
                    ),
                    normalized_name := LOWER(COALESCE(medication_raw, '')),
                    exact_name := COALESCE(medication_raw, ''),
                    source_table := source_table,
                    exposure_type := exposure_type
                )
            ) AS nearest_medication,
            MIN(ABS(DATE_DIFF('second', ecg_time, event_time))) / 3600.0
                AS nearest_delta_hours
        FROM candidate_matches
        GROUP BY subject_id, study_id, medication_class
    """


def _ecg_medication_exposure_pair_query(master: str, medications: str) -> str:
    """Nearest ECG strictly before and nearest ECG strictly after a single
    discrete AV-nodal-blocking medication exposure (administered/dispensed
    event, not a multi-day active order), within a tight +/-2h window on
    each side -- deliberately tighter than the -48h/+2h association
    window used elsewhere, since this needs temporal proximity to the
    exposure instant rather than mere prior exposure. Only exposures with
    BOTH a before- and an after-ECG are kept (that is the point: a
    before/after snapshot pair to compare)."""
    class_list_sql = ", ".join(
        f"'{name}'" for name in sorted(AV_NODAL_BLOCKING_MEDICATION_CLASSES)
    )
    return f"""
        WITH exposures AS (
            SELECT DISTINCT subject_id, medication_class, event_time
            FROM read_parquet('{medications}')
            WHERE source_table IN ('emar', 'pyxis')
              AND medication_class IN ({class_list_sql})
        ),
        ranked_before AS (
            SELECT
                e.subject_id,
                e.medication_class,
                e.event_time,
                m.study_id AS study_id_before,
                m.ecg_time AS ecg_time_before,
                ROW_NUMBER() OVER (
                    PARTITION BY e.subject_id, e.medication_class, e.event_time
                    -- Two ECGs can sit exactly the same distance from the
                    -- exposure instant. Without a final key the winner is
                    -- whichever row the scan happened to produce first, so
                    -- the same input yields different study_ids from run to
                    -- run -- measured on the live project: 10 of 4,296 pairs.
                    ORDER BY e.event_time - m.ecg_time ASC, m.study_id ASC
                ) AS rn
            FROM exposures e
            JOIN read_parquet('{master}') m
              ON m.subject_id = e.subject_id
             -- Strictly before: an ECG timestamped exactly at administration
             -- is neither a defensible baseline nor a reason to discard a
             -- valid earlier/later pair.
             AND m.ecg_time < e.event_time
             AND m.ecg_time >= e.event_time - INTERVAL 2 HOUR
        ),
        ranked_after AS (
            SELECT
                e.subject_id,
                e.medication_class,
                e.event_time,
                m.study_id AS study_id_after,
                m.ecg_time AS ecg_time_after,
                ROW_NUMBER() OVER (
                    PARTITION BY e.subject_id, e.medication_class, e.event_time
                    ORDER BY m.ecg_time - e.event_time ASC, m.study_id ASC
                ) AS rn
            FROM exposures e
            JOIN read_parquet('{master}') m
              ON m.subject_id = e.subject_id
             -- Strictly after for the same reason.  Using >= selected the
             -- simultaneous ECG on both sides and the later self-pair guard
             -- then threw away the exposure instead of choosing the next ECG.
             AND m.ecg_time > e.event_time
             AND m.ecg_time <= e.event_time + INTERVAL 2 HOUR
        )
        SELECT
            b.subject_id,
            b.medication_class,
            b.event_time AS exposure_event_time,
            b.study_id_before,
            b.ecg_time_before,
            a.study_id_after,
            a.ecg_time_after,
            DATE_DIFF('second', b.ecg_time_before, b.event_time) / 3600.0
                AS hours_before_to_exposure,
            DATE_DIFF('second', b.event_time, a.ecg_time_after) / 3600.0
                AS hours_exposure_to_after
        FROM (SELECT * FROM ranked_before WHERE rn = 1) b
        JOIN (SELECT * FROM ranked_after WHERE rn = 1) a
          ON b.subject_id = a.subject_id
         AND b.medication_class = a.medication_class
         AND b.event_time = a.event_time
         AND b.study_id_before != a.study_id_after
    """


def _medication_order_administration_context_query(
    source_relations: dict[str, str],
    medication_patterns: list[str],
) -> str:
    poe_relation = source_relations.get("poe")
    poe_detail_relation = source_relations.get("poe_detail")
    emar_detail_relation = source_relations.get("emar_detail")
    if poe_relation is not None:
        poe_orders = f"""
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                NULLIF(TRIM(poe_id), '') AS poe_id,
                TRY_CAST(poe_seq AS INTEGER) AS poe_seq,
                TRY_CAST(ordertime AS TIMESTAMP) AS order_time,
                NULLIF(TRIM(order_type), '') AS order_type,
                NULLIF(TRIM(order_subtype), '') AS order_subtype,
                NULLIF(TRIM(transaction_type), '') AS transaction_type,
                NULLIF(TRIM(order_status), '') AS order_status
            FROM {poe_relation}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND NULLIF(TRIM(poe_id), '') IS NOT NULL
        """
    else:
        poe_orders = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS hadm_id,
                NULL::VARCHAR AS poe_id,
                NULL::INTEGER AS poe_seq,
                NULL::TIMESTAMP AS order_time,
                NULL::VARCHAR AS order_type,
                NULL::VARCHAR AS order_subtype,
                NULL::VARCHAR AS transaction_type,
                NULL::VARCHAR AS order_status
            WHERE FALSE
        """
    if poe_detail_relation is not None:
        poe_details = f"""
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                NULLIF(TRIM(poe_id), '') AS poe_id,
                TRY_CAST(poe_seq AS INTEGER) AS poe_seq,
                MAX(NULLIF(TRIM(field_value), '')) FILTER (
                    WHERE LOWER(TRIM(field_name)) = 'route'
                ) AS detail_route,
                COUNT(*) AS detail_field_count
            FROM {poe_detail_relation}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND NULLIF(TRIM(poe_id), '') IS NOT NULL
              AND TRY_CAST(poe_seq AS INTEGER) IS NOT NULL
            GROUP BY subject_id, poe_id, poe_seq
        """
    else:
        poe_details = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::VARCHAR AS poe_id,
                NULL::INTEGER AS poe_seq,
                NULL::VARCHAR AS detail_route,
                NULL::BIGINT AS detail_field_count
            WHERE FALSE
        """
    if emar_detail_relation is not None:
        emar_details = f"""
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                NULLIF(TRIM(emar_id), '') AS emar_id,
                TRY_CAST(emar_seq AS INTEGER) AS emar_seq,
                MAX(NULLIF(TRIM(route), '')) AS route,
                MAX(NULLIF(TRIM(dose_given), '')) AS dose_given,
                MAX(NULLIF(TRIM(dose_given_unit), ''))
                    AS dose_given_unit,
                MAX(NULLIF(TRIM(administration_type), ''))
                    AS administration_type,
                BOOL_OR(
                    LOWER(COALESCE(complete_dose_not_given, '')) IN (
                        'y', 'yes', 'true', '1'
                    )
                ) AS incomplete_dose,
                COUNT(*) AS detail_row_count
            FROM {emar_detail_relation}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
              AND NULLIF(TRIM(emar_id), '') IS NOT NULL
              AND TRY_CAST(emar_seq AS INTEGER) IS NOT NULL
            GROUP BY subject_id, emar_id, emar_seq
        """
    else:
        emar_details = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::VARCHAR AS emar_id,
                NULL::INTEGER AS emar_seq,
                NULL::VARCHAR AS route,
                NULL::VARCHAR AS dose_given,
                NULL::VARCHAR AS dose_given_unit,
                NULL::VARCHAR AS administration_type,
                NULL::BOOLEAN AS incomplete_dose,
                NULL::BIGINT AS detail_row_count
            WHERE FALSE
        """

    source_queries: list[str] = []
    emar_relation = source_relations.get("emar")
    if emar_relation is not None:
        source_queries.append(
            f"""
            SELECT
                TRY_CAST(e.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(e.hadm_id AS BIGINT) AS hadm_id,
                'emar' AS source_table,
                NULLIF(TRIM(e.emar_id), '') AS source_record_id,
                TRY_CAST(e.charttime AS TIMESTAMP) AS event_time,
                TRY_CAST(e.charttime AS TIMESTAMP) AS start_time,
                TRY_CAST(e.charttime AS TIMESTAMP) AS end_time,
                NULLIF(TRIM(e.medication), '') AS medication_raw,
                NULLIF(TRIM(e.poe_id), '') AS poe_id,
                p.poe_seq,
                p.order_time,
                p.order_type,
                p.order_subtype,
                p.transaction_type,
                p.order_status,
                COALESCE(d.route, p.detail_route) AS route,
                d.dose_given AS dose_raw,
                d.dose_given_unit AS dose_unit,
                d.administration_type,
                d.incomplete_dose,
                d.detail_row_count,
                p.detail_field_count,
                NULLIF(TRIM(e.event_txt), '') AS source_status,
                'post_ecg_not_predictive_input' AS temporal_use_policy
            FROM {emar_relation} e
            LEFT JOIN emar_details d
              ON TRY_CAST(e.subject_id AS BIGINT) = d.subject_id
             AND NULLIF(TRIM(e.emar_id), '') = d.emar_id
             AND TRY_CAST(e.emar_seq AS INTEGER) = d.emar_seq
            LEFT JOIN poe_latest p
              ON TRY_CAST(e.subject_id AS BIGINT) = p.subject_id
             AND NULLIF(TRIM(e.poe_id), '') = p.poe_id
            WHERE TRY_CAST(e.subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(e.charttime AS TIMESTAMP) IS NOT NULL
              AND ({_name_condition("e.medication", medication_patterns)})
            """
        )
    prescriptions_relation = source_relations.get("prescriptions")
    if prescriptions_relation is not None:
        source_queries.append(
            f"""
            SELECT
                TRY_CAST(rx.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(rx.hadm_id AS BIGINT) AS hadm_id,
                'prescriptions' AS source_table,
                COALESCE(
                    NULLIF(TRIM(rx.pharmacy_id), ''),
                    NULLIF(TRIM(rx.poe_id), '')
                ) AS source_record_id,
                TRY_CAST(rx.starttime AS TIMESTAMP) AS event_time,
                TRY_CAST(rx.starttime AS TIMESTAMP) AS start_time,
                TRY_CAST(rx.stoptime AS TIMESTAMP) AS end_time,
                NULLIF(TRIM(rx.drug), '') AS medication_raw,
                NULLIF(TRIM(rx.poe_id), '') AS poe_id,
                TRY_CAST(rx.poe_seq AS INTEGER) AS poe_seq,
                p.order_time,
                p.order_type,
                p.order_subtype,
                p.transaction_type,
                p.order_status,
                COALESCE(NULLIF(TRIM(rx.route), ''), p.detail_route)
                    AS route,
                NULLIF(TRIM(rx.dose_val_rx), '') AS dose_raw,
                NULLIF(TRIM(rx.dose_unit_rx), '') AS dose_unit,
                NULL::VARCHAR AS administration_type,
                NULL::BOOLEAN AS incomplete_dose,
                NULL::BIGINT AS detail_row_count,
                p.detail_field_count,
                NULL::VARCHAR AS source_status,
                'post_ecg_not_predictive_input' AS temporal_use_policy
            FROM {prescriptions_relation} rx
            LEFT JOIN poe_enriched p
              ON TRY_CAST(rx.subject_id AS BIGINT) = p.subject_id
             AND NULLIF(TRIM(rx.poe_id), '') = p.poe_id
             AND TRY_CAST(rx.poe_seq AS INTEGER) = p.poe_seq
            WHERE TRY_CAST(rx.subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(rx.starttime AS TIMESTAMP) IS NOT NULL
              AND ({_name_condition("rx.drug", medication_patterns)})
            """
        )
    pharmacy_relation = source_relations.get("pharmacy")
    if pharmacy_relation is not None:
        source_queries.append(
            f"""
            SELECT
                TRY_CAST(ph.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(ph.hadm_id AS BIGINT) AS hadm_id,
                'pharmacy' AS source_table,
                COALESCE(
                    NULLIF(TRIM(ph.pharmacy_id), ''),
                    NULLIF(TRIM(ph.poe_id), '')
                ) AS source_record_id,
                TRY_CAST(ph.starttime AS TIMESTAMP) AS event_time,
                TRY_CAST(ph.starttime AS TIMESTAMP) AS start_time,
                TRY_CAST(ph.stoptime AS TIMESTAMP) AS end_time,
                NULLIF(TRIM(ph.medication), '') AS medication_raw,
                NULLIF(TRIM(ph.poe_id), '') AS poe_id,
                p.poe_seq,
                p.order_time,
                p.order_type,
                p.order_subtype,
                p.transaction_type,
                p.order_status,
                COALESCE(NULLIF(TRIM(ph.route), ''), p.detail_route)
                    AS route,
                NULL::VARCHAR AS dose_raw,
                NULL::VARCHAR AS dose_unit,
                NULL::VARCHAR AS administration_type,
                NULL::BOOLEAN AS incomplete_dose,
                NULL::BIGINT AS detail_row_count,
                p.detail_field_count,
                NULLIF(TRIM(ph.status), '') AS source_status,
                'post_ecg_not_predictive_input' AS temporal_use_policy
            FROM {pharmacy_relation} ph
            LEFT JOIN poe_latest p
              ON TRY_CAST(ph.subject_id AS BIGINT) = p.subject_id
             AND NULLIF(TRIM(ph.poe_id), '') = p.poe_id
            WHERE TRY_CAST(ph.subject_id AS BIGINT) IS NOT NULL
              AND TRY_CAST(ph.starttime AS TIMESTAMP) IS NOT NULL
              AND ({_name_condition("ph.medication", medication_patterns)})
            """
        )
    if not source_queries:
        source_queries.append(
            """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS hadm_id,
                NULL::VARCHAR AS source_table,
                NULL::VARCHAR AS source_record_id,
                NULL::TIMESTAMP AS event_time,
                NULL::TIMESTAMP AS start_time,
                NULL::TIMESTAMP AS end_time,
                NULL::VARCHAR AS medication_raw,
                NULL::VARCHAR AS poe_id,
                NULL::INTEGER AS poe_seq,
                NULL::TIMESTAMP AS order_time,
                NULL::VARCHAR AS order_type,
                NULL::VARCHAR AS order_subtype,
                NULL::VARCHAR AS transaction_type,
                NULL::VARCHAR AS order_status,
                NULL::VARCHAR AS route,
                NULL::VARCHAR AS dose_raw,
                NULL::VARCHAR AS dose_unit,
                NULL::VARCHAR AS administration_type,
                NULL::BOOLEAN AS incomplete_dose,
                NULL::BIGINT AS detail_row_count,
                NULL::BIGINT AS detail_field_count,
                NULL::VARCHAR AS source_status,
                NULL::VARCHAR AS temporal_use_policy
            WHERE FALSE
            """
        )
    union_sql = "\nUNION ALL\n".join(source_queries)
    return f"""
        WITH poe_orders AS (
            {poe_orders}
        ),
        poe_details AS (
            {poe_details}
        ),
        poe_enriched AS (
            SELECT
                o.*,
                d.detail_route,
                d.detail_field_count
            FROM poe_orders o
            LEFT JOIN poe_details d
              ON o.subject_id = d.subject_id
             AND o.poe_id = d.poe_id
             AND o.poe_seq = d.poe_seq
        ),
        poe_latest AS (
            SELECT *
            FROM poe_enriched
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY subject_id, poe_id
                ORDER BY
                    poe_seq DESC NULLS LAST,
                    order_time DESC NULLS LAST,
                    detail_route NULLS LAST,
                    detail_field_count DESC NULLS LAST
            ) = 1
        ),
        emar_details AS (
            {emar_details}
        )
        {union_sql}
    """


def _build_echo_evidence(
    connection,
    config: MimicProjectConfig,
    progress: Optional[Callable[[str], None]],
) -> list[EvidenceArtifact]:
    if progress:
        progress("Normalisiere strukturierte Echo-Messwerte")
    source_long_path = _sql_path(
        config.standardized_root / "echo_measurements_long.parquet"
    )
    normalized_path = config.evidence_root / "echo_measurements_normalized.parquet"
    value_status = _echo_value_status_sql()
    normalized_query = f"""
        WITH classified AS (
            SELECT
                *,
                {value_status} AS value_status
            FROM read_parquet('{source_long_path}')
        )
        SELECT
            *,
            CASE
                WHEN value_status = 'usable_numeric' THEN result_numeric
            END AS result_numeric_normalized
        FROM classified
    """
    normalized_count = _copy_query_atomic(
        connection,
        normalized_query,
        normalized_path,
    )
    issues_path = config.quarantine_root / "echo_measurement_value_issues.parquet"
    issues_query = f"""
        SELECT *
        FROM read_parquet('{_sql_path(normalized_path)}')
        WHERE value_status IN ('unexpected_unit', 'implausible_numeric')
    """
    issues_count = _copy_query_atomic(connection, issues_query, issues_path)

    if progress:
        progress("Erzeuge strukturierte Echo-Wide-Tabelle")
    long_path = _sql_path(normalized_path)
    wide_path = config.evidence_root / "echo_measurements_wide.parquet"
    valve_columns = {
        "aortic_stenosis": "aortic_stenosis",
        "aortic_regurgitation": "aortic_regurg",
        "mitral_stenosis": "mitral_stenosis",
        "mitral_regurgitation": "mitral_regurg",
        "tricuspid_regurgitation": "tricuspid_regurg",
    }
    valve_sql = []
    for output_name, measurement_name in valve_columns.items():
        raw_expression = (
            f"MAX(result_raw) FILTER (WHERE measurement = '{measurement_name}')"
        )
        valve_sql.append(
            f"{raw_expression} AS {output_name}_raw"
        )
        valve_sql.append(
            f"{_severity_sql(raw_expression)} AS {output_name}_severity"
        )
    valves = ",\n            ".join(valve_sql)
    # Diastolic function, aortic-valve haemodynamics, linear atrial
    # dimensions and the aortic root. None of these reached the wide table
    # before, which left the P-wave and P-loop markers -- the project's
    # primary markers for every atrial endpoint -- without any atrial-size
    # or filling reference at all. Generated rather than hand-written, for
    # the same reason the lab pivot is.
    numeric_terms = [
        f"MAX(result_numeric_normalized) FILTER "
        f"(WHERE measurement IN ({sources})) AS {column}"
        for column, sources in _echo_numeric_select_terms()
    ]
    echo_numeric = ",\n            ".join(numeric_terms)
    ordinal_terms: list[str] = []
    for column, source in _ECHO_ORDINAL_MEASUREMENTS:
        raw_expression = (
            f"MAX(result_raw) FILTER (WHERE measurement = '{source}')"
        )
        ordinal_terms.append(f"{raw_expression} AS {column}_raw")
        ordinal_terms.append(
            f"{_severity_sql(raw_expression)} AS {column}_severity"
        )
    echo_ordinal = ",\n            ".join(ordinal_terms)
    text_terms = [
        f"MAX(result_raw) FILTER (WHERE measurement = '{source}') AS {column}"
        for column, source in _ECHO_TEXT_MEASUREMENTS
    ]
    text_terms.extend(
        f"MAX(result_raw) FILTER (WHERE measurement = '{segment}') AS {segment}"
        for segment in _ECHO_WALL_MOTION_SEGMENTS
    )
    echo_text = ",\n            ".join(text_terms)

    wide_query = f"""
        SELECT
            measurement_id,
            MIN(subject_id) AS subject_id,
            MIN(measurement_datetime) AS echo_time,
            MIN(test_type) AS test_type,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement = 'lvef_3d'
            )
                AS lvef_3d,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement = 'biplane_lvef'
            )
                AS lvef_biplane,
            MAX(result_numeric_normalized) FILTER (WHERE measurement = 'lvef')
                AS lvef_generic,
            COALESCE(
                MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'lvef_3d'
                ),
                MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'biplane_lvef'
                ),
                MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'lvef'
                )
            ) AS preferred_lvef,
            CASE
                WHEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'lvef_3d'
                ) IS NOT NULL THEN '3d'
                WHEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'biplane_lvef'
                ) IS NOT NULL THEN 'biplane'
                WHEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'lvef'
                ) IS NOT NULL THEN 'generic'
            END AS preferred_lvef_method,
            (
                GREATEST(
                    COALESCE(MAX(result_numeric_normalized) FILTER (
                        WHERE measurement = 'lvef_3d'), -1e9),
                    COALESCE(MAX(result_numeric_normalized) FILTER (
                        WHERE measurement = 'biplane_lvef'), -1e9),
                    COALESCE(MAX(result_numeric_normalized) FILTER (
                        WHERE measurement = 'lvef'), -1e9)
                )
                -
                LEAST(
                    COALESCE(MIN(result_numeric_normalized) FILTER (
                        WHERE measurement = 'lvef_3d'), 1e9),
                    COALESCE(MIN(result_numeric_normalized) FILTER (
                        WHERE measurement = 'biplane_lvef'), 1e9),
                    COALESCE(MIN(result_numeric_normalized) FILTER (
                        WHERE measurement = 'lvef'), 1e9)
                )
            ) > 10 AS lvef_conflict_gt_10,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement IN ('lv_mass_index', 'lvmi')
            ) AS lv_mass_index,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement = 'septal_thickness'
            )
                AS septal_thickness_cm,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement = 'inf_lat_thickness'
            )
                AS posterior_wall_thickness_cm,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement IN (
                    'lvedd', 'lvid_d', 'lv_dimen_diastole'
                )
            ) AS lv_internal_diameter_diastole_cm,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement IN (
                    'lvesd', 'lvid_s', 'lv_dimen_systole'
                )
            ) AS lv_internal_diameter_systole_cm,
            MAX(result_raw) FILTER (WHERE measurement = 'lv_wall_thickness')
                AS lv_wall_category,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement IN ('la_vol_index', 'lavi')
            ) AS la_volume_index_direct,
            MAX(result_numeric_normalized) FILTER (WHERE measurement = 'la_vol')
                AS la_volume_ml,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement = 'body_surface_area'
            ) AS echo_body_surface_area_m2,
            COALESCE(
                MAX(result_numeric_normalized) FILTER (
                    WHERE measurement IN ('la_vol_index', 'lavi')
                ),
                MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'la_vol'
                ) / NULLIF(
                    MAX(result_numeric_normalized) FILTER (
                        WHERE measurement = 'body_surface_area'
                    ),
                    0
                )
            ) AS la_volume_index,
            CASE
                WHEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement IN ('la_vol_index', 'lavi')
                ) IS NOT NULL THEN 'direct'
                WHEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'la_vol'
                ) IS NOT NULL
                 AND MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'body_surface_area'
                ) IS NOT NULL THEN 'derived_la_volume_over_echo_bsa'
            END AS la_volume_index_method,
            MAX(result_raw) FILTER (WHERE measurement = 'la_size')
                AS la_size_category,
            MAX(result_raw) FILTER (WHERE measurement = 'ra_size')
                AS ra_size_category,
            MAX(result_numeric_normalized) FILTER (WHERE measurement = 'rv_diam')
                AS rv_basal_diameter_cm,
            MAX(result_raw) FILTER (WHERE measurement = 'rv_chamber_size')
                AS rv_size_category,
            MAX(result_raw) FILTER (WHERE measurement = 'rv_function')
                AS rv_function_category,
            MAX(result_numeric_normalized) FILTER (WHERE measurement = 'tapse')
                AS tapse_cm,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement IN (
                    'tv_est_pa_press_min', 'rvsp', 'pasp',
                    'rv_systolic_pressure'
                )
            ) AS direct_estimated_pa_pressure_mmhg,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement IN ('tr_velocity', 'tr_peak_velocity')
            ) AS tr_velocity_m_per_s,
            MAX(result_numeric_normalized) FILTER (
                WHERE measurement = 'tr_mmhg'
            ) AS tr_pressure_gradient_mmhg,
            MAX(result_raw) FILTER (WHERE measurement = 'ivc')
                AS ivc_ra_pressure_category,
            CASE
                WHEN LOWER(COALESCE(MAX(result_raw) FILTER (
                    WHERE measurement = 'ivc'
                ), '')) LIKE '%0-5mmhg%' THEN 3.0
                WHEN LOWER(COALESCE(MAX(result_raw) FILTER (
                    WHERE measurement = 'ivc'
                ), '')) LIKE '%5-10mmhg%' THEN 8.0
                WHEN LOWER(COALESCE(MAX(result_raw) FILTER (
                    WHERE measurement = 'ivc'
                ), '')) LIKE '%10-15mmhg%' THEN 12.5
                WHEN LOWER(COALESCE(MAX(result_raw) FILTER (
                    WHERE measurement = 'ivc'
                ), '')) LIKE '%>15mmhg%' THEN 15.0
            END AS estimated_ra_pressure_mmhg,
            MAX(result_raw) FILTER (WHERE measurement = 'phtn_severity')
                AS pulmonary_hypertension_category,
            {echo_numeric},
            {echo_ordinal},
            -- E/A prefers the reported ratio and falls back to the two
            -- velocities, so a study that recorded only the components is not
            -- silently lost.
            COALESCE(
                MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'mv_peak_e_a'
                ),
                MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'mv_peak_e'
                ) / NULLIF(
                    MAX(result_numeric_normalized) FILTER (
                        WHERE measurement = 'mv_peak_a'
                    ),
                    0
                )
            ) AS mitral_e_a_ratio,
            -- Relative wall thickness from the echo's own linear dimensions:
            -- 2 * posterior wall / LV internal diameter in diastole. Above
            -- ~0.42 separates concentric from eccentric remodelling.
            CASE
                WHEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement IN ('lvedd', 'lvid_d', 'lv_dimen_diastole')
                ) > 0
                THEN 2.0 * MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'inf_lat_thickness'
                ) / MAX(result_numeric_normalized) FILTER (
                    WHERE measurement IN ('lvedd', 'lvid_d', 'lv_dimen_diastole')
                )
            END AS relative_wall_thickness_echo,
            CASE
                WHEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'body_surface_area'
                ) > 0
                THEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'la_dimen'
                ) / MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'body_surface_area'
                )
            END AS la_dimension_indexed,
            {valves},
            {echo_text},
            -- Stroke volume from the outflow tract, the quantity the three
            -- LVOT measurements exist to produce: the cross-sectional area of
            -- a circle of diameter ``lvot_diam`` multiplied by the distance
            -- the blood column travels in one systole. With it a cardiac
            -- output becomes computable on 57.6 % of studies, and with the
            -- ECG's own QT interval an electromechanical window does.
            CASE
                WHEN MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'lvot_diam'
                ) > 0
                 AND MAX(result_numeric_normalized) FILTER (
                    WHERE measurement = 'lvot_vti'
                ) > 0
                THEN 0.785398163397448
                     * POWER(
                         MAX(result_numeric_normalized) FILTER (
                             WHERE measurement = 'lvot_diam'
                         ),
                         2
                       )
                     * MAX(result_numeric_normalized) FILTER (
                         WHERE measurement = 'lvot_vti'
                       )
            END AS lvot_stroke_volume_ml,
            COUNT(*) FILTER (
                WHERE measurement LIKE 'wm_%'
                  AND (
                       LOWER(result_raw) LIKE '%hypokinetic%'
                    OR LOWER(result_raw) LIKE '%akinetic%'
                    OR LOWER(result_raw) LIKE '%dyskinetic%'
                  )
            ) AS rwma_abnormal_segments,
            COUNT(*) FILTER (
                WHERE measurement LIKE 'wm_%'
                  AND LOWER(result_raw) IN ('normal', 'hyperdynamic')
            ) AS rwma_normal_segments,
            COUNT(*) FILTER (
                WHERE measurement LIKE 'wm_%'
                  AND LOWER(result_raw) LIKE '%not well seen%'
            ) AS rwma_not_visualized_segments,
            -- Whether the study carries a wall-motion section at all, which is
            -- not the same question as whether any segment was described.
            -- Measured over 784,482 segment rows: 94.7 % carry an empty result.
            -- Readers write down abnormalities; a segment they judged normal is
            -- usually left blank ('normal' appears on 0.94 % of rows). Counting
            -- only explicit values therefore found 464 negative studies out of
            -- 178,389 candidates, which emptied the endpoint. Separating "no
            -- wall-motion section" from "section present, nothing abnormal
            -- written" is what makes the negative arm definable -- and makes the
            -- assumption behind it visible instead of silent.
            --
            -- The blank rows are not in this table: the long table keeps only
            -- populated results, so counting here reproduced the explicit-only
            -- arm (pilot A117: 456 positives against 3 negatives). The section
            -- count therefore comes from the all-rows table, joined below.
            COUNT(*) FILTER (
                WHERE measurement LIKE 'wm_%'
            ) AS rwma_populated_segments
        FROM read_parquet('{long_path}')
        GROUP BY measurement_id
    """
    all_rows_path = config.standardized_root / "echo_measurements_all.parquet"
    section_source = _sql_path(all_rows_path if all_rows_path.is_file() else normalized_path)
    wide_query = f"""
        WITH populated AS ({wide_query}),
        wall_motion_sections AS (
            {resting_wall_motion_coverage_sql(section_source)}
        )
        SELECT
            populated.* EXCLUDE(rwma_normal_segments, rwma_abnormal_segments, rwma_not_visualized_segments),
            COALESCE(wall_motion_sections.rwma_normal_segments, 0) AS rwma_normal_segments,
            COALESCE(wall_motion_sections.rwma_abnormal_segments, 0) AS rwma_abnormal_segments,
            COALESCE(wall_motion_sections.rwma_not_visualized_segments, 0) AS rwma_not_visualized_segments,
            COALESCE(wall_motion_sections.rwma_uncertain_segments, 0) AS rwma_uncertain_segments,
            COALESCE(wall_motion_sections.rwma_reference_conflict, FALSE) AS rwma_reference_conflict,
            COALESCE(wall_motion_sections.rwma_complete_normal_coverage, FALSE) AS rwma_complete_normal_coverage,
            COALESCE(wall_motion_sections.rwma_reported_segments, 0)
                AS rwma_reported_segments
        FROM populated
        LEFT JOIN wall_motion_sections USING (measurement_id)
    """
    wide_count = _copy_query_atomic(connection, wide_query, wide_path)

    if progress:
        progress("Berechne nearest-before/after Echo-Matches fuer ECGs")
    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    echo_index = _sql_path(config.standardized_root / "echo_study_index.parquet")
    match_path = config.evidence_root / "ecg_echo_matches.parquet"
    match_query = _ecg_echo_match_query(master, echo_index)
    match_count = _copy_query_atomic(connection, match_query, match_path)
    by_type_path = config.evidence_root / "ecg_echo_matches_by_type.parquet"
    by_type_query = _ecg_echo_match_by_type_query(master, echo_index)
    by_type_count = _copy_query_atomic(
        connection,
        by_type_query,
        by_type_path,
    )
    return [
        EvidenceArtifact(
            "echo_measurements_normalized",
            normalized_path,
            normalized_count,
        ),
        EvidenceArtifact(
            "echo_measurement_value_issues",
            issues_path,
            issues_count,
        ),
        EvidenceArtifact("echo_measurements_wide", wide_path, wide_count),
        EvidenceArtifact("ecg_echo_matches", match_path, match_count),
        EvidenceArtifact(
            "ecg_echo_matches_by_type",
            by_type_path,
            by_type_count,
        ),
    ]


def _ecg_echo_match_query(master: str, echo_index: str) -> str:
    return f"""
        WITH candidates AS (
            SELECT
                m.subject_id,
                m.study_id,
                h.measurement_id,
                h.echo_time,
                DATE_DIFF('second', m.ecg_time, h.echo_time) / 3600.0
                    AS delta_hours
            FROM read_parquet('{master}') m
            JOIN read_parquet('{echo_index}') h
              ON m.subject_id = h.subject_id
             AND h.test_type = 'tte'
             AND NOT h.identity_conflict
             AND h.echo_time BETWEEN m.ecg_time - INTERVAL 90 DAY
                                 AND m.ecg_time + INTERVAL 90 DAY
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id, study_id
                    ORDER BY ABS(delta_hours), delta_hours ASC, measurement_id DESC
                ) AS nearest_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        subject_id,
                        study_id,
                        (delta_hours <= 0)
                    ORDER BY delta_hours DESC, measurement_id DESC
                ) AS before_rank,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        subject_id,
                        study_id,
                        (delta_hours >= 0)
                    ORDER BY delta_hours ASC, measurement_id DESC
                ) AS after_rank
            FROM candidates
        )
        SELECT
            subject_id,
            study_id,
            MAX(measurement_id) FILTER (WHERE nearest_rank = 1)
                AS nearest_tte_measurement_id,
            MAX(echo_time) FILTER (WHERE nearest_rank = 1)
                AS nearest_tte_time,
            MAX(delta_hours) FILTER (WHERE nearest_rank = 1)
                AS nearest_tte_delta_hours,
            MAX(measurement_id) FILTER (
                WHERE delta_hours <= 0 AND before_rank = 1
            ) AS nearest_tte_before_id,
            MAX(delta_hours) FILTER (
                WHERE delta_hours <= 0 AND before_rank = 1
            )
                AS nearest_tte_before_delta_hours,
            MAX(measurement_id) FILTER (
                WHERE delta_hours >= 0 AND after_rank = 1
            ) AS nearest_tte_after_id,
            MAX(delta_hours) FILTER (
                WHERE delta_hours >= 0 AND after_rank = 1
            )
                AS nearest_tte_after_delta_hours,
            BOOL_OR(ABS(delta_hours) <= 24 * 7) AS has_tte_7d,
            BOOL_OR(ABS(delta_hours) <= 24 * 30) AS has_tte_30d,
            BOOL_OR(ABS(delta_hours) <= 24 * 90) AS has_tte_90d,
            COUNT(*) AS tte_candidate_count_90d
        FROM ranked
        GROUP BY subject_id, study_id
    """


def _ecg_echo_match_by_type_query(
    master: str,
    echo_index: str,
) -> str:
    return f"""
        WITH candidates AS (
            SELECT
                m.subject_id,
                m.study_id,
                h.test_type,
                h.measurement_id,
                h.echo_time,
                DATE_DIFF('second', m.ecg_time, h.echo_time) / 3600.0
                    AS delta_hours
            FROM read_parquet('{master}') m
            JOIN read_parquet('{echo_index}') h
              ON m.subject_id = h.subject_id
             AND h.test_type IN ('tte', 'tee', 'stress')
             AND NOT h.identity_conflict
             AND h.echo_time BETWEEN m.ecg_time - INTERVAL 90 DAY
                                 AND m.ecg_time + INTERVAL 90 DAY
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id, study_id, test_type
                    ORDER BY ABS(delta_hours), delta_hours ASC, measurement_id DESC
                ) AS nearest_rank
            FROM candidates
        )
        SELECT
            subject_id,
            study_id,
            test_type,
            MAX(measurement_id) FILTER (WHERE nearest_rank = 1)
                AS nearest_measurement_id,
            MAX(echo_time) FILTER (WHERE nearest_rank = 1)
                AS nearest_echo_time,
            MAX(delta_hours) FILTER (WHERE nearest_rank = 1)
                AS nearest_delta_hours,
            BOOL_OR(ABS(delta_hours) <= 24 * 7) AS has_echo_7d,
            BOOL_OR(ABS(delta_hours) <= 24 * 30) AS has_echo_30d,
            BOOL_OR(ABS(delta_hours) <= 24 * 90) AS has_echo_90d,
            COUNT(*) AS candidate_count_90d
        FROM ranked
        GROUP BY subject_id, study_id, test_type
    """


def _build_context_features(
    connection,
    config: MimicProjectConfig,
    progress: Optional[Callable[[str], None]],
) -> list[EvidenceArtifact]:
    if progress:
        progress("Verdichte klinische Evidenz auf eine ECG-zentrierte Featurematrix")
    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    diagnoses = _sql_path(config.evidence_root / "diagnosis_evidence.parquet")
    procedures = _sql_path(config.evidence_root / "procedure_evidence.parquet")
    labs = _sql_path(config.evidence_root / "ecg_lab_windows.parquet")
    troponin = _sql_path(config.evidence_root / "ecg_troponin_serial.parquet")
    medications = _sql_path(config.evidence_root / "ecg_medication_windows.parquet")
    medication_coverage = config.evidence_root / (
        "ecg_medication_source_coverage.parquet"
    )
    echo_matches = _sql_path(config.evidence_root / "ecg_echo_matches.parquet")
    echo_wide = _sql_path(config.evidence_root / "echo_measurements_wide.parquet")
    report_labels = _sql_path(
        config.evidence_root / "machine_report_labels.parquet"
    )
    note_encounter_flags = config.evidence_root / "note_encounter_flags.parquet"
    admissions = _sql_path(config.standardized_root / "admissions.parquet")

    diagnosis_flags = config.evidence_root / "encounter_diagnosis_flags.parquet"
    diagnosis_query = f"""
        SELECT
            subject_id,
            hadm_id,
            {', '.join(
                f"BOOL_OR(code_group = '{group.group_id}') AS dx_{group.group_id}"
                for group in ICD_GROUPS
            )}
        FROM read_parquet('{diagnoses}')
        GROUP BY subject_id, hadm_id
    """
    diagnosis_flag_count = _copy_query_atomic(
        connection, diagnosis_query, diagnosis_flags
    )

    procedure_flags = config.evidence_root / "encounter_procedure_flags.parquet"
    procedure_query = f"""
        SELECT
            subject_id,
            hadm_id,
            {', '.join(
                f"BOOL_OR(code_group = '{group.group_id}') AS proc_{group.group_id}"
                for group in PROCEDURE_GROUPS
            )},
            MIN(chartdate) FILTER (WHERE code_group = 'pci') AS first_pci_date,
            MIN(chartdate) FILTER (
                WHERE code_group = 'coronary_angiography'
            ) AS first_cath_date
        FROM read_parquet('{procedures}')
        GROUP BY subject_id, hadm_id
    """
    procedure_flag_count = _copy_query_atomic(
        connection, procedure_query, procedure_flags
    )

    patient_history = config.evidence_root / "patient_code_history.parquet"
    patient_history_query = f"""
        WITH diagnosis_times AS (
            SELECT
                d.subject_id,
                MIN(a.admittime) FILTER (WHERE d.code_group = 'old_mi')
                    AS first_old_mi_time,
                MIN(a.dischtime) FILTER (
                    WHERE d.code_group = 'acute_mi'
                      AND a.dischtime IS NOT NULL
                ) AS first_completed_acute_mi_time
            FROM read_parquet('{diagnoses}') d
            JOIN read_parquet('{admissions}') a
              USING (subject_id, hadm_id)
            GROUP BY d.subject_id
        ),
        procedure_times AS (
            SELECT
                subject_id,
                MIN(chartdate) FILTER (WHERE code_group = 'pci')
                    AS first_pci_date,
                MIN(chartdate) FILTER (WHERE code_group = 'cabg')
                    AS first_cabg_date
            FROM read_parquet('{procedures}')
            GROUP BY subject_id
        ),
        subjects AS (
            SELECT subject_id FROM diagnosis_times
            UNION
            SELECT subject_id FROM procedure_times
        )
        SELECT
            s.subject_id,
            d.first_old_mi_time,
            d.first_completed_acute_mi_time,
            p.first_pci_date,
            p.first_cabg_date
        FROM subjects s
        LEFT JOIN diagnosis_times d USING (subject_id)
        LEFT JOIN procedure_times p USING (subject_id)
    """
    patient_history_count = _copy_query_atomic(
        connection,
        patient_history_query,
        patient_history,
    )

    encounter_microbiology = (
        config.evidence_root / "encounter_microbiology_flags.parquet"
    )
    organism_conditions = []
    for organism_name in TYPICAL_ENDOCARDITIS_ORGANISMS:
        escaped = organism_name.replace("'", "''")
        organism_conditions.append(
            f"LOWER(organism) LIKE '%{escaped}%'"
        )
    organisms = " OR ".join(organism_conditions)
    microbiology_path = (
        config.standardized_root / "microbiologyevents.parquet"
    )
    if microbiology_path.exists():
        microbiology_query = f"""
            SELECT
                subject_id,
                hadm_id,
                BOOL_OR({organisms}) AS typical_endocarditis_organism,
                COUNT(*) AS microbiology_specimen_count,
                BOOL_OR(organism IS NOT NULL) AS any_positive_culture
            FROM read_parquet('{_sql_path(microbiology_path)}')
            WHERE hadm_id IS NOT NULL
            GROUP BY subject_id, hadm_id
        """
    else:
        microbiology_query = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS hadm_id,
                FALSE AS typical_endocarditis_organism,
                0::BIGINT AS microbiology_specimen_count,
                FALSE AS any_positive_culture
            WHERE FALSE
        """
    encounter_microbiology_count = _copy_query_atomic(
        connection,
        microbiology_query,
        encounter_microbiology,
    )

    # care_setting: ICU-vs-ward exposure and admitting service line for
    # this admission -- both tables are already ingested (transfers.parquet,
    # services.parquet) but were previously unused downstream.
    encounter_care_setting = (
        config.evidence_root / "encounter_care_setting_flags.parquet"
    )
    transfers_path = config.standardized_root / "transfers.parquet"
    services_path = config.standardized_root / "services.parquet"
    services_join_sql = ""
    services_select_sql = "NULL::VARCHAR AS primary_service_this_admission"
    if services_path.exists():
        services_join_sql = f"""
            LEFT JOIN (
                SELECT
                    subject_id,
                    hadm_id,
                    ARG_MIN(current_service, transfer_time)
                        AS primary_service_this_admission
                FROM read_parquet('{_sql_path(services_path)}')
                WHERE hadm_id IS NOT NULL
                GROUP BY subject_id, hadm_id
            ) sv USING (subject_id, hadm_id)
        """
        services_select_sql = "sv.primary_service_this_admission"
    care_setting_query = f"""
        SELECT
            t.subject_id,
            t.hadm_id,
            t.ever_icu_this_admission,
            {services_select_sql} AS primary_service_this_admission
        FROM (
            SELECT
                subject_id,
                hadm_id,
                BOOL_OR(
                    care_unit ILIKE '%ICU%' OR care_unit ILIKE '%CCU%'
                    OR care_unit ILIKE '%SICU%' OR care_unit ILIKE '%MICU%'
                    OR care_unit ILIKE '%TSICU%' OR care_unit ILIKE '%NICU%'
                ) AS ever_icu_this_admission
            FROM read_parquet('{_sql_path(transfers_path)}')
            WHERE hadm_id IS NOT NULL
            GROUP BY subject_id, hadm_id
        ) t
        {services_join_sql}
    """
    encounter_care_setting_count = _copy_query_atomic(
        connection, care_setting_query, encounter_care_setting
    )

    # ed_vitals_trend: serial ED vital-sign variability over the stay, as
    # opposed to the single triage-snapshot vitals already used elsewhere
    # -- a genuine acute-physiological-state dimension the snapshot cannot
    # capture. Requires at least two readings to speak of a "trend".
    ed_vitals_trend = config.evidence_root / "ed_vitals_trend.parquet"
    vitalsign_path = config.standardized_root / "vitalsign.parquet"
    vitals_trend_query = f"""
        SELECT
            subject_id,
            ed_stay_id,
            COUNT(*) AS vitals_ed_reading_count,
            STDDEV(heart_rate) AS vitals_ed_hr_stddev,
            MAX(heart_rate) - MIN(heart_rate) AS vitals_ed_hr_range,
            ARG_MAX(heart_rate, charttime) - ARG_MIN(heart_rate, charttime)
                AS vitals_ed_hr_trend_delta,
            STDDEV(systolic_bp) AS vitals_ed_sbp_stddev,
            MAX(systolic_bp) - MIN(systolic_bp) AS vitals_ed_sbp_range,
            STDDEV(oxygen_saturation) AS vitals_ed_spo2_stddev,
            MIN(oxygen_saturation) AS vitals_ed_spo2_min
        FROM read_parquet('{_sql_path(vitalsign_path)}')
        GROUP BY subject_id, ed_stay_id
        HAVING COUNT(*) >= 2
    """
    ed_vitals_trend_count = _copy_query_atomic(
        connection, vitals_trend_query, ed_vitals_trend
    )

    # ICU feature family: icustay_id linkage, vitals-trend, ventilation,
    # vasopressor dose/duration, fluid balance, GCS at admission. All are
    # optional (icustays/chartevents/inputevents/outputevents/
    # procedureevents may not be ingested) and degrade to an empty
    # 0-row output rather than failing when the underlying table is
    # missing -- same pattern as the microbiology block above.
    # The ICU event tables are ingested into the partitioned, content-
    # addressed clinical-event registry -- there is no flat
    # standardized/chartevents.parquet and there never was. Resolving them as
    # flat files silently produced an empty ICU layer while 62.9 M chartevents
    # rows sat in the registry, so every source below is resolved through the
    # registry (with a raw-CSV fallback) exactly like the lab/medication
    # blocks, and availability is decided by the resolved relation.
    icustays_path = config.standardized_root / "icustays.parquet"
    icu_relations = _icu_event_relations(config)
    icustays_relation = (
        "read_parquet('" + _sql_path(icustays_path) + "')"
        if icustays_path.exists()
        else clinical_event_relation(config, "icustays")
    )
    chartevents_relation = icu_relations.get("chartevents")
    inputevents_relation = icu_relations.get("inputevents")
    outputevents_relation = icu_relations.get("outputevents")
    procedureevents_relation = icu_relations.get("procedureevents")

    icu_vitals_trend = config.evidence_root / "icu_vitals_trend.parquet"
    if chartevents_relation is not None and icustays_relation is not None:
        hr_ids = ", ".join(str(i) for i in ICU_CHARTEVENTS_ITEMIDS["heart_rate"])
        sbp_ids = ", ".join(str(i) for i in ICU_CHARTEVENTS_ITEMIDS["systolic_bp"])
        spo2_ids = ", ".join(str(i) for i in ICU_CHARTEVENTS_ITEMIDS["spo2"])
        # Every aggregate below is FILTERed to one of these three groups, and
        # so is the HAVING, so rows outside them cannot influence a single
        # output value -- they were only ever carried through the join. The
        # widened chart ingest made that visible: the intermediate result grew
        # 1.9x and ran the shadow build out of memory while producing exactly
        # the same numbers. Filtering here keeps the values identical and the
        # cost proportional to what is actually used.
        trend_ids = ", ".join(
            str(itemid)
            for itemid in sorted(
                set(ICU_CHARTEVENTS_ITEMIDS["heart_rate"])
                | set(ICU_CHARTEVENTS_ITEMIDS["systolic_bp"])
                | set(ICU_CHARTEVENTS_ITEMIDS["spo2"])
            )
        )
        icu_vitals_trend_query = f"""
            WITH windowed AS (
                SELECT c.subject_id, c.icustay_id, c.itemid, c.valuenum, c.charttime
                FROM {chartevents_relation} c
                JOIN {icustays_relation} i USING (icustay_id)
                WHERE c.charttime BETWEEN i.intime AND i.intime + INTERVAL 24 HOUR
                  AND c.valuenum IS NOT NULL
                  AND c.itemid IN ({trend_ids})
            )
            SELECT
                subject_id,
                icustay_id,
                COUNT(*) FILTER (WHERE itemid IN ({hr_ids}))
                    AS icu_vitals_reading_count,
                STDDEV(valuenum) FILTER (WHERE itemid IN ({hr_ids}))
                    AS icu_vitals_hr_stddev,
                ARG_MAX(valuenum, charttime) FILTER (WHERE itemid IN ({hr_ids}))
                    - ARG_MIN(valuenum, charttime) FILTER (WHERE itemid IN ({hr_ids}))
                    AS icu_vitals_hr_trend_delta,
                STDDEV(valuenum) FILTER (WHERE itemid IN ({sbp_ids}))
                    AS icu_vitals_sbp_stddev,
                MIN(valuenum) FILTER (WHERE itemid IN ({spo2_ids}))
                    AS icu_vitals_spo2_min
            FROM windowed
            GROUP BY subject_id, icustay_id
            HAVING COUNT(*) FILTER (WHERE itemid IN ({hr_ids})) >= 2
        """
    else:
        icu_vitals_trend_query = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS icustay_id,
                0::BIGINT AS icu_vitals_reading_count,
                NULL::DOUBLE AS icu_vitals_hr_stddev,
                NULL::DOUBLE AS icu_vitals_hr_trend_delta,
                NULL::DOUBLE AS icu_vitals_sbp_stddev,
                NULL::DOUBLE AS icu_vitals_spo2_min
            WHERE FALSE
        """
    icu_vitals_trend_count = _copy_query_atomic(
        connection, icu_vitals_trend_query, icu_vitals_trend
    )

    icu_ventilation_status = config.evidence_root / "icu_ventilation_status.parquet"
    if procedureevents_relation is not None and icustays_relation is not None:
        intubation_id = ICU_VENTILATION_ITEMIDS["intubation"]
        extubation_id = ICU_VENTILATION_ITEMIDS["extubation"]
        icu_ventilation_query = f"""
            SELECT
                i.subject_id,
                i.icustay_id,
                BOOL_OR(p.itemid = {intubation_id}) AS icu_ever_ventilated,
                DATE_DIFF(
                    'second',
                    MIN(p.starttime) FILTER (WHERE p.itemid = {intubation_id}),
                    MAX(COALESCE(p.endtime, p.starttime))
                        FILTER (WHERE p.itemid = {extubation_id})
                ) / 3600.0 AS icu_ventilation_duration_hours
            FROM {icustays_relation} i
            LEFT JOIN {procedureevents_relation} p
              ON i.icustay_id = p.icustay_id
             AND p.itemid IN ({intubation_id}, {extubation_id})
            GROUP BY i.subject_id, i.icustay_id
        """
    else:
        icu_ventilation_query = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS icustay_id,
                FALSE AS icu_ever_ventilated,
                NULL::DOUBLE AS icu_ventilation_duration_hours
            WHERE FALSE
        """
    icu_ventilation_status_count = _copy_query_atomic(
        connection, icu_ventilation_query, icu_ventilation_status
    )

    icu_vasopressor_exposure = (
        config.evidence_root / "icu_vasopressor_exposure.parquet"
    )
    if inputevents_relation is not None and icustays_relation is not None:
        vasopressor_ids = ", ".join(str(i) for i in ICU_VASOPRESSOR_ITEMIDS)
        icu_vasopressor_query = f"""
            SELECT
                i.subject_id,
                i.icustay_id,
                BOOL_OR(v.itemid IS NOT NULL) AS icu_ever_vasopressor,
                DATE_DIFF('second', MIN(v.starttime), MAX(v.endtime)) / 3600.0
                    AS icu_vasopressor_duration_hours
            FROM {icustays_relation} i
            LEFT JOIN {inputevents_relation} v
              ON i.icustay_id = v.icustay_id
             AND v.itemid IN ({vasopressor_ids})
            GROUP BY i.subject_id, i.icustay_id
        """
    else:
        icu_vasopressor_query = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS icustay_id,
                FALSE AS icu_ever_vasopressor,
                NULL::DOUBLE AS icu_vasopressor_duration_hours
            WHERE FALSE
        """
    icu_vasopressor_exposure_count = _copy_query_atomic(
        connection, icu_vasopressor_query, icu_vasopressor_exposure
    )

    icu_fluid_balance = config.evidence_root / "icu_fluid_balance_24h.parquet"
    if (
        inputevents_relation is not None
        and outputevents_relation is not None
        and icustays_relation is not None
    ):
        icu_fluid_query = f"""
            WITH intake AS (
                SELECT
                    i.icustay_id,
                    SUM(v.amount) AS total_in,
                    COUNT(v.amount) AS in_count
                FROM {icustays_relation} i
                LEFT JOIN {inputevents_relation} v
                  ON i.icustay_id = v.icustay_id
                 AND v.amountuom = 'mL'
                 AND v.starttime BETWEEN i.intime AND i.intime + INTERVAL 24 HOUR
                GROUP BY i.icustay_id
            ),
            output AS (
                SELECT
                    i.icustay_id,
                    SUM(o.value) AS total_out,
                    COUNT(o.value) AS out_count
                FROM {icustays_relation} i
                LEFT JOIN {outputevents_relation} o
                  ON i.icustay_id = o.icustay_id
                 AND o.charttime BETWEEN i.intime AND i.intime + INTERVAL 24 HOUR
                GROUP BY i.icustay_id
            )
            SELECT
                i.subject_id,
                i.icustay_id,
                COALESCE(intake.total_in, 0) - COALESCE(output.total_out, 0)
                    AS icu_fluid_balance_24h_ml,
                COALESCE(intake.in_count, 0) + COALESCE(output.out_count, 0)
                    AS icu_fluid_event_count
            FROM {icustays_relation} i
            LEFT JOIN intake USING (icustay_id)
            LEFT JOIN output USING (icustay_id)
        """
    else:
        icu_fluid_query = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS icustay_id,
                NULL::DOUBLE AS icu_fluid_balance_24h_ml,
                0::BIGINT AS icu_fluid_event_count
            WHERE FALSE
        """
    icu_fluid_balance_count = _copy_query_atomic(
        connection, icu_fluid_query, icu_fluid_balance
    )

    icu_gcs_admission = config.evidence_root / "icu_gcs_admission.parquet"
    if chartevents_relation is not None and icustays_relation is not None:
        gcs_ids = ", ".join(
            str(itemid)
            for key in ("gcs_eye", "gcs_verbal", "gcs_motor")
            for itemid in ICU_CHARTEVENTS_ITEMIDS[key]
        )
        icu_gcs_query = f"""
            WITH nearest AS (
                SELECT
                    i.subject_id,
                    i.icustay_id,
                    c.itemid,
                    c.valuenum,
                    ROW_NUMBER() OVER (
                        PARTITION BY i.icustay_id, c.itemid
                        ORDER BY
                            ABS(DATE_DIFF('second', i.intime, c.charttime)),
                            c.charttime,
                            c.valuenum
                    ) AS rn
                FROM {icustays_relation} i
                JOIN {chartevents_relation} c
                  USING (icustay_id)
                WHERE c.itemid IN ({gcs_ids})
                  AND c.valuenum IS NOT NULL
                  AND c.charttime BETWEEN i.intime - INTERVAL 6 HOUR
                                       AND i.intime + INTERVAL 24 HOUR
            )
            SELECT
                subject_id,
                icustay_id,
                SUM(valuenum) AS icu_gcs_admission_total,
                COUNT(*) AS icu_gcs_component_count
            FROM nearest
            WHERE rn = 1
            GROUP BY subject_id, icustay_id
        """
    else:
        icu_gcs_query = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS icustay_id,
                NULL::DOUBLE AS icu_gcs_admission_total,
                0::BIGINT AS icu_gcs_component_count
            WHERE FALSE
        """
    icu_gcs_admission_count = _copy_query_atomic(
        connection, icu_gcs_query, icu_gcs_admission
    )

    # Audit 5.2.9: invasive haemodynamics as surrogate references. Per ECG
    # the chart value closest in time within 6 hours (the lab window), one
    # per quantity; ties to the earlier chart time, then the lower value.
    icu_hemodynamics = config.evidence_root / "icu_hemodynamics_near_ecg.parquet"
    icu_hemodynamics_count = _copy_query_atomic(
        connection,
        _icu_hemodynamics_query(
            _sql_path(config.master_root / "ecg_master_index.parquet"),
            chartevents_relation,
        ),
        icu_hemodynamics,
    )

    if progress:
        progress("Verknuepfe ausschliesslich praee ECG-OMR-Kovariaten")
    omr_features = config.evidence_root / "ecg_omr_features.parquet"
    omr_path = config.standardized_root / "omr.parquet"
    if omr_path.exists():
        omr_query = f"""
            WITH normalized AS (
                SELECT
                    subject_id,
                    chartdate,
                    LOWER(TRIM(result_name)) AS result_name,
                    result_value,
                    TRY_CAST(
                        REGEXP_EXTRACT(
                            result_value,
                            '[-+]?[0-9]*\\.?[0-9]+',
                            0
                        ) AS DOUBLE
                    ) AS first_numeric,
                    TRY_CAST(
                        REGEXP_EXTRACT(result_value, '/\\s*([0-9]+)', 1)
                        AS DOUBLE
                    ) AS second_numeric
                FROM read_parquet('{_sql_path(omr_path)}')
            ),
            candidates AS (
                SELECT
                    m.subject_id,
                    m.study_id,
                    o.* EXCLUDE(subject_id),
                    DATE_DIFF(
                        'day',
                        o.chartdate,
                        CAST(m.ecg_time AS DATE)
                    ) AS delta_days
                FROM read_parquet('{master}') m
                JOIN normalized o USING (subject_id)
                WHERE o.chartdate <= CAST(m.ecg_time AS DATE)
                  AND o.chartdate >=
                      CAST(m.ecg_time AS DATE) - INTERVAL 365 DAY
            )
            SELECT
                subject_id,
                study_id,
                ARG_MIN(first_numeric * 0.45359237, delta_days) FILTER (
                    WHERE result_name = 'weight (lbs)'
                ) AS weight_kg_before_ecg,
                ARG_MIN(first_numeric * 2.54, delta_days) FILTER (
                    WHERE result_name = 'height (inches)'
                ) AS height_cm_before_ecg,
                ARG_MIN(first_numeric, delta_days) FILTER (
                    WHERE result_name IN ('bmi', 'bmi (kg/m2)')
                ) AS bmi_before_ecg,
                ARG_MIN(first_numeric, delta_days) FILTER (
                    WHERE result_name LIKE 'blood pressure%'
                ) AS systolic_bp_before_ecg,
                ARG_MIN(second_numeric, delta_days) FILTER (
                    WHERE result_name LIKE 'blood pressure%'
                ) AS diastolic_bp_before_ecg,
                MIN(delta_days) AS nearest_omr_delta_days,
                'date' AS omr_time_precision
            FROM candidates
            GROUP BY subject_id, study_id
        """
    else:
        omr_query = """
            SELECT
                NULL::BIGINT AS subject_id,
                NULL::BIGINT AS study_id,
                NULL::DOUBLE AS weight_kg_before_ecg,
                NULL::DOUBLE AS height_cm_before_ecg,
                NULL::DOUBLE AS bmi_before_ecg,
                NULL::DOUBLE AS systolic_bp_before_ecg,
                NULL::DOUBLE AS diastolic_bp_before_ecg,
                NULL::BIGINT AS nearest_omr_delta_days,
                NULL::VARCHAR AS omr_time_precision
            WHERE FALSE
        """
    omr_feature_count = _copy_query_atomic(
        connection,
        omr_query,
        omr_features,
    )

    lab_pivot = config.evidence_root / "ecg_lab_features.parquet"
    # Generated from LAB_ITEM_GROUPS so a new lab group cannot be defined,
    # windowed and then silently dropped before the feature table -- which is
    # exactly what happened to ck_mb and blood_gas_ph. potassium keeps its
    # hand-written multi-window block above and is excluded here.
    generated_lab_window_pivot = _lab_window_pivot_sql(
        LAB_ITEM_GROUPS,
        exclude=("potassium", "troponin_i", "troponin_t"),
    )

    def _lab_pivot_query(partition_index: int, partition_count: int) -> str:
        return _lab_feature_pivot_query(
            labs,
            generated_lab_window_pivot,
            partition_index=partition_index,
            partition_count=partition_count,
        )

    lab_pivot_count = _copy_partitioned_query_atomic(
        connection,
        _lab_pivot_query,
        lab_pivot,
        # 208 conditional aggregates over ~800k ECGs. Measured at full size:
        # six threads unbucketed peaked at 10.7 GiB, past this stage's 8 GB
        # allocation; two threads over four buckets peak at 3.7 GiB and finish
        # in 273 s. Neither number moves with row count -- both follow the
        # aggregate list -- so the settings are pinned rather than derived
        # from config.workers.
        # One thread (2026-09-25, same bucket, real data): Windows commit
        # 7.8 GiB instead of 14.4 GiB, resident 3.2 instead of 6.0 GiB, 206 s
        # instead of 201-302 s, identical rows. The aggregate states live
        # outside DuckDB's memory_limit (a 4 GB limit left the commit at
        # 14.4 GiB), so only the thread count bounds them; at two threads the
        # commit spike tripped the supervisor's memory pause and, because
        # buckets do not survive a restart, looped.
        initial_partitions=4,
        threads=1,
        progress=progress,
        label="Verdichte Laborfenster zu ECG-Merkmalen",
    )

    medication_pivot = config.evidence_root / "ecg_medication_features.parquet"
    medication_pivot_query = f"""
        SELECT
            subject_id,
            study_id,
            {', '.join(
                (
                    "BOOL_OR((administered_near_ecg OR active_order_at_ecg "
                    "OR ed_dispensed_near_ecg) "
                    f"AND medication_class = '{group}') AS med_{group}"
                )
                for group in MEDICATION_PATTERNS
            )},
            {', '.join(
                (
                    "BOOL_OR(home_medication_reported_near_ecg "
                    f"AND medication_class = '{group}') "
                    f"AS med_{group}_home_reported"
                )
                for group in MEDICATION_PATTERNS
            )}
        FROM read_parquet('{medications}')
        GROUP BY subject_id, study_id
    """
    medication_pivot_count = _copy_query_atomic(
        connection, medication_pivot_query, medication_pivot
    )

    if progress:
        progress("Erzeuge AF-Follow-up- und Vorerkrankungsachsen")
    patient_timeline = config.evidence_root / "patient_clinical_timeline.parquet"
    edstays_path = config.standardized_root / "edstays.parquet"
    ed_contacts_sql = (
        "SELECT subject_id, MAX(TRY_CAST(outtime AS TIMESTAMP)) "
        "AS last_ed_contact_time "
        f"FROM read_parquet('{_sql_path(edstays_path)}') GROUP BY subject_id"
        if edstays_path.is_file()
        else "SELECT NULL::BIGINT AS subject_id, "
        "NULL::TIMESTAMP AS last_ed_contact_time WHERE FALSE"
    )
    timeline_query = f"""
        WITH af_dx AS (
            SELECT
                d.subject_id,
                MIN(a.admittime) AS first_af_dx_time,
                ARG_MIN(
                    d.hadm_id,
                    STRUCT_PACK(admittime := a.admittime, hadm_id := d.hadm_id)
                ) AS first_af_dx_hadm_id
            FROM read_parquet('{diagnoses}') d
            JOIN read_parquet('{admissions}') a
              USING (subject_id, hadm_id)
            WHERE d.code_group = 'atrial_fibrillation_flutter'
            GROUP BY d.subject_id
        ),
        af_report AS (
            SELECT
                m.subject_id,
                MIN(m.ecg_time) FILTER (
                    WHERE r.report_af_state = 'positive'
                ) AS first_af_report_time,
                ARG_MIN(
                    m.study_id,
                    STRUCT_PACK(ecg_time := m.ecg_time, study_id := m.study_id)
                ) FILTER (
                    WHERE r.report_af_state = 'positive'
                ) AS first_af_report_study_id,
                MAX(m.ecg_time) AS last_ecg_time
            FROM read_parquet('{master}') m
            JOIN read_parquet('{report_labels}') r
              USING (subject_id, study_id)
            GROUP BY m.subject_id
        ),
        encounters AS (
            SELECT
                subject_id,
                MIN(admittime) AS first_encounter_time,
                MAX(dischtime) AS last_encounter_time
            FROM read_parquet('{admissions}')
            GROUP BY subject_id
        ),
        ed_contacts AS (
            {ed_contacts_sql}
        )
        SELECT
            p.subject_id,
            LEAST(d.first_af_dx_time, r.first_af_report_time)
                AS first_af_event_time,
            d.first_af_dx_time,
            d.first_af_dx_hadm_id,
            r.first_af_report_time,
            r.first_af_report_study_id,
            CASE
                WHEN d.first_af_dx_time IS NULL
                 AND r.first_af_report_time IS NULL THEN NULL
                WHEN r.first_af_report_time IS NOT NULL
                 AND (
                        d.first_af_dx_time IS NULL
                     OR r.first_af_report_time <= d.first_af_dx_time
                 ) THEN 'machine_report'
                ELSE 'icd_admission'
            END AS first_af_event_source,
            e.first_encounter_time,
            GREATEST(r.last_ecg_time, e.last_encounter_time)
                AS last_observed_time,
            e.last_encounter_time AS last_hospital_discharge_time,
            ed.last_ed_contact_time,
            -- Audit D4/T1: MIMIC-IV links state death records up to exactly
            -- one year after the last hospital discharge. A patient without
            -- a date of death is therefore known alive until then, and at
            -- every later ECG or ED contact. Patients never admitted keep
            -- only their contacts: registry coverage is not documented for
            -- them.
            GREATEST(
                r.last_ecg_time,
                ed.last_ed_contact_time,
                e.last_encounter_time + INTERVAL 365 DAY
            ) AS vital_status_known_until,
            p.date_of_death
        FROM read_parquet('{_sql_path(config.standardized_root / "patients.parquet")}') p
        LEFT JOIN af_dx d USING (subject_id)
        LEFT JOIN af_report r USING (subject_id)
        LEFT JOIN encounters e USING (subject_id)
        LEFT JOIN ed_contacts ed USING (subject_id)
    """
    timeline_count = _copy_query_atomic(connection, timeline_query, patient_timeline)

    if progress:
        progress("Berechne 30-Tage-Wiederaufnahme je Aufenthalt")
    admission_readmission = config.evidence_root / "admission_readmission.parquet"
    admission_readmission_query = f"""
        WITH ranked AS (
            SELECT
                subject_id,
                hadm_id,
                dischtime,
                LEAD(admittime) OVER (
                    PARTITION BY subject_id ORDER BY admittime, hadm_id
                ) AS next_admittime
            FROM read_parquet('{admissions}')
        )
        SELECT
            subject_id,
            hadm_id,
            (
                next_admittime IS NOT NULL
                AND DATE_DIFF('day', dischtime, next_admittime) BETWEEN 0 AND 30
            ) AS readmission_30d_flag
        FROM ranked
    """
    admission_readmission_count = _copy_query_atomic(
        connection, admission_readmission_query, admission_readmission
    )

    if progress:
        progress("Baue klinische ECG-Kontextmatrix")
    output = config.evidence_root / "clinical_context_features.parquet"
    query = _clinical_context_query(
        config,
        diagnosis_flags,
        procedure_flags,
        lab_pivot,
        troponin,
        medication_pivot,
        echo_matches,
        echo_wide,
        patient_timeline,
        patient_history,
        encounter_microbiology,
        omr_features,
        Path(report_labels),
        note_encounter_flags,
        medication_coverage,
        admission_readmission,
        encounter_care_setting,
        ed_vitals_trend,
        icu_vitals_trend,
        icu_ventilation_status,
        icu_vasopressor_exposure,
        icu_fluid_balance,
        icu_gcs_admission,
        icu_hemodynamics=icu_hemodynamics,
    )
    context_count = _copy_query_atomic(connection, query, output)
    return [
        EvidenceArtifact(
            "encounter_diagnosis_flags", diagnosis_flags, diagnosis_flag_count
        ),
        EvidenceArtifact(
            "encounter_care_setting_flags",
            encounter_care_setting,
            encounter_care_setting_count,
        ),
        EvidenceArtifact(
            "ed_vitals_trend", ed_vitals_trend, ed_vitals_trend_count
        ),
        EvidenceArtifact(
            "icu_vitals_trend", icu_vitals_trend, icu_vitals_trend_count
        ),
        EvidenceArtifact(
            "icu_ventilation_status",
            icu_ventilation_status,
            icu_ventilation_status_count,
        ),
        EvidenceArtifact(
            "icu_vasopressor_exposure",
            icu_vasopressor_exposure,
            icu_vasopressor_exposure_count,
        ),
        EvidenceArtifact(
            "icu_fluid_balance_24h", icu_fluid_balance, icu_fluid_balance_count
        ),
        EvidenceArtifact(
            "icu_gcs_admission", icu_gcs_admission, icu_gcs_admission_count
        ),
        EvidenceArtifact(
            "icu_hemodynamics_near_ecg",
            icu_hemodynamics,
            icu_hemodynamics_count,
        ),
        EvidenceArtifact(
            "admission_readmission",
            admission_readmission,
            admission_readmission_count,
        ),
        EvidenceArtifact(
            "encounter_procedure_flags", procedure_flags, procedure_flag_count
        ),
        EvidenceArtifact("ecg_lab_features", lab_pivot, lab_pivot_count),
        EvidenceArtifact(
            "ecg_medication_features", medication_pivot, medication_pivot_count
        ),
        EvidenceArtifact(
            "patient_clinical_timeline", patient_timeline, timeline_count
        ),
        EvidenceArtifact(
            "patient_code_history", patient_history, patient_history_count
        ),
        EvidenceArtifact(
            "encounter_microbiology_flags",
            encounter_microbiology,
            encounter_microbiology_count,
        ),
        EvidenceArtifact("ecg_omr_features", omr_features, omr_feature_count),
        EvidenceArtifact("clinical_context_features", output, context_count),
    ]


def _clinical_context_query(
    config: MimicProjectConfig,
    diagnosis_flags: Path,
    procedure_flags: Path,
    lab_features: Path,
    troponin_serial: str,
    medication_features: Path,
    echo_matches: str,
    echo_wide: str,
    patient_timeline: Path,
    patient_history: Path,
    encounter_microbiology: Path,
    omr_features: Path,
    report_labels: Path,
    note_encounter_flags: Path,
    medication_coverage: Path,
    admission_readmission: Path,
    encounter_care_setting: Path,
    ed_vitals_trend: Path,
    icu_vitals_trend: Path,
    icu_ventilation_status: Path,
    icu_vasopressor_exposure: Path,
    icu_fluid_balance: Path,
    icu_gcs_admission: Path,
    *,
    icu_hemodynamics: Optional[Path] = None,
) -> str:
    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    af_diagnoses = _sql_path(config.evidence_root / "diagnosis_evidence.parquet")
    af_admissions = _sql_path(config.standardized_root / "admissions.parquet")
    conduction_procedures = _sql_path(
        config.evidence_root / "procedure_evidence.parquet"
    )
    dx = _sql_path(diagnosis_flags)
    proc = _sql_path(procedure_flags)
    lab = _sql_path(lab_features)
    trop = troponin_serial
    meds = _sql_path(medication_features)
    echo_match = echo_matches
    echo = echo_wide
    timeline = _sql_path(patient_timeline)
    history = _sql_path(patient_history)
    microbiology = _sql_path(encounter_microbiology)
    omr = _sql_path(omr_features)
    reports = _sql_path(report_labels)
    notes = _sql_path(note_encounter_flags)
    medication_coverage_sql = _sql_path(medication_coverage)
    readmit = _sql_path(admission_readmission)
    care_setting = _sql_path(encounter_care_setting)
    vitals_trend = _sql_path(ed_vitals_trend)
    icu_vitals = _sql_path(icu_vitals_trend)
    icu_vent = _sql_path(icu_ventilation_status)
    icu_vaso = _sql_path(icu_vasopressor_exposure)
    icu_fluid = _sql_path(icu_fluid_balance)
    icu_gcs = _sql_path(icu_gcs_admission)
    hemodynamics_relation = (
        f"read_parquet('{_sql_path(icu_hemodynamics)}')"
        if icu_hemodynamics is not None
        else f"({_icu_hemodynamics_query(master, None)})"
    )
    return f"""
        SELECT
            m.*,
            l.* EXCLUDE(subject_id, study_id),
            t.acute_change_proxy AS acute_troponin_change_proxy_value,
            t.serial_assessment_available AS troponin_serial_assessed,
            t.undetectable_rise_state AS troponin_undetectable_rise_state,
            t.max_relative_delta AS troponin_max_relative_delta,
            t.qualifying_first_labevent_id,
            t.qualifying_last_labevent_id,
            DATE_DIFF(
                'second',
                m.ecg_time,
                COALESCE(t.acute_change_event_time, t.first_troponin_time)
            ) / 3600.0
                AS troponin_event_delta_hours,
            d.* EXCLUDE(subject_id, hadm_id),
            p.* EXCLUDE(subject_id, hadm_id),
            med.* EXCLUDE(subject_id, study_id),
            COALESCE(mc.medication_assessed, FALSE) AS medication_assessed,
            mc.* EXCLUDE(subject_id, study_id, medication_assessed),
            em.* EXCLUDE(subject_id, study_id),
            ew.* EXCLUDE(subject_id, measurement_id, echo_time, test_type),
            pt.first_af_event_time,
            pt.first_af_dx_time,
            pt.first_af_dx_hadm_id,
            pt.first_af_report_time,
            pt.first_af_report_study_id,
            pt.first_af_event_source,
            -- Audit D9: AF timing per ECG. The subject-level first AF time is
            -- the admission time of the first AF-coded admission, so AF that
            -- began after the ECG in the same admission (post-operative AF,
            -- the commonest new onset) read as prior AF. The index
            -- admission's code alone is now "timing unknown", and a later
            -- timed event (first AF report after the ECG, or a later
            -- AF-coded admission) decides.
            COALESCE(afx.af_dx_prior_admission, FALSE) AS af_dx_prior_admission,
            COALESCE(afx.af_dx_index_admission, FALSE) AS af_dx_index_admission,
            LEAST(
                CASE WHEN pt.first_af_report_time > m.ecg_time
                     THEN pt.first_af_report_time END,
                afx.next_af_dx_admission_time
            ) AS next_af_event_time,
            CASE
                WHEN pt.first_af_report_time > m.ecg_time
                 AND (afx.next_af_dx_admission_time IS NULL
                      OR pt.first_af_report_time <= afx.next_af_dx_admission_time)
                    THEN 'machine_report'
                WHEN afx.next_af_dx_admission_time IS NOT NULL
                    THEN 'icd_admission'
            END AS next_af_event_source,
            DATE_DIFF(
                'second',
                m.ecg_time,
                LEAST(
                    CASE WHEN pt.first_af_report_time > m.ecg_time
                         THEN pt.first_af_report_time END,
                    afx.next_af_dx_admission_time
                )
            ) / 3600.0 AS next_af_event_delta_hours,
            afx.next_af_dx_hadm_id,
            pt.first_encounter_time,
            pt.last_observed_time,
            pt.date_of_death,
            r.report_af_state AS current_report_af_state,
            -- Joined for the same reason the AF state is: an endpoint
            -- whose positives come from structured records still needs
            -- a way to keep paced tracings out of its controls. Used
            -- only to withhold a negative, never to assert a positive.
            r.report_pacemaker_state AS current_report_pacemaker_state,
            r.report_lbbb_state AS current_report_lbbb_state,
            -- Audit 5.2.5: conduction progression after the ECG. Prior
            -- evidence: an earlier LBBB/high-grade AV-block admission, a
            -- device-in-situ code up to the index admission, or a pacemaker
            -- implanted before the ECG date. Index-admission codes and a
            -- pacemaker on the ECG date carry no usable time.
            COALESCE(cpx.conduction_prior_evidence, FALSE)
                AS conduction_prior_evidence,
            COALESCE(cpx.conduction_index_untimed, FALSE)
                AS conduction_index_untimed,
            LEAST(cpx.next_conduction_dx_admission_time, cpx.next_pacemaker_date)
                AS next_conduction_event_time,
            CASE
                WHEN cpx.next_pacemaker_date IS NOT NULL
                 AND (cpx.next_conduction_dx_admission_time IS NULL
                      OR cpx.next_pacemaker_date <= cpx.next_conduction_dx_admission_time)
                    THEN 'pacemaker_procedure'
                WHEN cpx.next_conduction_dx_admission_time IS NOT NULL
                    THEN 'icd_admission'
            END AS next_conduction_event_source,
            DATE_DIFF(
                'second',
                m.ecg_time,
                LEAST(cpx.next_conduction_dx_admission_time, cpx.next_pacemaker_date)
            ) / 3600.0 AS next_conduction_event_delta_hours,
            DATE_DIFF(
                'second', m.ecg_time, cpx.next_device_status_admission_time
            ) / 3600.0 AS next_device_status_delta_hours,
            n.* EXCLUDE(subject_id, hadm_id),
            COALESCE(micro.typical_endocarditis_organism, FALSE)
                AS typical_endocarditis_organism,
            COALESCE(micro.microbiology_specimen_count, 0)
                AS microbiology_specimen_count,
            COALESCE(micro.any_positive_culture, FALSE)
                AS any_positive_culture,
            cs.ever_icu_this_admission,
            cs.primary_service_this_admission,
            vt.* EXCLUDE(subject_id, ed_stay_id),
            icuv.* EXCLUDE(subject_id, icustay_id),
            icuvent.* EXCLUDE(subject_id, icustay_id),
            icuvaso.* EXCLUDE(subject_id, icustay_id),
            icufl.* EXCLUDE(subject_id, icustay_id),
            icugcs.* EXCLUDE(subject_id, icustay_id),
            icuh.* EXCLUDE(subject_id, study_id),
            o.* EXCLUDE(subject_id, study_id),
            CASE
                WHEN h.first_old_mi_time IS NOT NULL
                 AND h.first_old_mi_time <= m.ecg_time THEN TRUE
                ELSE FALSE
            END AS history_old_mi_before_ecg,
            CASE
                WHEN (
                    h.first_old_mi_time IS NOT NULL
                    AND h.first_old_mi_time <= m.ecg_time
                ) OR (
                    h.first_completed_acute_mi_time IS NOT NULL
                    AND h.first_completed_acute_mi_time < m.ecg_time
                ) OR (
                    h.first_pci_date IS NOT NULL
                    AND h.first_pci_date < CAST(m.ecg_time AS DATE)
                ) OR (
                    h.first_cabg_date IS NOT NULL
                    AND h.first_cabg_date < CAST(m.ecg_time AS DATE)
                ) THEN TRUE
                ELSE FALSE
            END AS history_mi_or_revascularization_before_ecg,
            CASE
                WHEN m.machine_qt_interval_ms IS NOT NULL
                 AND m.machine_heart_rate_bpm BETWEEN 20 AND 300
                THEN m.machine_qt_interval_ms
                    / POWER(60.0 / m.machine_heart_rate_bpm, 1.0 / 3.0)
            END AS machine_qtc_fridericia_ms,
            CASE
                WHEN m.machine_qt_interval_ms IS NOT NULL
                 AND m.machine_heart_rate_bpm BETWEEN 20 AND 300
                THEN m.machine_qt_interval_ms
                    / SQRT(60.0 / m.machine_heart_rate_bpm)
            END AS machine_qtc_bazett_ms,
            CASE
                WHEN ew.posterior_wall_thickness_cm IS NOT NULL
                 AND ew.lv_internal_diameter_diastole_cm > 0
                THEN 2.0 * ew.posterior_wall_thickness_cm
                     / ew.lv_internal_diameter_diastole_cm
            END AS relative_wall_thickness,
            {_devereux_context_columns_sql()},
            CASE
                WHEN ew.direct_estimated_pa_pressure_mmhg IS NOT NULL
                    THEN ew.direct_estimated_pa_pressure_mmhg
                WHEN ew.tr_pressure_gradient_mmhg IS NOT NULL
                 AND ew.estimated_ra_pressure_mmhg IS NOT NULL
                    THEN ew.tr_pressure_gradient_mmhg
                         + ew.estimated_ra_pressure_mmhg
                WHEN ew.tr_velocity_m_per_s IS NOT NULL
                 AND ew.estimated_ra_pressure_mmhg IS NOT NULL
                    THEN 4.0 * POWER(ew.tr_velocity_m_per_s, 2)
                         + ew.estimated_ra_pressure_mmhg
            END AS derived_rvsp_or_pasp_mmhg,
            CASE
                WHEN ew.direct_estimated_pa_pressure_mmhg IS NOT NULL
                    THEN 'direct_estimated_pa_pressure'
                WHEN ew.tr_pressure_gradient_mmhg IS NOT NULL
                 AND ew.estimated_ra_pressure_mmhg IS NOT NULL
                    THEN 'tr_gradient_plus_ivc_rap'
                WHEN ew.tr_velocity_m_per_s IS NOT NULL
                 AND ew.estimated_ra_pressure_mmhg IS NOT NULL
                    THEN 'bernoulli_trv_plus_ivc_rap'
            END AS rvsp_or_pasp_method,
            DATE_DIFF('day', m.ecg_time, p.first_pci_date) AS pci_day_delta,
            DATE_DIFF('day', m.ecg_time, p.first_cath_date) AS cath_day_delta,
            DATE_DIFF('hour', m.ecg_time, pt.first_af_event_time)
                AS first_af_event_delta_hours,
            DATE_DIFF('hour', m.ecg_time, pt.last_observed_time)
                AS followup_hours,
            DATE_DIFF('hour', m.ecg_time, pt.vital_status_known_until)
                AS vital_status_followup_hours,
            DATE_DIFF('day', m.ecg_time, pt.date_of_death)
                AS death_delta_days,
            DATE_DIFF('day', m.admittime, m.dischtime) AS length_of_stay_days,
            COALESCE(rd.readmission_30d_flag, FALSE) AS readmission_30d_flag
        FROM read_parquet('{master}') m
        LEFT JOIN read_parquet('{lab}') l USING (subject_id, study_id)
        LEFT JOIN read_parquet('{trop}') t USING (subject_id, study_id)
        LEFT JOIN read_parquet('{dx}') d USING (subject_id, hadm_id)
        LEFT JOIN read_parquet('{proc}') p USING (subject_id, hadm_id)
        LEFT JOIN read_parquet('{meds}') med USING (subject_id, study_id)
        LEFT JOIN read_parquet('{medication_coverage_sql}') mc
          USING (subject_id, study_id)
        LEFT JOIN read_parquet('{echo_match}') em USING (subject_id, study_id)
        LEFT JOIN read_parquet('{echo}') ew
          ON em.nearest_tte_measurement_id = ew.measurement_id
        LEFT JOIN read_parquet('{timeline}') pt USING (subject_id)
        LEFT JOIN (
            SELECT
                mx.subject_id,
                mx.study_id,
                BOOL_OR(
                    af.admittime <= mx.ecg_time
                    AND NOT (
                        af.hadm_id = mx.hadm_id
                        OR (mx.hadm_id IS NULL
                            AND mx.ecg_time BETWEEN af.admittime AND af.dischtime)
                    )
                ) AS af_dx_prior_admission,
                BOOL_OR(
                    af.hadm_id = mx.hadm_id
                    OR (mx.hadm_id IS NULL
                        AND mx.ecg_time BETWEEN af.admittime AND af.dischtime)
                ) AS af_dx_index_admission,
                MIN(af.admittime) FILTER (
                    WHERE af.admittime > mx.ecg_time
                ) AS next_af_dx_admission_time,
                ARG_MIN(
                    af.hadm_id,
                    STRUCT_PACK(admittime := af.admittime, hadm_id := af.hadm_id)
                ) FILTER (
                    WHERE af.admittime > mx.ecg_time
                ) AS next_af_dx_hadm_id
            FROM read_parquet('{master}') mx
            JOIN (
                SELECT DISTINCT d.subject_id, d.hadm_id, a.admittime, a.dischtime
                FROM read_parquet('{af_diagnoses}') d
                JOIN read_parquet('{af_admissions}') a
                  USING (subject_id, hadm_id)
                WHERE d.code_group = 'atrial_fibrillation_flutter'
            ) af USING (subject_id)
            GROUP BY mx.subject_id, mx.study_id
        ) afx USING (subject_id, study_id)
        LEFT JOIN (
            {_conduction_timing_query(master, af_diagnoses, af_admissions, conduction_procedures)}
        ) cpx USING (subject_id, study_id)
        LEFT JOIN read_parquet('{history}') h USING (subject_id)
        LEFT JOIN read_parquet('{microbiology}') micro
          USING (subject_id, hadm_id)
        LEFT JOIN read_parquet('{care_setting}') cs USING (subject_id, hadm_id)
        LEFT JOIN read_parquet('{vitals_trend}') vt
          USING (subject_id, ed_stay_id)
        LEFT JOIN read_parquet('{icu_vitals}') icuv
          USING (subject_id, icustay_id)
        LEFT JOIN read_parquet('{icu_vent}') icuvent
          USING (subject_id, icustay_id)
        LEFT JOIN read_parquet('{icu_vaso}') icuvaso
          USING (subject_id, icustay_id)
        LEFT JOIN read_parquet('{icu_fluid}') icufl
          USING (subject_id, icustay_id)
        LEFT JOIN read_parquet('{icu_gcs}') icugcs
          USING (subject_id, icustay_id)
        LEFT JOIN {hemodynamics_relation} icuh
          USING (subject_id, study_id)
        LEFT JOIN read_parquet('{omr}') o USING (subject_id, study_id)
        LEFT JOIN read_parquet('{reports}') r USING (subject_id, study_id)
        LEFT JOIN read_parquet('{notes}') n USING (subject_id, hadm_id)
        LEFT JOIN read_parquet('{readmit}') rd USING (subject_id, hadm_id)
    """


def _icu_hemodynamics_query(master: str, chartevents_relation: Optional[str]) -> str:
    """Nearest invasive haemodynamic chart value per ECG within 6 hours.

    Columns ``icu_<quantity>_nearest`` and ``icu_<quantity>_delta_hours``
    (chart time minus ECG time) for every quantity in
    ``ICU_HEMODYNAMIC_ITEMIDS``. Without chartevents the query returns no
    rows with the same typed columns, so the context keeps its schema.
    """
    from .mappings import ICU_HEMODYNAMIC_ITEMIDS, ICU_HEMODYNAMIC_PLAUSIBLE_RANGES

    names = tuple(ICU_HEMODYNAMIC_ITEMIDS)
    if chartevents_relation is None:
        columns = ", ".join(
            f"NULL::DOUBLE AS icu_{name}_nearest, NULL::DOUBLE AS icu_{name}_delta_hours"
            for name in names
        )
        return f"SELECT NULL::BIGINT AS subject_id, NULL::BIGINT AS study_id, {columns} WHERE FALSE"
    quantity_case = " ".join(
        f"WHEN c.itemid IN ({', '.join(str(i) for i in ids)}) THEN '{name}'"
        for name, ids in ICU_HEMODYNAMIC_ITEMIDS.items()
    )
    plausible = " OR ".join(
        f"(quantity = '{name}' AND valuenum BETWEEN {low} AND {high})"
        for name, (low, high) in ICU_HEMODYNAMIC_PLAUSIBLE_RANGES.items()
    )
    all_ids = ", ".join(
        str(itemid)
        for ids in ICU_HEMODYNAMIC_ITEMIDS.values()
        for itemid in ids
    )
    pivot = ",\n            ".join(
        f"MAX(valuenum) FILTER (WHERE quantity = '{name}') AS icu_{name}_nearest,\n"
        f"            MAX(delta_hours) FILTER (WHERE quantity = '{name}') AS icu_{name}_delta_hours"
        for name in names
    )
    return f"""
        WITH charted AS (
            SELECT
                c.subject_id,
                c.charttime,
                c.valuenum,
                CASE {quantity_case} END AS quantity
            FROM {chartevents_relation} c
            WHERE c.itemid IN ({all_ids})
              AND c.valuenum IS NOT NULL
        ),
        plausible AS (
            SELECT * FROM charted WHERE {plausible}
        ),
        candidates AS (
            SELECT
                m.subject_id,
                m.study_id,
                p.quantity,
                p.valuenum,
                DATE_DIFF('second', m.ecg_time, p.charttime) / 3600.0
                    AS delta_hours,
                ROW_NUMBER() OVER (
                    PARTITION BY m.subject_id, m.study_id, p.quantity
                    ORDER BY
                        ABS(DATE_DIFF('second', m.ecg_time, p.charttime)),
                        p.charttime,
                        p.valuenum
                ) AS nearest_rank
            FROM read_parquet('{master}') m
            JOIN plausible p
              ON p.subject_id = m.subject_id
             AND p.charttime BETWEEN m.ecg_time - INTERVAL 6 HOUR
                                 AND m.ecg_time + INTERVAL 6 HOUR
            WHERE m.ecg_time IS NOT NULL
        )
        SELECT
            subject_id,
            study_id,
            {pivot}
        FROM candidates
        WHERE nearest_rank = 1
        GROUP BY subject_id, study_id
    """


def _conduction_timing_query(
    master: str, diagnoses: str, admissions: str, procedures: str
) -> str:
    """Audit 5.2.5: per ECG, prior and next conduction-progression evidence.

    Prior: an earlier admission coded LBBB or high-grade AV block, a
    device-in-situ code in an admission up to the ECG, or a pacemaker
    implanted before the ECG date. Untimed: codes of the index admission and
    a pacemaker on the ECG date. Next: the admission time of a later coded
    admission and the date of a later pacemaker implantation.
    """
    return f"""
        SELECT
            mx.subject_id,
            mx.study_id,
            BOOL_OR(
                (ev.kind = 'dx' AND ev.event_time <= mx.ecg_time
                 AND NOT (
                     ev.hadm_id = mx.hadm_id
                     OR (mx.hadm_id IS NULL
                         AND mx.ecg_time BETWEEN ev.event_time AND ev.dischtime)
                 ))
                OR (ev.kind = 'device_status' AND ev.event_time <= mx.ecg_time)
                OR (ev.kind = 'pacemaker'
                    AND CAST(ev.event_time AS DATE) < CAST(mx.ecg_time AS DATE))
            ) AS conduction_prior_evidence,
            BOOL_OR(
                (ev.kind = 'dx' AND (
                    ev.hadm_id = mx.hadm_id
                    OR (mx.hadm_id IS NULL
                        AND mx.ecg_time BETWEEN ev.event_time AND ev.dischtime)
                ))
                OR (ev.kind = 'pacemaker'
                    AND CAST(ev.event_time AS DATE) = CAST(mx.ecg_time AS DATE))
            ) AS conduction_index_untimed,
            MIN(ev.event_time) FILTER (
                WHERE ev.kind = 'dx' AND ev.event_time > mx.ecg_time
            ) AS next_conduction_dx_admission_time,
            MIN(ev.event_time) FILTER (
                WHERE ev.kind = 'pacemaker'
                  AND CAST(ev.event_time AS DATE) > CAST(mx.ecg_time AS DATE)
            ) AS next_pacemaker_date,
            -- A device-in-situ code in a later admission without a timed
            -- implantation: a device arrived at an unknown time (possibly
            -- elsewhere). Such a patient is no control.
            MIN(ev.event_time) FILTER (
                WHERE ev.kind = 'device_status' AND ev.event_time > mx.ecg_time
            ) AS next_device_status_admission_time
        FROM read_parquet('{master}') mx
        JOIN (
            SELECT DISTINCT
                d.subject_id,
                d.hadm_id,
                CASE WHEN d.code_group = 'cardiac_device_status'
                     THEN 'device_status' ELSE 'dx' END AS kind,
                a.admittime AS event_time,
                a.dischtime
            FROM read_parquet('{diagnoses}') d
            JOIN read_parquet('{admissions}') a
              USING (subject_id, hadm_id)
            WHERE d.code_group IN (
                'advanced_av_block', 'left_bundle_branch_block',
                'cardiac_device_status'
            )
            UNION ALL
            SELECT DISTINCT
                p.subject_id,
                p.hadm_id,
                'pacemaker' AS kind,
                CAST(p.chartdate AS TIMESTAMP) AS event_time,
                NULL::TIMESTAMP AS dischtime
            FROM read_parquet('{procedures}') p
            WHERE p.code_group = 'permanent_pacemaker_implant'
              AND p.chartdate IS NOT NULL
        ) ev USING (subject_id)
        GROUP BY mx.subject_id, mx.study_id
    """


def _group_union_query(path: str, groups, select_columns: str) -> str:
    unions: list[str] = []
    for group in groups:
        conditions: list[str] = []
        for version, prefixes in (
            (9, group.icd9_prefixes),
            (10, group.icd10_prefixes),
        ):
            conditions.extend(
                f"(icd_version = {version} AND icd_code LIKE '{prefix}%')"
                for prefix in prefixes
            )
        if not conditions:
            continue
        unions.append(
            f"""
            SELECT
                {select_columns},
                '{group.group_id}' AS code_group
            FROM read_parquet('{path}')
            WHERE {' OR '.join(conditions)}
            """
        )
    return " UNION ALL ".join(unions)



# The echocardiogram is the richest reference standard in this project and the
# least used: the source carries 302 distinct measurement codes and the wide
# table drew on 69 of them.
#
# Five of the columns declared below were, until this was measured, entirely
# empty -- the source names they asked for do not exist in MIMIC's echo
# vocabulary. The comment that used to stand here said they had been "verified
# against this project's own normalized echo table"; they had not. Counted over
# the 206,488 studies:
#
#     tissue_e_prime_septal      septal_e_prime  ->  sept_e_prime         63.9 %
#     tissue_e_prime_lateral     lateral_e_prime ->  lat_e_prime          64.5 %
#     aortic_mean_gradient_mmhg  av_mn_grad      ->  av_mean_grad         16.5 %
#     aortic_valve_area_cm2      av_area         ->  av_area_continuity   12.0 %
#     ascending_aorta_diameter   asc_ao_diam     ->  ascending_diam       70.7 %
#
# The tissue-Doppler columns were the worst loss: e-prime is what makes
# E/e-prime interpretable, and both halves were absent from every diastolic
# endpoint. The two are renamed to the unit they actually arrive in -- a median
# of 0.07 and 0.09 is metres per second, not centimetres -- because a column
# that lies about its unit is worse than one that is missing.
#
# Every entry below carries its measured coverage. ``_check_echo_admission``
# fails the audit if a declared column stays empty, so this cannot recur
# silently.
_ECHO_NUMERIC_MEASUREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # -- diastolic filling ---------------------------------------------------
    ("mitral_e_velocity_m_per_s", ("mv_peak_e",)),                    # 84.6 %
    ("mitral_a_velocity_m_per_s", ("mv_peak_a",)),                    # 75.1 %
    ("mitral_e_a_ratio_measured", ("mv_peak_e_a",)),                  # 74.7 %
    ("mitral_e_deceleration_time_ms", ("mv_e_decel",)),               # 82.1 %
    ("mitral_e_e_prime_ratio", ("e_e_prime",)),                       # 25.1 %
    ("tissue_e_prime_septal_m_per_s", ("sept_e_prime",)),             # 63.9 %
    ("tissue_e_prime_lateral_m_per_s", ("lat_e_prime",)),             # 64.5 %
    # -- aortic valve and root ----------------------------------------------
    ("aortic_peak_velocity_m_per_s", ("av_pk_vel",)),                 # 84.6 %
    ("aortic_mean_gradient_mmhg", ("av_mean_grad",)),                 # 16.5 %
    ("aortic_peak_gradient_mmhg", ("av_pk_grad",)),                   # 18.5 %
    ("aortic_valve_area_cm2", ("av_area_continuity",)),               # 12.0 %
    ("aortic_sinus_diameter_cm", ("sinus_diam",)),                    # 77.3 %
    ("ascending_aorta_diameter_cm", ("ascending_diam",)),             # 70.7 %
    ("aortic_arch_diameter_cm", ("arch_diam",)),                      # 49.7 %
    ("descending_aorta_diameter_cm", ("descending_diam",)),           # 14.9 %
    # -- atrial and right-heart dimensions ----------------------------------
    ("la_anteroposterior_dimension_cm", ("la_dimen",)),               # 77.7 %
    ("la_four_chamber_length_cm", ("la_4ch_length",)),                # 74.3 %
    ("ra_length_cm", ("ra_length",)),                                 # 74.4 %
    ("rv_diastolic_diameter_cm", ("rv_diam",)),                       # 42.9 %
    # -- left-ventricular outflow, the stroke-volume inputs ------------------
    #
    # These three are what let a stroke volume be computed at all, and with it
    # a cardiac output and the mechanical systole an electromechanical window
    # needs. None of them reached the wide table before.
    ("lvot_diameter_cm", ("lvot_diam",)),                             # 65.1 %
    ("lvot_velocity_time_integral_cm", ("lvot_vti",)),                # 57.6 %
    ("lvot_peak_velocity_m_per_s", ("lvot_pk_vel",)),                 #  6.9 %
    ("resting_lvot_gradient_mmhg", ("rest_lvot",)),                   #  5.4 %
    # -- ventricular volumes and shortening ----------------------------------
    ("fractional_shortening_percent", ("fs",)),                       # 25.9 %
    ("lv_end_diastolic_volume_ml", ("biplane_lvedv", "lvedv_3d")),    # 14.6 %
    ("lv_end_systolic_volume_ml", ("biplane_lvesv", "lvesv_3d")),     # 13.6 %
    # -- volume status -------------------------------------------------------
    #
    # The inferior vena cava is the echo's own volume-status reading, and the
    # obvious reference for any hypothesis that loop size tracks third-spacing.
    ("ivc_diameter_cm", ("ivc_diam",)),                               # 22.6 %
    # -- mitral valve haemodynamics ------------------------------------------
    ("mitral_valve_area_cm2", ("mv_valve_area", "mv_2dmva")),         #  0.9 %
    ("mitral_mean_gradient_mmhg", ("mv_mean_grad",)),                 #  6.2 %
    ("mitral_regurgitant_orifice_cm2", ("mv_ero",)),                  #  0.7 %
    # -- state at the time of the echo ---------------------------------------
    #
    # Recorded on almost every study and never carried across. The heart rate
    # in particular is the covariate whose absence let an unadjusted screen
    # report the calcium-QT relationship with the wrong sign.
    ("echo_resting_heart_rate_bpm", ("resting_hr",)),                 # 97.6 %
    ("echo_resting_systolic_bp_mmhg", ("resting_sbp",)),              # 97.8 %
    ("echo_resting_diastolic_bp_mmhg", ("resting_dbp",)),             # 97.5 %
    ("echo_weight_kg", ("weight_kg",)),                               # 61.7 %
    ("echo_height_cm", ("height_cm",)),                               # 61.3 %
)

# Ordinal text gradings that parallel the valve severities already handled.
_ECHO_ORDINAL_MEASUREMENTS: tuple[tuple[str, str], ...] = (
    ("rv_chamber_size", "rv_chamber_size"),
    ("phtn_grading", "phtn_severity"),
    # mild / mod mac / severe -- the one new categorical field whose vocabulary
    # maps onto the shared severity scale without interpretation.
    ("mitral_annular_calcification", "mac_severity"),                 # 26.0 %
)

# Categorical fields carried as text only.
#
# No severity is inferred for these: their vocabularies do not sit on the
# none/mild/moderate/severe scale, and guessing would invent an ordering the
# source does not have. Pericardial effusion, for instance, is graded by size
# ("trivial", "very small (<0.5cm)", "small (<1.0cm)", "moderate (1.0-2.0cm)",
# "large (>2.0cm)"), and the shared severity mapping would score three of those
# five as unknown. A label rule that wants an ordering states its own.
_ECHO_TEXT_MEASUREMENTS: tuple[tuple[str, str], ...] = (
    # The classic cause of low QRS voltage, and therefore the positive control
    # every amplitude marker in this project has been missing.
    ("pericardial_effusion", "pericardial_effusion"),                 # 18.9 %
    ("cardiac_tamponade", "tamponade"),                               #  6.2 %
    ("ivc_size_and_collapse", "ivc"),                                 # 21.4 %
    ("diastolic_function_statement", "diastolic_fcn"),                # 20.7 %
    ("diastolic_dysfunction_grade", "diastolic_grade"),               # 10.6 %
    ("lvot_obstruction_statement", "lv_obstruction"),                 # 73.8 %
    ("lvot_obstruction_valsalva", "lv_obstruction_valsalva"),         #  0.3 %
    ("mitral_systolic_anterior_motion", "mv_sam"),                    #  1.9 %
    # Septal motion names conduction abnormality and pacing directly, which
    # makes it both a useful reference and a circular one; it is carried so
    # that the circularity can be seen rather than met by surprise.
    ("septal_motion_statement", "septal_motion"),                     #  8.0 %
    ("lv_rv_dyssynchrony", "lv_dyssynchrony"),                        #  0.4 %
    ("lv_aneurysm", "lv_aneurysm"),                                   #  0.4 %
    ("lv_mass_or_thrombus", "lv_mass_throm"),                         #  3.0 %
    ("patent_foramen_or_asd", "pfo_asd"),                             # 18.8 %
    ("ventricular_septal_defect", "vsd"),                             #  0.2 %
    ("aortic_valve_leaflets", "av_leaflets"),                         # 86.2 %
    ("mitral_valve_leaflets", "mv_leaflets"),                         # 86.5 %
    ("rheumatic_mitral_deformity", "rheumatic_mv"),                   #  0.3 %
    # Quality on the reference side. Present on essentially every study and a
    # covariate the project has for the ECG but never had for the echo.
    ("echo_technical_quality", "technical_quality"),                  # 100  %
    # A pacing wire in the right heart is visible on the echo and on the ECG;
    # it is a confounder for both, and now a nameable one.
    ("right_heart_catheter_or_wire", "cath_wire"),                    #  3.4 %
)

# The seventeen wall-motion segments, individually.
#
# Only the counts were kept before, which answers "is anything abnormal" and
# nothing else. Vectorcardiography's oldest claim is that the loop says *where*
# the abnormality is, and that claim cannot be tested against a count.
_ECHO_WALL_MOTION_SEGMENTS: tuple[str, ...] = (
    "wm_ant_base", "wm_ant_mid", "wm_ant_apex",
    "wm_ant_sept_base", "wm_ant_sept_mid", "wm_sept_apex",
    "wm_ant_lat_base", "wm_ant_lat_mid", "wm_lat_apex",
    "wm_inf_base", "wm_inf_mid", "wm_inf_apex",
    "wm_inf_sept_base", "wm_inf_sept_mid",
    "wm_inf_lat_base", "wm_inf_lat_mid",
    "wm_apical_cap",
)


def _echo_numeric_select_terms() -> tuple[tuple[str, str], ...]:
    """Column name and quoted source-name list for the generated echo pivot."""

    return tuple(
        (column, ", ".join(f"'{name}'" for name in sources))
        for column, sources in _ECHO_NUMERIC_MEASUREMENTS
    )


def echo_derived_diastolic_columns() -> tuple[str, ...]:
    """Wide-table columns the diastolic and geometry extension contributes."""

    numeric = tuple(column for column, _ in _ECHO_NUMERIC_MEASUREMENTS)
    ordinal = tuple(
        f"{column}{suffix}"
        for column, _ in _ECHO_ORDINAL_MEASUREMENTS
        for suffix in ("_raw", "_severity")
    )
    text = tuple(column for column, _ in _ECHO_TEXT_MEASUREMENTS)
    return numeric + ordinal + text + _ECHO_WALL_MOTION_SEGMENTS + (
        "mitral_e_a_ratio",
        "relative_wall_thickness_echo",
        "la_dimension_indexed",
        "lvot_stroke_volume_ml",
    )


def _severity_sql(raw_expression: str) -> str:
    value = f"LOWER(COALESCE({raw_expression}, ''))"
    return f"""
        CASE
            WHEN {value} = '' OR {value} LIKE '%not quantified%'
                OR {value} LIKE '%cannot%' OR {value} LIKE '%unable%'
                THEN NULL
            -- A negated grade is not an affirmed diagnosis or a normal valve.
            -- Mixed statements remain indeterminate; no free-text adjudication.
            WHEN regexp_matches({value},
                '(^|[^a-z])(no|not|without)[ ]+(significant[ ]+)?(severe|moderate|mild|stenosis|regurgitation)')
                OR regexp_matches({value},
                '(^|[^a-z])(possible|possibly|suspected|questionable|uncertain)([^a-z]|$)')
                THEN NULL
            WHEN {value} LIKE '%severe%' OR {value} LIKE '%sev %'
                OR {value} LIKE '%3+%' OR {value} LIKE '%4+%' THEN 3
            WHEN (
                    {value} LIKE '%moderate%'
                 OR {value} LIKE '%mod %'
                 OR {value} LIKE '%mod [%'
                 OR {value} LIKE '%2+%'
                 )
                 AND {value} NOT LIKE '%mild-mod%'
                 AND {value} NOT LIKE '%mild moderate%'
                THEN 2
            WHEN {value} LIKE '%mild%' OR {value} LIKE '%1+%'
                OR {value} LIKE '%trace%' OR {value} LIKE '%trivial%'
                OR {value} LIKE '%minimal%' THEN 1
            WHEN {value} LIKE '%none%' OR {value} LIKE '%no valvular%'
                OR {value} LIKE '%physiologic%' OR {value} LIKE '%normal%'
                THEN 0
            ELSE NULL
        END
    """


def _echo_value_status_sql() -> str:
    unit = "LOWER(TRIM(COALESCE(unit, '')))"
    numeric = "result_numeric"
    expected_units = {
        ("lvef", "lvef_3d", "biplane_lvef"): ("%",),
        (
            "septal_thickness",
            "inf_lat_thickness",
            "lvedd",
            "lvesd",
            "lvid_d",
            "lvid_s",
            "lv_dimen_diastole",
            "lv_dimen_systole",
            "rv_diam",
            "tapse",
        ): ("cm",),
        ("la_vol",): ("cm3", "ml"),
        ("body_surface_area",): ("m2",),
        ("la_vol_index", "lavi"): ("ml/m2", "cm3/m2"),
        ("lv_mass_index", "lvmi"): ("g/m2",),
        ("tr_velocity", "tr_peak_velocity"): ("m/s",),
        (
            "tr_mmhg",
            "tv_est_pa_press_min",
            "rvsp",
            "pasp",
            "rv_systolic_pressure",
        ): ("mm hg", "mmhg"),
    }
    plausible_ranges = {
        ("lvef", "lvef_3d", "biplane_lvef"): (0.0, 100.0),
        ("septal_thickness", "inf_lat_thickness"): (0.2, 5.0),
        (
            "lvedd",
            "lvesd",
            "lvid_d",
            "lvid_s",
            "lv_dimen_diastole",
            "lv_dimen_systole",
        ): (1.0, 12.0),
        ("rv_diam",): (0.5, 10.0),
        ("tapse",): (0.1, 5.0),
        ("la_vol",): (1.0, 1000.0),
        ("body_surface_area",): (0.5, 4.5),
        ("la_vol_index", "lavi"): (1.0, 300.0),
        ("lv_mass_index", "lvmi"): (10.0, 500.0),
        ("tr_velocity", "tr_peak_velocity"): (0.5, 8.0),
        ("tr_mmhg",): (0.0, 300.0),
        (
            "tv_est_pa_press_min",
            "rvsp",
            "pasp",
            "rv_systolic_pressure",
        ): (0.0, 250.0),
    }

    unit_clauses: list[str] = []
    for measurements, units in expected_units.items():
        measurement_sql = ", ".join(f"'{item}'" for item in measurements)
        unit_sql = ", ".join(f"'{item}'" for item in units)
        unit_clauses.append(
            f"WHEN measurement IN ({measurement_sql}) "
            f"AND {unit} NOT IN ({unit_sql}) THEN 'unexpected_unit'"
        )
    range_clauses: list[str] = []
    for measurements, (lower, upper) in plausible_ranges.items():
        measurement_sql = ", ".join(f"'{item}'" for item in measurements)
        range_clauses.append(
            f"WHEN measurement IN ({measurement_sql}) "
            f"AND NOT ({numeric} BETWEEN {lower} AND {upper}) "
            "THEN 'implausible_numeric'"
        )
    return f"""
        CASE
            WHEN parse_status = 'categorical' THEN 'usable_categorical'
            WHEN parse_status <> 'numeric' OR result_numeric IS NULL
                THEN 'missing'
            WHEN NOT ISFINITE(result_numeric) THEN 'implausible_numeric'
            {' '.join(unit_clauses)}
            {' '.join(range_clauses)}
            ELSE 'usable_numeric'
        END
    """


def _name_condition(column: str, patterns: Iterable[str]) -> str:
    conditions = []
    for pattern in patterns:
        escaped = pattern.replace("'", "''")
        conditions.append(f"LOWER({column}) LIKE '%{escaped}%'")
    return " OR ".join(conditions)


# The unit each laboratory group is expected to arrive in, measured on the
# 52.1 million rows this project actually ingests rather than assumed.
#
# This table used to cover twelve groups. Every other group fell through to
# ``ELSE 'unit_unrecognized'`` and was then filtered out of every window
# aggregate, which silently removed **28.6 million rows** -- sodium, glucose,
# urea nitrogen, bicarbonate, leukocytes, albumin, TSH, free T4, HbA1c, CRP,
# D-dimer and the coagulation times. The thyroid axis, named in the project
# description as an example of the kind of finding being looked for, was
# therefore empty by construction.
#
# The measured share and median of the dominant unit are recorded next to each
# entry so that a future reviewer can see what was checked. Where a group
# arrives in two units that are not the same quantity, that is handled
# explicitly below rather than by accepting both.
LAB_EXPECTED_UNITS: dict[str, tuple[str, ...]] = {
    "albumin": ("g/dl",),                        # 100 %, median 3.9
    "bicarbonate": ("meq/l", "mmol/l"),          # 100 %, median 25.0
    "blood_gas_ph": ("units",),                  # 100 %, median 7.38
    "c_reactive_protein": ("mg/l",),             # 100 %, median 7.1
    "ck_mb": ("ng/ml",),                         #  88 %, median 4.0
    "creatinine": ("mg/dl",),                    # 100 %, median 0.9
    "d_dimer": ("ng/ml feu",),                   #  53 %, median 856.5
    "free_t4": ("ng/dl",),                       # 100 %, median 1.2
    "glucose": ("mg/dl",),                       # 100 %, median 112.0
    "hba1c": ("%",),                             # 100 %, median 6.1
    "hemoglobin": ("g/dl",),                     # 100 %, median 10.9
    "ionized_calcium": ("mmol/l",),              # 100 %, median 1.12
    "lactate": ("mmol/l",),                      # 100 %, median 1.7
    "magnesium": ("mg/dl",),                     # 100 %, median 2.0
    "ntprobnp": ("pg/ml",),                      # 100 %, median 1253
    "partial_thromboplastin_time": ("sec",),     # 100 %, median 33.1
    "phosphate": ("mg/dl",),                     # 100 %, median 3.4
    "potassium": ("meq/l", "mmol/l"),            # 100 %, median 4.1
    "prothrombin_time": ("sec",),                # 100 %, median 14.0
    "sodium": ("meq/l", "mmol/l"),               # 100 %, median 139.0
    "total_calcium": ("mg/dl",),                 # 100 %, median 8.8
    "troponin_i": ("ng/ml",),                    # 100 %, median 0.0
    "troponin_t": ("ng/ml",),                    # 100 %, median 0.09
    "tsh": ("uiu/ml", "uu/ml"),                  # 100 %, median 1.8
    "urea_nitrogen": ("mg/dl",),                 # 100 %, median 18.0
    "white_blood_cells": ("k/ul",),              # 100 %, median 7.6
}

#: Groups whose result is a ratio and legitimately carries no unit. Without
#: this the ``unit_missing`` rule discards all 1.77 million INR values.
LAB_UNITLESS_GROUPS: frozenset[str] = frozenset({"inr"})

#: Units that belong to the same group in MIMIC but measure a different
#: quantity or a different scale. Accepting them silently mixes two
#: distributions in one column, which is worse than dropping them: it produces
#: a bimodal marker that no threshold can be read off.
LAB_AMBIGUOUS_UNITS: dict[str, tuple[str, ...]] = {
    # The relative index (CK-MB as a percentage of total CK) is a different
    # quantity from the mass concentration; 12 % of rows, median 4.8 %.
    "ck_mb": ("%",),
    # Fibrinogen-equivalent units run about twice D-dimer units, and a bare
    # "ng/ml" does not say which assay produced it -- 46 % of rows, median 510
    # against 856 for the labelled FEU rows.
    "d_dimer": ("ng/ml", "ng/ml ddu"),
}


def _lab_unit_status_case_sql() -> str:
    """Classify each laboratory row as usable, unitless-usable, or quarantined.

    The order matters: unitless groups are decided before the general
    ``unit_missing`` rule, because for them a missing unit is correct.
    """

    unitless_clauses = []
    ambiguous_clauses = []
    unit_clauses = []
    for group, itemids in LAB_ITEM_GROUPS.items():
        item_sql = ", ".join(str(int(itemid)) for itemid in itemids)
        if group in LAB_UNITLESS_GROUPS:
            unitless_clauses.append(
                "WHEN TRY_CAST(itemid AS INTEGER) IN "
                f"({item_sql}) AND NULLIF(TRIM(valueuom), '') IS NULL "
                "THEN 'usable'"
            )
        ambiguous = LAB_AMBIGUOUS_UNITS.get(group)
        if ambiguous:
            ambiguous_sql = ", ".join(
                f"'{unit.replace(chr(39), chr(39) * 2)}'" for unit in ambiguous
            )
            ambiguous_clauses.append(
                "WHEN TRY_CAST(itemid AS INTEGER) IN "
                f"({item_sql}) AND LOWER(TRIM(valueuom)) IN ({ambiguous_sql}) "
                "THEN 'unit_scale_ambiguous'"
            )
        units = LAB_EXPECTED_UNITS.get(group)
        if not units:
            continue
        unit_sql = ", ".join(
            f"'{unit.replace(chr(39), chr(39) * 2)}'" for unit in units
        )
        unit_clauses.append(
            "WHEN TRY_CAST(itemid AS INTEGER) IN "
            f"({item_sql}) AND LOWER(TRIM(valueuom)) IN ({unit_sql}) "
            "THEN 'usable'"
        )
    ordered = unitless_clauses + ambiguous_clauses + unit_clauses
    return "\n                " + "\n                ".join(ordered)




def _lab_window_pivot_sql(
    lab_groups: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> str:
    """Generate the six-hour nearest-value pivot for every lab group.

    This used to be hand-written per group, and the predictable happened:
    ck_mb and blood_gas_ph were defined, windowed, and then silently dropped
    because nobody added their five lines here. Generating the block from
    LAB_ITEM_GROUPS makes adding a lab group a one-line change and makes the
    pivot impossible to forget.
    """

    skipped = {str(item) for item in exclude}
    fragments: list[str] = []
    for group in lab_groups:
        name = str(group)
        if name in skipped:
            continue
        fragments.append(
            f"""ARG_MIN(nearest_value_6h, ABS(nearest_delta_hours_6h)) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_nearest,
            ARG_MIN_NULL(nearest_bound_6h, ABS(nearest_delta_hours_6h)) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_nearest_bound,
            ARG_MIN_NULL(nearest_censoring_6h, ABS(nearest_delta_hours_6h)) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_nearest_censoring,
            ARG_MIN(nearest_delta_hours_6h,
                    ABS(nearest_delta_hours_6h)) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_delta_hours,
            ARG_MIN(nearest_reference_lower_6h,
                    ABS(nearest_delta_hours_6h)) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_reference_lower,
            ARG_MIN(nearest_reference_upper_6h,
                    ABS(nearest_delta_hours_6h)) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_reference_upper,
            ARG_MIN(nearest_unit_6h, ABS(nearest_delta_hours_6h)) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_unit,
            ARG_MIN(nearest_labevent_id_6h,
                    ABS(nearest_delta_hours_6h)) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_nearest_labevent_id,
            MAX(count_6h) FILTER (
                WHERE lab_group = '{name}'
            ) AS {name}_count_6h,"""
        )
    return "\n            ".join(fragments)


def _icu_event_relations(config: MimicProjectConfig) -> dict[str, str]:
    """Resolve the ICU event tables to queryable relations.

    Preference order per table: the partitioned clinical-event registry, then
    a raw MIMIC CSV. A table that resolves to neither is absent from the
    result, and the caller then writes an explicitly empty output rather than
    a silently empty one.
    """

    icu_root = config.mimic_iv_root / "icu"
    relations: dict[str, str] = {}
    for table_id in (
        "chartevents",
        "inputevents",
        "outputevents",
        "procedureevents",
    ):
        relation = clinical_event_relation(config, table_id)
        if relation is None:
            raw_path = icu_root / f"{table_id}.csv.gz"
            relation = _raw_csv(raw_path) if raw_path.is_file() else None
        if relation is not None:
            relations[table_id] = relation
    return relations


def icu_evidence_coverage(config: MimicProjectConfig) -> dict[str, object]:
    """Machine-readable state of the ICU evidence layer.

    Mirrors the note-coverage contract: an ICU table that was never resolved
    is reported as ``not_available``, so an empty ICU feature column can be
    told apart from "assessed, nothing found".
    """

    relations = _icu_event_relations(config)
    expected = (
        "chartevents",
        "inputevents",
        "outputevents",
        "procedureevents",
    )
    present = tuple(name for name in expected if name in relations)
    missing = tuple(name for name in expected if name not in relations)
    icustays_available = (
        config.standardized_root / "icustays.parquet"
    ).is_file() or clinical_event_relation(config, "icustays") is not None
    if not icustays_available:
        status = "no_icu_stays"
    elif not present:
        status = "not_available"
    elif missing:
        status = "partial"
    else:
        status = "complete"
    return {
        "status": status,
        "assessed": status == "complete",
        "icustays_available": bool(icustays_available),
        "present_tables": list(present),
        "missing_tables": list(missing),
        "zero_rows_are_not_evidence_of_assessed_absence": True,
    }


def _devereux_context_columns_sql() -> str:
    """LV mass (ASE linear, Devereux), its BSA index and the ASE geometry.

    Audit D5: MIMIC-IV reports no LV mass index (0 of 206,488 studies), so
    the "strict LVH" endpoint ran on the wall-category text alone -- walls
    >= 1.5 cm against < 1.2 cm, an extreme-group design that also counted
    eccentric LVH (normal walls, dilated cavity) as a control. The mass is
    reconstructed from the three linear measurements with the bounds of
    ``voltage_mass`` (one source for both analyses), indexed by the echo
    BSA. Geometry follows ASE 2015: LVMI > 115 / > 95 g/m2 (men/women) and
    RWT > 0.42.
    """
    from .voltage_mass import DIAMETER_BOUNDS_CM, WALL_BOUNDS_CM

    low_wall, high_wall = WALL_BOUNDS_CM
    low_diameter, high_diameter = DIAMETER_BOUNDS_CM
    septal = "ew.septal_thickness_cm"
    posterior = "ew.posterior_wall_thickness_cm"
    diameter = "ew.lv_internal_diameter_diastole_cm"
    bsa = "ew.echo_body_surface_area_m2"
    usable = (
        f"{septal} BETWEEN {low_wall} AND {high_wall} "
        f"AND {posterior} BETWEEN {low_wall} AND {high_wall} "
        f"AND {diameter} BETWEEN {low_diameter} AND {high_diameter}"
    )
    mass = (
        f"(0.8 * 1.04 * (POWER({diameter} + {posterior} + {septal}, 3) "
        f"- POWER({diameter}, 3)) + 0.6)"
    )
    index = f"({mass} / {bsa})"
    index_usable = f"{usable} AND {bsa} BETWEEN 0.5 AND 4.5"
    rwt = f"(2.0 * {posterior} / {diameter})"
    threshold = (
        "CASE WHEN m.sex = 'M' THEN 115.0 WHEN m.sex = 'F' THEN 95.0 END"
    )
    return f"""CASE WHEN {usable} THEN {mass} END AS lv_mass_devereux_g,
            CASE WHEN {index_usable} THEN {index} END
                AS lv_mass_index_devereux_g_m2,
            CASE
                WHEN NOT COALESCE({index_usable}, FALSE)
                     OR ({threshold}) IS NULL THEN NULL
                WHEN {index} > ({threshold}) AND {rwt} > 0.42
                    THEN 'concentric_hypertrophy'
                WHEN {index} > ({threshold}) THEN 'eccentric_hypertrophy'
                WHEN {rwt} > 0.42 THEN 'concentric_remodeling'
                ELSE 'normal_geometry'
            END AS lv_geometry_ase"""


def _event_or_raw_relation(
    config: MimicProjectConfig,
    table_id: str,
    raw_path: Path,
) -> str:
    standardized = clinical_event_relation(config, table_id)
    if standardized is not None:
        return standardized
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Neither standardized nor raw source exists for {table_id}: "
            f"{raw_path}"
        )
    return _raw_csv(raw_path)


def _raw_csv(path: Path) -> str:
    normalized = _sql_path(path)
    return (
        f"read_csv('{normalized}', header=true, all_varchar=true, "
        "auto_detect=true, ignore_errors=false)"
    )


def _copy_partitioned_query_atomic(
    connection,
    build_query: Callable[[int, int], str],
    output_path: Path,
    *,
    initial_partitions: int,
    max_partitions: int = 64,
    threads: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
    label: str,
) -> int:
    """Execute a per-subject aggregation in disjoint buckets and union them.

    Grouped aggregation over ECG-centred windows is the one shape in this
    module that DuckDB cannot spill.  ``ARG_MIN``/``ARG_MAX`` keep their
    ordering key in the aggregate state, and a state holding a STRUCT or a
    VARCHAR has no fixed size, so the hash table stays resident no matter how
    much temporary directory it is offered.  Peak memory is therefore set by
    *group count x aggregate count*, and the only lever that moves it is
    processing fewer groups at a time.

    ``hash(subject_id) % n`` splits the work into buckets that are disjoint in
    the grouping key, so every group lands whole in exactly one bucket and the
    union of the buckets is the unpartitioned result, row for row.  The same
    predicate is applied to both sides of the equi-join, which is redundant in
    logic but not in cost: it shrinks the join build side by the same factor.

    A guessed bucket count is a guess about data this code has not seen.  When
    a bucket still runs out of memory the count is doubled and the attempt
    restarts, so the query settles at a size the machine actually has rather
    than at one that looked plausible when the line was written.

    ``threads`` lowers DuckDB's parallelism for the duration.  That reads like
    a pessimisation and is the opposite for wide aggregations: each thread
    builds its own radix-partitioned hash table, so the resident state scales
    with *threads x aggregates*, not with rows.  Measured on the 208-aggregate
    laboratory pivot at full size, six threads needed 10.7 GiB and 237 s while
    two threads needed 6.6 GiB and 87 s -- faster and lighter, because the work
    that was parallelised was table setup rather than anything in the data.
    """

    if initial_partitions < 1:
        raise ValueError("initial_partitions must be positive")
    partitions = int(initial_partitions)
    restore_threads: Optional[str] = None
    if threads is not None:
        try:
            restore_threads = str(
                connection.execute(
                    "SELECT current_setting('threads')"
                ).fetchone()[0]
            )
            connection.execute(f"SET threads={max(1, int(threads))}")
        except Exception:  # noqa: BLE001 - parallelism is an optimisation
            restore_threads = None
    try:
        return _run_partitioned(
            connection,
            build_query,
            output_path,
            partitions=partitions,
            max_partitions=max_partitions,
            progress=progress,
            label=label,
        )
    finally:
        if restore_threads is not None:
            connection.execute(f"SET threads={restore_threads}")


def _run_partitioned(
    connection,
    build_query: Callable[[int, int], str],
    output_path: Path,
    *,
    partitions: int,
    max_partitions: int,
    progress: Optional[Callable[[str], None]],
    label: str,
) -> int:
    # The label is shown to the user, so it carries spaces and umlauts; the
    # bucket files are read back through a glob, so they must not.
    slug = "".join(
        character if character.isascii() and character.isalnum() else "-"
        for character in label.lower()
    ).strip("-") or "bucket"
    # Keep successful buckets when an OOM forces a finer modulus. If bucket
    # i/n is complete, it is exactly the disjoint union of i/(2n) and
    # (i+n)/(2n); recomputing both used to throw away all work before the
    # failed bucket. A wide lab aggregation can spend hours reaching a late
    # skewed bucket, so that retry behaviour was a material full-run cost.
    completed: list[tuple[int, int, Path]] = []
    with tempfile.TemporaryDirectory(
        prefix=f"{slug}-parts-",
        dir=output_path.parent,
    ) as temporary_root:
        while True:
            try:
                for partition_index in range(partitions):
                    covered = any(
                        partitions % completed_count == 0
                        and partition_index % completed_count == completed_index
                        for completed_index, completed_count, _ in completed
                    )
                    if covered:
                        continue
                    if progress:
                        progress(
                            f"{label} ({partition_index + 1}/{partitions})"
                        )
                    part_path = (
                        Path(temporary_root)
                        / (
                            f"{slug}-n{partitions:04d}-"
                            f"i{partition_index:04d}.parquet"
                        )
                    )
                    _copy_query_atomic(
                        connection,
                        build_query(partition_index, partitions),
                        part_path,
                    )
                    completed.append(
                        (partition_index, partitions, part_path)
                    )
                part_glob = _sql_path(Path(temporary_root) / "*.parquet")
                return _copy_query_atomic(
                    connection,
                    f"SELECT * FROM read_parquet('{part_glob}')",
                    output_path,
                )
            except Exception as exc:  # noqa: BLE001 - only OOM is retried
                if not _is_out_of_memory(exc):
                    raise
                if partitions >= max_partitions:
                    raise RuntimeError(
                        f"{label}: bucket count reached the {max_partitions} "
                        "limit and DuckDB still ran out of memory. The stage "
                        "needs a larger memory allocation, not finer buckets."
                    ) from exc
                partitions *= 2
                if progress:
                    progress(
                        f"{label}: Speicher erschoepft, feinere Aufteilung "
                        f"({partitions} Buckets; fertige Buckets bleiben erhalten)"
                    )


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether a DuckDB failure is an allocation failure we can bucket around.

    Only allocation failures may be answered with finer buckets. A binder or
    conversion error answered the same way would retry a broken query eight
    times and then report a memory problem, which is how a one-line typo turns
    into an afternoon of profiling the wrong thing.
    """

    try:
        import duckdb

        if isinstance(exc, duckdb.OutOfMemoryException):
            return True
    except (ImportError, AttributeError):
        pass
    if type(exc).__name__ == "OutOfMemoryException":
        return True
    return "out of memory" in str(exc).lower()


def _copy_checkpointed_medication_query(
    connection, query: str, output_path: Path, *,
    checkpoint_context: Optional[tuple[Path, dict]], threads: Optional[int] = None,
) -> int:
    """Persist expensive COPY results inside the validated stage checkpoint.

    The caller's checkpoint binds input data, configuration and stage version.
    SQL and output content are additionally checked here. Independent shadow
    builds pass no context and always execute their own query.
    """
    identity = content_hash({"sql": query, "output": str(output_path.resolve())})
    if checkpoint_context is not None:
        checkpoint_path, checkpoint = checkpoint_context
        previous = checkpoint.get("medication_queries", {}).get(identity)
        if isinstance(previous, dict) and output_path.is_file():
            fingerprint = file_content_fingerprint(output_path)
            count = previous.get("row_count")
            if (type(count) is int and count >= 0
                    and previous.get("sha256") == fingerprint["sha256"]
                    and previous.get("size") == fingerprint["size"]):
                return count
    count = (
        _copy_query_atomic(connection, query, output_path) if threads is None
        else _copy_query_atomic_with_threads(connection, query, output_path, threads=threads)
    )
    if checkpoint_context is not None:
        fingerprint = file_content_fingerprint(output_path)
        checkpoint.setdefault("medication_queries", {})[identity] = {
            "row_count": count, "sha256": fingerprint["sha256"], "size": fingerprint["size"],
        }
        # Persist before the next progress callback can honor a pause request.
        atomic_write_json(checkpoint_path, checkpoint)
    return count


def _copy_query_atomic(connection, query: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    if temporary.exists():
        temporary.unlink()
    temporary_sql = _sql_path(temporary)
    with profile_query(f"evidence:{output_path.name}") as profile:
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


def _copy_query_atomic_with_threads(
    connection,
    query: str,
    output_path: Path,
    *,
    threads: int,
) -> int:
    """Run one COPY with bounded parallel state and restore the session.

    DuckDB's hash joins and grouped aggregates may allocate operator state per
    worker.  A query can therefore be both faster and substantially smaller
    with fewer threads.  This helper deliberately changes only the selected
    COPY and restores the caller's setting even when DuckDB raises.
    """

    if int(threads) < 1:
        raise ValueError("threads must be positive")
    previous_threads = int(
        connection.execute(
            "SELECT current_setting('threads')"
        ).fetchone()[0]
    )
    connection.execute(f"SET threads={int(threads)}")
    try:
        return _copy_query_atomic(connection, query, output_path)
    finally:
        connection.execute(f"SET threads={previous_threads}")


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")
