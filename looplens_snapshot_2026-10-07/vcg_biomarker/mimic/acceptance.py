from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ..core.delineation.benchmark_suite import (
    DELINEATION_RELEASE_POLICY_VERSION,
    DELINEATION_SUITE_VERSION,
    delineation_implementation_hash,
    evaluate_delineation_release_gate,
)
from .config import MimicProjectConfig
from .event_equivalence import (
    verify_clinical_event_evidence_provenance,
)
from .event_ingest import load_clinical_event_registry
from .evidence import echo_derived_diastolic_columns
from .feature_extraction import verify_feature_artifacts
from .group_vcg import group_vcg_implementation_hash
from .labels import verify_development_label_attestation
from .phenome_promotions import verify_phenome_promotion_refresh
from .ptb_transfer import ptb_implementation_hash
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    feature_parquet_clause,
    parquet_columns,
)
from .validation import verify_validation_input_payload_files
from .validation_provenance import verify_validation_code_provenance


ACCEPTANCE_AUDIT_VERSION = "1.8.0"
MINIMUM_FEATURE_SUCCESS_FRACTION = 0.95
MAXIMUM_PROCESSING_ERROR_FRACTION = 0.01

DEFAULT_GROUP_VCG_ENDPOINTS = (
    "acute_mi_high_specificity",
    "echo_lvh_strict",
    "reduced_lvef",
    "right_heart_strain_proxy",
    "echo_lae",
    "hyperkalemia_near_ecg",
)

REQUIRED_EVENT_TABLES = (
    "labevents",
    "prescriptions",
    "pharmacy",
    "emar",
    "emar_detail",
    "poe",
    "poe_detail",
    "medrecon",
    "pyxis",
)

REQUIRED_CANONICAL_REPORTS = (
    "run_manifest",
    "clinical_utility_summary",
    "data_inventory",
    "ecg_master_index_summary",
    "label_summary",
    "split_summary",
    "feature_failure_report",
    "screening_results",
    "classical_baseline_results",
    "vcg_vs_classical_comparison",
    "combination_results",
    "overfitting_stability_report",
    "discordance_report",
    "review_candidates",
    "final_test_report",
)


@dataclass(frozen=True)
class AcceptanceExpectations:
    ecg_count: int = 800_035
    subject_count: int = 161_352
    echo_study_count: int = 206_488
    #: ``None`` derives the expectation from the extraction manifest
    #: (``row_counts.selected_ecgs``). A fixed constant cannot follow the
    #: selection rules: A131 selected 640,110 development ECGs, while the old
    #: default of 640,351 matched neither that nor the 640,376 development
    #: rows, so the final stage of every full run failed by construction.
    development_feature_rows: Optional[int] = None
    label_definition_count: int = 63
    minimum_bootstrap_iterations: int = 1_000
    #: A laboratory group losing more than this share of its numeric results to
    #: the unit rule means the rule does not know that group, not that the data
    #: are bad. The previous rule declared units for twelve of twenty-seven
    #: groups and quarantined 28.6 million rows in silence; a group at 100 %
    #: quarantine looked exactly like a group nobody had ordered.
    maximum_laboratory_quarantine_share: float = 0.5
    group_vcg_endpoints: tuple[str, ...] = DEFAULT_GROUP_VCG_ENDPOINTS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceCheck:
    check_id: str
    category: str
    status: str
    observed: Any
    expected: Any
    evidence: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PendingArtifact(Exception):
    """A check whose artifact a later user action produces, not a stage."""

    def __init__(self, message: str, *, observed: Any, expected: Any) -> None:
        super().__init__(message)
        self.observed = observed
        self.expected = expected


def run_development_acceptance_audit(
    config: MimicProjectConfig,
    *,
    run_id: str,
    ptb_run_id: str,
    expectations: AcceptanceExpectations = AcceptanceExpectations(),
    require_freeze: bool = False,
    strict: bool = False,
    deep_feature_verification: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Audit a development run without opening either locked outcome set."""

    run_root = config.runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    checks: list[AcceptanceCheck] = []

    def record(
        check_id: str,
        category: str,
        evaluator: Callable[[], tuple[Any, Any, str]],
    ) -> None:
        if progress:
            progress(
                f"Acceptance audit {len(checks) + 1}/16: {check_id}"
            )
        try:
            observed, expected, evidence = evaluator()
        except PendingArtifact as exc:
            # The artifact is produced by an action after the guided study
            # (group VCG selection, print build, PTB revalidation), not by
            # a stage before this audit. Pending, like the freeze itself:
            # it blocks the final audit with ``require_freeze`` and is
            # reported, not failed, in the pre-freeze audit. Before this the
            # study's own last stage failed on every run because it demanded
            # outputs no study stage writes.
            checks.append(
                AcceptanceCheck(
                    check_id=check_id,
                    category=category,
                    status="pending",
                    observed=exc.observed,
                    expected=exc.expected,
                    evidence="",
                    detail=str(exc),
                )
            )
            return
        except Exception as exc:
            checks.append(
                AcceptanceCheck(
                    check_id=check_id,
                    category=category,
                    status="fail",
                    observed={"error_type": type(exc).__name__},
                    expected="check completes without an integrity violation",
                    evidence="",
                    detail=str(exc),
                )
            )
            return
        checks.append(
            AcceptanceCheck(
                check_id=check_id,
                category=category,
                status="pass",
                observed=observed,
                expected=expected,
                evidence=evidence,
            )
        )

    record(
        "raw_data_inventory",
        "data",
        lambda: _check_inventory(config),
    )
    record(
        "standardized_ingest_counts",
        "data",
        lambda: _check_standardized_ingest(config, expectations),
    )
    record(
        "ecg_master_and_subject_split",
        "data",
        lambda: _check_master_index(config, expectations),
    )
    record(
        "clinical_event_registry",
        "provenance",
        lambda: _check_event_registry(config),
    )
    record(
        "clinical_event_evidence",
        "provenance",
        lambda: _check_event_evidence(config),
    )
    record(
        "laboratory_admission_rate",
        "evidence",
        lambda: _check_laboratory_admission(config, expectations),
    )
    record(
        "echo_column_admission",
        "evidence",
        lambda: _check_echo_admission(config),
    )
    record(
        "development_label_attestation",
        "labels",
        lambda: _check_label_attestation(config, expectations),
    )
    record(
        "delineation_release_gate",
        "measurement_validity",
        lambda: _check_delineation_release_gate(config),
    )
    record(
        "complete_development_features",
        "features",
        lambda: _check_development_features(
            config,
            run_root,
            expectations,
            deep_verify=deep_feature_verification,
        ),
    )
    record(
        "formal_train_validation",
        "validation",
        lambda: _check_formal_validation(run_root, expectations),
    )
    record(
        "development_phewas",
        "discovery",
        lambda: _check_phewas(run_root),
    )
    record(
        "group_vcg_visualizations",
        "visualization",
        lambda: _check_group_vcg(run_root, expectations),
    )
    record(
        "canonical_and_printable_reports",
        "reports",
        lambda: _check_reports(run_root),
    )
    record(
        "ptb_transfer_reference_sealed",
        "test_lock",
        lambda: _check_ptb_lock(config, ptb_run_id),
    )

    if progress:
        progress("Acceptance audit 16/16: mimic_locked_test_sealed")
    mimic_lock_check = _check_mimic_lock(
        run_root,
        require_freeze=require_freeze,
    )
    checks.append(mimic_lock_check)

    failing = [item for item in checks if item.status == "fail"]
    pending = [item for item in checks if item.status == "pending"]
    if failing:
        status = "failed"
    elif pending:
        status = "ready_to_freeze"
    else:
        status = "accepted_frozen_development"
    payload: dict[str, Any] = {
        "audit_version": ACCEPTANCE_AUDIT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "run_id": run_id,
        "ptb_run_id": ptb_run_id,
        "expectations": expectations.to_dict(),
        "require_freeze": bool(require_freeze),
        "checks": [item.to_dict() for item in checks],
        "summary": {
            "pass": sum(item.status == "pass" for item in checks),
            "pending": len(pending),
            "fail": len(failing),
        },
        "locked_test_outcomes_read": False,
        "ptb_fold_10_outcomes_read": False,
        "scientific_scope": (
            "Train/validation development acceptance only. Locked outcome "
            "tables, locked feature partitions, and PTB fold 10 are not read."
        ),
    }
    payload["audit_hash"] = content_hash(payload)
    audit_stem = (
        "acceptance_audit"
        if require_freeze
        else "acceptance_pre_freeze_audit"
    )
    json_path = run_root / f"{audit_stem}.json"
    parquet_path = run_root / f"{audit_stem}.parquet"
    atomic_write_json(json_path, payload)
    atomic_write_parquet(_checks_frame(checks), parquet_path)

    blocking = failing + (pending if require_freeze else [])
    if strict and blocking:
        unresolved = blocking
        summary = ", ".join(
            f"{item.check_id}={item.status}" for item in unresolved
        )
        raise RuntimeError(
            f"Development acceptance audit did not pass: {summary}. "
            f"Details were written to {json_path}."
        )
    return payload


def refresh_pre_freeze_acceptance_audit(
    config: MimicProjectConfig,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Re-run the recorded acceptance policy immediately before freezing."""

    path = (
        config.runs_root
        / run_id
        / "acceptance_pre_freeze_audit.json"
    )
    recorded = _load_json(path)
    recorded_hash = str(recorded.get("audit_hash", ""))
    hash_payload = dict(recorded)
    hash_payload.pop("audit_hash", None)
    if not recorded_hash or recorded_hash != content_hash(hash_payload):
        raise RuntimeError(
            "Pre-freeze acceptance audit hash verification failed"
        )
    if (
        recorded.get("run_id") != run_id
        or recorded.get("locked_test_outcomes_read") is not False
        or recorded.get("ptb_fold_10_outcomes_read") is not False
    ):
        raise RuntimeError(
            "Pre-freeze acceptance audit has unsafe or mismatched scope"
        )
    expectation_payload = recorded.get("expectations")
    if not isinstance(expectation_payload, dict):
        raise RuntimeError(
            "Pre-freeze acceptance audit has no expectation policy"
        )
    expectations = AcceptanceExpectations(
        ecg_count=int(expectation_payload["ecg_count"]),
        subject_count=int(expectation_payload["subject_count"]),
        echo_study_count=int(expectation_payload["echo_study_count"]),
        development_feature_rows=(
            None
            if expectation_payload.get("development_feature_rows") is None
            else int(expectation_payload["development_feature_rows"])
        ),
        label_definition_count=int(
            expectation_payload["label_definition_count"]
        ),
        minimum_bootstrap_iterations=int(
            expectation_payload["minimum_bootstrap_iterations"]
        ),
        group_vcg_endpoints=tuple(
            str(item)
            for item in expectation_payload.get(
                "group_vcg_endpoints",
                (),
            )
        ),
    )
    refreshed = run_development_acceptance_audit(
        config,
        run_id=run_id,
        ptb_run_id=str(recorded.get("ptb_run_id", "")),
        expectations=expectations,
        require_freeze=False,
        strict=True,
        deep_feature_verification=True,
    )
    if refreshed.get("status") != "ready_to_freeze":
        raise RuntimeError(
            "Development acceptance audit is not ready to freeze"
        )
    return refreshed


def _check_inventory(
    config: MimicProjectConfig,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = config.work_root / "data_inventory.json"
    payload = _load_json(path)
    items = payload.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Data inventory has no item registry")
    required_failures = [
        str(item.get("table_id"))
        for item in items
        if item.get("required")
        and (
            item.get("status") != "ok"
            or item.get("exists") is not True
            or item.get("missing_columns")
        )
    ]
    waveform_samples = [
        item
        for item in items
        if item.get("table_id") == "ecg_waveform_path_sample"
    ]
    if (
        payload.get("error_count") != 0
        or required_failures
        or len(waveform_samples) != 1
        or waveform_samples[0].get("status") != "ok"
    ):
        raise RuntimeError(
            "Inventory is not clean: "
            f"errors={payload.get('error_count')}, "
            f"required_failures={required_failures}, "
            f"waveform_samples={len(waveform_samples)}"
        )
    observed = {
        "table_count": len(items),
        "error_count": int(payload.get("error_count", 0)),
        "warning_count": int(payload.get("warning_count", 0)),
        "waveform_sample": waveform_samples[0].get("diagnostic", ""),
    }
    expected = {
        "required_tables": "present with expected columns",
        "error_count": 0,
        "waveform_path_sample": "ok",
    }
    return observed, expected, str(path.resolve())


def _check_standardized_ingest(
    config: MimicProjectConfig,
    expectations: AcceptanceExpectations,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = config.standardized_root / "ingest.manifest.json"
    manifest = StageManifest.load(path)
    if manifest.status != "complete":
        raise RuntimeError("Standardized ingest manifest is not complete")
    missing_outputs = [
        output for output in manifest.output_paths if not Path(output).exists()
    ]
    counts = manifest.row_counts
    observed = {
        "ecg_records": int(counts.get("ecg_records", -1)),
        "ecg_records_quarantine": int(
            counts.get("ecg_records_quarantine", -1)
        ),
        "echo_study_index": int(counts.get("echo_study_index", -1)),
        "missing_output_count": len(missing_outputs),
    }
    expected = {
        "ecg_records": expectations.ecg_count,
        "ecg_records_quarantine": 0,
        "echo_study_index": expectations.echo_study_count,
        "missing_output_count": 0,
    }
    if observed != expected:
        raise RuntimeError(
            f"Standardized ingest counts differ: {observed} != {expected}"
        )
    return observed, expected, str(path.resolve())


def _check_master_index(
    config: MimicProjectConfig,
    expectations: AcceptanceExpectations,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    summary_path = config.master_root / "ecg_master_index_summary.json"
    master_path = config.master_root / "ecg_master_index.parquet"
    summary = _load_json(summary_path)
    summary_counts = summary.get("counts") or {}
    expected_split_names = {"train", "validation", "locked_test"}
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    master_sql = _sql_path(master_path)
    with catalog.connect(isolated=True) as connection:
        row = connection.execute(
            f"""
            SELECT
                COUNT(*) AS ecg_count,
                COUNT(DISTINCT study_id) AS study_count,
                COUNT(DISTINCT subject_id) AS subject_count,
                COUNT(*) FILTER (
                    WHERE subject_id IS NULL
                       OR study_id IS NULL
                       OR ecg_time IS NULL
                ) AS invalid_identity_or_time,
                COUNT(*) FILTER (
                    WHERE waveform_path IS NULL
                       OR TRIM(waveform_path) = ''
                ) AS missing_waveform_path,
                COUNT(DISTINCT split_spec_hash) AS split_spec_hashes,
                COUNT(*) FILTER (
                    WHERE split_spec_hash IS NULL
                       OR TRIM(split_spec_hash) = ''
                ) AS missing_split_spec_hash
            FROM read_parquet('{master_sql}')
            """
        ).fetchone()
        split_rows = connection.execute(
            f"""
            SELECT split, COUNT(*) AS ecgs, COUNT(DISTINCT subject_id) AS subjects
            FROM read_parquet('{master_sql}')
            GROUP BY split
            ORDER BY split
            """
        ).fetchall()
        leaking_subjects = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT subject_id
                    FROM read_parquet('{master_sql}')
                    GROUP BY subject_id
                    HAVING COUNT(DISTINCT split) <> 1
                )
                """
            ).fetchone()[0]
        )
    split_counts = {
        str(split): {"ecg_count": int(ecgs), "subject_count": int(subjects)}
        for split, ecgs, subjects in split_rows
    }
    observed = {
        "ecg_count": int(row[0]),
        "study_count": int(row[1]),
        "subject_count": int(row[2]),
        "invalid_identity_or_time": int(row[3]),
        "missing_waveform_path": int(row[4]),
        "split_spec_hashes": int(row[5]),
        "missing_split_spec_hash": int(row[6]),
        "subject_split_overlap": leaking_subjects,
        "splits": split_counts,
        "locked_test_outcomes_read": False,
    }
    expected = {
        "ecg_count": expectations.ecg_count,
        "study_count": expectations.ecg_count,
        "subject_count": expectations.subject_count,
        "invalid_identity_or_time": 0,
        "missing_waveform_path": 0,
        "split_spec_hashes": 1,
        "missing_split_spec_hash": 0,
        "subject_split_overlap": 0,
        "split_names": sorted(expected_split_names),
        "locked_test_outcomes_read": False,
    }
    violations = []
    for key in (
        "ecg_count",
        "study_count",
        "subject_count",
        "invalid_identity_or_time",
        "missing_waveform_path",
        "split_spec_hashes",
        "missing_split_spec_hash",
        "subject_split_overlap",
    ):
        if observed[key] != expected[key]:
            violations.append(
                f"{key}={observed[key]} (expected {expected[key]})"
            )
    if set(split_counts) != expected_split_names:
        violations.append(f"split_names={sorted(split_counts)}")
    if int(summary_counts.get("ecg_count", -1)) != expectations.ecg_count:
        violations.append("summary ecg_count mismatch")
    if int(summary_counts.get("subject_count", -1)) != expectations.subject_count:
        violations.append("summary subject_count mismatch")
    if sum(item["ecg_count"] for item in split_counts.values()) != int(row[0]):
        violations.append("split ECG counts do not sum to master rows")
    if sum(item["subject_count"] for item in split_counts.values()) != int(
        row[2]
    ):
        violations.append("split subject counts do not sum to unique subjects")
    if violations:
        raise RuntimeError("Master-index integrity failed: " + "; ".join(violations))
    return observed, expected, str(summary_path.resolve())


def _check_event_registry(
    config: MimicProjectConfig,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    registry = load_clinical_event_registry(config)
    if registry is None:
        raise RuntimeError("Clinical-event registry is absent or incomplete")
    records = {
        str(item.get("table_id")): item
        for item in registry.get("tables", ())
    }
    missing = sorted(set(REQUIRED_EVENT_TABLES) - set(records))
    invalid = sorted(
        table_id
        for table_id in REQUIRED_EVENT_TABLES
        if table_id in records and records[table_id].get("status") != "ok"
    )
    if (
        missing
        or invalid
        or registry.get("immutable") is not True
        or registry.get("raw_data_modified") is not False
    ):
        raise RuntimeError(
            "Clinical-event registry failed: "
            f"missing={missing}, invalid={invalid}, "
            f"immutable={registry.get('immutable')}"
        )
    observed = {
        "verified_table_count": len(REQUIRED_EVENT_TABLES),
        "partition_count": registry.get("partition_count"),
        "immutable": True,
        "raw_data_modified": False,
    }
    expected = {
        "required_tables": list(REQUIRED_EVENT_TABLES),
        "immutable": True,
        "raw_data_modified": False,
    }
    return observed, expected, str(registry["_registry_path"])


def _check_event_evidence(
    config: MimicProjectConfig,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    provenance = verify_clinical_event_evidence_provenance(config)
    status = str(provenance.get("status", ""))
    if status not in (
        "direct_event_backed",
        "verified_exact_equivalence",
        "verified_exact_label_relevant_equivalence",
    ):
        raise RuntimeError(
            f"Clinical-event evidence has unsupported status {status!r}"
        )
    if provenance.get("locked_test_associations_read") not in (None, False):
        raise RuntimeError(
            "Event-evidence provenance reports locked-test association access"
        )
    path = (
        config.evidence_root / "evidence_provenance.json"
        if status == "direct_event_backed"
        else (
            config.work_root
            / "provenance"
            / "clinical_event_evidence_equivalence.json"
        )
    )
    observed = {
        "status": status,
        "registry_input_hash": provenance.get(
            "clinical_event_registry_input_hash"
        ),
        "locked_test_associations_read": False,
    }
    expected = {
        "status": [
            "direct_event_backed",
            "verified_exact_equivalence",
            "verified_exact_label_relevant_equivalence",
        ],
        "locked_test_associations_read": False,
    }
    return observed, expected, str(path.resolve())


def _check_laboratory_admission(
    config: MimicProjectConfig,
    expectations: AcceptanceExpectations,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Fail when a laboratory group is mostly missing from the evidence.

    Every window aggregate in the evidence build filters on
    ``value_status = 'usable'``, so a group the unit rule does not recognise
    disappears from the analysis without any error. That is how sodium,
    glucose, urea nitrogen, bicarbonate, leukocytes, albumin, TSH, free T4,
    HbA1c, CRP, D-dimer and the coagulation times came to be absent from a
    screen that reported 727 laboratory targets.

    Rows with no numeric result at all are excluded from the denominator: a
    troponin ordered and never resulted is a property of the hospital, not of
    this pipeline. What is measured here is the share of *numeric* results the
    unit rule refuses.
    """

    labs_path = config.evidence_root / "labs.parquet"
    if not labs_path.is_file():
        raise RuntimeError(f"Laboratory evidence is missing: {labs_path}")
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=min(4.0, float(config.duckdb_memory_limit_gb)),
        threads=max(1, min(4, int(config.workers))),
    )
    query = f"""
        SELECT
            lab_group,
            COUNT(*) AS numeric_rows,
            -- Hemolysis (D2) is a specimen finding, not a unit refusal.
            COUNT(*) FILTER (
                WHERE value_status NOT IN ('usable', 'hemolysis_reported')
            ) AS refused_rows
        FROM read_parquet('{_sql_path(labs_path)}')
        WHERE value_numeric IS NOT NULL
        GROUP BY lab_group
        ORDER BY lab_group
    """
    with catalog.connect(isolated=True) as connection:
        frame = connection.execute(query).df()
    limit = float(expectations.maximum_laboratory_quarantine_share)
    offenders = {}
    for row in frame.itertuples(index=False):
        numeric_rows = int(row.numeric_rows)
        if numeric_rows == 0:
            continue
        share = int(row.refused_rows) / numeric_rows
        if share > limit:
            offenders[str(row.lab_group)] = {
                "numeric_rows": numeric_rows,
                "refused_rows": int(row.refused_rows),
                "refused_share": round(share, 4),
            }
    if offenders:
        raise RuntimeError(
            "Laboratory groups whose numeric results are mostly refused by the "
            f"unit rule: {sorted(offenders)}"
        )
    worst = (
        frame.assign(share=frame["refused_rows"] / frame["numeric_rows"])
        .sort_values("share", ascending=False)
        .head(3)
    )
    observed = {
        "groups": int(len(frame)),
        "numeric_rows": int(frame["numeric_rows"].sum()),
        "refused_rows": int(frame["refused_rows"].sum()),
        "worst_groups": {
            str(row.lab_group): round(float(row.share), 4)
            for row in worst.itertuples(index=False)
        },
    }
    expected = {
        "maximum_refused_share_per_group": limit,
        "groups_above_limit": 0,
    }
    return observed, expected, str(labs_path.resolve())


def _check_echo_admission(
    config: MimicProjectConfig,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Fail when a declared echo column carries no value at all.

    Five columns of the diastolic and aortic extension were entirely empty for
    the life of the project, because the measurement names they asked for do
    not exist in MIMIC's echo vocabulary -- ``septal_e_prime`` instead of
    ``sept_e_prime``, ``av_area`` instead of ``av_area_continuity``, and three
    more. Nothing failed: an empty column and an unmeasured quantity look the
    same downstream, and the tissue-Doppler half of every diastolic endpoint
    was simply absent.

    A declared column that is empty on 206,488 studies is a naming error, not a
    property of the data. Coverage is otherwise not judged here -- a rare
    measurement is allowed to be rare.
    """

    wide_path = config.evidence_root / "echo_measurements_wide.parquet"
    if not wide_path.is_file():
        raise RuntimeError(f"Echo evidence is missing: {wide_path}")
    declared = tuple(echo_derived_diastolic_columns())
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=min(4.0, float(config.duckdb_memory_limit_gb)),
        threads=max(1, min(4, int(config.workers))),
    )
    wide_sql = _sql_path(wide_path)
    with catalog.connect(isolated=True) as connection:
        available = set(parquet_columns(connection, wide_sql))
        missing = sorted(name for name in declared if name not in available)
        if missing:
            raise RuntimeError(
                f"Echo columns declared but never written: {missing}"
            )
        counts = ", ".join(
            f'COUNT("{name}") AS "{name}"' for name in declared
        )
        row = connection.execute(
            f"SELECT COUNT(*) AS studies, {counts} "
            f"FROM read_parquet('{wide_sql}')"
        ).fetchone()
    studies = int(row[0])
    filled = {name: int(value) for name, value in zip(declared, row[1:])}
    empty = sorted(name for name, value in filled.items() if value == 0)
    if empty:
        raise RuntimeError(
            "Echo columns declared and populated on no study at all -- the "
            f"source measurement name is wrong: {empty}"
        )
    thinnest = sorted(filled.items(), key=lambda item: item[1])[:3]
    observed = {
        "studies": studies,
        "declared_columns": len(declared),
        "empty_columns": 0,
        "thinnest_columns": {
            name: round(value / studies, 4) if studies else 0.0
            for name, value in thinnest
        },
    }
    expected = {"empty_columns": 0}
    return observed, expected, str(wide_path.resolve())


def _check_label_attestation(
    config: MimicProjectConfig,
    expectations: AcceptanceExpectations,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = verify_development_label_attestation(config)
    comparisons = payload.get("comparisons") or {}
    mode = str(payload.get("mode", ""))
    generated_current = (
        mode == "generated_by_current_implementation"
        and comparisons.get("status") == "generated_current"
    )
    rule_results = comparisons.get("long_rule_results") or []
    observed = {
        "scope": payload.get("scope"),
        "mode": mode,
        "definition_comparisons": len(rule_results),
        "all_exact": comparisons.get("all_exact"),
        "locked_test_rows_materialized": comparisons.get(
            "locked_test_rows_materialized"
        ),
        "locked_test_read_for_attestation": payload.get(
            "locked_test_read_for_attestation"
        ),
    }
    # Two attestation modes exist. Labels built by the current implementation
    # attest themselves at build time without a rule-by-rule rebuild; only a
    # legacy artifact is re-derived rule by rule (63 comparisons). The check
    # expected the legacy shape for both, so every run whose labels were
    # freshly built failed here -- the production-path preview of 2026-09-19
    # was the first run to reach the audit with such labels.
    expected = {
        "scope": "train_validation_only",
        "mode": mode if generated_current else "legacy_artifact_exact_rebuild",
        "definition_comparisons": (
            0 if generated_current else expectations.label_definition_count
        ),
        "all_exact": True,
        "locked_test_rows_materialized": None if generated_current else False,
        "locked_test_read_for_attestation": False,
    }
    if observed != expected:
        raise RuntimeError(
            f"Development label attestation differs: {observed} != {expected}"
        )
    path = config.labels_root / "development_label_attestation.json"
    return observed, expected, str(path.resolve())


def _check_delineation_release_gate(
    config: MimicProjectConfig,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Re-evaluate the measurement gate instead of trusting a status flag."""

    path = (
        config.resources_root
        / "delineation"
        / "benchmark_suite.json"
    )
    payload = _load_json(path)
    run_configuration = payload.get("run_configuration") or {}
    results = payload.get("results") or {}
    if not isinstance(results, dict):
        raise RuntimeError("Delineation benchmark results are malformed")
    recomputed = evaluate_delineation_release_gate(
        results,
        implementation_hash=str(payload.get("implementation_hash", "")),
        suite_version=str(payload.get("suite_version", "")),
        max_records=run_configuration.get("max_records"),
        max_seconds_per_record=int(
            run_configuration.get("max_seconds_per_record", 120)
        ),
    )
    recorded = payload.get("release_gate") or {}
    current_hash = delineation_implementation_hash()
    observed = {
        "passed": bool(recomputed.get("passed")),
        "recorded_passed": bool(recorded.get("passed")),
        "suite_version": payload.get("suite_version"),
        "release_policy_version": payload.get("release_policy_version"),
        "implementation_hash": payload.get("implementation_hash"),
        "record_capped": bool(run_configuration.get("record_capped")),
        "required_datasets_missing": payload.get(
            "required_datasets_missing",
            [],
        ),
        "failures": recomputed.get("failures", []),
    }
    expected = {
        "passed": True,
        "recorded_passed": True,
        "suite_version": DELINEATION_SUITE_VERSION,
        "release_policy_version": DELINEATION_RELEASE_POLICY_VERSION,
        "implementation_hash": current_hash,
        "record_capped": False,
        "required_datasets_missing": [],
        "failures": [],
    }
    if observed != expected:
        raise RuntimeError(
            "Delineation measurement release gate is not current and passing: "
            f"{observed}"
        )
    if recorded != recomputed:
        raise RuntimeError(
            "Recorded delineation release gate differs from an independent "
            "re-evaluation of its dataset results"
        )
    return observed, expected, str(path.resolve())


def _check_development_features(
    config: MimicProjectConfig,
    run_root: Path,
    expectations: AcceptanceExpectations,
    *,
    deep_verify: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    provenance_path = run_root / "development_provenance.json"
    provenance = _load_json(provenance_path)
    feature_files = [
        Path(str(path)) for path in provenance.get("feature_files", ())
    ]
    if not feature_files:
        raise RuntimeError("Validation provenance has no feature partitions")
    lowered = [str(path).lower() for path in feature_files]
    if any(
        "split=locked_test" in path or "locked-test-" in path
        for path in lowered
    ):
        raise RuntimeError(
            "Development validation references locked-test feature partitions"
        )
    recorded = provenance.get("feature_integrity") or {}
    if deep_verify:
        verified = verify_feature_artifacts(feature_files)
    else:
        verified = recorded
    feature_root = Path(str(verified.get("root", "")))
    feature_glob = feature_root / "split=*" / "part-*.parquet"
    expected_rows, expected_rows_source = _expected_development_feature_rows(
        feature_root,
        expectations,
    )
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=min(2, config.workers),
    )
    with catalog.connect(isolated=True) as connection:
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=_sql_path(feature_glob),
        )
        status_rows = connection.execute(
            f"""
            SELECT extraction_status, COUNT(*) AS record_count
            FROM {read_features}
            GROUP BY extraction_status
            """
        ).fetchall()
    status_counts = {
        str(status): int(count) for status, count in status_rows
    }
    status_total = sum(status_counts.values())
    success_fraction = (
        status_counts.get("ok", 0) / status_total
        if status_total
        else 0.0
    )
    processing_error_fraction = (
        status_counts.get("error", 0) / status_total
        if status_total
        else 1.0
    )
    observed = {
        "feature_scope": provenance.get("feature_scope"),
        "partition_count": int(verified.get("partition_count", -1)),
        "row_count": int(verified.get("row_count", -1)),
        "complete_index": verified.get("complete_index"),
        "artifact_hash": verified.get("artifact_hash"),
        "source_code_hash": verified.get("source_code_hash"),
        "status_counts": status_counts,
        "success_fraction": success_fraction,
        "processing_error_fraction": processing_error_fraction,
        "locked_test_feature_partitions": 0,
    }
    expected = {
        "feature_scope": "development_only",
        "row_count": expected_rows,
        "row_count_source": expected_rows_source,
        "complete_index": True,
        "artifact_hash": "non-empty and integrity-verified",
        "source_code_hash": "current feature implementation",
        "minimum_success_fraction": MINIMUM_FEATURE_SUCCESS_FRACTION,
        "maximum_processing_error_fraction": (
            MAXIMUM_PROCESSING_ERROR_FRACTION
        ),
        "locked_test_feature_partitions": 0,
    }
    if (
        observed["feature_scope"] != "development_only"
        or observed["row_count"] != expected_rows
        or observed["complete_index"] is not True
        or not observed["artifact_hash"]
        or not observed["source_code_hash"]
        or status_total != observed["row_count"]
        or success_fraction < MINIMUM_FEATURE_SUCCESS_FRACTION
        or processing_error_fraction
        > MAXIMUM_PROCESSING_ERROR_FRACTION
        or (
            recorded.get("artifact_hash")
            and observed["artifact_hash"] != recorded.get("artifact_hash")
        )
    ):
        raise RuntimeError(
            f"Development feature artifact is incomplete: {observed}"
        )
    return observed, expected, str(provenance_path.resolve())


def _expected_development_feature_rows(
    feature_root: Path,
    expectations: AcceptanceExpectations,
) -> tuple[int, str]:
    """Explicit expectation, otherwise the extraction's own selected count."""

    if expectations.development_feature_rows is not None:
        return int(expectations.development_feature_rows), "explicit_expectation"
    manifest_path = feature_root / "features.manifest.json"
    manifest = _load_json(manifest_path)
    row_counts = manifest.get("row_counts")
    selected = (
        row_counts.get("selected_ecgs") if isinstance(row_counts, dict) else None
    )
    if type(selected) is not int or selected < 1:
        raise RuntimeError(
            "Extraction manifest records no selected ECG count; the expected "
            f"development feature rows cannot be derived: {manifest_path}"
        )
    return int(selected), f"extraction_manifest:{manifest_path.resolve()}"


def _check_formal_validation(
    run_root: Path,
    expectations: AcceptanceExpectations,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = run_root / "validate.manifest.json"
    provenance_path = run_root / "development_provenance.json"
    spec_path = run_root / "validation_spec.json"
    manifest = StageManifest.load(manifest_path)
    provenance = _load_json(provenance_path)
    spec = _load_json(spec_path)
    payload_verification = verify_validation_input_payload_files(
        provenance.get("validation_input_payload") or {}
    )
    code_verification = verify_validation_code_provenance(run_root)
    missing_outputs = [
        output for output in manifest.output_paths if not Path(output).exists()
    ]
    bootstrap_iterations = int(
        spec.get(
            "development_bootstrap_iterations",
            spec.get("bootstrap_iterations", 0),
        )
    )
    observed = {
        "manifest_status": manifest.status,
        "preview": provenance.get("preview"),
        "analysis_completeness": provenance.get(
            "analysis_completeness"
        ),
        "feature_scope": provenance.get("feature_scope"),
        "locked_test_labels_read": provenance.get(
            "locked_test_labels_read"
        ),
        "bootstrap_iterations": bootstrap_iterations,
        "missing_output_count": len(missing_outputs),
        "input_payload_status": payload_verification.get("status"),
        "validation_code_status": code_verification.get("status"),
        "validation_implementation_hash": code_verification.get(
            "implementation_hash"
        ),
    }
    expected = {
        "manifest_status": "complete",
        "preview": False,
        "analysis_completeness": "complete_snapshot",
        "feature_scope": "development_only",
        "locked_test_labels_read": False,
        "minimum_bootstrap_iterations": (
            expectations.minimum_bootstrap_iterations
        ),
        "missing_output_count": 0,
        "input_payload_status": "verified",
        "validation_code_status": (
            "verified_scoped_implementation or "
            "verified_legacy_scientific_scope_attestation"
        ),
        "validation_implementation_hash": "non-empty",
    }
    if (
        manifest.status != "complete"
        or provenance.get("preview") is not False
        or provenance.get("analysis_completeness") != "complete_snapshot"
        or provenance.get("feature_scope") != "development_only"
        or provenance.get("locked_test_labels_read") is not False
        or bootstrap_iterations < expectations.minimum_bootstrap_iterations
        or missing_outputs
        or payload_verification.get("status") != "verified"
        or code_verification.get("status")
        not in {
            "verified_scoped_implementation",
            "verified_legacy_scientific_scope_attestation",
        }
        or not code_verification.get("implementation_hash")
    ):
        raise RuntimeError(
            f"Formal train/validation provenance failed: {observed}"
        )
    return observed, expected, str(manifest_path.resolve())


def _check_phewas(
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = run_root / "phenome" / "phenome.manifest.json"
    promotion_path = run_root / "phenome" / "promotion_candidates.json"
    manifest = StageManifest.load(manifest_path)
    promotion = _load_json(promotion_path)
    promotion_refresh = verify_phenome_promotion_refresh(run_root)
    notes = {str(item).lower() for item in manifest.notes}
    locked_note = "locked_test_scanned=false" in notes
    missing_outputs = [
        output for output in manifest.output_paths if not Path(output).exists()
    ]
    observed = {
        "manifest_status": manifest.status,
        "locked_test_scanned": promotion.get("locked_test_scanned"),
        "locked_test_guard_note": locked_note,
        "missing_output_count": len(missing_outputs),
        "eligible_phenotypes": int(
            manifest.row_counts.get("eligible_phenotypes", 0)
        ),
        "features": int(manifest.row_counts.get("features", 0)),
        "promotion_refresh_status": (
            "verified" if promotion_refresh.get("valid") else "invalid"
        ),
        "promotion_candidate_count": int(
            promotion_refresh.get("candidate_count", 0)
        ),
        "invalid_promotion_feature_role_counts": (
            promotion_refresh.get("invalid_feature_role_counts", {})
        ),
        "excluded_methodological_top_candidates": int(
            promotion_refresh.get(
                "excluded_methodological_top_candidates",
                0,
            )
        ),
    }
    expected = {
        "manifest_status": "complete",
        "locked_test_scanned": False,
        "locked_test_guard_note": True,
        "missing_output_count": 0,
        "eligible_phenotypes": ">0",
        "features": ">0",
        "promotion_refresh_status": "verified",
        "invalid_promotion_feature_role_counts": {},
    }
    if (
        manifest.status != "complete"
        or promotion.get("locked_test_scanned") is not False
        or not locked_note
        or missing_outputs
        or observed["eligible_phenotypes"] <= 0
        or observed["features"] <= 0
        or promotion_refresh.get("valid") is not True
        or promotion_refresh.get("invalid_feature_role_counts")
    ):
        raise RuntimeError(f"Development PheWAS is incomplete: {observed}")
    return observed, expected, str(manifest_path.resolve())


def _check_group_vcg(
    run_root: Path,
    expectations: AcceptanceExpectations,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    missing: list[str] = []
    incomplete: dict[str, str] = {}
    expected_implementation_hash = group_vcg_implementation_hash()
    for endpoint_id in expectations.group_vcg_endpoints:
        path = (
            run_root
            / "group_vcg"
            / f"endpoint={endpoint_id}"
            / "split=validation"
            / "group_vcg.manifest.json"
        )
        if not path.is_file():
            missing.append(endpoint_id)
            continue
        try:
            _verify_group_vcg_output_set(
                StageManifest.load(path),
                endpoint_id=endpoint_id,
                expected_implementation_hash=expected_implementation_hash,
            )
        except (
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            incomplete[endpoint_id] = str(exc)
    if (
        missing
        and not incomplete
        and len(missing) == len(expectations.group_vcg_endpoints)
        and not (run_root / "group_vcg").is_dir()
    ):
        raise PendingArtifact(
            "Group VCG visualizations are produced by the group-vcg action "
            f"after the study; none exist yet for {list(missing)}",
            observed={"completed_endpoint_count": 0},
            expected={
                "completed_endpoint_count": len(expectations.group_vcg_endpoints),
                "split": "validation",
            },
        )
    if missing or incomplete:
        raise RuntimeError(
            f"Group VCG artifacts missing={missing}, incomplete={incomplete}"
        )
    observed = {
        "completed_endpoint_count": len(expectations.group_vcg_endpoints),
        "endpoints": list(expectations.group_vcg_endpoints),
        "split": "validation",
    }
    expected = {
        "completed_endpoint_count": len(expectations.group_vcg_endpoints),
        "split": "validation",
        "locked_test_used": False,
    }
    return observed, expected, str((run_root / "group_vcg").resolve())


def _verify_group_vcg_output_set(
    manifest: StageManifest,
    *,
    endpoint_id: str,
    expected_implementation_hash: str,
) -> None:
    notes = {str(note).strip().lower() for note in manifest.notes}
    outputs_by_name = {
        Path(output).name: Path(output) for output in manifest.output_paths
    }
    required_outputs = {
        "group_vcg_summary.parquet",
        "group_vcg_difference.parquet",
        "group_vcg_case_audit.parquet",
        "group_vcg_provenance.json",
    }
    positive_count = int(manifest.row_counts.get("included_positive", 0))
    negative_count = int(manifest.row_counts.get("included_negative", 0))
    time_points = int(manifest.row_counts.get("time_points", 0))
    bootstrap_iterations = int(
        manifest.row_counts.get("bootstrap_iterations", 0)
    )
    if (
        manifest.status != "complete"
        or "locked_test_read=false" not in notes
        or "scope=development_only" not in notes
        or not required_outputs.issubset(outputs_by_name)
        or any(not Path(output).is_file() for output in manifest.output_paths)
        or positive_count < 10
        or negative_count < 10
        or time_points <= 0
        or bootstrap_iterations < 500
    ):
        raise RuntimeError("manifest or required outputs are incomplete")

    provenance = _load_json(outputs_by_name["group_vcg_provenance.json"])
    class_counts = provenance.get("class_counts") or {}
    event_provenance = (
        provenance.get("clinical_event_evidence_provenance") or {}
    )
    if (
        provenance.get("stage_version") != manifest.stage_version
        or provenance.get("implementation_hash")
        != expected_implementation_hash
        or provenance.get("input_hash") != manifest.input_hash
        or provenance.get("endpoint_id") != endpoint_id
        or provenance.get("split") != "validation"
        or provenance.get("locked_test_read") is not False
        or provenance.get("feature_snapshot_verification", {}).get("valid")
        is not True
        or event_provenance.get("status")
        not in {
            "direct_event_backed",
            "verified_exact_equivalence",
            "verified_exact_label_relevant_equivalence",
        }
        or int(class_counts.get("positive", -1)) != positive_count
        or int(class_counts.get("negative", -1)) != negative_count
    ):
        raise RuntimeError("provenance does not match the development manifest")

    summary = pd.read_parquet(
        outputs_by_name["group_vcg_summary.parquet"]
    )
    difference = pd.read_parquet(
        outputs_by_name["group_vcg_difference.parquet"]
    )
    audit = pd.read_parquet(
        outputs_by_name["group_vcg_case_audit.parquet"]
    )
    shared_columns = {
        "endpoint_id",
        "split",
        "sample_index",
        "time_ms",
        "feature_snapshot_hash",
        "artifact_completeness",
        "exploration_grade",
    }
    summary_columns = shared_columns | {
        "label_state",
        "n_subjects",
        "median_x_mV",
        "median_y_mV",
        "median_z_mV",
    }
    difference_columns = shared_columns | {
        "delta_x_mV",
        "delta_y_mV",
        "delta_z_mV",
        "delta_magnitude_mV",
    }
    audit_columns = {
        "subject_id",
        "study_id",
        "endpoint_id",
        "label_state",
        "split",
        "status",
    }
    if (
        not summary_columns.issubset(summary.columns)
        or not difference_columns.issubset(difference.columns)
        or not audit_columns.issubset(audit.columns)
    ):
        raise RuntimeError("group VCG output schema is incomplete")
    if len(summary) != 2 * time_points or len(difference) != time_points:
        raise RuntimeError("group VCG row counts do not match the manifest")
    if set(summary["label_state"].astype(str)) != {"positive", "negative"}:
        raise RuntimeError("group VCG summary lacks one reference group")
    summary_counts = summary["label_state"].astype(str).value_counts()
    if any(
        int(summary_counts.get(state, 0)) != time_points
        for state in ("positive", "negative")
    ):
        raise RuntimeError("group VCG reference groups have unequal timelines")
    if (
        not difference["sample_index"].is_unique
        or any(
            not group["sample_index"].is_unique
            for _, group in summary.groupby("label_state")
        )
    ):
        raise RuntimeError("group VCG timelines contain duplicate samples")
    for frame in (summary, difference):
        if (
            set(frame["endpoint_id"].astype(str)) != {endpoint_id}
            or set(frame["split"].astype(str)) != {"validation"}
            or set(frame["artifact_completeness"].astype(str)) != {"complete"}
            or set(frame["exploration_grade"].astype(str))
            != {"descriptive_development"}
        ):
            raise RuntimeError("group VCG table metadata is inconsistent")
    numeric_columns = (
        "median_x_mV",
        "median_y_mV",
        "median_z_mV",
    )
    if not np.isfinite(
        summary.loc[:, numeric_columns].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("group VCG summary contains missing VCG coordinates")
    delta_columns = (
        "delta_x_mV",
        "delta_y_mV",
        "delta_z_mV",
        "delta_magnitude_mV",
    )
    if not np.isfinite(
        difference.loc[:, delta_columns].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("group VCG difference contains missing coordinates")
    expected_subject_counts = {
        "positive": positive_count,
        "negative": negative_count,
    }
    for state, group in summary.groupby("label_state"):
        if set(group["n_subjects"].astype(int)) != {
            expected_subject_counts[str(state)]
        }:
            raise RuntimeError("group VCG subject counts are inconsistent")

    included = audit.loc[audit["status"].astype(str).eq("included")]
    included_counts = included["label_state"].astype(str).value_counts()
    if (
        set(audit["endpoint_id"].astype(str)) != {endpoint_id}
        or set(audit["split"].astype(str)) != {"validation"}
        or int(included_counts.get("positive", 0)) != positive_count
        or int(included_counts.get("negative", 0)) != negative_count
        or included["subject_id"].duplicated().any()
        or included["study_id"].duplicated().any()
    ):
        raise RuntimeError("group VCG case audit does not match class counts")


def _check_reports(
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    reports_root = run_root / "reports"
    manifest_path = reports_root / "report.manifest.json"
    manifest = StageManifest.load(manifest_path)
    missing = []
    for report_name in REQUIRED_CANONICAL_REPORTS:
        for suffix in (".parquet", ".csv"):
            path = reports_root / f"{report_name}{suffix}"
            if not path.is_file():
                missing.append(path.name)
    convenience = (
        reports_root / f"{run_root.name}_report.html",
        reports_root / f"{run_root.name}_reports.xlsx",
    )
    missing.extend(path.name for path in convenience if not path.is_file())
    build_manifests = sorted(
        (reports_root / "builds").glob("*/report_build.manifest.json")
    )
    valid_print_builds = [
        str(build_manifest_path.parent.resolve())
        for build_manifest_path in build_manifests
        if _is_complete_anonymized_print_build(build_manifest_path)
    ]
    missing_manifest_outputs = [
        output for output in manifest.output_paths if not Path(output).exists()
    ]
    clinical_summary_path = reports_root / "clinical_utility_summary.parquet"
    clinical_summary: Optional[pd.Series] = None
    clinical_error = ""
    if clinical_summary_path.is_file():
        frame = pd.read_parquet(clinical_summary_path)
        if len(frame) != 1:
            clinical_error = "clinical utility summary must contain one row"
        else:
            clinical_summary = frame.iloc[0]
            process_claim = bool(
                clinical_summary.get(
                    "decision_process_utility_demonstrated", False
                )
            )
            patient_claim = bool(
                clinical_summary.get("patient_benefit_demonstrated", False)
            )
            if patient_claim and not process_claim:
                clinical_error = "patient benefit cannot precede process utility"
            if not str(clinical_summary.get("disclaimer", "")).strip():
                clinical_error = "clinical utility disclaimer is missing"
            if process_claim or patient_claim:
                from .reports import resolve_clinical_utility_report_paths

                if resolve_clinical_utility_report_paths(run_root) is None:
                    clinical_error = (
                        "clinical utility claim has no verified assessment bundle"
                    )
    if (
        manifest.status == "complete"
        and not missing
        and not missing_manifest_outputs
        and not clinical_error
        and not build_manifests
    ):
        raise PendingArtifact(
            "Canonical reports are complete; the anonymized print build is "
            "produced by the report-build action after the study",
            observed={"canonical_reports": "complete", "print_builds": 0},
            expected={"canonical_reports": "complete", "print_builds": ">= 1"},
        )
    if (
        manifest.status != "complete"
        or missing
        or missing_manifest_outputs
        or not valid_print_builds
        or clinical_error
    ):
        raise RuntimeError(
            "Report acceptance failed: "
            f"status={manifest.status}, missing={missing}, "
            f"missing_manifest_outputs={len(missing_manifest_outputs)}, "
            f"valid_print_builds={len(valid_print_builds)}"
            f", clinical_utility_error={clinical_error!r}"
        )
    observed = {
        "canonical_report_tables": len(REQUIRED_CANONICAL_REPORTS),
        "canonical_formats": ["parquet", "csv", "html", "xlsx"],
        "anonymized_print_builds": len(valid_print_builds),
        "print_formats": ["pdf", "html", "xlsx"],
        "clinical_utility_claim_level": (
            str(clinical_summary.get("claim_level", "hypothesis_only"))
            if clinical_summary is not None
            else "not_available"
        ),
        "decision_process_utility_demonstrated": (
            bool(
                clinical_summary.get(
                    "decision_process_utility_demonstrated", False
                )
            )
            if clinical_summary is not None
            else False
        ),
        "patient_benefit_demonstrated": (
            bool(clinical_summary.get("patient_benefit_demonstrated", False))
            if clinical_summary is not None
            else False
        ),
    }
    expected = {
        "canonical_report_tables": list(REQUIRED_CANONICAL_REPORTS),
        "anonymized_print_builds": ">=1",
        "case_ids_by_default": False,
    }
    return observed, expected, str(manifest_path.resolve())


def _is_complete_anonymized_print_build(
    manifest_path: Path,
) -> bool:
    try:
        manifest = StageManifest.load(manifest_path)
        metadata = _load_json(manifest_path.parent / "report_spec.json")
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    specification = metadata.get("report_spec")
    provenance = metadata.get("provenance")
    if not isinstance(specification, dict) or not isinstance(provenance, dict):
        return False
    if (
        manifest.status != "complete"
        or specification.get("include_case_ids") is not False
        or specification.get("include_case_appendix") is not False
        or provenance.get("analysis_status")
        not in {"complete_development", "frozen_development"}
        or provenance.get("artifact_completeness") != "complete"
        or provenance.get("locked_test_status") != "sealed_not_evaluated"
        or any(not Path(output).is_file() for output in manifest.output_paths)
    ):
        return False
    output_suffixes = {
        Path(output).suffix.lower() for output in manifest.output_paths
    }
    return {".pdf", ".html", ".xlsx"}.issubset(output_suffixes)


def _check_ptb_lock(
    config: MimicProjectConfig,
    ptb_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = config.work_root / "ptb_revalidation" / ptb_run_id
    freeze_path = root / "ptb_freeze_manifest.json"
    lock_path = root / "ptb_test_lock.json"
    provenance_path = root / "ptb_development_provenance.json"
    freeze = _load_json(freeze_path)
    lock = _load_json(lock_path)
    provenance = _load_json(provenance_path)
    payload_hash = content_hash(freeze.get("payload"))
    folds_read = [int(item) for item in provenance.get("folds_read", ())]
    forbidden_outputs = sorted(
        path.name
        for path in root.glob("*fold*10*")
        if path.name not in {
            "ptb_freeze_manifest.json",
            "ptb_test_lock.json",
        }
    )
    observed = {
        "freeze_hash_matches_payload": payload_hash
        == freeze.get("freeze_hash")
        == lock.get("freeze_hash"),
        "implementation_hash_current": (
            freeze.get("payload", {}).get("code_hash")
            == ptb_implementation_hash()
        ),
        "folds_read": folds_read,
        "locked_fold_10_read": provenance.get("locked_fold_10_read"),
        "ptb_fold_10_status": freeze.get("ptb_fold_10_status"),
        "evaluation_started_at": lock.get("evaluation_started_at"),
        "consumed_at": lock.get("consumed_at"),
        "invalidated_at": lock.get("invalidated_at"),
        "forbidden_output_count": len(forbidden_outputs),
    }
    expected = {
        "freeze_hash_matches_payload": True,
        "implementation_hash_current": True,
        "folds_read": list(range(1, 10)),
        "locked_fold_10_read": False,
        "ptb_fold_10_status": "sealed_unread",
        "evaluation_started_at": None,
        "consumed_at": None,
        "invalidated_at": None,
        "forbidden_output_count": 0,
    }
    if observed != expected:
        seal_intact = {
            key: value for key, value in observed.items()
            if key != "implementation_hash_current"
        } == {
            key: value for key, value in expected.items()
            if key != "implementation_hash_current"
        }
        if seal_intact and not observed["implementation_hash_current"]:
            raise PendingArtifact(
                "PTB fold-10 seal is intact but the revalidation was frozen "
                "with an earlier marker implementation; repeat the PTB "
                "transfer with the current one before the final audit",
                observed=observed,
                expected=expected,
            )
        raise RuntimeError(f"PTB fold-10 seal failed: {observed}")
    return observed, expected, str(lock_path.resolve())


def _check_mimic_lock(
    run_root: Path,
    *,
    require_freeze: bool,
) -> AcceptanceCheck:
    freeze_path = run_root / "freeze_manifest.json"
    lock_path = run_root / "test_lock.json"
    expected = {
        "locked_test_status": "sealed_unread",
        "evaluation_started_at": None,
        "consumed_at": None,
        "invalidated_at": None,
        "root_locked_test_results": "absent",
    }
    if not freeze_path.is_file() and not lock_path.is_file():
        if require_freeze:
            return AcceptanceCheck(
                check_id="mimic_locked_test_sealed",
                category="test_lock",
                status="fail",
                observed="freeze and test lock are absent",
                expected=expected,
                evidence="",
                detail="A final acceptance audit requires the new freeze.",
            )
        return AcceptanceCheck(
            check_id="mimic_locked_test_sealed",
            category="test_lock",
            status="pending",
            observed="development checks complete; freeze not created yet",
            expected=expected,
            evidence="",
            detail=(
                "This is the intended pre-freeze state. Run freeze only after "
                "all other checks pass, then repeat with require_freeze."
            ),
        )
    try:
        freeze = _load_json(freeze_path)
        lock = _load_json(lock_path)
        root_results = [
            path.name
            for path in (
                run_root / "final_test_report.parquet",
                run_root / "evaluate_test.manifest.json",
            )
            if path.exists()
        ]
        observed = {
            "freeze_hash_matches_payload": (
                content_hash(freeze.get("payload"))
                == freeze.get("freeze_hash")
                == lock.get("freeze_hash")
            ),
            "locked_test_status": freeze.get("locked_test_status"),
            "evaluation_started_at": lock.get("evaluation_started_at"),
            "consumed_at": lock.get("consumed_at"),
            "invalidated_at": lock.get("invalidated_at"),
            "root_locked_test_results": root_results,
            "locked_test_outcomes_read": False,
        }
        valid = (
            observed["freeze_hash_matches_payload"] is True
            and observed["locked_test_status"] == "sealed_unread"
            and observed["evaluation_started_at"] is None
            and observed["consumed_at"] is None
            and observed["invalidated_at"] is None
            and not root_results
        )
        return AcceptanceCheck(
            check_id="mimic_locked_test_sealed",
            category="test_lock",
            status="pass" if valid else "fail",
            observed=observed,
            expected=expected,
            evidence=str(lock_path.resolve()),
            detail="" if valid else "MIMIC locked-test seal is not pristine.",
        )
    except Exception as exc:
        return AcceptanceCheck(
            check_id="mimic_locked_test_sealed",
            category="test_lock",
            status="fail",
            observed={"error_type": type(exc).__name__},
            expected=expected,
            evidence="",
            detail=str(exc),
        )


def _checks_frame(checks: list[AcceptanceCheck]) -> pd.DataFrame:
    rows = []
    for order, check in enumerate(checks, start=1):
        rows.append(
            {
                "order": order,
                "check_id": check.check_id,
                "category": check.category,
                "status": check.status,
                "observed_json": json.dumps(
                    check.observed,
                    ensure_ascii=True,
                    sort_keys=True,
                    default=str,
                ),
                "expected_json": json.dumps(
                    check.expected,
                    ensure_ascii=True,
                    sort_keys=True,
                    default=str,
                ),
                "evidence": check.evidence,
                "detail": check.detail,
            }
        )
    return pd.DataFrame(rows)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Audit artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Audit artifact is not a JSON object: {path}")
    return payload


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")
