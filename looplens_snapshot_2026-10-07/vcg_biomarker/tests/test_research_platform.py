from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from vcg_biomarker.mimic_job_worker import (
    _apply_job_resource_environment,
    _finalize_progress,
    _runtime_config_for_job,
)
from vcg_biomarker.core.data_loader import LEAD_NAMES_12
from vcg_biomarker.core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
)
from vcg_biomarker.core.external_loader import (
    ECGImportMappingRequired,
    ExternalECGMeta,
    load_external_ecg_result,
)
from vcg_biomarker.mimic.association_domains import (
    ContextAssociationDefinition,
    _build_association_index,
    _overlay_context_intervals,
    _screen_association_domains,
    default_context_association_definitions,
)
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.baseline_registry import BaselineRegistry, _rule
from vcg_biomarker.mimic.discovery_statistics import (
    add_auc_replication_status,
    add_effect_replication_status,
    discovery_marker_bootstrap_intervals,
)
from vcg_biomarker.mimic.ecg_lab import (
    ECGCaseLibrary,
    _accepted_beat_wave_segments,
    _apply_annotation_revision,
    _apply_single_case_score,
    _release_component_names,
    _source_bundle_fingerprint,
    _source_bundle_paths,
)
from vcg_biomarker.mimic.event_ingest import (
    clinical_event_quarantine_relation,
    clinical_event_relation,
    ingest_clinical_event_tables,
    load_clinical_event_registry,
)
from vcg_biomarker.mimic.event_equivalence import (
    _classify_evidence_table_scope,
    _compare_or_resume_medication_windows,
    _compare_or_resume_table,
    _load_equivalence_checkpoint,
    verify_clinical_event_evidence_provenance,
)
from vcg_biomarker.mimic.evidence import (
    EvidenceArtifact,
    _ecg_echo_match_query,
    _load_evidence_checkpoint,
    _medication_order_administration_context_query,
    _resume_evidence_group,
    _save_evidence_group,
)
from vcg_biomarker.mimic.feature_extraction import (
    FeatureExtractionOptions,
    _DynamicWorkerHysteresis,
    _AdmissionPool,
    _cohort_record_payloads,
    _load_feature_cohort,
    _load_resume_checkpoint,
    _partition_reusable_for_execution,
    _write_resume_checkpoint,
    current_development_feature_input_hash,
    feature_implementation_hash,
    feature_stage_input_payload,
    resolve_feature_panel_selection,
)
from vcg_biomarker.mimic.feature_record import (
    _finalize_biomarker_failure_provenance,
    _store_biomarker_results,
)
from vcg_biomarker.mimic.frame_checkpoints import (
    checkpointed_frame_chunk,
)
from vcg_biomarker.mimic.jobs import (
    JOB_PROJECT_CONFIG_HASH_KEY,
    JOB_PROJECT_CONFIG_KEY,
    JobControl,
    JobPauseRequested,
    JobStore,
    resource_profile_for_config,
)
from vcg_biomarker.mimic.locking import (
    FROZEN_DEVELOPMENT_TABLES,
    _locked_report_record,
    _update_score_release_stage,
)
from vcg_biomarker.mimic.methodology_registry import MethodologyRegistry
from vcg_biomarker.mimic.models import (
    AnnotationRevision,
    ArtifactSnapshot,
    JobSpec,
    JobState,
    ParameterPolicy,
    ReportSpec,
    ReportType,
    ResourceProfile,
    ScoreReleaseStage,
    ValidationSpec,
)
from vcg_biomarker.mimic.optimization import grouped_score_search
from vcg_biomarker.mimic.phenome import (
    PhenomeDiscoveryOptions,
    _auc_and_p_from_sorted_controls,
    _broad_discovery_table,
    _build_phenome_cohorts,
    _case_control_values,
    _checkpointed_train_screen,
    _checkpointed_validation_replication,
    _overlay_replicated_bootstraps,
    _read_parquet_columns,
    _replicate_hits,
    _replicated_bootstrap_hits,
    _promotion_candidates,
    _screen_split,
)
from vcg_biomarker.mimic.future_phenome import (
    _fit_cause_specific_cox,
    _auc_and_p as _future_auc_and_p,
    _checkpointed_future_replication,
    _future_replicated_bootstrap_hits,
    _future_label_summary,
    _future_state_masks,
    _overlay_future_bootstraps,
    _replicate_future_hits,
)
from vcg_biomarker.mimic.phenotypes import _derived_outputs_match
from vcg_biomarker.mimic.phenome_promotions import (
    refresh_phenome_promotions,
    verify_phenome_promotion_refresh,
)
from vcg_biomarker.mimic.presentation import comparison_effect_frame
from vcg_biomarker.mimic.presets import (
    analysis_presets,
    get_analysis_preset,
)
from vcg_biomarker.mimic.report_builder import (
    _excel_export_tables,
    _excel_table_chunks,
    _formal_candidate_marker_rows,
    build_configured_report,
)
from vcg_biomarker.mimic.feature_roles import feature_analysis_role
from vcg_biomarker.mimic.reports import (
    _run_feature_failure_path,
    resolve_acceptance_audit_paths,
)
from vcg_biomarker.mimic.resources import (
    HardwareInventory,
    MemoryStatus,
    apply_worker_thread_limits,
    cpu_token_capacity,
    critical_available_memory_gb,
    detect_hardware,
    global_memory_budget_gb,
    launch_available_memory_gb,
    recommend_dynamic_workers,
)
from vcg_biomarker.mimic.snapshots import (
    _is_complete,
    create_artifact_snapshot,
    verify_artifact_snapshot,
)
from vcg_biomarker.mimic.statistics import (
    _weighted_bootstrap_metric_batch,
    classification_metrics,
    exact_pr_auc,
    fit_marker_model,
    marker_bootstrap_intervals,
    paired_operating_point_bootstrap,
)
from vcg_biomarker.mimic.transfer import _build_transfer_matrix
from vcg_biomarker.mimic.supervisor import (
    MEMORY_PRESSURE_SAMPLES,
    MEMORY_READY_SAMPLES,
    LocalJobSupervisor,
    _is_process_pool_child_of,
)
from vcg_biomarker.mimic.storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    file_content_fingerprint,
    file_fingerprint,
    file_sha256,
    scientific_config_hash,
)
from vcg_biomarker.mimic.validation import (
    VALIDATION_INPUT_PAYLOAD_VERSION,
    _build_operating_point_results,
    _comparison_verdict,
    _fixed_reference_curve_points,
    _load_or_evaluate_marker_chunks,
    _machine_report_records,
    _paired_primary_baseline_comparison,
    _prevalence_metrics_allowed,
    _subgroup_stability_records,
    _probability_models_allowed,
    _suppress_prevalence_dependent_metrics,
    _suppress_prevalence_metrics_in_payload,
    _validation_json_value,
    _prepare_validation_feature_cache,
    _verify_current_development_feature_inputs,
    _visualization_records_for_marker,
    verify_validation_input_payload_files,
    PRIMARY_CLASSICAL_BASELINES,
)


def _config(tmp_path: Path) -> MimicProjectConfig:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "structured.csv",
        echo_study_list_path=tmp_path / "study.csv",
        echo_record_list_path=tmp_path / "record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    return config


def test_baseline_registry_covers_every_historical_primary_classical_baseline() -> (
    None
):
    """Regression guard for the paired-comparison generalization
    (Work Package A1): run_development_validation now derives the
    classical comparators for a marker purely from
    BaselineRegistry.for_endpoint(...), not from the older hardcoded
    PRIMARY_CLASSICAL_BASELINES mapping. Every endpoint that mapping
    already asserted a classical comparator for must still have that
    exact criterion registered, or be one of the below, deliberately
    documented exceptions — otherwise an endpoint would silently lose
    its guideline comparison."""
    registry = BaselineRegistry()
    # hypokalemia_near_ecg / severe_hypokalemia_near_ecg: baseline_registry.py
    # has an explicit, cited "no established stand-alone ECG rule for
    # hypokalemia" entry (BaselineKind.NONE_ESTABLISHED) that deliberately
    # supersedes the older qtc_fridericia proxy used here.
    # hypomagnesemia_near_ecg / hypocalcemia_near_ecg: not yet reviewed in
    # baseline_registry.py at all; asserting a classical comparator for
    # them is a clinical-content decision outside the scope of this
    # generalization, so they are left unregistered (and therefore
    # correctly skipped, not compared) rather than guessed at.
    known_unregistered = {
        "hypokalemia_near_ecg",
        "severe_hypokalemia_near_ecg",
        "hypomagnesemia_near_ecg",
        "hypocalcemia_near_ecg",
    }
    missing = []
    for endpoint_id, old_baseline_feature_id in (
        PRIMARY_CLASSICAL_BASELINES.items()
    ):
        if endpoint_id in known_unregistered:
            continue
        old_criterion_id = old_baseline_feature_id.removeprefix("classical_")
        registered = {
            item.criterion_id
            for item in registry.for_endpoint(endpoint_id)
            if item.criterion_id
        }
        if old_criterion_id not in registered:
            missing.append((endpoint_id, old_criterion_id))
    assert missing == []


def test_partitioned_clinical_event_ingest_is_immutable_and_typed(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    hosp = config.mimic_iv_root / "hosp"
    hosp.mkdir(parents=True)
    labevents = pd.DataFrame(
        {
            "labevent_id": ["1", "2", "invalid"],
            "subject_id": ["100", "101", "102"],
            "hadm_id": ["10", "11", "12"],
            "specimen_id": ["20", "21", "22"],
            "itemid": ["51003", "51003", "51003"],
            "order_provider_id": ["P1", "P2", "P3"],
            "charttime": [
                "2024-01-01 10:00:00",
                "2024-01-02 11:00:00",
                "2024-01-03 12:00:00",
            ],
            "storetime": [
                "2024-01-01 10:05:00",
                "2024-01-02 11:05:00",
                "2024-01-03 12:05:00",
            ],
            "value": ["5.0", "6.0", "7.0"],
            "valuenum": ["5.0", "6.0", "7.0"],
            "valueuom": ["ng/L", "ng/L", "ng/L"],
            "ref_range_lower": ["0", "0", "0"],
            "ref_range_upper": ["14", "14", "14"],
            "flag": ["", "", ""],
            "priority": ["ROUTINE", "ROUTINE", "ROUTINE"],
            "comments": ["", "", ""],
        }
    )
    labevents.to_csv(
        hosp / "labevents.csv.gz",
        index=False,
        compression="gzip",
    )

    manifest = ingest_clinical_event_tables(config)
    relation = clinical_event_relation(config, "labevents")
    quarantine_relation = clinical_event_quarantine_relation(
        config,
        "labevents",
    )
    assert relation is not None
    assert quarantine_relation is not None
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"SELECT subject_id, valuenum FROM {relation} "
            "ORDER BY subject_id"
        ).fetchall()
        quarantined = connection.execute(
            f"SELECT subject_id, _diagnostic FROM {quarantine_relation}"
        ).fetchall()
    finally:
        connection.close()

    assert manifest.row_counts["labevents"] == 2
    assert manifest.row_counts["labevents_quarantine"] == 1
    assert rows == [(100, 5.0), (101, 6.0)]
    assert quarantined == [
        ("102", "invalid required key or timestamp")
    ]
    assert ingest_clinical_event_tables(config).input_hash == manifest.input_hash
    assert len(pd.read_csv(hosp / "labevents.csv.gz")) == 3
    registry = load_clinical_event_registry(config)
    assert registry is not None
    atomic_write_json(
        config.evidence_root / "evidence_provenance.json",
        {
            "clinical_event_source_policy": (
                "verified_partitioned_parquet"
            ),
            "clinical_event_registry": {
                "input_hash": registry["input_hash"]
            },
        },
    )
    verified = verify_clinical_event_evidence_provenance(config)
    assert verified["status"] == "direct_event_backed"
    assert (
        verified["clinical_event_registry_input_hash"]
        == registry["input_hash"]
    )


def test_clinical_event_ingest_resumes_after_atomic_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import event_ingest

    config = _config(tmp_path)
    hosp = config.mimic_iv_root / "hosp"
    hosp.mkdir(parents=True)
    row_count = 500
    labevents = pd.DataFrame(
        {
            "labevent_id": [str(index + 1) for index in range(row_count)],
            "subject_id": [
                str(100 + index % 17) for index in range(row_count)
            ],
            "hadm_id": ["10"] * row_count,
            "specimen_id": ["20"] * row_count,
            "itemid": ["51003"] * row_count,
            "order_provider_id": ["P1"] * row_count,
            "charttime": ["2024-01-01 10:00:00"] * row_count,
            "storetime": ["2024-01-01 10:05:00"] * row_count,
            "value": ["5.0"] * row_count,
            "valuenum": ["5.0"] * row_count,
            "valueuom": ["ng/L"] * row_count,
            "ref_range_lower": ["0"] * row_count,
            "ref_range_upper": ["14"] * row_count,
            "flag": [""] * row_count,
            "priority": ["ROUTINE"] * row_count,
            "comments": ["checkpoint-test"] * row_count,
        }
    )
    source = hosp / "labevents.csv.gz"
    labevents.to_csv(source, index=False, compression="gzip")
    raw_hash = file_sha256(source)
    monkeypatch.setattr(event_ingest, "EVENT_BATCH_BLOCK_BYTES", 2048)

    def pause_after_first_batch(message: str) -> None:
        if "Batch 1 atomar abgeschlossen" in message:
            raise RuntimeError("simulated supervisor stop")

    with pytest.raises(RuntimeError, match="simulated supervisor stop"):
        ingest_clinical_event_tables(
            config,
            progress=pause_after_first_batch,
        )

    work_roots = list(
        (config.standardized_root / "clinical_events" / "versions").glob(
            ".work-schema-*"
        )
    )
    assert len(work_roots) == 1
    checkpoint = json.loads(
        (work_roots[0] / "ingest_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(
        checkpoint["tables"]["labevents"]["completed_batches"]
    ) == {"0"}
    assert (
        checkpoint["tables"]["labevents"]["source_staging"][
            "stream_complete"
        ]
        is True
    )

    def compressed_source_must_not_be_reopened(*_args, **_kwargs):
        raise AssertionError(
            "completed source staging must make transform resume seekable"
        )

    monkeypatch.setattr(
        event_ingest,
        "_open_event_batch_reader",
        compressed_source_must_not_be_reopened,
    )

    resumed_messages: list[str] = []
    manifest = ingest_clinical_event_tables(
        config,
        progress=resumed_messages.append,
    )
    assert manifest.row_counts["labevents"] == row_count
    assert not any(
        "Batch 1 atomar abgeschlossen" in message
        for message in resumed_messages
    )
    assert any(
        "Batch 2 atomar abgeschlossen" in message
        for message in resumed_messages
    )
    assert file_sha256(source) == raw_hash
    assert not list(
        (config.standardized_root / "clinical_events" / "versions").glob(
            ".work-schema-*"
        )
    )


def test_validation_feature_cache_compacts_and_resumes_by_split(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "features" / "development"
    partitions = []
    source_hashes = {}
    for split, subject_ids in (
        ("train", [1, 2, 3]),
        ("validation", [4, 5]),
    ):
        path = source_root / f"split={split}" / "part-000000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "subject_id": subject_ids,
                "study_id": [value * 10 for value in subject_ids],
                "split": [split] * len(subject_ids),
                "ecg_time": ["2024-01-01"] * len(subject_ids),
                "marker_a": np.arange(len(subject_ids), dtype=float),
            }
        ).to_parquet(path, index=False)
        source_hashes[str(path)] = file_sha256(path)
        partitions.append(
            {
                "relative_path": path.relative_to(
                    source_root
                ).as_posix(),
                "row_count": len(subject_ids),
            }
        )
    index_path = source_root / "partition_index.json"
    atomic_write_json(index_path, {"partitions": partitions})
    integrity = {
        "root": str(source_root),
        "index_path": str(index_path),
        "artifact_hash": "artifact-123",
        "partition_count": 2,
        "row_count": 5,
    }
    run_root = tmp_path / "run"

    def pause_after_train(message: str) -> None:
        if "train atomar abgeschlossen" in message:
            raise RuntimeError("simulated validation pause")

    with pytest.raises(RuntimeError, match="simulated validation pause"):
        _prepare_validation_feature_cache(
            catalog_path=tmp_path / "catalog.duckdb",
            memory_limit_gb=1,
            threads=1,
            run_root=run_root,
            feature_integrity=integrity,
            feature_snapshot_hash="snapshot-123",
            progress=pause_after_train,
        )

    resumed_messages: list[str] = []
    feature_glob, payload, manifest_path = (
        _prepare_validation_feature_cache(
            catalog_path=tmp_path / "catalog.duckdb",
            memory_limit_gb=1,
            threads=1,
            run_root=run_root,
            feature_integrity=integrity,
            feature_snapshot_hash="snapshot-123",
            progress=resumed_messages.append,
        )
    )
    frame = duckdb.connect().execute(
        "SELECT subject_id, split FROM read_parquet("
        "?, hive_partitioning=true) ORDER BY subject_id",
        [feature_glob],
    ).fetchdf()
    assert payload["row_count"] == 5
    assert manifest_path.is_file()
    assert frame["subject_id"].tolist() == [1, 2, 3, 4, 5]
    assert frame["split"].tolist() == [
        "train",
        "train",
        "train",
        "validation",
        "validation",
    ]
    assert not any(
        "train atomar abgeschlossen" in message
        for message in resumed_messages
    )
    assert all(
        file_sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    )


def test_validation_marker_chunks_resume_without_recalculation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import validation as validation_module

    train = pd.DataFrame(
        {
            "subject_id": np.arange(20),
            "label": np.repeat([0, 1], 10),
            "marker_a": np.r_[np.arange(10), np.arange(10) + 20],
            "marker_b": np.r_[np.arange(10) + 20, np.arange(10)],
        }
    )
    validation = train.copy()
    specification = ValidationSpec(
        run_id="checkpoint-test",
        split_hash="split-hash",
        endpoint_ids=("endpoint",),
        panel_versions={"primary_core": "1"},
        development_bootstrap_iterations=0,
        random_seed=17,
    )
    calls: list[tuple[str, ...]] = []
    original = validation_module._evaluate_endpoint_markers

    def record_call(**kwargs):
        calls.append(tuple(kwargs["feature_ids"]))
        return original(**kwargs)

    monkeypatch.setattr(
        validation_module,
        "_evaluate_endpoint_markers",
        record_call,
    )

    def pause_after_first_chunk(message: str) -> None:
        if "Marker 2/2 atomar" in message:
            raise RuntimeError("simulated marker pause")

    arguments = {
        "endpoint_id": "endpoint",
        "feature_ids": ("marker_a", "marker_b"),
        "panel_map": {
            "marker_a": "primary_core",
            "marker_b": "primary_core",
        },
        "train": train,
        "validation": validation,
        "specification": specification,
        "bootstrap_iterations": 0,
        "worker_count": 1,
        "checkpoint_root": tmp_path / "checkpoints",
        "input_hash": "input-hash",
        "endpoint_index": 1,
        "endpoint_total": 1,
    }
    with pytest.raises(RuntimeError, match="simulated marker pause"):
        _load_or_evaluate_marker_chunks(
            **arguments,
            progress=pause_after_first_chunk,
        )
    evaluations = _load_or_evaluate_marker_chunks(
        **arguments,
        progress=None,
    )
    assert [item.feature_id for item in evaluations] == [
        "marker_a",
        "marker_b",
    ]
    # Checkpoint chunking is intentionally independent of worker count.
    assert calls == [("marker_a", "marker_b")]
    assert evaluations[0].model is not None
    assert evaluations[0].model.direction == 1
    assert evaluations[1].model is not None
    assert evaluations[1].model.direction == -1


def test_emar_detail_and_poe_enrich_medication_context() -> None:
    connection = duckdb.connect()
    try:
        connection.register(
            "emar_source",
            pd.DataFrame(
                {
                    "subject_id": [1],
                    "hadm_id": [10],
                    "emar_id": ["E1"],
                    "emar_seq": [1],
                    "poe_id": ["P1"],
                    "charttime": ["2024-01-01 10:00:00"],
                    "medication": ["Amiodarone"],
                    "event_txt": ["Administered"],
                }
            ),
        )
        connection.register(
            "emar_detail_source",
            pd.DataFrame(
                {
                    "subject_id": [1],
                    "emar_id": ["E1"],
                    "emar_seq": [1],
                    "route": ["IV"],
                    "dose_given": ["150"],
                    "dose_given_unit": ["mg"],
                    "administration_type": ["Bolus"],
                    "complete_dose_not_given": ["N"],
                }
            ),
        )
        connection.register(
            "poe_source",
            pd.DataFrame(
                {
                    "subject_id": [1],
                    "hadm_id": [10],
                    "poe_id": ["P1"],
                    "poe_seq": [1],
                    "ordertime": ["2024-01-01 09:45:00"],
                    "order_type": ["Medications"],
                    "order_subtype": ["General"],
                    "transaction_type": ["New"],
                    "order_status": ["Active"],
                }
            ),
        )
        connection.register(
            "poe_detail_source",
            pd.DataFrame(
                {
                    "subject_id": [1],
                    "poe_id": ["P1"],
                    "poe_seq": [1],
                    "field_name": ["Route"],
                    "field_value": ["Intravenous"],
                }
            ),
        )
        query = _medication_order_administration_context_query(
            {
                "emar": "emar_source",
                "emar_detail": "emar_detail_source",
                "poe": "poe_source",
                "poe_detail": "poe_detail_source",
            },
            ["amiodarone"],
        )
        row = connection.execute(query).fetchdf().iloc[0]
    finally:
        connection.close()

    assert row["source_table"] == "emar"
    assert row["route"] == "IV"
    assert row["dose_raw"] == "150"
    assert row["dose_unit"] == "mg"
    assert row["order_status"] == "Active"
    assert row["temporal_use_policy"] == "post_ecg_not_predictive_input"


def test_job_store_pause_resume_and_atomic_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    project_file = config.save()
    store = JobStore.for_project(config)
    record = store.create(
        stage="extract-features",
        project_file=project_file,
        arguments={"panel_ids": ["primary_core"]},
        job_id="feature-test",
    )
    assert record.state == JobState.QUEUED
    assert store.request_pause(record.spec.job_id).state == JobState.PAUSED
    assert store.request_resume(record.spec.job_id).state == JobState.QUEUED

    store.claim(record.spec.job_id, worker_pid=999_999_999)
    assert store.request_pause(record.spec.job_id).state == JobState.PAUSING
    with pytest.raises(JobPauseRequested, match="atomic checkpoint"):
        JobControl(store, record.spec.job_id).checkpoint(
            "partition complete",
            current=25,
            total=100,
        )
    paused = store.transition(record.spec.job_id, JobState.PAUSED)
    assert paused.progress_current == 25
    assert paused.progress_total == 100
    assert store.events(record.spec.job_id)


def test_resumed_job_eta_uses_only_new_session_work(tmp_path: Path) -> None:
    config = _config(tmp_path)
    project_file = config.save()
    store = JobStore.for_project(config)
    record = store.create(
        stage="extract-features",
        project_file=project_file,
        job_id="resume-eta",
    )
    store.claim(record.spec.job_id, worker_pid=999_999_999)
    store.heartbeat(
        record.spec.job_id,
        progress_current=430_000,
        progress_total=640_351,
        progress_message="Paused",
    )
    ticks = iter((100.0, 110.0, 160.0))
    control = JobControl(
        store,
        record.spec.job_id,
        clock=lambda: next(ticks),
    )

    control.checkpoint(
        "Features 430,025/640,351; ETA=0m 37s",
        current=430_025,
        total=640_351,
    )
    assert store.get(record.spec.job_id).progress_message.endswith(
        "ETA=wird neu kalibriert"
    )

    control.checkpoint(
        "Features 430,600/640,351; ETA=2m 27s",
        current=430_600,
        total=640_351,
    )
    assert store.get(record.spec.job_id).progress_message.endswith(
        "ETA=5h 49m"
    )
    assert store.events(record.spec.job_id)[-1]["message"].endswith(
        "ETA=5h 49m"
    )


def test_resumed_study_keeps_stage_eta_when_checkpoint_counts_workflow_units(
    tmp_path: Path,
) -> None:
    """E13: a stage message must not get an ETA computed in other units."""
    config = _config(tmp_path)
    project_file = config.save()
    store = JobStore.for_project(config)
    record = store.create(
        stage="study-run",
        project_file=project_file,
        job_id="resume-study-eta",
    )
    store.claim(record.spec.job_id, worker_pid=999_999_999)
    store.heartbeat(
        record.spec.job_id,
        progress_current=40_000,
        progress_total=488_650,
        progress_message="Paused",
    )
    ticks = iter((100.0, 5_000.0))
    control = JobControl(store, record.spec.job_id, clock=lambda: next(ticks))
    message = "Features 168,575/640,110; OK=167,033, Reused=6,725, ETA=78h 10m"
    control.checkpoint(message, current=43_573, total=488_650)
    assert store.get(record.spec.job_id).progress_message == message


def test_completed_job_progress_is_closed_at_total(tmp_path: Path) -> None:
    config = _config(tmp_path)
    project_file = config.save()
    store = JobStore.for_project(config)
    record = store.create(
        stage="inventory",
        project_file=project_file,
        job_id="complete-progress",
    )
    store.claim(record.spec.job_id, worker_pid=999_999_999)
    store.heartbeat(
        record.spec.job_id,
        progress_current=947,
        progress_total=1_000,
        progress_message="Final report",
    )

    _finalize_progress(store, record.spec.job_id)

    completed = store.get(record.spec.job_id)
    assert completed.progress_current == 1_000
    assert completed.progress_total == 1_000
    assert completed.progress_message == "Job completed"


def test_job_resource_profiles_match_workers_and_artifact_namespaces(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.workers = 7
    config.save()
    capacity = cpu_token_capacity(detect_hardware())
    feature = resource_profile_for_config(
        "extract-features",
        config,
        {
            "panel_ids": ["primary_core", "sentinel"],
            "max_records": 1000,
        },
    )
    assert feature.cpu_tokens == min(7, capacity)
    assert feature.writer_namespace == (
        "features:primary_core_sentinel:smoke-1000"
    )
    first_run = resource_profile_for_config(
        "validate",
        config,
        {"run_id": "run-a"},
    )
    second_run = resource_profile_for_config(
        "validate",
        config,
        {"run_id": "run-b"},
    )
    report_same_run = resource_profile_for_config(
        "report",
        config,
        {"run_id": "run-a"},
    )
    validation_preview = resource_profile_for_config(
        "validate",
        config,
        {"run_id": "run-preview", "interactive_preview": True},
    )
    phenome = resource_profile_for_config(
        "discover-phenome",
        config,
        {"run_id": "run-a"},
    )
    promotion_refresh = resource_profile_for_config(
        "refresh-phenome-promotions",
        config,
        {"run_id": "run-a"},
    )
    event_attestation = resource_profile_for_config(
        "attest-event-evidence",
        config,
    )
    population = resource_profile_for_config(
        "population-cohorts",
        config,
        {"run_id": "run-a", "workers": 7},
    )
    deep_combinations = resource_profile_for_config(
        "deep-combination-search",
        config,
        {"run_id": "run-a"},
    )
    audit = resource_profile_for_config(
        "audit",
        config,
        {"run_id": "run-a"},
    )
    assert first_run.writer_namespace == report_same_run.writer_namespace
    assert first_run.writer_namespace != second_run.writer_namespace
    assert first_run.cpu_tokens == min(7, capacity)
    assert not first_run.interactive
    assert phenome.cpu_tokens == min(7, capacity)
    assert not phenome.interactive
    assert promotion_refresh.writer_namespace == "run:run-a"
    assert promotion_refresh.cpu_tokens <= 2
    assert promotion_refresh.interactive
    assert validation_preview.cpu_tokens <= 2
    assert validation_preview.interactive
    assert event_attestation.memory_gb == min(
        global_memory_budget_gb(
            detect_hardware(),
            reserve_gb=config.os_memory_reserve_gb,
        ),
        8.0,
    )
    assert event_attestation.writer_namespace == "evidence-migration"
    assert audit.writer_namespace == "run:run-a"
    assert audit.interactive
    assert population.cpu_tokens == min(7, capacity)
    assert population.writer_namespace == "external:run-a"
    assert deep_combinations.cpu_tokens == min(7, capacity)
    assert deep_combinations.writer_namespace == "run:run-a"
    assert deep_combinations.memory_gb == min(
        global_memory_budget_gb(
            detect_hardware(),
            reserve_gb=config.os_memory_reserve_gb,
        ),
        10.0,
    )

    record = JobStore.for_project(config).create(
        stage="extract-features",
        project_file=config.project_file,
        arguments={"panel_ids": ["primary_core"], "max_records": 20},
    )
    assert record.spec.resource_profile.cpu_tokens == min(7, capacity)
    assert record.spec.resource_profile.writer_namespace.endswith("smoke-20")


def test_duckdb_batch_stages_scale_with_configured_workers_not_fixed_at_two(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.workers = 7
    config.save()
    capacity = cpu_token_capacity(detect_hardware())
    for stage in (
        "ingest",
        "build-index",
        "build-evidence",
        "build-labels",
        "parse-reports",
        "ingest-events",
        "attest-event-evidence",
    ):
        profile = resource_profile_for_config(stage, config)
        assert profile.cpu_tokens == min(7, capacity), stage

    # Genuinely interactive stages must stay conservatively capped -- they
    # are meant to run alongside other work, not to claim most of the
    # machine.
    for stage, arguments in (
        ("validate", {"run_id": "run-a", "interactive_preview": True}),
        ("refresh-phenome-promotions", {"run_id": "run-a"}),
        ("report", {"run_id": "run-a"}),
        ("audit", {"run_id": "run-a"}),
    ):
        profile = resource_profile_for_config(stage, config, arguments)
        assert profile.cpu_tokens <= 2, stage


def test_dynamic_worker_hysteresis_requires_consecutive_same_direction_signals() -> None:
    hysteresis = _DynamicWorkerHysteresis()
    # Two "scale down" signals are not enough -- no thrash on a single
    # noisy sample.
    assert hysteresis.step(current_workers=6, recommended_workers=2) is None
    assert hysteresis.step(current_workers=6, recommended_workers=2) is None
    # Third consecutive "scale down" signal triggers exactly a one-step
    # reduction, not a jump straight to the recommended value.
    assert hysteresis.step(current_workers=6, recommended_workers=2) == 5


def test_dynamic_worker_hysteresis_resets_on_direction_change() -> None:
    hysteresis = _DynamicWorkerHysteresis()
    assert hysteresis.step(current_workers=4, recommended_workers=1) is None
    assert hysteresis.step(current_workers=4, recommended_workers=1) is None
    # Direction flips before reaching the threshold -- streak must reset,
    # not carry over toward the opposite direction.
    assert hysteresis.step(current_workers=4, recommended_workers=8) is None
    assert hysteresis.step(current_workers=4, recommended_workers=8) is None
    assert hysteresis.step(current_workers=4, recommended_workers=8) == 5


def test_dynamic_worker_hysteresis_no_op_when_already_at_target() -> None:
    hysteresis = _DynamicWorkerHysteresis()
    assert hysteresis.step(current_workers=4, recommended_workers=4) is None
    assert hysteresis.step(current_workers=4, recommended_workers=4) is None
    assert hysteresis.step(current_workers=4, recommended_workers=4) is None


def test_admission_pool_is_sequential_for_one_worker_and_pooled_above() -> None:
    single = _AdmissionPool(1)
    assert single.executor is None
    single.close()
    pooled = _AdmissionPool(3)
    try:
        assert pooled.executor is not None
        assert pooled.executor.submit(pow, 2, 5).result() == 32
    finally:
        pooled.close()
    assert pooled.executor is None


def test_recommend_dynamic_workers_is_bounded_by_admitted_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakePsutil:
        Error = RuntimeError

        @staticmethod
        def cpu_count(logical: bool = True) -> int:
            return 8

        @staticmethod
        def cpu_percent(interval: float) -> float:
            return 0.0

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil())
    # Fully idle system (0% load, 8 logical cores) still must never
    # recommend more than the job's own admitted budget.
    assert recommend_dynamic_workers(3, sample_interval=0.0) == 3
    assert recommend_dynamic_workers(1, sample_interval=0.0) == 1


def test_recommend_dynamic_workers_shrinks_under_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeBusyPsutil:
        Error = RuntimeError

        @staticmethod
        def cpu_count(logical: bool = True) -> int:
            return 8

        @staticmethod
        def cpu_percent(interval: float) -> float:
            return 95.0

    monkeypatch.setitem(sys.modules, "psutil", _FakeBusyPsutil())
    recommended = recommend_dynamic_workers(6, sample_interval=0.0)
    assert 1 <= recommended < 6


def test_study_run_resources_respect_configured_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.workers = 2
    monkeypatch.setattr(
        "vcg_biomarker.mimic.resources.detect_hardware",
        lambda: HardwareInventory(
            physical_cores=8,
            logical_cores=16,
            total_memory_gb=64.0,
            available_memory_gb=48.0,
            cpu_name="test-cpu",
            platform="test-platform",
        ),
    )

    profile = resource_profile_for_config(
        "study-run",
        config,
        {"preset_id": "exhaustive_research", "run_id": "study-a"},
    )

    assert profile.cpu_tokens == 2
    assert profile.memory_gb == float(config.duckdb_memory_limit_gb)


def test_guided_study_reports_stage_progress_to_job_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    calls: list[tuple[str, int | None, int | None]] = []

    class FakeJobControl:
        def checkpoint(
            self,
            message: str = "",
            *,
            current: int | None = None,
            total: int | None = None,
        ) -> None:
            calls.append((message, current, total))

    def fake_service_stage(*_args, **_kwargs):
        class FakeResult:
            status = "complete"
            row_counts = {
                "selected_ecgs": 1000,
                "successful_ecgs": 800,
            }

        return FakeResult()

    monkeypatch.setattr(
        "vcg_biomarker.mimic.services.run_service_stage",
        fake_service_stage,
    )

    from vcg_biomarker.mimic.workflow import run_guided_study

    run_guided_study(
        config,
        {"preset_id": "fast_preview", "run_id": "study-a"},
        progress=None,
        job_control=FakeJobControl(),
    )

    assert calls
    first_message, first_current, first_total = calls[0]
    # Progress/ETA is reported in weighted overall-run units (see
    # workflow._STAGE_WEIGHTS), not raw stage-index/stage-count: nothing
    # is complete yet before stage 1 starts, so current=0 here, and the
    # total reflects the sum of every stage's weight (scaled), not the
    # number of stages.
    assert first_current == 0
    assert first_total is not None and first_total >= 1
    assert "Schritt 1/" in first_message
    currents = [item[1] for item in calls if item[1] is not None]
    assert currents == sorted(currents)
    assert currents[-1] > first_current


def test_job_project_snapshot_survives_live_config_changes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.workers = 7
    config.duckdb_memory_limit_gb = 20
    config.powerline_hz = 60.0
    config.save()
    profile = ResourceProfile(
        cpu_tokens=2,
        memory_gb=4.0,
        writer_namespace="snapshot-test",
    )
    record = JobStore.for_project(config).create(
        stage="extract-features",
        project_file=config.project_file,
        arguments={"panel_ids": ["primary_core"]},
        resource_profile=profile,
    )
    assert record.spec.arguments[JOB_PROJECT_CONFIG_HASH_KEY]
    assert (
        record.spec.arguments[JOB_PROJECT_CONFIG_KEY]["powerline_hz"]
        == 60.0
    )

    config.workers = 1
    config.duckdb_memory_limit_gb = 2
    config.powerline_hz = 50.0
    config.save()
    live = MimicProjectConfig.load(config.project_file)
    runtime = _runtime_config_for_job(live, record.spec)

    assert runtime.powerline_hz == 60.0
    assert runtime.workers == 2
    assert runtime.duckdb_memory_limit_gb == 4
    assert runtime.to_dict()["workers"] == 7
    assert runtime.to_dict()["duckdb_memory_limit_gb"] == 20


def test_legacy_job_binds_project_snapshot_on_first_resume(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.workers = 6
    config.save()
    store = JobStore.for_project(config)
    legacy = store.enqueue(
        JobSpec(
            job_id="legacy-resume",
            stage="inventory",
            project_file=str(config.project_file),
        )
    )
    assert JOB_PROJECT_CONFIG_KEY not in legacy.spec.arguments
    store.request_pause(legacy.spec.job_id)
    resumed = store.request_resume(legacy.spec.job_id)

    assert resumed.spec.arguments[JOB_PROJECT_CONFIG_KEY]["workers"] == 6
    assert resumed.spec.arguments[
        JOB_PROJECT_CONFIG_HASH_KEY
    ] == content_hash(
        dict(resumed.spec.arguments[JOB_PROJECT_CONFIG_KEY])
    )


def test_stale_worker_is_recoverable_without_losing_job_spec(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = JobStore.for_project(config)
    record = store.create(
        stage="validate",
        project_file=config.save(),
        arguments={"run_id": "development"},
        job_id="stale-test",
    )
    store.claim(record.spec.job_id, worker_pid=999_999_999)
    recovered = store.recover_stale_jobs(stale_after_seconds=-1)
    assert recovered == (record.spec.job_id,)
    paused = store.get(record.spec.job_id)
    assert paused.state == JobState.PAUSED
    assert paused.worker_pid is None
    assert paused.spec.arguments["run_id"] == "development"
    assert store.request_resume(record.spec.job_id).state == JobState.QUEUED


def test_job_priority_is_live_persistent_and_reorders_queue(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = JobStore.for_project(config)
    store.create(
        stage="inventory",
        project_file=config.save(),
        job_id="low",
        priority=10,
    )
    store.create(
        stage="validate",
        project_file=config.project_file,
        job_id="high",
        priority=20,
    )
    updated = store.set_priority("low", 90)
    assert updated.spec.priority == 90
    assert [record.spec.job_id for record in store.list(limit=2)] == [
        "low",
        "high",
    ]
    assert store.events("low")[-1]["event_type"] == "priority"
    with pytest.raises(ValueError, match="between 0 and 100"):
        store.set_priority("low", 101)


def test_job_launch_reservation_is_atomic_across_store_instances(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = JobStore.for_project(config)
    second = JobStore.for_project(config)
    first.create(
        stage="inventory",
        project_file=config.save(),
        job_id="single-launch",
    )
    reserved = first.reserve_for_launch(
        "single-launch",
        supervisor_pid=111,
    )
    assert reserved is not None
    assert reserved.state == JobState.RUNNING
    assert second.reserve_for_launch(
        "single-launch",
        supervisor_pid=222,
    ) is None
    assert [
        event["event_type"]
        for event in first.events("single-launch")
    ].count("reserved") == 1


def test_supervisor_reserves_one_core_and_blocks_oversized_jobs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = JobStore.for_project(config)
    store.create(
        stage="inventory",
        project_file=config.save(),
        job_id="oversized",
        resource_profile=ResourceProfile(
            cpu_tokens=999,
            memory_gb=999.0,
        ),
    )
    supervisor = LocalJobSupervisor(config)
    assert supervisor.cpu_capacity == cpu_token_capacity(supervisor.hardware)
    supervisor.run_once()
    blocked = store.get("oversized")
    assert blocked.state == JobState.BLOCKED
    assert "requested 999 CPU tokens" in blocked.error_message


def test_supervisor_runs_bounded_interactive_reader_beside_feature_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = JobStore.for_project(config)
    supervisor = LocalJobSupervisor(
        config,
        memory_probe=lambda: MemoryStatus(64.0, 48.0, 25.0, 0.0),
    )
    standard = store.create(
        stage="extract-features",
        project_file=config.save(),
        job_id="feature-writer",
        priority=90,
        resource_profile=ResourceProfile(
            cpu_tokens=supervisor.cpu_capacity - supervisor.interactive_cpu_capacity,
            memory_gb=1.0,
            writer_namespace="features",
        ),
    )
    interactive = store.create(
        stage="report",
        project_file=config.project_file,
        job_id="read-only-analysis",
        priority=80,
        resource_profile=ResourceProfile(
            cpu_tokens=supervisor.interactive_cpu_capacity,
            memory_gb=1.0,
            writer_namespace="run:previous",
            interactive=True,
        ),
    )
    launched: list[str] = []

    def launch(record) -> bool:
        store.reserve_for_launch(record.spec.job_id, supervisor_pid=123)
        launched.append(record.spec.job_id)
        return True

    monkeypatch.setattr(supervisor, "_launch", launch)
    for _ in range(MEMORY_READY_SAMPLES):
        supervisor.run_once()
    assert launched == [standard.spec.job_id, interactive.spec.job_id]


def test_supervisor_waits_for_live_memory_then_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = JobStore.for_project(config)
    memory = [MemoryStatus(32.0, 3.0, 90.0, 0.5)]
    supervisor = LocalJobSupervisor(
        config,
        memory_probe=lambda: memory[0],
        clock=lambda: 100.0,
    )
    record = store.create(
        stage="extract-features",
        project_file=config.save(),
        job_id="memory-gated",
        resource_profile=ResourceProfile(
            cpu_tokens=1,
            memory_gb=10.0,
            writer_namespace="features",
        ),
    )
    launched: list[str] = []

    def launch(candidate) -> bool:
        store.reserve_for_launch(candidate.spec.job_id, supervisor_pid=123)
        launched.append(candidate.spec.job_id)
        return True

    monkeypatch.setattr(supervisor, "_launch", launch)
    supervisor.run_once()
    assert launched == []
    waiting = store.get(record.spec.job_id)
    assert waiting.state == JobState.QUEUED
    assert "Wartet auf freien Arbeitsspeicher" in waiting.progress_message
    assert any(
        event["event_type"] == "memory_pressure_wait"
        for event in store.events(record.spec.job_id)
    )

    memory[0] = MemoryStatus(32.0, 12.0, 62.5, 0.2)
    for _ in range(MEMORY_READY_SAMPLES - 1):
        supervisor.run_once()
        assert launched == []
    supervisor.run_once()
    assert launched == [record.spec.job_id]
    assert any(
        event["event_type"] == "memory_pressure_wait_cleared"
        for event in store.events(record.spec.job_id)
    )


def test_supervisor_pauses_background_job_after_sustained_memory_pressure(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = JobStore.for_project(config)
    supervisor = LocalJobSupervisor(
        config,
        memory_probe=lambda: MemoryStatus(32.0, 1.0, 96.9, 1.0),
    )
    record = store.create(
        stage="extract-features",
        project_file=config.save(),
        job_id="memory-pressure-pause",
        resource_profile=ResourceProfile(
            cpu_tokens=1,
            memory_gb=10.0,
            writer_namespace="features",
        ),
    )
    store.claim(record.spec.job_id, os.getpid())

    for _ in range(MEMORY_PRESSURE_SAMPLES - 1):
        supervisor.run_once()
        assert store.get(record.spec.job_id).state == JobState.RUNNING
    supervisor.run_once()
    paused = store.get(record.spec.job_id)
    assert paused.state == JobState.PAUSING
    assert "kritischem Arbeitsspeicher" in paused.error_message
    events = store.events(record.spec.job_id)
    pressure = [
        event
        for event in events
        if event["event_type"] == "memory_pressure_pause_requested"
    ]
    assert len(pressure) == 1
    assert pressure[0]["payload"]["available_memory_gb"] == 1.0
    store.transition(record.spec.job_id, JobState.PAUSED)
    resumed = store.request_resume(record.spec.job_id)
    assert resumed.error_message == ""


def test_memory_pressure_thresholds_keep_reserve_and_startup_headroom() -> None:
    assert critical_available_memory_gb(reserve_gb=6.0) == 6.0
    assert launch_available_memory_gb(
        reserve_gb=6.0,
        requested_memory_gb=10.0,
    ) == 8.0
    assert launch_available_memory_gb(
        reserve_gb=6.0,
        requested_memory_gb=100.0,
    ) == 10.0


def test_orphan_cleanup_matches_only_the_recorded_pool_parent() -> None:
    command = (
        "python.exe",
        "-c",
        (
            "from multiprocessing.spawn import spawn_main; "
            "spawn_main(parent_pid=30536, pipe_handle=3636)"
        ),
        "--multiprocessing-fork",
    )
    assert _is_process_pool_child_of(command, 30536)
    assert not _is_process_pool_child_of(command, 30537)
    assert not _is_process_pool_child_of(
        ("python.exe", "unrelated_script.py", "--infinite"),
        30536,
    )


def test_forced_feature_job_resumes_only_its_own_atomic_outputs(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / ".part.resume.parquet"
    manifest_path = tmp_path / ".part.resume.json"
    frame = pd.DataFrame(
        {"study_id": [11], "extraction_status": ["ok"]}
    )
    _write_resume_checkpoint(
        frame,
        parquet_path,
        manifest_path,
        part_hash="part-hash",
        execution_id="job-a",
    )

    own = _load_resume_checkpoint(
        parquet_path,
        manifest_path,
        part_hash="part-hash",
        expected_study_ids={11, 12},
        force=True,
        execution_id="job-a",
    )
    foreign = _load_resume_checkpoint(
        parquet_path,
        manifest_path,
        part_hash="part-hash",
        expected_study_ids={11, 12},
        force=True,
        execution_id="job-b",
    )
    assert own["study_id"].tolist() == [11]
    assert foreign.empty
    assert _partition_reusable_for_execution(
        {"execution_id": "job-a"},
        FeatureExtractionOptions(force=True, execution_id="job-a"),
    )
    assert not _partition_reusable_for_execution(
        {"execution_id": "job-a"},
        FeatureExtractionOptions(force=True, execution_id="job-b"),
    )


def test_worker_honors_pause_requested_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker import mimic_job_worker

    config = _config(tmp_path)
    store = JobStore.for_project(config)
    store.create(
        stage="inventory",
        project_file=config.save(),
        job_id="prestart-pause",
    )
    store.reserve_for_launch("prestart-pause", supervisor_pid=111)
    store.request_pause("prestart-pause")
    monkeypatch.setattr(
        "sys.argv",
        [
            "vcg-mimic-job-worker",
            "--project",
            str(config.project_file),
            "--job-id",
            "prestart-pause",
        ],
    )
    assert mimic_job_worker.main() == 75
    assert store.get("prestart-pause").state == JobState.PAUSED


def test_worker_thread_limits_cover_common_blas_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vcg_biomarker.mimic.resources as resource_module

    monkeypatch.setattr(resource_module, "_THREAD_LIMITS_APPLIED", False)
    monkeypatch.setattr(resource_module, "_THREADPOOL_LIMITER", None)
    apply_worker_thread_limits()
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert __import__("os").environ[name] == "1"


def test_job_resource_environment_clamps_duckdb_without_mutating_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = ResourceProfile(cpu_tokens=2, memory_gb=6.0)
    _apply_job_resource_environment(profile)
    catalog = DuckDBCatalog(
        tmp_path / "catalog.duckdb",
        memory_limit_gb=20,
        threads=7,
    )
    assert catalog.threads == 2
    assert catalog.memory_limit_gb == 6.0
    with catalog.connect(isolated=True) as connection:
        assert connection.execute(
            "SELECT current_setting('threads')"
        ).fetchone()[0] == 2
    monkeypatch.delenv("VCG_JOB_CPU_TOKENS")
    monkeypatch.delenv("VCG_JOB_MEMORY_GB")
    monkeypatch.delenv("VCG_JOB_MEMORY_ALLOCATION_GB")
    monkeypatch.delenv("VCG_JOB_OS_RESERVE_GB")


def test_cli_parses_persistent_jobs_reports_and_ecg_labor() -> None:
    from vcg_biomarker import mimic_cli

    parser = mimic_cli._build_parser()
    job = parser.parse_args(
        [
            "--project",
            "project.json",
            "jobs",
            "start",
            "extract-features",
            "--arguments-json",
            '{"panel_ids":["primary_core"]}',
            "--priority",
            "80",
        ]
    )
    assert job.jobs_command == "start"
    assert job.stage == "extract-features"
    assert job.priority == 80

    report = parser.parse_args(
        [
            "report",
            "build",
            "--run-id",
            "development",
            "--formats",
            "pdf,html",
        ]
    )
    assert report.report_action == "build"
    assert report.run_id == "development"

    ecg = parser.parse_args(
        [
            "ecg",
            "analyze",
            "case-123",
            "--panels",
            "primary_core,sentinel",
        ]
    )
    assert ecg.ecg_command == "analyze"
    assert ecg.case_id == "case-123"

    locked_features = parser.parse_args(
        [
            "extract-features",
            "--locked-test-for-frozen-run",
            "development",
        ]
    )
    assert locked_features.locked_test_for_frozen_run == "development"


def test_locked_test_feature_extraction_requires_an_explicit_freeze() -> None:
    with pytest.raises(ValueError, match="Union extraction"):
        FeatureExtractionOptions(include_locked_test=True)
    with pytest.raises(ValueError, match="freeze_run_id"):
        FeatureExtractionOptions(locked_test_only=True)
    options = FeatureExtractionOptions(
        locked_test_only=True,
        freeze_run_id="development",
    )
    assert options.locked_test_only
    with pytest.raises(ValueError, match="max_records"):
        FeatureExtractionOptions(
            locked_test_only=True,
            freeze_run_id="development",
            max_records=100,
        )


def test_snapshot_detects_data_mutation_even_with_stale_sidecar(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    root = config.features_root / "primary_core" / "development"
    partition = root / "split=train" / "part-00000.parquet"
    partition.parent.mkdir(parents=True)
    pd.DataFrame({"study_id": [1], "value": [1.0]}).to_parquet(
        partition,
        index=False,
    )
    atomic_write_json(
        partition.with_suffix(".json"),
        {"parquet_sha256": file_sha256(partition)},
    )
    atomic_write_json(
        root / "features.manifest.json",
        {"status": "complete", "input_hash": "input-v1"},
    )
    atomic_write_json(root / "partition_index.json", {"partition_count": 1})

    snapshot = create_artifact_snapshot(
        config,
        artifact_kind="features",
        source_glob=root / "split=*" / "part-*.parquet",
        snapshot_id="immutable-test",
        require_complete=True,
    )
    assert snapshot.complete
    assert verify_artifact_snapshot(snapshot)["valid"]

    pd.DataFrame({"study_id": [1], "value": [2.0]}).to_parquet(
        partition,
        index=False,
    )
    verification = verify_artifact_snapshot(snapshot)
    assert not verification["valid"]
    assert verification["changed"] == ["split=train/part-00000.parquet"]


def test_duckdb_catalog_serializes_writers_but_allows_isolated_reads(
    tmp_path: Path,
) -> None:
    catalog = DuckDBCatalog(tmp_path / "catalog.duckdb")
    entered: list[str] = []
    first_ready = threading.Event()
    release_first = threading.Event()

    def first_writer() -> None:
        with catalog.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS example(value INTEGER)")
            entered.append("first")
            first_ready.set()
            release_first.wait(timeout=5)

    def second_writer() -> None:
        first_ready.wait(timeout=5)
        with catalog.connect():
            entered.append("second")

    first = threading.Thread(target=first_writer)
    second = threading.Thread(target=second_writer)
    first.start()
    second.start()
    assert first_ready.wait(timeout=5)
    with catalog.connect(isolated=True) as isolated:
        assert isolated.execute("SELECT 42").fetchone()[0] == 42
    time.sleep(0.15)
    assert entered == ["first"]
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert entered == ["first", "second"]

    lock_path = catalog.path.with_suffix(f"{catalog.path.suffix}.lock")
    atomic_write_json(lock_path, {"pid": 999_999_999})
    with catalog.connect() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    assert not lock_path.exists()


def test_partial_snapshot_cannot_be_frozen() -> None:
    snapshot = ArtifactSnapshot(
        snapshot_id="partial",
        artifact_kind="features",
        source_root=".",
        files=("part.parquet",),
        file_hashes={"part.parquet": "abc"},
        snapshot_hash="def",
        complete=False,
    )
    with pytest.raises(RuntimeError, match="partial"):
        snapshot.assert_freezable()


def test_smoke_feature_snapshot_is_always_partial(tmp_path: Path) -> None:
    root = tmp_path / "smoke-development-1000"
    root.mkdir()
    atomic_write_json(root / "features.manifest.json", {"status": "complete"})
    atomic_write_json(root / "partition_index.json", {"partition_count": 1})
    partition = root / "split=train" / "part-000000.parquet"
    partition.parent.mkdir()
    pd.DataFrame({"study_id": [1]}).to_parquet(partition, index=False)

    assert not _is_complete(root, file_count=1, expected_files=None)


def test_analysis_presets_cannot_optimize_scientific_guards() -> None:
    """The guards hold for every preset, however many there are.

    The set collapsed from five to two in August 2026: three of the five had
    decayed into duplicates or into knobs the runtime now decides better, and
    the dial they sat behind was labelled as thoroughness while its real
    effect was a five-fold change in scope. The guard assertions below are the
    point of this test and are unchanged.
    """

    presets = analysis_presets()
    assert {item.preset_id for item in presets} == {
        "exhaustive_research",
        "fast_preview",
        "production_path_preview",
    }
    for preset in presets:
        assert not preset.policy.allow_outcome_tuned_filters
        assert not preset.policy.allow_label_rule_optimization
        assert not preset.policy.allow_locked_test_optimization
        assert preset.policy.max_development_score_components <= 3
    with pytest.raises(ValueError, match="Scientific guard"):
        ParameterPolicy(allow_locked_test_optimization=True)


def test_retired_preset_ids_resolve_instead_of_crashing_a_run() -> None:
    """An old manifest must not end a month-long study over a name.

    Resolution is safe because the preset is part of the configuration hash:
    the scope difference is caught there and forces recomputation, rather than
    silently reusing results from the narrower study the old name meant.
    """

    from vcg_biomarker.mimic.presets import RETIRED_PRESET_IDS

    for retired in RETIRED_PRESET_IDS:
        preset = get_analysis_preset(retired)
        assert preset.preset_id == "exhaustive_research"
    with pytest.raises(KeyError):
        get_analysis_preset("never_existed")


def test_a_retired_preset_id_builds_the_identical_run() -> None:
    """Resolving the preset is not enough if the branches see the old name.

    ``guided_study_spec`` compares the identifier directly in several places,
    so a retired name first produced a quietly narrower run -- 25 stages
    instead of 27, missing the 10.000-record scaling pilot -- and recorded the
    retired name in the spec as though that were the configuration that ran.
    A resolution that stops at the registry lookup is worse than none, because
    it looks like it worked.
    """

    from vcg_biomarker.mimic.presets import RETIRED_PRESET_IDS
    from vcg_biomarker.mimic.workflow import guided_study_spec

    canonical = guided_study_spec(
        preset_id="exhaustive_research", run_id="r", study_id="s"
    )
    for retired in RETIRED_PRESET_IDS:
        spec = guided_study_spec(preset_id=retired, run_id="r", study_id="s")
        assert spec.stage_ids == canonical.stage_ids
        # The spec must record what ran, not the name that was typed.
        assert spec.preset_id == "exhaustive_research"


def test_cornell_product_registry_uses_the_implemented_units() -> None:
    definition = BaselineRegistry().get("cornell_product")
    assert "244 mV*ms" in definition.formula
    assert "2440 mm*ms" in definition.formula
    assert "2440 mV*ms" not in definition.formula


def test_potassium_baselines_do_not_reverse_hyperkalemia_pattern() -> None:
    registry = BaselineRegistry()
    hyper = registry.get("hyperkalemia_ecg_heuristic")
    assert set(hyper.endpoint_ids) == {
        "hyperkalemia_near_ecg",
        "severe_hyperkalemia_near_ecg",
    }
    for endpoint_id in (
        "hypokalemia_near_ecg",
        "severe_hypokalemia_near_ecg",
    ):
        definitions = registry.for_endpoint(endpoint_id)
        assert definitions
        assert all(
            item.baseline_id != "hyperkalemia_ecg_heuristic"
            for item in definitions
        )
        assert any(
            item.baseline_id == "no_established_ecg_rule_hypokalemia"
            and item.kind.value == "none_established"
            for item in definitions
        )


def test_grouped_score_search_aligns_opposite_marker_directions() -> None:
    rng = np.random.default_rng(20260726)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 120)
    subjects = np.arange(len(labels), dtype=np.int64)
    frame = pd.DataFrame(
        {
            "higher_in_cases": labels + rng.normal(0, 0.05, len(labels)),
            "lower_in_cases": -labels + rng.normal(0, 0.05, len(labels)),
        }
    )
    family = {
        "higher_in_cases": "qrs_t",
        "lower_in_cases": "t_loop",
    }
    candidates = grouped_score_search(
        frame,
        labels=labels,
        subject_ids=subjects,
        feature_ids=tuple(frame.columns),
        family_for_feature=family.__getitem__,
        folds=5,
        max_components=2,
    )
    assert len(candidates) == 1
    assert candidates[0].components == (
        "higher_in_cases",
        "lower_in_cases",
    )
    assert candidates[0].stable
    assert candidates[0].mean_auc > 0.99


def test_grouped_score_search_uses_reproducible_stratified_subject_folds() -> None:
    labels = np.asarray([1] * 5 + [0] * 45, dtype=np.int8)
    subjects = np.arange(len(labels), dtype=np.int64)
    frame = pd.DataFrame(
        {
            "marker_a": labels.astype(float),
            "marker_b": -labels.astype(float),
        }
    )
    kwargs = {
        "labels": labels,
        "subject_ids": subjects,
        "feature_ids": tuple(frame.columns),
        "family_for_feature": {
            "marker_a": "qrs_t",
            "marker_b": "t_loop",
        }.__getitem__,
        "folds": 5,
        "max_components": 2,
        "random_seed": 71,
    }
    first = grouped_score_search(frame, **kwargs)
    second = grouped_score_search(frame, **kwargs)

    assert first == second
    assert len(first) == 1
    assert first[0].evaluable_folds == 5
    assert first[0].positive_direction_folds == 5


def test_grouped_score_search_block_loader_is_result_identical() -> None:
    rng = np.random.default_rng(20260903)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 160)
    subjects = np.arange(len(labels), dtype=np.int64)
    names = (
        "qrs_t_hausdorff_distance",
        "t_loop_area",
        "svg_magnitude",
        "qrs_loop_volume",
    )
    frame = pd.DataFrame(
        {
            name: (1 if index % 2 == 0 else -1) * labels
            + rng.normal(0.0, 0.15 + 0.02 * index, len(labels))
            for index, name in enumerate(names)
        }
    )
    families = dict(zip(names, ("qrs_t", "t_loop", "svg", "qrs_loop")))
    kwargs = {
        "labels": labels,
        "subject_ids": subjects,
        "feature_ids": names,
        "family_for_feature": families.__getitem__,
        "folds": 5,
        "max_components": 3,
        "representatives_per_family": 1,
        "selected_pairs": 4,
        "selected_triples": 2,
        "triple_seed_pairs": 4,
        "random_seed": 83,
    }
    expected = grouped_score_search(frame, **kwargs)
    requested_blocks: list[tuple[str, ...]] = []

    def load(columns: tuple[str, ...]) -> dict[str, np.ndarray]:
        requested_blocks.append(columns)
        return {
            column: frame[column].to_numpy(dtype=float)
            for column in columns
        }

    observed = grouped_score_search(
        frame.iloc[:, 0:0],
        **kwargs,
        feature_value_loader=load,
        feature_block_size=2,
    )

    assert observed == expected
    assert requested_blocks[:2] == [names[:2], names[2:]]
    assert max(map(len, requested_blocks)) <= 2


def test_grouped_score_search_refuses_fewer_than_five_class_subjects() -> None:
    labels = np.asarray([1] * 4 + [0] * 46, dtype=np.int8)
    frame = pd.DataFrame(
        {
            "marker_a": labels.astype(float),
            "marker_b": -labels.astype(float),
        }
    )
    result = grouped_score_search(
        frame,
        labels=labels,
        subject_ids=np.arange(len(labels)),
        feature_ids=tuple(frame.columns),
        family_for_feature={
            "marker_a": "qrs_t",
            "marker_b": "t_loop",
        }.__getitem__,
        folds=5,
        max_components=2,
    )
    assert result == ()


def test_feature_union_uses_exact_phenome_indices_and_clinical_ecgs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    master = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 3, 4, 9],
            "study_id": [101, 102, 201, 301, 302, 401, 999],
            "hadm_id": [10, 10, 20, 30, 30, pd.NA, 90],
            "split": [
                "train",
                "train",
                "validation",
                "train",
                "train",
                "validation",
                "locked_test",
            ],
            "ecg_time": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-03-02",
                    "2020-04-01",
                    "2020-05-01",
                ]
            ),
            "waveform_path": [f"record-{value}" for value in range(7)],
            "age": [60] * 7,
            "sex": ["M"] * 7,
            "machine_qrs_axis_deg": [0.0] * 7,
            "machine_report": [""] * 7,
            "machine_cart_id": [1] * 7,
            "machine_time_mismatch": [False] * 7,
            "machine_time_mismatch_seconds": [0] * 7,
            "machine_metadata_temporal_status": [
                "time_consistent_le_60s"
            ] * 7,
            "machine_metadata_usable_for_primary": [True] * 7,
        }
    )
    master.to_parquet(
        config.master_root / "ecg_master_index.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "subject_id": [1, 4],
            "study_id": [102, 401],
            "label_state": ["positive", "negative"],
            "eligibility_state": ["eligible", "eligible"],
        }
    ).to_parquet(
        config.labels_root / "clinical_label_evidence.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 9],
            "hadm_id": [10, 20, 30, 90],
            "icd_version": [10, 10, 9, 10],
            "icd_code": ["A", "UNMAPPED", "B", "A"],
        }
    ).to_parquet(
        config.standardized_root / "diagnoses_icd.parquet",
        index=False,
    )
    mapping = tmp_path / "phecodes.parquet"
    pd.DataFrame(
        {
            "icd_version": [10, 9],
            "icd_code": ["A", "B"],
            "phenotype_id": ["P1", "P2"],
        }
    ).to_parquet(mapping, index=False)

    selected = _load_feature_cohort(
        config,
        FeatureExtractionOptions(),
        phenotype_mapping_path=mapping,
    )

    assert set(selected["study_id"]) == {101, 102, 201, 301, 401}
    assert selected["study_id"].is_unique
    assert 302 not in set(selected["study_id"])
    assert 999 not in set(selected["study_id"])


def test_capped_feature_pilot_covers_observed_endpoint_classes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    rows = []
    labels = []
    for split_index, split in enumerate(("train", "validation")):
        for row_index in range(6):
            study_id = 10_000 * (split_index + 1) + row_index
            subject_id = 1_000 * (split_index + 1) + row_index
            rows.append(
                {
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "hadm_id": subject_id,
                    "split": split,
                    "ecg_time": pd.Timestamp("2020-01-01")
                    + pd.Timedelta(days=row_index),
                    "waveform_path": f"record-{study_id}",
                    "age": 60,
                    "sex": "M",
                    "machine_qrs_axis_deg": 0.0,
                    "machine_report": "",
                    "machine_cart_id": 1,
                    "machine_time_mismatch": False,
                    "machine_time_mismatch_seconds": 0,
                    "machine_metadata_temporal_status": (
                        "time_consistent_le_60s"
                    ),
                    "machine_metadata_usable_for_primary": True,
                }
            )
            labels.append(
                {
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "label_id": "endpoint_a",
                    "label_state": (
                        "positive" if row_index >= 3 else "negative"
                    ),
                    "eligibility_state": "eligible",
                }
            )
    pd.DataFrame(rows).to_parquet(
        config.master_root / "ecg_master_index.parquet",
        index=False,
    )
    label_frame = pd.DataFrame(labels)
    label_frame.to_parquet(
        config.labels_root / "clinical_label_evidence.parquet",
        index=False,
    )
    label_frame.to_parquet(
        config.labels_root / "endpoint_index_cohorts.parquet",
        index=False,
    )

    selected = _load_feature_cohort(
        config,
        FeatureExtractionOptions(max_records=4),
    )

    selected_labels = selected[["study_id", "split"]].merge(
        label_frame[["study_id", "label_state"]],
        on="study_id",
        how="left",
        validate="one_to_one",
    )
    assert len(selected_labels) == 4
    for split in ("train", "validation"):
        scoped = selected_labels.loc[selected_labels["split"].eq(split)]
        assert set(scoped["label_state"]) == {"positive", "negative"}
    # A master without recorder intervals yields typed NULLs, not a binder error.
    for name in ("machine_qrs_duration_ms", "machine_qt_interval_ms", "machine_pr_interval_ms"):
        assert name in selected.columns
        assert selected[name].isna().all()


def test_recorder_intervals_reach_the_cohort_only_when_metadata_is_usable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    rows = []
    labels = []
    for row_index in range(4):
        rows.append(
            {
                "subject_id": 1_000 + row_index,
                "study_id": 10_000 + row_index,
                "hadm_id": 1_000 + row_index,
                "split": "train",
                "ecg_time": pd.Timestamp("2020-01-01") + pd.Timedelta(days=row_index),
                "waveform_path": f"record-{row_index}",
                "age": 60,
                "sex": "M",
                "machine_qrs_axis_deg": 0.0,
                "machine_qrs_duration_ms": 90.0 + row_index,
                "machine_qt_interval_ms": 400.0,
                "machine_pr_interval_ms": 160.0,
                "machine_report": "",
                "machine_cart_id": 1,
                "machine_time_mismatch": False,
                "machine_time_mismatch_seconds": 0,
                "machine_metadata_temporal_status": "time_consistent_le_60s",
                "machine_metadata_usable_for_primary": row_index != 0,
            }
        )
        labels.append(
            {
                "subject_id": 1_000 + row_index,
                "study_id": 10_000 + row_index,
                "label_id": "endpoint_a",
                "label_state": "positive" if row_index >= 2 else "negative",
                "eligibility_state": "eligible",
            }
        )
    pd.DataFrame(rows).to_parquet(config.master_root / "ecg_master_index.parquet", index=False)
    label_frame = pd.DataFrame(labels)
    label_frame.to_parquet(config.labels_root / "clinical_label_evidence.parquet", index=False)
    label_frame.to_parquet(config.labels_root / "endpoint_index_cohorts.parquet", index=False)

    selected = _load_feature_cohort(config, FeatureExtractionOptions()).set_index("study_id")
    assert pd.isna(selected.loc[10_000, "machine_qrs_duration_ms"])
    assert selected.loc[10_001, "machine_qrs_duration_ms"] == 91.0
    assert selected.loc[10_001, "machine_qt_interval_ms"] == 400.0
    assert selected.loc[10_001, "machine_pr_interval_ms"] == 160.0


def test_probability_models_require_complete_nonpreview_sampling() -> None:
    assert not _probability_models_allowed(
        preview=True,
        sampling_design="endpoint_enriched_pilot",
    )
    assert not _probability_models_allowed(
        preview=False,
        sampling_design="endpoint_enriched_pilot",
    )
    assert _probability_models_allowed(
        preview=False,
        sampling_design="complete_clinical_phenome_union",
    )
    assert not _prevalence_metrics_allowed(
        preview=True,
        sampling_design="endpoint_enriched_pilot",
    )
    assert _prevalence_metrics_allowed(
        preview=False,
        sampling_design="complete_clinical_phenome_union",
    )


def test_enriched_preview_suppresses_prevalence_dependent_metrics() -> None:
    frames = {
        "screening_results": pd.DataFrame(
            {
                "roc_auc": [0.81],
                "average_precision": [0.73],
                "specificity_90_observed_ppv": [0.44],
                "specificity_90_observed_npv": [0.91],
            }
        ),
        "validation_label_counts": pd.DataFrame(
            {"prevalence": [0.60], "n_positive": [60]}
        ),
        "performance_curve_points": pd.DataFrame(
            {
                "curve_type": ["roc", "precision_recall"],
                "x": [0.1, 0.2],
                "y": [0.8, 0.7],
                "average_precision": [0.73, 0.73],
            }
        ),
    }

    sanitized = _suppress_prevalence_dependent_metrics(frames)

    screening = sanitized["screening_results"].iloc[0]
    assert screening["roc_auc"] == pytest.approx(0.81)
    assert pd.isna(screening["average_precision"])
    assert pd.isna(screening["specificity_90_observed_ppv"])
    assert pd.isna(screening["specificity_90_observed_npv"])
    counts = sanitized["validation_label_counts"].iloc[0]
    assert pd.isna(counts["prevalence"])
    assert counts["pilot_sample_prevalence"] == pytest.approx(0.60)
    assert sanitized["performance_curve_points"]["curve_type"].tolist() == [
        "roc"
    ]
    payload = _suppress_prevalence_metrics_in_payload(
        {
            "roc_auc": 0.81,
            "grouped_cv": {"mean_average_precision": 0.73},
            "threshold": {"ppv_validation": 0.44},
        }
    )
    assert payload["roc_auc"] == pytest.approx(0.81)
    assert payload["grouped_cv"]["mean_average_precision"] is None
    assert payload["threshold"]["ppv_validation"] is None


def test_atomic_json_serializes_nonfinite_values_as_null(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict.json"
    payload = {
        "nan": float("nan"),
        "positive_infinity": float("inf"),
        "negative_infinity": np.float64("-inf"),
        "nested": [np.float32("nan"), 1.25],
    }

    atomic_write_json(path, payload)

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(
        raw,
        parse_constant=lambda token: pytest.fail(
            f"Non-standard JSON token written: {token}"
        ),
    )
    assert parsed == {
        "nan": None,
        "positive_infinity": None,
        "negative_infinity": None,
        "nested": [None, 1.25],
    }
    assert content_hash(payload) == content_hash(parsed)
    assert _validation_json_value(np.float64("nan")) is None


def test_feature_extraction_records_individual_calculator_failures() -> None:
    destination: dict = {}
    failures: list[dict[str, str]] = []
    results = [
        SimpleNamespace(
            name="valid_marker",
            value=1.25,
            computation_notes="",
        ),
        SimpleNamespace(
            name="failed_marker",
            value=None,
            computation_notes="Error: singular morphology matrix",
        ),
    ]

    _store_biomarker_results(
        destination,
        results,
        failures,
        prefix="sensitivity_inverse_dower__",
    )
    _finalize_biomarker_failure_provenance(destination, failures)

    assert (
        destination["sensitivity_inverse_dower__valid_marker"]
        == pytest.approx(1.25)
    )
    assert destination["biomarker_calculation_failure_count"] == 1
    assert destination["biomarker_calculation_failure_names"] == (
        "sensitivity_inverse_dower__failed_marker"
    )
    details = json.loads(
        destination["biomarker_calculation_failure_details"]
    )
    assert details == [
        {
            "feature_id": "sensitivity_inverse_dower__failed_marker",
            "message": "singular morphology matrix",
        }
    ]


def test_machine_report_uncertainty_is_not_counted_as_negative() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "label": [1, 0, 1, 0],
            "report_reference_state": [
                "positive",
                "negative",
                "uncertain",
                "not_mentioned",
            ],
        }
    )
    specification = ValidationSpec(
        run_id="development",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={},
        development_bootstrap_iterations=0,
        min_class_inference=1,
        min_class_primary=1,
    )
    records = _machine_report_records(
        "endpoint",
        "ischemia",
        frame,
        frame,
        specification,
    )

    assert len(records) == 2
    for record in records:
        assert record["n_source_rows"] == 4
        assert record["n_total"] == 3
        assert record["uncertain_count"] == 1
        assert record["uncertain_excluded"] is True
        assert record["missingness"] == pytest.approx(0.25)
        assert record["sensitivity"] == pytest.approx(1.0)
        assert record["specificity"] == pytest.approx(1.0)

    locked_specification = ValidationSpec(
        run_id="development",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={},
        final_bootstrap_iterations=0,
        min_class_inference=1,
        min_class_primary=1,
    )
    locked = _locked_report_record(
        "endpoint",
        "ischemia",
        frame,
        locked_specification,
    )
    assert locked is not None
    assert locked["n_total"] == 3
    assert locked["uncertain_count"] == 1
    assert locked["uncertain_excluded"] is True
    assert locked["sensitivity"] == pytest.approx(1.0)


def test_report_feature_qc_is_bound_to_validation_feature_root(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_root = config.runs_root / "development"
    feature_root = config.features_root / "panel-hash" / "development"
    run_root.mkdir(parents=True)
    feature_root.mkdir(parents=True)
    bound = feature_root / "feature_failure_report.parquet"
    pd.DataFrame([{"status": "ok"}]).to_parquet(bound, index=False)
    unrelated = config.runs_root / "feature_failure_report.parquet"
    pd.DataFrame([{"status": "wrong"}]).to_parquet(unrelated, index=False)
    atomic_write_json(
        run_root / "development_provenance.json",
        {"feature_integrity": {"root": str(feature_root)}},
    )

    assert _run_feature_failure_path(run_root) == bound.resolve()


def test_ptb_mimic_transfer_matrix_keeps_reference_performance_separate() -> None:
    feature_id = "qrs_t_hausdorff_distance"
    ptb_marker_rows = pd.DataFrame(
        [
            {
                "endpoint_id": "ptb_ischemia",
                "feature_id": feature_id,
                "evaluation_split": "train",
                "direction": 1,
                "roc_auc": 0.70,
            },
            {
                "endpoint_id": "ptb_ischemia",
                "feature_id": feature_id,
                "evaluation_split": "validation",
                "direction": 1,
                "roc_auc": 0.65,
                "rank_biserial": 0.30,
                "missing_fraction": 0.02,
                "inference_status": "formal_inference",
            },
        ]
    )
    ptb_score_id = "ptb_ischemia__ptb_frozen_score"
    ptb_score_rows = pd.DataFrame(
        [
            {
                "endpoint_id": "ptb_ischemia",
                "feature_id": ptb_score_id,
                "evaluation_split": "train",
                "direction": 1,
                "roc_auc": 0.75,
            },
            {
                "endpoint_id": "ptb_ischemia",
                "feature_id": ptb_score_id,
                "evaluation_split": "validation",
                "direction": 1,
                "roc_auc": 0.68,
                "rank_biserial": 0.36,
                "missing_fraction": 0.10,
                "inference_status": "formal_inference",
            },
        ]
    )
    mimic_marker_rows = pd.DataFrame(
        [
            {
                "endpoint_id": "acute_mi_high_specificity",
                "feature_id": feature_id,
                "panel_id": "primary_core",
                "evaluation_split": "validation",
                "direction": -1,
                "roc_auc": 0.61,
                "rank_biserial": -0.22,
                "missingness": 0.03,
                "inference_status": "formal_inference",
            }
        ]
    )
    mimic_score_id = (
        "acute_mi_high_specificity__ptb_ischemia"
        "__ptb_frozen_transfer"
    )
    mimic_score_rows = pd.DataFrame(
        [
            {
                "endpoint_id": "acute_mi_high_specificity",
                "ptb_endpoint_id": "ptb_ischemia",
                "feature_id": mimic_score_id,
                "analysis_type": "ptb_frozen_transfer",
                "evaluation_split": "validation",
                "direction": 1,
                "roc_auc": 0.62,
                "rank_biserial": 0.24,
                "missingness": 0.12,
                "inference_status": "formal_inference",
                "components": "a|b|c",
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "endpoint_id": "acute_mi_high_specificity",
                "feature_id": feature_id,
                "analysis_type": "single_marker",
                "train_direction": -1,
                "validation_direction": -1,
                "direction_agreement": True,
                "train_validation_auc_gap": 0.04,
            },
            {
                "endpoint_id": "acute_mi_high_specificity",
                "feature_id": mimic_score_id,
                "analysis_type": "ptb_frozen_transfer",
                "train_direction": 1,
                "validation_direction": 1,
                "direction_agreement": True,
                "train_validation_auc_gap": 0.03,
            },
        ]
    )

    matrix = _build_transfer_matrix(
        ptb_marker_rows,
        ptb_score_rows,
        mimic_marker_rows,
        mimic_score_rows,
        stability,
    )

    assert set(matrix["transfer_type"]) == {
        "shared_single_marker",
        "ptb_frozen_score",
    }
    marker = matrix.loc[
        matrix["transfer_type"].eq("shared_single_marker")
    ].iloc[0]
    assert marker["ptb_train_direction"] == 1
    assert marker["mimic_train_direction"] == -1
    assert not bool(marker["train_direction_agreement"])
    assert not bool(marker["pooled_performance_permitted"])
    assert not bool(marker["direct_auc_delta_permitted"])
    assert "auc_delta" not in matrix.columns
    assert marker["ptb_reference_type"] != marker["mimic_reference_type"]


def test_phenome_uses_phenotype_ecg_and_only_explicit_control_index() -> None:
    features = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "study_id": [10, 11, 20, 30],
            "split": ["train"] * 4,
            "is_control_index": [True, False, True, False],
            "is_case_index": [False, True, False, False],
            "marker": [-100.0, 10.0, 0.0, -500.0],
        }
    )
    current = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [11],
            "split": ["train"],
            "phenotype_id": ["P1"],
            "body_system": ["circulatory"],
        }
    )
    values, labels, subjects = _case_control_values(
        features,
        current,
        "P1",
        "marker",
        history_subjects={1},
    )
    assert values.tolist() == [10.0, 0.0]
    assert labels.tolist() == [1, 0]
    assert subjects.tolist() == [1, 2]

    records = _screen_split(
        features,
        current,
        pd.DataFrame(
            {
                "subject_id": [1],
                "body_system": ["circulatory"],
            }
        ),
        phenotype_ids=("P1",),
        feature_ids=("marker",),
        panel_map={"marker": "primary_core"},
        definition_lookup=pd.DataFrame(
            {
                "phenotype_id": ["P1"],
                "display_name": ["Test phenotype"],
                "body_system": ["circulatory"],
            }
        ).set_index("phenotype_id"),
        min_inference=1,
        min_formal=1,
        progress=None,
    )
    assert len(records) == 1
    assert records[0]["n_case"] == 1
    assert records[0]["n_control"] == 1
    assert records[0]["raw_auc"] == pytest.approx(1.0)


def test_phenome_projected_parquet_read_is_pyarrow_compatible(
    tmp_path: Path,
) -> None:
    path = tmp_path / "phenome-index.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "study_id": [10, 20],
            "split": ["train", "validation"],
        }
    ).to_parquet(path, index=False)

    projected = _read_parquet_columns(
        path,
        ("subject_id", "split"),
    )

    assert projected.to_dict(orient="list") == {
        "subject_id": [1, 2],
        "split": ["train", "validation"],
    }


def test_frame_checkpoint_is_reused_and_detects_data_corruption(
    tmp_path: Path,
) -> None:
    calls = []

    def compute() -> pd.DataFrame:
        calls.append("computed")
        return pd.DataFrame({"value": [1.0, 2.0]})

    first, resumed = checkpointed_frame_chunk(
        tmp_path,
        namespace="train",
        chunk_index=0,
        input_hash="scientific-input",
        item_keys=("P1", "P2"),
        compute=compute,
    )
    assert not resumed
    second, resumed = checkpointed_frame_chunk(
        tmp_path,
        namespace="train",
        chunk_index=0,
        input_hash="scientific-input",
        item_keys=("P1", "P2"),
        compute=lambda: pytest.fail("completed checkpoint was recomputed"),
    )
    assert resumed
    assert calls == ["computed"]
    pd.testing.assert_frame_equal(second, first)

    pd.DataFrame({"value": [9.0]}).to_parquet(
        tmp_path / "train" / "part-000000.parquet",
        index=False,
    )
    with pytest.raises(RuntimeError, match="data hash mismatch"):
        checkpointed_frame_chunk(
            tmp_path,
            namespace="train",
            chunk_index=0,
            input_hash="scientific-input",
            item_keys=("P1", "P2"),
            compute=compute,
        )


def test_clinical_evidence_group_checkpoint_resumes_atomically(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = _load_evidence_checkpoint(
        checkpoint_path,
        input_hash="events-v1",
        config_hash="config-v1",
    )
    artifact_path = tmp_path / "labs.parquet"
    pd.DataFrame({"subject_id": [1, 2]}).to_parquet(
        artifact_path,
        index=False,
    )
    _save_evidence_group(
        checkpoint_path,
        checkpoint,
        group_id="laboratory",
        artifacts=[
            EvidenceArtifact(
                name="labs",
                path=artifact_path,
                row_count=2,
            )
        ],
    )
    reloaded = _load_evidence_checkpoint(
        checkpoint_path,
        input_hash="events-v1",
        config_hash="config-v1",
    )
    resumed = _resume_evidence_group(
        reloaded,
        group_id="laboratory",
    )
    assert resumed == [
        EvidenceArtifact(
            name="labs",
            path=artifact_path.resolve(),
            row_count=2,
        )
    ]

    # A changed artifact means the checkpoint can no longer describe what is
    # on disk, so the group is recomputed rather than resumed. It used to
    # raise instead, which ended the run over a file the stage can rebuild --
    # and did, once the equivalence shadow build was found writing into the
    # canonical quarantine directory.
    notes: list[str] = []
    pd.DataFrame({"subject_id": [9]}).to_parquet(
        artifact_path,
        index=False,
    )
    assert (
        _resume_evidence_group(
            reloaded,
            group_id="laboratory",
            progress=notes.append,
        )
        is None
    )
    assert any("neu berechnet" in note for note in notes)

    artifact_path.unlink()
    assert (
        _resume_evidence_group(reloaded, group_id="laboratory") is None
    )


def test_event_evidence_equivalence_is_exact_and_resumable(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.parquet"
    event_backed = tmp_path / "event.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "value": [2.0, 2.0, np.nan],
        }
    ).to_parquet(canonical, index=False)
    pd.DataFrame(
        {
            "subject_id": [2, 1, 1],
            "value": [np.nan, 2.0, 2.0],
        }
    ).to_parquet(event_backed, index=False)
    checkpoint_path = tmp_path / "equivalence.json"
    checkpoint = _load_equivalence_checkpoint(
        checkpoint_path,
        input_hash="migration-v1",
    )
    with duckdb.connect(":memory:") as connection:
        comparison, resumed = _compare_or_resume_table(
            connection,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            table_name="labs.parquet",
            canonical_path=canonical,
            event_backed_path=event_backed,
        )
        assert not resumed
        assert comparison["equivalent"]
        comparison, resumed = _compare_or_resume_table(
            connection,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            table_name="labs.parquet",
            canonical_path=canonical,
            event_backed_path=event_backed,
        )
        assert resumed
        assert comparison["equivalent"]

        pd.DataFrame(
            {
                "subject_id": [1, 2],
                "value": [2.0, np.nan],
            }
        ).to_parquet(event_backed, index=False)
        comparison, resumed = _compare_or_resume_table(
            connection,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            table_name="labs.parquet",
            canonical_path=canonical,
            event_backed_path=event_backed,
        )
        assert not resumed
        assert not comparison["equivalent"]
        assert comparison["canonical_row_count"] == 3
        assert comparison["event_backed_row_count"] == 2


def test_event_evidence_scope_allows_only_additive_audit_context(
    tmp_path: Path,
) -> None:
    canonical = {
        "laboratory_evidence.parquet": tmp_path / "canonical-labs.parquet",
        "medication_evidence.parquet": (
            tmp_path / "canonical-medications.parquet"
        ),
    }
    additive_name = "medication_order_administration_context.parquet"
    event_backed = {
        **canonical,
        additive_name: tmp_path / "event-medication-context.parquet",
    }

    compared, additive = _classify_evidence_table_scope(
        canonical,
        event_backed,
    )
    assert compared == sorted(canonical)
    assert additive == [additive_name]

    with pytest.raises(RuntimeError, match="canonical_only"):
        _classify_evidence_table_scope(
            canonical,
            {
                "laboratory_evidence.parquet": (
                    tmp_path / "event-labs.parquet"
                )
            },
        )
    with pytest.raises(RuntimeError, match="unexpected_event_only"):
        _classify_evidence_table_scope(
            canonical,
            {
                **canonical,
                "unplanned_evidence.parquet": (
                    tmp_path / "unplanned.parquet"
                ),
            },
        )


def test_medication_window_display_tie_requires_real_minimum_candidates(
    tmp_path: Path,
) -> None:
    canonical_windows = tmp_path / "canonical-windows.parquet"
    event_windows = tmp_path / "event-windows.parquet"
    canonical_medications = tmp_path / "canonical-medications.parquet"
    event_medications = tmp_path / "event-medications.parquet"
    master = tmp_path / "master.parquet"
    checkpoint_path = tmp_path / "equivalence.json"
    base_window = {
        "subject_id": [1],
        "study_id": [10],
        "medication_class": ["antiplatelet"],
        "administered_near_ecg": [True],
        "nearest_delta_hours": [1.0],
    }
    pd.DataFrame(
        {
            **base_window,
            "nearest_medication": ["Aspirin"],
        }
    ).to_parquet(canonical_windows, index=False)
    pd.DataFrame(
        {
            **base_window,
            "nearest_medication": ["Clopidogrel"],
        }
    ).to_parquet(event_windows, index=False)
    medications = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "medication_class": ["antiplatelet", "antiplatelet"],
            "medication_raw": ["Aspirin", "Clopidogrel"],
            "event_time": pd.to_datetime(
                ["2024-01-01 11:00:00", "2024-01-01 11:00:00"]
            ),
        }
    )
    medications.to_parquet(canonical_medications, index=False)
    medications.iloc[::-1].to_parquet(event_medications, index=False)
    pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [10],
            "ecg_time": pd.to_datetime(["2024-01-01 12:00:00"]),
        }
    ).to_parquet(master, index=False)
    checkpoint = _load_equivalence_checkpoint(
        checkpoint_path,
        input_hash="medication-tie-v1",
    )
    with duckdb.connect(":memory:") as connection:
        comparison, resumed = _compare_or_resume_medication_windows(
            connection,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            canonical_path=canonical_windows,
            event_backed_path=event_windows,
            canonical_medication_path=canonical_medications,
            event_backed_medication_path=event_medications,
            master_path=master,
        )
        assert not resumed
        assert comparison["equivalent"]
        assert not comparison["full_table_equivalent"]
        assert comparison["nondiagnostic_tie_audit"]["valid"]

        pd.DataFrame(
            {
                **base_window,
                "nearest_medication": ["Invented Drug"],
            }
        ).to_parquet(event_windows, index=False)
        comparison, resumed = _compare_or_resume_medication_windows(
            connection,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            canonical_path=canonical_windows,
            event_backed_path=event_windows,
            canonical_medication_path=canonical_medications,
            event_backed_medication_path=event_medications,
            master_path=master,
        )
    assert not resumed
    assert not comparison["equivalent"]
    assert (
        comparison["nondiagnostic_tie_audit"][
            "event_backed_choice_not_at_minimum"
        ]
        == 1
    )


def test_echo_nearest_tie_is_deterministic_and_prefers_earlier_study(
    tmp_path: Path,
) -> None:
    master_path = tmp_path / "master.parquet"
    echo_path = tmp_path / "echo.parquet"
    pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [100],
            "ecg_time": pd.to_datetime(["2024-01-10 12:00:00"]),
        }
    ).to_parquet(master_path, index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "measurement_id": [20, 10],
            "echo_time": pd.to_datetime(
                ["2024-01-11 12:00:00", "2024-01-09 12:00:00"]
            ),
            "test_type": ["tte", "tte"],
            "identity_conflict": [False, False],
        }
    ).to_parquet(echo_path, index=False)

    query = _ecg_echo_match_query(
        master_path.as_posix(),
        echo_path.as_posix(),
    )
    with duckdb.connect(":memory:") as connection:
        row = connection.execute(query).fetchone()

    # Audit D10: on equal distance the registry promises the earlier study.
    assert row[2] == 10
    assert row[4] == pytest.approx(-24.0)
    assert row[5] == 10
    assert row[7] == 20


def test_phenome_train_screen_resumes_after_atomic_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import phenome as phenome_module

    features = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "study_id": [10, 20, 30, 40],
            "split": ["train"] * 4,
            "is_control_index": [False, False, True, True],
            "is_case_index": [True, True, False, False],
            "marker": [2.0, 3.0, 0.0, 0.5],
        }
    )
    current = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "study_id": [10, 20],
            "split": ["train", "train"],
            "phenotype_id": ["P1", "P2"],
        }
    )
    history = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "body_system": ["circulatory", "circulatory"],
        }
    )
    definitions = pd.DataFrame(
        {
            "phenotype_id": ["P1", "P2"],
            "display_name": ["Phenotype one", "Phenotype two"],
            "body_system": ["circulatory", "circulatory"],
        }
    ).set_index("phenotype_id")
    original_screen = phenome_module._screen_split
    calls: list[tuple[str, ...]] = []

    def counted_screen(*args, **kwargs):
        calls.append(tuple(kwargs["phenotype_ids"]))
        return original_screen(*args, **kwargs)

    monkeypatch.setattr(phenome_module, "_screen_split", counted_screen)
    monkeypatch.setattr(phenome_module, "PHENOME_TRAIN_CHUNK_SIZE", 1)

    with pytest.raises(JobPauseRequested):
        _checkpointed_train_screen(
            tmp_path,
            input_hash="phenome-input",
            features=features,
            current=current,
            history_body=history,
            phenotype_ids=("P1", "P2"),
            feature_ids=("marker",),
            panel_map={"marker": "primary_core"},
            definition_lookup=definitions,
            min_inference=1,
            min_formal=1,
            progress=lambda message: (_ for _ in ()).throw(
                JobPauseRequested(message)
            ),
        )
    assert calls == [("P1",)]

    resumed = _checkpointed_train_screen(
        tmp_path,
        input_hash="phenome-input",
        features=features,
        current=current,
        history_body=history,
        phenotype_ids=("P1", "P2"),
        feature_ids=("marker",),
        panel_map={"marker": "primary_core"},
        definition_lookup=definitions,
        min_inference=1,
        min_formal=1,
        progress=None,
    )
    assert calls == [("P1",), ("P2",)]
    assert resumed["phenotype_id"].tolist() == ["P1", "P2"]


def test_phenome_fast_mann_whitney_matches_scipy_with_ties() -> None:
    cases = np.asarray([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    controls = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    auc, p_value = _auc_and_p_from_sorted_controls(
        cases,
        np.sort(controls),
    )
    reference = mannwhitneyu(
        cases,
        controls,
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    )
    assert auc == pytest.approx(
        reference.statistic / (len(cases) * len(controls))
    )
    assert p_value == pytest.approx(reference.pvalue, rel=1e-12)
    future_auc, future_p = _future_auc_and_p(cases, np.sort(controls))
    assert future_auc == pytest.approx(auc)
    assert future_p == pytest.approx(reference.pvalue, rel=1e-12)

    constant_auc, constant_p = _auc_and_p_from_sorted_controls(
        np.ones(5),
        np.ones(7),
    )
    assert constant_auc == pytest.approx(0.5)
    assert constant_p == pytest.approx(1.0)


def test_phenome_parallel_replication_matches_sequential_results() -> None:
    rng = np.random.default_rng(20260726)
    feature_rows = []
    current_rows = []
    history_rows = []
    for split, subject_start in (("train", 1), ("validation", 101)):
        for offset in range(40):
            subject_id = subject_start + offset
            study_id = subject_id * 10
            positive = offset < 20
            feature_rows.append(
                {
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "split": split,
                    "is_control_index": True,
                    "marker_up": float(positive) + rng.normal(0.0, 0.2),
                    "marker_down": -float(positive) + rng.normal(0.0, 0.2),
                }
            )
            if positive:
                current_rows.append(
                    {
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "split": split,
                        "phenotype_id": "P1",
                    }
                )
                history_rows.append(
                    {
                        "subject_id": subject_id,
                        "body_system": "circulatory",
                    }
                )
    train_hits = pd.DataFrame(
        [
            {
                "phenotype_id": "P1",
                "feature_id": "marker_up",
                "body_system": "circulatory",
                "panel_id": "primary_core",
                "q_value": 0.001,
                "aligned_auc": 0.95,
                "direction": 1,
            },
            {
                "phenotype_id": "P1",
                "feature_id": "marker_down",
                "body_system": "circulatory",
                "panel_id": "sentinel",
                "q_value": 0.002,
                "aligned_auc": 0.94,
                "direction": -1,
            },
        ]
    )
    arguments = {
        "train_hits": train_hits,
        "features": pd.DataFrame(feature_rows),
        "current": pd.DataFrame(current_rows),
        "history_body": pd.DataFrame(history_rows),
        "definition_lookup": pd.DataFrame(
            {
                "phenotype_id": ["P1"],
                "display_name": ["Test phenotype"],
            }
        ).set_index("phenotype_id"),
        "min_inference": 10,
        "min_formal": 20,
        "bootstrap_iterations": 20,
        "random_seed": 20260726,
        "progress": None,
    }

    sequential = _replicate_hits(**arguments, worker_count=1)
    parallel = _replicate_hits(**arguments, worker_count=2)
    chunked = [
        *_replicate_hits(
            **{
                **arguments,
                "train_hits": train_hits.iloc[:1],
            },
            worker_count=1,
            seed_offset=0,
        ),
        *_replicate_hits(
            **{
                **arguments,
                "train_hits": train_hits.iloc[1:],
            },
            worker_count=1,
            seed_offset=1,
        ),
    ]

    pd.testing.assert_frame_equal(
        pd.DataFrame(parallel),
        pd.DataFrame(sequential),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(chunked),
        pd.DataFrame(sequential),
        check_exact=True,
    )


def test_phenome_replication_prepares_indices_once_across_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import phenome as phenome_module

    feature_rows = []
    current_rows = []
    history_rows = []
    for split, start in (("train", 1), ("validation", 101)):
        for offset in range(4):
            subject_id = start + offset
            feature_rows.append(
                {
                    "subject_id": subject_id,
                    "study_id": subject_id * 10,
                    "split": split,
                    "is_control_index": offset >= 2,
                    "marker_up": float(offset < 2),
                    "marker_down": -float(offset < 2),
                }
            )
        for offset, phenotype_id in enumerate(("P1", "P2")):
            subject_id = start + offset
            current_rows.append(
                {
                    "subject_id": subject_id,
                    "study_id": subject_id * 10,
                    "split": split,
                    "phenotype_id": phenotype_id,
                }
            )
            history_rows.append(
                {
                    "subject_id": subject_id,
                    "body_system": "circulatory",
                }
            )
    train_hits = pd.DataFrame(
        [
            {
                "phenotype_id": "P1",
                "feature_id": "marker_up",
                "body_system": "circulatory",
                "panel_id": "primary_core",
                "q_value": 0.001,
                "aligned_auc": 1.0,
                "direction": 1,
            },
            {
                "phenotype_id": "P2",
                "feature_id": "marker_down",
                "body_system": "circulatory",
                "panel_id": "sentinel",
                "q_value": 0.001,
                "aligned_auc": 1.0,
                "direction": -1,
            },
        ]
    )
    original_index = phenome_module._phenome_replication_row_index
    index_calls = 0

    def counted_index(*args, **kwargs):
        nonlocal index_calls
        index_calls += 1
        return original_index(*args, **kwargs)

    monkeypatch.setattr(
        phenome_module,
        "_phenome_replication_row_index",
        counted_index,
    )
    monkeypatch.setattr(
        phenome_module,
        "PHENOME_REPLICATION_CHUNK_SIZE",
        1,
    )

    result = _checkpointed_validation_replication(
        tmp_path,
        input_hash="replication-cache",
        train_hits=train_hits,
        features=pd.DataFrame(feature_rows),
        current=pd.DataFrame(current_rows),
        history_body=pd.DataFrame(history_rows),
        definition_lookup=pd.DataFrame(
            {
                "phenotype_id": ["P1", "P2"],
                "display_name": ["Phenotype one", "Phenotype two"],
            }
        ).set_index("phenotype_id"),
        min_inference=1,
        min_formal=1,
        bootstrap_iterations=0,
        random_seed=20260726,
        worker_count=1,
        progress=None,
    )

    assert len(result) == 2
    assert index_calls == 2


def test_phenome_bootstraps_only_validation_replicated_hits() -> None:
    train_hits = pd.DataFrame(
        {
            "phenotype_id": ["P1", "P2"],
            "feature_id": ["marker_a", "marker_b"],
            "panel_id": ["primary_core", "sentinel"],
            "replication_seed_index": [1, 2],
        }
    )
    validation = pd.DataFrame(
        {
            "phenotype_id": ["P1", "P2"],
            "feature_id": ["marker_a", "marker_b"],
            "panel_id": ["primary_core", "sentinel"],
            "replicated": [False, True],
            "roc_auc_ci_low": [np.nan, np.nan],
            "roc_auc_ci_high": [np.nan, np.nan],
            "bootstrap_iterations": [0, 0],
            "bootstrap_successful_iterations": [0, 0],
            "bootstrap_unit": ["subject_id", "subject_id"],
        }
    )

    selected = _replicated_bootstrap_hits(train_hits, validation)
    assert selected["phenotype_id"].tolist() == ["P2"]
    assert selected["replication_seed_index"].tolist() == [2]

    bootstrapped = pd.DataFrame(
        {
            "phenotype_id": ["P2"],
            "feature_id": ["marker_b"],
            "panel_id": ["sentinel"],
            "roc_auc_ci_low": [0.61],
            "roc_auc_ci_high": [0.78],
            "bootstrap_iterations": [1000],
            "bootstrap_successful_iterations": [998],
            "bootstrap_unit": ["subject_id"],
        }
    )
    output = _overlay_replicated_bootstraps(
        validation,
        bootstrapped,
    )

    assert np.isnan(output.loc[0, "roc_auc_ci_low"])
    assert output.loc[0, "bootstrap_iterations"] == 0
    assert (
        output.loc[0, "bootstrap_scope"]
        == "not_requested_not_replicated"
    )
    assert output.loc[1, "roc_auc_ci_low"] == pytest.approx(0.61)
    assert output.loc[1, "roc_auc_ci_high"] == pytest.approx(0.78)
    assert output.loc[1, "bootstrap_iterations"] == 1000
    assert output.loc[1, "bootstrap_successful_iterations"] == 998
    assert (
        output.loc[1, "bootstrap_scope"]
        == "subject_cluster_replicated_hit"
    )


def test_discovery_replication_requires_practical_stability() -> None:
    auc_rows = pd.DataFrame(
        {
            "train_aligned_auc": [0.65, 0.65, 0.75, 0.65],
            "roc_auc": [0.61, 0.59, 0.62, 0.64],
            "q_value": [0.01, 0.01, 0.01, 0.20],
            "direction_consistent": [True, True, True, True],
            "inference_status": ["formal_inference"] * 4,
        }
    )
    auc_status = add_auc_replication_status(
        auc_rows,
        validation_fdr_alpha=0.05,
    )

    assert auc_status["fdr_direction_replicated"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert auc_status["validation_stable"].tolist() == [
        True,
        False,
        False,
        True,
    ]
    assert auc_status["replicated"].tolist() == [
        True,
        False,
        False,
        False,
    ]

    effect_rows = pd.DataFrame(
        {
            "train_effect": [0.30, 0.30, 0.45],
            "validation_effect": [0.25, 0.10, 0.30],
            "validation_q_value": [0.01, 0.01, 0.01],
            "direction_consistent": [True, True, True],
            "inference_status": ["formal_inference"] * 3,
        }
    )
    effect_status = add_effect_replication_status(
        effect_rows,
        validation_fdr_alpha=0.05,
    )
    assert effect_status["replicated"].tolist() == [
        True,
        False,
        False,
    ]


def test_discovery_auc_bootstrap_matches_full_auc_interval() -> None:
    rng = np.random.default_rng(20260729)
    subjects = np.arange(120)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 60)
    values = labels + rng.normal(0.0, 0.8, len(labels))
    arguments = {
        "subject_ids": subjects,
        "values": values,
        "labels": labels,
        "direction": 1,
        "thresholds": {},
        "iterations": 200,
        "seed": 90210,
    }

    full = marker_bootstrap_intervals(**arguments)
    discovery = discovery_marker_bootstrap_intervals(**arguments)

    assert discovery["roc_auc_ci_low"] == pytest.approx(
        full["roc_auc_ci_low"],
        abs=0.0,
    )
    assert discovery["roc_auc_ci_high"] == pytest.approx(
        full["roc_auc_ci_high"],
        abs=0.0,
    )
    assert (
        discovery["bootstrap_successful_iterations"]
        == full["bootstrap_successful_iterations"]
    )
    assert np.isnan(discovery["average_precision_ci_low"])
    assert (
        discovery["bootstrap_metric_scope"]
        == "roc_auc_and_aligned_rank_biserial"
    )


def test_context_domains_use_deterministic_subject_indices_without_test_rows(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    rows = []
    feature_rows = []
    # Wide enough that the train split still holds several exposed subjects
    # once subject 2 (exposed only on its second ECG) no longer counts as
    # exposed at a target-blind index.
    for subject_id in range(1, 13):
        split = "train" if subject_id <= 8 else "validation"
        positive = subject_id % 2 == 0
        for offset in (0, 1):
            study_id = subject_id * 10 + offset
            target = float(subject_id + (0.2 if offset == 0 else 0.0))
            rows.append(
                {
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "split": split,
                    "ecg_time": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=study_id),
                    "medication_assessed": True,
                    # Exposure is a property of the encounter, so it is
                    # present on both recordings -- except for subject 2,
                    # which is exposed only on its second ECG. That subject is
                    # the probe for outcome-dependent index selection below.
                    "med_exposed": positive
                    and (subject_id != 2 or offset == 1),
                    "lab_value": target,
                    "lab_distance": 5.0 if offset == 0 else 1.0,
                }
            )
            feature_rows.append(
                {
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "split": split,
                    "extraction_status": "ok",
                    "marker": (
                        float(positive)
                        if offset == 1
                        else target
                    ),
                }
            )
    rows.append(
        {
            "subject_id": 99,
            "study_id": 999,
            "split": "locked_test",
            "ecg_time": pd.Timestamp("2026-01-01"),
            "medication_assessed": True,
            "med_exposed": True,
            "lab_value": 999.0,
            "lab_distance": 0.0,
        }
    )
    feature_rows.append(
        {
            "subject_id": 99,
            "study_id": 999,
            "split": "locked_test",
            "extraction_status": "ok",
            "marker": 999.0,
        }
    )
    context_path = tmp_path / "context.parquet"
    feature_path = tmp_path / "features.parquet"
    index_path = tmp_path / "association-index.parquet"
    pd.DataFrame(rows).to_parquet(context_path, index=False)
    pd.DataFrame(feature_rows).to_parquet(feature_path, index=False)
    definitions = (
        ContextAssociationDefinition(
            association_id="medication_test",
            display_name="Medication",
            domain="medication",
            target_kind="binary",
            value_sql="CASE WHEN med_exposed THEN 1.0 ELSE 0.0 END",
            eligibility_sql="medication_assessed",
            temporal_role="retrospective_exposure_context",
            index_policy="first eligible ECG, target-blind",
        ),
        ContextAssociationDefinition(
            association_id="laboratory_test",
            display_name="Laboratory",
            domain="laboratory",
            target_kind="continuous",
            value_sql="lab_value",
            eligibility_sql="lab_value IS NOT NULL",
            temporal_role="concurrent_clinical_measurement",
            index_policy="first eligible ECG, target-blind",
            distance_sql="ABS(lab_distance)",
        ),
    )
    _build_association_index(
        config,
        feature_glob=str(feature_path),
        context_path=context_path,
        definitions=definitions,
        output_path=index_path,
    )
    index = pd.read_parquet(index_path)

    assert not index["split"].eq("locked_test").any()
    assert not index.duplicated(["association_id", "subject_id"]).any()
    # Subject 2 is exposed only on its *second* ECG (study 21). A target-blind
    # index must still take the first eligible ECG (study 20); taking 21 would
    # be outcome-dependent index selection, which silently enriched the
    # exposed arm with whichever recording carried the exposure.
    exposed_subject_two = index.loc[
        (index["association_id"] == "medication_test")
        & (index["subject_id"] == 2)
    ].iloc[0]
    assert exposed_subject_two["study_id"] == 20
    assert exposed_subject_two["target_value"] == 0.0
    assert (
        exposed_subject_two["index_selection_strategy"]
        == "first_eligible_ecg_target_blind"
    )
    # Same for the continuous target: study 11 is the closer measurement, but
    # the index is the earlier ECG regardless. The realised distance stays
    # available as a column so pairing quality remains auditable.
    nearest_subject_one = index.loc[
        (index["association_id"] == "laboratory_test")
        & (index["subject_id"] == 1)
    ].iloc[0]
    assert nearest_subject_one["study_id"] == 10
    assert nearest_subject_one["target_time_distance"] == 5.0
    # The index must be identical across every association: it is chosen once
    # per subject, before any target is looked at.
    per_association = index.pivot_table(
        index="subject_id",
        columns="association_id",
        values="study_id",
        aggfunc="first",
    ).dropna()
    assert (
        per_association["medication_test"]
        == per_association["laboratory_test"]
    ).all()

    screened = _screen_association_domains(
        config,
        feature_glob=str(feature_path),
        index_path=index_path,
        definitions=definitions,
        feature_ids=("marker",),
        panel_map={"marker": "primary_core"},
        split="train",
        min_inference=1,
        min_formal=2,
        selected_pairs=None,
        progress=None,
    )
    assert set(screened["domain"]) == {"laboratory", "medication"}
    assert screened["locked_test_scanned"].eq(False).all()
    assert screened["inference_status"].eq("formal_inference").all()
    assert (
        screened["index_selection_strategy"]
        .eq("first_eligible_ecg_target_blind")
        .all()
    )
    # The laboratory definition declares a time distance, so its realised
    # pairing quality is reported; the medication definition declares none and
    # is therefore NaN rather than a fabricated zero.
    by_domain = screened.set_index("domain")[
        "target_pairing_distance_median"
    ]
    assert by_domain.loc["laboratory"] == 5.0
    assert pd.isna(by_domain.loc["medication"])


def test_default_context_domains_are_explicitly_non_pathology() -> None:
    definitions = default_context_association_definitions()
    assert {item.domain for item in definitions} == {
        "laboratory",
        "medication",
        "procedure",
        "outcome",
        "demographic",
        "socioeconomic",
        "vitals_ed",
        "care_setting",
        "vitals_ed_trend",
        "infection",
        "vitals_pre_ecg",
        "ed_context",
        "icu_vitals_trend",
        "icu_context",
        "note_radiology_findings",
        "note_risk_factors",
        "note_diagnosis_mentions",
        "note_head_imaging_findings",
        "note_cta_chest_findings",
        "note_functional_status",
        # Audit 5.2.2 and 5.2.9.
        "echocardiography_continuous",
        "icu_invasive_hemodynamics",
    }
    assert all("patholog" not in item.domain for item in definitions)


def test_baseline_registry_is_bound_to_local_source_lock() -> None:
    registry = BaselineRegistry()
    assert len(registry.source_lock_hash) == 64
    assert registry.source_lock_path.is_file()
    assert file_sha256(registry.source_lock_path) == registry.source_lock_hash
    assert registry.source_lock["runtime_network_access"] is False
    assert "ase_right_heart_ph_2025" in {
        source["source_id"] for source in registry.source_lock["sources"]
    }
    frame = registry.frame()
    assert frame["source_lock_sha256"].eq(
        registry.source_lock_hash
    ).all()
    assert len(registry.registry_hash) == 64
    for baseline_id in (
        "wellens_ecg_pattern",
        "de_winter_ecg_pattern",
        "s1q3t3_pattern",
    ):
        definition = registry.get(baseline_id)
        assert definition.kind.value == "expert_pattern"
        assert definition.source_url
        assert definition.source_url.rstrip("/") in {
            url.rstrip("/")
            for source in registry.source_lock["sources"]
            for url in source["urls"]
        }
    assert {
        "baseline_registry.parquet",
        "descriptive_validation_ranking.parquet",
        "methodology_registry.parquet",
        "ptb_transfer_matrix.parquet",
    } <= set(FROZEN_DEVELOPMENT_TABLES)


def test_baseline_definition_literature_reference_metrics_is_optional() -> None:
    """literature_reference_metrics (Work Package A4 infrastructure) must
    default to None for every existing entry — populating it with a real
    published Se/Sp figure and citation is a clinical-content decision
    reserved for a reviewed follow-up, not this generalization pass — and
    must round-trip cleanly through to_dict() when a caller does supply
    it."""
    registry = BaselineRegistry()
    for baseline_id in ("cornell_voltage", "p_terminal_force_v1", "qtc_fridericia"):
        definition = registry.get(baseline_id)
        assert definition.literature_reference_metrics is None
        assert "literature_reference_metrics" in definition.to_dict()

    example = _rule(
        "example_with_literature_metrics",
        ("echo_lvh_strict",),
        "Beispiel",
        "example_criterion",
        "example formula",
        (),
        literature_reference_metrics={
            "reported_sensitivity": 0.42,
            "reported_specificity": 0.95,
            "note": "illustrative only, not a real registered baseline",
        },
    )
    assert example.literature_reference_metrics == {
        "reported_sensitivity": 0.42,
        "reported_specificity": 0.95,
        "note": "illustrative only, not a real registered baseline",
    }
    assert example.to_dict()["literature_reference_metrics"][
        "reported_sensitivity"
    ] == 0.42


def test_methodology_registry_pins_current_reporting_and_bias_frameworks() -> None:
    registry = MethodologyRegistry()
    assert registry.source_lock_path.is_file()
    assert file_sha256(registry.source_lock_path) == registry.source_lock_hash
    assert registry.source_lock["runtime_network_access"] is False
    frame = registry.frame()
    assert set(frame["source_id"]) == {
        "aha_biomarker_utility_2026",
        "tripod_ai_2024",
        "probast_ai_2025",
    }
    assert frame["source_lock_sha256"].eq(
        registry.source_lock_hash
    ).all()
    assert frame["runtime_network_access"].eq(False).all()


def test_validation_rejects_stale_development_feature_inputs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    cohort_path = config.labels_root / "clinical_label_evidence.parquet"
    master_path = config.master_root / "ecg_master_index.parquet"
    master = pd.DataFrame(
        {
            "subject_id": [10, 20],
            "study_id": [1, 2],
            "split": ["train", "validation"],
            "ecg_time": pd.to_datetime(
                ["2026-01-01T00:00:00", "2026-01-02T00:00:00"]
            ),
            "waveform_path": ["one", "two"],
            "age": [60, 70],
            "sex": ["F", "M"],
            "machine_qrs_axis_deg": [10.0, 20.0],
            "machine_report": ["normal", "normal"],
            "machine_cart_id": ["a", "b"],
            "machine_time_mismatch": [False, False],
            "machine_time_mismatch_seconds": [0.0, 0.0],
            "machine_metadata_temporal_status": ["matched", "matched"],
            "machine_metadata_usable_for_primary": [True, True],
        }
    )
    master.to_parquet(master_path, index=False)
    initial_labels = pd.DataFrame(
        {
            "subject_id": [10],
            "study_id": [1],
            "label_id": ["endpoint"],
            "label_state": ["positive"],
            "eligibility_state": ["eligible"],
        }
    )
    initial_labels.to_parquet(cohort_path, index=False)

    panel_ids = ("primary_core", "sentinel")
    options = FeatureExtractionOptions(panel_ids=panel_ids, partition_size=250)
    selection = resolve_feature_panel_selection(config, options)
    panel_hash = str(selection["panel_hash"])
    producer_cohort = _load_feature_cohort(config, options)
    stage_input_payload = feature_stage_input_payload(
        _cohort_record_payloads(producer_cohort),
        panel_hash=panel_hash,
        feature_code_hash=feature_implementation_hash(),
        options=options,
        sensitivity_vcg_methods=tuple(config.sensitivity_vcg_methods),
        freeze_hash=None,
    )
    stage_input_hash = content_hash(stage_input_payload)
    assert current_development_feature_input_hash(
        config,
        panel_ids=panel_ids,
        partition_size=options.partition_size,
        panel_hash=panel_hash,
    ) == stage_input_hash

    root = config.features_root / "primary_core_sentinel" / "development"
    root.mkdir(parents=True)
    manifest_path = root / "features.manifest.json"
    index_path = root / "partition_index.json"
    atomic_write_json(
        manifest_path,
        {
            "config_hash": scientific_config_hash(config),
        },
    )
    atomic_write_json(
        index_path,
        {
            "stage_input_hash": stage_input_hash,
            "stage_input_payload": stage_input_payload,
            "partitions": [
                {"relative_path": "split=train/part-000000.parquet", "row_count": 250}
            ]
        },
    )
    integrity = {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "index_path": str(index_path),
        "panel_hash": panel_hash,
        "stage_input_hash": stage_input_hash,
    }
    _verify_current_development_feature_inputs(
        config,
        feature_integrity=integrity,
        panel_ids=panel_ids,
    )

    pd.concat(
        [
            initial_labels,
            pd.DataFrame(
                {
                    "subject_id": [20],
                    "study_id": [2],
                    "label_id": ["endpoint"],
                    "label_state": ["negative"],
                    "eligibility_state": ["eligible"],
                }
            ),
        ],
        ignore_index=True,
    ).to_parquet(cohort_path, index=False)
    with pytest.raises(RuntimeError, match="provenance is stale"):
        _verify_current_development_feature_inputs(
            config,
            feature_integrity=integrity,
            panel_ids=panel_ids,
        )


def test_preview_feature_input_contract_cannot_alias_full_development(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    master_path = config.master_root / "ecg_master_index.parquet"
    labels_path = config.labels_root / "clinical_label_evidence.parquet"
    pilot_index_path = config.labels_root / "endpoint_index_cohorts.parquet"
    rows = 4
    pd.DataFrame(
        {
            "subject_id": list(range(10, 10 + rows)),
            "study_id": list(range(1, rows + 1)),
            "split": ["train", "train", "validation", "validation"],
            "ecg_time": pd.date_range("2026-01-01", periods=rows),
            "waveform_path": [f"wave-{index}" for index in range(rows)],
            "age": [50, 51, 52, 53],
            "sex": ["F", "M", "F", "M"],
            "machine_qrs_axis_deg": [0.0] * rows,
            "machine_report": ["normal"] * rows,
            "machine_cart_id": ["cart"] * rows,
            "machine_time_mismatch": [False] * rows,
            "machine_time_mismatch_seconds": [0.0] * rows,
            "machine_metadata_temporal_status": ["matched"] * rows,
            "machine_metadata_usable_for_primary": [True] * rows,
        }
    ).to_parquet(master_path, index=False)
    labels = pd.DataFrame(
        {
            "subject_id": list(range(10, 10 + rows)),
            "study_id": list(range(1, rows + 1)),
            "label_id": ["endpoint"] * rows,
            "label_state": ["positive", "negative"] * 2,
            "eligibility_state": ["eligible"] * rows,
        }
    )
    labels.to_parquet(labels_path, index=False)
    labels.to_parquet(pilot_index_path, index=False)

    panel_ids = ("primary_core", "sentinel")
    full_options = FeatureExtractionOptions(
        panel_ids=panel_ids,
        partition_size=250,
    )
    preview_options = FeatureExtractionOptions(
        panel_ids=panel_ids,
        partition_size=250,
        max_records=2,
    )
    panel_hash = str(
        resolve_feature_panel_selection(config, full_options)["panel_hash"]
    )
    preview_payload = feature_stage_input_payload(
        _cohort_record_payloads(_load_feature_cohort(config, preview_options)),
        panel_hash=panel_hash,
        feature_code_hash=feature_implementation_hash(),
        options=preview_options,
        sensitivity_vcg_methods=tuple(config.sensitivity_vcg_methods),
        freeze_hash=None,
    )
    full_hash = current_development_feature_input_hash(
        config,
        panel_ids=panel_ids,
        partition_size=250,
        panel_hash=panel_hash,
    )

    assert preview_payload["max_records"] == 2
    assert preview_payload["capped_sampling_version"] is not None
    assert content_hash(preview_payload) != full_hash


def test_freeze_provenance_rejects_changed_validation_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "development_input.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    payload = {
        "payload_version": VALIDATION_INPUT_PAYLOAD_VERSION,
        "analysis_scope": "development_only",
        "locked_test_included": False,
        "clinical_event_evidence_provenance": {
            "status": "direct_event_backed"
        },
        "scientific_files": {
            "development_input": file_content_fingerprint(path)
        },
        "feature_files": [],
    }
    assert (
        verify_validation_input_payload_files(payload)["status"]
        == "verified"
    )

    pd.DataFrame({"value": [2]}).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="changed after analysis"):
        verify_validation_input_payload_files(payload)


def test_future_phenome_states_require_clean_history_and_followup() -> None:
    index_time = pd.Timestamp("2020-01-10")
    frame = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6, 7],
            "index_end_time": [index_time] * 7,
            "last_observed_time": [
                "2020-03-10",
                "2020-03-10",
                "2020-03-10",
                "2020-01-20",
                "2020-03-10",
                "2020-03-10",
                "2020-03-10",
            ],
            "date_of_death": [
                None,
                None,
                None,
                None,
                "2020-01-20",
                None,
                None,
            ],
            "sex": ["F", "F", "F", "F", "F", "M", "F"],
        }
    )
    events = pd.Series(
        {
            1: pd.Timestamp("2020-01-20"),
            2: pd.Timestamp("2020-01-05"),
            7: pd.Timestamp("2020-03-01"),
        }
    )
    definition = pd.Series({"male_only": False, "female_only": True})

    states = _future_state_masks(
        frame,
        events,
        definition,
        horizon_days=30,
    )

    assert set(np.flatnonzero(states["positive"])) == {0}
    assert set(np.flatnonzero(states["negative"])) == {2, 6}
    assert set(np.flatnonzero(states["unavailable"])) == {3, 4}
    assert set(np.flatnonzero(states["excluded_history"])) == {1}
    assert set(np.flatnonzero(states["excluded_sex"])) == {5}
    total = sum(
        np.asarray(states[key], dtype=int)
        for key in (
            "positive",
            "negative",
            "unavailable",
            "excluded_history",
            "excluded_sex",
        )
    )
    assert np.array_equal(total, np.ones(len(frame), dtype=int))
    assert set(np.flatnonzero(states["competing_death"])) == {4}
    assert 3 in set(np.flatnonzero(states["censored"]))


def test_future_phenome_excludes_date_only_same_day_death_event_ties() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": [1],
            "index_end_time": [pd.Timestamp("2020-01-01 12:00")],
            "last_observed_time": [pd.Timestamp("2020-02-01")],
            "date_of_death": [pd.Timestamp("2020-01-10")],
            "sex": ["F"],
        }
    )
    events = pd.Series({1: pd.Timestamp("2020-01-10 18:00")})
    definition = pd.Series({"male_only": False, "female_only": False})

    primary = _future_state_masks(
        frame, events, definition, horizon_days=30
    )
    event_first = _future_state_masks(
        frame,
        events,
        definition,
        horizon_days=30,
        same_day_tie_policy="event_first",
    )
    death_first = _future_state_masks(
        frame,
        events,
        definition,
        horizon_days=30,
        same_day_tie_policy="death_first",
    )

    assert primary["excluded_time_tie"].tolist() == [True]
    assert not primary["at_risk"].any()
    assert event_first["positive"].tolist() == [True]
    assert death_first["competing_death"].tolist() == [True]


def test_future_target_cap_is_selected_from_train_only() -> None:
    index_time = pd.Timestamp("2020-01-01")
    subjects = np.arange(1, 31)
    index = pd.DataFrame(
        {
            "subject_id": subjects,
            "split": ["train"] * 10 + ["validation"] * 20,
            "index_end_time": [index_time] * len(subjects),
            "last_observed_time": [pd.Timestamp("2021-01-01")] * len(subjects),
            "date_of_death": [pd.NaT] * len(subjects),
            "sex": ["F"] * len(subjects),
        }
    )
    records = []
    # A has the larger Train event count; B has far more Validation events.
    for subject_id in (1, 2, 3, 11):
        records.append(
            {
                "subject_id": subject_id,
                "phenotype_id": "A",
                "first_event_time": pd.Timestamp("2020-01-10"),
            }
        )
    for subject_id in (4, 5, *range(12, 30)):
        records.append(
            {
                "subject_id": subject_id,
                "phenotype_id": "B",
                "first_event_time": pd.Timestamp("2020-01-10"),
            }
        )
    definitions = pd.DataFrame(
        {
            "phenotype_id": ["A", "B"],
            "display_name": ["A", "B"],
            "body_system": ["x", "x"],
            "male_only": [False, False],
            "female_only": [False, False],
        }
    )

    summary, selected = _future_label_summary(
        index,
        pd.DataFrame(records),
        definitions,
        horizons_days=(30,),
        min_inference=2,
        max_phenotypes=1,
    )

    assert selected == (("A", 30),)
    assert summary["validation_outcomes_used_for_selection"].eq(False).all()


def test_current_phenome_uses_prior_history_but_not_future_diagnoses(
    tmp_path: Path,
) -> None:
    standardized = tmp_path / "standardized"
    standardized.mkdir()
    output = tmp_path / "phenome"
    output.mkdir()
    diagnosis_path = standardized / "diagnoses_icd.parquet"
    admissions_path = standardized / "admissions.parquet"
    master_path = tmp_path / "master.parquet"
    mapping_path = tmp_path / "mapping.parquet"
    definitions_path = tmp_path / "definitions.parquet"
    ccsr_path = tmp_path / "ccsr.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 11, 12],
            "icd_version": [10, 10, 10],
            "icd_code": ["A", "B", "C"],
            "seq_num": [1, 1, 1],
        }
    ).to_parquet(diagnosis_path, index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 11, 12],
            "admittime": pd.to_datetime(
                ["2020-01-01", "2020-01-10", "2020-02-01"]
            ),
            "dischtime": pd.to_datetime(
                ["2020-01-05", "2020-01-12", "2020-02-03"]
            ),
        }
    ).to_parquet(admissions_path, index=False)
    pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [101],
            "hadm_id": [11],
            "ecg_time": pd.to_datetime(["2020-01-11"]),
            "dischtime": pd.to_datetime(["2020-01-12"]),
            "split": ["train"],
            "age": [60.0],
            "sex": ["F"],
            "race": ["X"],
            "waveform_path": ["x"],
        }
    ).to_parquet(master_path, index=False)
    pd.DataFrame(
        {
            "icd_version": [10, 10, 10],
            "icd_code": ["A", "B", "C"],
            "phenotype_id": ["prior", "index", "future"],
        }
    ).to_parquet(mapping_path, index=False)
    pd.DataFrame(
        {
            "phenotype_id": ["prior", "index", "future"],
            "body_system": ["x", "x", "x"],
        }
    ).to_parquet(definitions_path, index=False)
    pd.DataFrame(
        {
            "icd_code": ["A", "B", "C"],
            "ccsr_category": ["a", "b", "c"],
            "ccsr_description": ["a", "b", "c"],
            "body_system_code": ["x", "x", "x"],
        }
    ).to_parquet(ccsr_path, index=False)
    config = SimpleNamespace(
        catalog_path=tmp_path / "catalog.duckdb",
        duckdb_memory_limit_gb=1,
        workers=1,
        standardized_root=standardized,
    )
    resources = SimpleNamespace(
        phecodex_mapping_path=mapping_path,
        phecodex_definitions_path=definitions_path,
        ccsr_mapping_path=ccsr_path,
    )

    paths = _build_phenome_cohorts(
        config,
        resources=resources,
        diagnosis_path=diagnosis_path,
        master_path=master_path,
        output_root=output,
    )
    current = pd.read_parquet(paths["current_phenotypes"])

    assert set(current["phenotype_id"]) == {"prior", "index"}
    assert current["phenotype_time_basis"].eq(
        "pre_or_index_discharge_history"
    ).all()


def test_future_phenome_cox_uses_censored_person_time() -> None:
    rng = np.random.default_rng(20260812)
    marker = rng.normal(size=400)
    event_time = rng.exponential(np.exp(-0.7 * marker))
    censor_time = rng.exponential(2.0, size=len(marker))
    duration = np.minimum(event_time, censor_time)
    event = event_time <= censor_time

    result = _fit_cause_specific_cox(marker, duration, event)

    assert result["status"] == "evaluable"
    assert result["direction"] == 1
    assert result["hazard_ratio_per_sd"] > 1.0
    assert result["n_at_risk"] == len(marker)
    assert result["n_event"] == int(event.sum())


def test_future_phenome_parallel_replication_matches_sequential_results() -> None:
    rng = np.random.default_rng(20260726)
    rows = []
    event_times: dict[int, pd.Timestamp] = {}
    index_time = pd.Timestamp("2020-01-01")
    for split, subject_start in (("train", 1), ("validation", 101)):
        for offset in range(40):
            subject_id = subject_start + offset
            positive = offset < 20
            rows.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "index_end_time": index_time,
                    "last_observed_time": pd.Timestamp("2021-12-31"),
                    "date_of_death": pd.NaT,
                    "sex": "F",
                    "marker_up": float(positive) + rng.normal(0.0, 0.2),
                    "marker_down": -float(positive) + rng.normal(0.0, 0.2),
                }
            )
            if positive:
                event_times[subject_id] = index_time + pd.Timedelta(days=10)
    train_hits = pd.DataFrame(
        [
            {
                "phenotype_id": "P1",
                "horizon_days": 30,
                "feature_id": "marker_up",
                "panel_id": "primary_core",
                "q_value": 0.001,
                "aligned_auc": 0.95,
                "direction": 1,
            },
            {
                "phenotype_id": "P1",
                "horizon_days": 30,
                "feature_id": "marker_down",
                "panel_id": "sentinel",
                "q_value": 0.002,
                "aligned_auc": 0.94,
                "direction": -1,
            },
        ]
    )
    arguments = {
        "train_hits": train_hits,
        "features": pd.DataFrame(rows),
        "event_lookup": {"P1": pd.Series(event_times)},
        "definition_lookup": pd.DataFrame(
            {
                "phenotype_id": ["P1"],
                "display_name": ["Future phenotype"],
                "body_system": ["circulatory"],
                "male_only": [False],
                "female_only": [False],
            }
        ).set_index("phenotype_id"),
        "min_inference": 10,
        "min_formal": 20,
        "bootstrap_iterations": 20,
        "random_seed": 20260726,
        "progress": None,
    }

    sequential = _replicate_future_hits(**arguments, worker_count=1)
    parallel = _replicate_future_hits(**arguments, worker_count=2)

    pd.testing.assert_frame_equal(parallel, sequential, check_exact=True)


def test_future_phenome_replication_cache_is_shared_across_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import future_phenome as future_module

    rows = []
    event_times: dict[int, pd.Timestamp] = {}
    index_time = pd.Timestamp("2020-01-01")
    for split, subject_start in (("train", 1), ("validation", 101)):
        for offset in range(20):
            subject_id = subject_start + offset
            positive = offset < 10
            rows.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "index_end_time": index_time,
                    "last_observed_time": pd.Timestamp("2021-12-31"),
                    "date_of_death": pd.NaT,
                    "sex": "F",
                    "marker_up": float(positive) + 0.01 * offset,
                    "marker_down": -float(positive) - 0.01 * offset,
                }
            )
            if positive:
                event_times[subject_id] = (
                    index_time + pd.Timedelta(days=10)
                )
    train_hits = pd.DataFrame(
        [
            {
                "phenotype_id": "P1",
                "horizon_days": 30,
                "feature_id": feature_id,
                "panel_id": panel_id,
                "q_value": 0.001,
                "aligned_auc": 1.0,
                "direction": direction,
            }
            for feature_id, panel_id, direction in (
                ("marker_up", "primary_core", 1),
                ("marker_down", "sentinel", -1),
            )
        ]
    )
    original_prepare = future_module._prepare_future_replication_cache
    prepare_calls = 0

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        future_module,
        "_prepare_future_replication_cache",
        counted_prepare,
    )
    monkeypatch.setattr(
        future_module,
        "FUTURE_REPLICATION_CHUNK_SIZE",
        1,
    )
    result = _checkpointed_future_replication(
        tmp_path,
        checkpoint_input_hash="future-cache",
        train_hits=train_hits,
        features=pd.DataFrame(rows),
        event_lookup={"P1": pd.Series(event_times)},
        definition_lookup=pd.DataFrame(
            {
                "phenotype_id": ["P1"],
                "display_name": ["Future phenotype"],
                "body_system": ["circulatory"],
                "male_only": [False],
                "female_only": [False],
            }
        ).set_index("phenotype_id"),
        min_inference=1,
        min_formal=1,
        bootstrap_iterations=0,
        random_seed=20260726,
        worker_count=1,
        progress=None,
    )

    assert len(result) == 2
    assert prepare_calls == 1


def test_future_and_context_bootstraps_are_validation_gated() -> None:
    train_hits = pd.DataFrame(
        {
            "phenotype_id": ["P1", "P2"],
            "horizon_days": [30, 90],
            "feature_id": ["marker_a", "marker_b"],
            "panel_id": ["primary_core", "sentinel"],
        }
    )
    validation = pd.DataFrame(
        {
            "phenotype_id": ["P1", "P2"],
            "horizon_days": [30, 90],
            "feature_id": ["marker_a", "marker_b"],
            "panel_id": ["primary_core", "sentinel"],
            "replicated": [False, True],
            "roc_auc_ci_low": [np.nan, np.nan],
            "roc_auc_ci_high": [np.nan, np.nan],
            "bootstrap_iterations": [0, 0],
            "bootstrap_successful_iterations": [0, 0],
            "bootstrap_unit": ["subject_id", "subject_id"],
        }
    )
    selected = _future_replicated_bootstrap_hits(
        train_hits,
        validation,
    )
    assert selected["phenotype_id"].tolist() == ["P2"]
    future_output = _overlay_future_bootstraps(
        validation,
        pd.DataFrame(
            {
                "phenotype_id": ["P2"],
                "horizon_days": [90],
                "feature_id": ["marker_b"],
                "panel_id": ["sentinel"],
                "roc_auc_ci_low": [0.6],
                "roc_auc_ci_high": [0.8],
                "bootstrap_iterations": [1000],
                "bootstrap_successful_iterations": [997],
                "bootstrap_unit": ["subject_id"],
            }
        ),
    )
    assert future_output.loc[0, "bootstrap_iterations"] == 0
    assert future_output.loc[1, "bootstrap_iterations"] == 1000
    assert future_output.loc[1, "roc_auc_ci_low"] == pytest.approx(0.6)

    context = pd.DataFrame(
        {
            "association_id": ["lab_a", "lab_b"],
            "feature_id": ["marker_a", "marker_b"],
            "replicated": [False, True],
        }
    )
    context_output = _overlay_context_intervals(
        context,
        pd.DataFrame(
            {
                "association_id": ["lab_b"],
                "feature_id": ["marker_b"],
                "effect_ci_low": [0.2],
                "effect_ci_high": [0.4],
                "bootstrap_successful_iterations": [999],
                "bootstrap_unit": ["subject_id"],
            }
        ),
    )
    assert np.isnan(context_output.loc[0, "effect_ci_low"])
    assert context_output.loc[0, "bootstrap_successful_iterations"] == 0
    assert context_output.loc[1, "effect_ci_low"] == pytest.approx(0.2)
    assert (
        context_output.loc[1, "bootstrap_successful_iterations"]
        == 999
    )


def test_future_phenome_options_and_broad_results_remain_separate() -> None:
    options = PhenomeDiscoveryOptions.from_mapping(
        {"future_horizons_days": "365,30,90"}
    )
    assert options.future_horizons_days == (30, 90, 365)
    assert options.max_future_phenotypes == 100
    assert PhenomeDiscoveryOptions.from_mapping(
        {"max_future_phenotypes": 0}
    ).max_future_phenotypes is None
    future = pd.DataFrame(
        {
            "phenotype_id": ["CV_001"],
            "phenotype_name": ["Future condition"],
            "body_system": ["Circulatory"],
            "horizon_days": [90],
            "feature_id": ["marker"],
            "panel_id": ["primary_core"],
            "train_aligned_auc": [0.8],
            "train_direction": [-1],
            "train_q_value": [0.001],
            "validation_raw_auc": [0.2],
            "validation_q_value": [0.01],
            "direction_consistent": [True],
            "replicated": [True],
            "inference_status": ["formal_inference"],
        }
    )

    broad = _broad_discovery_table(
        pd.DataFrame(),
        pd.DataFrame(),
        future,
    )

    assert broad.loc[0, "discovery_class"] == "future_outcome_hypothesis"
    assert broad.loc[0, "target_kind"] == (
        "post_discharge_prognostic_association"
    )
    assert broad.loc[0, "target_id"] == "CV_001@90d"
    assert not bool(broad.loc[0, "locked_test_scanned"])
    assert broad.loc[0, "confounding_control"] == "none_broad_screen"
    assert bool(
        broad.loc[0, "requires_endpoint_specific_adjustment"]
    )
    assert broad.loc[0, "feature_role_policy_version"]


def test_broad_discovery_table_surfaces_context_association_domains() -> None:
    """Regression guard: association_domains.py hits (labs, meds,
    procedures, outcomes, and the newer demographic/socioeconomic/
    vitals_ed domains) must flow into the same unified discovery table as
    pathology and future-outcome hits, tagged
    discovery_class="non_pathology_association" — this is what the GUI's
    "Neue Zusammenhänge" scope filter "Labor, Medikamente, Prozeduren und
    Outcomes" (value="non_pathology_association") reads."""
    context_hits = pd.DataFrame(
        {
            "domain": ["demographic"],
            "association_id": ["demographic_age"],
            "association_name": ["Alter bei ECG"],
            "target_kind": ["continuous"],
            "feature_id": ["marker"],
            "panel_id": ["primary_core"],
            "train_effect": [0.3],
            "train_q_value": [0.001],
            "validation_effect": [0.25],
            "validation_q_value": [0.02],
            "direction_consistent": [True],
            "replicated": [True],
            "inference_status": ["formal_inference"],
        }
    )
    broad = _broad_discovery_table(pd.DataFrame(), context_hits, None)
    assert len(broad) == 1
    assert broad.loc[0, "discovery_class"] == "non_pathology_association"
    assert broad.loc[0, "discovery_domain"] == "demographic"
    assert broad.loc[0, "target_id"] == "demographic_age"
    assert broad.loc[0, "validation_effect"] == pytest.approx(0.25)
    assert bool(broad.loc[0, "replicated"])
    assert not bool(broad.loc[0, "locked_test_scanned"])


def test_derived_phenotype_outputs_require_matching_checksums(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(first, index=False)
    pd.DataFrame({"value": [2]}).to_parquet(second, index=False)
    payload = {
        "output_fingerprints": {
            path.name: {
                "size": fingerprint["size"],
                "sha256": fingerprint["sha256"],
            }
            for path in (first, second)
            for fingerprint in (
                file_fingerprint(path, include_sha256=True),
            )
        }
    }
    assert _derived_outputs_match((first, second), payload)
    pd.DataFrame({"value": [3, 4]}).to_parquet(second, index=False)
    assert not _derived_outputs_match((first, second), payload)


def test_fixed_rule_comparison_uses_a_paired_operating_point() -> None:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 50)
    result = paired_operating_point_bootstrap(
        np.arange(len(labels)),
        labels,
        candidate_predictions=labels.astype(bool),
        baseline_predictions=np.zeros(len(labels), dtype=bool),
        metric="sensitivity",
        iterations=100,
        seed=17,
    )
    assert result["candidate_sensitivity"] == 1.0
    assert result["baseline_sensitivity"] == 0.0
    assert result["delta_sensitivity"] == 1.0
    assert result["bootstrap_successful_iterations"] > 0


def test_weighted_cluster_bootstrap_metrics_match_explicit_resampling() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    scores = np.asarray([0.1, 0.9, 0.4, 0.4, 0.8, 0.2])
    weights = np.asarray(
        [
            [1, 2, 0, 1, 3, 2],
            [2, 1, 3, 2, 1, 1],
        ],
        dtype=np.int64,
    )
    prevalence = float(np.mean(labels))
    result = _weighted_bootstrap_metric_batch(
        labels,
        scores,
        weights,
        thresholds={"fixed": 0.4},
        prevalence=prevalence,
    )

    for bootstrap_index, row_weights in enumerate(weights):
        indices = np.repeat(np.arange(len(labels)), row_weights)
        sampled_labels = labels[indices]
        sampled_scores = scores[indices]
        expected_classification = classification_metrics(
            sampled_labels,
            sampled_scores >= 0.4,
            prevalence=prevalence,
        )
        assert result["roc_auc"][bootstrap_index] == pytest.approx(
            roc_auc_score(sampled_labels, sampled_scores)
        )
        assert result["average_precision"][bootstrap_index] == pytest.approx(
            average_precision_score(sampled_labels, sampled_scores)
        )
        assert result["pr_auc"][bootstrap_index] == pytest.approx(
            exact_pr_auc(sampled_labels, sampled_scores)
        )
        for metric in (
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "balanced_accuracy",
        ):
            assert result[f"fixed_{metric}"][bootstrap_index] == pytest.approx(
                expected_classification[metric]
            )


def test_weighted_bootstrap_prepared_order_is_exact_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import statistics as statistics_module

    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
    scores = np.linspace(-2.0, 2.0, len(labels))
    subjects = np.arange(len(labels), dtype=np.int64)

    original_argsort = statistics_module.np.argsort
    calls = 0

    def tracked_argsort(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_argsort(*args, **kwargs)

    monkeypatch.setattr(statistics_module.np, "argsort", tracked_argsort)
    result = statistics_module.marker_bootstrap_intervals(
        subjects,
        scores,
        labels,
        direction=1,
        thresholds={"fixed": 0.0},
        # The production batching cap is 64, so this requires three batches.
        iterations=130,
        seed=20260902,
    )

    assert result["bootstrap_successful_iterations"] == 130
    assert calls == 1


def test_comparison_presentation_keeps_fixed_and_continuous_metrics() -> None:
    frame = pd.DataFrame(
        {
            "endpoint_id": ["acs", "qtc"],
            "vcg_feature_id": ["marker_st", "marker_qt"],
            "sensitivity_delta_vs_fixed_rule": [0.12, np.nan],
            "sensitivity_delta_ci_low": [0.04, np.nan],
            "sensitivity_delta_ci_high": [0.20, np.nan],
            "specificity_delta_vs_fixed_rule": [0.08, np.nan],
            "specificity_delta_ci_low": [0.01, np.nan],
            "specificity_delta_ci_high": [0.15, np.nan],
            "delta_auc": [np.nan, 0.09],
            "delta_auc_ci_low": [np.nan, 0.03],
            "delta_auc_ci_high": [np.nan, 0.15],
            "delta_average_precision": [np.nan, 0.07],
            "delta_average_precision_ci_low": [np.nan, 0.01],
            "delta_average_precision_ci_high": [np.nan, 0.13],
        }
    )
    effects = comparison_effect_frame(frame)
    assert set(effects["effect_metric"]) == {
        "sensitivity_at_baseline_specificity",
        "specificity_at_baseline_sensitivity",
        "roc_auc",
        "average_precision",
    }
    assert len(effects) == 4


def test_continuous_baseline_comparison_reports_paired_auc_and_ap() -> None:
    rng = np.random.default_rng(441)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 120)
    train = pd.DataFrame(
        {
            "subject_id": np.arange(len(labels)),
            "label": labels,
            "vcg_marker": labels + rng.normal(0, 0.08, len(labels)),
            "classical_qtc_fridericia": (
                labels + rng.normal(0, 1.2, len(labels))
            ),
        }
    )
    validation = pd.DataFrame(
        {
            "subject_id": np.arange(10_000, 10_000 + len(labels)),
            "label": labels,
            "vcg_marker": labels + rng.normal(0, 0.08, len(labels)),
            "classical_qtc_fridericia": (
                labels + rng.normal(0, 1.2, len(labels))
            ),
        }
    )
    vcg_model = fit_marker_model(
        train["vcg_marker"],
        train["label"],
        "vcg_marker",
    )
    comparison = _paired_primary_baseline_comparison(
        "qtc_gt_500",
        "vcg_marker",
        "classical_qtc_fridericia",
        vcg_model,
        train,
        validation,
        ValidationSpec(
            run_id="test",
            split_hash="split",
            endpoint_ids=("qtc_gt_500",),
            panel_versions={"primary_core": "test"},
            development_bootstrap_iterations=100,
            min_class_inference=30,
            min_class_primary=100,
        ),
        100,
    )
    assert comparison["comparison_type"] == (
        "paired_continuous_marker_discrimination"
    )
    assert comparison["delta_auc"] > 0.1
    assert comparison["delta_average_precision"] > 0.1
    assert comparison["bootstrap_successful_iterations"] > 0
    assert comparison["reference_independence"] == "ecg_derived_non_independent"
    # A clearly, robustly separating new marker against a noisy classical
    # baseline must produce a clear "new marker wins" verdict, not just
    # raw numbers the reader has to interpret themselves.
    verdict, verdict_de = _comparison_verdict(comparison)
    assert verdict == "new_marker_significantly_better"
    assert verdict_de == "neuer Marker signifikant überlegen"


def test_comparison_verdict_requires_formal_inference_power() -> None:
    verdict, verdict_de = _comparison_verdict(
        {
            "inference_status": "underpowered",
            "comparison_type": "paired_continuous_marker_discrimination",
            "delta_auc_ci_low": 0.05,
            "delta_auc_ci_high": 0.20,
        }
    )
    assert verdict == "insufficient_evidence"
    assert verdict_de == "nicht genügend Evidenz"


def test_comparison_verdict_missing_bootstrap_ci_is_insufficient_evidence() -> None:
    verdict, _ = _comparison_verdict(
        {
            "inference_status": "formal_inference",
            "comparison_type": "paired_continuous_marker_discrimination",
            "delta_auc_ci_low": float("nan"),
            "delta_auc_ci_high": float("nan"),
        }
    )
    assert verdict == "insufficient_evidence"


def test_comparison_verdict_continuous_ci_excludes_zero_below() -> None:
    verdict, _ = _comparison_verdict(
        {
            "inference_status": "formal_inference",
            "comparison_type": "paired_continuous_marker_discrimination",
            "delta_auc_ci_low": -0.20,
            "delta_auc_ci_high": -0.02,
        }
    )
    assert verdict == "classical_rule_significantly_better"


def test_comparison_verdict_continuous_ci_straddles_zero_is_equivalent() -> None:
    verdict, _ = _comparison_verdict(
        {
            "inference_status": "formal_inference",
            "comparison_type": "paired_continuous_marker_discrimination",
            "delta_auc_ci_low": -0.05,
            "delta_auc_ci_high": 0.05,
        }
    )
    assert verdict == "equivalent"


def test_comparison_verdict_fixed_rule_uses_sensitivity_delta_ci() -> None:
    verdict, _ = _comparison_verdict(
        {
            "inference_status": "formal_inference",
            "comparison_type": "paired_fixed_rule_operating_point",
            "sensitivity_delta_ci_low": 0.02,
            "sensitivity_delta_ci_high": 0.18,
            # A stray continuous-style key must not be picked up for the
            # fixed-rule comparison type.
            "delta_auc_ci_low": -0.9,
            "delta_auc_ci_high": -0.8,
        }
    )
    assert verdict == "new_marker_significantly_better"


def test_subgroup_stability_includes_a_race_performance_audit_axis() -> None:
    """Work Package A5: race/ethnicity must already be one of the
    subgroup axes checked for stability, alongside age/sex/BMI/renal
    function/signal quality/care setting — labeled as an
    administrative-audit-only fairness check, never as a clinical
    subgroup with its own threshold."""
    rng = np.random.default_rng(2026)
    n = 260
    # Three race groups: two with >=100 subjects (must be included), one
    # with <100 (must be excluded to avoid an unstable small-N estimate).
    race = np.array(["White"] * 120 + ["Black"] * 110 + ["Asian"] * 30)
    label = rng.integers(0, 2, n)
    marker = label + rng.normal(0.0, 0.3, n)
    validation = pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "label": label,
            "marker": marker,
            "age": rng.uniform(40, 85, n),
            "machine_qrs_duration_ms": rng.uniform(80, 140, n),
            "bmi_before_ecg": rng.uniform(18, 40, n),
            "creatinine_nearest": rng.uniform(0.5, 2.5, n),
            "sex": rng.choice(["M", "F"], n),
            "signal_quality_tier": "high",
            "ed_stay_id": np.nan,
            "hadm_id": np.arange(n),
            "race": race,
        }
    )
    model = fit_marker_model(validation["marker"], validation["label"], "marker")
    specification = ValidationSpec(
        run_id="test",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={"primary_core": "test"},
        min_class_inference=10,
        min_class_primary=20,
    )
    rows = _subgroup_stability_records(
        "endpoint",
        "marker",
        model,
        validation,
        specification,
    )
    race_rows = {
        row["subgroup_level"]: row
        for row in rows
        if row["subgroup_variable"] == "race_performance_audit"
    }
    assert set(race_rows) == {"White", "Black"}
    assert "Asian" not in race_rows
    for row in race_rows.values():
        assert row["subgroup_time_scope"] == "administrative_audit_only"
        assert row["analysis_type"] == "subgroup_stability"

    # The established clinical/technical subgroup axes must still be
    # present alongside the fairness audit, not replaced by it.
    other_variables = {row["subgroup_variable"] for row in rows}
    assert {"age", "sex", "bmi", "creatinine", "signal_quality"} <= (
        other_variables
    )


def test_visualization_artifacts_align_direction_and_keep_rules_as_points() -> None:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
    train = pd.DataFrame(
        {
            "subject_id": np.arange(len(labels)),
            "label": labels,
            "marker": -labels + np.linspace(-0.02, 0.02, len(labels)),
        }
    )
    validation = train.copy()
    validation["subject_id"] += 10_000
    model = fit_marker_model(train["marker"], labels, "marker")
    metric_record = {
        "roc_auc": 1.0,
        "roc_auc_ci_low": 0.98,
        "roc_auc_ci_high": 1.0,
        "average_precision": 1.0,
        "average_precision_ci_low": 0.98,
        "average_precision_ci_high": 1.0,
        "inference_status": "formal_inference",
    }
    curves, distributions = _visualization_records_for_marker(
        "endpoint",
        "marker",
        "primary_core",
        "prespecified_primary_marker",
        model,
        train,
        validation,
        metric_record,
        metric_record,
    )
    curve_frame = pd.DataFrame(curves)
    assert model.direction == -1
    assert set(curve_frame["curve_type"]) == {"roc", "precision_recall"}
    roc = curve_frame.loc[
        (curve_frame["evaluation_split"] == "validation")
        & (curve_frame["curve_type"] == "roc")
    ]
    assert roc["x"].min() == 0.0
    assert roc["y"].max() == 1.0
    distribution = pd.DataFrame(distributions)
    fractions = distribution.groupby(
        ["evaluation_split", "label"]
    )["fraction_within_class"].sum()
    assert np.allclose(fractions.to_numpy(), 1.0)

    fixed = pd.DataFrame(
        {
            "endpoint_id": ["endpoint"],
            "baseline_id": ["st_rule"],
            "evaluation_split": ["validation"],
            "threshold_source": ["predefined_clinical_rule"],
            "sensitivity": [0.7],
            "specificity": [0.9],
            "ppv": [0.6],
            "n_positive": [100],
            "n_negative": [100],
            "inference_status": ["formal_inference"],
        }
    )
    reference_points = _fixed_reference_curve_points(
        fixed,
        pd.DataFrame(),
    )
    assert len(reference_points) == 2
    roc_point = reference_points.loc[
        reference_points["curve_type"] == "roc"
    ].iloc[0]
    assert roc_point["x"] == pytest.approx(0.1)
    assert roc_point["y"] == pytest.approx(0.7)
    assert roc_point["point_kind"] == "fixed_reference_point"

    candidate = pd.DataFrame(
        {
            "endpoint_id": ["endpoint"],
            "feature_id": ["marker"],
            "panel_id": ["primary_core"],
            "analysis_type": ["single_marker"],
            "evaluation_split": ["validation"],
            "threshold_source": ["train"],
            "specificity_90_sensitivity": [0.8],
            "specificity_90_specificity": [0.9],
            "specificity_90_observed_ppv": [0.5],
            "specificity_90_observed_npv": [0.95],
            "specificity_90_balanced_accuracy": [0.85],
            "specificity_90_threshold_original": [1.2],
            "inference_status": ["formal_inference"],
            "exploratory_score": [False],
        }
    )
    operating = _build_operating_point_results(
        candidate,
        fixed,
        pd.DataFrame(),
    )
    assert set(operating["operating_point"]) == {
        "sensitivity_at_90_specificity",
        "fixed_reference_point",
    }


def test_csv_import_is_strict_and_converts_microvolts(tmp_path: Path) -> None:
    source = tmp_path / "case.csv"
    reversed_leads = tuple(reversed(LEAD_NAMES_12))
    pd.DataFrame(
        {
            lead: np.full(20, 1000.0 + index)
            for index, lead in enumerate(reversed_leads)
        }
    ).to_csv(source, index=False)
    result = load_external_ecg_result(
        source,
        ExternalECGMeta(sampling_rate=500.0, default_unit="\u00b5V"),
    )
    assert result.lead_names == tuple(LEAD_NAMES_12)
    assert result.signal_millivolts.shape == (20, 12)
    assert result.signal_millivolts[0, 0] == pytest.approx(
        (1000.0 + reversed_leads.index("I")) * 0.001
    )

    missing = tmp_path / "missing.csv"
    pd.DataFrame({"I": [0.0], "II": [0.0]}).to_csv(missing, index=False)
    with pytest.raises(ECGImportMappingRequired) as error:
        load_external_ecg_result(
            missing,
            ExternalECGMeta(sampling_rate=500.0, default_unit="mV"),
        )
    assert "V6" in error.value.missing_leads


def test_npz_import_uses_embedded_leads_units_and_sampling_rate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "case.npz"
    reversed_leads = tuple(reversed(LEAD_NAMES_12))
    signal_microvolts = np.column_stack(
        [
            np.full(20, 1000.0 + index)
            for index, _lead in enumerate(reversed_leads)
        ]
    )
    np.savez(
        source,
        signal=signal_microvolts,
        sampling_rate_hz=np.asarray(250.0),
        lead_names=np.asarray(reversed_leads),
        units=np.asarray(["uV"] * 12),
    )

    result = load_external_ecg_result(source)

    assert result.source_format == "numpy_research"
    assert result.sampling_rate_hz == 250.0
    assert result.lead_names == tuple(LEAD_NAMES_12)
    assert result.signal_millivolts[0, 0] == pytest.approx(
        (1000.0 + reversed_leads.index("I")) * 0.001
    )


def test_wfdb_import_roundtrip_preserves_named_lead_order(
    tmp_path: Path,
) -> None:
    wfdb = pytest.importorskip("wfdb")
    reversed_leads = tuple(reversed(LEAD_NAMES_12))
    time_axis = np.arange(250, dtype=float) / 250.0
    physical = np.column_stack(
        [
            0.25 * np.sin(2.0 * np.pi * (index + 1) * time_axis)
            for index, _lead in enumerate(reversed_leads)
        ]
    )
    wfdb.wrsamp(
        record_name="case",
        fs=250,
        units=["mV"] * 12,
        sig_name=list(reversed_leads),
        p_signal=physical,
        write_dir=str(tmp_path),
    )

    result = load_external_ecg_result(tmp_path / "case.hea")

    assert result.source_format == "wfdb"
    assert result.sampling_rate_hz == 250.0
    assert result.lead_names == tuple(LEAD_NAMES_12)
    expected_i = physical[:, reversed_leads.index("I")]
    assert result.signal_millivolts[:, 0] == pytest.approx(
        expected_i,
        abs=2e-4,
    )


def test_edf_import_roundtrip_converts_units_and_orders_leads(
    tmp_path: Path,
) -> None:
    pyedflib = pytest.importorskip("pyedflib")
    highlevel = pytest.importorskip("pyedflib.highlevel")
    source = tmp_path / "case.edf"
    reversed_leads = tuple(reversed(LEAD_NAMES_12))
    time_axis = np.arange(250, dtype=float) / 250.0
    signals_microvolts = np.vstack(
        [
            (index + 1) * 100.0 * np.sin(2.0 * np.pi * time_axis)
            for index, _lead in enumerate(reversed_leads)
        ]
    )
    signal_headers = [
        highlevel.make_signal_header(
            lead,
            dimension="uV",
            sample_frequency=250,
            physical_min=-1500.0,
            physical_max=1500.0,
        )
        for lead in reversed_leads
    ]
    assert highlevel.write_edf(
        str(source),
        signals_microvolts,
        signal_headers,
        file_type=pyedflib.FILETYPE_EDFPLUS,
    )

    result = load_external_ecg_result(source)

    assert result.source_format == "edf"
    assert result.sampling_rate_hz == 250.0
    assert result.lead_names == tuple(LEAD_NAMES_12)
    expected_i = signals_microvolts[reversed_leads.index("I")] * 0.001
    assert result.signal_millivolts[:, 0] == pytest.approx(
        expected_i,
        abs=5e-5,
    )


def test_dicom_waveform_import_roundtrip_applies_channel_sensitivity(
    tmp_path: Path,
) -> None:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    source = tmp_path / "case.dcm"
    reversed_leads = tuple(reversed(LEAD_NAMES_12))
    raw = np.column_stack(
        [
            np.full(20, 1000 + index, dtype=np.int16)
            for index, _lead in enumerate(reversed_leads)
        ]
    )
    storage_uid = generate_uid()
    instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = storage_uid
    file_meta.MediaStorageSOPInstanceUID = instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(
        str(source),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    dataset.SOPClassUID = storage_uid
    dataset.SOPInstanceUID = instance_uid
    dataset.PatientName = "LOCAL^RESEARCH"
    dataset.PatientID = "LOCAL"
    dataset.Modality = "ECG"

    multiplex = Dataset()
    multiplex.NumberOfWaveformChannels = 12
    multiplex.NumberOfWaveformSamples = raw.shape[0]
    multiplex.SamplingFrequency = 500.0
    multiplex.WaveformBitsAllocated = 16
    multiplex.WaveformSampleInterpretation = "SS"
    multiplex.WaveformData = raw.astype("<i2").tobytes()
    channel_definitions = []
    for lead in reversed_leads:
        channel = Dataset()
        channel.ChannelLabel = lead
        channel.ChannelSensitivity = 0.001
        channel.ChannelSensitivityCorrectionFactor = 1.0
        unit = Dataset()
        unit.CodeValue = "mV"
        unit.CodingSchemeDesignator = "UCUM"
        unit.CodeMeaning = "millivolt"
        channel.ChannelSensitivityUnitsSequence = Sequence([unit])
        channel_definitions.append(channel)
    multiplex.ChannelDefinitionSequence = Sequence(channel_definitions)
    dataset.WaveformSequence = Sequence([multiplex])
    pydicom.dcmwrite(source, dataset, enforce_file_format=True)

    result = load_external_ecg_result(source)

    assert result.source_format == "dicom"
    assert result.sampling_rate_hz == 500.0
    assert result.lead_names == tuple(LEAD_NAMES_12)
    assert result.signal_millivolts[0, 0] == pytest.approx(
        (1000 + reversed_leads.index("I")) * 0.001
    )


def test_case_library_is_pseudonymized_and_revisioned(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / "case.csv"
    pd.DataFrame(
        {
            lead: np.sin(np.linspace(0, 4 * np.pi, 100))
            for lead in LEAD_NAMES_12
        }
    ).to_csv(source, index=False)
    library = ECGCaseLibrary(config)
    case = library.import_case(
        source,
        metadata=ExternalECGMeta(
            sampling_rate=500.0,
            default_unit="mV",
            age=55,
            sex="female",
        ),
    )
    assert case.source_path == "<local-redacted>"
    assert case.metadata["age"] == 55
    assert len(library.list_cases()) == 1
    revision = library.add_annotation(
        case.case_id,
        fiducials={"qrs_onset": 10, "qrs_offset": 40},
        notes="local review",
    )
    assert library.annotations(case.case_id) == (revision,)


def test_ecg_lab_blocks_marker_calculation_for_unusable_signal(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "flatline.csv"
    pd.DataFrame(
        {
            lead: np.zeros(1_000, dtype=float)
            for lead in LEAD_NAMES_12
        }
    ).to_csv(source, index=False)
    library = ECGCaseLibrary(config)
    case = library.import_case(
        source,
        metadata=ExternalECGMeta(
            sampling_rate=500.0,
            default_unit="mV",
        ),
    )

    payload = library.analyze_case(case.case_id)

    assert payload["analysis_status"] == "blocked_signal_quality"
    assert payload["markers_calculated"] is False
    assert payload["feature_values"] == {}
    assert payload["classical_criteria"] == {}
    assert payload["score_releases"] == []


def test_manual_fiducial_revision_is_validated_and_applied() -> None:
    beat = BeatDelineation(
        r_peak=MarkerPoint(100, 0.8, "auto"),
        r_onset=MarkerPoint(80, 0.8, "auto"),
        r_offset=MarkerPoint(120, 0.8, "auto"),
        t_onset=MarkerPoint(150, 0.8, "auto"),
        t_peak=MarkerPoint(180, 0.8, "auto"),
        t_offset=MarkerPoint(220, 0.8, "auto"),
    )
    delineation = RecordDelineation(
        beats=[beat],
        fs=500,
        n_samples=500,
        ecg_id=1,
    )
    revision = AnnotationRevision(
        revision_id="manual-1",
        case_id="case-1",
        parent_revision_id="",
        fiducials={
            "qrs_onset": 75,
            "r_peak": 100,
            "qrs_offset": 125,
            "t_onset": 150,
            "t_peak": 180,
            "t_offset": 225,
        },
    )

    _apply_annotation_revision(delineation, revision)

    assert delineation.beats[0].r_onset.sample == 75
    assert delineation.beats[0].r_offset.sample == 125
    assert delineation.beats[0].r_onset.source == "manual:manual-1"
    assert delineation.source_method == "fusion+manual_revision"

    invalid = AnnotationRevision(
        revision_id="manual-invalid",
        case_id="case-1",
        parent_revision_id="manual-1",
        fiducials={"qrs_onset": 130, "qrs_offset": 110},
    )
    with pytest.raises(ValueError, match="physiological ordering"):
        _apply_annotation_revision(delineation, invalid)


def test_ecg_lab_exposes_colored_segments_for_every_accepted_beat() -> None:
    beats = []
    def point(sample: int) -> MarkerPoint:
        return MarkerPoint(sample, 0.9, "test")

    for r_peak in (300, 700, 1100):
        beats.append(
            BeatDelineation(
                r_peak=point(r_peak),
                p_onset=point(r_peak - 110),
                p_peak=point(r_peak - 85),
                p_offset=point(r_peak - 60),
                r_onset=point(r_peak - 25),
                r_offset=point(r_peak + 30),
                t_onset=point(r_peak + 80),
                t_peak=point(r_peak + 130),
                t_offset=point(r_peak + 200),
            )
        )
    delineation = RecordDelineation(
        beats=beats,
        fs=500,
        n_samples=1_500,
        ecg_id=2,
    )

    segments = _accepted_beat_wave_segments(delineation)
    measurable = [item for item in segments if item["status"] == "measurable"]

    assert {item["beat_index"] for item in measurable} == {0, 1, 2}
    assert len(measurable) == 15
    assert {
        item["segment"] for item in measurable
    } == {
        "p_wave",
        "pr_segment",
        "qrs_complex",
        "st_segment",
        "t_wave",
    }


def test_ecg_lab_loads_all_release_components_and_classical_values() -> None:
    releases = [
        {
            "score_spec": {
                "components": [
                    "qrs_t_hausdorff_distance",
                    "classical_sokolow_lyon",
                    "unknown_marker",
                ]
            }
        }
    ]
    assert _release_component_names(releases) == (
        "qrs_t_hausdorff_distance",
    )
    score = _apply_single_case_score(
        {
            "qrs_t_hausdorff_distance": 3.0,
            "classical_sokolow_lyon": 4.0,
        },
        {
            "score_id": "score",
            "endpoint_id": "echo_lvh_strict",
            "components": [
                "qrs_t_hausdorff_distance",
                "classical_sokolow_lyon",
            ],
            "directions": [1, -1],
            "means": [1.0, 2.0],
            "standard_deviations": [2.0, 1.0],
            "training_split_hash": "train",
            "origin": "development",
            "version": "1",
        },
    )
    assert score == pytest.approx(-0.5)


def test_wfdb_case_fingerprints_and_copies_the_complete_record(
    tmp_path: Path,
) -> None:
    wfdb = pytest.importorskip("wfdb")
    samples = np.arange(200, dtype=float) / 25.0
    signal = np.column_stack(
        [
            np.sin(samples + channel / 7.0) * (1.0 + channel / 20.0)
            for channel in range(12)
        ]
    )
    wfdb.wrsamp(
        "bundle",
        fs=500,
        units=["mV"] * 12,
        sig_name=list(LEAD_NAMES_12),
        p_signal=signal,
        write_dir=str(tmp_path),
    )
    header = tmp_path / "bundle.hea"
    bundle_paths = _source_bundle_paths(header, "wfdb")
    assert {path.suffix for path in bundle_paths} == {".hea", ".dat"}
    before = _source_bundle_fingerprint(bundle_paths)

    config = _config(tmp_path / "project")
    case = ECGCaseLibrary(config).import_case(header, copy_original=True)
    assert case.source_sha256 == before
    assert case.metadata["source_file_count"] == 2
    copied = config.cases_root / case.case_id / "source"
    assert (copied / "bundle.hea").is_file()
    assert (copied / "bundle.dat").is_file()

    data_path = tmp_path / "bundle.dat"
    with data_path.open("r+b") as handle:
        handle.seek(-1, 2)
        final_byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final_byte[0] ^ 1]))
    assert _source_bundle_fingerprint(bundle_paths) != before


def test_freeze_updates_score_release_without_enabling_case_display(
    tmp_path: Path,
) -> None:
    path = tmp_path / "score_releases.json"
    atomic_write_json(
        path,
        {
            "release_registry_version": "1.0.0",
            "releases": [
                {
                    "release_id": "score@1",
                    "stage": "development",
                    "primary_case_display_allowed": False,
                }
            ],
        },
    )
    _update_score_release_stage(
        path,
        ScoreReleaseStage.FROZEN,
        freeze_hash="freeze-123",
    )
    release = json.loads(path.read_text(encoding="utf-8"))["releases"][0]
    assert release["stage"] == "frozen"
    assert release["freeze_hash"] == "freeze-123"
    assert not release["primary_case_display_allowed"]


def test_run_all_never_extracts_locked_test_waveforms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker import mimic_cli

    config = _config(tmp_path)
    captured = {}
    monkeypatch.setattr(
        mimic_cli,
        "build_inventory",
        lambda *args, **kwargs: SimpleNamespace(errors=[]),
    )
    for name in (
        "ingest_standardized_tables",
        "build_ecg_master_index",
        "build_machine_report_labels",
        "build_note_labels",
            "build_clinical_evidence",
            "build_clinical_labels",
            "attest_development_label_artifacts",
            "ingest_clinical_event_tables",
        "run_development_validation",
        "generate_run_reports",
    ):
        monkeypatch.setattr(mimic_cli, name, lambda *args, **kwargs: None)

    def capture_features(_config, options, **_kwargs):
        captured["include_locked_test"] = options.include_locked_test

    monkeypatch.setattr(mimic_cli, "extract_mimic_features", capture_features)
    args = SimpleNamespace(
        count_rows=False,
        force=False,
        ptb_root=None,
        ptb_feature_cache=None,
        panels="primary_core,sentinel",
        development_only=False,
        max_records=10,
        partition_size=10,
        run_id="development",
        endpoints=None,
        bootstrap_iterations=0,
        no_predefined_scores=False,
        no_ptb_frozen_scores=True,
        ptb_transfer_run_id="ptb-development",
        no_exploratory_scores=False,
        no_all_ecg_secondary=True,
    )
    assert mimic_cli._run_all(config, args) == 0
    assert captured["include_locked_test"] is False


def test_configured_report_is_printable_and_anonymous_by_default(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_root = config.runs_root / "development"
    canonical = run_root / "reports"
    canonical.mkdir(parents=True)
    atomic_write_json(canonical / "report.manifest.json", {"status": "complete"})
    atomic_write_json(
        run_root / "acceptance_pre_freeze_audit.json",
        {
            "status": "ready_to_freeze",
            "audit_hash": "audit-123",
            "locked_test_outcomes_read": False,
            "ptb_fold_10_outcomes_read": False,
        },
    )
    acceptance_frame = pd.DataFrame(
        {
            "check_id": ["formal_train_validation"],
            "status": ["pass"],
        }
    )
    acceptance_frame.to_parquet(
        run_root / "acceptance_pre_freeze_audit.parquet",
        index=False,
    )
    acceptance_frame.to_parquet(
        canonical / "acceptance_audit.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef", "reduced_lvef"],
            "feature_id": ["marker_a", "marker_b"],
            "panel_id": ["primary_core", "sentinel"],
            "evaluation_split": ["validation", "validation"],
            "roc_auc": [0.82, 0.73],
            "average_precision": [0.61, 0.52],
        }
    ).to_parquet(canonical / "screening_results.parquet", index=False)
    pd.DataFrame(
        {
            "subject_id": [17],
            "study_id": [23],
            "endpoint_id": ["reduced_lvef"],
            "panel_id": ["primary_core"],
            "review_status": ["pending"],
        }
    ).to_parquet(canonical / "review_candidates.parquet", index=False)
    pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef"],
            "vcg_feature_id": ["marker_a"],
            "sensitivity_delta_vs_fixed_rule": [0.14],
            "sensitivity_delta_ci_low": [0.04],
            "sensitivity_delta_ci_high": [0.24],
        }
    ).to_parquet(
        canonical / "vcg_vs_classical_comparison.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef"] * 6,
            "feature_id": ["marker_a"] * 6,
            "evaluation_split": ["validation"] * 6,
            "selection_role": ["prespecified_primary_marker"] * 6,
            "point_kind": ["continuous_marker_curve"] * 4
            + ["fixed_reference_point"] * 2,
            "curve_type": [
                "roc",
                "roc",
                "precision_recall",
                "precision_recall",
                "roc",
                "precision_recall",
            ],
            "point_order": [0, 1, 0, 1, 0, 0],
            "x": [0.0, 1.0, 0.0, 1.0, 0.1, 0.4],
            "y": [0.0, 1.0, 1.0, 0.4, 0.7, 0.7],
        }
    ).to_parquet(
        canonical / "performance_curve_points.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef"] * 4,
            "feature_id": ["marker_a"] * 4,
            "evaluation_split": ["validation"] * 4,
            "selection_role": ["prespecified_primary_marker"] * 4,
            "label": [0, 0, 1, 1],
            "bin_center": [-1.0, 0.0, 0.0, 1.0],
            "fraction_within_class": [0.7, 0.3, 0.2, 0.8],
        }
    ).to_parquet(
        canonical / "score_distribution_bins.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef", "reduced_lvef"],
            "evaluation_split": ["validation", "validation"],
            "operating_point": [
                "specificity_90",
                "fixed_reference_point",
            ],
            "specificity": [0.90, 0.88],
            "sensitivity": [0.70, 0.55],
        }
    ).to_parquet(
        canonical / "operating_point_results.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef"],
            "analysis_type": ["vcg_transform_sensitivity"],
            "transform_method": ["inverse_dower"],
            "validation_auc": [0.79],
        }
    ).to_parquet(
        canonical / "overfitting_stability_report.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef"],
            "evaluation_split": ["validation"],
            "calibration_slope": [0.94],
            "brier_score": [0.16],
        }
    ).to_parquet(
        canonical / "adjusted_model_results.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef"] * 2,
            "evaluation_split": ["validation"] * 2,
            "model_id": ["adjusted_a"] * 2,
            "threshold_probability": [0.1, 0.2],
            "net_benefit": [0.08, 0.05],
        }
    ).to_parquet(
        canonical / "decision_curve_results.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "transfer_type": [
                "shared_single_marker",
                "ptb_frozen_score",
            ],
            "ptb_validation_rank_biserial": [0.3, 0.4],
            "mimic_validation_rank_biserial": [0.2, -0.1],
            "train_direction_agreement": [True, False],
        }
    ).to_parquet(
        canonical / "ptb_transfer_matrix.parquet",
        index=False,
    )
    phenome_root = run_root / "phenome"
    phenome_root.mkdir()
    pd.DataFrame(
        {
            "phenotype_id": ["phecode:411.2"],
            "feature_id": ["marker_a"],
            "replicated": [True],
        }
    ).to_parquet(
        phenome_root / "phenome_replicated_hits.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "subject_id": [17],
            "study_id": [23],
            "phenotype_id": ["phecode:411.2"],
        }
    ).to_parquet(
        phenome_root / "current_phenotypes.parquet",
        index=False,
    )
    stale_output = canonical / "builds" / "anonymous-study"
    stale_output.mkdir(parents=True)
    (stale_output / "stale_partial_export.txt").write_text(
        "must not survive atomic publication",
        encoding="utf-8",
    )

    manifest = build_configured_report(
        config,
        ReportSpec(
            report_id="anonymous-study",
            report_type=ReportType.STUDY,
            run_id="development",
            formats=("pdf", "html", "xlsx", "csv", "parquet"),
            sections=(
                "performance",
                "baselines",
                "stability",
                "review",
                "provenance",
            ),
            include_case_appendix=True,
        ),
    )
    outputs = {Path(path).suffix for path in manifest.output_paths}
    assert {".pdf", ".html", ".xlsx", ".csv", ".parquet", ".json"} <= outputs
    output_root = canonical / "builds" / "anonymous-study"
    anonymized = pd.read_parquet(
        output_root / "parquet" / "review_candidates.parquet"
    )
    assert "subject_id" not in anonymized
    assert "study_id" not in anonymized
    assert (
        output_root / "parquet" / "ptb_transfer_matrix.parquet"
    ).is_file()
    assert (
        output_root / "parquet" / "acceptance_audit.parquet"
    ).is_file()
    assert (
        output_root / "parquet" / "phenome_replicated_hits.parquet"
    ).is_file()
    assert not (
        output_root / "parquet" / "phenome_current_phenotypes.parquet"
    ).exists()
    assert not (output_root / "stale_partial_export.txt").exists()
    provenance = pd.read_parquet(
        output_root / "parquet" / "report_provenance.parquet"
    ).iloc[0]
    assert provenance["analysis_status"] == "partial_preview"
    assert provenance["artifact_completeness"] == "partial"
    assert provenance["evaluation_splits"] == "validation"
    assert provenance["acceptance_status"] == "ready_to_freeze"
    assert provenance["acceptance_audit_hash"] == "audit-123"
    assert not provenance["acceptance_locked_test_outcomes_read"]
    assert not provenance["acceptance_ptb_fold_10_outcomes_read"]
    html_text = (output_root / "anonymous-study.html").read_text(
        encoding="utf-8"
    )
    assert "data:image/png;base64," in html_text
    assert "Baseline Delta Forest" in html_text
    assert "Validation Roc Pr" in html_text
    assert "Score Distribution" in html_text
    assert "Operating Points" in html_text
    assert "Sensitivity Stability" in html_text
    assert "Adjusted Utility" in html_text
    assert "Status: partial_preview" in html_text
    assert "Locked test: sealed_not_evaluated" in html_text
    assert "Acceptance: ready_to_freeze" in html_text
    assert (output_root / "anonymous-study.pdf").stat().st_size > 10_000


def test_methodological_controls_are_not_ranked_or_promoted_as_biomarkers() -> None:
    ranked = _formal_candidate_marker_rows(
        pd.DataFrame(
            {
                "feature_id": [
                    "fiducial_confidence",
                    "qrs_t_hausdorff_distance",
                    "spatial_qrs_t_angle_deg",
                ],
                "inference_status": [
                    "formal_inference",
                    "descriptive_only",
                    "formal_inference",
                ],
                "roc_auc": [0.91, 0.89, 0.72],
            }
        )
    )
    assert ranked["feature_id"].tolist() == [
        "spatial_qrs_t_angle_deg"
    ]
    assert (
        feature_analysis_role("fiducial_confidence")
        == "methodological_qc_control"
    )

    promoted = _promotion_candidates(
        pd.DataFrame(
            {
                "phenotype_id": ["CV_401", "CV_401", "CV_999"],
                "feature_id": [
                    "fiducial_confidence",
                    "spatial_qrs_t_angle_deg",
                    "quality_score",
                ],
                "q_value": [0.0001, 0.01, 0.0001],
                "roc_auc": [0.91, 0.72, 0.95],
            }
        ),
        snapshot_hash="a" * 64,
    )
    assert len(promoted) == 1
    assert promoted[0].phenotype_id == "CV_401"
    assert promoted[0].feature_ids == ("spatial_qrs_t_angle_deg",)
    # roc_auc/q_value are present on the source frame -> populated;
    # CI bounds and train_q_value are absent on this synthetic frame ->
    # must degrade to None rather than raising a KeyError.
    assert promoted[0].effect_size == pytest.approx(0.72)
    assert promoted[0].validation_q_value == pytest.approx(0.01)
    assert promoted[0].effect_ci_low is None
    assert promoted[0].effect_ci_high is None
    assert promoted[0].train_q_value is None


def test_promotion_candidates_populate_effect_size_and_ci_when_available() -> None:
    promoted = _promotion_candidates(
        pd.DataFrame(
            {
                "phenotype_id": ["CV_401"],
                "feature_id": ["spatial_qrs_t_angle_deg"],
                "q_value": [0.01],
                "train_q_value": [0.002],
                "roc_auc": [0.72],
                "roc_auc_ci_low": [0.65],
                "roc_auc_ci_high": [0.79],
            }
        ),
        snapshot_hash="b" * 64,
    )
    assert len(promoted) == 1
    candidate = promoted[0]
    assert candidate.effect_size == pytest.approx(0.72)
    assert candidate.effect_ci_low == pytest.approx(0.65)
    assert candidate.effect_ci_high == pytest.approx(0.79)
    assert candidate.train_q_value == pytest.approx(0.002)
    assert candidate.validation_q_value == pytest.approx(0.01)
    # Round-trips through to_dict() (the exact path phenome_promotions.py
    # uses to build the persisted parquet/JSON) without dropping fields.
    as_dict = candidate.to_dict()
    assert as_dict["effect_size"] == pytest.approx(0.72)
    assert as_dict["effect_ci_low"] == pytest.approx(0.65)


def test_promotion_candidates_tag_seeded_from_promotion_with_caveat() -> None:
    frame = pd.DataFrame(
        {
            "phenotype_id": ["CV_401", "CV_500"],
            "feature_id": ["spatial_qrs_t_angle_deg", "spatial_qrs_t_angle_deg"],
            "q_value": [0.01, 0.02],
            "roc_auc": [0.72, 0.70],
        }
    )
    unseeded = _promotion_candidates(frame, snapshot_hash="c" * 64)
    assert all(
        candidate.proposed_label_definition["seeded_from_promotion"] is False
        for candidate in unseeded
    )
    assert all(
        "seeded_from_promotion_caveat" not in candidate.proposed_label_definition
        for candidate in unseeded
    )

    seeded = _promotion_candidates(
        frame,
        snapshot_hash="c" * 64,
        seeded_phenotype_ids=frozenset({"CV_401"}),
    )
    by_phenotype = {c.phenotype_id: c for c in seeded}
    assert by_phenotype["CV_401"].proposed_label_definition[
        "seeded_from_promotion"
    ] is True
    assert (
        "seeded_from_promotion_caveat"
        in by_phenotype["CV_401"].proposed_label_definition
    )
    # Only the phenotype actually in seeded_phenotype_ids is tagged --
    # this must never silently spread to unrelated candidates.
    assert by_phenotype["CV_500"].proposed_label_definition[
        "seeded_from_promotion"
    ] is False


def test_phenome_promotion_refresh_replaces_legacy_qc_candidates(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    phenome_root = config.runs_root / "development" / "phenome"
    phenome_root.mkdir(parents=True)
    replicated_path = phenome_root / "phenome_replicated_hits.parquet"
    atomic_write_parquet(
        pd.DataFrame(
            {
                "phenotype_id": ["P1", "P1", "P2"],
                "feature_id": [
                    "fiducial_confidence",
                    "spatial_qrs_t_angle_deg",
                    "quality_score",
                ],
                "q_value": [0.0001, 0.01, 0.001],
                "roc_auc": [0.91, 0.72, 0.95],
            }
        ),
        replicated_path,
    )
    source_manifest = StageManifest.complete(
        stage="discover-phenome",
        stage_version="test",
        input_hash="source-input",
        config_hash="source-config",
        output_paths=(str(replicated_path.resolve()),),
        notes=(
            "locked_test_scanned=false",
            f"feature_snapshot_hash={'a' * 64}",
        ),
    )
    source_manifest.save(phenome_root / "phenome.manifest.json")

    first = refresh_phenome_promotions(config, "development")
    refreshed = pd.read_parquet(
        phenome_root / "phenome_promotion_candidates.parquet"
    )
    verification = verify_phenome_promotion_refresh(
        config.runs_root / "development"
    )
    second = refresh_phenome_promotions(config, "development")

    assert refreshed["phenotype_id"].tolist() == ["P1"]
    assert tuple(refreshed.iloc[0]["feature_ids"]) == (
        "spatial_qrs_t_angle_deg",
    )
    assert (
        first.row_counts["excluded_methodological_top_candidates"]
        == 2
    )
    assert first.row_counts["invalid_output_feature_roles"] == 0
    assert verification["valid"] is True
    assert verification["invalid_feature_role_counts"] == {}
    assert second.created_at == first.created_at


def test_excel_export_splits_tables_before_the_sheet_row_limit() -> None:
    frame = pd.DataFrame(
        {
            "row_id": np.arange(11),
            "value": np.arange(11) * 2,
        }
    )

    chunks = _excel_table_chunks(
        "large_discovery_table",
        frame,
        max_data_rows=5,
    )

    assert [name for name, _ in chunks] == [
        "large_discovery_table_part_01",
        "large_discovery_table_part_02",
        "large_discovery_table_part_03",
    ]
    assert [len(chunk) for _, chunk in chunks] == [5, 5, 1]
    pd.testing.assert_frame_equal(
        pd.concat([chunk for _, chunk in chunks]),
        frame,
    )


def test_excel_preview_budget_is_fair_deterministic_and_documented() -> None:
    tables = {
        "wide_results": pd.DataFrame(
            np.arange(20_000).reshape(1_000, 20)
        ),
        "narrow_results": pd.DataFrame(
            np.arange(2_000).reshape(1_000, 2)
        ),
    }

    first = _excel_export_tables(
        tables,
        max_total_cells=1_000,
        max_rows_per_table=1_000,
        min_rows_per_table=10,
    )
    second = _excel_export_tables(
        tables,
        max_total_cells=1_000,
        max_rows_per_table=1_000,
        min_rows_per_table=10,
    )

    exported_cells = sum(
        len(first[name]) * len(tables[name].columns)
        for name in tables
    )
    assert exported_cells <= 1_000
    assert all(len(first[name]) >= 10 for name in tables)
    assert {
        row.table_name: row.workbook_rows
        for row in first["export_index"].itertuples()
    } == {
        row.table_name: row.workbook_rows
        for row in second["export_index"].itertuples()
    }
    assert set(first["export_index"]["complete_formats"]) == {
        "CSV; Parquet"
    }
    assert first["export_index"]["omitted_rows"].gt(0).all()


def test_report_acceptance_path_prefers_post_freeze_audit(
    tmp_path: Path,
) -> None:
    pre_json = tmp_path / "acceptance_pre_freeze_audit.json"
    pre_parquet = tmp_path / "acceptance_pre_freeze_audit.parquet"
    post_json = tmp_path / "acceptance_audit.json"
    post_parquet = tmp_path / "acceptance_audit.parquet"
    atomic_write_json(pre_json, {"status": "ready_to_freeze"})
    pd.DataFrame({"status": ["pending"]}).to_parquet(pre_parquet, index=False)
    assert resolve_acceptance_audit_paths(tmp_path) == (
        pre_json,
        pre_parquet,
    )

    atomic_write_json(post_json, {"status": "accepted_frozen_development"})
    pd.DataFrame({"status": ["pass"]}).to_parquet(post_parquet, index=False)
    assert resolve_acceptance_audit_paths(tmp_path) == (
        post_json,
        post_parquet,
    )


def test_resume_fragments_with_all_na_columns_concatenate_without_a_warning(tmp_path):
    """One fragment abstains for a marker, the next measures it; a marker no
    fragment measured keeps its column. pandas' concat deprecation warning must
    not fire, because the extraction runs under ``-W error``."""
    import warnings

    from vcg_biomarker.mimic.feature_extraction import (
        _ResumeFragmentState, _materialize_resume_fragments,
    )

    root = tmp_path / "resume"
    (root / "split=train").mkdir(parents=True)
    first = pd.DataFrame({
        "study_id": [1, 2], "extraction_status": ["ok", "ok"],
        "marker": [np.nan, np.nan], "never": [np.nan, np.nan], "split": ["train"] * 2,
    })
    second = pd.DataFrame({
        "study_id": [3, 4], "extraction_status": ["ok", "failed"],
        "marker": [1.5, 2.5], "never": [np.nan, np.nan], "split": ["train"] * 2,
    })
    first.to_parquet(root / "split=train" / "a.parquet", index=False)
    second.to_parquet(root / "split=train" / "b.parquet", index=False)
    state = _ResumeFragmentState(
        root=root, manifest_path=root / "manifest.json", part_hash="p", execution_id="e",
        expected_study_ids=frozenset({1, 2, 3, 4}),
        fragments=({"relative_path": "split=train/a.parquet", "study_ids": [1, 2]},
                   {"relative_path": "split=train/b.parquet", "study_ids": [3, 4]}),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = _materialize_resume_fragments(state)
    assert frame["study_id"].tolist() == [1, 2, 3, 4]
    assert frame["marker"].tolist()[2:] == [1.5, 2.5] and frame["marker"].isna().tolist()[:2] == [True, True]
    assert "never" in frame and frame["never"].isna().all()
