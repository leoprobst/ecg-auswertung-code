from __future__ import annotations

import gc
import json
import os
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.config import (
    MimicProjectConfig,
    scientific_config_payload,
)
from vcg_biomarker.mimic.evidence import _medication_windows_query
from vcg_biomarker.mimic import feature_extraction, feature_record
from vcg_biomarker.mimic.jobs import (
    JOB_PROJECT_CONFIG_KEY,
    JobStore,
    remove_confirmed_job_run_manifest_root,
    validate_job_run_manifest_root,
)
from vcg_biomarker.mimic.models import JobState
from vcg_biomarker.mimic.resources import HardwareInventory
from vcg_biomarker.mimic.storage import atomic_write_json, content_hash
from vcg_biomarker.mimic_cli import _build_parser


def _config(tmp_path: Path, *, workers: int = 4, memory_gb: int = 12) -> MimicProjectConfig:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic-iv",
        mimic_ed_root=tmp_path / "mimic-ed",
        mimic_ecg_root=tmp_path / "mimic-ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "echo-study.csv",
        echo_record_list_path=tmp_path / "echo-record.csv",
        work_root=tmp_path / "work",
        workers=workers,
        duckdb_memory_limit_gb=memory_gb,
        os_memory_reserve_gb=6.0,
    )
    config.ensure_directories()
    config.save()
    return config


def test_scientific_config_hash_ignores_only_operational_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    baseline = content_hash(scientific_config_payload(config))
    config.workers = 9
    config.duckdb_memory_limit_gb = 7
    config.feature_micro_batch_size = 100
    config.metadata = {"operator_note": "resource tuning"}
    assert content_hash(scientific_config_payload(config)) == baseline

    config.powerline_hz = 50.0
    assert content_hash(scientific_config_payload(config)) != baseline


def test_resume_rebinds_resources_but_preserves_scientific_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import resources

    hardware = HardwareInventory(
        physical_cores=8,
        logical_cores=16,
        total_memory_gb=32.0,
        available_memory_gb=24.0,
        cpu_name="test",
        platform="test",
    )
    monkeypatch.setattr(resources, "detect_hardware", lambda: hardware)
    config = _config(tmp_path, workers=14, memory_gb=24)
    store = JobStore.for_project(config)
    record = store.create(
        stage="study-run",
        project_file=config.project_file,
        arguments={
            "run_id": "resume-rebind-test",
            "preset_id": "exhaustive_research",
        },
    )
    store.request_pause(record.spec.job_id)

    config.workers = 3
    config.duckdb_memory_limit_gb = 7
    config.powerline_hz = 50.0
    config.save()
    resumed = store.request_resume(record.spec.job_id)

    assert resumed.state == JobState.QUEUED
    assert resumed.spec.resource_profile.cpu_tokens == 3
    assert resumed.spec.resource_profile.memory_gb == 7.0
    snapshot = dict(resumed.spec.arguments[JOB_PROJECT_CONFIG_KEY])
    assert snapshot["workers"] == 3
    assert snapshot["duckdb_memory_limit_gb"] == 7
    assert snapshot["powerline_hz"] == 60.0
    assert any(
        event["event_type"] == "operational_config_rebound"
        for event in store.events(record.spec.job_id)
    )


def test_retry_rebinds_resources_but_preserves_scientific_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import resources

    hardware = HardwareInventory(
        physical_cores=8,
        logical_cores=16,
        total_memory_gb=32.0,
        available_memory_gb=24.0,
        cpu_name="test",
        platform="test",
    )
    monkeypatch.setattr(resources, "detect_hardware", lambda: hardware)
    config = _config(tmp_path, workers=3, memory_gb=4)
    store = JobStore.for_project(config)
    record = store.create(
        stage="study-run",
        project_file=config.project_file,
        arguments={"run_id": "retry-rebind-test"},
    )
    store.claim(record.spec.job_id, worker_pid=12345)
    store.transition(record.spec.job_id, JobState.FAILED, message="oom")

    config.workers = 6
    config.duckdb_memory_limit_gb = 14
    config.powerline_hz = 50.0
    config.save()
    retried = store.retry(record.spec.job_id)

    assert retried.state == JobState.QUEUED
    assert retried.spec.resource_profile.cpu_tokens == 6
    assert retried.spec.resource_profile.memory_gb == 14.0
    snapshot = dict(retried.spec.arguments[JOB_PROJECT_CONFIG_KEY])
    assert snapshot["workers"] == 6
    assert snapshot["duckdb_memory_limit_gb"] == 14
    assert snapshot["powerline_hz"] == 60.0


def test_job_delete_is_atomic_and_rejects_active_jobs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = JobStore.for_project(config)
    paused = store.create(
        stage="inventory",
        project_file=config.project_file,
        job_id="paused-delete",
    )
    store.request_pause(paused.spec.job_id)
    assert store.events(paused.spec.job_id)
    deleted = store.delete(paused.spec.job_id)
    assert deleted.state == JobState.PAUSED
    with pytest.raises(KeyError):
        store.get(paused.spec.job_id)
    assert store.events(paused.spec.job_id) == []

    running = store.create(
        stage="inventory",
        project_file=config.project_file,
        job_id="running-keep",
    )
    store.claim(running.spec.job_id, 12345)
    with pytest.raises(ValueError, match="cannot be deleted"):
        store.delete(running.spec.job_id)
    assert store.get(running.spec.job_id).state == JobState.RUNNING


def test_jobs_delete_cli_requires_explicit_identity_confirmation() -> None:
    args = _build_parser().parse_args(
        [
            "jobs",
            "delete",
            "job-123",
            "--confirm-job-id",
            "job-123",
            "--delete-run-manifest-root",
            "run-root",
            "--confirm-delete-run-manifest-root",
        ]
    )
    assert args.jobs_command == "delete"
    assert args.job_id == args.confirm_job_id == "job-123"
    assert args.confirm_delete_run_manifest_root is True


def test_feature_worker_batches_records_without_repeating_task_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []

    def fake_worker(row: dict) -> dict:
        seen.append(int(row["study_id"]))
        return {"study_id": int(row["study_id"]), "value": 1.0}

    monkeypatch.setattr(feature_record, "_extract_record_worker", fake_worker)
    rows = [{"study_id": 1}, {"study_id": 2}, {"study_id": 3}]
    assert feature_record._extract_record_batch_worker(rows) == [
        {"study_id": 1, "value": 1.0},
        {"study_id": 2, "value": 1.0},
        {"study_id": 3, "value": 1.0},
    ]
    assert seen == [1, 2, 3]


def test_feature_partition_sha_is_reused_only_while_file_identity_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "part.parquet"
    path.write_bytes(b"first")
    payload = {
        "parquet_size": path.stat().st_size,
        "parquet_sha256": feature_extraction.file_sha256(path),
    }
    feature_extraction._VERIFIED_PARTITION_SHA_CACHE.clear()
    original = feature_extraction.file_sha256
    calls: list[Path] = []

    def counted(source):
        calls.append(Path(source))
        return original(source)

    monkeypatch.setattr(feature_extraction, "file_sha256", counted)
    assert feature_extraction._partition_file_matches(path, payload)
    assert feature_extraction._partition_file_matches(path, payload)
    assert len(calls) == 1

    path.write_bytes(b"other")
    assert not feature_extraction._partition_file_matches(path, payload)
    assert len(calls) == 2


def test_isolated_duckdb_scratch_is_project_local_unique_and_cleaned(tmp_path):
    from vcg_biomarker.mimic.storage import DuckDBCatalog

    root = tmp_path / "project's catalog"
    root.mkdir()
    sentinel = root / "unrelated.txt"
    sentinel.write_text("preserve")
    catalog = DuckDBCatalog(root / "catalog.duckdb")
    with catalog.connect(isolated=True) as first:
        first_path = Path(first.execute("SELECT current_setting('temp_directory')").fetchone()[0])
        assert first_path.parent == root.resolve()
        assert first_path.is_dir()
        with pytest.raises(RuntimeError, match="synthetic failure"):
            with catalog.connect(isolated=True) as second:
                second_path = Path(second.execute("SELECT current_setting('temp_directory')").fetchone()[0])
                assert second_path != first_path
                raise RuntimeError("synthetic failure")
        assert not second_path.exists()
        assert first_path.is_dir()
    assert not first_path.exists()
    assert sentinel.read_text() == "preserve"


def _complete_test_feature_artifact(root: Path) -> Path:
    partition = root / "split=train" / "part-000000.parquet"
    partition.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "study_id": [11, 12],
            "split": ["train", "train"],
            "value": [0.5, 0.6],
        }
    ).to_parquet(partition, index=False)
    atomic_write_json(
        partition.with_suffix(".json"),
        {
            "part_hash": "part-hash",
            "study_ids": [11, 12],
            "row_count": 2,
            "parquet_size": partition.stat().st_size,
            "parquet_sha256": feature_extraction.file_sha256(partition),
        },
    )
    atomic_write_json(root / "features.manifest.json", {"status": "complete"})
    stage_input_payload = {"test": "persistent-verification"}
    feature_extraction._write_feature_artifact_index(
        root,
        stage_input_hash=content_hash(stage_input_payload),
        stage_input_payload=stage_input_payload,
        source_code_hash=feature_extraction.feature_implementation_hash(),
        panel_hash="panel-hash",
        feature_ids=("value",),
        parquet_paths=(partition,),
        destination=root / "partition_index.json",
    )
    return partition


def test_feature_index_excludes_timing_metadata_but_checks_partition_integrity(tmp_path):
    root = tmp_path / "features"
    partition = _complete_test_feature_artifact(root)
    timing = root / "feature_timing_contract.json"
    atomic_write_json(timing, {"stage": "extract-features"})
    paths = feature_extraction._feature_partition_paths([str(timing), partition])
    assert paths == [partition]
    kwargs = dict(
        stage_input_hash="input", stage_input_payload={}, source_code_hash="source",
        panel_hash="panel", feature_ids=("value",), parquet_paths=paths,
        destination=root / "partition_index.json",
    )
    indexed = feature_extraction._write_feature_artifact_index(root, **kwargs)
    assert indexed["partition_count"] == 1
    assert indexed["row_count"] == 2
    assert timing.is_file()
    partition.write_bytes(b"corrupt partition must still fail")
    with pytest.raises(RuntimeError, match="incomplete or corrupt feature partition"):
        feature_extraction._write_feature_artifact_index(root, **kwargs)


def test_complete_feature_verification_attestation_survives_process_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "features"
    partition = _complete_test_feature_artifact(root)
    first = feature_extraction.verify_feature_artifacts((partition,))
    attestation_path = (
        root / feature_extraction.FEATURE_VERIFICATION_ATTESTATION_NAME
    )
    assert first["verification_mode"] == "full"
    assert first["verification_attestation_status"] == "written"
    assert attestation_path.is_file()
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation["complete_index"] is True
    assert attestation["binding"]["partition_index"]["sha256"]
    assert attestation["binding"]["feature_manifest"]["sha256"]
    assert attestation["files"][0]["manifest"]["sha256"]
    assert {
        "size",
        "mtime_ns",
        "ctime_ns",
        "change_time_ns",
    } <= set(attestation["files"][0]["parquet"])

    # Clearing the process-local cache emulates the next supervised stage.
    feature_extraction._VERIFIED_PARTITION_SHA_CACHE.clear()
    original_sha = feature_extraction.file_sha256
    original_read = feature_extraction._read_single_parquet
    partition_hashes: list[Path] = []
    partition_reads: list[Path] = []

    def counted_sha(path):
        source = Path(path).resolve()
        if source == partition.resolve():
            partition_hashes.append(source)
        return original_sha(path)

    def counted_read(path, *args, **kwargs):
        source = Path(path).resolve()
        if source == partition.resolve():
            partition_reads.append(source)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(feature_extraction, "file_sha256", counted_sha)
    monkeypatch.setattr(feature_extraction, "_read_single_parquet", counted_read)
    second = feature_extraction.verify_feature_artifacts((partition,))
    assert second["verification_mode"] == "persistent_attestation"
    assert partition_hashes == []
    assert partition_reads == []


def test_feature_attestation_fails_closed_after_partition_or_manifest_change(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    partition = _complete_test_feature_artifact(root)
    feature_extraction.verify_feature_artifacts((partition,))
    feature_extraction._VERIFIED_PARTITION_SHA_CACHE.clear()

    # A restored mtime and unchanged size are insufficient on Windows: the
    # native filesystem ChangeTime token must still invalidate the cache.
    original = partition.read_bytes()
    original_stat = partition.stat()
    changed = bytearray(original)
    changed[len(changed) // 2] ^= 1
    partition.write_bytes(changed)
    os.utime(
        partition,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        feature_extraction.verify_feature_artifacts((partition,))

    partition.write_bytes(original)
    partition_manifest_path = partition.with_suffix(".json")
    partition_manifest = json.loads(
        partition_manifest_path.read_text(encoding="utf-8")
    )
    partition_manifest["audit_note"] = "descriptor changed"
    atomic_write_json(partition_manifest_path, partition_manifest)
    feature_extraction._VERIFIED_PARTITION_SHA_CACHE.clear()
    refreshed = feature_extraction.verify_feature_artifacts((partition,))
    assert refreshed["verification_mode"] == "full"
    assert refreshed["verification_attestation_status"] == "written"


def test_feature_resume_fragments_are_immutable_and_fault_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".part-000000.resume"
    manifest = tmp_path / ".part-000000.resume.json"
    expected = {11, 12}
    state, resumed = feature_extraction._open_resume_fragments(
        root,
        manifest,
        part_hash="part-hash",
        expected_study_ids=expected,
        force=True,
        execution_id="job-a",
    )
    assert resumed.empty
    first = pd.DataFrame(
        {"study_id": [11], "extraction_status": ["ok"], "value": [1.0]}
    )

    original_write_json = feature_extraction.atomic_write_json

    def fail_manifest_commit(*args, **kwargs):
        raise OSError("injected manifest commit failure")

    monkeypatch.setattr(
        feature_extraction,
        "atomic_write_json",
        fail_manifest_commit,
    )
    with pytest.raises(OSError, match="injected"):
        feature_extraction._append_resume_fragment(
            first,
            root,
            manifest,
            part_hash="part-hash",
            execution_id="job-a",
            expected_study_ids=expected,
            state=state,
        )
    assert not manifest.exists()
    assert len(list(root.glob("fragment-*.parquet"))) == 1

    monkeypatch.setattr(
        feature_extraction,
        "atomic_write_json",
        original_write_json,
    )
    state = feature_extraction._append_resume_fragment(
        first,
        root,
        manifest,
        part_hash="part-hash",
        execution_id="job-a",
        expected_study_ids=expected,
        state=state,
    )
    first_path = root / state.fragments[0]["relative_path"]
    first_bytes = first_path.read_bytes()
    state = feature_extraction._append_resume_fragment(
        pd.DataFrame(
            {"study_id": [12], "extraction_status": ["failed"], "value": [2.0]}
        ),
        root,
        manifest,
        part_hash="part-hash",
        execution_id="job-a",
        expected_study_ids=expected,
        state=state,
    )
    assert first_path.read_bytes() == first_bytes
    assert len(state.fragments) == 2
    loaded = feature_extraction._load_resume_fragments(
        root,
        manifest,
        part_hash="part-hash",
        expected_study_ids=expected,
        force=True,
        execution_id="job-a",
    ).sort_values("study_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        loaded,
        pd.concat(
            [
                first,
                pd.DataFrame(
                    {
                        "study_id": [12],
                        "extraction_status": ["failed"],
                        "value": [2.0],
                    }
                ),
            ],
            ignore_index=True,
        ),
    )


def test_streamed_phenome_train_screen_matches_wide_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import phenome

    config = _config(tmp_path)
    full, current, definitions = _phenome_stream_fixture()
    train = full.loc[full["split"].eq("train")].reset_index(drop=True)
    reference = phenome._checkpointed_train_screen(
        tmp_path / "wide",
        input_hash="wide",
        features=train,
        current=current.loc[current["split"].eq("train")],
        history_body=pd.DataFrame(),
        phenotype_ids=("p1", "p2"),
        feature_ids=("f1", "f2"),
        panel_map={"f1": "panel", "f2": "panel"},
        definition_lookup=definitions,
        min_inference=1,
        min_formal=1,
        progress=None,
    )
    widths: list[int] = []

    def load_block(*args, feature_ids, splits=("train", "validation"), **kwargs):
        widths.append(len(feature_ids))
        selected = full.loc[full["split"].isin(splits)]
        return selected[["subject_id", "study_id", "split", *feature_ids]].reset_index(
            drop=True
        )

    monkeypatch.setattr(phenome, "_load_phenome_feature_block", load_block)
    monkeypatch.setattr(
        phenome,
        "_feature_blocks",
        lambda feature_ids, size=32: tuple((item,) for item in feature_ids),
    )
    index = full.drop(columns=["f1", "f2"])
    streamed, available = phenome._checkpointed_train_screen_streaming(
        tmp_path / "stream",
        config=config,
        feature_glob="unused",
        index_path=tmp_path / "unused-index.parquet",
        input_hash="stream",
        feature_index=index,
        current=current.loc[current["split"].eq("train")],
        history_body=pd.DataFrame(),
        phenotype_ids=("p1", "p2"),
        feature_ids=("f1", "f2"),
        panel_map={"f1": "panel", "f2": "panel"},
        definition_lookup=definitions,
        min_inference=1,
        min_formal=1,
        progress=None,
    )
    assert available == ("f1", "f2")
    assert widths == [1, 1]
    keys = ["phenotype_id", "feature_id"]
    pd.testing.assert_frame_equal(
        streamed.sort_values(keys).reset_index(drop=True),
        reference.sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )


def test_streamed_phenome_replication_matches_wide_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import phenome

    config = _config(tmp_path, workers=1)
    full, current, definitions = _phenome_stream_fixture()
    hits = pd.DataFrame(
        {
            "phenotype_id": ["p1", "p2"],
            "phenotype_name": ["P1", "P2"],
            "body_system": ["A", "B"],
            "panel_id": ["panel", "panel"],
            "feature_id": ["f1", "f2"],
            "q_value": [0.01, 0.02],
            "aligned_auc": [0.8, 0.8],
            "direction": [1, 1],
            "replication_seed_index": [1, 2],
        }
    )
    reference = phenome._checkpointed_validation_replication(
        tmp_path / "wide-replication",
        input_hash="wide-replication",
        train_hits=hits,
        features=full,
        current=current,
        history_body=pd.DataFrame(),
        definition_lookup=definitions,
        min_inference=1,
        min_formal=1,
        bootstrap_iterations=0,
        random_seed=123,
        worker_count=1,
        progress=None,
    )
    widths: list[int] = []

    def load_block(*args, feature_ids, splits=("train", "validation"), **kwargs):
        widths.append(len(feature_ids))
        selected = full.loc[full["split"].isin(splits)]
        return selected[["subject_id", "study_id", "split", *feature_ids]].reset_index(
            drop=True
        )

    monkeypatch.setattr(phenome, "_load_phenome_feature_block", load_block)
    monkeypatch.setattr(
        phenome,
        "_feature_blocks",
        lambda feature_ids, size=32: tuple((item,) for item in feature_ids),
    )
    streamed = phenome._checkpointed_validation_replication_streaming(
        tmp_path / "stream-replication",
        config=config,
        feature_glob="unused",
        index_path=tmp_path / "unused-index.parquet",
        input_hash="stream-replication",
        train_hits=hits,
        feature_index=full.drop(columns=["f1", "f2"]),
        current=current,
        history_body=pd.DataFrame(),
        definition_lookup=definitions,
        min_inference=1,
        min_formal=1,
        bootstrap_iterations=0,
        random_seed=123,
        worker_count=1,
        progress=None,
        checkpoint_namespace="validation",
        progress_label="Validation",
    )
    assert widths == [1, 1]
    keys = ["phenotype_id", "feature_id", "panel_id"]
    pd.testing.assert_frame_equal(
        streamed.sort_values(keys).reset_index(drop=True),
        reference.sort_values(keys).reset_index(drop=True),
        check_dtype=False,
    )


def test_all_project_stages_use_scientific_config_identity() -> None:
    mimic_root = Path(__file__).resolve().parents[1] / "mimic"
    stage_files = (
        "combination_stage.py",
        "group_vcg.py",
        "ingest.py",
        "inventory.py",
        "labels.py",
        "locking.py",
        "master_index.py",
        "note_labels.py",
        "phenome_promotions.py",
        "physiological_gap_models.py",
        "ptb_transfer.py",
        "repeatability_stage.py",
        "reports.py",
        "report_builder.py",
        "validation.py",
        "workflow.py",
    )
    for filename in stage_files:
        source = (mimic_root / filename).read_text(encoding="utf-8")
        assert "content_hash(config.to_dict())" not in source, filename
        assert "scientific_config_hash" in source, filename
        assert "operational_config_hash" in source, filename


def test_run_manifest_root_requires_exact_job_owned_single_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = JobStore.for_project(config)
    record = store.create(
        stage="study-run",
        project_file=config.project_file,
        arguments={"run_id": "owned-run", "preset_id": "exhaustive_research"},
        job_id="owned-job",
    )
    record = store.request_pause(record.spec.job_id)
    root = config.runs_root / "owned-run"
    root.mkdir(parents=True)
    (root / "guided_workflow.manifest.json").write_text(
        json.dumps(
            {
                "study_spec": {
                    "run_id": "owned-run",
                    "preset_id": "exhaustive_research",
                }
            }
        ),
        encoding="utf-8",
    )
    assert validate_job_run_manifest_root(record, config, root) == root.resolve()
    (root / "unexpected.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="nothing else"):
        validate_job_run_manifest_root(record, config, root)
    (root / "unexpected.txt").unlink()

    deleted = store.delete(record.spec.job_id)
    removed = remove_confirmed_job_run_manifest_root(deleted, config, root)
    assert removed == root.resolve()
    assert not root.exists()


def test_phenome_screen_does_not_retain_control_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import phenome

    features = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "study_id": [11, 22, 33, 44, 55, 66],
            "is_control_index": [False, False, True, True, True, True],
            "f1": [3.0, 4.0, 1.0, 1.5, 2.0, 2.5],
            "f2": [8.0, 7.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    current = pd.DataFrame(
        {
            "phenotype_id": ["p1", "p2"],
            "subject_id": [1, 2],
            "study_id": [11, 22],
        }
    )
    definitions = pd.DataFrame(
        {
            "phenotype_id": ["p1", "p2"],
            "display_name": ["P1", "P2"],
            "body_system": ["A", "B"],
        }
    ).set_index("phenotype_id")
    original = phenome.prepare_sorted_control_distribution
    references: list[weakref.ReferenceType[np.ndarray]] = []
    peak_alive = 0

    def tracked(values):
        nonlocal peak_alive
        distribution = original(values)
        references.append(weakref.ref(distribution.sorted_values))
        peak_alive = max(
            peak_alive,
            sum(reference() is not None for reference in references),
        )
        return distribution

    monkeypatch.setattr(phenome, "prepare_sorted_control_distribution", tracked)
    records = phenome._screen_split(
        features,
        current,
        pd.DataFrame(),
        phenotype_ids=("p1", "p2"),
        feature_ids=("f1", "f2"),
        panel_map={"f1": "test", "f2": "test"},
        definition_lookup=definitions,
        min_inference=1,
        min_formal=1,
        progress=None,
    )
    gc.collect()
    assert len(records) == 4
    assert peak_alive <= 2
    assert sum(reference() is not None for reference in references) == 0


def test_partitioned_medication_range_join_matches_original_or_query(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    ecg_time = pd.Timestamp("2024-01-03 12:00:00")
    master_frame = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "study_id": [101, 202],
            "ecg_time": [ecg_time, ecg_time],
        }
    )
    medication_frame = pd.DataFrame(
        [
            _medication_row(1, "emar", "digoxin", ecg_time - pd.Timedelta(hours=30)),
            _medication_row(
                1,
                "prescriptions",
                "beta_blocker",
                ecg_time - pd.Timedelta(hours=1),
                start=ecg_time - pd.Timedelta(days=2),
                end=ecg_time + pd.Timedelta(days=2),
            ),
            _medication_row(
                1,
                "pharmacy",
                "ace_inhibitor",
                ecg_time - pd.Timedelta(days=10),
                start=ecg_time - pd.Timedelta(days=20),
                end=None,
            ),
            _medication_row(2, "pyxis", "diuretic", ecg_time),
            _medication_row(2, "medrecon", "diuretic", ecg_time - pd.Timedelta(hours=3)),
            _medication_row(2, "emar", "ignored", ecg_time - pd.Timedelta(days=5)),
        ]
    )
    master_path = tmp_path / "master.parquet"
    medication_path = tmp_path / "medications.parquet"
    connection.register("master_frame", master_frame)
    connection.register("medication_frame", medication_frame)
    connection.execute(f"COPY master_frame TO '{master_path.as_posix()}' (FORMAT PARQUET)")
    connection.execute(
        f"COPY medication_frame TO '{medication_path.as_posix()}' (FORMAT PARQUET)"
    )

    old_query = _original_medication_window_query(
        master_path.as_posix(),
        medication_path.as_posix(),
    )
    partition_queries = [
        _medication_windows_query(
            master_path.as_posix(),
            medication_path.as_posix(),
            partition_index=index,
            partition_count=4,
        )
        for index in range(4)
    ]
    assert all(query.count("hash(subject_id)") == 2 for query in partition_queries)
    new_query = " UNION ALL ".join(f"({query})" for query in partition_queries)
    old = connection.execute(old_query).fetchdf()
    new = connection.execute(new_query).fetchdf()
    sort_columns = ["subject_id", "study_id", "medication_class"]
    old = old.sort_values(sort_columns).reset_index(drop=True)
    new = new.sort_values(sort_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(new, old, check_dtype=False)


def _phenome_stream_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = list(range(1, 9)) + list(range(101, 109))
    splits = ["train"] * 8 + ["validation"] * 8
    full = pd.DataFrame(
        {
            "subject_id": subjects,
            "study_id": [value * 10 for value in subjects],
            "split": splits,
            "is_control_index": True,
            "is_case_index": True,
            "f1": [8.0, 7.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0] * 2,
            "f2": [1.0, 2.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0] * 2,
        }
    )
    current = pd.DataFrame(
        {
            "phenotype_id": ["p1", "p1", "p2", "p2"] * 2,
            "subject_id": [1, 2, 3, 4, 101, 102, 103, 104],
            "study_id": [10, 20, 30, 40, 1010, 1020, 1030, 1040],
            "split": ["train"] * 4 + ["validation"] * 4,
        }
    )
    definitions = pd.DataFrame(
        {
            "phenotype_id": ["p1", "p2"],
            "display_name": ["P1", "P2"],
            "body_system": ["A", "B"],
        }
    ).set_index("phenotype_id")
    return full, current, definitions


def _medication_row(
    subject_id: int,
    source_table: str,
    medication_class: str,
    event_time: pd.Timestamp,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict:
    return {
        "subject_id": subject_id,
        "hadm_id": subject_id * 10,
        "ed_stay_id": subject_id * 100,
        "event_time": event_time,
        "start_time": start if start is not None else event_time,
        "end_time": end,
        "medication_raw": f"raw-{source_table}-{medication_class}",
        "medication_class": medication_class,
        "source_table": source_table,
        "exposure_type": "test",
        "confidence": "high",
        "source_status": "active",
    }


def _original_medication_window_query(master: str, medications: str) -> str:
    return f"""
        SELECT
            m.subject_id,
            m.study_id,
            e.medication_class,
            BOOL_OR(
                e.source_table = 'emar'
                AND e.event_time BETWEEN
                    m.ecg_time
                    - CASE
                        WHEN e.medication_class IN (
                            'digoxin', 'antiarrhythmic_class_i_iii'
                        ) THEN INTERVAL 48 HOUR
                        ELSE INTERVAL 24 HOUR
                      END
                    AND m.ecg_time + INTERVAL 2 HOUR
            ) AS administered_near_ecg,
            BOOL_OR(
                e.source_table IN ('prescriptions', 'pharmacy')
                AND e.start_time <= m.ecg_time
                AND COALESCE(e.end_time, m.ecg_time) >= m.ecg_time
            ) AS active_order_at_ecg,
            BOOL_OR(
                e.source_table = 'pyxis'
                AND e.event_time BETWEEN
                    m.ecg_time
                    - CASE
                        WHEN e.medication_class IN (
                            'digoxin', 'antiarrhythmic_class_i_iii'
                        ) THEN INTERVAL 48 HOUR
                        ELSE INTERVAL 24 HOUR
                      END
                    AND m.ecg_time + INTERVAL 2 HOUR
            ) AS ed_dispensed_near_ecg,
            BOOL_OR(
                e.source_table = 'medrecon'
                AND e.event_time BETWEEN
                    m.ecg_time
                    - CASE
                        WHEN e.medication_class IN (
                            'digoxin', 'antiarrhythmic_class_i_iii'
                        ) THEN INTERVAL 48 HOUR
                        ELSE INTERVAL 24 HOUR
                      END
                    AND m.ecg_time + INTERVAL 2 HOUR
            ) AS home_medication_reported_near_ecg,
            COUNT(*) AS evidence_count,
            ARG_MIN(
                e.medication_raw,
                STRUCT_PACK(
                    distance_seconds := ABS(
                        DATE_DIFF('second', m.ecg_time, e.event_time)
                    ),
                    normalized_name := LOWER(COALESCE(e.medication_raw, '')),
                    exact_name := COALESCE(e.medication_raw, ''),
                    source_table := e.source_table,
                    exposure_type := e.exposure_type
                )
            ) AS nearest_medication,
            MIN(ABS(DATE_DIFF('second', m.ecg_time, e.event_time))) / 3600.0
                AS nearest_delta_hours
        FROM read_parquet('{master}') m
        JOIN read_parquet('{medications}') e
          ON m.subject_id = e.subject_id
         AND (
                e.event_time BETWEEN m.ecg_time - INTERVAL 48 HOUR
                                 AND m.ecg_time + INTERVAL 2 HOUR
             OR (
                    e.source_table IN ('prescriptions', 'pharmacy')
                AND e.start_time <= m.ecg_time
                AND COALESCE(e.end_time, m.ecg_time) >= m.ecg_time
             )
         )
        GROUP BY m.subject_id, m.study_id, e.medication_class
    """


def test_validation_bootstrap_skips_only_conservatively_null_markers() -> None:
    from vcg_biomarker.mimic.statistics import marker_bootstrap_intervals

    n = 4_000
    labels = np.tile(np.array([0, 1], dtype=np.int8), n // 2)
    subjects = np.arange(n)
    null_values = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), n // 4)
    screened = marker_bootstrap_intervals(
        subjects,
        null_values,
        labels,
        direction=1,
        iterations=200,
        seed=7,
        screen_uninformative=True,
        negligible_auc_margin=0.10,
    )
    assert screened["screened_out_as_uninformative"] is True
    assert screened["bootstrap_iterations"] == 0
    assert screened["interval_method"] == (
        "hanley_mcneil_design_effect_screen"
    )

    rng = np.random.default_rng(7)
    strong_values = labels.astype(float) + rng.normal(0.0, 0.25, n)
    retained = marker_bootstrap_intervals(
        subjects,
        strong_values,
        labels,
        direction=1,
        iterations=20,
        seed=7,
        screen_uninformative=True,
        negligible_auc_margin=0.10,
    )
    assert retained["screened_out_as_uninformative"] is False
    assert retained["bootstrap_iterations"] == 20
