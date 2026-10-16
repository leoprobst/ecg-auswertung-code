from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.jobs import JobStore
from vcg_biomarker.mimic.models import JobState
from vcg_biomarker.mimic.pilot_archive import archive_partial_feature_pilot


def _config(tmp_path: Path) -> MimicProjectConfig:
    config = MimicProjectConfig.discover(
        tmp_path / "database",
        tmp_path / "work",
    )
    config.save()
    return config


def _paused_feature_job(
    config: MimicProjectConfig,
    *,
    job_id: str,
    locked: bool = False,
):
    store = JobStore.for_project(config)
    record = store.create(
        stage="extract-features",
        project_file=config.project_file,
        job_id=job_id,
        arguments={
            "panel_ids": ["primary_core", "sentinel"],
            "include_locked_test": locked,
            "max_records": None,
        },
    )
    store.transition(record.spec.job_id, JobState.RUNNING)
    store.heartbeat(
        record.spec.job_id,
        progress_current=2,
        progress_total=10,
        progress_message="Features 2/10",
    )
    store.request_pause(record.spec.job_id)
    return store.transition(record.spec.job_id, JobState.PAUSED)


def test_partial_feature_pilot_is_atomically_archived(tmp_path: Path) -> None:
    config = _config(tmp_path)
    record = _paused_feature_job(config, job_id="partial-development")
    source = (
        config.features_root
        / "primary_core_sentinel"
        / "development"
        / "split=train"
    )
    source.mkdir(parents=True)
    parquet_path = source / "part-000000.parquet"
    pd.DataFrame(
        {"study_id": [1, 2], "extraction_status": ["ok", "ok"]}
    ).to_parquet(parquet_path, index=False)
    (source / "part-000000.json").write_text(
        json.dumps(
            {
                "part_hash": "part-hash",
                "row_count": 2,
                "parquet_size": parquet_path.stat().st_size,
                "parquet_sha256": "recorded-sha",
            }
        ),
        encoding="utf-8",
    )

    archived = archive_partial_feature_pilot(
        config,
        record.spec.job_id,
        reason="implementation changed",
    )

    assert not (config.features_root / "primary_core_sentinel" / "development").exists()
    assert Path(archived.artifact_root, "split=train", parquet_path.name).is_file()
    assert archived.completed_rows == 2
    assert archived.completed_partitions == 1
    payload = json.loads(Path(archived.manifest_path).read_text(encoding="utf-8"))
    assert payload["scientific_role"] == "historical_partial_pilot_hypothesis_only"
    assert payload["formal_analysis_allowed"] is False
    assert payload["locked_test_read"] is False
    assert JobStore.for_project(config).get(record.spec.job_id).state == JobState.CANCELLED


def test_locked_feature_job_cannot_be_archived_as_pilot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    record = _paused_feature_job(
        config,
        job_id="locked-feature-job",
        locked=True,
    )
    with pytest.raises(PermissionError, match="Locked-test"):
        archive_partial_feature_pilot(config, record.spec.job_id, reason="no")
