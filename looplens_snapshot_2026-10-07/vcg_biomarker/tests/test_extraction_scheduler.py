"""The extraction scheduler: one pool, admission, no barriers (audit E1/E2).

Real process pools, real resume fragments and real partition files; only the
per-record computation is replaced by a deterministic stand-in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic import feature_extraction as module
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.tests import _fake_feature_worker as fake


def _config(tmp_path: Path, workers: int) -> MimicProjectConfig:
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
    config.workers = workers
    (config.labels_root / "clinical_label_evidence.parquet").write_bytes(b"exists")
    return config


class _Pause(Exception):
    pass


def _patch(monkeypatch: pytest.MonkeyPatch, study_ids: list[int]) -> None:
    import vcg_biomarker.mimic.runtime_calibration as calibration

    cohort = pd.DataFrame(
        {"study_id": study_ids, "split": ["train"] * len(study_ids)}
    )
    monkeypatch.setattr(module, "resolve_feature_panel_selection", lambda *a: {
        "panel_payload": {"panel_hash": "a" * 64},
        "feature_ids": ("marker_a", "marker_b"),
        "vcg_feature_ids": ("marker_a", "marker_b"),
        "classical_feature_ids": (),
        "sensitivity_feature_ids": (),
    })
    monkeypatch.setattr(module, "feature_implementation_hash", lambda: "b" * 64)
    monkeypatch.setattr(module, "scientific_config_hash", lambda _: "c" * 64)
    monkeypatch.setattr(module, "operational_config_hash", lambda _: "d" * 64)
    monkeypatch.setattr(module, "_load_feature_cohort", lambda *a, **k: cohort.copy())
    monkeypatch.setattr(
        module,
        "_cohort_record_payloads",
        lambda frame: {
            int(sid): json.dumps({"study_id": int(sid)})
            for sid in frame["study_id"]
        },
    )
    monkeypatch.setattr(module, "build_feature_task_config", lambda *a, **k: {})
    monkeypatch.setattr(
        calibration,
        "build_feature_timing_contract",
        lambda *a, **k: {"stage": "extract-features"},
    )
    monkeypatch.setattr(module, "_extract_record_worker", fake.record)
    monkeypatch.setattr(module, "_extract_record_batch_worker", fake.batch)
    monkeypatch.setattr(module, "_initialize_feature_worker", fake.initialize)
    monkeypatch.delenv("VCG_RESOURCE_GRANT_FILE", raising=False)


def _partitions(config: MimicProjectConfig) -> dict[str, pd.DataFrame]:
    root = config.features_root / "synthetic" / "development"
    return {
        path.name: pd.read_parquet(path)
        for path in sorted(root.glob("split=*/part-*.parquet"))
    }


def _run(config, **kwargs):
    return module.extract_mimic_features(
        config,
        module.FeatureExtractionOptions(
            panel_ids=("synthetic",),
            partition_size=40,
            micro_batch_size=7,
        ),
        **kwargs,
    )


STUDIES = list(range(990, 1090))  # 100 records, 3 partitions of 40/40/20


def test_pooled_run_matches_sequential_run_exactly(tmp_path, monkeypatch):
    _patch(monkeypatch, STUDIES)
    sequential = _config(tmp_path / "seq", workers=1)
    pooled = _config(tmp_path / "pool", workers=3)
    manifest_seq = _run(sequential)
    manifest_pool = _run(pooled)
    left, right = _partitions(sequential), _partitions(pooled)
    assert list(left) == list(right) and len(left) == 3
    for name in left:
        pd.testing.assert_frame_equal(left[name], right[name])
    combined = pd.concat(right.values(), ignore_index=True)
    assert sorted(combined["study_id"]) == STUDIES
    assert manifest_pool.row_counts["selected_ecgs"] == len(STUDIES)
    assert manifest_pool.row_counts["successful_ecgs"] == sum(
        1 for sid in STUDIES if sid % 11
    )
    assert manifest_seq.row_counts == manifest_pool.row_counts
    # No resume fragments survive a completed partition.
    root = pooled.features_root / "synthetic" / "development"
    assert not list(root.glob("split=*/.part-*.resume*"))


def test_pause_mid_partition_resumes_without_loss_or_duplicates(tmp_path, monkeypatch):
    _patch(monkeypatch, STUDIES)
    reference = _config(tmp_path / "ref", workers=1)
    _run(reference)
    config = _config(tmp_path / "pause", workers=3)
    calls = {"n": 0}

    def pausing_checkpoint(message, *, current=None, total=None):
        calls["n"] += 1
        if calls["n"] == 5:
            raise _Pause(message)

    with pytest.raises(_Pause):
        _run(config, checkpoint=pausing_checkpoint)
    root = config.features_root / "synthetic" / "development"
    # Something was durably committed before the pause ...
    committed = list(root.glob("split=*/part-*.parquet")) + list(
        root.glob("split=*/.part-*.resume.json")
    )
    assert committed
    messages: list[str] = []
    _run(config, progress=messages.append)
    left, right = _partitions(reference), _partitions(config)
    assert list(left) == list(right)
    for name in left:
        pd.testing.assert_frame_equal(left[name], right[name])
    # ... and reused on resume instead of being computed again.
    final = [m for m in messages if m.startswith("Features ")]
    assert final and "Reused=0," not in final[-1]


def test_progress_reports_every_micro_batch_and_reaches_total(tmp_path, monkeypatch):
    _patch(monkeypatch, STUDIES)
    config = _config(tmp_path, workers=2)
    seen: list[tuple[int, int]] = []

    def checkpoint(message, *, current=None, total=None):
        seen.append((current, total))

    _run(config, checkpoint=checkpoint)
    assert seen[-1] == (len(STUDIES), len(STUDIES))
    currents = [current for current, _ in seen]
    assert currents == sorted(currents)


def test_admission_pool_spawns_no_process_for_a_single_worker():
    pool = module._AdmissionPool(1, None)
    try:
        assert pool.executor is None
    finally:
        pool.close()


def test_supervised_run_admits_by_grant_and_acknowledges_in_flight(tmp_path, monkeypatch):
    import os
    import time

    from vcg_biomarker.mimic.storage import atomic_write_json

    _patch(monkeypatch, STUDIES)
    reference = _config(tmp_path / "ref", workers=1)
    _run(reference)
    config = _config(tmp_path / "managed", workers=3)
    grant = tmp_path / "grant.json"

    def refresh() -> None:
        atomic_write_json(grant, {
            "version": "1.0.0", "job_id": "t", "owner_pid": os.getpid(),
            "sampled_at": time.time(), "generation": 1, "workers": 2,
            "ceiling": 15, "unacknowledged_workers": 2,
        })

    refresh()
    monkeypatch.setenv("VCG_RESOURCE_GRANT_FILE", str(grant))

    def checkpoint(message, *, current=None, total=None):
        refresh()  # a live supervisor renews the grant

    _run(config, checkpoint=checkpoint)
    left, right = _partitions(reference), _partitions(config)
    for name in left:
        pd.testing.assert_frame_equal(left[name], right[name])
    ack = json.loads(grant.with_suffix(".ack.json").read_text(encoding="utf-8"))
    assert ack["owner_pid"] == os.getpid()
    assert 0 <= ack["resident_workers"] <= 2
