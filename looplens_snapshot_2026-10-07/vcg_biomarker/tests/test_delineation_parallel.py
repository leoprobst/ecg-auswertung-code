"""E7: record-parallel delineation benchmarks produce the serial result."""
from __future__ import annotations

from pathlib import Path

import pytest

from vcg_biomarker.core.delineation import benchmark_suite as suite

DATASETS = Path(r"E:\VCG\mimic_validation\resources\delineation\datasets")


@pytest.mark.skipif(not DATASETS.is_dir(), reason="benchmark datasets not available locally")
def test_staffiii_parallel_records_equal_serial(tmp_path):
    root = suite.resolve_benchmark_root(DATASETS, "staffiii")
    if root is None:
        pytest.skip("STAFF III root not recognised")
    serial = suite.run_staffiii_event_stability_benchmark(root, max_records=4)
    parallel = suite.run_staffiii_event_stability_benchmark(
        root, max_records=4, record_workers=4,
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert serial["windows"] == parallel["windows"]
    assert serial["quarantine"] == parallel["quarantine"]
    assert serial["phase_summary"] == parallel["phase_summary"]
    # Checkpoints written by workers are reused without recomputation.
    again = suite.run_staffiii_event_stability_benchmark(
        root, max_records=4, record_workers=4,
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert again["windows"] == serial["windows"]


def test_record_worker_admission_never_exceeds_request(monkeypatch):
    import vcg_biomarker.mimic.resources as resources

    monkeypatch.setattr(resources, "available_memory_gb", lambda: 100.0)
    assert suite._admitted_record_workers(1) == 1
    assert suite._admitted_record_workers(6) == 6
    monkeypatch.setattr(resources, "available_memory_gb", lambda: 6.5)
    assert suite._admitted_record_workers(6) == 2
    monkeypatch.setattr(resources, "available_memory_gb", lambda: None)
    assert suite._admitted_record_workers(6) == 1
    with pytest.raises(ValueError):
        suite._admitted_record_workers(0)


@pytest.mark.skipif(not DATASETS.is_dir(), reason="benchmark datasets not available locally")
def test_qtdb_parallel_records_equal_serial_and_adapt_to_memory(tmp_path, monkeypatch):
    root = suite.resolve_benchmark_root(DATASETS, "qtdb")
    if root is None:
        pytest.skip("QTDB root not recognised")
    serial = suite.run_qtdb_boundary_benchmark(root, max_records=3)
    monkeypatch.setattr(suite, "_available_gb", lambda _psutil: 64.0)
    parallel = suite.run_qtdb_boundary_benchmark(root, max_records=3, record_workers=3)
    assert parallel["record_workers"] == 3
    for key in ("records_completed", "records_quarantined", "windows_processed", "waves"):
        assert serial[key] == parallel[key], key
    # Tight memory: fewer workers instead of a failed stage.
    monkeypatch.setattr(suite, "_available_gb", lambda _psutil: 7.0)
    tight = suite.run_qtdb_boundary_benchmark(root, max_records=3, record_workers=3)
    assert tight["record_workers"] == 1
    assert tight["waves"] == serial["waves"]
