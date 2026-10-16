from pathlib import Path
from types import SimpleNamespace

import pytest

from vcg_biomarker.mimic.storage import DuckDBCatalog, isolated_analysis_connection


def test_direct_analysis_honors_worker_caps_and_owns_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("VCG_JOB_MEMORY_GB", "1")
    monkeypatch.setenv("VCG_JOB_CPU_TOKENS", "1")
    monkeypatch.setenv("VCG_JOB_SPILL_GB", "2")
    config = SimpleNamespace(catalog_path=tmp_path / "catalog.duckdb",
                             duckdb_memory_limit_gb=20, workers=16)
    with isolated_analysis_connection(config) as connection:
        memory, threads, spill, scratch = connection.execute(
            "SELECT current_setting('memory_limit'), current_setting('threads'), "
            "current_setting('max_temp_directory_size'), current_setting('temp_directory')"
        ).fetchone()
        assert threads == 1
        assert spill == "2.0 GiB"
        assert memory != "18.6 GiB"  # configured 20 GB must not override worker's cap
        assert DuckDBCatalog(config.catalog_path).memory_limit_gb == 1
        scratch_path = Path(scratch).resolve()
        assert scratch_path.is_relative_to(tmp_path.resolve())
        assert scratch_path.exists()
    assert not scratch_path.exists()
    assert not config.catalog_path.exists()


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "invalid"])
def test_invalid_environment_spill_cannot_disable_default_limit(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("VCG_JOB_SPILL_GB", bad)
    assert DuckDBCatalog(tmp_path / "c.duckdb").spill_limit_gb == 16
