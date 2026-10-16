"""The active manifest, not lexical ordering of archived hashes, selects data."""
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic import event_ingest
from vcg_biomarker.mimic.within_subject import _events_glob
from vcg_biomarker.tests.test_inputevent_provenance_fields import input_source


def test_actual_ingest_used_even_when_other_version_sorts_later(tmp_path, monkeypatch):
    config, spec, _ = input_source(tmp_path)
    monkeypatch.setattr(event_ingest, "_clinical_event_specs", lambda _: (spec,))
    event_ingest.ingest_clinical_event_tables(config)
    stale = config.standardized_root / "clinical_events/versions/zzzz-old/inputevents/part.parquet"
    stale.parent.mkdir(parents=True)
    pd.DataFrame(dict(orderid=[999])).to_parquet(stale, index=False)
    selected = _events_glob(config, "inputevents")
    assert "zzzz-old" not in selected
    with duckdb.connect() as con:
        assert con.execute(f"SELECT orderid FROM read_parquet('{selected}') ORDER BY orderid").fetchall() == [(123,), (124,)]


def test_unregistered_files_cannot_replace_missing_manifest(tmp_path):
    stale = tmp_path / "clinical_events/versions/schema-1.2.0/inputevents/part.parquet"
    stale.parent.mkdir(parents=True)
    pd.DataFrame(dict(orderid=[999])).to_parquet(stale, index=False)
    with pytest.raises(FileNotFoundError, match="Manifest"):
        _events_glob(SimpleNamespace(standardized_root=tmp_path), "inputevents")


@pytest.mark.parametrize("variant", ["missing", "duplicate", "escape"])
def test_registry_contract_does_not_fall_back_to_files(tmp_path, monkeypatch, variant):
    record = dict(table_id="inputevents", status="ok", relative_glob="inputevents/*.parquet")
    records = [] if variant == "missing" else [record, record] if variant == "duplicate" else [
        {**record, "relative_glob": "../outside/*.parquet"}]
    monkeypatch.setattr(event_ingest, "load_clinical_event_registry", lambda _: dict(
        _registry_path=str(tmp_path / "registry.json"), tables=records))
    with pytest.raises(RuntimeError):
        _events_glob(SimpleNamespace(), "inputevents")


def test_actual_partition_corruption_propagates(tmp_path, monkeypatch):
    config, spec, _ = input_source(tmp_path)
    monkeypatch.setattr(event_ingest, "_clinical_event_specs", lambda _: (spec,))
    event_ingest.ingest_clinical_event_tables(config)
    registry = event_ingest.load_clinical_event_registry(config)
    record = next(item for item in registry["tables"] if item["table_id"] == "inputevents")
    partition = Path(registry["_registry_path"]).parent / record["partitions"][0]["relative_path"]
    rows = pd.read_parquet(partition)
    rows["amount"] = 999.
    rows.to_parquet(partition, index=False)
    with pytest.raises(RuntimeError, match="partition integrity failed"):
        _events_glob(config, "inputevents")
