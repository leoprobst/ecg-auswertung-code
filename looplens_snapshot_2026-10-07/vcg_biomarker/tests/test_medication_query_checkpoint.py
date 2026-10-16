import json

import duckdb
import pytest

from vcg_biomarker.mimic import evidence


def test_query_checkpoint_resume_sql_change_corruption_and_shadow(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.json"
    output = tmp_path / "medication.parquet"
    checkpoint = {}
    original = evidence._copy_query_atomic
    calls = []

    def counted(connection, query, path):
        calls.append(query)
        return original(connection, query, path)

    monkeypatch.setattr(evidence, "_copy_query_atomic", counted)
    with duckdb.connect() as connection:
        def run(query="SELECT 1 AS value", context=True):
            return evidence._copy_checkpointed_medication_query(
                connection, query, output, threads=1,
                checkpoint_context=(checkpoint_path, checkpoint) if context else None,
            )
        assert run() == 1
        assert len(calls) == 1
        # Recreate in-memory state exactly as after an interrupted process.
        checkpoint = json.loads(checkpoint_path.read_text())
        assert run() == 1
        assert len(calls) == 1
        output.write_bytes(b"corrupt")
        assert run() == 1
        assert len(calls) == 2
        assert run("SELECT 2 AS value") == 1
        assert len(calls) == 3
        assert run("SELECT 2 AS value", context=False) == 1
        assert len(calls) == 4


def test_failed_copy_does_not_publish_query_checkpoint(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic COPY failure")
    monkeypatch.setattr(evidence, "_copy_query_atomic", fail)
    checkpoint = {}
    checkpoint_path = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="synthetic COPY failure"):
        evidence._copy_checkpointed_medication_query(
            None, "SELECT 1", tmp_path / "out.parquet",
            checkpoint_context=(checkpoint_path, checkpoint),
        )
    assert checkpoint == {}
    assert not checkpoint_path.exists()


def test_medication_builder_persists_both_queries_before_pause_callback(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pandas as pd

    config = SimpleNamespace(mimic_iv_root=tmp_path, mimic_ed_root=tmp_path,
                             evidence_root=tmp_path, master_root=tmp_path, workers=1)
    monkeypatch.setattr(evidence, "clinical_event_relation", lambda *args: None)
    monkeypatch.setattr(evidence, "clinical_event_quarantine_relation", lambda *args: None)
    calls = []

    def copy(connection, query, path):
        calls.append(path.name)
        pd.DataFrame({"synthetic": [1]}).to_parquet(path, index=False)
        return 1

    monkeypatch.setattr(evidence, "_copy_query_atomic", copy)
    monkeypatch.setattr(evidence, "_copy_partitioned_query_atomic",
                        lambda connection, query, path, **kwargs: copy(connection, "", path))
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = evidence._load_evidence_checkpoint(checkpoint_path, input_hash="input", config_hash="config")

    def pause(message):
        if message == "Verknuepfe Medikamentenfenster mit ECGs":
            assert len(json.loads(checkpoint_path.read_text())["medication_queries"]) == 2
            raise evidence.JobPauseRequested("synthetic pause")

    with duckdb.connect() as connection:
        with pytest.raises(evidence.JobPauseRequested):
            evidence._build_medication_evidence(connection, config, pause,
                                                query_checkpoint=(checkpoint_path, checkpoint))
        resumed = evidence._load_evidence_checkpoint(checkpoint_path, input_hash="input", config_hash="config")
        artifacts = evidence._build_medication_evidence(connection, config, None,
                                                        query_checkpoint=(checkpoint_path, resumed))
    assert calls.count("medication_evidence.parquet") == 1
    assert calls.count("medication_order_administration_context.parquet") == 1
    assert len(artifacts) == 6
    assert len(calls) == 6
