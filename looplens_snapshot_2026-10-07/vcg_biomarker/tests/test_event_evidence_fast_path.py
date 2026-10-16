"""E9: attest-event-evidence skips the shadow rebuild for direct event evidence."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vcg_biomarker.mimic import event_equivalence
from vcg_biomarker.mimic.storage import StageManifest


def _config(tmp_path, *, policy="verified_partitioned_parquet", registry_hash="h" * 64):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    output = evidence_root / "diagnoses.parquet"
    output.write_bytes(b"rows")
    StageManifest.complete(
        stage="build-evidence",
        stage_version="1.0.0",
        input_hash="a" * 64,
        config_hash="b" * 64,
        output_paths=(str(output.resolve()),),
    ).save(evidence_root / "evidence.manifest.json")
    (evidence_root / "evidence_provenance.json").write_text(
        json.dumps({
            "clinical_event_source_policy": policy,
            "clinical_event_registry": {"input_hash": registry_hash},
        }),
        encoding="utf-8",
    )
    return SimpleNamespace(evidence_root=evidence_root, work_root=tmp_path / "work"), output


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(
        event_equivalence,
        "load_clinical_event_registry",
        lambda _config: {"input_hash": "h" * 64},
    )


def _record_attest(monkeypatch):
    calls = []

    def attest(config, *, progress=None):
        calls.append(config)
        return {"status": event_equivalence.EXACT_EVENT_EVIDENCE_STATUS}

    monkeypatch.setattr(event_equivalence, "attest_clinical_event_evidence_equivalence", attest)
    return calls


def test_direct_event_evidence_needs_no_shadow_rebuild(tmp_path, monkeypatch, registry):
    config, _ = _config(tmp_path)
    calls = _record_attest(monkeypatch)
    messages = []
    result = event_equivalence.confirm_or_attest_clinical_event_evidence(
        config, progress=messages.append,
    )
    assert calls == []
    assert result["status"] == event_equivalence.DIRECT_EVENT_EVIDENCE_STATUS
    assert result["shadow_rebuild"] is False
    assert result["clinical_event_registry_input_hash"] == "h" * 64
    assert messages
    # Every consumer binds to the same record.
    assert event_equivalence.verify_clinical_event_evidence_provenance(config)["status"] == "direct_event_backed"


def test_forced_or_non_direct_evidence_runs_the_full_attestation(tmp_path, monkeypatch, registry):
    config, _ = _config(tmp_path)
    calls = _record_attest(monkeypatch)
    event_equivalence.confirm_or_attest_clinical_event_evidence(config, full_equivalence=True)
    assert len(calls) == 1

    other = tmp_path / "csv"
    other.mkdir()
    config, _ = _config(other, policy="raw_csv_fallback")
    result = event_equivalence.confirm_or_attest_clinical_event_evidence(config)
    assert len(calls) == 2
    assert result["status"] == event_equivalence.EXACT_EVENT_EVIDENCE_STATUS

    stale = tmp_path / "stale"
    stale.mkdir()
    config, _ = _config(stale, registry_hash="x" * 64)
    event_equivalence.confirm_or_attest_clinical_event_evidence(config)
    assert len(calls) == 3


def test_incomplete_canonical_evidence_is_refused(tmp_path, monkeypatch, registry):
    config, output = _config(tmp_path)
    calls = _record_attest(monkeypatch)
    output.unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        event_equivalence.confirm_or_attest_clinical_event_evidence(config)
    assert calls == []


def test_stage_service_routes_force_to_full_equivalence(monkeypatch):
    from vcg_biomarker.mimic import services

    seen = []

    def fake(config, *, full_equivalence, progress):
        seen.append(full_equivalence)
        return {"status": "ok"}

    monkeypatch.setattr(services, "confirm_or_attest_clinical_event_evidence", fake)
    services.run_service_stage(SimpleNamespace(), "attest-event-evidence", {})
    services.run_service_stage(SimpleNamespace(), "attest-event-evidence", {"force": True})
    assert seen == [False, True]
