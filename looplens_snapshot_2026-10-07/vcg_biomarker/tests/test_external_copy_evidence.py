from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.external_copy_evidence import KnownCopyEvidence, apply_known_copy_evidence
from vcg_biomarker.mimic.external_identity_audit import duplicate_contract, waveform_fingerprint
from vcg_biomarker.mimic.external_validation import ExternalExtractionOptions, extract_external_features


def test_historical_match_excludes_copy_even_when_other_source_is_not_extracted():
    known = duplicate_contract(pd.DataFrame({"source_id": ["a", "b"], "record_id": ["1", "2"],
        "source_role": ["independent_external"]*2, "fingerprint_status": ["measured"]*2,
        "waveform_sha256": ["same"]*2}))
    reps = known.loc[known.exact_copy_representative].set_index("waveform_sha256")
    current = duplicate_contract(known.iloc[[1]])
    assert current.exact_copy_representative.iloc[0]
    result = apply_known_copy_evidence(current, KnownCopyEvidence(reps, "manifest"))
    assert not result.exact_copy_representative.iloc[0]
    assert result.known_copy_evidence_matched.iloc[0]
    assert not result.independence_proven_by_no_exact_copy.iloc[0]


def test_changed_waveform_is_not_excluded_by_old_record_name():
    known = duplicate_contract(pd.DataFrame({"source_id": ["a"], "record_id": ["1"],
        "source_role": ["independent_external"], "fingerprint_status": ["measured"], "waveform_sha256": ["old"]}))
    current = known.copy()
    current["waveform_sha256"] = "new"
    result = apply_known_copy_evidence(current, KnownCopyEvidence(known.set_index("waveform_sha256"), "manifest"))
    assert result.exact_copy_representative.iloc[0]
    assert not result.known_copy_evidence_matched.iloc[0]


def test_known_copy_never_enters_signal_quality_or_delineation(monkeypatch):
    signal = np.arange(120, dtype=float).reshape(10, 12)
    digest = waveform_fingerprint(signal, 500)
    monkeypatch.setattr("vcg_biomarker.core.external_loader.load_external_ecg_result",
        lambda *a, **kw: SimpleNamespace(signal_millivolts=signal, sampling_rate_hz=500))
    monkeypatch.setattr("vcg_biomarker.mimic.signals.assess_signal_quality",
        lambda *a, **kw: pytest.fail("Copy reached expensive processing"))
    inventory = pd.DataFrame({"source_id": ["cpsc_2018_extra"], "record_id": ["Q1"],
        "source_role": ["independent_external"], "header_path": [str(Path("Q1.hea"))],
        "sampling_rate_hz": [500], "eligible_for_diagnostic_validation": [True]})
    result = extract_external_features(inventory, ExternalExtractionOptions(marker_ids=("heart_rate",)),
        known_copy_representatives={digest: ("cpsc_2018", "A1", False)})
    assert result.extraction_status.iloc[0] == "excluded_exact_copy"


def test_modified_copy_artifact_is_rejected(tmp_path):
    import json
    from vcg_biomarker.mimic.external_copy_evidence import load_known_copy_evidence
    artifact = tmp_path / "copies.parquet"
    artifact.write_bytes(b"modified")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "complete", "artifact": str(artifact), "artifact_sha256": "0"*64}))
    with pytest.raises(ValueError, match="changed"):
        load_known_copy_evidence(manifest)


def test_incomplete_restoration_cannot_refresh_identity(tmp_path):
    import json
    from vcg_biomarker.mimic.external_identity_refresh import refresh
    previous = tmp_path / "previous.json"
    restore = tmp_path / "restore.json"
    previous.write_text(json.dumps({"status": "complete"}))
    restore.write_text(json.dumps({"status": "incomplete"}))
    with pytest.raises(ValueError, match="completed"):
        refresh(previous, restore, tmp_path / "new")


def test_copy_evidence_is_scientific_configuration_not_only_resource_metadata():
    from vcg_biomarker.mimic.config import scientific_config_payload
    from vcg_biomarker.mimic.storage import content_hash
    first = {"workers": 1, "external_identity_manifest": "audit.json", "external_identity_manifest_sha256": "a"*64}
    changed_workers = dict(first, workers=8)
    changed_evidence = dict(first, external_identity_manifest_sha256="b"*64)
    assert content_hash(scientific_config_payload(first)) == content_hash(scientific_config_payload(changed_workers))
    assert content_hash(scientific_config_payload(first)) != content_hash(scientific_config_payload(changed_evidence))


def test_frozen_digest_mismatch_is_rejected_before_reading_artifact(tmp_path):
    from vcg_biomarker.mimic.external_copy_evidence import load_known_copy_evidence
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="scientifically frozen"):
        load_known_copy_evidence(manifest, "0"*64)


@pytest.mark.parametrize("digest", [None, "short", "g"*64])
def test_config_requires_valid_scientific_copy_digest(tmp_path, digest):
    from dataclasses import replace
    from vcg_biomarker.mimic.config import MimicProjectConfig
    config = MimicProjectConfig.discover(tmp_path, tmp_path / "work")
    with pytest.raises(ValueError, match="frozen lowercase"):
        replace(config, external_identity_manifest=tmp_path / "identity.json", external_identity_manifest_sha256=digest)
