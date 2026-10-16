from __future__ import annotations

import json

import pytest

from vcg_biomarker.mimic.clinical_utility import ScopeFreezeAttestation
from vcg_biomarker.mimic.locking import (
    CLINICAL_UTILITY_SCOPE_ATTESTATION,
    _clinical_utility_scope_freeze,
)


def _attestation(**overrides: object) -> ScopeFreezeAttestation:
    payload: dict[str, object] = {
        "attestation_id": "scope-1",
        "candidate_id": "marker-1",
        "use_id": "use-1",
        "intended_use_sha256": "1" * 64,
        "comparator_sha256": "2" * 64,
        "thresholds_sha256": "3" * 64,
        "evidence_registry_sha256": "4" * 64,
        "custody_evidence_id": "custody-1",
        "custody_manifest_sha256": "5" * 64,
        "attested_by": "independent-custodian",
        "attested_at": "2026-08-12T00:00:00+00:00",
        "frozen_before_evaluation": True,
        "locked_test_data_attested_unread": True,
        "locked_test_outcomes_attested_unread": True,
    }
    payload.update(overrides)
    return ScopeFreezeAttestation(**payload)


def test_missing_clinical_scope_is_sealed_as_not_prespecified(tmp_path) -> None:
    result = _clinical_utility_scope_freeze(tmp_path)

    assert result["status"] == "not_prespecified"
    assert result["clinical_utility_claim_scope_prespecified"] is False


def test_valid_clinical_scope_binds_use_comparator_and_thresholds(tmp_path) -> None:
    attestation = _attestation()
    (tmp_path / CLINICAL_UTILITY_SCOPE_ATTESTATION).write_text(
        json.dumps(attestation.to_dict()),
        encoding="utf-8",
    )

    result = _clinical_utility_scope_freeze(tmp_path)

    assert result["status"] == "prespecified_and_bound"
    assert result["clinical_utility_claim_scope_prespecified"] is True
    assert result["intended_use_sha256"] == "1" * 64
    assert result["comparator_sha256"] == "2" * 64
    assert result["thresholds_sha256"] == "3" * 64
    assert result["attestation_sha256"] == attestation.attestation_sha256


def test_clinical_scope_cannot_claim_locked_outcomes_were_read(tmp_path) -> None:
    attestation = _attestation(locked_test_outcomes_attested_unread=False)
    (tmp_path / CLINICAL_UTILITY_SCOPE_ATTESTATION).write_text(
        json.dumps(attestation.to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="locked data and outcomes unread"):
        _clinical_utility_scope_freeze(tmp_path)


def test_tampered_clinical_scope_attestation_is_rejected(tmp_path) -> None:
    payload = _attestation().to_dict()
    payload["candidate_id"] = "tampered-marker"
    (tmp_path / CLINICAL_UTILITY_SCOPE_ATTESTATION).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="attestation is invalid"):
        _clinical_utility_scope_freeze(tmp_path)
