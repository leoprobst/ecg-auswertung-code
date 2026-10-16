"""Conservative use of hash-matched historical measurement-copy evidence."""
from __future__ import annotations

import json
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

@dataclass(frozen=True)
class KnownCopyEvidence:
    representatives: pd.DataFrame
    manifest_sha256: str


def load_known_copy_evidence(manifest_path: Path, expected_sha256: str | None = None) -> KnownCopyEvidence:
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_sha256 is not None and manifest_digest != expected_sha256:
        raise ValueError("Copy evidence differs from the scientifically frozen configuration")
    manifest = json.loads(manifest_bytes)
    artifact = Path(manifest["artifact"]).resolve()
    if manifest.get("status") != "complete" or not artifact.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("Historical copy evidence is incomplete or has changed")
    artifact_bytes = artifact.read_bytes()
    if hashlib.sha256(artifact_bytes).hexdigest() != manifest.get("artifact_sha256"):
        raise ValueError("Historical copy evidence is incomplete or has changed")
    known = pd.read_parquet(io.BytesIO(artifact_bytes))
    known = known.loc[known.fingerprint_status.eq("measured") & known.waveform_sha256.notna()]
    representatives = known.loc[known.exact_copy_representative.eq(True)].set_index("waveform_sha256")
    if representatives.index.duplicated().any():
        raise ValueError("Ambiguous historical representative")
    return KnownCopyEvidence(representatives, manifest_digest)


def apply_known_copy_evidence(features: pd.DataFrame, evidence: KnownCopyEvidence) -> pd.DataFrame:
    """Historical matches may exclude copies; nonmatches never prove independence."""
    if features.empty:
        return features.copy()
    representatives = evidence.representatives
    result = features.copy()
    fingerprints = result.waveform_sha256
    matched = result.fingerprint_status.eq("measured") & fingerprints.isin(representatives.index)
    expected_source = fingerprints.map(representatives.source_id)
    expected_record = fingerprints.map(representatives.record_id)
    result.loc[matched, "exact_copy_representative"] &= (
        result.loc[matched, "source_id"].eq(expected_source.loc[matched])
        & result.loc[matched, "record_id"].astype(str).eq(expected_record.loc[matched].astype(str)))
    for column in ("exact_copy_group_size", "exact_copy_source_count"):
        previous = fingerprints.map(representatives[column]).fillna(0)
        result[column] = pd.concat([result[column], previous], axis=1).max(axis=1).astype(int)
    development_seen = fingerprints.map(representatives.copy_overlaps_development_source).eq(True)
    result["copy_overlaps_development_source"] |= development_seen
    result["known_copy_evidence_matched"] = matched
    result["known_copy_manifest_sha256"] = evidence.manifest_sha256
    result["exact_copy_evidence_scope"] = "selected_extraction_plus_hash_matched_historical_scan_not_proof_of_independence"
    return result
