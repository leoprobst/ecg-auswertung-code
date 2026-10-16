"""Validate returned judgments without adopting them as clinical truth."""
from pathlib import Path
import json

import pandas as pd

from .reviewer_packet import ReviewerPacket
from .reviewer_responses import VERDICTS
from .reviewer_export import REVIEW_EXPORT_VERSION


def validate_export_packets(manifest: dict, packets: list[ReviewerPacket]) -> None:
    """Bind all supplied packets to the trusted coordinator export manifest.

    Checksums are consistency checks, not signatures or proof of authorship.
    The caller must obtain the original manifest through a trusted channel.
    """
    if (not isinstance(manifest, dict) or manifest.get("status") != "complete"
            or manifest.get("version") != REVIEW_EXPORT_VERSION
            or manifest.get("clinical_claim_allowed") is not False
            or type(manifest.get("n_reviewers")) is not int
            or manifest["n_reviewers"] < 2 or manifest["n_reviewers"] != len(packets)
            or type(manifest.get("n_cases")) is not int or manifest["n_cases"] <= 0):
        raise ValueError("Incomplete or incompatible original export manifest")
    expected = manifest.get("signal_sha256")
    if not isinstance(expected, dict) or len(expected) != manifest["n_cases"]:
        raise ValueError("Original export signal inventory missing or inconsistent")
    for packet in packets:
        actual = {case: packet.hashes[f"{case}.npz"] for case in packet.cases.review_case_id}
        if actual != expected:
            raise ValueError("Packet cases or signals do not belong to the original export")
    _validate_packet_cohort(packets)


def _validate_packet_cohort(packets: list[ReviewerPacket]) -> None:
    if not packets:
        raise ValueError("Original issued reviewer packets required")
    first = packets[0]
    expected = dict(zip(first.cases.review_case_id, first.cases.endpoint_id))
    for packet in packets[1:]:
        actual = dict(zip(packet.cases.review_case_id, packet.cases.endpoint_id))
        if actual != expected:
            raise ValueError("Reviewer packets disagree on cases or endpoint questions")
        if any(packet.hashes[f"{case}.npz"] != first.hashes[f"{case}.npz"] for case in expected):
            raise ValueError("Reviewer packets contain different signals for the same case")


def import_review_responses(packets: list[ReviewerPacket], response_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fail the batch on mismatches; report missing assignments separately.

    Slot/reviewer consistency is not proof of identity, independence, authorship
    or blinding. Coordinator verification is still required. No dataset labels
    or thresholds are written. File order never resolves conflicting answers.
    """
    _validate_packet_cohort(packets)
    by_hash = {packet.manifest_sha256: packet for packet in packets}
    slots = [int(packet.cases.reviewer_slot.iloc[0]) for packet in packets]
    if len(by_hash) != len(packets) or len(set(slots)) != len(slots):
        raise ValueError("Duplicate issued packet or reviewer slot")
    assignments = []
    for issued_packet in packets:
        for row in issued_packet.cases.itertuples():
            assignments.append(dict(review_case_id=row.review_case_id,reviewer_slot=int(row.reviewer_slot),
                endpoint_id=row.endpoint_id,packet_sha256=issued_packet.manifest_sha256,status="missing_response"))
    coverage = pd.DataFrame(assignments)
    records: list[dict] = []
    seen = set()
    slot_identities: dict[int,str] = {}
    required = {"version","review_case_id","reviewer_slot","endpoint_id","reviewer_id","reviewer_verdict",
                "recorded_at","packet_sha256","signal_sha256","clinical_claim_allowed",
                "assessment_scope","reviewer_independence"}
    for filename in response_files:
        path = Path(filename)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
            raise ValueError("Missing, redirected or oversized response file")
        with path.open("rb") as handle:
            raw = handle.read(1024**2+1)
        if len(raw) > 1024**2:
            raise ValueError("Response exceeds size limit")
        record = json.loads(raw)
        if not isinstance(record,dict) or set(record) != required:
            raise ValueError("Unexpected response schema")
        if (record["version"] != "1.0.0" or record["clinical_claim_allowed"] is not False
                or record["reviewer_verdict"] not in VERDICTS
                or record["assessment_scope"] != "human_signal_review_not_independent_clinical_confirmation"
                or record["reviewer_independence"] != "requires_coordinator_verification_not_proven_by_slot"):
            raise ValueError("Invalid response interpretation or verdict")
        reviewer = record["reviewer_id"]
        if not isinstance(reviewer,str) or not reviewer.strip():
            raise ValueError("Missing reviewer identity")
        reviewer = reviewer.strip()
        stamp = pd.to_datetime(record["recorded_at"], errors="coerce")
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("Response requires a timezone-aware timestamp")
        packet = by_hash.get(record["packet_sha256"])
        if packet is None:
            raise ValueError("Response belongs to an unissued packet")
        case, slot = record["review_case_id"], record["reviewer_slot"]
        if type(slot) is not int:
            raise ValueError("Invalid reviewer slot")
        assigned = packet.cases.loc[packet.cases.review_case_id.eq(case) & packet.cases.reviewer_slot.eq(slot)]
        if len(assigned) != 1 or record["endpoint_id"] != assigned.endpoint_id.iloc[0]:
            raise ValueError("Response does not match its issued assignment")
        if record["signal_sha256"] != packet.hashes[f"{case}.npz"]:
            raise ValueError("Response refers to a different signal")
        key = (case,slot)
        if key in seen:
            raise ValueError("Duplicate response; amendments require explicit adjudication")
        seen.add(key)
        if slot in slot_identities and slot_identities[slot] != reviewer:
            raise ValueError("Multiple reviewer identities used for one slot")
        if any(other != slot and identity.casefold() == reviewer.casefold() for other,identity in slot_identities.items()):
            raise ValueError("Same reviewer cannot constitute independent slots")
        slot_identities[slot] = reviewer
        records.append({**record,"reviewer_id":reviewer,"import_status":"provenance_consistent_not_clinically_validated"})
        mask = coverage.review_case_id.eq(case) & coverage.reviewer_slot.eq(slot)
        coverage.loc[mask,"status"] = "received"
    coverage["clinical_claim_allowed"] = False
    return pd.DataFrame(records), coverage
