"""Explicit human judgments in a separate write-once response directory."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os

from .reviewer_packet import ReviewerPacket

VERDICTS = ("positive", "negative", "uncertain", "unreadable")


def save_review(packet: ReviewerPacket, *, case_id: str, verdict: str,
                reviewer_id: str, response_root: Path) -> Path:
    if verdict not in VERDICTS or not reviewer_id.strip():
        raise ValueError("Explicit verdict and reviewer identifier required")
    if case_id not in set(packet.cases.review_case_id):
        raise ValueError("Case not assigned to reviewer")
    root = Path(response_root).resolve()
    if root.is_relative_to(packet.folder) or packet.folder.is_relative_to(root):
        raise ValueError("Responses must be separate from the immutable packet")
    packet.signal(case_id)  # Verify the source again at the point of judgment.
    row = packet.cases.loc[packet.cases.review_case_id.eq(case_id)].iloc[0]
    folder = root / packet.manifest_sha256
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{case_id}.json"
    record = dict(version="1.0.0", review_case_id=case_id, reviewer_slot=int(row.reviewer_slot),
        endpoint_id=str(row.endpoint_id), reviewer_id=reviewer_id.strip(), reviewer_verdict=verdict,
        recorded_at=datetime.now(timezone.utc).isoformat(), packet_sha256=packet.manifest_sha256,
        signal_sha256=packet.hashes[f"{case_id}.npz"], clinical_claim_allowed=False,
        assessment_scope="human_signal_review_not_independent_clinical_confirmation",
        reviewer_independence="requires_coordinator_verification_not_proven_by_slot")
    # Exclusive create: no silent overwrite, including concurrent double-clicks.
    with target.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    return target
