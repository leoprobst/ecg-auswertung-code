"""Refresh exactly the recordings changed by a verified missing-file restore."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .external_identity_audit import duplicate_contract, waveform_fingerprint
from .external_cohorts import parse_external_header, source_by_directory
from .storage import atomic_write_json, atomic_write_parquet, file_sha256


def refresh(previous_manifest: Path, restoration_manifest: Path, output: Path) -> dict:
    from ..core.external_loader import load_external_ecg_result
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    restoration = json.loads(restoration_manifest.read_text(encoding="utf-8"))
    if previous.get("status") != "complete" or restoration.get("status") != "complete":
        raise ValueError("Refresh requires completed scan and verified restoration")
    source_path = Path(previous["artifact"])
    if file_sha256(source_path) != previous["artifact_sha256"]:
        raise ValueError("Previous identity artifact has changed")
    root = Path(previous["collection_root"]).resolve()
    frame = pd.read_parquet(source_path)
    touched = set()
    for item in restoration["results"]:
        if item["status"] != "restored_verified":
            raise ValueError("Restoration contains unverified files")
        target = (root.parent / item["relative_path"]).resolve()
        if not target.is_relative_to(root) or file_sha256(target) != item["sha256"]:
            raise ValueError("Restored input escaped root or changed")
        touched.add(target.with_suffix(".hea"))
    rows = []
    keys = set()
    for header in sorted(touched):
        relative = header.relative_to(root)
        source = source_by_directory(relative.parts[0])
        if source is None:
            raise ValueError("Unknown restored ECG source")
        record = parse_external_header(header, source.source_id)
        if record is None:
            raise ValueError("Restored header is not parseable")
        keys.add((source.source_id, record.record_id))
        row = dict(source_id=source.source_id, source_role=source.role, record_id=record.record_id,
                   header_path=str(header), header_sha256=file_sha256(header),
                   fingerprint_status="failed", waveform_sha256=None, error_type="", error_message="")
        try:
            loaded = load_external_ecg_result(header.with_suffix(""))
            row["waveform_sha256"] = waveform_fingerprint(loaded.signal_millivolts, loaded.sampling_rate_hz)
            row["fingerprint_status"] = "measured"
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error_message=str(exc))
        rows.append(row)
    retain = ~pd.MultiIndex.from_frame(frame[["source_id", "record_id"]]).isin(keys)
    updated = duplicate_contract(pd.concat([frame.loc[retain], pd.DataFrame(rows)], ignore_index=True))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "external_signal_identity.parquet"
    if path.resolve() == source_path.resolve():
        raise ValueError("Never overwrite the original identity audit")
    atomic_write_parquet(updated, path)
    copies = updated.loc[updated.exact_copy_group_size.gt(1)]
    result = {"status": "complete", "collection_root": str(root), "records": len(updated),
        "fingerprinted": int(updated.fingerprint_status.eq("measured").sum()),
        "failed": int(updated.fingerprint_status.ne("measured").sum()),
        "recordings_reloaded": len(rows), "prior_recordings_retained": int(retain.sum()),
        "prior_measurements_retained_under_additions_only_restoration": True,
        "duplicate_measurement_groups": int(copies.waveform_sha256.nunique()),
        "records_in_duplicate_groups": len(copies),
        "redundant_measurements": int(copies.exact_copy_representative.eq(False).sum()),
        "cross_source_duplicate_groups": int(copies.loc[copies.exact_copy_source_count.gt(1), "waveform_sha256"].nunique()),
        "previous_manifest_sha256": file_sha256(previous_manifest),
        "restoration_manifest_sha256": file_sha256(restoration_manifest),
        "artifact": str(path.resolve()), "artifact_sha256": file_sha256(path),
        "clinical_claim_allowed": False, "locked_test_read": False}
    atomic_write_json(output / "external_signal_identity.manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--restoration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refresh(args.previous_manifest, args.restoration_manifest, args.output)))
