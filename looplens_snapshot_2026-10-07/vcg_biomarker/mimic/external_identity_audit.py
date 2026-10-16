"""Exact measurement-copy audit, not patient identification or fuzzy linkage."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import atomic_write_json, atomic_write_parquet, file_sha256

IDENTITY_AUDIT_VERSION = "1.0.0"


def measurement_representative_mask(frame: pd.DataFrame) -> pd.Series:
    """Keep unmeasured failures, exclude identified copies; not a patient mask."""
    if "exact_copy_representative" not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)
    return ((frame.exact_copy_representative.eq(True) | frame.fingerprint_status.ne("measured"))
            & ~frame.copy_overlaps_development_source.eq(True))


def waveform_fingerprint(signal: np.ndarray, fs: float) -> str:
    """Hash canonical ordered 12-lead mV samples, before any filtering.

    No rounding/resampling: similar tracings or repeat ECGs are not duplicates.
    Signed zeros are normalized; dtype/endianness/layout do not affect the hash.
    """
    data = np.asarray(signal)
    if data.ndim != 2 or data.shape[1] != 12 or not len(data):
        raise ValueError("Expected nonempty ordered twelve-lead signal")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling rate")
    digest = hashlib.sha256(b"vcg-exact-12lead-mv-v1")
    digest.update(np.asarray([float(fs)], dtype="<f8").tobytes())
    digest.update(np.asarray(data.shape, dtype="<i8").tobytes())
    for start in range(0, len(data), 8192):
        block = np.array(data[start:start+8192], dtype="<f8", order="C", copy=True)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite samples cannot establish signal identity")
        block[block == 0] = 0.0
        digest.update(block.tobytes())
    return digest.hexdigest()


def duplicate_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag duplicate measurement groups without changing diagnoses/patient IDs."""
    result = frame.copy()
    if result.duplicated(["source_id", "record_id"]).any():
        raise ValueError("Ambiguous source/record identity")
    result["exact_copy_group_size"] = 0
    result["exact_copy_source_count"] = 0
    result["copy_overlaps_development_source"] = False
    result["exact_copy_representative"] = False
    measured = result["fingerprint_status"].eq("measured") & result["waveform_sha256"].notna()
    valid = result.loc[measured]
    groups = valid.groupby("waveform_sha256", sort=False)
    result.loc[measured, "exact_copy_group_size"] = groups.record_id.transform("size")
    result.loc[measured, "exact_copy_source_count"] = groups.source_id.transform("nunique")
    overlap = valid.source_role.eq("development_reused_not_independent").groupby(valid.waveform_sha256).transform("any")
    result.loc[measured, "copy_overlaps_development_source"] = overlap
    # Reference/label/marker-blind choice; no outcome-dependent winner.
    representatives = valid.sort_values(["source_id", "record_id"], kind="stable").drop_duplicates("waveform_sha256")
    result.loc[representatives.index, "exact_copy_representative"] = True
    result["patient_identity_established_by_fingerprint"] = False
    result["independence_proven_by_no_exact_copy"] = False
    return result


def audit_collection(root: Path, output: Path, *, workers: int = 4,
                     limit_per_source: int | None = None) -> dict:
    from ..core.external_loader import load_external_ecg_result
    from .external_cohorts import inventory_external_collection
    if not 1 <= workers <= 4:
        raise ValueError("Identity audit uses one to four bounded workers")
    inventory = inventory_external_collection(root, limit_per_source=limit_per_source)
    if inventory.empty:
        raise ValueError("No external records available")
    records = inventory[["source_id", "source_role", "record_id", "header_path"]].to_dict("records")
    output.mkdir(parents=True, exist_ok=True)

    def inspect(record):
        row = dict(record, fingerprint_status="failed", waveform_sha256=None,
                   error_type="", error_message="")
        try:
            header = Path(record["header_path"])
            row["header_sha256"] = file_sha256(header)
            loaded = load_external_ecg_result(header.with_suffix(""))
            row["waveform_sha256"] = waveform_fingerprint(loaded.signal_millivolts, loaded.sampling_rate_hz)
            row["fingerprint_status"] = "measured"
        except Exception as exc:
            # Keep structured failed records in the denominator, never a false identity.
            row.update(error_type=type(exc).__name__, error_message=str(exc))
        return row

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(records), 128):
            rows.extend(executor.map(inspect, records[start:start+128]))
            print(f"identity_audit {len(rows)}/{len(records)}", flush=True)
    table = duplicate_contract(pd.DataFrame(rows))
    path = output / "external_signal_identity.parquet"
    atomic_write_parquet(table, path)
    duplicated = table.loc[table.exact_copy_group_size.gt(1)]
    summary = {
        "version": IDENTITY_AUDIT_VERSION, "status": "complete",
        "collection_root": str(root.resolve()), "records": len(table),
        "fingerprinted": int(table.fingerprint_status.eq("measured").sum()),
        "failed": int(table.fingerprint_status.ne("measured").sum()),
        "duplicate_measurement_groups": int(duplicated.waveform_sha256.nunique()),
        "records_in_duplicate_groups": len(duplicated),
        "redundant_measurements": int((duplicated.exact_copy_representative.eq(False)).sum()),
        "cross_source_duplicate_groups": int(duplicated.loc[duplicated.exact_copy_source_count.gt(1), "waveform_sha256"].nunique()),
        "limit_per_source": limit_per_source,
        "complete_collection": limit_per_source is None,
        "scope": "exact_decoded_12lead_samples_and_fs_only_not_fuzzy_copies_or_patient_identity",
        "source_summary": table.groupby("source_id").agg(records=("record_id", "size"),
            exact_copy_records=("exact_copy_group_size", lambda v: int(v.gt(1).sum()))).reset_index().to_dict("records"),
        "artifact": str(path.resolve()), "artifact_sha256": file_sha256(path),
        "clinical_claim_allowed": False, "locked_test_read": False,
    }
    atomic_write_json(output / "external_signal_identity.manifest.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-source", type=int)
    args = parser.parse_args()
    audit_collection(args.root, args.output, workers=args.workers, limit_per_source=args.limit_per_source)
