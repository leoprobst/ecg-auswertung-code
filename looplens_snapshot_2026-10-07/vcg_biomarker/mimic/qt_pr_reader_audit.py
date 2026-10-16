"""Independent-reader PR disagreement in QTDB; no 12-lead VCG substitution."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import wfdb

from .pr_error_calibration import match_reference_beats
from .storage import atomic_write_json, atomic_write_parquet

VERSION = "1.1.0"


def annotated_pr(annotation, fs):
    """QTDB boundary num codes 0=P, 1=QRS. Require complete explicit triplets."""
    waves = {"p": [], "qrs": []}
    pending = {}
    for sample, symbol, number in zip(annotation.sample, annotation.symbol, annotation.num):
        wave = {0: "p", 1: "qrs"}.get(int(number))
        if symbol == "(":
            if wave is not None:
                pending[wave] = {"onset": int(sample)}
        elif symbol == ")":
            entry = pending.pop(wave, {})
            if "peak" in entry and entry["onset"] < entry["peak"] < sample:
                waves[wave].append(SimpleNamespace(**entry, offset=int(sample)))
        else:
            peak_wave = "p" if symbol == "p" else (
                "qrs" if symbol not in {"t", "u", "[", "]", "|", "+", "~"} else None)
            if peak_wave in pending:
                pending[peak_wave]["peak"] = int(sample)
    return [(q.peak, (q.onset-p.onset)/fs*1000) for p, q in match_reference_beats(waves)]


def run_reader_audit(root, output_root):
    root, output = Path(root), Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    rows, records = [], []
    for path in sorted(root.glob("*.q2c")):
        record = path.with_suffix("")
        header = wfdb.rdheader(str(record))
        ann1, ann2 = (wfdb.rdann(str(record), reader) for reader in ("q1c", "q2c"))
        first, second = annotated_pr(ann1, header.fs), annotated_pr(ann2, header.fs)
        candidates = [[j for j, (peak, _) in enumerate(second) if abs(peak-a) <= .1*header.fs] for a, _ in first]
        uses = {j: sum(j in ids for ids in candidates) for ids in candidates for j in ids}
        matched = 0
        for (peak, pr), ids in zip(first, candidates):
            if len(ids) == 1 and uses[ids[0]] == 1:
                other = second[ids[0]][1]
                rows.append({"record_id": record.name, "r_sample_reader1": peak,
                    "pr_reader1_ms": pr, "pr_reader2_ms": other, "difference_ms": pr-other})
                matched += 1
        records.append({"record_id": record.name, "n_pr_reader1": len(first), "n_pr_reader2": len(second),
            "n_p_symbols_reader2": ann2.symbol.count("p"),
            "n_matched": matched, "n_signal_channels": header.n_sig,
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (path, record.with_suffix(".q1c"), record.with_suffix(".hea"))}})
    frame = pd.DataFrame(rows, columns=["record_id", "r_sample_reader1", "pr_reader1_ms", "pr_reader2_ms", "difference_ms"])
    summary = {"version": VERSION, "source": "https://physionet.org/content/qtdb/1.0.0/", "n_dual_reader_records": len(records),
        "n_p_symbols_reader2": sum(r["n_p_symbols_reader2"] for r in records),
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_records_with_matched_PR": frame.record_id.nunique(), "n_matched_PR": len(frame),
        "mean_signed_reader_difference_ms": float(frame.difference_ms.mean()) if len(frame) else None,
        "mean_absolute_reader_difference_ms": float(frame.difference_ms.abs().mean()) if len(frame) else None,
        "sd_reader_difference_ms": float(frame.difference_ms.std(ddof=1)) if len(frame)>1 else None,
        "approved_for_correction": False, "clinical_claim_allowed": False,
        "limits": ["two_channels_not_12_lead_VCG_geometry", "reader_errors_not_proven_independent",
                   "disagreement_variance_is_not_halved_to_invent_single_reader_error",
                   "few_records_no_independent_clinical_validation"], "records": records}
    atomic_write_parquet(frame, output / "qt_dual_reader_pr.parquet")
    atomic_write_json(output / "qt_reader_audit.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(run_reader_audit(args.database_root, args.output_root))
