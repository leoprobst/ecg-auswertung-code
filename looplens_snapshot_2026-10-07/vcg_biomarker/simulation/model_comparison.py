"""Paired model discrepancy, not surrogate error, patient accuracy or agreement truth."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .learning import OUTPUTS, digest


def compare_campaigns(root, manifest):
    reference = manifest["comparison_reference"]
    old_root = Path(reference["root"])
    parent = json.loads((old_root/"manifest.json").read_text(encoding="utf-8"))
    unsigned = dict(parent)
    if unsigned.pop("job_sha256") != reference["job_sha256"] or digest(unsigned) != reference["job_sha256"]:
        raise ValueError("Reference manifest changed")
    pairs, excluded, inputs = [], [], {}
    for case in manifest["cases"]:
        name = f"{case['id']:05d}.json"
        paths = (old_root/"cases"/name, Path(root)/"cases"/name)
        rows = []
        for path in paths:
            raw = path.read_bytes()
            inputs[str(path)] = hashlib.sha256(raw).hexdigest()
            rows.append(json.loads(raw))
        if inputs[str(paths[0])] != reference["case_sha256"][str(case["id"])]:
            raise ValueError("Reference result changed since comparison freeze")
        for row, expected_hash in zip(rows, (reference["job_sha256"], manifest["job_sha256"])):
            if row["job_sha256"] != expected_hash or any(row[k] != case[k] for k in ("id", "candidate_id", "cycle_ms", "scales", "partition")):
                raise ValueError("Model comparison is not paired")
        if any(r["status"] != "computed" or not r.get("periodicity_check_passed") for r in rows):
            excluded.append({"case_id": case["id"], "status": [r["status"] for r in rows],
                             "periodic": [r.get("periodicity_check_passed") for r in rows]})
            continue
        pairs.append({"case_id": case["id"], "cycle_ms": case["cycle_ms"],
                      "delta_new_minus_reference": [rows[1]["biomarkers"][o]-rows[0]["biomarkers"][o] for o in OUTPUTS]})
    summary = []
    for cycle in manifest["cycles_ms"]:
        data = np.array([r["delta_new_minus_reference"] for r in pairs if r["cycle_ms"] == cycle])
        summary.append({"cycle_ms": cycle, "n": len(data), "status": "descriptive_model_discrepancy" if len(data) >= 2 else "insufficient_pairs",
                        "mean_difference": dict(zip(OUTPUTS, data.mean(0).tolist())) if len(data) >= 2 else None,
                        "mean_absolute_difference": dict(zip(OUTPUTS, np.abs(data).mean(0).tolist())) if len(data) >= 2 else None})
    return {"schema": "paired-cell-model-discrepancy/1", "clinical_use": False,
            "model": manifest["model"], "reference_model": parent["model"], "source_hashes": inputs,
            "summary": summary, "excluded": excluded, "pairs": pairs,
            "limitations": ["Neither mathematical model is clinical ground truth.",
                            "Same relative multipliers need not represent the same absolute current density.",
                            "Failure to reach periodicity is not discarded silently and can bias complete-pair comparisons.",
                            "This comparison does not quantify anatomical or torso model discrepancy."]}
