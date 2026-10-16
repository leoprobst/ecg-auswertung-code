"""Synthetic production marker-path benchmark, never a scientific run.

Includes forward/reverse fits, operating thresholds and both held-out
bootstrap directions. Excludes database IO, lead models and verification-bias
models; it cannot alone establish a whole-run ETA.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from .models import ValidationSpec
from .storage import atomic_write_json, content_hash
from .validation import _evaluate_endpoint_markers
from .validation_provenance import validation_implementation_hash


def benchmark(output: Path, *, records: int = 10000, markers: int = 12,
              iterations: int = 1000, repeats: int = 3) -> dict:
    if records < 200 or markers < 1 or iterations < 0 or repeats < 1:
        raise ValueError("Invalid benchmark dimensions")
    rng = np.random.default_rng(20260908)
    names = tuple(f"synthetic_{i}" for i in range(markers))
    y = rng.binomial(1, .25, records)
    frame = pd.DataFrame({"subject_id": np.arange(records), "label": y,
        "signal_quality_score": rng.uniform(.6, 1, records),
        "reference_kind": "simulation", "reference_alignment_confirmed": False})
    for name in names:
        frame[name] = rng.normal(size=records) + .7*y
        frame.loc[rng.random(records) < .03, name] = np.nan
    train, validation = frame.iloc[:records//2].copy(), frame.iloc[records//2:].copy()
    spec = ValidationSpec(run_id="synthetic_performance_only", split_hash="synthetic_disjoint",
        endpoint_ids=("synthetic",), panel_versions={"primary_core": "synthetic"},
        development_bootstrap_iterations=iterations, min_class_inference=10, min_class_primary=20)
    process = psutil.Process()
    measurements, reference = [], None
    for workers in (1, 6):
        for repeat in range(repeats):
            before = process.cpu_times()
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                results = _evaluate_endpoint_markers(endpoint_id="synthetic", feature_ids=names,
                    panel_map={name: "primary_core" for name in names}, train=train, validation=validation,
                    specification=spec, bootstrap_iterations=iterations, worker_count=workers)
            elapsed = time.perf_counter()-start
            after = process.cpu_times()
            digest = content_hash([asdict(result) for result in results])
            if reference is None:
                reference = digest
            measurement = {"workers": workers, "repeat": repeat, "seconds": elapsed,
                "process_cpu_seconds": after.user+after.system-before.user-before.system,
                "post_task_rss_gib_not_peak": process.memory_info().rss/1024**3,
                "result_hash": digest, "identical": digest == reference}
            measurements.append(measurement)
            print(f"workers={workers} repeat={repeat+1} seconds={elapsed:.3f} identical={digest == reference}", flush=True)
            atomic_write_json(output, {"status": "running", "measurements": measurements})
    payload = {"status": "completed", "scope": "synthetic_production_marker_path_not_full_stage",
        "records": records, "markers": markers, "bootstrap_iterations": iterations,
        "repeats": repeats, "seed": 20260908, "native_threads": 1,
        "validation_implementation_hash": validation_implementation_hash(),
        "measurements": measurements, "identical": all(x["identical"] for x in measurements),
        "median_seconds": {str(w): float(np.median([x["seconds"] for x in measurements if x["workers"] == w])) for w in (1, 6)},
        "excluded": ["database_io", "verification_bias_models", "lead_models", "stage_reports", "feature_extraction"],
        "clinical_evidence": False, "locked_test_read": False, "full_run_eta_verified": False}
    atomic_write_json(output, payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=10000)
    parser.add_argument("--markers", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark(args.output, records=args.records, markers=args.markers,
              iterations=args.iterations, repeats=args.repeats)
