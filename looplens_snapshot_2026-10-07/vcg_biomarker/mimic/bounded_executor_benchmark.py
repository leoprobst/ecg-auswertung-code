"""Synthetic scheduler/row-materialization benchmark, not a clinical pipeline ETA."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics
import time
import tracemalloc

import pandas as pd

from .bounded_executor import bounded_ordered_map
from .storage import atomic_write_json, content_hash


def benchmark(*, records: int = 20000, workers: int = 4, repeats: int = 3) -> dict[str, object]:
    if min(records, workers, repeats) < 1:
        raise ValueError("Positive records, workers and repeats required")
    frame = pd.DataFrame({f"field_{i}": [f"synthetic_{i}_{j}" for j in range(records)] for i in range(12)})
    trials = []
    def work(row):
        return len(row["field_0"]) + len(row["field_11"])
    # Alternate execution order to reduce systematic warm-up/order bias.
    for repetition in range(repeats):
        for strategy in (["eager", "bounded"] if repetition % 2 == 0 else ["bounded", "eager"]):
            tracemalloc.start()
            started = time.perf_counter()
            if strategy == "eager":
                values = [row for _, row in frame.iterrows()]
            else:
                columns = tuple(frame.columns)
                values = (dict(zip(columns, row)) for row in frame.itertuples(index=False, name=None))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(work, values) if strategy == "eager" else
                               bounded_ordered_map(pool, work, values, max_pending=4*workers))
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            trials.append({"strategy": strategy, "repetition": repetition, "seconds": elapsed,
                           "python_traced_peak_bytes": peak, "result_hash": content_hash(results)})
            del results, values
    return {
        "schema_version": "1.0.0", "scope": "synthetic_scheduler_and_row_materialization_not_feature_extraction",
        "records": records, "workers": workers, "repeats": repeats, "bounded_max_pending": 4*workers,
        "input_frame_allocation_excluded": True, "native_memory_measured": False,
        "identical_results": len({row["result_hash"] for row in trials}) == 1,
        "trials": trials,
        "median_by_strategy": {
            strategy: {key: statistics.median([row[key] for row in trials if row["strategy"] == strategy])
                       for key in ("seconds", "python_traced_peak_bytes")}
            for strategy in ("eager", "bounded")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    result = benchmark(records=args.records, workers=args.workers, repeats=args.repeats)
    atomic_write_json(args.output, result)
    print(result["median_by_strategy"])
    print("identical_results", result["identical_results"])


if __name__ == "__main__":
    main()
