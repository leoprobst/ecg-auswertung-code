"""Reproducible isolated statistics benchmark; never starts a scientific run.

Windows spawn and threads consume the same read-only mmap. Startup is included
in end-to-end timings. No benchmark output is eligible as clinical evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
import multiprocessing
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from .statistics import fit_marker_model, evaluate_marker, marker_bootstrap_intervals
from .storage import atomic_write_json

_ARRAYS = None
_LIMIT = None


def _initialize(path: str):
    global _ARRAYS, _LIMIT
    _LIMIT = threadpool_limits(limits=1)
    _ARRAYS = np.load(path, mmap_mode="r", allow_pickle=False)


def _marker(task):
    index, iterations = task
    matrix = _ARRAYS
    values, labels = matrix[:, index+2], matrix[:, 1]
    n_train = len(values)//2
    fit = fit_marker_model(values[:n_train], labels[:n_train], f"synthetic_{index}")
    metrics = evaluate_marker(values[n_train:], labels[n_train:], direction=fit.direction, thresholds={})
    intervals = marker_bootstrap_intervals(matrix[n_train:, 0], values[n_train:], labels[n_train:],
        direction=fit.direction, thresholds={}, iterations=iterations, seed=20260907+index)
    return [float(metrics["roc_auc"]), *[float(intervals.get(key, np.nan)) for key in ("roc_auc_ci_low", "roc_auc_ci_high")]]


def benchmark(output: Path, *, records: int = 20000, markers: int = 12, iterations: int = 200, repeats: int = 3):
    rng = np.random.default_rng(20260907)
    labels = rng.binomial(1, .25, records)
    matrix = np.column_stack((np.arange(records), labels,
        rng.normal(size=(records, markers)) + labels[:, None]*.7))
    measurements = []
    reference = None
    with tempfile.TemporaryDirectory(prefix="vcg-statistics-benchmark-") as scratch:
        path = Path(scratch)/"input.npy"; np.save(path, matrix)
        for backend, workers in (("serial", 1), ("threads", 6), ("spawn_mmap", 6)):
            for repeat in range(repeats):
                stop = threading.Event(); samples = []
                def sample():
                    parent = psutil.Process()
                    while not stop.is_set():
                        rss = 0
                        for proc in [parent, *parent.children(recursive=True)]:
                            try: rss += proc.memory_info().rss
                            except psutil.Error: pass
                        samples.append(rss)
                        stop.wait(.1)
                sampler = threading.Thread(target=sample, daemon=True); sampler.start()
                started = time.perf_counter()
                try:
                    with threadpool_limits(limits=1):
                        _initialize(str(path))
                        tasks = [(i, iterations) for i in range(markers)]
                        executor = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_initialize, initargs=(str(path),)) if backend == "spawn_mmap" else
                                    ThreadPoolExecutor(max_workers=workers) if backend == "threads" else nullcontext())
                        with executor as pool:
                            values = list(pool.map(_marker, tasks)) if pool else list(map(_marker, tasks))
                finally:
                    stop.set(); sampler.join()
                elapsed = time.perf_counter()-started
                if reference is None: reference = np.asarray(values)
                same = bool(np.allclose(reference, values, rtol=0, atol=1e-12, equal_nan=True))
                measurements.append({"backend": backend, "workers": workers, "repeat": repeat,
                    "seconds": elapsed, "peak_tree_rss_gib": max(samples, default=0)/1024**3, "identical": same})
                print(f"{backend} repeat={repeat+1} {elapsed:.3f}s identical={same}", flush=True)
        # Explicitly release the mmap before TemporaryDirectory on Windows.
        global _ARRAYS
        _ARRAYS = None
    medians = {backend: float(np.median([x["seconds"] for x in measurements if x["backend"] == backend]))
               for backend in ("serial", "threads", "spawn_mmap")}
    speedup = medians["threads"]/medians["spawn_mmap"]
    payload = {"scope": "synthetic_statistics_hotspot_not_full_validation", "seed": 20260907,
        "records": records, "markers": markers, "bootstrap_iterations": iterations,
        "native_threads_per_worker": 1, "startup_included": True, "measurements": measurements,
        "median_seconds": medians, "spawn_speedup_vs_threads": speedup,
        "process_adoption_gate_passed": speedup >= 1.25 and all(x["identical"] and x["peak_tree_rss_gib"] <= 14 for x in measurements),
        "clinical_evidence": False, "locked_test_read": False,
        "limitation": "Not a whole-stage or full-cohort prediction; machine contention must be recorded separately."}
    atomic_write_json(output, payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark(args.output, records=args.records, repeats=args.repeats)
