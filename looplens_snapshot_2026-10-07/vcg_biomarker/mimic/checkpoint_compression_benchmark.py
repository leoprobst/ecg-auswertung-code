"""Benchmark one real pilot checkpoint without modifying the source artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

from .checkpoint_codec import write_group, read_group
from .storage import atomic_write_json, content_hash, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("Positive repeats required")
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Never overwrite the source checkpoint")
    before = file_sha256(args.source)
    payload = json.loads(args.source.read_bytes())
    semantic_hash = content_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trials = []
    with tempfile.TemporaryDirectory(prefix="codec-benchmark-", dir=args.output.parent) as directory:
        root = Path(directory).resolve()
        if not root.is_relative_to(args.output.parent.resolve()):
            raise RuntimeError("Temporary benchmark directory escaped output root")
        for repetition in range(args.repeats):
            for mode in (["pretty_json", "compact_gzip1"] if repetition % 2 == 0 else ["compact_gzip1", "pretty_json"]):
                target = root / ("sample.json" if mode == "pretty_json" else "sample.json.gz")
                started = time.perf_counter()
                if mode == "pretty_json":
                    atomic_write_json(target, payload)
                else:
                    metadata = write_group(target, payload)
                written = time.perf_counter()
                if mode == "pretty_json":
                    decoded = json.loads(target.read_bytes())
                else:
                    decoded = read_group(target, sha256=metadata["sha256"], uncompressed_bytes=metadata["uncompressed_bytes"])
                read = time.perf_counter()
                trials.append({"strategy": mode, "repeat": repetition, "write_seconds": written-started,
                    "read_seconds": read-written, "roundtrip_seconds": read-started, "bytes": target.stat().st_size,
                    "semantically_identical": content_hash(decoded) == semantic_hash})
                del decoded
    if file_sha256(args.source) != before:
        raise RuntimeError("Source checkpoint changed during measurement")
    report = {"source_sha256": before, "source_path": str(args.source.resolve()),
        "source_bytes": args.source.stat().st_size, "repeats": args.repeats,
        "scope": "single_real_checkpoint_codec_not_whole_validation", "source_modified": False,
        "trials": trials, "all_results_identical": all(row["semantically_identical"] for row in trials),
        "median_by_strategy": {mode: {key: statistics.median([row[key] for row in trials if row["strategy"] == mode])
            for key in ("write_seconds", "read_seconds", "roundtrip_seconds", "bytes")}
            for mode in ("pretty_json", "compact_gzip1")}}
    atomic_write_json(args.output, report)
    print(report["median_by_strategy"])
    print("all_results_identical", report["all_results_identical"])


if __name__ == "__main__":
    main()
