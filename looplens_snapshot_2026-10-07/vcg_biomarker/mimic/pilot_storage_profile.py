"""Read-only pilot file/Parquet-footer accounting; never a whole-run size extrapolation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from .storage import atomic_write_json


def profile(result_root: Path, feature_root: Path) -> dict:
    result_root, feature_root = result_root.resolve(), feature_root.resolve()
    if not result_root.is_dir() or not feature_root.is_dir():
        raise ValueError("Both explicit pilot roots must exist")
    categories = defaultdict(lambda: {"files": 0, "logical_bytes": 0})
    group_sizes = defaultdict(int)
    files = []
    for path in sorted(result_root.rglob("*")):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(result_root):
            raise ValueError("Pilot file escapes the selected root")
        relative = path.relative_to(result_root)
        category = relative.parts[0] if len(relative.parts) > 1 else "root_results"
        size = path.stat().st_size
        categories[category]["files"] += 1
        categories[category]["logical_bytes"] += size
        if category == "validation_checkpoints":
            group_sizes[path.name] += size
        files.append({"path": relative.as_posix(), "logical_bytes": size})
    partitions = []
    # Never use a broad recursive glob that might enter a held-out split.
    for split in ("train", "validation"):
        for path in sorted(feature_root.glob(f"split={split}/part-*.parquet")):
            if not path.resolve().is_relative_to(feature_root):
                raise ValueError("Feature partition escapes the selected pilot root")
            metadata = pq.read_metadata(path)
            partitions.append({"path": path.relative_to(feature_root).as_posix(), "split": split,
                "logical_bytes": path.stat().st_size, "rows": metadata.num_rows, "columns": metadata.num_columns,
                "parquet_encoded_uncompressed_bytes_not_dataframe_ram": sum(metadata.row_group(i).total_byte_size
                                                                            for i in range(metadata.num_row_groups))})
    return {"result_root": str(result_root), "feature_root": str(feature_root),
        "scope": "historical_pilot_storage_baseline_not_full_run_capacity_release",
        "file_sizes_are_logical_not_allocated_disk_bytes": True,
        "source_contents_hashed": False, "waveforms_or_parquet_rows_read": False,
        "locked_test_read": False, "result_files": len(files),
        "result_logical_bytes": sum(row["logical_bytes"] for row in files),
        "categories": dict(categories), "checkpoint_bytes_by_filename": dict(sorted(group_sizes.items())),
        "largest_result_files": sorted(files, key=lambda row: row["logical_bytes"], reverse=True)[:20],
        "feature_partitions": partitions,
        "feature_partition_logical_bytes": sum(row["logical_bytes"] for row in partitions),
        "feature_partition_rows_not_verified_unique_patients": sum(row["rows"] for row in partitions),
        "volume_free_bytes_observed": shutil.disk_usage(result_root).free,
        "full_run_capacity_verified": False,
        "cannot_scale_whole_result_folder_by_record_count": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(root.resolve()) for root in (args.results, args.features)):
        raise ValueError("Store the audit separately from original pilot inputs")
    report = profile(args.results, args.features)
    atomic_write_json(args.output, report)
    print({key: report[key] for key in ("result_logical_bytes", "feature_partition_logical_bytes",
                                      "feature_partition_rows_not_verified_unique_patients", "volume_free_bytes_observed")})


if __name__ == "__main__":
    main()
