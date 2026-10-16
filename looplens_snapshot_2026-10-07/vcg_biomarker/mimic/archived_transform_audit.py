"""Descriptive agreement from explicit archived training partitions only."""
import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .storage import atomic_write_json, atomic_write_parquet, file_sha256
from .transform_agreement import agreement_table, _sample_training_partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.features.resolve()):
        raise ValueError("Audit output must be outside archived features")
    paths = _sample_training_partitions(str(args.features / "split=train" / "part-*.parquet"), 1000)
    identities = {str(path): file_sha256(path) for path in paths}
    schema = set.intersection(*(set(pq.ParquetFile(path).schema_arrow.names) for path in paths))
    prefix = "sensitivity_inverse_dower__"
    markers = sorted(name[len(prefix):] for name in schema if name.startswith(prefix) and name[len(prefix):] in schema)
    if not markers:
        raise ValueError("No paired Kors/inverse-Dower markers in selected archive")
    columns = [name for name in ("subject_id", "study_id", "extraction_status") if name in schema]
    columns += [name for marker in markers for name in (marker, prefix+marker)]
    frame = pd.concat([pq.ParquetFile(path).read(columns=columns).to_pandas() for path in paths], ignore_index=True)
    result = agreement_table(frame, markers, primary_transform="kors", comparison_transform="inverse_dower")
    if any(file_sha256(path) != identities[str(path)] for path in paths):
        raise RuntimeError("Archive changed during audit")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "archived_transform_agreement.parquet"
    atomic_write_parquet(result, target)
    atomic_write_json(args.output / "manifest.json", {
        "scope": "historical_training_features_descriptive_not_current_marker_validation",
        "sources": identities, "output_sha256": file_sha256(target),
        "marker_count": len(result), "loaded_record_count": len(frame),
        "source_modified": False, "locked_test_read": False, "clinical_release": False,
    })
    print({"markers": len(result), "records": len(frame), "status_counts": result.status.value_counts().to_dict()})


if __name__ == "__main__":
    main()
