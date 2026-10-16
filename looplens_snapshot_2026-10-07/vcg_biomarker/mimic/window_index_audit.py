"""Read-only diagnostic of index drift across development time-window profiles."""
from __future__ import annotations

import argparse
from pathlib import Path

from .storage import DuckDBCatalog, atomic_write_json, file_sha256


def audit(path: Path) -> dict:
    before = file_sha256(path)
    with DuckDBCatalog(path.parent / ".window-audit.duckdb", memory_limit_gb=2,
                       threads=2).connect(isolated=True) as connection:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='2GB'")
        connection.execute("""CREATE TEMP TABLE selected AS SELECT subject_id, study_id, split,
            label_id, window_profile, window_rule_version FROM read_parquet(?)
            WHERE split IN ('train', 'validation')""", [str(path)])
        connection.execute("""CREATE TEMP TABLE pairs AS SELECT subject_id,label_id,split,
            COUNT(DISTINCT study_id) AS n_indices, COUNT(DISTINCT window_profile) AS n_profiles
            FROM selected GROUP BY subject_id,label_id,split""")
        rows = connection.execute("""SELECT label_id, split, COUNT(*) AS patient_endpoint_pairs,
            COUNT(*) FILTER (WHERE n_profiles > 1) AS pairs_with_multiple_profiles,
            COUNT(*) FILTER (WHERE n_indices > 1) AS pairs_with_index_drift
            FROM pairs GROUP BY label_id,split ORDER BY label_id,split""").fetchdf().to_dict("records")
        total = connection.execute("""SELECT COUNT(*), COUNT(*) FILTER (WHERE n_profiles > 1),
            COUNT(*) FILTER (WHERE n_indices > 1) FROM pairs""").fetchone()
        duplicate_profiles = connection.execute("""SELECT COUNT(*) FROM (
            SELECT subject_id,label_id,window_profile,split FROM selected
            GROUP BY subject_id,label_id,window_profile,split HAVING COUNT(*) > 1)""").fetchone()[0]
        split_conflicts = connection.execute("""SELECT COUNT(*) FROM (
            SELECT subject_id FROM selected GROUP BY subject_id HAVING COUNT(DISTINCT split) > 1)""").fetchone()[0]
        versions = [row[0] for row in connection.execute("SELECT DISTINCT window_rule_version FROM selected ORDER BY 1").fetchall()]
    if file_sha256(path) != before:
        raise RuntimeError("Window cohort changed during read-only audit")
    return {"scope": "current_materialized_development_window_cohorts_not_a_new_marker_analysis",
        "source": str(path.resolve()), "source_sha256": before, "window_rule_versions": versions,
        "patient_endpoint_pairs": total[0], "pairs_with_multiple_profiles": total[1],
        "pairs_with_index_drift": total[2], "duplicate_patient_endpoint_profile_groups": duplicate_profiles,
        "patients_with_split_conflicts": split_conflicts, "by_endpoint_and_split": rows,
        "source_modified": False, "locked_test_read": False,
        "clinical_effect_of_index_drift_quantified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cohorts.resolve() == args.output.resolve():
        raise ValueError("Audit output must not overwrite source cohorts")
    result = audit(args.cohorts)
    atomic_write_json(args.output, result)
    print({key: result[key] for key in ("patient_endpoint_pairs", "pairs_with_multiple_profiles", "pairs_with_index_drift")})


if __name__ == "__main__":
    main()
