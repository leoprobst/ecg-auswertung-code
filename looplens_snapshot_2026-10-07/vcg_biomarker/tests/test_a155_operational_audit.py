import json

import pyarrow as pa
import pyarrow.parquet as pq

from audit_tools.a155_input_keys import scan
from audit_tools.a155_stage_audit import manifest_findings, performance_summary


def test_join_key_audit_detects_duplicates_and_missing_keys(tmp_path):
    path = tmp_path / "keys.parquet"
    pq.write_table(pa.table({"subject_id": [1, 1, None],
                             "study_id": [4, 4, 6]}), path)
    result = scan([path])
    assert result["rows"] == 3
    assert result["unique_keys"] == 1
    assert result["duplicate_rows"] == 1
    assert result["missing_key_rows"] == 1


def test_manifest_audit_detects_changed_completed_output(tmp_path):
    work = tmp_path
    root = work / "runs" / "A155"
    root.mkdir(parents=True)
    output = root / "result.json"
    output.write_text("ok", encoding="utf-8")
    stat = output.stat()
    manifest = root / "validate.manifest.json"
    manifest.write_text(json.dumps({"stage": "validate", "status": "complete",
        "output_paths": [str(output)], "output_integrity": [{
            "path": str(output), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}]}),
        encoding="utf-8")
    assert manifest_findings(root, work, ["validate"])[0]["status"] == "ok"
    output.write_text("damaged", encoding="utf-8")
    assert manifest_findings(root, work, ["validate"])[0]["status"] == "damaged"


def test_completed_stage_requires_its_declared_manifest(tmp_path):
    root = tmp_path / "runs" / "A155"
    root.mkdir(parents=True)
    result = manifest_findings(root, tmp_path, ["thesis-tests"])
    assert result == [{"stage": "thesis-tests",
                       "path": str(root / "theses/thesis_tests.manifest.json"),
                       "status": "missing_expected_output"}]


def test_performance_summary_keeps_real_peak_and_duration(tmp_path):
    path = tmp_path / "run.jsonl"
    entries = [
        {"kind": "sample", "stage": "validate", "rss_gb": 5.5,
         "tree_read_bytes_per_second": 2**20, "tree_write_bytes_per_second": 0,
         "active_queries": [{"elapsed_seconds": 32.0}]},
        {"kind": "stage", "stage": "validate", "event": "end",
         "duration_seconds": 40.0},
    ]
    path.write_text("\n".join(json.dumps(x) for x in entries), encoding="utf-8")
    result = performance_summary(path, ["validate"])["validate"]
    assert result["max_rss_gib"] == 5.5
    assert result["max_read_mib_s"] == 1.0
    assert result["completed_seconds"] == 40.0
    assert result["max_query_seconds"] == 32.0
