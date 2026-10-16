import pandas as pd

from vcg_biomarker.mimic.pilot_storage_profile import profile


def test_storage_profile_does_not_count_locked_or_diagnostic_parquet_as_features(tmp_path):
    results, features = tmp_path / "results", tmp_path / "features"
    results.mkdir(); features.mkdir()
    (results / "model.json").write_text("{}")
    train = features / "split=train"; train.mkdir()
    pd.DataFrame({"marker": [1., 2.]}).to_parquet(train / "part-0.parquet")
    locked = features / "split=locked_test"; locked.mkdir()
    # This would fail footer decoding if a broad glob accidentally touched it.
    (locked / "part-0.parquet").write_bytes(b"do not read")
    (features / "feature_failure_report.parquet").write_bytes(b"not a record partition")
    result = profile(results, features)
    assert result["feature_partition_rows_not_verified_unique_patients"] == 2
    assert len(result["feature_partitions"]) == 1
    assert not result["full_run_capacity_verified"]
    assert not result["locked_test_read"]
