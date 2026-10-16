from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vcg_biomarker.mimic.jobs import JobPauseRequested
from vcg_biomarker.mimic.storage import feature_parquet_clause, file_sha256
from vcg_biomarker.mimic.validation import _stream_validation_cache_split, _validation_cache_schema


def _sources(root: Path, *, wide=False):
    root.mkdir(parents=True)
    for i in (2, 0, 1):
        frame = pd.DataFrame({
            "subject_id": np.arange(i * 16, (i + 1) * 16, dtype=np.int64),
            "study_id": np.arange(i * 16, (i + 1) * 16, dtype=np.int64) + 100,
            "split": ["train"] * 16,
            "float": [None, 1.5, -2.25, float('inf')] * 4,
            "boolean": pd.array([True, False, None, True] * 4, dtype="boolean"),
            "text": ["ä", None, "", "abc"] * 4,
            "date": pd.date_range("2020-01-01", periods=16),
            "nullable": [None] * 16 if i == 0 else list(range(16)),
        })
        if i == 2:
            frame["later_column"] = 0.25
        if wide:
            frame = pd.concat([frame, pd.DataFrame(np.ones((16, 2200)),
                              columns=[f"m{j}" for j in range(2200)])], axis=1)
        frame.to_parquet(root / f"part-{i:06d}.parquet", index=False)
    return sorted(root.glob("part-*.parquet"))


@pytest.mark.parametrize("threads", [1, 4])
def test_stream_matches_duckdb_values_types_and_sorted_analysis_input(tmp_path, threads):
    paths = _sources(tmp_path / "source" / "split=train")
    before = [file_sha256(p) for p in paths]
    with duckdb.connect() as db:
        db.execute(f"SET threads={threads}")
        db.execute("SET preserve_insertion_order=false")
        clause = feature_parquet_clause(sql_path=(paths[0].parent / "part-*.parquet").as_posix())
        old = tmp_path / "old.parquet"
        db.execute(f"COPY (SELECT * EXCLUDE (split) FROM {clause}) "
                   f"TO '{old.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)")
        schema = db.execute(f"SELECT * EXCLUDE (split) FROM {clause} LIMIT 0").to_arrow_table().schema
        streamed_schema = _validation_cache_schema(db, paths)
        assert schema.equals(streamed_schema, check_metadata=False)
        target = tmp_path / "new" / "features.parquet"
        target.parent.mkdir()
        rows = _stream_validation_cache_split(db, source_paths=paths, schema=streamed_schema,
            temporary=target, split="train", expected_rows=48, progress=None, block_files=1)
    assert rows == 48
    old_frame = pd.read_parquet(old)
    new_frame = pd.read_parquet(target)
    sort_keys = ["subject_id", "study_id"]
    pd.testing.assert_frame_equal(old_frame.sort_values(sort_keys).reset_index(drop=True),
                                  new_frame.sort_values(sort_keys).reset_index(drop=True))
    if threads == 1:
        pd.testing.assert_frame_equal(old_frame, new_frame)
    assert new_frame.subject_id.tolist() == list(range(48))
    assert before == [file_sha256(p) for p in paths]


def test_stream_pause_keeps_verified_blocks_and_resumes_without_rewriting(tmp_path):
    paths = _sources(tmp_path / "source" / "split=train")
    target = tmp_path / "output" / "features.parquet"
    target.parent.mkdir()
    with duckdb.connect() as db:
        clause = feature_parquet_clause(sql_path=(paths[0].parent / "part-*.parquet").as_posix())
        schema = db.execute(f"SELECT * EXCLUDE (split) FROM {clause} LIMIT 0").to_arrow_table().schema
        def stop(message):
            if "16/48" in message:
                raise JobPauseRequested("test checkpoint")
        args = dict(source_paths=paths, schema=schema, temporary=target,
                    split="train", expected_rows=48, block_files=1)
        with pytest.raises(JobPauseRequested):
            _stream_validation_cache_split(db, **args, progress=stop)
        block = target.parent / "_stream_blocks" / "block-000000.parquet"
        before = (file_sha256(block), block.stat().st_mtime_ns)
        messages = []
        assert _stream_validation_cache_split(db, **args, progress=messages.append) == 48
    assert before == (file_sha256(block), block.stat().st_mtime_ns)
    assert any("wiederverwendet" in m for m in messages)
    assert pq.ParquetFile(target).metadata.num_rows == 48


def test_stream_wide_schema_and_row_count_guard(tmp_path):
    paths = _sources(tmp_path / "source" / "split=train", wide=True)
    target = tmp_path / "output" / "features.parquet"
    target.parent.mkdir()
    with duckdb.connect() as db:
        clause = feature_parquet_clause(sql_path=(paths[0].parent / "part-*.parquet").as_posix())
        schema = db.execute(f"SELECT * EXCLUDE (split) FROM {clause} LIMIT 0").to_arrow_table().schema
        with pytest.raises(RuntimeError, match="source row mismatch"):
            _stream_validation_cache_split(db, source_paths=paths, schema=schema,
                temporary=target, split="train", expected_rows=49, progress=None)
    assert not target.exists()
    checkpoint = json.loads((target.parent / "_stream_blocks/checkpoint.json").read_text())
    assert checkpoint["blocks"]["0"]["row_count"] == 48
