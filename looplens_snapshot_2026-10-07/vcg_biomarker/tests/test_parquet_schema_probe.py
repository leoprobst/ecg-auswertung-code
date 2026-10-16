"""Asking which columns exist must not cost gigabytes.

``union_by_name=true`` makes DuckDB open every file and merge every schema.
Over the feature partitions -- 1,889 columns across 1,527 files -- that was
measured at 42 s and 3.6 GB against stage profiles of 2 GB, and it paused a
running study seconds after every resume. Four call sites did it, three of
them in stages the run had not reached yet.

The probe only ever answers "is this column present", which one schema read
answers just as well.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic.storage import (
    _UNIFORMITY_CACHE,
    _UNIFORMITY_SIDECAR,
    feature_parquet_clause,
    feature_partitions_share_a_schema,
    parquet_columns,
)


def _partitions(tmp_path, columns_per_file):
    for index, columns in enumerate(columns_per_file):
        frame = pd.DataFrame({name: [float(index)] for name in columns})
        frame.to_parquet(tmp_path / f"part-{index:03d}.parquet", index=False)
    return str(tmp_path / "part-*.parquet").replace("\\", "/")


@pytest.fixture()
def connection():
    con = duckdb.connect()
    con.execute("SET memory_limit='512MB'; SET threads=1;")
    yield con
    con.close()


def test_it_reports_the_columns(tmp_path, connection):
    glob = _partitions(tmp_path, [("a", "b"), ("a", "b")])
    assert parquet_columns(connection, glob) == {"a", "b"}


class _Spy:
    """DuckDB connections do not allow patching execute, so wrap one."""

    def __init__(self, connection):
        self._connection = connection
        self.statements: list[str] = []

    def execute(self, statement, *args, **kwargs):
        self.statements.append(statement)
        return self._connection.execute(statement, *args, **kwargs)


def test_it_never_unifies_schemas_when_a_plain_read_answers(tmp_path, connection):
    glob = _partitions(tmp_path, [("a", "b"), ("a", "b")])
    spy = _Spy(connection)
    assert parquet_columns(spy, glob) == {"a", "b"}
    assert not any("union_by_name" in s for s in spy.statements)
    # First and last partition, not all of them.
    assert len(spy.statements) <= 2


def test_drift_between_first_and_last_partition_is_caught(tmp_path, connection):
    """A plain read does not raise on mismatch -- it reports the first file.

    Measured: DuckDB silently returns the first schema. Trusting it would
    quietly drop markers, so both ends are read.
    """

    glob = _partitions(tmp_path, [("a", "b"), ("a", "b"), ("a", "c")])
    assert parquet_columns(connection, glob) == {"a", "b", "c"}


def test_a_column_only_in_the_middle_is_missed_and_that_is_documented(
    tmp_path, connection
):
    """The limit of the cheap probe, pinned so nobody assumes more."""

    glob = _partitions(tmp_path, [("a",), ("a", "mitte"), ("a",)])
    assert "mitte" not in parquet_columns(connection, glob)


def test_an_unreadable_path_raises_rather_than_returning_nothing(connection):
    """An empty set would read as 'the column is absent' and pick a fallback."""

    with pytest.raises(RuntimeError, match="nicht lesbar"):
        parquet_columns(connection, "Z:/nicht/vorhanden/*.parquet")


def test_no_stage_probes_schemas_the_expensive_way():
    """Four call sites had this; a fifth must not appear unnoticed."""

    from pathlib import Path

    mimic = Path("vcg_biomarker/mimic")
    offenders = []
    for path in mimic.glob("*.py"):
        if path.name == "storage.py":
            continue
        text = path.read_text(encoding="utf-8")
        for block in text.split("DESCRIBE")[1:]:
            if "union_by_name" in block[:200]:
                offenders.append(path.name)
    assert offenders == []


def test_feature_reading_stages_are_not_left_on_the_default_profile():
    """A stage that reads the whole feature glob needs room to do it.

    therapy-response and physiological-gap-markers were not listed in the
    profile table and fell through to 1 CPU / 2 GB, which left DuckDB a 1 GB
    budget for a read measured at roughly 16 GB. It did not fail -- it
    spilled, and one query spent 25.9 hours producing 3,357 rows.
    """

    from types import SimpleNamespace

    from vcg_biomarker.mimic.jobs import resource_profile_for_config

    config = SimpleNamespace(
        workers=6,
        duckdb_memory_limit_gb=16,
        os_memory_reserve_gb=6.0,
        resource_mode="auto",
    )
    for stage in ("therapy-response", "physiological-gap-markers"):
        profile = resource_profile_for_config(stage, config, {})
        assert profile.memory_gb >= 8.0, stage
        assert profile.cpu_tokens > 1, stage


# --- Choosing the read clause from the data ----------------------------------


def _partitioned(root, *, columns_per_part):
    """Write ``split=<n>/part-0.parquet`` with the given column sets."""

    for index, columns in enumerate(columns_per_part):
        folder = root / f"split={index}"
        folder.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({name: [float(index)] for name in columns})
        frame.to_parquet(folder / "part-0.parquet", index=False)
    return str(root / "split=*" / "part-*.parquet").replace("\\", "/")


def test_uniform_partitions_are_read_without_unifying(tmp_path):
    glob = _partitioned(
        tmp_path / "features",
        columns_per_part=[("study_id", "marker_a")] * 4,
    )
    clause = feature_parquet_clause(sql_path=glob)
    assert "union_by_name" not in clause
    assert glob in clause


def test_a_marker_added_partway_through_keeps_the_expensive_read(tmp_path):
    """The case broad_axis_stage documents: later parts carry a new column."""

    glob = _partitioned(
        tmp_path / "features",
        columns_per_part=[
            ("study_id", "marker_a"),
            ("study_id", "marker_a"),
            ("study_id", "marker_a", "marker_b"),
        ],
    )
    assert "union_by_name=true" in feature_parquet_clause(sql_path=glob)


def test_drift_only_in_the_middle_is_still_caught(tmp_path):
    """Every footer is read, so this is not the first/last blind spot."""

    glob = _partitioned(
        tmp_path / "features",
        columns_per_part=[
            ("study_id", "marker_a"),
            ("study_id", "marker_a", "marker_mid"),
            ("study_id", "marker_a"),
        ],
    )
    assert "union_by_name=true" in feature_parquet_clause(sql_path=glob)


def test_the_answer_survives_a_fresh_process_via_the_sidecar(tmp_path):
    glob = _partitioned(
        tmp_path / "features",
        columns_per_part=[("study_id", "marker_a")] * 3,
    )
    assert feature_partitions_share_a_schema(glob) is True
    sidecar = tmp_path / "features" / _UNIFORMITY_SIDECAR
    assert sidecar.is_file()

    _UNIFORMITY_CACHE.clear()  # what a restarted worker sees
    assert feature_partitions_share_a_schema(glob) is True


def test_a_new_extraction_invalidates_the_cached_answer(tmp_path):
    """The sidecar must not outlive the partitions it describes.

    The partition list is now memoised for a short window of its own, so this
    clears that too. The two caches guard different things and both must be
    seen to expire: this test covers the *key* -- partition count and newest
    modification time -- while
    ``test_a_new_extraction_is_seen_once_the_window_passes`` covers the
    window. Without the clear, this test would pass on a stale listing and
    prove nothing about the key at all.
    """

    from vcg_biomarker.mimic import storage

    root = tmp_path / "features"
    glob = _partitioned(root, columns_per_part=[("study_id", "marker_a")] * 3)
    assert feature_partitions_share_a_schema(glob) is True

    _partitioned_extra = root / "split=3"
    _partitioned_extra.mkdir(parents=True)
    pd.DataFrame({"study_id": [3.0], "marker_new": [1.0]}).to_parquet(
        _partitioned_extra / "part-0.parquet", index=False
    )
    _UNIFORMITY_CACHE.clear()
    storage._PROBE_CACHE.clear()
    assert feature_partitions_share_a_schema(glob) is False


def test_the_sidecar_never_lands_above_the_path_it_describes(tmp_path):
    """A pattern without wildcards used to write one directory too high."""

    folder = tmp_path / "nested" / "features"
    folder.mkdir(parents=True)
    single = folder / "part-0.parquet"
    pd.DataFrame({"study_id": [1.0]}).to_parquet(single, index=False)
    sql_path = str(single).replace("\\", "/")

    assert feature_partitions_share_a_schema(sql_path) is True
    assert (folder / _UNIFORMITY_SIDECAR).is_file()
    assert not (tmp_path / "nested" / _UNIFORMITY_SIDECAR).exists()


def test_no_matches_chooses_the_safe_expensive_read(tmp_path):
    missing = str(tmp_path / "absent" / "*.parquet").replace("\\", "/")
    assert feature_partitions_share_a_schema(missing) is False
    assert "union_by_name=true" in feature_parquet_clause(sql_path=missing)


def test_an_unreadable_footer_chooses_the_safe_expensive_read(tmp_path):
    root = tmp_path / "features"
    (root / "split=0").mkdir(parents=True)
    (root / "split=0" / "part-0.parquet").write_bytes(b"not a parquet file")
    glob = str(root / "split=*" / "part-*.parquet").replace("\\", "/")
    assert feature_partitions_share_a_schema(glob) is False


def test_extra_options_are_passed_through_verbatim(tmp_path):
    """``hive_partitioning=true`` must survive the memory fix.

    ``split`` is a physical column in these partitions, so the option is
    redundant -- but a memory fix has no business quietly dropping a read
    option it did not measure.
    """

    glob = _partitioned(
        tmp_path / "features", columns_per_part=[("study_id", "marker_a")] * 2
    )
    clause = feature_parquet_clause(
        sql_path=glob, options=("hive_partitioning=true",)
    )
    assert "hive_partitioning=true" in clause
    assert "union_by_name" not in clause


def test_extra_options_survive_the_expensive_read_too(tmp_path):
    glob = _partitioned(
        tmp_path / "features",
        columns_per_part=[("study_id",), ("study_id", "marker_b")],
    )
    clause = feature_parquet_clause(
        sql_path=glob, options=("hive_partitioning=true",)
    )
    assert "union_by_name=true" in clause
    assert "hive_partitioning=true" in clause


def test_every_feature_read_defines_its_clause_in_the_same_function():
    """An f-string with an undefined name fails only at run time.

    That is mid-run, on a stage that may have been queued for days. A syntax
    check will not catch it and neither will an import.
    """

    import ast
    from pathlib import Path as _Path

    for name in (
        "validation",
        "phenome",
        "trajectory",
        "physiological_gap_models",
        "broad_axis_stage",
        "therapy_response",
    ):
        tree = ast.parse(
            _Path(f"vcg_biomarker/mimic/{name}.py").read_text(encoding="utf-8")
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            assigned, used = set(), False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    used = used or "{read_features}" in sub.value
                if isinstance(sub, ast.FormattedValue) and isinstance(
                    sub.value, ast.Name
                ):
                    used = used or sub.value.id == "read_features"
            if used:
                assert "read_features" in assigned, f"{name}.{node.name}"


def test_the_directory_listing_is_not_repeated_per_query(tmp_path, monkeypatch):
    """Validation builds this clause once per endpoint and marker.

    Measured on the real partitions: 191 ms per call, because listing 2,562
    files and stat-ing each happened *before* the result cache was consulted.
    Over the validation stage that is 32 minutes of directory traffic added by
    a change meant to save memory.
    """

    import glob as _glob

    from vcg_biomarker.mimic import storage

    glob_pattern = _partitioned(
        tmp_path / "features", columns_per_part=[("study_id", "marker_a")] * 3
    )
    storage._PROBE_CACHE.clear()
    _UNIFORMITY_CACHE.clear()

    calls = {"n": 0}
    real = _glob.glob

    def counted(pattern, **kwargs):
        calls["n"] += 1
        return real(pattern, **kwargs)

    monkeypatch.setattr(storage.glob if hasattr(storage, "glob") else _glob,
                        "glob", counted, raising=False)
    monkeypatch.setattr(_glob, "glob", counted)

    for _ in range(50):
        feature_parquet_clause(sql_path=glob_pattern)
    assert calls["n"] == 1, f"{calls['n']} Verzeichnislisten statt einer"


def test_a_new_extraction_is_seen_once_the_window_passes(tmp_path, monkeypatch):
    """The memo may shorten directory traffic, never hide a rewrite."""

    from vcg_biomarker.mimic import storage

    root = tmp_path / "features"
    glob_pattern = _partitioned(
        root, columns_per_part=[("study_id", "marker_a")] * 2
    )
    storage._PROBE_CACHE.clear()
    _UNIFORMITY_CACHE.clear()

    now = [1000.0]
    monkeypatch.setattr(storage.time, "monotonic", lambda: now[0])
    assert "union_by_name" not in feature_parquet_clause(sql_path=glob_pattern)

    extra = root / "split=9"
    extra.mkdir()
    pd.DataFrame({"study_id": [9.0], "marker_new": [1.0]}).to_parquet(
        extra / "part-0.parquet", index=False
    )

    # Inside the window the old answer stands -- and cannot be wrong, because
    # an extraction takes days and no consumer runs during one.
    assert "union_by_name" not in feature_parquet_clause(sql_path=glob_pattern)

    now[0] += storage._PROBE_TTL_SECONDS + 1
    _UNIFORMITY_CACHE.clear()
    assert "union_by_name=true" in feature_parquet_clause(sql_path=glob_pattern)


def test_an_unusable_path_is_remembered_as_unusable(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import storage

    storage._PROBE_CACHE.clear()
    missing = str(tmp_path / "absent" / "*.parquet").replace("\\", "/")
    assert feature_partitions_share_a_schema(missing) is False
    assert feature_partitions_share_a_schema(missing) is False


def test_a_column_that_is_null_in_the_first_partition_keeps_the_expensive_read(tmp_path):
    """A092's validate-pilot-10000: same names everywhere, but a marker with
    no value in the first partition was written as the ``null`` type and as
    ``double`` in the fourth. A plain read fails to cast double to null, so
    the type must be part of what "the same schema" means."""
    import duckdb as _duckdb

    root = tmp_path / "features"
    root.mkdir()
    pd.DataFrame({"study_id": [1.0], "flag": [None]}).to_parquet(root / "part-000.parquet", index=False)
    pd.DataFrame({"study_id": [2.0], "flag": [1.0]}).to_parquet(root / "part-001.parquet", index=False)
    glob = str(root / "part-*.parquet").replace("\\", "/")
    clause = feature_parquet_clause(sql_path=glob)
    assert "union_by_name" in clause
    rows = _duckdb.connect().execute(f"SELECT flag FROM {clause} ORDER BY study_id").fetchall()
    assert rows == [(None,), (1.0,)]


def test_the_writer_casts_an_all_null_feature_to_double(tmp_path):
    from vcg_biomarker.mimic.feature_extraction import _cast_all_null_features_to_float
    import pyarrow.parquet as pq

    frame = pd.DataFrame({"study_id": [1, 2], "flag": [None, None], "value": [None, 2.0], "note": ["a", None]})
    _cast_all_null_features_to_float(frame, ["flag", "value"])
    assert frame["flag"].dtype == "float64"
    assert frame["value"].dtype == "float64"
    assert frame["note"].dtype == object
    frame.to_parquet(tmp_path / "p.parquet", index=False)
    types = dict(zip(pq.ParquetFile(tmp_path / "p.parquet").schema_arrow.names,
                     map(str, pq.ParquetFile(tmp_path / "p.parquet").schema_arrow.types)))
    assert types["flag"] == "double"


def test_split_caches_with_bool_double_and_null_columns_concatenate():
    """A093's validate-pilot-1000: the train cache held a classical flag as
    double, the validation cache as bool, and ``concat_tables`` refused."""
    import pyarrow as pa
    from vcg_biomarker.mimic.validation import _unify_column_types

    a = pa.table({"flag": pa.array([True, None]), "value": pa.array([1.0, 2.0]), "empty": pa.nulls(2)})
    b = pa.table({"flag": pa.array([1.0, None]), "value": pa.array([3.0, None]), "empty": pa.nulls(2)})
    table = pa.concat_tables(_unify_column_types([a, b]))
    assert table.schema.field("flag").type == pa.float64()
    assert table.column("flag").to_pylist() == [1.0, None, 1.0, None]
    assert table.schema.field("value").type == pa.float64()
    assert table.schema.field("empty").type == pa.float64()
