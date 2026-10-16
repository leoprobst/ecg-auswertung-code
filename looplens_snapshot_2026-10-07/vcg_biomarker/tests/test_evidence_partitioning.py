"""Bucketed execution of the ECG-centred evidence windows.

Three queries in the evidence layer aggregate per ECG with ``ARG_MIN``/
``ARG_MAX`` over STRUCT ordering keys. DuckDB keeps those aggregate states
resident -- a state holding a STRUCT or a VARCHAR has no fixed size and cannot
spill -- so peak memory follows *group count x aggregate count* and no amount
of temporary directory helps. The laboratory window carries 42 such aggregates
over roughly 3.5 million groups, which is what ran the stage out of its 6 GiB.

The fix splits the work into subject buckets. That is only sound if the buckets
are disjoint in the grouping key and their union reproduces the unpartitioned
result exactly -- otherwise the stage would trade an out-of-memory error for
silently missing or duplicated rows, which is far worse. Every test below is
that equality, checked in both directions with ``EXCEPT ALL`` so neither a
dropped nor a duplicated row can hide.
"""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from vcg_biomarker.mimic.evidence import (  # noqa: E402
    LAB_ITEM_GROUPS,
    _is_out_of_memory,
    _lab_feature_pivot_query,
    _lab_troponin_serial_query,
    _lab_window_pivot_sql,
    _lab_windows_query,
)


TROPONIN_ITEMIDS = sorted(
    set(LAB_ITEM_GROUPS["troponin_t"]) | set(LAB_ITEM_GROUPS["troponin_i"])
)
POTASSIUM_ITEMID = LAB_ITEM_GROUPS["potassium"][0]


@pytest.fixture(scope="module")
def synthetic_sources(tmp_path_factory) -> tuple[str, str]:
    """A small master index and lab table with the shapes that matter.

    Subjects carry several ECGs, labs fall inside and outside the 24 h window,
    admission identifiers are present, absent and mismatched, and some values
    are unusable -- so the aggregate FILTERs are actually exercised rather than
    seeing one clean row per group.
    """

    root = tmp_path_factory.mktemp("evidence-partitioning")
    master = (root / "master.parquet").as_posix()
    labs = (root / "labs.parquet").as_posix()
    troponin_list = ", ".join(str(itemid) for itemid in TROPONIN_ITEMIDS)
    connection = duckdb.connect()
    connection.execute("SET memory_limit='1GB'")
    connection.execute(
        f"""
        COPY (
            SELECT
                (s % 97)::BIGINT AS subject_id,
                s::BIGINT AS study_id,
                TIMESTAMP '2150-01-01 00:00:00'
                    + INTERVAL (s * 7) HOUR AS ecg_time,
                CASE WHEN s % 5 = 0 THEN NULL ELSE (s % 31)::BIGINT END
                    AS hadm_id
            FROM range(600) t(s)
        ) TO '{master}' (FORMAT PARQUET)
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                (r * 10 + itemid % 10)::BIGINT AS labevent_id,
                (r % 97)::BIGINT AS subject_id,
                CASE WHEN r % 7 = 0 THEN NULL ELSE (r % 31)::BIGINT END
                    AS hadm_id,
                itemid::INTEGER AS itemid,
                CASE
                    WHEN itemid = {TROPONIN_ITEMIDS[0]} THEN 'troponin_t'
                    WHEN itemid IN ({troponin_list}) THEN 'troponin_i'
                    ELSE 'potassium'
                END AS lab_group,
                TIMESTAMP '2150-01-01 00:00:00'
                    + INTERVAL (r * 3) HOUR AS charttime,
                TIMESTAMP '2150-01-01 00:00:00'
                    + INTERVAL (r * 3 + 1) HOUR AS storetime,
                ((r % 40) / 4.0)::VARCHAR AS value_raw,
                (r % 40) / 4.0 AS value_numeric,
                (r % 40) / 4.0 AS value_bound,
                NULL::VARCHAR AS censoring,
                (r % 40) / 4.0 AS value_interval_lower,
                (r % 40) / 4.0 AS value_interval_upper,
                FALSE AS value_interval_lower_open,
                FALSE AS value_interval_upper_open,
                CASE WHEN r % 11 = 0 THEN NULL ELSE 'mEq/L' END AS unit,
                3.5 AS reference_lower,
                4.0 AS reference_upper,
                (r % 40) / 16.0 AS upper_reference_ratio,
                CASE WHEN (r % 40) / 16.0 > 1 THEN 'positive' ELSE 'negative' END
                    AS upper_reference_state,
                CASE WHEN r % 17 = 0 THEN 'abnormal' ELSE NULL END
                    AS abnormal_flag,
                CASE WHEN r % 13 = 0 THEN 'unit_missing' ELSE 'usable' END
                    AS value_status
            FROM range(4000) t(r)
            CROSS JOIN (
                SELECT unnest([{troponin_list}, {POTASSIUM_ITEMID}]) AS itemid
            ) items
        ) TO '{labs}' (FORMAT PARQUET)
        """
    )
    connection.close()
    return master, labs


def _assert_bucketing_is_lossless(build, *args, buckets: int = 8) -> int:
    """Union of the buckets must equal the unpartitioned result, row for row."""

    connection = duckdb.connect()
    connection.execute("SET memory_limit='2GB'")
    # One thread: for wide aggregations DuckDB builds a hash table per thread,
    # and on fixture-sized input that setup is the entire runtime.
    connection.execute("SET threads=1")
    try:
        connection.execute(
            "CREATE TABLE whole AS "
            + build(*args, partition_index=0, partition_count=1)
        )
        # One statement per bucket, exactly as the stage executes them. A
        # single UNION ALL over every bucket would instead hand the planner
        # eight copies of sixty STRUCT aggregates and spend minutes in
        # optimisation that production never pays.
        for index in range(buckets):
            bucket_sql = build(
                *args, partition_index=index, partition_count=buckets
            )
            if index == 0:
                connection.execute(f"CREATE TABLE bucketed AS {bucket_sql}")
            else:
                connection.execute(f"INSERT INTO bucketed {bucket_sql}")
        whole = connection.execute("SELECT count(*) FROM whole").fetchone()[0]
        bucketed = connection.execute(
            "SELECT count(*) FROM bucketed"
        ).fetchone()[0]
        assert whole > 0, "fixture produced no groups; the test proves nothing"
        assert whole == bucketed
        missing = connection.execute(
            "SELECT count(*) FROM "
            "(SELECT * FROM whole EXCEPT ALL SELECT * FROM bucketed)"
        ).fetchone()[0]
        extra = connection.execute(
            "SELECT count(*) FROM "
            "(SELECT * FROM bucketed EXCEPT ALL SELECT * FROM whole)"
        ).fetchone()[0]
        assert (missing, extra) == (0, 0)
        return whole
    finally:
        connection.close()


def test_lab_window_bucketing_is_lossless(synthetic_sources):
    master, labs = synthetic_sources
    _assert_bucketing_is_lossless(_lab_windows_query, master, labs)


def test_nearest_reference_is_not_borrowed_from_another_sample(synthetic_sources, tmp_path):
    master, labs = synthetic_sources
    with duckdb.connect() as connection:
        query = _lab_windows_query(master, labs, partition_index=0, partition_count=1)
        first = connection.execute(
            f"SELECT study_id, lab_group, nearest_labevent_id_6h FROM ({query}) "
            "WHERE count_6h > 1 AND nearest_reference_lower_6h IS NOT NULL LIMIT 1"
        ).fetchone()
        assert first is not None
        study_id, group, event_id = first
        modified = connection.execute(
            "SELECT * REPLACE (CASE WHEN labevent_id = ? THEN NULL ELSE reference_lower END "
            "AS reference_lower, CASE WHEN labevent_id = ? THEN NULL ELSE reference_upper END "
            "AS reference_upper) FROM read_parquet(?)", [event_id, event_id, labs]
        ).fetchdf()
        updated = tmp_path / "missing_reference.parquet"
        modified.to_parquet(updated, index=False)
        query = _lab_windows_query(master, updated.as_posix(), partition_index=0, partition_count=1)
        result = connection.execute(
            f"SELECT nearest_labevent_id_6h, nearest_reference_lower_6h, "
            f"nearest_reference_upper_6h FROM ({query}) WHERE study_id = ? AND lab_group = ?",
            [study_id, group],
        ).fetchone()
    assert result == (event_id, None, None)


def test_troponin_serial_bucketing_is_lossless(synthetic_sources):
    master, labs = synthetic_sources
    _assert_bucketing_is_lossless(_lab_troponin_serial_query, master, labs)


def test_lab_feature_pivot_bucketing_is_lossless(synthetic_sources, tmp_path):
    master, labs = synthetic_sources
    windows = (tmp_path / "windows.parquet").as_posix()
    connection = duckdb.connect()
    connection.execute("SET memory_limit='1GB'")
    connection.execute(
        "COPY ("
        + _lab_windows_query(master, labs, partition_index=0, partition_count=1)
        + f") TO '{windows}' (FORMAT PARQUET)"
    )
    connection.close()
    # Losslessness is a property of the bucket predicate, not of the aggregate
    # list, so a handful of groups proves it. The full panel generates 208
    # conditional aggregates, and DuckDB's per-statement cost for a wide
    # aggregation is fixed rather than data-driven -- nine such statements take
    # minutes even on a hundred rows, which is a bad trade for a unit test.
    pivot_sql = _lab_window_pivot_sql(
        {
            name: LAB_ITEM_GROUPS[name]
            for name in ("creatinine", "hemoglobin", "lactate")
        },
        exclude=(),
    )
    _assert_bucketing_is_lossless(_lab_feature_pivot_query, windows, pivot_sql)


def test_unpartitioned_queries_carry_no_bucket_predicate():
    """A single bucket must not leave a hash filter in the plan."""

    query = _lab_windows_query("m", "l", partition_index=0, partition_count=1)
    assert "hash(" not in query
    bucketed = _lab_windows_query("m", "l", partition_index=3, partition_count=8)
    # Both sides of the equi-join are filtered: redundant in logic, but it is
    # what shrinks the join build side rather than only the probe side.
    assert "hash(m.subject_id) % 8 = 3" in bucketed
    assert "hash(l.subject_id) % 8 = 3" in bucketed


@pytest.mark.parametrize(
    "build",
    [_lab_windows_query, _lab_troponin_serial_query],
)
@pytest.mark.parametrize(
    "index, count",
    [(0, 0), (-1, 4), (4, 4), (5, 4)],
)
def test_query_builders_reject_impossible_buckets(build, index, count):
    with pytest.raises(ValueError):
        build("m", "l", partition_index=index, partition_count=count)


class OutOfMemoryException(Exception):
    """Stands in for the DuckDB exception, which is matched by class name."""


def test_out_of_memory_is_recognised_by_class_name_and_message():
    assert _is_out_of_memory(OutOfMemoryException("failed to pin block"))
    assert _is_out_of_memory(RuntimeError("Out of Memory Error: 5.5 GiB used"))
    assert not _is_out_of_memory(RuntimeError("Binder Error: no such column"))


def test_bucket_count_doubles_until_the_query_fits(monkeypatch, tmp_path):
    """A guessed bucket count is a guess; the helper must correct it."""

    from vcg_biomarker.mimic import evidence

    attempts: list[int] = []
    notes: list[str] = []

    def fake_copy(connection, query: str, output_path: Path) -> int:
        if query.startswith("SELECT * FROM read_parquet"):
            return 7
        count = int(query.rsplit(" of ", 1)[1])
        attempts.append(count)
        if count < 16:
            raise OutOfMemoryException("failed to pin block of size 256.0 KiB")
        return 1

    monkeypatch.setattr(evidence, "_copy_query_atomic", fake_copy)
    total = evidence._copy_partitioned_query_atomic(
        object(),
        lambda index, count: f"bucket {index} of {count}",
        tmp_path / "out.parquet",
        initial_partitions=4,
        progress=notes.append,
        label="test",
    )

    assert total == 7
    # Four buckets fail on the first one, eight fail, sixteen go through.
    assert sorted(set(attempts)) == [4, 8, 16]
    assert attempts.count(16) == 16
    assert any("16 Buckets" in note for note in notes)


def test_bucket_refinement_does_not_recompute_completed_subjects(
    monkeypatch,
    tmp_path,
):
    """Completed coarse buckets cover both of their refined descendants."""

    from vcg_biomarker.mimic import evidence

    attempts: list[tuple[int, int]] = []
    failed_once = False

    def fake_copy(connection, query: str, output_path: Path) -> int:
        nonlocal failed_once
        if query.startswith("SELECT * FROM read_parquet"):
            return 11
        _, index, _, count = query.split()
        bucket = (int(index), int(count))
        attempts.append(bucket)
        if bucket == (2, 4) and not failed_once:
            failed_once = True
            raise OutOfMemoryException("failed to pin block")
        output_path.touch()
        return 1

    monkeypatch.setattr(evidence, "_copy_query_atomic", fake_copy)
    total = evidence._copy_partitioned_query_atomic(
        object(),
        lambda index, count: f"bucket {index} of {count}",
        tmp_path / "out.parquet",
        initial_partitions=4,
        progress=None,
        label="test",
    )

    assert total == 11
    assert attempts[:3] == [(0, 4), (1, 4), (2, 4)]
    assert (0, 8) not in attempts
    assert (4, 8) not in attempts
    assert (1, 8) not in attempts
    assert (5, 8) not in attempts
    assert set(attempts[3:]) == {(2, 8), (3, 8), (6, 8), (7, 8)}


def test_refinement_gives_up_instead_of_looping_forever(monkeypatch, tmp_path):
    from vcg_biomarker.mimic import evidence

    def always_out_of_memory(connection, query: str, output_path: Path) -> int:
        raise OutOfMemoryException("failed to pin block")

    monkeypatch.setattr(evidence, "_copy_query_atomic", always_out_of_memory)
    with pytest.raises(RuntimeError, match="larger memory allocation"):
        evidence._copy_partitioned_query_atomic(
            object(),
            lambda index, count: f"bucket {index} of {count}",
            tmp_path / "out.parquet",
            initial_partitions=4,
            max_partitions=16,
            label="test",
        )


def test_non_memory_failures_are_not_bucketed_around(monkeypatch, tmp_path):
    """Refining buckets must not paper over an actual SQL error."""

    from vcg_biomarker.mimic import evidence

    def binder_error(connection, query: str, output_path: Path) -> int:
        raise RuntimeError("Binder Error: referenced column not found")

    monkeypatch.setattr(evidence, "_copy_query_atomic", binder_error)
    with pytest.raises(RuntimeError, match="Binder Error"):
        evidence._copy_partitioned_query_atomic(
            object(),
            lambda index, count: f"bucket {index} of {count}",
            tmp_path / "out.parquet",
            initial_partitions=2,
            label="test",
        )


@pytest.mark.parametrize("fail", [False, True])
def test_single_copy_thread_limit_is_always_restored(
    monkeypatch, tmp_path, fail
):
    """A memory-bound COPY must not throttle later evidence queries."""

    from vcg_biomarker.mimic import evidence

    connection = duckdb.connect()
    connection.execute("SET threads=4")
    observed: list[int] = []

    def fake_copy(connection, query: str, output_path: Path) -> int:
        observed.append(
            int(
                connection.execute(
                    "SELECT current_setting('threads')"
                ).fetchone()[0]
            )
        )
        if fail:
            raise RuntimeError("synthetic failure")
        return 7

    monkeypatch.setattr(evidence, "_copy_query_atomic", fake_copy)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="synthetic failure"):
                evidence._copy_query_atomic_with_threads(
                    connection,
                    "SELECT 1",
                    tmp_path / "out.parquet",
                    threads=1,
                )
        else:
            assert evidence._copy_query_atomic_with_threads(
                connection,
                "SELECT 1",
                tmp_path / "out.parquet",
                threads=1,
            ) == 7
        assert observed == [1]
        assert int(
            connection.execute(
                "SELECT current_setting('threads')"
            ).fetchone()[0]
        ) == 4
    finally:
        connection.close()
