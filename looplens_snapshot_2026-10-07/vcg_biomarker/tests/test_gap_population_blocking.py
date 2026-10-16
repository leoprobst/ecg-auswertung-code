"""Reading the cohort in blocks must return exactly the wide result.

``_load_gap_population`` selects every marker it was given for the whole
cohort and calls ``fetchdf``. That was assumed to be 640,104 x 1,753 doubles,
about 9 GB. Measured, it is 80 markers -- the ``primary_core`` panel -- and
0.41 GB, so blocking buys nothing at today's size and exists only to bound the
cost if the panel grows.

The tests are the part that keeps its value regardless: whatever the size,
blocking may only change the peak. The row order, the
values, and the alignment between a marker and its row are all load-bearing --
a block that lined up one row off would silently attach every marker to the
wrong patient, which is worse than running out of memory.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.physiological_gap_models import (
    GAP_COLUMN_BLOCK_SIZE,
    _fetch_gap_population_in_blocks,
    _quote_identifier,
)


MARKERS = tuple(f"marker_{index:03d}" for index in range(37))


@pytest.fixture()
def cohort(tmp_path):
    rng = np.random.default_rng(4)
    rows = 240
    features = pd.DataFrame(
        {
            "subject_id": rng.integers(1, 60, size=rows),
            "study_id": np.arange(rows) + 1000,
            "split": rng.choice(["train", "validation", "locked_test"], rows),
            "ecg_time": pd.Timestamp("2026-01-01"),
            "extraction_status": rng.choice(
                ["ok", "ok", "ok", "signal_qc_failed"], rows
            ),
        }
    )
    for name in MARKERS:
        features[name] = rng.normal(size=rows)
    feature_path = tmp_path / "features.parquet"
    features.to_parquet(feature_path, index=False)

    context = features[["subject_id", "study_id", "split"]].copy()
    context["age"] = rng.integers(30, 90, size=rows)
    context["bmi_before_ecg"] = rng.normal(27.0, 4.0, size=rows)
    context["sex"] = rng.choice(["M", "F"], rows)
    context_path = tmp_path / "context.parquet"
    context.to_parquet(context_path, index=False)

    return (
        str(feature_path).replace("\\", "/"),
        str(context_path).replace("\\", "/"),
    )


@pytest.fixture()
def connection():
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'; SET threads=1;")
    yield con
    con.close()


def _wide_reference(connection, feature_sql, context_sql, selected):
    """The query as it stood before blocking."""

    columns = ", ".join(f"f.{_quote_identifier(item)}" for item in selected)
    return connection.execute(
        f"""
        SELECT
            f.subject_id, f.study_id, f.split,
            c.age, c.bmi_before_ecg, c.sex,
            {columns}
        FROM read_parquet('{feature_sql}', union_by_name=true) f
        JOIN read_parquet('{context_sql}') c
          USING (subject_id, study_id, split)
        WHERE f.split IN ('train', 'validation')
          AND f.extraction_status = 'ok'
        ORDER BY f.subject_id, f.study_id, f.split
        """
    ).fetchdf()


def test_blocks_reproduce_the_wide_result_exactly(cohort, connection):
    feature_sql, context_sql = cohort
    reference = _wide_reference(connection, feature_sql, context_sql, MARKERS)
    blocked = _fetch_gap_population_in_blocks(
        connection,
        feature_sql=feature_sql,
        context_sql=context_sql,
        selected=MARKERS,
        block_size=7,
    )
    pd.testing.assert_frame_equal(
        blocked.reset_index(drop=True),
        reference.reset_index(drop=True),
        check_like=False,
    )


@pytest.mark.parametrize("block_size", [1, 5, 37, 1000])
def test_the_result_does_not_depend_on_the_block_size(
    cohort, connection, block_size
):
    feature_sql, context_sql = cohort
    reference = _fetch_gap_population_in_blocks(
        connection,
        feature_sql=feature_sql,
        context_sql=context_sql,
        selected=MARKERS,
        block_size=GAP_COLUMN_BLOCK_SIZE,
    )
    blocked = _fetch_gap_population_in_blocks(
        connection,
        feature_sql=feature_sql,
        context_sql=context_sql,
        selected=MARKERS,
        block_size=block_size,
    )
    pd.testing.assert_frame_equal(blocked, reference)


def test_a_marker_stays_attached_to_its_own_row(cohort, connection):
    """The failure that would be worse than running out of memory."""

    feature_sql, context_sql = cohort
    blocked = _fetch_gap_population_in_blocks(
        connection,
        feature_sql=feature_sql,
        context_sql=context_sql,
        selected=MARKERS,
        block_size=4,
    )
    raw = pd.read_parquet(feature_sql)
    raw = raw[
        raw["extraction_status"].eq("ok")
        & raw["split"].isin(["train", "validation"])
    ]
    merged = blocked.merge(
        raw[["study_id", MARKERS[0], MARKERS[-1]]],
        on="study_id",
        suffixes=("", "_raw"),
    )
    assert len(merged) == len(blocked)
    np.testing.assert_allclose(
        merged[MARKERS[0]].to_numpy(), merged[f"{MARKERS[0]}_raw"].to_numpy()
    )
    np.testing.assert_allclose(
        merged[MARKERS[-1]].to_numpy(), merged[f"{MARKERS[-1]}_raw"].to_numpy()
    )


def test_no_markers_returns_only_the_keys(cohort, connection):
    feature_sql, context_sql = cohort
    frame = _fetch_gap_population_in_blocks(
        connection,
        feature_sql=feature_sql,
        context_sql=context_sql,
        selected=(),
    )
    assert list(frame.columns) == [
        "subject_id",
        "study_id",
        "split",
        "age",
        "bmi_before_ecg",
        "sex",
    ]


def test_the_dtypes_are_not_narrowed(cohort, connection):
    """A memory fix has no business moving the numbers."""

    feature_sql, context_sql = cohort
    frame = _fetch_gap_population_in_blocks(
        connection,
        feature_sql=feature_sql,
        context_sql=context_sql,
        selected=MARKERS,
        block_size=9,
    )
    assert frame[MARKERS[0]].dtype == np.float64


def test_locked_test_rows_never_enter(cohort, connection):
    feature_sql, context_sql = cohort
    frame = _fetch_gap_population_in_blocks(
        connection,
        feature_sql=feature_sql,
        context_sql=context_sql,
        selected=MARKERS[:3],
    )
    assert set(frame["split"].unique()) <= {"train", "validation"}
