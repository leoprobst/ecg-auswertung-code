"""Reading the replicate pairs in blocks must change nothing but the peak.

The pair artifact is the widest table this project writes: two columns per
marker, 2 x 1,753 = 3,506 of them, one row per replicate pair. It was read
whole into pandas on a stage that had never run.

Exact rather than approximate: ``marker_repeatability`` consumes exactly one
marker's two columns, so which markers share a read cannot move a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.repeatability import (
    REPEATABILITY_COLUMN_BLOCK,
    repeatability_table,
    repeatability_table_from_parquet,
)


MARKERS = tuple(f"marker_{index:02d}" for index in range(31))


@pytest.fixture()
def pairs_file(tmp_path):
    rng = np.random.default_rng(21)
    rows = 500
    frame = pd.DataFrame({"subject_id": rng.integers(1, 90, rows)})
    for name in MARKERS:
        first = rng.normal(size=rows)
        frame[f"{name}_first"] = first
        frame[f"{name}_second"] = first + rng.normal(0.0, 0.3, rows)
    # A marker with gaps and one that is entirely missing on the second read.
    frame.loc[frame.index[:200], f"{MARKERS[4]}_second"] = np.nan
    frame[f"{MARKERS[9]}_second"] = np.nan
    path = tmp_path / "replicate_pairs.parquet"
    frame.to_parquet(path, index=False)
    return path, frame


def test_the_blocked_read_reproduces_the_whole_frame_exactly(pairs_file):
    path, frame = pairs_file
    whole = repeatability_table(frame, MARKERS)
    blocked = repeatability_table_from_parquet(path, MARKERS)
    pd.testing.assert_frame_equal(
        blocked.reset_index(drop=True),
        whole.reset_index(drop=True),
        check_like=False,
    )


@pytest.mark.parametrize("block_size", [1, 3, 31, 500])
def test_the_result_does_not_depend_on_the_block_size(pairs_file, block_size):
    path, _frame = pairs_file
    reference = repeatability_table_from_parquet(
        path, MARKERS, block_size=REPEATABILITY_COLUMN_BLOCK
    )
    blocked = repeatability_table_from_parquet(
        path, MARKERS, block_size=block_size
    )
    pd.testing.assert_frame_equal(blocked, reference)


def test_a_marker_absent_from_the_artifact_is_skipped_not_invented(pairs_file):
    path, _frame = pairs_file
    table = repeatability_table_from_parquet(path, (*MARKERS, "never_written"))
    assert "never_written" not in set(table["marker_id"])
    assert len(table) == len(MARKERS)


def test_no_markers_yields_the_empty_shape_not_an_error(pairs_file):
    path, _frame = pairs_file
    table = repeatability_table_from_parquet(path, ())
    assert table.empty
    assert "marker_id" in table.columns


def test_only_the_block_columns_are_read(pairs_file, monkeypatch):
    """The whole point: a 3,506-column table must never be read whole."""

    import pyarrow.parquet as pq

    widths = []
    real = pq.read_table

    def counted(source, **kwargs):
        widths.append(len(kwargs.get("columns") or []))
        return real(source, **kwargs)

    monkeypatch.setattr(pq, "read_table", counted)
    path, _frame = pairs_file
    repeatability_table_from_parquet(path, MARKERS, block_size=5)
    assert widths, "es wurde gar nicht gelesen"
    assert max(widths) == 10, widths
