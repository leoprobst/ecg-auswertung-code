"""Grouping markers by missingness must not change a single correlation.

``_screen_continuous`` ranked both the marker and the target once per
(target, split, marker).  Its binary counterpart had already been optimised --
the comments there record 3,426 sorts instead of 1.15 million -- but the
continuous path was left as it was, and it carries the laboratory axis, the
widest of the five: 460 targets over 13.2 million rows.

Markers observed on the same recordings share the same usable rows, so the
*target* is ranked once per pattern instead of once per marker -- the marker
rankings themselves cannot be shared, and that is where most of the cost stays.
Measured on the real feature store, 1,821 numeric columns fall into 153
patterns; end to end the change takes about a quarter off.  A saving that small
is only worth taking if the numbers are unchanged, so this test recomputes every
record the old way and compares exactly.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.broad_axis_stage import (
    BroadAxisPolicy,
    _missingness_groups,
    _screen_continuous,
    _spearman,
)


def _policy() -> BroadAxisPolicy:
    return BroadAxisPolicy(
        axis="laboratory_item",
        minimum_subjects=30,
        target_kind="continuous",
    )


def _frame(seed: int, n_markers: int = 6, n_rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    markers = [f"marker_{index}" for index in range(n_markers)]
    rows = {
        "target_id": rng.choice(["t_a", "t_b"], size=n_rows),
        "split": rng.choice(["train", "validation"], size=n_rows),
        "value": np.round(rng.normal(size=n_rows), 1),
        "study_id": np.arange(n_rows),
    }
    for index, name in enumerate(markers):
        column = np.round(rng.normal(size=n_rows), 1)
        # Two markers deliberately share a missingness pattern, one has its own,
        # and one is complete -- the three cases the grouping has to handle.
        if index in (0, 1):
            column[rng.random(n_rows) < 0.05] = np.nan
        elif index == 2:
            column[rng.random(n_rows) < 0.2] = np.nan
        rows[name] = column
    frame = pd.DataFrame(rows)
    # Force markers 0 and 1 onto exactly the same pattern.
    frame["marker_1"] = np.where(
        frame["marker_0"].isna(), np.nan, frame["marker_1"]
    )
    return frame


def _reference_records(frame: pd.DataFrame, markers, policy) -> dict:
    """The pre-optimisation loop: rank both series for every marker."""

    from vcg_biomarker.mimic.broad_axis_stage import (
        MINIMUM_CASES,
        MINIMUM_MARKER_COVERAGE,
    )

    out: dict[tuple, tuple[float, float, int]] = {}
    for target_id, block in frame.groupby("target_id", sort=True):
        for split in ("train", "validation"):
            scoped = block.loc[block["split"] == split]
            if len(scoped) < MINIMUM_CASES:
                continue
            target_values = pd.to_numeric(
                scoped["value"], errors="coerce"
            ).to_numpy(dtype=float)
            for marker_id in markers:
                values = pd.to_numeric(
                    scoped[marker_id], errors="coerce"
                ).to_numpy(dtype=float)
                usable = np.isfinite(values) & np.isfinite(target_values)
                n_usable = int(usable.sum())
                if n_usable < MINIMUM_CASES:
                    continue
                if n_usable / max(1, len(scoped)) < MINIMUM_MARKER_COVERAGE:
                    continue
                rho, p_value = _spearman(values[usable], target_values[usable])
                if not np.isfinite(p_value):
                    continue
                out[(str(target_id), split, marker_id)] = (rho, p_value, n_usable)
    return out


@pytest.mark.parametrize("seed", [1, 5, 9, 17])
def test_the_grouped_screen_reproduces_every_record(seed: int) -> None:
    markers = [f"marker_{index}" for index in range(6)]
    frame = _frame(seed)
    policy = _policy()
    records = _screen_continuous(frame, "laboratory_item", policy, {}, markers)
    reference = _reference_records(frame, markers, policy)

    produced = {
        (r["target_id"], r["evaluation_split"], r["feature_id"]): (
            r["effect"],
            r["p_value"],
            r["n_observations"],
        )
        for r in records
    }
    assert set(produced) == set(reference)
    for key, (rho, p_value, n_usable) in reference.items():
        assert produced[key][0] == rho, key
        assert produced[key][1] == p_value, key
        assert produced[key][2] == n_usable, key


def test_markers_sharing_a_pattern_share_a_group() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, np.nan, 4.0],
            [np.nan, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ]
    )
    groups = _missingness_groups(matrix, ["a", "b", "c", "d"])
    sizes = sorted(len(positions) for _, positions in groups)
    # b and d are complete; a and c each miss a different row.
    assert sizes == [1, 1, 2]
    total = sum(len(positions) for _, positions in groups)
    assert total == matrix.shape[1]


def test_a_group_mask_has_one_entry_per_row() -> None:
    matrix = np.array([[1.0, np.nan], [2.0, 3.0]])
    for mask, positions in _missingness_groups(matrix, ["a", "b"]):
        assert mask.shape == (matrix.shape[0],)
        assert mask.dtype == bool
        assert positions
