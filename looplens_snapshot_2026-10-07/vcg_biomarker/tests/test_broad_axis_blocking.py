"""Blocking the broad screen may change the peak, never a number.

Measured on the real laboratory axis before this change: 16,527,138 value
rows, 13,232,238 after the split filter, joined against 1,713 marker columns
in a single ``fetchdf()`` -- **181 GB** on a machine with 32 GB. The ICU axis
was 74 GB. The stage had never run, so nobody had met it; the run would have
reached it after days and died rather than paused.

Blocking is exact rather than approximate, and that is the whole argument for
doing it this way: ``_screen_continuous`` computes one Spearman correlation
per (target, split, marker) from a single marker column, and ``_screen_binary``
ranks each marker on its own before comparing groups. Which markers happen to
share a materialised frame therefore cannot enter any result. These tests hold
that claim to the data instead of to the reasoning.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.broad_axis_stage import (
    BROAD_AXIS_CELL_BUDGET,
    DEVELOPMENT_SPLITS,
    _marker_blocks,
    _ordered_records,
    _screen_binary,
    _screen_continuous,
)
from vcg_biomarker.mimic.broad_event_screen import policy_for


MARKERS = tuple(f"marker_{index:02d}" for index in range(23))
TARGETS = tuple(f"t{index}" for index in range(6))


# The real prespecified policies, not hand-built ones: a floor below 30
# subjects per class is refused on purpose, and a test that works around that
# would be exercising a screen the project does not run.
@pytest.fixture()
def policy():
    return policy_for("laboratory_item")


@pytest.fixture()
def binary_policy():
    return policy_for("procedure_code")


@pytest.fixture()
def value_frame():
    """One row per (ECG, target), the shape the value join produces."""

    rng = np.random.default_rng(11)
    ecgs = 400
    rows = []
    for target in TARGETS:
        for study in range(ecgs):
            rows.append(
                {
                    "target_id": target,
                    "split": "train" if study % 3 else "validation",
                    "value": rng.normal(140.0, 5.0),
                    "study_id": study,
                }
            )
    frame = pd.DataFrame(rows)
    markers = pd.DataFrame(
        rng.normal(size=(ecgs, len(MARKERS))), columns=list(MARKERS)
    )
    markers["study_id"] = np.arange(ecgs)
    # A marker with gaps, so the coverage rule is exercised too.
    markers.loc[markers.index[:120], MARKERS[3]] = np.nan
    merged = frame.merge(markers, on="study_id").drop(columns=["study_id"])
    return merged


@pytest.fixture()
def cohort_and_presence():
    rng = np.random.default_rng(5)
    ecgs = 500
    cohort = pd.DataFrame(
        {
            "study_id": np.arange(ecgs),
            "split": ["train" if i % 3 else "validation" for i in range(ecgs)],
        }
    )
    for name in MARKERS:
        cohort[name] = rng.normal(size=ecgs)
    cohort.loc[cohort.index[:90], MARKERS[7]] = np.nan
    presence = pd.DataFrame(
        [
            {
                "study_id": int(study),
                "target_id": target,
                "temporal_relation": "before",
            }
            for target in TARGETS
            for study in rng.choice(ecgs, size=140, replace=False)
        ]
    )
    return cohort, presence


def _records(frame, policy, marker_ids):
    return _screen_continuous(frame, "laboratory_item", policy, {}, marker_ids)


def test_blocking_reproduces_the_continuous_screen_exactly(value_frame, policy):
    whole = _ordered_records(_records(value_frame, policy, MARKERS), MARKERS)

    blocked = []
    for block in _marker_blocks(MARKERS, len(value_frame)):
        columns = ["target_id", "split", "value", *block]
        blocked.extend(_records(value_frame[columns], policy, block))
    blocked = _ordered_records(blocked, MARKERS)

    assert len(blocked) == len(whole)
    pd.testing.assert_frame_equal(pd.DataFrame(blocked), pd.DataFrame(whole))


@pytest.mark.parametrize("per_block", [1, 2, 5, 23, 100])
def test_the_continuous_result_does_not_depend_on_the_block_size(
    value_frame, policy, per_block, monkeypatch
):
    whole = _ordered_records(_records(value_frame, policy, MARKERS), MARKERS)
    blocked = []
    for start in range(0, len(MARKERS), per_block):
        block = MARKERS[start : start + per_block]
        columns = ["target_id", "split", "value", *block]
        blocked.extend(_records(value_frame[columns], policy, block))
    pd.testing.assert_frame_equal(
        pd.DataFrame(_ordered_records(blocked, MARKERS)), pd.DataFrame(whole)
    )


def test_blocking_reproduces_the_binary_screen_exactly(
    cohort_and_presence, binary_policy
):
    cohort, presence = cohort_and_presence
    whole = _ordered_records(
        _screen_binary(
            cohort, presence, "procedure_code", binary_policy, {}, MARKERS
        ),
        MARKERS,
    )
    blocked = []
    for block in _marker_blocks(MARKERS, len(cohort)):
        columns = ["study_id", "split", *block]
        blocked.extend(
            _screen_binary(
                cohort[columns], presence, "procedure_code", binary_policy,
                {}, block,
            )
        )
    blocked = _ordered_records(blocked, MARKERS)
    assert len(blocked) == len(whole)
    pd.testing.assert_frame_equal(pd.DataFrame(blocked), pd.DataFrame(whole))


def test_a_marker_keeps_its_own_effect(value_frame, policy):
    """The failure that would be worse than running out of memory."""

    whole = {
        (r["target_id"], r["evaluation_split"], r["feature_id"]): r["effect"]
        for r in _records(value_frame, policy, MARKERS)
    }
    for block in _marker_blocks(MARKERS, len(value_frame)):
        columns = ["target_id", "split", "value", *block]
        for r in _records(value_frame[columns], policy, block):
            key = (r["target_id"], r["evaluation_split"], r["feature_id"])
            assert r["effect"] == pytest.approx(whole[key], rel=0, abs=0)


def test_the_block_size_is_bounded_by_the_cell_budget():
    markers = tuple(f"m{i}" for i in range(1713))
    blocks = _marker_blocks(markers, 13_232_238)
    assert sum(len(b) for b in blocks) == len(markers)
    assert len(set(sum(blocks, ()))) == len(markers), "Marker doppelt oder fehlend"
    for block in blocks:
        assert len(block) * 13_232_238 <= BROAD_AXIS_CELL_BUDGET


def test_small_data_is_still_one_block():
    """A pilot must not pay for a guard the full cohort needs."""

    markers = tuple(f"m{i}" for i in range(200))
    assert len(_marker_blocks(markers, 1_000)) == 1


def test_no_markers_yields_no_blocks():
    assert _marker_blocks((), 10_000) == ()


def test_the_order_is_stable_whatever_the_block_size(value_frame, policy):
    orders = []
    for per_block in (1, 4, 23):
        records = []
        for start in range(0, len(MARKERS), per_block):
            block = MARKERS[start : start + per_block]
            columns = ["target_id", "split", "value", *block]
            records.extend(_records(value_frame[columns], policy, block))
        ordered = _ordered_records(records, MARKERS)
        orders.append(
            [
                (r["target_id"], r["evaluation_split"], r["feature_id"])
                for r in ordered
            ]
        )
    assert orders[0] == orders[1] == orders[2]


# --- The prespecified effect floor -------------------------------------------


def test_the_effect_floor_is_fixed_before_any_screen_runs():
    """A floor chosen after seeing results is not a floor.

    At 13,232,238 rows a Spearman rho of 0.001 is significant at any alpha
    anybody would write down, so without this the word "significant" measures
    the cohort size.
    """

    from vcg_biomarker.mimic import broad_axis_stage as module

    assert module.MINIMUM_ABSOLUTE_RHO == 0.10
    assert module.MINIMUM_ABSOLUTE_AUC_MARGIN == 0.05


@pytest.mark.parametrize(
    ("effect", "kind", "expected"),
    [
        (0.10, "continuous", True),
        (0.0999, "continuous", False),
        (-0.42, "continuous", True),
        (0.0, "continuous", False),
        (0.55, "binary", True),
        (0.45, "binary", True),
        (0.5499, "binary", False),
        (0.50, "binary", False),
    ],
)
def test_the_floor_is_two_sided_and_measured_from_no_effect(
    effect, kind, expected
):
    from vcg_biomarker.mimic.broad_axis_stage import meets_effect_floor

    assert meets_effect_floor(effect, kind) is expected


@pytest.mark.parametrize("effect", [float("nan"), float("inf"), None, "n/a"])
def test_an_unusable_effect_never_clears_the_floor(effect):
    from vcg_biomarker.mimic.broad_axis_stage import meets_effect_floor

    assert meets_effect_floor(effect, "continuous") is False
    assert meets_effect_floor(effect, "binary") is False
