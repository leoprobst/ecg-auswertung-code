"""Six stages produced results; almost none of them had a picture.

Tables were added when the stages were wired, and tables are where a reader
stops. The additions tested here are the ones that carry something a table
cannot: a curve that shows a non-monotone relationship a hazard ratio hides, a
pair of bars whose *gap* is the finding, and a correction that only means
anything when it spans every arm at once.

Three things this file holds in place:

* the segment-availability comparison, because the class of defect it shows --
  a marker missing more often in the patients who have the condition -- was
  invisible in every view this project had;
* the survival curves, cut at tertiles fixed on training, because cutting each
  split at its own boundaries pulls the curves apart for any marker at all;
* the multiplicity correction spanning the arms rather than sitting inside
  them, because the selection happens between arms.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.segment_availability import (
    SEGMENT_SENTINELS,
    AvailabilityPolicy,
    availability_query,
    availability_table,
    count_by_state,
    worst_differential,
)
from vcg_biomarker.mimic.survival import (
    SURVIVAL_GROUP_COUNT,
    SURVIVAL_GROUP_SPLIT,
    tertile_curves,
)
from vcg_biomarker.mimic.targeted_multiplicity import (
    TARGETED_ARMS,
    collect_arm_results,
    correct_across_arms,
    multiplicity_summary,
)


# --- Segment availability -----------------------------------------------------


def _counts(positive, negative, segments):
    rows = []
    for state, (total, available) in (
        ("positive", positive),
        ("negative", negative),
    ):
        row = {"endpoint_state": state, "n_ecgs": total}
        for segment in segments:
            row[f"n_{segment}"] = available[segment]
        rows.append(row)
    return pd.DataFrame(rows)


def test_a_marker_missing_equally_in_both_groups_is_not_flagged():
    """Missing at random costs power and nothing else."""

    counts = _counts((1000, {"p": 400}), (1000, {"p": 400}), ("p",))
    table = availability_table(counts, "endpoint", sentinels={"p": "marker"})
    assert float(table.iloc[0]["gap_points"]) == pytest.approx(0.0)
    assert bool(table.iloc[0]["differential"]) is False


def test_a_marker_missing_more_often_in_the_cases_is_flagged():
    """Measured for real: 12,7 % against 38,0 % for digoxin exposure."""

    counts = _counts((1000, {"p": 127}), (1000, {"p": 380}), ("p",))
    row = availability_table(
        counts, "digoxin", sentinels={"p": "marker"}
    ).iloc[0]
    assert float(row["gap_points"]) == pytest.approx(-25.3, abs=0.1)
    assert bool(row["differential"]) is True


def test_the_gap_is_signed_towards_the_damaging_direction():
    """Negative must mean 'missing more often in the cases'."""

    worse_in_cases = availability_table(
        _counts((1000, {"p": 100}), (1000, {"p": 500}), ("p",)),
        "e", sentinels={"p": "m"},
    ).iloc[0]
    worse_in_controls = availability_table(
        _counts((1000, {"p": 500}), (1000, {"p": 100}), ("p",)),
        "e", sentinels={"p": "m"},
    ).iloc[0]
    assert float(worse_in_cases["gap_points"]) < 0
    assert float(worse_in_controls["gap_points"]) > 0


def test_a_group_too_small_to_read_yields_nothing():
    counts = _counts((20, {"p": 5}), (1000, {"p": 400}), ("p",))
    assert availability_table(counts, "e", sentinels={"p": "m"}).empty


def test_a_state_that_never_occurs_yields_nothing():
    counts = pd.DataFrame(
        [{"endpoint_state": "positive", "n_ecgs": 500, "n_p": 100}]
    )
    assert availability_table(counts, "e", sentinels={"p": "m"}).empty


def test_the_worst_differential_is_the_largest_absolute_gap():
    table = pd.concat(
        [
            availability_table(
                _counts((1000, {"p": 900}), (1000, {"p": 880}), ("p",)),
                "mild", sentinels={"p": "m"},
            ),
            availability_table(
                _counts((1000, {"p": 100}), (1000, {"p": 500}), ("p",)),
                "severe", sentinels={"p": "m"},
            ),
        ],
        ignore_index=True,
    )
    assert worst_differential(table)["endpoint_id"] == "severe"


def test_the_query_reads_the_cohort_once_rather_than_once_per_endpoint():
    """The first version issued one scan per endpoint: 70 scans for five
    columns of answer."""

    signature = inspect.signature(availability_query)
    assert "endpoint_id" not in signature.parameters
    sql = availability_query(
        "master.parquet", "read_parquet('f/*.parquet')",
        sentinels={"p": "p_marker", "qrs": "qrs_marker"},
    )
    assert "IS NOT NULL) AS has_p" in sql
    assert "IS NOT NULL) AS has_qrs" in sql
    assert "extraction_status = 'ok'" in sql


def test_the_locked_test_split_is_never_read():
    sql = availability_query(
        "master.parquet", "read_parquet('f/*.parquet')",
        sentinels={"p": "p_marker"},
    )
    assert "'train', 'validation'" in sql
    assert "locked_test" not in sql


def test_counting_by_state_matches_the_table_it_feeds():
    availability = pd.DataFrame({"has_p": [True, True, False, False, True]})
    states = pd.Series(
        ["positive", "positive", "positive", "negative", "negative"]
    )
    counts = count_by_state(availability, states, sentinels={"p": "m"})
    indexed = counts.set_index("endpoint_state")
    assert int(indexed.loc["positive", "n_ecgs"]) == 3
    assert int(indexed.loc["positive", "n_p"]) == 2
    assert int(indexed.loc["negative", "n_p"]) == 1


def test_the_tq_window_is_one_of_the_segments_watched():
    """It is the answer to what this view exposes."""

    assert "tq" in SEGMENT_SENTINELS
    assert "p" in SEGMENT_SENTINELS
    assert "qrs" in SEGMENT_SENTINELS


# --- Survival curves ----------------------------------------------------------


def _survival_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    marker = rng.normal(size=n)
    hazard = 0.002 * np.exp(0.8 * marker)
    time = rng.exponential(1.0 / hazard)
    censor = rng.exponential(500.0, n)
    return pd.DataFrame(
        {
            "marker": marker,
            "time_days": np.minimum(time, censor),
            "event": (time <= censor).astype(int),
            "split": rng.choice(["train", "validation"], n),
        }
    )


def test_the_curves_are_cut_into_the_declared_number_of_groups():
    curves = tertile_curves(_survival_frame(), "marker", minimum_per_group=50)
    assert curves["group"].nunique() == SURVIVAL_GROUP_COUNT


def test_risk_rises_with_the_marker_when_the_hazard_does():
    curves = tertile_curves(_survival_frame(), "marker", minimum_per_group=50)
    final = (
        curves.sort_values("time_days")
        .groupby("group")
        .tail(1)
        .sort_values("group")["cumulative_incidence"]
        .tolist()
    )
    assert final[0] < final[-1], "das oberste Terzil muss mehr Risiko tragen"


def test_the_boundaries_come_from_training_not_from_each_split():
    """Cutting each split at its own tertiles pulls the curves apart for any
    marker at all."""

    assert SURVIVAL_GROUP_SPLIT == "train"
    source = inspect.getsource(tertile_curves)
    assert "boundary_split" in source
    assert "reference" in source


def test_a_marker_with_a_mass_point_yields_no_curves():
    """Identical boundaries would not produce three groups."""

    frame = _survival_frame()
    frame["marker"] = 1.0
    assert tertile_curves(frame, "marker", minimum_per_group=50).empty


def test_a_cohort_too_small_to_split_yields_no_curves():
    assert tertile_curves(_survival_frame(n=100), "marker").empty


def test_a_missing_marker_yields_no_curves():
    assert tertile_curves(_survival_frame(), "gibt_es_nicht").empty


def test_the_curves_stop_at_the_horizon():
    curves = tertile_curves(
        _survival_frame(), "marker", horizon_days=180.0, minimum_per_group=50
    )
    assert float(curves["time_days"].max()) <= 180.0


def test_every_curve_carries_its_own_group_size():
    curves = tertile_curves(_survival_frame(), "marker", minimum_per_group=50)
    for group in curves["group"].unique():
        block = curves.loc[curves["group"].eq(group)]
        assert int(block["n"].iloc[0]) > 0
        assert block["n"].nunique() == 1


def test_repeated_ecgs_do_not_duplicate_people_in_survival_curves():
    first = _survival_frame(n=2400, seed=17)
    first["subject_id"] = np.arange(len(first))
    first["ecg_rank"] = 1
    repeated = first.copy()
    repeated["ecg_rank"] = 2
    repeated["marker"] = repeated["marker"] + 100.0
    doubled = pd.concat([first, repeated], ignore_index=True)

    curves = tertile_curves(doubled, "marker", minimum_per_group=50)
    group_sizes = curves.groupby("group")["n"].first()
    expected = int(first["split"].ne(SURVIVAL_GROUP_SPLIT).sum())

    assert int(group_sizes.sum()) == expected


# --- Multiplicity across the arms ---------------------------------------------


def _arm_results():
    return pd.DataFrame(
        {
            "arm": ["a"] * 5 + ["b"] * 5,
            "feature_id": [f"m{i}" for i in range(10)],
            "context": "x",
            "effect": np.linspace(0.1, 1.0, 10),
            "p_value": [1e-9, 1e-6, 0.001, 0.02, 0.4] * 2,
        }
    )


def test_the_correction_spans_every_arm_as_one_family():
    """Correcting inside each arm leaves the choice between arms uncorrected,
    and that is where the selection happens."""

    corrected = correct_across_arms(_arm_results())
    assert len(corrected) == 10
    assert corrected["q_value"].notna().any()
    # Ten tests in one family: the largest p-value cannot keep its own value
    # as a q-value.
    worst = corrected.sort_values("p_value").iloc[-1]
    assert float(worst["q_value"]) >= float(worst["p_value"])


def test_a_clearly_null_test_does_not_survive_the_correction():
    corrected = correct_across_arms(_arm_results())
    weakest = corrected.sort_values("p_value").iloc[-1]
    assert bool(weakest["fdr_significant"]) is False


def test_an_empty_collection_yields_an_empty_correction():
    assert correct_across_arms(pd.DataFrame()).empty
    assert multiplicity_summary(pd.DataFrame())["n_tests"] == 0


def test_the_summary_carries_the_noise_floor_beside_the_q_values():
    """A q-value asks whether a p-value is small for the number of tests. The
    noise floor asks whether an effect is large for a cohort where almost
    everything tracks being ill."""

    summary = multiplicity_summary(
        correct_across_arms(_arm_results()), noise_floor=0.116
    )
    assert summary["confounding_noise_floor"] == pytest.approx(0.116)
    assert "Konfundierung" in summary["noise_floor_note"]


def test_only_arms_that_report_uncertainty_take_part():
    """Correcting an AUC that never had a p-value would mean inventing one."""

    for spec in TARGETED_ARMS.values():
        assert spec["p_value"]
    assert "occlusion" not in TARGETED_ARMS
    assert "voltage_mass" not in TARGETED_ARMS


def test_collecting_from_a_run_without_the_arms_yields_nothing(tmp_path):
    assert collect_arm_results(tmp_path).empty
