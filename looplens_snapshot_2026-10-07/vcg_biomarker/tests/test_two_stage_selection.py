"""A discovery must be able to become a claim, without becoming a fishing trip.

The two halves of this project were never joined: 21,099 replicated hits, 493
promotion candidates all still marked draft, and exactly 12 markers that ever
reach confirmatory inference. ``PRIMARY_ADJUSTED_MARKERS`` pairs one
hand-curated marker per endpoint -- the right instinct, and the reason 0.7 % of
the catalog can ever support a claim.

This adds a second arm and replaces nothing. Its whole validity rests on one
property, tested here from several directions: **the selection sees the
training split and nothing else.**

Two defects found while building it, both pinned below:

* ``heart_rate`` and ``heart_rate_bpm`` were ranked first for acute myocardial
  infarction. They carry the role ``ecg_context_comparator`` -- they are what a
  finding is adjusted *for*.
* ``conduction_detour_index`` and ``qrs_loop_tortuosity`` were selected as two
  markers, sharing an AUC, a p-value to seven digits, an n and a missingness
  across all 63 endpoints -- while their values correlate at rho = 0.34 and
  their formulas differ. Two confirmation slots for one measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.two_stage_selection import (
    TWO_STAGE_TOP_K,
    TwoStagePolicy,
    ValidationLeak,
    assert_train_only,
    confirmation_plan,
    select_from_train,
    selection_summary,
)


def _screen(rows, split="train"):
    frame = pd.DataFrame(
        rows, columns=["endpoint_id", "feature_id", "roc_auc", "p_value"]
    )
    frame["evaluation_split"] = split
    frame["n_total"] = 1000
    return frame


def _many(endpoint, features, base=0.70):
    return [
        (endpoint, name, base - 0.01 * index, 1e-10)
        for index, name in enumerate(features)
    ]


REAL = [
    "conduction_detour_index",
    "beat_sharpness_index",
    "qrs_loop_horizontal_xz_winding_number",
    "electric_diastole_ms",
    "spatial_magnitude_p95",
]


# --- The property everything else rests on -----------------------------------


def test_validation_rows_are_refused_not_filtered():
    """Silently dropping them would let a caller believe it got what it asked."""

    frame = _screen(_many("e1", REAL), split="validation")
    with pytest.raises(ValidationLeak):
        select_from_train(frame)


def test_a_mixed_frame_is_refused():
    train = _screen(_many("e1", REAL))
    held = _screen(_many("e1", REAL), split="validation")
    with pytest.raises(ValidationLeak):
        select_from_train(pd.concat([train, held], ignore_index=True))


def test_a_frame_without_a_split_column_is_allowed():
    """Callers that already sliced their data must not be blocked."""

    frame = _screen(_many("e1", REAL)).drop(columns=["evaluation_split"])
    assert not select_from_train(frame).empty


def test_assert_train_only_names_the_intruder():
    frame = _screen(_many("e1", REAL), split="locked_test")
    with pytest.raises(ValidationLeak, match="locked_test"):
        assert_train_only(frame)


# --- The selection rule ------------------------------------------------------


def test_at_most_k_markers_per_endpoint():
    frame = _screen(_many("e1", REAL))
    selection = select_from_train(frame)
    assert len(selection) == TWO_STAGE_TOP_K
    assert list(selection["selection_rank"]) == [1, 2, 3]


def test_the_ranking_follows_effect_size():
    frame = _screen(_many("e1", REAL))
    selection = select_from_train(frame)
    assert selection.iloc[0]["feature_id"] == REAL[0]
    assert (
        selection["train_effect"].is_monotonic_decreasing
    ), "die Rangfolge muss der Effektstaerke folgen"


def test_a_marker_below_the_effect_floor_is_not_selected():
    frame = _screen([("e1", REAL[0], 0.52, 1e-30)])
    assert select_from_train(frame).empty


def test_a_marker_without_train_significance_is_not_selected():
    frame = _screen([("e1", REAL[0], 0.70, 0.2)])
    assert select_from_train(frame).empty


def test_a_context_comparator_is_never_selected():
    """They were ranked first for acute myocardial infarction."""

    frame = _screen(
        [
            ("e1", "heart_rate", 0.80, 1e-30),
            ("e1", "heart_rate_bpm", 0.79, 1e-30),
            ("e1", REAL[0], 0.70, 1e-30),
        ]
    )
    chosen = set(select_from_train(frame)["feature_id"])
    assert "heart_rate" not in chosen
    assert "heart_rate_bpm" not in chosen
    assert REAL[0] in chosen


def test_two_markers_with_identical_statistics_take_one_slot():
    """Bit-identical statistics mean one measurement, not two markers."""

    frame = _screen(
        [
            ("e1", "conduction_detour_index", 0.688279, 5.409323e-99),
            ("e1", "qrs_loop_tortuosity", 0.688279, 5.409323e-99),
            ("e1", "beat_sharpness_index", 0.684112, 1e-90),
        ]
    )
    chosen = list(select_from_train(frame)["feature_id"])
    assert len(chosen) == 2
    assert "beat_sharpness_index" in chosen


def test_one_marker_per_redundancy_cluster():
    frame = _screen(_many("e1", REAL))
    redundancy = pd.DataFrame(
        {
            "feature_id": REAL,
            "cluster_id": ["C1", "C1", "C1", "C2", "C3"],
        }
    )
    chosen = list(select_from_train(frame, redundancy=redundancy)["feature_id"])
    assert chosen == [REAL[0], REAL[3], REAL[4]]


def test_deduplication_can_be_switched_off_deliberately():
    frame = _screen(_many("e1", REAL))
    redundancy = pd.DataFrame(
        {"feature_id": REAL, "cluster_id": ["C1"] * 5}
    )
    policy = TwoStagePolicy(deduplicate_by_cluster=False)
    chosen = select_from_train(frame, redundancy=redundancy, policy=policy)
    assert len(chosen) == TWO_STAGE_TOP_K


def test_each_selection_records_how_many_markers_it_beat():
    """A reader has to be able to weigh the multiplicity behind a hit."""

    frame = _screen(_many("e1", REAL))
    selection = select_from_train(frame)
    assert set(selection["n_screened_for_endpoint"]) == {len(REAL)}


def test_agreement_with_the_curated_pairing_is_recorded():
    frame = _screen(_many("e1", REAL))
    selection = select_from_train(
        frame, prespecified={"e1": REAL[1]}
    )
    row = selection.loc[selection["feature_id"].eq(REAL[1])].iloc[0]
    assert bool(row["also_prespecified"]) is True
    assert int(selection["also_prespecified"].sum()) == 1


# --- Both arms stay separate -------------------------------------------------


def test_the_plan_labels_which_arm_each_pair_belongs_to():
    frame = _screen(_many("e1", REAL))
    selection = select_from_train(frame)
    plan = confirmation_plan(selection, prespecified={"e1": "svg_magnitude"})
    assert set(plan["arm"]) == {"two_stage_confirmatory", "prespecified_curated"}
    curated = plan.loc[plan["arm"].eq("prespecified_curated")]
    assert list(curated["feature_id"]) == ["svg_magnitude"]


def test_the_curated_arm_survives_an_empty_selection():
    """Adding an arm must never be able to remove the existing one."""

    plan = confirmation_plan(
        pd.DataFrame(), prespecified={"e1": "svg_magnitude", "e2": "qt_interval_ms"}
    )
    assert len(plan) == 2
    assert set(plan["arm"]) == {"prespecified_curated"}


def test_the_summary_counts_endpoints_the_curation_never_covered():
    frame = pd.concat(
        [_screen(_many("e1", REAL)), _screen(_many("e2", REAL))],
        ignore_index=True,
    )
    selection = select_from_train(frame, prespecified={"e1": "svg_magnitude"})
    summary = selection_summary(selection, {"e1": "svg_magnitude"})
    assert summary["n_endpoints"] == 2
    assert summary["n_endpoints_without_curation"] == 1
    assert summary["curated_endpoints"] == 1


def test_the_policy_is_recorded_in_full():
    payload = TwoStagePolicy().to_dict()
    for key in ("top_k", "train_alpha", "minimum_auc_margin", "version"):
        assert key in payload


def test_the_rule_is_written_into_every_row():
    frame = _screen(_many("e1", REAL))
    selection = select_from_train(frame)
    assert selection["selection_rule"].str.contains("top-3").all()


def test_an_empty_screen_yields_an_empty_selection():
    selection = select_from_train(pd.DataFrame())
    assert selection.empty
    assert selection_summary(selection)["n_selected"] == 0
