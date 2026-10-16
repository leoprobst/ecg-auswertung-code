"""A screen that cannot recover the textbook has no informative nulls.

Nothing in the pipeline tested the pipeline.  Millions of associations were
screened with a controlled false-discovery rate, but "no association between
marker X and phenotype Y" was reported by an instrument that had never been
shown to detect an association of that size -- and the false-discovery rate
bounds chance, not confounding by acuity, which is the dominant error in an
intensive-care cohort.

These tests pin the two properties that make the controls worth declaring: a
known relationship is recognised only when it appears with the right direction
and a stated minimum size, and a control that cannot be run is reported as such
rather than quietly passing.  The heart-rate control is the live example --
``heart_rate`` carries the role ``ecg_context_comparator`` and is written by no
panel, so the best-established relationship in electrocardiography is currently
uncheckable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.prespecified_controls import (
    EFFECT_AUC,
    EFFECT_RHO,
    NEGATIVE_CONTROLS,
    POSITIVE_CONTROLS,
    VERDICT_FOUND,
    VERDICT_MARKER_MISSING,
    VERDICT_TARGET_MISSING,
    VERDICT_TOO_WEAK,
    VERDICT_WRONG_DIRECTION,
    control_summary,
    evaluate_negative_controls,
    evaluate_positive_controls,
)


def _results(rows):
    return pd.DataFrame(rows, columns=["target_id", "feature_id", "effect"])


def test_every_declared_control_names_a_mechanism_and_a_source():
    for control in POSITIVE_CONTROLS:
        assert control.mechanism.strip(), control.control_id
        assert control.source.strip(), control.control_id
        assert control.expected_direction in (1, -1), control.control_id
        assert control.marker_ids, control.control_id
        assert control.effect_kind in (EFFECT_RHO, EFFECT_AUC)


def test_potassium_raising_the_t_wave_is_recognised():
    results = _results(
        [
            ("lab_potassium", "t_vector_magnitude", 0.24),
            ("lab_potassium", "qrs_duration_ms", 0.15),
        ]
    )
    table = evaluate_positive_controls(results, target_column="target_id")
    row = table.set_index("control_id").loc["hyperkalaemia_tall_t_waves"]
    assert row["verdict"] == VERDICT_FOUND
    assert row["best_marker"] == "t_vector_magnitude"


def test_the_wrong_direction_is_not_a_recovery():
    # Potassium rising while the T wave falls contradicts the mechanism; the
    # screen finding "something" is not the same as finding the right thing.
    results = _results([("lab_potassium", "t_vector_magnitude", -0.24)])
    table = evaluate_positive_controls(results, target_column="target_id")
    row = table.set_index("control_id").loc["hyperkalaemia_tall_t_waves"]
    assert row["verdict"] == VERDICT_WRONG_DIRECTION
    assert row["observed_direction"] == -1


def test_a_correct_but_tiny_effect_is_reported_as_too_weak():
    results = _results([("lab_potassium", "t_vector_magnitude", 0.03)])
    table = evaluate_positive_controls(results, target_column="target_id")
    row = table.set_index("control_id").loc["hyperkalaemia_tall_t_waves"]
    assert row["verdict"] == VERDICT_TOO_WEAK


def test_calcium_shortening_the_qt_is_recognised_in_the_negative_direction():
    results = _results([("lab_total_calcium", "qtc_fridericia_ms", -0.18)])
    table = evaluate_positive_controls(results, target_column="target_id")
    row = table.set_index("control_id").loc["calcium_qt_shortening"]
    assert row["verdict"] == VERDICT_FOUND


def test_an_auc_control_uses_distance_from_one_half():
    # Digoxin deepens ST depression, so the marker falls: an AUC below 0.5.
    results = _results(
        [("medication_digoxin", "classical_st_depression_contiguous", 0.40)]
    )
    table = evaluate_positive_controls(results, target_column="target_id")
    row = table.set_index("control_id").loc["digoxin_st_depression"]
    assert row["verdict"] == VERDICT_FOUND
    assert row["observed_direction"] == -1


def test_an_absent_target_and_an_absent_marker_are_told_apart():
    table = evaluate_positive_controls(
        _results([("lab_potassium", "some_other_marker", 0.9)]),
        target_column="target_id",
    ).set_index("control_id")
    assert table.loc["hyperkalaemia_tall_t_waves", "verdict"] == (
        VERDICT_MARKER_MISSING
    )
    assert table.loc["calcium_qt_shortening", "verdict"] == VERDICT_TARGET_MISSING


def test_the_heart_rate_control_is_declared_even_though_it_cannot_run():
    # Declaring it is what makes the gap visible; a silently omitted control
    # would leave the pipeline looking complete.
    ids = {control.control_id for control in POSITIVE_CONTROLS}
    assert "heart_rate_qt_dependence" in ids
    table = evaluate_positive_controls(
        _results([("lab_potassium", "t_vector_magnitude", 0.2)]),
        target_column="target_id",
    ).set_index("control_id")
    assert table.loc["heart_rate_qt_dependence", "verdict"] == (
        VERDICT_TARGET_MISSING
    )


def test_negative_controls_report_a_rate_not_a_pass_per_target():
    targets = NEGATIVE_CONTROLS[0].target_ids
    rows = [(target, "qrs_loop_tortuosity", 0.51) for target in targets]
    rows[0] = (targets[0], "qrs_loop_tortuosity", 0.72)
    table = evaluate_negative_controls(
        _results(rows), target_column="target_id"
    )
    row = table.iloc[0]
    assert row["targets_present"] == len(targets)
    assert row["targets_with_association"] == 1
    assert row["hit_rate"] == pytest.approx(1 / len(targets))
    assert row["within_expectation"] in (True, False)


def test_a_screen_full_of_implausible_hits_fails_its_negative_controls():
    targets = NEGATIVE_CONTROLS[0].target_ids
    rows = [(target, "qrs_loop_tortuosity", 0.80) for target in targets]
    table = evaluate_negative_controls(
        _results(rows), target_column="target_id"
    )
    assert table.iloc[0]["hit_rate"] == 1.0
    assert not bool(table.iloc[0]["within_expectation"])


def test_nulls_are_only_interpretable_when_every_runnable_control_passed():
    good = _results(
        [
            ("lab_potassium", "t_vector_magnitude", 0.24),
            ("lab_potassium", "qrs_duration_ms", 0.15),
            ("lab_total_calcium", "qtc_fridericia_ms", -0.18),
            ("lab_ionized_calcium", "qt_interval_ms", -0.14),
            (
                "medication_digoxin",
                "classical_st_depression_contiguous",
                0.40,
            ),
        ]
    )
    positive = evaluate_positive_controls(good, target_column="target_id")
    negative = evaluate_negative_controls(
        _results(
            [
                (target, "qrs_loop_tortuosity", 0.51)
                for target in NEGATIVE_CONTROLS[0].target_ids
            ]
        ),
        target_column="target_id",
    )
    summary = control_summary(positive, negative)
    assert summary["positive_controls_recovered"] == 5
    assert summary["positive_controls_not_evaluable"] == 1
    assert summary["negative_findings_interpretable"] is True

    # One control flipping direction is enough to withdraw the claim.
    broken = good.copy()
    broken.loc[0, "effect"] = -0.24
    summary = control_summary(
        evaluate_positive_controls(broken, target_column="target_id"), negative
    )
    assert summary["negative_findings_interpretable"] is False


def test_an_empty_results_table_does_not_pretend_the_controls_passed():
    positive = evaluate_positive_controls(
        pd.DataFrame(), target_column="target_id"
    )
    negative = evaluate_negative_controls(
        pd.DataFrame(), target_column="target_id"
    )
    assert len(positive) == len(POSITIVE_CONTROLS)
    assert set(positive["verdict"]) == {VERDICT_TARGET_MISSING}
    summary = control_summary(positive, negative)
    assert summary["negative_findings_interpretable"] is False
