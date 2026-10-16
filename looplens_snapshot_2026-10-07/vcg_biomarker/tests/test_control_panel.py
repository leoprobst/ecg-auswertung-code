"""The screen tests two million associations; something must test the screen.

Measured against the real screening results on 2026-08-30:

===========================  ========  ==========
Kontrolle                    AUC       Befund
===========================  ========  ==========
hyperkalemia_peaked_t           0.543  recovered
hyperkalemia_qrs_widening       0.537  recovered
hypocalcemia_qt                 0.519  recovered
qt_prolonging_drug_qt           0.517  recovered
hypokalemia_tpeak_tend          0.510  direction only
**digoxin_p_duration**          0.616  **false positive**
===========================  ========  ==========

Two lessons are built into the code and pinned here.

**A positive control is judged by direction and significance, never by size.**
The first draft required an AUC margin of 0.05 and declared a working pipeline
broken: QTc is 3.4 ms higher under QT-prolonging drugs (436.4 vs 433.0,
p = 2.4e-05, n = 2,912 vs 6,727) against a population SD of 44.5 ms. Correct,
highly significant, and far too small for any threshold on magnitude.

**A negative control is judged by size, and its size is the point.** At these
sample sizes confounding is always significant, so the question is how large it
gets. Digoxin exposure against P-wave duration -- a pair with no mechanism --
reached AUC 0.616, larger than every mechanism-backed control in the same run.
Below that figure an association cannot be told apart from being ill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.control_panel import (
    CONTROL_ALPHA,
    CONTROL_PANEL,
    NEGATIVE_CONTROLS,
    POSITIVE_CONTROLS,
    control_panel_verdict,
    evaluate_controls,
    render_control_panel,
)


def _results(rows):
    return pd.DataFrame(
        rows, columns=["endpoint_id", "feature_id", "roc_auc", "p_value"]
    )


def _evaluate(rows, **kwargs):
    return evaluate_controls(
        _results(rows), effect_column="roc_auc", kind_column=None, **kwargs
    )


def test_every_control_names_an_endpoint_a_marker_and_a_mechanism():
    for control in CONTROL_PANEL:
        assert control.endpoint_id
        assert control.feature_id
        assert len(control.mechanism) > 40, control.control_id
        assert control.kind in {"positive", "negative"}


def test_every_control_resolves_against_the_real_catalog():
    """A control pointing at a marker that does not exist tests nothing."""

    from vcg_biomarker.mimic.label_registry import default_label_definitions
    from vcg_biomarker.mimic.panels import default_feature_panels

    labels = {x.label_id for x in default_label_definitions()}
    markers = set()
    for panel in default_feature_panels():
        markers |= {str(f) for f in panel.feature_ids}
    for control in CONTROL_PANEL:
        assert control.endpoint_id in labels, control.control_id
        assert control.feature_id in markers, control.control_id


def test_every_positive_control_carries_a_reference():
    for control in POSITIVE_CONTROLS:
        assert control.reference, control.control_id


def test_a_small_but_real_effect_counts_as_recovered():
    """The defect that made a working pipeline look broken."""

    table = _evaluate(
        [("qt_prolonging_drug_exposure", "qtc_fridericia_ms", 0.527, 2.4e-05)]
    )
    row = table.loc[table["control_id"].eq("qt_prolonging_drug_qt")].iloc[0]
    assert row["status"] == "recovered"
    assert bool(row["effect_above_floor"]) is False, (
        "genau deshalb darf die Groesse nicht entscheiden"
    )


def test_a_significant_effect_in_the_wrong_direction_is_not_recovered():
    table = _evaluate(
        [("hypocalcemia_near_ecg", "qtc_fridericia_ms", 0.42, 1e-20)]
    )
    row = table.loc[table["control_id"].eq("hypocalcemia_qt")].iloc[0]
    assert row["status"] == "wrong_direction"
    assert bool(row["meets_expectation"]) is False


def test_the_right_direction_without_significance_is_not_recovered():
    table = _evaluate(
        [("severe_hypokalemia_near_ecg", "tpeak_tend_ms", 0.510, 0.4)]
    )
    row = table.loc[table["control_id"].eq("hypokalemia_tpeak_tend")].iloc[0]
    assert row["status"] == "direction_only"


def test_a_result_without_uncertainty_is_not_silently_accepted():
    table = evaluate_controls(
        pd.DataFrame(
            [("hypocalcemia_near_ecg", "qtc_fridericia_ms", 0.9)],
            columns=["endpoint_id", "feature_id", "roc_auc"],
        ),
        effect_column="roc_auc",
        kind_column=None,
        p_value_column=None,
    )
    row = table.loc[table["control_id"].eq("hypocalcemia_qt")].iloc[0]
    assert row["status"] == "no_uncertainty_reported"


def test_a_control_that_was_never_screened_says_so():
    table = _evaluate([])
    assert set(table["status"]) == {"not_tested"}
    assert not table["found"].any()


def test_a_negative_control_above_the_floor_is_a_false_positive():
    table = _evaluate(
        [("digoxin_exposure_near_ecg", "p_duration_ms", 0.616, 1e-30)]
    )
    row = table.loc[table["control_id"].eq("digoxin_p_duration")].iloc[0]
    assert row["status"] == "false_positive"


def test_a_negative_control_below_the_floor_is_silent():
    table = _evaluate(
        [("qt_prolonging_drug_exposure", "p_loop_width_major", 0.506, 1e-8)]
    )
    row = table.loc[table["control_id"].eq("qt_drug_p_width")].iloc[0]
    assert row["status"] == "silent_as_expected"
    assert bool(row["meets_expectation"]) is True


def test_the_noise_floor_is_the_largest_effect_without_a_mechanism():
    """The most useful number the panel produces."""

    table = _evaluate(
        [
            ("digoxin_exposure_near_ecg", "p_duration_ms", 0.616, 1e-30),
            ("hypocalcemia_near_ecg", "p_loop_width_major", 0.542, 1e-9),
        ]
    )
    verdict = control_panel_verdict(table)
    assert verdict["confounding_noise_floor"] == pytest.approx(0.116, abs=1e-9)


def test_a_run_that_recovers_nothing_is_declared_unreadable():
    """Its null results are as uninformative as its hits."""

    table = _evaluate(
        [
            ("severe_hyperkalemia_near_ecg", "classical_peaked_t_wave_heuristic", 0.50, 0.9),
            ("hypocalcemia_near_ecg", "qtc_fridericia_ms", 0.50, 0.9),
        ]
    )
    verdict = control_panel_verdict(table)
    assert verdict["trustworthy"] is False
    assert "nicht als Abwesenheit" in verdict["reason"]


def test_a_run_that_recovers_the_majority_is_declared_readable():
    table = _evaluate(
        [
            ("severe_hyperkalemia_near_ecg", "classical_peaked_t_wave_heuristic", 0.543, 1e-12),
            ("severe_hyperkalemia_near_ecg", "qrs_duration_ms", 0.537, 1e-10),
            ("hypocalcemia_near_ecg", "qtc_fridericia_ms", 0.519, 1e-6),
        ]
    )
    verdict = control_panel_verdict(table)
    assert verdict["trustworthy"] is True
    assert verdict["n_positive_recovered"] == 3


def test_the_rendered_panel_names_the_noise_floor():
    table = _evaluate(
        [("digoxin_exposure_near_ecg", "p_duration_ms", 0.616, 1e-30)]
    )
    text = render_control_panel(table, control_panel_verdict(table))
    assert "Rauschgrenze" in text
    assert "0.116" in text


def test_the_alpha_is_prespecified():
    assert CONTROL_ALPHA == 0.01


def test_positive_and_negative_controls_are_both_declared():
    assert len(POSITIVE_CONTROLS) >= 5
    assert len(NEGATIVE_CONTROLS) >= 3
    assert {c.kind for c in POSITIVE_CONTROLS} == {"positive"}
    assert {c.kind for c in NEGATIVE_CONTROLS} == {"negative"}


def test_no_control_appears_twice():
    ids = [c.control_id for c in CONTROL_PANEL]
    assert len(ids) == len(set(ids))
    pairs = [(c.endpoint_id, c.feature_id) for c in CONTROL_PANEL]
    assert len(pairs) == len(set(pairs))


def test_an_empty_verdict_does_not_claim_trustworthiness():
    verdict = control_panel_verdict(pd.DataFrame())
    assert verdict["trustworthy"] is False


# --- The split the panel reports on ------------------------------------------
#
# The defect: a screening table holds each endpoint-marker pair twice, once for
# training and once for validation, and ``evaluate_controls`` took
# ``match.iloc[0]`` -- the training row. On the development run it reported
# ``hyperkalemia_peaked_t`` at AUC 0,568 with no column saying where that came
# from, while the validation value was 0,543. A reader would have taken the
# certificate of the run's honesty for a held-out result when it was measured
# on the data the thresholds were chosen on.


def _split_results(rows):
    frame = pd.DataFrame(
        rows,
        columns=["endpoint_id", "feature_id", "roc_auc", "p_value",
                 "evaluation_split"],
    )
    return frame


def _both_splits(control, train_auc, validation_auc, p_value=1e-9):
    return _split_results(
        [
            (control.endpoint_id, control.feature_id, train_auc, p_value, "train"),
            (
                control.endpoint_id,
                control.feature_id,
                validation_auc,
                p_value,
                "validation",
            ),
        ]
    )


def test_each_control_is_reported_once_per_split():
    control = POSITIVE_CONTROLS[0]
    table = evaluate_controls(
        _both_splits(control, 0.568, 0.543),
        effect_column="roc_auc",
        kind_column=None,
    )
    rows = table.loc[table["control_id"].eq(control.control_id)]
    assert len(rows) == 2
    assert set(rows["evaluation_split"]) == {"train", "validation"}


def test_the_reported_effect_says_which_split_it_came_from():
    """The column that was missing entirely."""

    control = POSITIVE_CONTROLS[0]
    table = evaluate_controls(
        _both_splits(control, 0.568, 0.543),
        effect_column="roc_auc",
        kind_column=None,
    )
    assert "evaluation_split" in table.columns
    held = table.loc[
        table["control_id"].eq(control.control_id)
        & table["evaluation_split"].eq("validation")
    ].iloc[0]
    assert float(held["observed_effect"]) == pytest.approx(0.543)


def test_the_verdict_is_taken_on_validation_not_on_training():
    """A control that only works on training must not count as recovered."""

    control = POSITIVE_CONTROLS[0]
    table = evaluate_controls(
        _split_results(
            [
                (control.endpoint_id, control.feature_id, 0.70, 1e-30, "train"),
                # Held out: right direction, no significance.
                (control.endpoint_id, control.feature_id, 0.51, 0.4, "validation"),
            ]
        ),
        effect_column="roc_auc",
        kind_column=None,
    )
    verdict = control_panel_verdict(table)
    assert verdict["verdict_split"] == "validation"
    assert verdict["n_positive_recovered"] == 0


def test_a_table_without_splits_is_still_judged():
    """Sources other than the screen carry no split column."""

    control = POSITIVE_CONTROLS[0]
    table = _evaluate([(control.endpoint_id, control.feature_id, 0.70, 1e-30)])
    verdict = control_panel_verdict(table)
    assert verdict["n_positive_recovered"] == 1


def test_a_table_whose_only_split_is_training_refuses_a_verdict():
    """Judging it would be exactly the defect, one level quieter."""

    control = POSITIVE_CONTROLS[0]
    table = evaluate_controls(
        _split_results(
            [(control.endpoint_id, control.feature_id, 0.70, 1e-30, "train")]
        ),
        effect_column="roc_auc",
        kind_column=None,
    )
    verdict = control_panel_verdict(table)
    assert verdict["trustworthy"] is False
    assert "validation" in verdict["reason"]


def test_the_shrinkage_between_splits_is_reported():
    from vcg_biomarker.mimic.control_panel import effect_shrinkage

    control = POSITIVE_CONTROLS[0]
    table = evaluate_controls(
        _both_splits(control, 0.70, 0.60),
        effect_column="roc_auc",
        kind_column=None,
    )
    shrinkage = effect_shrinkage(table)
    row = shrinkage.loc[shrinkage["control_id"].eq(control.control_id)].iloc[0]
    # Margin 0,20 on training, 0,10 held out: half of it is gone.
    assert float(row["shrinkage"]) == pytest.approx(0.5, abs=1e-9)
    assert bool(row["shrinkage_defined"]) is True


def test_shrinkage_is_undefined_when_there_was_no_effect_to_lose():
    """``qt_drug_p_width`` scored 0,5002 on training and the ratio came out at
    -32,02 -- a number that says nothing except that the denominator was
    noise."""

    from vcg_biomarker.mimic.control_panel import effect_shrinkage

    control = POSITIVE_CONTROLS[0]
    table = evaluate_controls(
        _both_splits(control, 0.5002, 0.5059),
        effect_column="roc_auc",
        kind_column=None,
    )
    row = effect_shrinkage(table).iloc[0]
    assert bool(row["shrinkage_defined"]) is False
    assert np.isnan(float(row["shrinkage"]))


def test_the_verdict_counts_how_many_controls_clear_the_effect_floor():
    """Measured on the development run: 1 of 10 on validation."""

    control = POSITIVE_CONTROLS[0]
    table = evaluate_controls(
        _both_splits(control, 0.70, 0.70),
        effect_column="roc_auc",
        kind_column=None,
    )
    assert control_panel_verdict(table)["n_above_effect_floor"] == 1
