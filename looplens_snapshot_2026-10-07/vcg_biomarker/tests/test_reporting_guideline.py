"""What a report must contain, decided by measurement rather than by choice.

The screens say which markers separate which endpoints.  That is not a
guideline.  A guideline says what a report carries, in what unit, against what
reference, what change is large enough to mean anything, and what the number
may be used to claim.

These tests pin the two properties that make the set defensible: that a marker
enters by clearing four measured conditions rather than by being picked, and
that a marker failing one is listed **with** the condition it failed -- a
hand-chosen list cannot be audited, and one that silently omits its failures
cannot either.

They also pin the claim ceiling, which is the part a strong result must not be
able to override.  An endpoint the electrocardiogram itself defines is analytic
concordance at any area under the curve, and an endpoint whose reference
standard was obtained for two per cent of recordings cannot be confirmed as a
diagnostic-accuracy claim at all.

Crossing those two filters over the 72 endpoints leaves **one**: six are
verified widely enough, five of the six are read off the same tracing, and what
survives is 30-day mortality -- a prognosis rather than a diagnosis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.reporting_guideline import (
    MANDATORY_CONTEXT_FIELDS,
    MINIMUM_REPORTABLE_ICC,
    ReportingSetPolicy,
    build_reporting_algorithm,
    build_reporting_set,
    intraindividual_change_note,
    mandatory_context_table,
)


def _ranges(marker_id: str, estimable: int, total: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "marker_id": [marker_id] * total,
            "estimable": [True] * estimable + [False] * (total - estimable),
        }
    )


def _repeatability(marker_id: str, icc: float, coefficient: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "marker_id": [marker_id],
            "icc": [icc],
            "repeatability_coefficient": [coefficient],
            "smallest_detectable_change": [coefficient / np.sqrt(2.0)],
        }
    )


def _agreement(marker_id: str, agreement_class: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "marker_id": [marker_id],
            "agreement_class": [agreement_class],
            "bias": [11.6],
            "limit_of_agreement_low": [-26.5],
            "limit_of_agreement_high": [49.7],
        }
    )


def _screening(marker_id: str, auc: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_id": [marker_id],
            "endpoint_id": ["reduced_lvef_strict"],
            "evaluation_split": ["validation"],
            "roc_auc": [auc],
        }
    )


def _set(marker_id="spatial_qrs_t_angle_deg", **overrides):
    parts = {
        "reference_ranges": _ranges(marker_id, 10, 12),
        "repeatability": _repeatability(marker_id, 0.85, 12.0),
        "transform_agreement": _agreement(
            marker_id, "convertible_with_declared_error"
        ),
        "screening": _screening(marker_id, 0.71),
        "marker_units": {marker_id: "degrees"},
    }
    parts.update(overrides)
    return build_reporting_set(**parts)


def test_a_marker_clearing_all_four_conditions_is_included():
    table = _set()
    row = table.iloc[0]
    assert bool(row["included"])
    assert row["excluded_reason"] == ""
    assert row["unit"] == "degrees"
    assert row["evidence_source"] == "external_literature"
    assert row["transform_agreement_class"] == "convertible_with_declared_error"


def test_a_marker_without_a_reference_range_is_excluded_and_says_so():
    table = _set(reference_ranges=_ranges("spatial_qrs_t_angle_deg", 2, 12))
    row = table.iloc[0]
    assert not bool(row["included"])
    assert "no_reference_range_in_enough_strata" in row["excluded_reason"]


def test_a_marker_whose_change_cannot_be_interpreted_is_excluded():
    """Follow-up is most of what a report is for."""

    table = _set(
        repeatability=_repeatability("spatial_qrs_t_angle_deg", 0.40, 12.0)
    )
    row = table.iloc[0]
    assert not bool(row["included"])
    assert (
        "repeatability_below_reportable_reliability" in row["excluded_reason"]
    )


def test_an_unmeasured_transform_dependence_is_a_veto():
    """Not that it must be small -- that it must be known and printed."""

    table = _set(
        transform_agreement=_agreement("spatial_qrs_t_angle_deg", "not_assessable")
    )
    row = table.iloc[0]
    assert not bool(row["included"])
    assert "transform_dependence_not_measured" in row["excluded_reason"]


def test_a_strongly_transform_dependent_marker_may_still_be_reported():
    table = _set(
        transform_agreement=_agreement(
            "spatial_qrs_t_angle_deg", "not_interchangeable"
        )
    )
    row = table.iloc[0]
    assert bool(row["included"])
    assert row["transform_agreement_class"] == "not_interchangeable"


def test_a_marker_with_neither_an_anchor_nor_a_finding_is_excluded():
    table = build_reporting_set(
        reference_ranges=_ranges("qrs_loop_writhe_probe", 10, 12),
        repeatability=_repeatability("qrs_loop_writhe_probe", 0.85, 1.0),
        transform_agreement=_agreement("qrs_loop_writhe_probe", "convertible_with_declared_error"),
        screening=_screening("qrs_loop_writhe_probe", 0.52),
        marker_units={"qrs_loop_writhe_probe": "dimensionless"},
    )
    row = table.iloc[0]
    assert not bool(row["included"])
    assert (
        "no_external_anchor_and_no_replicated_association"
        in row["excluded_reason"]
    )
    assert row["evidence_source"] == "none"


def test_a_strong_auc_alone_does_not_establish_replication():
    table = build_reporting_set(
        reference_ranges=_ranges("qrs_loop_tortuosity", 10, 12),
        repeatability=_repeatability("qrs_loop_tortuosity", 0.85, 1.0),
        transform_agreement=_agreement(
            "qrs_loop_tortuosity", "convertible_with_declared_error"
        ),
        screening=_screening("qrs_loop_tortuosity", 0.708),
        marker_units={"qrs_loop_tortuosity": "dimensionless"},
    )
    row = table.iloc[0]
    assert row["evidence_source"] == "observed_association_not_replication"
    assert not bool(row["included"])


def test_every_failure_names_every_condition_it_failed():
    table = build_reporting_set(
        reference_ranges=_ranges("weak", 1, 12),
        repeatability=_repeatability("weak", 0.10, float("nan")),
        transform_agreement=_agreement("weak", "not_assessable"),
        screening=_screening("weak", 0.50),
        marker_units={"weak": "dimensionless"},
    )
    reason = table.iloc[0]["excluded_reason"]
    assert reason.count("|") == 3
    for expected in (
        "no_reference_range_in_enough_strata",
        "repeatability_below_reportable_reliability",
        "transform_dependence_not_measured",
        "no_external_anchor_and_no_replicated_association",
    ):
        assert expected in reason


def test_a_direction_reversed_association_still_counts():
    """An AUC of 0,30 separates as well as one of 0,70."""

    table = build_reporting_set(
        reference_ranges=_ranges("inverse_marker", 10, 12),
        repeatability=_repeatability("inverse_marker", 0.85, 1.0),
        transform_agreement=_agreement(
            "inverse_marker", "convertible_with_declared_error"
        ),
        screening=_screening("inverse_marker", 0.29),
        marker_units={"inverse_marker": "dimensionless"},
    )
    assert not bool(table.iloc[0]["included"])
    assert table.iloc[0]["evidence_source"] == "observed_association_not_replication"


def test_an_ecg_defined_endpoint_cannot_reach_a_diagnostic_accuracy_claim():
    reporting_set = _set()
    operating = pd.DataFrame(
        {
            "endpoint_id": ["rbbb", "reduced_lvef_strict"],
            "feature_id": ["spatial_qrs_t_angle_deg"] * 2,
            "evaluation_split": ["validation"] * 2,
            "operating_point": ["youden"] * 2,
            "threshold_source": ["train"] * 2,
            "threshold_original": [95.0, 95.0],
            "sensitivity": [0.92, 0.61],
            "specificity": [0.88, 0.70],
        }
    )
    circularity = pd.DataFrame(
        {
            "target_id": ["rbbb", "reduced_lvef_strict"],
            "circularity_class": ["ecg_defined", "independent"],
        }
    )
    reachability = pd.DataFrame(
        {
            "label_id": ["rbbb", "reduced_lvef_strict"],
            "diagnostic_accuracy_claim_reachable": [True, True],
            "verification_rate": [0.99, 0.95],
        }
    )
    table = build_reporting_algorithm(
        reporting_set=reporting_set,
        operating_points=operating,
        circularity=circularity,
        reachability=reachability,
    ).set_index("endpoint_id")
    # A very high AUC does not promote the claim.
    assert table.loc["rbbb", "claim_ceiling"] == "analytic_concordance"
    assert "ecg_derived" in table.loc["rbbb", "claim_ceiling_reason"]
    assert (
        table.loc["reduced_lvef_strict", "claim_ceiling"] == "diagnostic_accuracy"
    )


def test_an_endpoint_that_was_rarely_verified_stays_descriptive():
    reporting_set = _set()
    operating = pd.DataFrame(
        {
            "endpoint_id": ["significant_valvular_disease"],
            "feature_id": ["spatial_qrs_t_angle_deg"],
            "evaluation_split": ["validation"],
            "operating_point": ["youden"],
            "threshold_source": ["train"],
            "threshold_original": [95.0],
            "sensitivity": [0.66],
            "specificity": [0.71],
        }
    )
    reachability = pd.DataFrame(
        {
            "label_id": ["significant_valvular_disease"],
            "diagnostic_accuracy_claim_reachable": [False],
            "verification_rate": [0.0162],
        }
    )
    row = build_reporting_algorithm(
        reporting_set=reporting_set,
        operating_points=operating,
        reachability=reachability,
    ).iloc[0]
    assert row["claim_ceiling"] == "descriptive_association"
    assert row["verification_rate"] == pytest.approx(0.0162)


def test_the_algorithm_marks_rows_whose_marker_is_not_in_the_reporting_set():
    reporting_set = _set()
    operating = pd.DataFrame(
        {
            "endpoint_id": ["reduced_lvef_strict"],
            "feature_id": ["a_marker_that_failed_the_conditions"],
            "evaluation_split": ["validation"],
            "operating_point": ["youden"],
            "threshold_source": ["train"],
            "threshold_original": [1.0],
            "sensitivity": [0.6],
            "specificity": [0.6],
        }
    )
    row = build_reporting_algorithm(
        reporting_set=reporting_set, operating_points=operating
    ).iloc[0]
    assert not bool(row["reportable"])


def test_the_four_mandatory_fields_are_declared_with_a_reason_each():
    table = mandatory_context_table()
    assert len(table) == 4
    assert set(table["field"]) == {
        "vcg_transform",
        "vcg_origin_method",
        "signal_quality_score",
        "heart_rate_bpm",
    }
    assert all(len(reason) > 40 for reason in table["reason"])
    assert len(MANDATORY_CONTEXT_FIELDS) == 4


def test_the_follow_up_formula_names_what_it_needs():
    note = intraindividual_change_note()
    assert "repeatability_coefficient" in note["formula"]
    assert "within measurement error" in note["interpretation_below_one"]
    requirements = " ".join(note["requires"])
    assert "transform" in requirements
    assert "quality" in requirements
    assert "rate" in requirements


def test_the_reliability_floor_is_the_declared_one():
    assert MINIMUM_REPORTABLE_ICC == 0.75
    assert ReportingSetPolicy().minimum_icc == 0.75


def test_no_inputs_yields_an_empty_but_typed_set():
    empty = build_reporting_set(
        reference_ranges=pd.DataFrame(),
        repeatability=pd.DataFrame(),
        transform_agreement=pd.DataFrame(),
        screening=pd.DataFrame(),
        marker_units={},
    )
    assert empty.empty
    assert "excluded_reason" in empty.columns


def test_the_endpoint_that_survives_both_filters_is_a_prognosis():
    """One endpoint of 72, and it is not a diagnosis.

    Two independent things block a claim, and crossing them is the plainest
    statement this project can make about its own design. Six endpoints are
    verified widely enough for a diagnostic-accuracy claim to be confirmable;
    five of those six are read off the same tracing. What is left is 30-day
    mortality.
    """

    from vcg_biomarker.mimic.ecg_derived_targets import classify_target

    reachable = {
        "mortality_30d": 1.000,
        "new_onset_af_30d": 1.000,
        "new_onset_af_90d": 1.000,
        "new_onset_af_1y": 1.000,
        "qtc_gt_480": 0.890,
        "qtc_gt_500": 0.890,
    }
    independent = {
        endpoint
        for endpoint in reachable
        if classify_target(
            target_id=endpoint, target_name=endpoint
        ).classification
        == "independent_of_ecg"
    }
    assert independent == {"mortality_30d"}

    reporting_set = _set()
    operating = pd.DataFrame(
        {
            "endpoint_id": list(reachable),
            "feature_id": ["spatial_qrs_t_angle_deg"] * len(reachable),
            "evaluation_split": ["validation"] * len(reachable),
            "operating_point": ["youden"] * len(reachable),
            "threshold_source": ["train"] * len(reachable),
            "threshold_original": [95.0] * len(reachable),
            "sensitivity": [0.7] * len(reachable),
            "specificity": [0.7] * len(reachable),
        }
    )
    reachability = pd.DataFrame(
        {
            "label_id": list(reachable),
            "diagnostic_accuracy_claim_reachable": [True] * len(reachable),
            "verification_rate": list(reachable.values()),
        }
    )
    table = build_reporting_algorithm(
        reporting_set=reporting_set,
        operating_points=operating,
        reachability=reachability,
    )
    accuracy = table[table["claim_ceiling"] == "diagnostic_accuracy"]
    assert list(accuracy["endpoint_id"]) == ["mortality_30d"]


def test_a_laboratory_endpoint_timed_to_the_ecg_keeps_its_full_claim():
    """The electrolyte endpoints are the reference standards, not readings.

    An earlier version of the circularity registry flagged all seven
    ``*_near_ecg`` endpoints, because their identifier contains "ecg" -- which
    would have downgraded the project's own positive controls to analytic
    concordance.
    """

    from vcg_biomarker.mimic.ecg_derived_targets import classify_target

    for endpoint in (
        "hyperkalemia_near_ecg",
        "severe_hypokalemia_near_ecg",
        "hypocalcemia_near_ecg",
        "hypomagnesemia_near_ecg",
        "digoxin_exposure_near_ecg",
        "qt_prolonging_drug_exposure",
    ):
        verdict = classify_target(target_id=endpoint, target_name=endpoint)
        assert verdict.classification == "independent_of_ecg", endpoint


def test_an_unmeasured_verification_rate_does_not_grant_the_highest_ceiling():
    """An unmeasured gate is an open gate only if it is written that way.

    Measured on the development run, where ``build-labels`` had not yet
    written the reachability table: defaulting a missing endpoint to
    "reachable" handed ``diagnostic_accuracy`` to 56.602 rows, among them
    ``aortic_stenosis`` and ``significant_valvular_disease`` -- endpoints
    whose reference standard exists for one or two per cent of recordings.
    """

    reporting_set = _set()
    operating = pd.DataFrame(
        {
            "endpoint_id": ["aortic_stenosis", "significant_valvular_disease"],
            "feature_id": ["spatial_qrs_t_angle_deg"] * 2,
            "evaluation_split": ["validation"] * 2,
            "operating_point": ["youden"] * 2,
            "threshold_source": ["train"] * 2,
            "threshold_original": [95.0, 95.0],
            "sensitivity": [0.7, 0.7],
            "specificity": [0.7, 0.7],
        }
    )
    # No reachability table at all.
    table = build_reporting_algorithm(
        reporting_set=reporting_set, operating_points=operating
    )
    assert set(table["claim_ceiling"]) == {"descriptive_association"}
    assert all(
        reason == "verification_rate_not_measured_for_this_endpoint"
        for reason in table["claim_ceiling_reason"]
    )


def test_the_claim_ceiling_survives_a_missing_marker_level_input():
    """The ceiling is a property of the endpoint, not of the reporting set.

    Returning nothing when the reporting set is empty hid the stage's most
    important column behind an unrelated file that had not been produced yet.
    """

    operating = pd.DataFrame(
        {
            "endpoint_id": ["qtc_gt_480", "mortality_30d"],
            "feature_id": ["some_marker"] * 2,
            "evaluation_split": ["validation"] * 2,
            "operating_point": ["youden"] * 2,
            "threshold_source": ["train"] * 2,
            "threshold_original": [95.0, 95.0],
            "sensitivity": [0.7, 0.7],
            "specificity": [0.7, 0.7],
        }
    )
    reachability = pd.DataFrame(
        {
            "label_id": ["qtc_gt_480", "mortality_30d"],
            "diagnostic_accuracy_claim_reachable": [True, True],
            "verification_rate": [0.89, 1.0],
        }
    )
    table = build_reporting_algorithm(
        reporting_set=pd.DataFrame(),
        operating_points=operating,
        reachability=reachability,
    ).set_index("endpoint_id")
    assert len(table) == 2
    assert table.loc["qtc_gt_480", "claim_ceiling"] == "analytic_concordance"
    assert table.loc["mortality_30d", "claim_ceiling"] == "diagnostic_accuracy"
    # Nothing is reportable, because no marker cleared the conditions.
    assert not table["reportable"].any()
