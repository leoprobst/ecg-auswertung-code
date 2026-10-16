import pandas as pd
import pytest

from vcg_biomarker.mimic.recovery_reference import attach_observed_normalization


def fixture():
    start = pd.Timestamp("2020-01-02")
    events = pd.DataFrame(dict(subject_id=[1], hadm_id=[10], episode_id=["a"],
        landmark_time=[start], episode_end=[start+pd.Timedelta(hours=48)]))
    labs = pd.DataFrame(dict(subject_id=1, hadm_id=10, labevent_id=[1, 2, 3], itemid=50971,
        lab_group="potassium", charttime=[start+pd.Timedelta(hours=h) for h in (-1, 2, 5)],
        storetime=[start+pd.Timedelta(hours=h) for h in (0, 3, 6)],
        value_numeric=[6., 4., 4.2], reference_lower=3.5, reference_upper=5., unit="mEq/L", value_status="usable"))
    return events, labs


def run(events, labs):
    return attach_observed_normalization(events, labs, lab_group="potassium").iloc[0]


def test_first_observed_same_assay_normalization_and_availability():
    events, labs = fixture()
    row = run(events, labs)
    assert row.reference_normalization_status == "observed"
    assert row.normalization_time == labs.charttime.iloc[1]
    assert row.reference_result_available_at == labs.storetime.iloc[1].isoformat()
    assert row.reference_normalization_labevent_id == 2
    assert not row.clinical_claim_allowed


@pytest.mark.parametrize("column,value", [("unit", "mmol/L"), ("itemid", 50822),
    ("reference_upper", 5.1), ("value_status", "unit_unrecognized"), ("hadm_id", 99)])
def test_incompatible_followup_is_not_pooled(column, value):
    events, labs = fixture()
    labs.loc[1:, column] = value
    row = run(events, labs)
    assert row.reference_normalization_status == "not_evaluable"
    assert pd.isna(row.normalization_time)


def test_invalid_nearest_baseline_is_not_replaced_by_older_abnormal_value():
    events, labs = fixture()
    invalid = labs.iloc[[0]].assign(labevent_id=4, value_status="unit_missing",
                                   charttime=events.landmark_time.iloc[0])
    row = run(events, pd.concat([labs, invalid]))
    assert row.reference_normalization_reason == "invalid_baseline_value_unit_or_interval"


def test_no_reference_interval_and_already_normal_baseline():
    events, labs = fixture()
    labs.loc[0, "reference_lower"] = float("nan")
    assert run(events, labs).reference_normalization_reason == "invalid_baseline_value_unit_or_interval"
    labs.loc[0, "reference_lower"] = 3.5
    labs.loc[0, "value_numeric"] = 4.
    assert run(events, labs).reference_normalization_reason == "baseline_already_in_range"


def test_conflicting_normalization_cannot_be_cherry_picked():
    events, labs = fixture()
    conflict = labs.iloc[[1]].assign(labevent_id=4, value_numeric=6.)
    assert run(events, pd.concat([labs, conflict])).reference_normalization_reason == "conflicting_normalization_measurements"


def test_lab_at_episode_end_is_not_included():
    events, labs = fixture()
    events["episode_end"] = labs.charttime.iloc[1]
    assert run(events, labs).reference_normalization_reason == "no_observed_compatible_normalization"


def test_reference_contract_survives_through_ecg_episode_result():
    from vcg_biomarker.mimic.recovery_episodes import episode_recovery_table
    events, labs = fixture()
    attached = attach_observed_normalization(events, labs, lab_group="potassium")
    start = events.landmark_time.iloc[0]
    ecgs = pd.DataFrame(dict(subject_id=1, hadm_id=10, study_id=range(4), split="train",
        ecg_time=[start+pd.Timedelta(hours=h) for h in (-2, 1, 3, 7)],
        extraction_status="ok", signal_quality_score=.9, marker=[0., 10., 8., 4.]))
    row = episode_recovery_table(ecgs, attached, marker_ids=("marker",)).iloc[0]
    assert row.normalization_residual == 8.
    assert row.reference_normalization_labevent_id == 2
    assert row.reference_kind == "independent_laboratory_measurement"
    assert row.reference_result_available_at == labs.storetime.iloc[1].isoformat()


@pytest.mark.parametrize("values,expected", [
    ([4., 4., 4.2], "baseline_and_all_observed_followups_in_range"),
    ([4., 6., 4.2], "baseline_in_range_followup_excursion_observed"),
    ([6., 6., 5.5], "baseline_abnormal_no_in_range_followup_observed"),
    ([6., 4., 6.], "baseline_abnormal_in_range_then_excursion_observed"),
    ([6., 4., 4.2], "baseline_abnormal_in_range_without_observed_recurrence"),
])
def test_observed_course_is_not_inferred_from_ecg(values, expected):
    events, labs = fixture()
    labs["value_numeric"] = values
    row = run(events, labs)
    assert row.reference_observed_course == expected
    assert row.reference_n_followup_times == 2
    assert row.reference_followup_span_hours == 3.
    assert "not_continuous_stability" in row.reference_course_interpretation


@pytest.mark.parametrize("kind", ["none", "one", "duplicate", "invalid", "conflict"])
def test_missing_incompatible_or_duplicate_followup_never_creates_stable_course(kind):
    events, labs = fixture()
    labs["value_numeric"] = 4.
    if kind == "none":
        labs = labs.iloc[:1]
    elif kind == "one":
        labs = labs.iloc[:2]
    elif kind == "duplicate":
        labs.loc[2, "charttime"] = labs.charttime.iloc[1]
    elif kind == "invalid":
        labs.loc[2, "value_status"] = "unit_missing"
    else:
        labs = pd.concat([labs, labs.iloc[[1]].assign(labevent_id=99, value_numeric=6.)])
    row = run(events, labs)
    assert row.reference_observed_course == "unknown"
    assert row.reference_course_reason
