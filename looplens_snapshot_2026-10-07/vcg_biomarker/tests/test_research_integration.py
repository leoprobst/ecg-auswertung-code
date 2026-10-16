import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.derived_markers import recovery_kinetics
from vcg_biomarker.mimic.feature_roles import feature_profile


@pytest.mark.parametrize("marker", ["v6_r_peak_time_ms", "v6_v1_interpeak_interval_ms"])
def test_native_lead_modality_preserves_discovery_guards(marker):
    profile = feature_profile(marker)
    assert profile.modality == "ecg"
    assert profile.post_outcome_hypothesis
    assert not profile.core_sentinel_eligible
    assert not profile.combination_eligible


def episode(values=(0., 10., 8., 4.), key=1, day=0):
    landmark = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day)
    return pd.DataFrame({"subject_id": 1, "episode": key,
                         "landmark_time": landmark,
                         "ecg_time": [landmark + pd.Timedelta(hours=h) for h in [-1, 1, 3, 7]],
                         "marker": values})


def test_observed_crossing_reports_bracket_and_retrospective_availability():
    row = recovery_kinetics(episode(), marker_column="marker").iloc[0]
    assert row.observed_half_peak_crossing_hours == 6
    assert row.half_peak_crossing_lower_hours == 2
    assert row.half_peak_crossing_upper_hours == 6
    assert row.half_peak_crossing_status == "interval_censored"
    assert not row.available_at_landmark
    assert not row.physiological_half_life
    assert not row.clinical_claim_allowed


def test_no_observed_crossing_is_censored_not_no_recovery():
    row = recovery_kinetics(episode((0., 10., 8., 7.)), marker_column="marker").iloc[0]
    assert row.half_peak_crossing_status == "right_censored"
    assert row.half_peak_crossing_lower_hours == 6
    assert np.isinf(row.half_peak_crossing_upper_hours)
    assert np.isnan(row.observed_half_peak_crossing_hours)


def test_multiple_landmarks_require_explicit_episode_identity():
    frame = pd.concat([episode(), episode(key=2, day=4)], ignore_index=True)
    ambiguous = recovery_kinetics(frame, marker_column="marker")
    assert ambiguous.failure_reason.eq("ambiguous_episode_landmark").all()
    separated = recovery_kinetics(frame, marker_column="marker", episode_column="episode")
    assert len(separated) == 2
    assert separated.status.eq("evaluable").all()
    assert set(separated.episode_id) == {1, 2}


def test_recovery_duplicates_cannot_satisfy_minimum_followup():
    frame = episode().iloc[:2]
    frame = pd.concat([frame, frame.iloc[[1]], frame.iloc[[1]]], ignore_index=True)
    row = recovery_kinetics(frame, marker_column="marker").iloc[0]
    assert row.status == "not_evaluable"
    assert row.n_post == 1
    assert row.n_duplicate_observations == 2


def test_conflicting_same_time_values_cannot_create_zero_time_recovery():
    frame = episode()
    conflict = frame.iloc[[1]].assign(marker=0.)
    row = recovery_kinetics(pd.concat([frame, conflict]), marker_column="marker").iloc[0]
    assert row.failure_reason == "conflicting_same_time_measurements"
    assert not row.clinical_claim_allowed
    assert not row.available_at_landmark


def test_all_missing_episode_remains_in_result_denominator():
    frame = episode((np.nan, np.inf, -np.inf, np.nan))
    row = recovery_kinetics(frame, marker_column="marker").iloc[0]
    assert row.failure_reason == "no_finite_timed_observations"
    assert row.n_input_rows == row.n_unusable_observations == 4


def test_invalid_landmark_cannot_be_silently_removed():
    frame = episode()
    frame.loc[3, "landmark_time"] = pd.NaT
    row = recovery_kinetics(frame, marker_column="marker").iloc[0]
    assert row.failure_reason == "ambiguous_episode_landmark"


@pytest.mark.parametrize("kwargs", [{"baseline_window_hours": 0},
    {"baseline_window_hours": np.inf}, {"minimum_post_landmark": 2},
    {"minimum_post_landmark": True}, {"minimum_post_landmark": 3.5}])
def test_invalid_recovery_design_rejected(kwargs):
    with pytest.raises(ValueError, match="positive baseline"):
        recovery_kinetics(episode(), marker_column="marker", **kwargs)


def test_recovery_rejects_unknown_identity_and_locked_or_split_patient():
    with pytest.raises(ValueError, match="known patient"):
        recovery_kinetics(episode().assign(subject_id="unknown"), marker_column="marker")
    with pytest.raises(PermissionError, match="development"):
        recovery_kinetics(episode().assign(split="locked_test"), marker_column="marker")
    frame = episode().assign(split=["train", "validation", "validation", "validation"])
    with pytest.raises(ValueError, match="across development"):
        recovery_kinetics(frame, marker_column="marker")


def test_duplicate_imports_preserve_crossing_and_input_order_does_not_matter():
    frame = episode()
    duplicate = pd.concat([frame, frame]).sample(frac=1, random_state=5)
    original = recovery_kinetics(frame, marker_column="marker").iloc[0]
    repeated = recovery_kinetics(duplicate, marker_column="marker").iloc[0]
    assert repeated.n_duplicate_observations == 4
    for field in ("n_pre", "n_post", "observed_half_peak_crossing_hours",
                  "half_peak_crossing_lower_hours", "deviation_area_hours", "observation_density_per_day"):
        assert repeated[field] == original[field]


def test_normalization_residual_uses_first_ecg_and_not_smallest_residual():
    frame = episode().assign(normalized_at=pd.Timestamp("2020-01-01 02:00"))
    row = recovery_kinetics(frame, marker_column="marker", normalization_column="normalized_at").iloc[0]
    assert row.normalization_status == "evaluable"
    assert row.normalization_residual == 8.
    assert row.normalization_ecg_lag_hours == 1.
    assert not row.available_at_landmark


def test_missing_first_normalization_marker_is_not_replaced_with_later_ecg():
    frame = episode().assign(normalized_at=pd.Timestamp("2020-01-01 02:00"))
    frame.loc[2, "marker"] = np.nan
    row = recovery_kinetics(frame, marker_column="marker", normalization_column="normalized_at").iloc[0]
    assert row.normalization_reason == "selected_ecg_marker_missing_or_conflicting"
    assert row.normalization_status == "not_evaluable"


def test_normalization_outside_window_remains_missing():
    frame = episode().assign(normalized_at=pd.Timestamp("2020-01-01 02:00"))
    row = recovery_kinetics(frame, marker_column="marker", normalization_column="normalized_at",
                           normalization_window_hours=.5).iloc[0]
    assert row.normalization_reason == "no_ecg_in_prespecified_normalization_window"
