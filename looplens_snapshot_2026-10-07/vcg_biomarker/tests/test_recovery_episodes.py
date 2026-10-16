import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.recovery_episodes import episode_recovery_table


@pytest.mark.parametrize("seed", range(12))
def test_time_index_matches_direct_predicates(seed):
    from vcg_biomarker.mimic.recovery_episodes import _episode_time_index
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2020-01-01")
    starts = pd.Series(origin + pd.to_timedelta(rng.integers(0, 300, 180), unit="h"))
    ends = starts + pd.to_timedelta(rng.integers(-10, 40, 180), unit="h")
    starts.iloc[::17] = pd.NaT
    ends.iloc[::19] = pd.NaT
    group = pd.DataFrame(dict(landmark_time=starts, episode_end=ends)).sort_values(
        "landmark_time", kind="stable").reset_index(drop=True)
    overlap, prior = _episode_time_index(group)
    for i, row in group.iterrows():
        start, end = row.landmark_time, row.episode_end
        if pd.isna(start) or pd.isna(end) or start >= end:
            continue
        direct = group.landmark_time.lt(end) & group.episode_end.gt(start)
        direct.iloc[i] = False
        assert overlap[i] == direct.any()
        expected = group.loc[group.episode_end.le(start), "episode_end"].max()
        assert (pd.isna(expected) and np.isnat(prior[i])) or pd.Timestamp(prior[i]) == expected


def fixture():
    stamp = pd.Timestamp("2020-01-02")
    ecgs = pd.DataFrame(dict(subject_id=1, hadm_id=10, study_id=range(5), split="train",
        ecg_time=[stamp+pd.Timedelta(hours=h) for h in (-2, 1, 3, 7, 48)],
        extraction_status="ok", signal_quality_score=.9, marker=[0., 10., 8., 4., 99.]))
    events = pd.DataFrame(dict(subject_id=[1], hadm_id=[10], episode_id=["a"],
        landmark_time=[stamp], episode_end=[stamp+pd.Timedelta(hours=48)],
        normalization_time=[stamp+pd.Timedelta(hours=2)]))
    return ecgs, events


def test_bounded_episode_and_normalization():
    ecgs, events = fixture()
    row = episode_recovery_table(ecgs, events, marker_ids=("marker",)).iloc[0]
    assert row.status == "evaluable"
    assert row.n_post == 3  # exactly at the exclusive end is not admitted
    assert row.peak_deviation == 10.
    assert row.normalization_residual == 8.
    assert row.observed_half_peak_crossing_hours == 6.
    assert not row.clinical_claim_allowed


def test_overlapping_episodes_abstain_and_never_duplicate_ecgs():
    ecgs, events = fixture()
    events = pd.concat([events, events.assign(episode_id="b")], ignore_index=True)
    result = episode_recovery_table(ecgs, events, marker_ids=("marker",))
    assert len(result) == 2
    assert result.failure_reason.eq("overlapping_episode_windows").all()


def test_other_admission_and_absent_marker_preserve_failure_rows():
    ecgs, events = fixture()
    result = episode_recovery_table(ecgs.assign(hadm_id=99), events, marker_ids=("marker", "absent"))
    assert len(result) == 2
    assert result.failure_reason.eq("no_ecg_in_episode_window").all()
    row = episode_recovery_table(ecgs, events, marker_ids=("absent",)).iloc[0]
    assert row.failure_reason == "no_finite_timed_observations"


def test_qc_failure_does_not_select_later_normalization_ecg():
    ecgs, events = fixture()
    ecgs.loc[2, "signal_quality_score"] = .1
    row = episode_recovery_table(ecgs, events, marker_ids=("marker",)).iloc[0]
    assert row.normalization_reason == "selected_ecg_marker_missing_or_conflicting"
    assert row.n_unusable_observations == 1


def test_additional_reference_does_not_replace_failed_first_ecg():
    ecgs, events = fixture()
    events["magnesium_time"] = events.normalization_time
    ecgs.loc[2, "signal_quality_score"] = .1
    row = episode_recovery_table(ecgs, events, marker_ids=("marker",),
        additional_normalization_columns={"magnesium": "magnesium_time"}).iloc[0]
    assert row.magnesium__normalization_reason == "selected_ecg_marker_missing_or_conflicting"
    assert row.magnesium__normalization_status == "not_evaluable"
    assert row.magnesium__normalization_ecg_lag_hours == 1.


def test_additional_reference_cannot_overwrite_ecg_context():
    ecgs, events = fixture()
    with pytest.raises(ValueError, match="cannot overwrite"):
        episode_recovery_table(ecgs, events, marker_ids=("marker",),
            additional_normalization_columns={"magnesium": "landmark_time"})


def test_duplicate_and_locked_input_rejected():
    ecgs, events = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        episode_recovery_table(pd.concat([ecgs, ecgs]), events, marker_ids=("marker",))
    with pytest.raises(PermissionError):
        episode_recovery_table(ecgs.assign(split="locked_test"), events, marker_ids=("marker",))
    with pytest.raises(ValueError, match="finite"):
        episode_recovery_table(ecgs, events, marker_ids=("marker",), followup_hours=np.inf)


def test_prior_episode_cannot_be_used_as_next_baseline():
    ecgs, events = fixture()
    previous = events.copy()
    previous["episode_id"] = "previous"
    previous["landmark_time"] -= pd.Timedelta(days=1)
    previous["episode_end"] = events.landmark_time - pd.Timedelta(hours=1)
    result = episode_recovery_table(ecgs, pd.concat([previous, events]), marker_ids=("marker",))
    row = result.set_index("episode_id").loc["a"]
    assert row.n_pre == 0  # ECG two hours before belongs to the prior episode
    assert row.status == "not_evaluable"
