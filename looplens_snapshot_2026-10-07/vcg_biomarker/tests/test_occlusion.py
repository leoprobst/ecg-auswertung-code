"""The criterion and the reference must not be the same tracing.

Testing "does ST elevation detect occlusion" against a cardiologist's reading
of the same ECG settles nothing. ``omi_revascularization_proxy`` is built from
a troponin rise, a PCI within a day, and a coronary presentation -- so here the
question has an answer, and the answer is that 81,3 % of 2.120 events carry no
ST-elevation criterion.

Two things this file is built to protect:

* the caveat has to travel with the number. A revascularization proxy is not
  angiographic occlusion, and "81 % of occlusions are missed" is a claim the
  data does not support. ``reference_note`` is therefore part of the result,
  not part of the documentation.
* a marker must not be allowed to "keep" its discrimination by silently
  reversing direction between the two populations. That would turn a marker
  which fails in the criterion-negative into one that looks robust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import duckdb

from vcg_biomarker.mimic.occlusion import (
    CLASSICAL_CRITERIA,
    OCCLUSION_ENDPOINT,
    OMI_EQUIVALENT_PATTERNS,
    OcclusionPolicy,
    STEMI_CRITERION,
    criterion_yield,
    discrimination_without_the_criterion,
    missed_by_criterion,
    occlusion_frame_query,
)


def _cohort(n=2000, prevalence=0.1, sensitivity=0.2, seed=0):
    """A cohort where the criterion has a known sensitivity."""

    rng = np.random.default_rng(seed)
    occlusion = (rng.random(n) < prevalence).astype(int)
    criterion = np.zeros(n, dtype=bool)
    positives = np.flatnonzero(occlusion == 1)
    criterion[rng.choice(positives, int(len(positives) * sensitivity), replace=False)] = True
    negatives = np.flatnonzero(occlusion == 0)
    criterion[rng.choice(negatives, int(len(negatives) * 0.08), replace=False)] = True
    frame = pd.DataFrame({"occlusion": occlusion, STEMI_CRITERION: criterion})
    for name in CLASSICAL_CRITERIA:
        if name not in frame:
            frame[name] = rng.random(n) < 0.05
    frame["marker"] = rng.normal(size=n) + occlusion * 0.4
    return frame


# --- The yield table ---------------------------------------------------------


def test_a_known_sensitivity_is_recovered():
    frame = _cohort(n=4000, sensitivity=0.2, seed=1)
    table = criterion_yield(frame).set_index("criterion")
    assert table.loc[STEMI_CRITERION, "sensitivity"] == pytest.approx(20.0, abs=1.0)


def test_the_unassessable_share_is_reported_next_to_the_sensitivity():
    """11,2 % of real ECGs cannot be assessed for the ST criteria."""

    frame = _cohort(n=2000)
    frame[STEMI_CRITERION] = frame[STEMI_CRITERION].astype("boolean")
    frame.loc[frame.index[:200], STEMI_CRITERION] = pd.NA
    table = criterion_yield(frame).set_index("criterion")
    assert table.loc[STEMI_CRITERION, "n_unassessable_pct"] == pytest.approx(10.0)


def test_an_unassessable_criterion_is_not_counted_as_met():
    frame = _cohort(n=2000)
    frame[STEMI_CRITERION] = pd.Series([pd.NA] * len(frame), dtype="boolean")
    table = criterion_yield(frame).set_index("criterion")
    assert table.loc[STEMI_CRITERION, "n_positive"] == 0
    assert table.loc[STEMI_CRITERION, "sensitivity"] == 0.0


def test_a_rule_that_fires_on_everyone_is_visible_as_such():
    """``pathological_q_waves`` was positive in 81 % of a hospital cohort."""

    frame = _cohort(n=2000, prevalence=0.1)
    frame["classical_pathological_q_waves_positive"] = True
    table = criterion_yield(frame).set_index("criterion")
    row = table.loc["classical_pathological_q_waves_positive"]
    assert row["sensitivity"] == pytest.approx(100.0)
    assert row["specificity"] == pytest.approx(0.0)
    # The tell: PPV collapses to the prevalence.
    assert row["positive_predictive_value"] == pytest.approx(
        frame["occlusion"].mean() * 100, abs=0.1
    )


def test_a_cohort_with_too_few_events_yields_nothing():
    frame = _cohort(n=200, prevalence=0.01, seed=3)
    assert criterion_yield(frame).empty


def test_an_absent_criterion_column_does_not_crash_the_table():
    frame = _cohort(n=2000).drop(columns=["classical_wellens_ecg_pattern_positive"])
    table = criterion_yield(frame).set_index("criterion")
    assert table.loc["classical_wellens_ecg_pattern_positive", "n_positive"] == 0


# --- What the criterion misses -----------------------------------------------


def test_the_missed_count_is_the_complement_of_the_sensitivity():
    frame = _cohort(n=4000, sensitivity=0.2, seed=4)
    summary = missed_by_criterion(frame)
    assert summary["missed_share_pct"] == pytest.approx(80.0, abs=1.5)
    assert summary["n_missed"] + int(
        (frame["occlusion"].astype(bool) & frame[STEMI_CRITERION]).sum()
    ) == summary["n_events"]


def test_the_equivalent_patterns_recover_only_the_missed():
    """A pattern firing on an already-caught event must not be counted."""

    frame = _cohort(n=3000, seed=5)
    for name in OMI_EQUIVALENT_PATTERNS:
        frame[name] = False
    # A pattern that only fires where the criterion already did.
    frame.loc[frame[STEMI_CRITERION], OMI_EQUIVALENT_PATTERNS[0]] = True
    assert missed_by_criterion(frame)["n_recovered_by_equivalents"] == 0


def test_a_pattern_covering_every_missed_event_recovers_all_of_them():
    frame = _cohort(n=3000, seed=6)
    for name in OMI_EQUIVALENT_PATTERNS:
        frame[name] = False
    missed = frame["occlusion"].astype(bool) & ~frame[STEMI_CRITERION]
    frame.loc[missed, OMI_EQUIVALENT_PATTERNS[1]] = True
    summary = missed_by_criterion(frame)
    assert summary["recovered_share_pct"] == pytest.approx(100.0)


def test_the_caveat_travels_with_the_number():
    """The result must not be quotable without what the endpoint actually is."""

    summary = missed_by_criterion(_cohort(n=3000))
    assert "Revaskularisations" in summary["reference_note"]
    assert "angiographisch" in summary["reference_note"].lower()


def test_an_empty_cohort_reports_no_events_rather_than_dividing_by_zero():
    summary = missed_by_criterion(pd.DataFrame())
    assert summary["n_events"] == 0


# --- Discrimination without the criterion ------------------------------------


def test_a_marker_independent_of_the_criterion_keeps_its_discrimination():
    frame = _cohort(n=6000, seed=7)
    result = discrimination_without_the_criterion(frame, ["marker"])
    row = result.iloc[0]
    assert row["retained"] > 0.7, "ein unabhaengiger Marker darf nicht einbrechen"


def test_a_marker_that_only_re_expresses_the_criterion_collapses():
    """This is what the comparison exists to detect."""

    frame = _cohort(n=6000, seed=8)
    frame["marker"] = frame[STEMI_CRITERION].astype(float)
    result = discrimination_without_the_criterion(frame, ["marker"])
    assert result.empty or abs(result.iloc[0]["auc_criterion_negative"] - 0.5) < 0.01


def test_direction_is_fixed_on_the_full_cohort_not_re_chosen():
    """Otherwise a marker keeps its score by reversing, which is not the same
    finding.

    Building this took a second attempt. At the realistic sensitivity of 20 %
    the criterion-negative are four fifths of the events, so they *set* the
    overall direction and there is nothing for them to disagree with. The
    disagreement only exists when the criterion catches most events, so the
    sensitivity is turned up to 80 % here -- a deliberately unrealistic cohort
    built to exercise one branch, not to resemble the data.
    """

    rng = np.random.default_rng(9)
    frame = _cohort(n=8000, sensitivity=0.8, seed=9)
    criterion = frame[STEMI_CRITERION].to_numpy()
    outcome = frame["occlusion"].to_numpy().astype(bool)
    # Predictive among the criterion-positive, anti-predictive among the rest.
    values = rng.normal(size=len(frame))
    values[criterion & outcome] += 3.0
    values[~criterion & outcome] -= 3.0
    frame["marker"] = values

    row = discrimination_without_the_criterion(frame, ["marker"]).iloc[0]
    assert row["auc_all"] > 0.5, "der Aufbau braucht einen insgesamt positiven Marker"
    assert row["auc_criterion_negative"] < 0.5, (
        "die Richtung muss aus der Gesamtkohorte stammen; sonst sieht ein "
        "umgedrehter Marker robust aus"
    )
    assert row["retained"] < 0


def test_both_populations_are_reported_side_by_side():
    result = discrimination_without_the_criterion(_cohort(n=6000), ["marker"])
    for column in ("auc_all", "auc_criterion_negative", "n_events_criterion_negative"):
        assert column in result


def test_the_criterion_negative_population_is_smaller():
    result = discrimination_without_the_criterion(_cohort(n=6000), ["marker"])
    row = result.iloc[0]
    assert row["n_criterion_negative"] < row["n_all"]


def test_a_marker_absent_from_the_frame_is_skipped():
    result = discrimination_without_the_criterion(
        _cohort(n=6000), ["marker", "gibt_es_nicht"]
    )
    assert list(result["marker"]) == ["marker"]


def test_too_few_events_after_restriction_yields_no_row():
    frame = _cohort(n=3000, seed=10)
    # Almost every event meets the criterion, so nothing is left to judge.
    frame[STEMI_CRITERION] = frame["occlusion"].astype(bool)
    assert discrimination_without_the_criterion(frame, ["marker"]).empty


# --- The query ---------------------------------------------------------------


def test_indeterminate_endpoints_are_excluded_not_counted_as_negative():
    """Folding partial evidence into the negatives inflates every specificity."""

    sql = occlusion_frame_query(
        "master.parquet", "labels.parquet", "read_parquet('f/*.parquet')",
        marker_columns="f.svg_magnitude",
    )
    assert "IN ('positive', 'negative')" in sql
    assert "indeterminate" not in sql


def test_the_query_names_the_endpoint_state_column():
    sql = occlusion_frame_query(
        "master.parquet", "labels.parquet", "read_parquet('f/*.parquet')",
        marker_columns="f.svg_magnitude",
    )
    assert f"{OCCLUSION_ENDPOINT}_state" in sql


def test_the_query_uses_the_first_ecg_of_each_clinical_episode(tmp_path):
    master = pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "study_id": [1, 2, 3, 4, 5],
            "split": ["train"] * 5,
            "ecg_time": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 09:00",
                    "2026-02-01 08:00",
                    "2026-03-01 08:00",
                    "2026-03-01 09:00",
                ]
            ),
            "hadm_id": [10, 10, 11, np.nan, np.nan],
            "ed_stay_id": [np.nan] * 5,
        }
    )
    labels = master[["subject_id", "study_id"]].copy()
    labels[f"{OCCLUSION_ENDPOINT}_state"] = [
        "positive", "positive", "negative", "negative", "negative"
    ]
    features = master[["subject_id", "study_id"]].copy()
    features["extraction_status"] = "ok"
    features["svg_magnitude"] = np.arange(5, dtype=float)
    master_path = tmp_path / "master.parquet"
    labels_path = tmp_path / "labels.parquet"
    features_path = tmp_path / "features.parquet"
    master.to_parquet(master_path, index=False)
    labels.to_parquet(labels_path, index=False)
    features.to_parquet(features_path, index=False)

    query = occlusion_frame_query(
        str(master_path).replace("\\", "/"),
        str(labels_path).replace("\\", "/"),
        f"read_parquet('{str(features_path).replace('\\', '/')}')",
        marker_columns="f.svg_magnitude",
    )
    result = duckdb.connect().execute(query).fetchdf()

    assert set(result["study_id"]) == {1, 3, 4}
    assert len(result) == 3
    assert not result.duplicated(["subject_id", "clinical_episode_id"]).any()


def test_the_query_only_reads_successfully_extracted_ecgs():
    sql = occlusion_frame_query(
        "master.parquet", "labels.parquet", "read_parquet('f/*.parquet')",
        marker_columns="f.svg_magnitude",
    )
    assert "extraction_status = 'ok'" in sql


def test_the_locked_test_split_is_never_read():
    sql = occlusion_frame_query(
        "master.parquet", "labels.parquet", "read_parquet('f/*.parquet')",
        marker_columns="f.svg_magnitude",
    )
    assert "'train', 'validation'" in sql
    assert "locked_test" not in sql


def test_the_two_missingness_readings_actually_differ():
    """The defect: both branches of ``_flag`` computed the same thing.

    ``raw.fillna(False) & raw.notna()`` is ``raw.fillna(False)`` -- a missing
    value is already False by the time the second term applies. So
    ``treat_unassessable_as_negative=False`` changed nothing, while the setting
    was written into the policy and the run summary as though it were a choice.
    Excluding an ECG means removing it from the *denominator*, which a flag
    alone cannot express.
    """

    from vcg_biomarker.mimic.occlusion import _flag

    frame = pd.DataFrame(
        {STEMI_CRITERION: pd.Series([True, False, pd.NA], dtype="boolean")}
    )
    _, lenient, _ = _flag(frame, STEMI_CRITERION, unassessable_negative=True)
    _, strict, _ = _flag(frame, STEMI_CRITERION, unassessable_negative=False)
    assert list(lenient) == [True, True, True]
    assert list(strict) == [True, True, False]


def test_the_strict_reading_shrinks_the_denominator():
    frame = _cohort(n=3000, seed=12)
    frame[STEMI_CRITERION] = frame[STEMI_CRITERION].astype("boolean")
    frame.loc[frame.index[:300], STEMI_CRITERION] = pd.NA

    lenient = criterion_yield(frame).set_index("criterion")
    strict = criterion_yield(
        frame, policy=OcclusionPolicy(treat_unassessable_as_negative=False)
    ).set_index("criterion")
    assert (
        strict.loc[STEMI_CRITERION, "n_assessed"]
        < lenient.loc[STEMI_CRITERION, "n_assessed"]
    )
    assert lenient.loc[STEMI_CRITERION, "n_assessed"] == len(frame)


def test_the_missed_share_barely_moves_between_the_two_readings():
    """Measured on the real cohort: 81,3 % against 80,4 %.

    If the choice moved the headline, having a default would not be safe.
    """

    frame = _cohort(n=4000, sensitivity=0.2, seed=13)
    frame[STEMI_CRITERION] = frame[STEMI_CRITERION].astype("boolean")
    frame.loc[frame.index[:400], STEMI_CRITERION] = pd.NA

    lenient = missed_by_criterion(frame)["missed_share_pct"]
    strict = missed_by_criterion(
        frame, policy=OcclusionPolicy(treat_unassessable_as_negative=False)
    )["missed_share_pct"]
    assert abs(lenient - strict) < 15.0


def test_the_policy_records_how_missingness_was_handled():
    payload = OcclusionPolicy().to_dict()
    assert payload["treat_unassessable_as_negative"] is True
    assert payload["criterion"] == STEMI_CRITERION
    assert payload["version"]
