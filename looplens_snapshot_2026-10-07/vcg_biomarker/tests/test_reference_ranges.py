"""The reference range is the guideline component the project did not have.

A guideline needs six things and the audit found two.  The missing one that
everything else waits on is a reference value per marker, per age, per sex, per
rate band, with the number of people behind it.

These tests pin the four decisions that make the table usable rather than
merely present: that it is labelled as a hospital population and not as normal
values, that one patient contributes once, that a stratum too small to estimate
says so instead of disappearing, and that the transform travels with every row.

Measured while building it: the training half holds 477.493 recordings from
96.738 patients, and 76.977 of those patients contribute one recording each to
the reference cohort.  The signal-quality criterion had to move: the score is
null for every row of the clinical context table and of the master index --
it is computed during extraction and lives only in the feature partitions --
so filtering on it before the feature join removed the entire cohort while
reporting a clean-looking hundred per cent exclusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.reference_ranges import (
    MINIMUM_STRATUM_SUBJECTS,
    NORMATIVE_COHORT_VERSION,
    POPULATION_LABEL,
    NormativeCohortPolicy,
    apply_cohort_policy,
    apply_signal_quality_gate,
    assign_strata,
    band_label,
    cohort_summary,
    one_recording_per_subject,
    reference_ranges_for_markers,
)


def _context(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": list(range(rows)),
            "study_id": [100 + index for index in range(rows)],
            "ecg_time": pd.date_range("2150-01-01", periods=rows, freq="D"),
            "split": ["train"] * rows,
            "age": [25, 35, 45, 55, 65, 75, 85, 95][:rows],
            "sex": ["F", "M"] * (rows // 2),
            "hadm_id": [None] * rows,
            "icustay_id": [None] * rows,
            "machine_heart_rate_bpm": [55, 65, 75, 85, 95, 105, 115, 125][:rows],
            "dx_heart_failure": [False] * rows,
            "med_beta_blocker": [False] * rows,
            "proc_pacemaker_or_icd": [False] * rows,
        }
    )


def test_a_clean_cohort_is_admitted_whole():
    frame = apply_cohort_policy(_context(), NormativeCohortPolicy())
    assert frame["reference_eligible"].all()
    assert (frame["population"] == POPULATION_LABEL).all()
    assert (frame["cohort_version"] == NORMATIVE_COHORT_VERSION).all()


def test_each_criterion_is_named_rather_than_filtered_away():
    """The first question about a reference range is what was excluded."""

    context = _context()
    context.loc[0, "dx_heart_failure"] = True
    context.loc[1, "med_beta_blocker"] = True
    context.loc[2, "proc_pacemaker_or_icd"] = True
    context.loc[3, "icustay_id"] = 4242
    frame = apply_cohort_policy(context, NormativeCohortPolicy())
    reasons = dict(zip(frame["subject_id"], frame["exclusion_reasons"]))
    assert reasons[0] == "cardiac_or_systemic_diagnosis"
    assert reasons[1] == "cardiotropic_medication"
    assert reasons[2] == "implanted_device"
    assert reasons[3] == "intensive_care_episode"
    assert reasons[4] == ""
    summary = cohort_summary(frame)
    criteria = set(summary["criterion"])
    assert {"considered", "reference_eligible"} <= criteria
    assert "cardiac_or_systemic_diagnosis" in criteria


def test_a_nullable_boolean_flag_does_not_raise():
    """The evidence table stores the flags in pandas' nullable boolean dtype.

    ``fillna(0)`` raises on that dtype -- an integer is not a boolean -- and
    the first version of the cohort rule did exactly that on the real table.
    """

    context = _context(4)
    context["dx_heart_failure"] = pd.array([True, None, False, None], dtype="boolean")
    frame = apply_cohort_policy(context, NormativeCohortPolicy())
    reasons = list(frame["exclusion_reasons"])
    assert reasons[0] == "cardiac_or_systemic_diagnosis"
    # An unrecorded diagnosis is not a recorded one.
    assert reasons[1] == ""
    assert reasons[3] == ""


def test_the_quality_gate_runs_where_the_quality_score_exists():
    policy = NormativeCohortPolicy(minimum_signal_quality=0.80)
    joined = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "signal_quality_score": [0.95, 0.42, None],
        }
    )
    gated = apply_signal_quality_gate(joined, policy)
    assert list(gated["signal_quality_excluded"]) == [False, True, True]
    # A frame without the column is not silently emptied.
    missing = apply_signal_quality_gate(
        pd.DataFrame({"subject_id": [1, 2]}), policy
    )
    assert not missing["signal_quality_excluded"].any()


def test_one_patient_contributes_one_recording():
    """800.035 recordings come from 161.352 patients.

    A patient with forty recordings would contribute forty times, and those
    patients are systematically the sicker ones -- so the pseudo-replication
    would move the estimate, not only narrow its interval.
    """

    frame = pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "study_id": [10, 11, 12, 20, 21],
            "ecg_time": pd.to_datetime(
                [
                    "2150-03-01",
                    "2150-01-01",
                    "2150-02-01",
                    "2151-05-05",
                    "2151-04-04",
                ]
            ),
        }
    )
    single = one_recording_per_subject(frame)
    assert len(single) == 2
    # The earliest one, so the choice does not depend on how many follow.
    assert list(single["study_id"]) == [11, 21]


def test_a_stratum_below_the_minimum_is_declared_not_dropped():
    policy = NormativeCohortPolicy(
        minimum_stratum_subjects=50, bootstrap_iterations=0
    )
    rng = np.random.default_rng(3)
    small = pd.DataFrame(
        {
            "subject_id": range(20),
            "sex_band": ["F"] * 20,
            "age_band": ["40-50y"] * 20,
            "rate_band": ["60-70bpm"] * 20,
            "qt_interval_ms": rng.normal(400, 20, 20),
        }
    )
    table = reference_ranges_for_markers(
        small, ["qt_interval_ms"], policy, transform="kors"
    )
    assert len(table) == 1
    row = table.iloc[0]
    assert not bool(row["estimable"])
    assert "stratum_below_minimum_subjects" in row["unestimable_reason"]
    assert np.isnan(row["percentile_50"])
    # The row still carries what it is and where it came from.
    assert row["population"] == POPULATION_LABEL
    assert row["vcg_transform"] == "kors"


def test_a_large_stratum_yields_percentiles_with_an_interval():
    policy = NormativeCohortPolicy(
        minimum_stratum_subjects=50, bootstrap_iterations=80
    )
    rng = np.random.default_rng(7)
    values = rng.normal(400.0, 20.0, 400)
    frame = pd.DataFrame(
        {
            "subject_id": range(400),
            "sex_band": ["M"] * 400,
            "age_band": ["50-60y"] * 400,
            "rate_band": ["70-80bpm"] * 400,
            "qt_interval_ms": values,
        }
    )
    row = reference_ranges_for_markers(
        frame, ["qt_interval_ms"], policy, transform="kors"
    ).iloc[0]
    assert bool(row["estimable"])
    assert row["percentile_2_5"] < row["percentile_50"] < row["percentile_97_5"]
    assert row["percentile_50"] == pytest.approx(400.0, abs=5.0)
    assert (
        row["percentile_2_5_ci_low"]
        <= row["percentile_2_5"]
        <= row["percentile_2_5_ci_high"]
    )
    assert row["subject_count"] == 400


def test_the_transform_is_a_column_and_not_an_assumption():
    """A threshold without its transform is not a threshold.

    Two thirds of the VCG markers move between Kors and inverse Dower; for the
    spatial QRS-T angle the offset is 11,6 degrees with limits of agreement
    from -26,5 to +49,7.
    """

    policy = NormativeCohortPolicy(
        minimum_stratum_subjects=10, bootstrap_iterations=0
    )
    frame = pd.DataFrame(
        {
            "subject_id": range(50),
            "sex_band": ["F"] * 50,
            "age_band": ["30-40y"] * 50,
            "rate_band": ["60-70bpm"] * 50,
            "spatial_qrs_t_angle_deg": np.linspace(20.0, 120.0, 50),
        }
    )
    for transform in ("kors", "inverse_dower"):
        table = reference_ranges_for_markers(
            frame, ["spatial_qrs_t_angle_deg"], policy, transform=transform
        )
        assert (table["vcg_transform"] == transform).all()


def test_bands_are_readable_and_open_at_both_ends():
    edges = (18.0, 30.0, 40.0)
    assert band_label(10.0, edges, "y") == "<18y"
    assert band_label(25.0, edges, "y") == "18-30y"
    assert band_label(35.0, edges, "y") == "30-40y"
    assert band_label(99.0, edges, "y") == ">=40y"
    assert band_label(float("nan"), edges, "y") == "unknown"


def test_strata_are_assigned_from_age_sex_and_rate():
    policy = NormativeCohortPolicy()
    frame = assign_strata(
        pd.DataFrame(
            {
                "subject_id": [1, 2],
                "age": [45.0, 82.0],
                "sex": ["f", "M"],
                "heart_rate_bpm": [72.0, 130.0],
            }
        ),
        policy,
    )
    assert list(frame["age_band"]) == ["40-50y", "80-90y"]
    assert list(frame["sex_band"]) == ["F", "M"]
    assert list(frame["rate_band"]) == ["70-80bpm", "120-200bpm"]


def test_the_declared_minimum_is_the_one_that_is_used():
    assert MINIMUM_STRATUM_SUBJECTS == 200
    assert NormativeCohortPolicy().minimum_stratum_subjects == 200
