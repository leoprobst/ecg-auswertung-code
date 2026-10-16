"""A gap marker must be a property of the heart, not of the label.

Three defects, all measured on the 2026-08-28 run over 630,044 recordings:

* **The gap was mostly the target.** ``age_gap`` correlated with actual age at
  **r = -0.872**. Mean gap by age band: +24.4 years under 40, +9.2 at 40-60,
  -2.3 at 60-75, **-15.4 over 75**. That is regression towards the mean --
  any imperfect regression is biased towards its target's mean, so the
  residual it produces is a function of that target. Seventy-six per cent of
  the "cardiac ageing" signal was age.
* **A model with no signal still emitted a marker.** ``bmi_gap`` reached a
  held-out R^2 of **-0.0000** and its gap correlated with BMI at **r =
  -1.000**, exactly: the model predicts a constant, so the gap is the negated
  target. Reporting that as an ECG-derived quantity is reporting BMI with
  extra steps.
* **Nothing checked the target.** BMI ranged to **107,840** -- 149 records
  above 200, unit errors -- and those 0.05 % of rows moved the gap standard
  deviation from 7.06 to 298.89.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.physiological_gap_models import (
    GAP_MINIMUM_VALIDATION_R2,
    GAP_TARGET_BOUNDS,
    _corrected_gap,
    _fit_gap_bias,
    _safe_correlation,
    _within_target_bounds,
    target_bounds_for,
)


def _biased_gap(rng, n=4000):
    """A gap with exactly the defect the real run showed."""

    actual = rng.uniform(20.0, 90.0, size=n)
    # An imperfect predictor shrinks towards the mean.
    predicted = 55.0 + 0.35 * (actual - 55.0) + rng.normal(0.0, 6.0, size=n)
    return actual, predicted - actual


def test_the_uncorrected_gap_is_mostly_the_target():
    """States the defect, so a future change cannot quietly reintroduce it."""

    rng = np.random.default_rng(3)
    actual, gap = _biased_gap(rng)
    assert _safe_correlation(gap, actual) < -0.8


def test_the_correction_removes_the_dependence_on_the_target():
    rng = np.random.default_rng(3)
    actual, gap = _biased_gap(rng)
    slope, intercept = _fit_gap_bias(gap, actual)
    corrected = _corrected_gap(gap, actual, slope=slope, intercept=intercept)
    assert abs(_safe_correlation(corrected, actual)) < 0.01


def test_the_correction_is_fitted_on_train_and_still_works_on_held_out_data():
    """Fitting it on the evaluation split would be circular.

    Measured on the real run: the line fitted on train moved the validation
    correlation from -0.871 to +0.023.
    """

    rng = np.random.default_rng(7)
    train_actual, train_gap = _biased_gap(rng, n=5000)
    held_actual, held_gap = _biased_gap(rng, n=5000)
    slope, intercept = _fit_gap_bias(train_gap, train_actual)
    corrected = _corrected_gap(
        held_gap, held_actual, slope=slope, intercept=intercept
    )
    assert abs(_safe_correlation(corrected, held_actual)) < 0.05


def test_the_correction_shrinks_the_band_bias_not_just_the_correlation():
    """A near-zero correlation with a lopsided mean would still mislead."""

    rng = np.random.default_rng(9)
    actual, gap = _biased_gap(rng, n=8000)
    slope, intercept = _fit_gap_bias(gap, actual)
    corrected = _corrected_gap(gap, actual, slope=slope, intercept=intercept)
    young = actual < 40
    old = actual > 75
    assert abs(float(np.mean(gap[young]))) > 8.0
    assert abs(float(np.mean(gap[old]))) > 8.0
    assert abs(float(np.mean(corrected[young]))) < 2.0
    assert abs(float(np.mean(corrected[old]))) < 2.0


def test_a_constant_prediction_makes_the_gap_the_negated_target():
    """The bmi_gap signature: r = -1.000 exactly."""

    rng = np.random.default_rng(4)
    actual = rng.uniform(18.0, 45.0, size=2000)
    gap = np.full_like(actual, 28.0) - actual
    assert _safe_correlation(gap, actual) == pytest.approx(-1.0, abs=1e-9)


def test_the_usability_floor_is_prespecified_and_rejects_a_null_model():
    assert GAP_MINIMUM_VALIDATION_R2 == 0.05
    assert -0.0000 < GAP_MINIMUM_VALIDATION_R2, "bmi_gap muss durchfallen"
    assert 0.2528 > GAP_MINIMUM_VALIDATION_R2, "age_gap muss bestehen"


def test_the_target_bounds_reject_the_unit_errors_that_were_measured():
    assert target_bounds_for("bmi_gap") == (12.0, 70.0)
    values = np.array([22.0, 31.5, 107840.2, 0.0, 250.0, np.nan, 68.0])
    keep = _within_target_bounds(values, "bmi_gap")
    assert keep.tolist() == [True, True, False, False, False, False, True]


def test_an_unknown_target_is_not_silently_bounded():
    lower, upper = target_bounds_for("something_new")
    assert lower == float("-inf") and upper == float("inf")
    assert _within_target_bounds(np.array([1e9]), "something_new").all()


def test_one_bad_row_no_longer_governs_the_dispersion():
    """149 rows moved the real standard deviation by a factor of 42."""

    rng = np.random.default_rng(2)
    clean = rng.normal(28.0, 6.0, size=10_000)
    contaminated = np.concatenate([clean, [107840.2] * 5])
    assert float(np.std(contaminated)) > 20 * float(np.std(clean))
    keep = _within_target_bounds(contaminated, "bmi_gap")
    assert float(np.std(contaminated[keep])) == pytest.approx(
        float(np.std(clean)), rel=0.05
    )


def test_the_bias_fit_refuses_degenerate_input_instead_of_guessing():
    assert _fit_gap_bias(np.array([1.0]), np.array([2.0])) == (0.0, 0.0)
    constant = np.full(50, 40.0)
    assert _fit_gap_bias(np.arange(50.0), constant) == (0.0, 0.0)
    with_nans = np.array([np.nan] * 50)
    assert _fit_gap_bias(with_nans, np.arange(50.0)) == (0.0, 0.0)


def test_every_declared_target_has_bounds():
    from vcg_biomarker.mimic.physiological_gap_models import GAP_TARGETS

    for target in GAP_TARGETS:
        assert target in GAP_TARGET_BOUNDS, target
        lower, upper = GAP_TARGET_BOUNDS[target]
        assert np.isfinite(lower) and np.isfinite(upper) and lower < upper
