"""The cheaper metric path must return the same numbers as the ranked one.

``evaluate_marker`` used to call ``roc_auc_score`` three times on at most two
distinct rankings, and ``_missingness_auc_sensitivity`` ranked the cohort twice
more to obtain bounds that follow arithmetically from the observed AUC.  At the
scale this project screens at -- roughly five million marker-by-phenotype pairs
per split, each over hundreds of thousands of recordings -- that is the
difference between ranking the cohort five times and ranking it once.

Replacing a ranking with arithmetic is only allowed if it returns the same
answer, so that is what these tests check, against the ranked computation
itself rather than against a stored expectation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from vcg_biomarker.mimic.statistics import (
    _missingness_auc_sensitivity,
    evaluate_marker,
    screen_marker_assessability_from_counts,
)


def _bounds_by_ranking(values, labels, *, direction):
    """The bounds as they were computed before: by ranking twice."""

    values_array = np.asarray(values, dtype=float)
    labels_array = np.asarray(labels, dtype=float)
    label_mask = np.isfinite(labels_array) & np.isin(labels_array, (0, 1))
    scoped_labels = labels_array[label_mask].astype(np.int8)
    scoped_values = int(direction) * values_array[label_mask]
    missing = ~np.isfinite(scoped_values)
    observed = scoped_values[~missing]
    if len(observed):
        low = float(np.min(observed) - max(np.ptp(observed), 1.0))
        high = float(np.max(observed) + max(np.ptp(observed), 1.0))
    else:
        low, high = -1.0, 1.0
    lower = scoped_values.copy()
    upper = scoped_values.copy()
    lower[missing & (scoped_labels == 1)] = low
    lower[missing & (scoped_labels == 0)] = high
    upper[missing & (scoped_labels == 1)] = high
    upper[missing & (scoped_labels == 0)] = low
    return (
        float(roc_auc_score(scoped_labels, lower)),
        float(roc_auc_score(scoped_labels, upper)),
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("direction", [1, -1])
def test_the_bounds_match_the_ranked_computation(seed: int, direction: int):
    rng = np.random.default_rng(seed)
    size = int(rng.integers(60, 400))
    labels = rng.integers(0, 2, size=size)
    values = rng.normal(size=size) + 0.7 * labels
    # Missing at a rate that varies with the seed, including none and many.
    missing_fraction = (seed % 6) / 6.0
    values[rng.random(size) < missing_fraction] = np.nan
    if len(np.unique(labels)) < 2 or not np.isfinite(values).any():
        pytest.skip("degenerate draw")

    expected_lower, expected_upper = _bounds_by_ranking(
        values, labels, direction=direction
    )
    result = _missingness_auc_sensitivity(
        values, labels, direction=direction, assessability_gate_passed=True
    )
    assert result["missingness_auc_lower_bound"] == pytest.approx(
        expected_lower, abs=1e-12
    )
    assert result["missingness_auc_upper_bound"] == pytest.approx(
        expected_upper, abs=1e-12
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("direction", [1, -1])
def test_the_auc_family_matches_three_separate_rankings(seed: int, direction: int):
    rng = np.random.default_rng(1000 + seed)
    size = int(rng.integers(60, 400))
    labels = rng.integers(0, 2, size=size)
    values = rng.normal(size=size) + 0.5 * labels
    values[rng.random(size) < (seed % 5) / 8.0] = np.nan
    finite = np.isfinite(values)
    if len(np.unique(labels[finite])) < 2:
        pytest.skip("degenerate draw")

    clean_values = values[finite]
    clean_labels = labels[finite]
    aligned = int(direction) * clean_values

    result = evaluate_marker(values, labels, direction=direction)
    assert result["roc_auc"] == float(roc_auc_score(clean_labels, aligned))
    assert result["rank_biserial"] == pytest.approx(
        2.0 * float(roc_auc_score(clean_labels, clean_values)) - 1.0, abs=1e-12
    )
    assert result["aligned_rank_biserial"] == pytest.approx(
        2.0 * float(roc_auc_score(clean_labels, aligned)) - 1.0, abs=1e-12
    )


def test_a_marker_observed_everywhere_has_bounds_equal_to_its_auc():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=300)
    values = rng.normal(size=300) + labels
    result = evaluate_marker(values, labels, direction=1)
    assert result["missingness_auc_lower_bound"] == pytest.approx(
        result["roc_auc"], abs=1e-12
    )
    assert result["missingness_auc_upper_bound"] == pytest.approx(
        result["roc_auc"], abs=1e-12
    )
    assert result["missingness_sensitivity_status"] == "complete_marker_observation"


def test_the_observed_auc_does_not_leak_into_the_result_table():
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 2, size=120)
    values = rng.normal(size=120)
    result = evaluate_marker(values, labels, direction=1)
    assert "observed_aligned_auc" not in result


@pytest.mark.parametrize("seed", range(12))
def test_count_only_screen_assessability_matches_full_evaluation(seed: int):
    rng = np.random.default_rng(2000 + seed)
    labels = np.repeat((1, 0), (37 + seed, 83 + 2 * seed))
    values = rng.integers(-4, 5, size=len(labels)).astype(float)
    values[rng.random(len(values)) < (seed % 7) / 10.0] = np.nan
    finite = np.isfinite(values)
    finite_labels = labels[finite]
    if len(np.unique(finite_labels)) < 2:
        raw_auc = float("nan")
    else:
        raw_auc = float(roc_auc_score(finite_labels, values[finite]))
    direction = 1 if raw_auc >= 0.5 else -1

    full = evaluate_marker(values, labels, direction=direction)
    count_only = screen_marker_assessability_from_counts(
        n_positive_input=int(np.sum(labels == 1)),
        n_negative_input=int(np.sum(labels == 0)),
        n_positive_evaluable=int(np.sum((labels == 1) & finite)),
        n_negative_evaluable=int(np.sum((labels == 0) & finite)),
        raw_auc=raw_auc,
        direction=direction,
    )
    keys = (
        "test_yield",
        "positive_test_yield",
        "negative_test_yield",
        "test_yield_gap",
        "overall_missingness",
        "assessability_status",
        "assessability_gate_passed",
        "missingness_auc_lower_bound",
        "missingness_auc_upper_bound",
        "missingness_sensitivity_status",
        "missingness_sensitivity_gate_passed",
        "confirmatory_assessability_gate_passed",
    )
    for key in keys:
        if isinstance(full[key], float):
            assert count_only[key] == pytest.approx(full[key], abs=1e-15)
        else:
            assert count_only[key] == full[key]
