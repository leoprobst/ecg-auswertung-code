"""An unadjusted rank correlation reported the calcium-QT sign backwards.

The curated association axis correlates each marker with each context value and
reports the result.  The prespecified control for "calcium shortens the QT" --
one of the most reliable relationships in clinical electrophysiology -- came
back as rho = +0.128 against ionised calcium.

Checked on the raw evidence, 25,909 recordings carrying both an ionised calcium
and a measurable QT:

    QT vs ionised calcium, unadjusted   +0.030
    QT vs heart rate                    -0.667
    heart rate vs ionised calcium       -0.138

Sicker patients have lower calcium and faster hearts, and the rate shortens the
QT by more than the low calcium lengthens it.  Within every heart-rate band the
sign is correct again.  Removing the rate as a covariate turns the reported
figure from +0.029 into -0.085 on the same 25,335 recordings, and strengthens
the potassium relationship from -0.049 to -0.095 on 360,167.

Most VCG markers are rate-dependent and the rate correlates with nearly every
laboratory value in this cohort, so this is not specific to calcium.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from vcg_biomarker.mimic.association_domains import _partial_spearman


def test_a_confounder_strong_enough_to_invert_a_sign_is_removed():
    # Built to the measured shape: the rate drives the marker hard and the
    # target weakly, in opposite directions, and the true effect is negative.
    rng = np.random.default_rng(20260830)
    size = 5000
    rate = rng.normal(size=size)
    target = -0.4 * rate + 0.9 * rng.normal(size=size)
    marker = -0.7 * rate - 0.15 * target + 0.2 * rng.normal(size=size)

    unadjusted = float(stats.spearmanr(marker, target).statistic)
    adjusted, count = _partial_spearman(marker, target, rate)

    assert unadjusted > 0.1, "fixture must reproduce the inverted sign"
    assert adjusted < -0.1, "removing the rate must restore the true sign"
    assert count == size


def test_an_unconfounded_relationship_is_left_where_it_was():
    rng = np.random.default_rng(7)
    size = 4000
    rate = rng.normal(size=size)
    target = rng.normal(size=size)
    marker = 0.6 * target + 0.4 * rng.normal(size=size)

    unadjusted = float(stats.spearmanr(marker, target).statistic)
    adjusted, _ = _partial_spearman(marker, target, rate)
    assert adjusted == pytest.approx(unadjusted, abs=0.05)


def test_a_relationship_that_is_entirely_the_covariate_collapses():
    rng = np.random.default_rng(11)
    size = 4000
    rate = rng.normal(size=size)
    marker = rate + 0.05 * rng.normal(size=size)
    target = rate + 0.05 * rng.normal(size=size)

    assert float(stats.spearmanr(marker, target).statistic) > 0.9
    adjusted, _ = _partial_spearman(marker, target, rate)
    assert abs(adjusted) < 0.2


def test_rows_without_a_heart_rate_are_dropped_not_imputed():
    rng = np.random.default_rng(3)
    size = 600
    rate = rng.normal(size=size)
    target = rng.normal(size=size)
    marker = 0.5 * target + rng.normal(size=size)
    rate[:200] = np.nan
    adjusted, count = _partial_spearman(marker, target, rate)
    assert count == 400
    assert np.isfinite(adjusted)


def test_too_few_usable_rows_gives_no_number_rather_than_a_bad_one():
    marker = np.arange(8, dtype=float)
    adjusted, count = _partial_spearman(marker, marker, marker)
    assert np.isnan(adjusted)
    assert count == 8


def test_a_constant_covariate_cannot_adjust_anything():
    rng = np.random.default_rng(5)
    size = 300
    target = rng.normal(size=size)
    marker = target + rng.normal(size=size)
    adjusted, _ = _partial_spearman(marker, target, np.zeros(size))
    assert np.isnan(adjusted)


def test_a_binary_target_is_adjustable_too():
    # The digoxin control is binary, and the rate confounds it as well.
    rng = np.random.default_rng(13)
    size = 3000
    rate = rng.normal(size=size)
    exposed = (rate + rng.normal(size=size) > 0).astype(float)
    marker = -0.8 * rate + 0.1 * rng.normal(size=size)
    adjusted, count = _partial_spearman(marker, exposed, rate)
    assert count == size
    assert abs(adjusted) < abs(float(stats.spearmanr(marker, exposed).statistic))
