import numpy as np
import pytest

from vcg_biomarker.mimic.statistics import _weighted_bootstrap_metric_batch


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("single_class", [False, True])
def test_auc_only_exactly_matches_full_metrics(tied, single_class):
    rng = np.random.default_rng(73)
    labels = rng.integers(0, 2, 200)
    if single_class:
        labels[:] = 0
    scores = rng.normal(size=200)
    if tied:
        scores = scores.round(0)
    weights = rng.multinomial(200, np.full(200, 1 / 200), size=12)
    weights[0] = 0
    kwargs = dict(thresholds={}, prevalence=float(labels.mean()))
    full = _weighted_bootstrap_metric_batch(labels, scores, weights, **kwargs)
    fast = _weighted_bootstrap_metric_batch(labels, scores, weights, auc_only=True, **kwargs)
    assert set(fast) == {"roc_auc", "_valid"}
    for key in fast:
        np.testing.assert_array_equal(fast[key], full[key])


def test_auc_only_rejects_thresholds():
    with pytest.raises(ValueError, match="threshold"):
        _weighted_bootstrap_metric_batch(
            np.array([0, 1]), np.array([0., 1.]), np.ones((1, 2)),
            thresholds={"clinical": .5}, prevalence=.5, auc_only=True)
