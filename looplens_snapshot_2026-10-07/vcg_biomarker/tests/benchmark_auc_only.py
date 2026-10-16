"""Synthetic weighted metric kernel benchmark, not total analysis runtime."""
import json
from statistics import median
from time import perf_counter

import numpy as np

from vcg_biomarker.mimic.statistics import (
    _prepare_weighted_bootstrap_score_order,
    _weighted_bootstrap_metric_batch,
)


def main():
    rng = np.random.default_rng(73)
    n = 10_000
    labels = rng.integers(0, 2, n)
    scores = rng.normal(size=n)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=25)
    order, groups, positive = _prepare_weighted_bootstrap_score_order(labels, scores)

    def run(fast):
        return _weighted_bootstrap_metric_batch(
            labels, scores, weights, thresholds={}, prevalence=float(labels.mean()),
            score_order=order, score_group_starts=groups, sorted_positive=positive,
            auc_only=fast)["roc_auc"]

    np.testing.assert_array_equal(run(False), run(True))
    elapsed = {False: [], True: []}
    for repeat in range(10):
        for fast in ([False, True] if repeat % 2 == 0 else [True, False]):
            start = perf_counter()
            run(fast)
            elapsed[fast].append(perf_counter() - start)
    old, new = median(elapsed[False]), median(elapsed[True])
    print(json.dumps({"patients": n, "draws": 25, "repeats": 10,
                      "full_seconds": old, "auc_only_seconds": new,
                      "kernel_speedup": old / new, "exact_auc": True}, indent=2))


if __name__ == "__main__":
    main()
