"""Synthetic preparation-only benchmark; no claim about total model-fit time."""
import json
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    n = 100_000
    covariates = ["age", "sex"]
    groups = [covariates] + [covariates + [f"lead_{i}_{j}" for j in range(4)] for i in range(12)]
    columns = list(dict.fromkeys(c for group in groups for c in group))
    values = pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)
    train = pd.Series(np.arange(n) < n * .7)
    valid = ~train

    def previous():
        return [(values.loc[train, names].to_numpy(), values.loc[valid, names].to_numpy()) for names in groups]

    def cached():
        tr, va = values.loc[train].to_numpy(), values.loc[valid].to_numpy()
        positions = {name: i for i, name in enumerate(values.columns)}
        return [(tr[:, [positions[c] for c in names]], va[:, [positions[c] for c in names]]) for names in groups]

    for old, new in zip(previous(), cached(), strict=True):
        for a, b in zip(old, new, strict=True):
            np.testing.assert_array_equal(a, b)
    timings = {}
    for name, function in (("previous", previous), ("cached", cached)):
        measurements = []
        for _ in range(7):
            start = perf_counter()
            function()
            measurements.append(perf_counter() - start)
        timings[name] = median(measurements)
    print(json.dumps({"patients": n, "models": len(groups), "repeats": 7,
                      "median_seconds": timings,
                      "preparation_speedup": timings["previous"] / timings["cached"],
                      "exact_arrays": True}, indent=2))


if __name__ == "__main__":
    main()
