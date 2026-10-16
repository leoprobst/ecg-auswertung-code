"""R15: the parallel hyperparameter grid selects exactly what the serial one does."""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from vcg_biomarker.mimic import physiological_gap_models as gap
from vcg_biomarker.mimic import resource_grants as grants


def test_parallel_grid_search_is_identical_to_serial(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    rows, columns = 1500, 8
    matrix = rng.normal(size=(rows, columns))
    matrix[rng.random((rows, columns)) < 0.05] = np.nan
    target = np.nan_to_num(matrix[:, 0]) * 2 + rng.normal(size=rows)
    sex = (np.nan_to_num(matrix[:, 1]) + rng.normal(size=rows) > 0).astype(int)
    subjects = rng.integers(0, 400, rows)
    monkeypatch.delenv(grants.GRANT_ENV, raising=False)
    serial = gap._select_hyperparameters_by_grouped_cv(
        matrix, target, subjects, n_folds=3, random_seed=3)
    serial_sex = gap._best_sex_parameter(matrix, sex, subjects, n_folds=3, random_seed=3)

    grant = tmp_path / "grant.json"
    stop = threading.Event()

    def refresh():
        while not stop.is_set():
            grants.atomic_write_json(grant, {
                "owner_pid": os.getpid(), "workers": 4, "ceiling": 4,
                "generation": 1, "sampled_at": time.time()})
            stop.wait(1.0)

    refresh_thread = threading.Thread(target=refresh, daemon=True)
    grants.atomic_write_json(grant, {"owner_pid": os.getpid(), "workers": 4, "ceiling": 4,
                                     "generation": 1, "sampled_at": time.time()})
    refresh_thread.start()
    monkeypatch.setenv(grants.GRANT_ENV, str(grant))
    monkeypatch.setattr(gap, "GRID_PARALLEL_MINIMUM_ROWS", 0)
    try:
        assert gap._grid_workers(10) == 4
        parallel = gap._select_hyperparameters_by_grouped_cv(
            matrix, target, subjects, n_folds=3, random_seed=3)
        parallel_sex = gap._best_sex_parameter(matrix, sex, subjects, n_folds=3, random_seed=3)
    finally:
        stop.set()
        refresh_thread.join()
    for key in ("alpha", "l1_ratio", "selection_r2", "mae", "r2"):
        assert serial[key] == parallel[key], key
    assert np.array_equal(serial["out_of_fold_predictions"],
                          parallel["out_of_fold_predictions"], equal_nan=True)
    assert serial_sex == parallel_sex


def test_block_parallel_repeatability_and_reference_ranges_equal_serial(tmp_path, monkeypatch):
    import pandas as pd

    from vcg_biomarker.mimic.grant_parallel import PARALLEL_ENV
    from vcg_biomarker.mimic.reference_ranges import NormativeCohortPolicy, _ranges_in_blocks
    from vcg_biomarker.mimic.repeatability import repeatability_table_from_parquet

    rng = np.random.default_rng(9)
    markers = [f"m{index}" for index in range(6)]
    pairs = pd.DataFrame({"subject_id": np.arange(80)})
    for marker in markers:
        base = rng.normal(size=80)
        pairs[f"{marker}_first"] = base + rng.normal(scale=0.2, size=80)
        pairs[f"{marker}_second"] = base + rng.normal(scale=0.2, size=80)
    pairs_root = tmp_path / "pairs"
    pairs_root.mkdir()
    pairs.to_parquet(pairs_root / "part-000.parquet", index=False)

    cohort = pd.DataFrame({"study_id": np.arange(900), **{m: rng.normal(size=900) for m in markers}})
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    strata = pd.DataFrame({"study_id": np.arange(900), "subject_id": np.arange(900),
                           "sex_band": "F", "age_band": "40-50", "rate_band": "60-70"})
    policy = NormativeCohortPolicy(bootstrap_iterations=50)

    results = {}
    for setting in ("0", "1"):
        monkeypatch.setenv(PARALLEL_ENV, setting)
        monkeypatch.delenv(grants.GRANT_ENV, raising=False)
        results[setting] = (
            repeatability_table_from_parquet(pairs_root, markers, block_size=2),
            _ranges_in_blocks(cohort_path, strata, markers, policy, transform="kors", block_size=2),
        )
    pd.testing.assert_frame_equal(results["0"][0], results["1"][0], check_exact=True)
    pd.testing.assert_frame_equal(results["0"][1], results["1"][1], check_exact=True)
