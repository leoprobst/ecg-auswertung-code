from dataclasses import replace
import numpy as np

from vcg_biomarker.tests.test_score_research import fixture_data
from vcg_biomarker.mimic.score_research import SearchPolicy, discover_binary_score
from vcg_biomarker.mimic.combination_search import (
    beam_search_combinations,
    SearchBudget,
)


def test_process_score_search_equals_serial():
    x, y, groups = fixture_data()
    policy = SearchPolicy(
        outer_folds=2,
        inner_folds=2,
        max_components=2,
        minimum_class_patients=10,
        bootstrap_iterations=2,
        benchmark=True,
    )
    kwargs = dict(
        endpoint_id="synthetic",
        model_group="vcg",
        family=lambda name: name,
        provenance={"split": "train", "input_hash": "synthetic"},
    )
    first = discover_binary_score(x, y, groups, policy=policy, **kwargs)
    updates = []
    second = discover_binary_score(
        x,
        y,
        groups,
        policy=replace(policy, workers=2),
        progress=lambda *args: updates.append(args),
        **kwargs,
    )
    assert first["score_spec"] == second["score_spec"]
    assert first["folds"] == second["folds"]
    np.testing.assert_array_equal(first["oof_predictions"], second["oof_predictions"])
    assert first["evaluation"] == second["evaluation"]
    assert ("outer_evaluation", 2, 2) in updates


def test_parallel_beam_equals_serial_with_missingness():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 600)
    arrays = {f"feature_{i}": rng.normal(size=600) + 0.2 * y for i in range(6)}
    arrays["feature_0"][::10] = np.nan
    kwargs = dict(
        family_for_feature=lambda x: x,
        budget=SearchBudget(max_depth=3),
        minimum_rows=30,
        beam_width=3,
    )
    a = beam_search_combinations(arrays, y, np.arange(600), worker_count=1, **kwargs)
    b = beam_search_combinations(arrays, y, np.arange(600), worker_count=3, **kwargs)
    for first, second in zip(a, b):
        assert first.best.components == second.best.components
        assert first.best.confirmation_auc == second.best.confirmation_auc
