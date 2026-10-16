from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.score_research import (
    SearchPolicy,
    ScoreSpec,
    NotEstimable,
    discover_binary_score,
    evaluate_frozen_score,
    probability_metrics,
    _splits,
)
from vcg_biomarker.mimic.decision_research import TreatmentProfile, PROFILES
from vcg_biomarker.mimic.score_research import operating_points


def fixture_data():
    rng = np.random.default_rng(32)
    x = rng.normal(size=(120, 3))
    y = (x[:, 0] + rng.normal(size=120) > 0).astype(int)
    return (
        pd.DataFrame(x, columns=["qrs_t_angle", "qrs_area", "t_area"]),
        y,
        np.arange(120),
    )


def test_formula_and_points_are_explicit_and_abstain_when_all_missing():
    spec = ScoreSpec("ef", "vcg", ("x",), (2.0,), (2.0,), (2.0,), (1.0,), 0.0, {})
    frame = pd.DataFrame({"x": [2, 4, np.nan]})
    assert spec.predict(frame)[0] == 0.5
    assert spec.predict(frame)[1] == pytest.approx(1 / (1 + np.exp(-1)))
    assert np.isnan(spec.predict(frame)[2])
    assert np.allclose(
        spec.predict(frame)[:2], spec.predict(frame, point_step=0.25)[:2]
    )
    with pytest.raises(ValueError):
        replace(spec, clinical_claim_allowed=True)


def test_nested_search_repeatable_and_frozen_validation_cannot_fit():
    x, y, groups = fixture_data()
    policy = SearchPolicy(
        outer_folds=3,
        inner_folds=2,
        max_components=1,
        minimum_class_patients=10,
        bootstrap_iterations=5,
        benchmark=False,
    )
    kwargs = dict(
        endpoint_id="ef",
        model_group="vcg",
        family=lambda x: x,
        policy=policy,
        provenance={"split": "train", "input_hash": "synthetic"},
    )
    first = discover_binary_score(x, y, groups, **kwargs)
    second = discover_binary_score(x, y, groups, **kwargs)
    assert np.array_equal(first["oof_predictions"], second["oof_predictions"])
    assert first["evaluation"]["metrics"]["auc"] > 0.7
    payload = dict(first["score_spec"])
    payload.pop("formula")
    spec = ScoreSpec(**payload)
    before = spec.to_dict()
    evaluate_frozen_score(
        spec,
        x,
        1 - y,
        groups + 1000,
        training_patient_ids=groups,
        provenance={"split": "validation", "input_hash": "v"},
        iterations=2,
    )
    assert spec.to_dict() == before
    with pytest.raises(ValueError, match="patient_leakage"):
        evaluate_frozen_score(
            spec,
            x,
            y,
            groups,
            training_patient_ids=groups,
            provenance={"split": "validation", "input_hash": "v"},
        )


def test_group_splits_never_separate_repeated_patients():
    groups = np.repeat(np.arange(30), 2)
    y = np.repeat(np.arange(30) % 2, 2)
    for fit, test in _splits(y, groups, 3, 2):
        assert not set(groups[fit]) & set(groups[test])


def test_capacity_waves_preserve_real_nested_score_predictions(monkeypatch):
    from vcg_biomarker.mimic import resources
    x, y, groups = fixture_data()
    policy = SearchPolicy(outer_folds=3, inner_folds=2, max_components=1,
                          minimum_class_patients=10, bootstrap_iterations=2,
                          benchmark=False, workers=1)
    arguments = dict(endpoint_id="synthetic", model_group="vcg", family=lambda name: name,
                     provenance={"split": "train", "input_hash": "synthetic-capacity"})
    serial = discover_binary_score(x, y, groups, policy=policy, **arguments)
    monkeypatch.setattr(resources, "recommend_dynamic_workers", lambda *a, **kw: 2)
    parallel = discover_binary_score(x, y, groups, policy=replace(policy, workers=2), **arguments)
    np.testing.assert_array_equal(serial["oof_predictions"], parallel["oof_predictions"])
    assert serial["score_spec"] == parallel["score_spec"]
    assert parallel["execution"]["admitted_workers_by_wave"] == [2, 1]


def test_locked_or_unspecified_provenance_rejected():
    x, y, groups = fixture_data()
    for split in ("validation", "locked_test", None):
        with pytest.raises(ValueError, match="train_provenance"):
            discover_binary_score(
                x,
                y,
                groups,
                endpoint_id="ef",
                model_group="vcg",
                family=lambda x: x,
                provenance={"split": split, "input_hash": "x"},
            )


def test_non_estimable_not_a_fabricated_score():
    x, y, groups = fixture_data()
    with pytest.raises(NotEstimable):
        discover_binary_score(
            x,
            np.zeros_like(y),
            groups,
            endpoint_id="ef",
            model_group="vcg",
            family=lambda x: x,
            provenance={"split": "train", "input_hash": "x"},
        )


def test_calibration_separation_is_an_explicit_abstention():
    result = probability_metrics(
        np.array([0] * 20 + [1] * 20),
        np.array([0.01] * 20 + [0.99] * 20),
        np.arange(40),
        iterations=2,
    )
    assert result["calibration_status"] == "non_estimable"
    assert result["auc"] == 1


def test_decision_curve_requires_prespecified_thresholds():
    assert operating_points([0, 1], [0.1, 0.9])["status"] == "not_computed"
    point = operating_points([0, 1], [0.1, 0.9], [0.5])["operating_points"][0]
    assert point["sensitivity"] == point["specificity"] == 1
    assert point["net_benefit"] == 0.5
    with pytest.raises(ValueError):
        operating_points([0, 1], [0.1, 0.9], [1.0])


def test_treatment_execution_is_not_benefit():
    profile = TreatmentProfile("pci", "treatment_benefit_research", treatment="PCI")
    result = profile.readiness(
        verified_fields=PROFILES["pci"], record_level_verified=True
    )
    assert result["status"] == "non_estimable"
    assert "missing_protocol:comparator" in result["reasons"]
    assert not result["clinical_claim_allowed"]
    assert not result["performed_treatment_is_benefit_label"]


def test_full_three_component_search_is_bounded_and_numerically_evaluable():
    rng = np.random.default_rng(84)
    x = rng.normal(size=(180, 3))
    y = (x[:, 0] + x[:, 1] + rng.normal(size=180) * 0.5 > 0).astype(int)
    frame = pd.DataFrame(x, columns=["qrs_area", "t_area", "qrs_t_angle"])
    result = discover_binary_score(
        frame,
        y,
        np.arange(180),
        endpoint_id="ef",
        model_group="vcg",
        family=lambda name: name,
        policy=SearchPolicy(
            outer_folds=2,
            inner_folds=2,
            max_components=3,
            minimum_class_patients=10,
            bootstrap_iterations=2,
            benchmark=True,
        ),
        provenance={"split": "train", "input_hash": "synthetic"},
    )
    assert result["evaluation"]["metrics"]["auc"] > 0.7
    assert max(len(fold["components"]) for fold in result["folds"]) <= 3
    assert result["benchmark"]["n_predicted"] == 180


def test_fixed_clinical_baseline_stays_in_every_fold():
    x, y, groups = fixture_data()
    x["context_age"] = np.arange(len(x)) / 10 + 40
    result = discover_binary_score(
        x,
        y,
        groups,
        endpoint_id="ef",
        model_group="clinical_vcg",
        family=lambda name: name,
        fixed_components=("context_age",),
        policy=SearchPolicy(
            outer_folds=2,
            inner_folds=2,
            max_components=1,
            minimum_class_patients=10,
            bootstrap_iterations=2,
            benchmark=False,
        ),
        provenance={
            "split": "train",
            "input_hash": "synthetic",
            "context_available_at_decision_verified": True,
        },
    )
    assert all("context_age" in fold["components"] for fold in result["folds"])
    assert "context_age" in result["score_spec"]["components"]
