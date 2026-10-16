from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.broad_axis_adjustment import (
    PRIMARY_COVARIATES,
    add_adjustment_effect_change,
    add_adjusted_fdr,
    fit_adjusted_association,
)


def _frame(seed: int = 17, n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(64.0, 11.0, n)
    marker = 0.075 * age + rng.normal(0.0, 0.65, n)
    target = rng.binomial(1, 1.0 / (1.0 + np.exp(-(age - 64.0) / 6.0)))
    return pd.DataFrame(
        {
            "study_id": np.arange(n),
            "subject_id": np.arange(n),
            "ecg_time": pd.date_range("2020-01-01", periods=n, freq="min"),
            "marker_value": marker,
            "target_value": target,
            "age": age,
            "sex": np.where(np.arange(n) % 2, "F", "M"),
            "heart_rate_bpm": rng.normal(75.0, 9.0, n),
            "signal_quality_score": rng.uniform(0.7, 1.0, n),
            "care_setting": np.where(np.arange(n) % 3, "inpatient", "ED"),
        }
    )


def test_categories_only_in_missing_outcomes_do_not_create_false_collinearity():
    frame = _frame(n=240)
    frame["target_value"] = np.random.default_rng(11).normal(size=len(frame))
    frame.loc[:39, "care_setting"] = "absent_from_fit"
    frame.loc[:39, "target_value"] = np.nan
    expected = fit_adjusted_association(frame.dropna(subset=["target_value"]), target_kind="continuous")
    actual = fit_adjusted_association(frame, target_kind="continuous")
    assert actual["status"] == expected["status"] == "estimable"
    assert actual["effect"] == pytest.approx(expected["effect"])
    assert actual["n"] == 200


def test_adjustment_reduces_age_confounding():
    frame = _frame()
    unadjusted = fit_adjusted_association(
        frame,
        target_kind="binary",
        covariates=(),
        model_id="unadjusted_model_check",
    )
    adjusted = fit_adjusted_association(
        frame,
        target_kind="binary",
        covariates=PRIMARY_COVARIATES,
        model_id="primary",
    )
    assert unadjusted["status"] == "estimable"
    assert adjusted["status"] == "estimable"
    assert abs(adjusted["estimate"]) < abs(unadjusted["estimate"]) * 0.35


def test_adjustment_retains_an_independent_continuous_effect():
    frame = _frame()
    rng = np.random.default_rng(23)
    frame["target_value"] = (
        0.8 * frame["marker_value"]
        + 0.03 * frame["age"]
        + rng.normal(0.0, 0.5, len(frame))
    )
    result = fit_adjusted_association(
        frame,
        target_kind="continuous",
        covariates=PRIMARY_COVARIATES,
    )
    assert result["status"] == "estimable"
    assert result["effect"] > 0.45
    assert result["p_value"] < 1e-12


def test_perfect_separation_is_a_typed_abstention_without_warning():
    frame = _frame(n=200)
    frame["marker_value"] = frame["target_value"].astype(float)
    result = fit_adjusted_association(
        frame,
        target_kind="binary",
        covariates=(),
    )
    assert result["status"] == "non_estimable"
    assert result["reason"] == "perfect_or_quasi_separation"


def test_an_overflowing_odds_ratio_is_quasi_separation_not_an_infinite_effect(monkeypatch):
    """The fitter returned a finite but absurd log-odds; ``exp`` overflowed.

    Seen on the CCSR axis of the production-path preview. The overflow
    warning ended the stage; without ``-W error`` it would have written an
    infinite effect with a finite confidence bound beside it.
    """
    import warnings

    import vcg_biomarker.mimic.broad_axis_adjustment as module

    class _Fitted:
        params = pd.Series({"marker_z": 800.0})
        bse = pd.Series({"marker_z": 1.0})
        pvalues = pd.Series({"marker_z": 0.01})

        def conf_int(self):
            return pd.DataFrame({0: [795.0], 1: [805.0]}, index=["marker_z"])

    class _GLM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, **kwargs):
            return _Fitted()

    import statsmodels.api as sm

    assert module  # the fitter imports statsmodels.api at call time
    monkeypatch.setattr(sm, "GLM", _GLM)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fit_adjusted_association(_frame(n=200), target_kind="binary", covariates=())
    assert result["status"] == "non_estimable"
    assert result["reason"] == "perfect_or_quasi_separation"


def test_adjusted_fdr_excludes_non_estimable_rows():
    frame = pd.DataFrame(
        [
            {"axis": "lab", "model_id": "primary", "evaluation_split": "validation", "status": "estimable", "p_value": 0.01},
            {"axis": "lab", "model_id": "primary", "evaluation_split": "validation", "status": "estimable", "p_value": 0.04},
            {"axis": "lab", "model_id": "primary", "evaluation_split": "validation", "status": "non_estimable", "p_value": np.nan},
        ]
    )
    result = add_adjusted_fdr(frame)
    assert result.loc[:1, "domain_q_value"].tolist() == [0.02, 0.04]
    assert np.isnan(result.loc[2, "domain_q_value"])


def test_required_covariate_cannot_be_silently_omitted():
    frame = _frame().drop(columns="heart_rate_bpm")
    result = fit_adjusted_association(
        frame,
        target_kind="binary",
        covariates=PRIMARY_COVARIATES,
    )
    assert result["status"] == "non_estimable"
    assert result["reason"] == "required_covariate_unavailable:heart_rate_bpm"
    assert result["covariates_requested"] == "|".join(PRIMARY_COVARIATES)


def test_invariant_covariate_is_declared_but_does_not_block_fit():
    frame = _frame()
    frame["care_setting"] = "inpatient"
    result = fit_adjusted_association(
        frame,
        target_kind="binary",
        covariates=PRIMARY_COVARIATES,
    )
    assert result["status"] == "estimable"
    assert "care_setting" in result["covariates_invariant"].split("|")


def test_missing_categorical_covariate_values_are_complete_case_excluded():
    frame = _frame()
    frame.loc[:9, "sex"] = None
    result = fit_adjusted_association(
        frame,
        target_kind="binary",
        covariates=PRIMARY_COVARIATES,
    )
    assert result["status"] == "estimable"
    assert result["n"] == len(frame) - 10


def test_effect_change_uses_same_scale_unadjusted_reference():
    frame = pd.DataFrame(
        [
            {
                "axis": "lab",
                "target_id": "k",
                "temporal_relation": "undated",
                "feature_id": "marker",
                "evaluation_split": "validation",
                "model_id": "unadjusted_regression",
                "status": "estimable",
                "estimate": 0.8,
            },
            {
                "axis": "lab",
                "target_id": "k",
                "temporal_relation": "undated",
                "feature_id": "marker",
                "evaluation_split": "validation",
                "model_id": "primary",
                "status": "estimable",
                "estimate": 0.2,
            },
        ]
    )
    result = add_adjustment_effect_change(frame)
    primary = result.loc[result["model_id"].eq("primary")].iloc[0]
    assert primary["estimate_change_from_unadjusted"] == pytest.approx(-0.6)
    assert primary["absolute_estimate_attenuation"] == pytest.approx(0.6)
    assert primary["relative_absolute_estimate_attenuation"] == pytest.approx(0.75)
