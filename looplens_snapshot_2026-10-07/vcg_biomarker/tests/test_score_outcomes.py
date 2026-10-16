import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.score_outcomes import discover_outcome_score
from vcg_biomarker.mimic.score_research import SearchPolicy, ScoreSpec


def test_continuous_score_keeps_physical_outcome_scale():
    rng = np.random.default_rng(3)
    x = rng.normal(size=120)
    y = 40 + 7 * x + rng.normal(size=120)
    result = discover_outcome_score(
        pd.DataFrame({"qrs_area": x}),
        y,
        np.arange(120),
        endpoint_id="lvef",
        model_group="vcg",
        family=lambda x: x,
        provenance={
            "split": "train",
            "input_hash": "test",
            "outcome_reference_verified": True,
        },
        policy=SearchPolicy(
            outer_folds=2, inner_folds=2, max_components=1, benchmark=False
        ),
    )
    assert result["evaluation"]["metrics"]["r2"] > 0.9
    assert result["score_spec"]["link"] == "identity"
    payload = dict(result["score_spec"])
    payload.pop("formula")
    assert ScoreSpec(**payload).predict(pd.DataFrame({"qrs_area": [0.0]}))[
        0
    ] == pytest.approx(40, abs=1)


def test_missing_survival_followup_is_not_censoring():
    with pytest.raises(ValueError, match="followup"):
        discover_outcome_score(
            pd.DataFrame({"x": [1.0, 2.0]}),
            [0, 1],
            [1, 2],
            time=[1.0, 2.0],
            endpoint_id="death",
            model_group="vcg",
            family=lambda x: x,
            provenance={
                "split": "train",
                "input_hash": "x",
                "outcome_reference_verified": True,
            },
        )


def test_censored_score_is_relative_hazard_not_probability():
    rng = np.random.default_rng(52)
    x = rng.normal(size=200)
    event_time = rng.exponential(np.exp(-x))
    censor_time = rng.exponential(3, size=200)
    result = discover_outcome_score(
        pd.DataFrame({"qrs_area": x}),
        (event_time <= censor_time).astype(int),
        np.arange(200),
        time=np.minimum(event_time, censor_time),
        endpoint_id="mortality",
        model_group="vcg",
        family=lambda x: x,
        provenance={
            "split": "train",
            "input_hash": "synthetic",
            "outcome_reference_verified": True,
            "followup_verified": True,
            "time_zero": "index_ecg",
        },
        policy=SearchPolicy(
            outer_folds=2, inner_folds=2, max_components=1, benchmark=False
        ),
    )
    assert result["score_spec"]["link"] == "log_relative_hazard"
    assert result["evaluation"]["metrics"]["absolute_risk"] == "not_estimated"
    assert result["evaluation"]["metrics"]["harrell_c"] > 0.65
