import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.mechanism_models import compare_mechanism_adjustment


def cohort():
    rng = np.random.default_rng(812)
    z, e, noise = rng.normal(size=(3, 500))
    return pd.DataFrame({"subject_id": np.arange(500), "split": "train",
                         "marker_value": z + e, "target_value": 3*z + noise, "rv": z})


def compare(frame):
    return compare_mechanism_adjustment(frame, target_kind="continuous",
                                       baseline_covariates=[], additional_covariates=["rv"],
                                       numeric_covariates=["rv"])


def test_common_patients_and_simulated_confounding():
    frame = cohort()
    frame.loc[:49, "rv"] = np.nan
    result = compare(frame)
    assert result["status"] == "estimable"
    assert result["n_index"] == 500 and result["n_common"] == 450
    base, adjusted = result["models"]
    assert base["n"] == adjusted["n"] == 450
    assert abs(adjusted["estimate"]) < abs(base["estimate"])/4
    assert not result["clinical_claim_allowed"]


@pytest.mark.parametrize("invalid", ["locked", "mixed", "duplicate"])
def test_invalid_cohort_rejected_before_fit(invalid):
    frame = cohort()
    if invalid == "duplicate":
        frame.loc[1, "subject_id"] = 0
    elif invalid == "mixed":
        frame.loc[0, "split"] = "validation"
    else:
        frame["split"] = "locked_test"
    with pytest.raises(ValueError):
        compare(frame)


def test_missing_covariate_does_not_fall_back_to_baseline():
    result = compare(cohort().drop(columns="rv"))
    assert result["reason"] == "required_column_missing:rv"
    assert result["models"] == []
