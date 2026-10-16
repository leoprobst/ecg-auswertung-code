import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.broad_axis_adjustment import _design_matrix, fit_adjusted_association


def test_explicit_echo_covariate_is_one_numeric_column():
    frame = pd.DataFrame({"marker_value": [1., 3., 2., 4.],
                          "rv_size": [2., 3., 4., None]})
    design, used, invariant, missing, complete = _design_matrix(
        frame, marker_column="marker_value", covariates=["rv_size"],
        numeric_covariates=["rv_size"],
    )
    assert list(design.columns) == ["marker_z", "rv_size"]
    assert used == ["rv_size"] and not invariant and not missing
    assert complete.tolist() == [True, True, True, False]


def test_numeric_mechanism_adjustment_matches_existing_numeric_path():
    rng = np.random.default_rng(42)
    x, z, noise = rng.normal(size=(3, 500))
    frame = pd.DataFrame({"marker_value": x, "target_value": x + 2*z + noise,
                          "rv_size": z, "age": z})
    actual = fit_adjusted_association(frame, target_kind="continuous",
                                     covariates=["rv_size"], numeric_covariates=["rv_size"])
    reference = fit_adjusted_association(frame, target_kind="continuous", covariates=["age"])
    assert actual["status"] == reference["status"] == "estimable"
    for key in ("effect", "standard_error", "ci_low", "ci_high", "p_value"):
        assert actual[key] == pytest.approx(reference[key])


def test_undeclared_numeric_covariate_rejected():
    with pytest.raises(ValueError, match="must be declared"):
        fit_adjusted_association(pd.DataFrame(), target_kind="binary",
                                 covariates=[], numeric_covariates=["rv_size"])
