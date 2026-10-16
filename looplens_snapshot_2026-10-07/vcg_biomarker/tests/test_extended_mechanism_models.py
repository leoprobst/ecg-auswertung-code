import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.extended_mechanism_models import (
    pr_geometry_coupling, electromechanical_residuals, within_between_echo_model,
)


def test_PR_slope_and_error_variance_abstention():
    x = np.linspace(100, 200, 100)
    y = 2*x + 7
    assert pr_geometry_coupling(x, y)["status"] == "non_estimable"
    result = pr_geometry_coupling(x, y, pr_error_variance_ms2=0., error_covariance=[0.])
    assert result["slope_per_ms"] == pytest.approx([2.])
    assert pr_geometry_coupling(x, y, pr_error_variance_ms2=1e6, error_covariance=[0.])["status"] == "non_estimable"


def test_PR_known_error_correction_recovers_latent_slope():
    # Orthogonal centered errors give exactly known finite-sample moments.
    x = np.linspace(-1, 1, 100)
    error = np.cos(np.linspace(0, 2*np.pi, 100))
    error -= error.mean()
    error -= x * (x @ error) / (x @ x)
    result = pr_geometry_coupling(150 + x + error, 3*x,
        pr_error_variance_ms2=np.var(error, ddof=1), error_covariance=[0.])
    assert result["slope_per_ms"] == pytest.approx([3.])
    assert result["naive_slope_per_ms"][0] < 3.


def residual_frame():
    rng = np.random.default_rng(7)
    n = 500
    x = rng.normal(size=n)
    return pd.DataFrame({"subject_id": np.arange(n), "split": ["train"]*400 + ["validation"]*100,
        "x": x, "m": 3*x + rng.normal(size=n), "m2": -x + rng.normal(size=n)})


def test_residual_fit_never_uses_validation_and_keeps_signed_components():
    frame = residual_frame()
    original = frame.copy(deep=True)
    out, model = electromechanical_residuals(frame, predictors=["x"], mechanical=["m", "m2"])
    frame.loc[frame.split.eq("validation"), ["m", "m2"]] += 100
    changed, model2 = electromechanical_residuals(frame, predictors=["x"], mechanical=["m", "m2"])
    assert model == model2
    assert changed.residual_norm.tail(100).mean() > out.residual_norm.tail(100).mean()
    assert "m_whitened_component" in out and "m2_whitened_component" in out
    assert set(out.role) == {"model_fit", "covariance_calibration", "validation"}
    pd.testing.assert_frame_equal(original, residual_frame())


@pytest.mark.parametrize("problem", ["locked", "duplicate"])
def test_residual_identity_safety(problem):
    frame = residual_frame()
    if problem == "locked":
        frame.loc[0, "split"] = "locked_test"
    else:
        frame.loc[1, "subject_id"] = 0
    with pytest.raises(ValueError):
        electromechanical_residuals(frame, predictors=["x"], mechanical=["m"])


def test_residual_singular_covariance_abstains():
    frame = residual_frame()
    frame["m2"] = frame.m
    _, model = electromechanical_residuals(frame, predictors=["x"], mechanical=["m", "m2"])
    assert model["reason"] == "singular_residual_covariance"


def test_within_between_recovers_opposite_effects():
    rng = np.random.default_rng(10)
    n, repeats = 100, 5
    patient_mean = np.repeat(rng.normal(size=n), repeats)
    delta = rng.normal(size=(n, repeats))
    delta -= delta.mean(axis=1)[:, None]
    delta = delta.ravel()
    frame = pd.DataFrame({"subject_id": np.repeat(np.arange(n), repeats), "study_id": np.arange(n*repeats),
        "split": "train", "marker_value": patient_mean + delta,
        "echo_value": 2*delta - 3*patient_mean + rng.normal(size=n*repeats)*.1})
    fit = within_between_echo_model(frame)
    assert fit["status"] == "estimable"
    assert [t["estimate"] for t in fit["terms"]] == pytest.approx([2., -3.], abs=.05)
    frame.loc[0, "split"] = "validation"
    with pytest.raises(ValueError):
        within_between_echo_model(frame)
