"""Physical sampling contracts, independent of clinical labels."""

from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.temporal_sampling import speed_smoothing_sigma_samples
from vcg_biomarker.core.views import smoothed_speed
from vcg_biomarker.core.biomarkers.loop_parametrisation_biomarkers import _speed
from vcg_biomarker.core.biomarkers import conduction_uniformity_biomarkers as uniformity


@pytest.mark.parametrize("fs,sigma", [(250, .5), (500, 1.), (1000, 2.), (2000, 4.)])
def test_sigma_is_two_milliseconds(fs, sigma):
    assert speed_smoothing_sigma_samples(fs) == sigma


@pytest.mark.parametrize("fs", [0, -1, np.nan, np.inf])
def test_invalid_sampling_rate_is_rejected(fs):
    with pytest.raises(ValueError):
        speed_smoothing_sigma_samples(fs)


def trajectory(fs):
    t = np.arange(round(.16 * fs) + 1) / fs
    points = np.column_stack((np.sin(2*np.pi*7*t), np.cos(2*np.pi*7*t), .4*np.sin(2*np.pi*3*t)))
    return t, points


@pytest.mark.parametrize("fs", [250, 500, 1000])
def test_marker_and_view_share_speed(fs):
    _, points = trajectory(fs)
    ctx = SimpleNamespace(fs=fs, vcg_segment_origin_corrected=lambda segment: points)
    np.testing.assert_allclose(_speed(ctx), smoothed_speed(points, fs), rtol=0, atol=0)


def test_smooth_trajectory_speed_agrees_across_sampling_rates():
    # Compare interval midpoints, away from Gaussian boundary handling.
    target = np.linspace(.02, .14, 81)
    curves = []
    for fs in [250, 500, 1000]:
        t, points = trajectory(fs)
        curves.append(np.interp(target, (t[:-1]+t[1:])/2, smoothed_speed(points, fs)))
    np.testing.assert_allclose(curves[0], curves[2], rtol=.015)
    np.testing.assert_allclose(curves[1], curves[2], rtol=.015)


@pytest.mark.parametrize("fs,length,evaluable", [(500,4,True),(1000,4,False),(1000,8,True)])
def test_cv_requires_physical_duration_and_sample_count(monkeypatch, fs, length, evaluable):
    import vcg_biomarker.core.biomarkers.loop_parametrisation_biomarkers as loop
    values = np.r_[1., np.full(length, .1), 1.]
    monkeypatch.setattr(loop, "_speed", lambda ctx: values)
    result = uniformity._slow_run_speed_cv(SimpleNamespace(fs=fs))
    assert (result.value is not None) is evaluable


def test_slow_run_cv_can_exceed_one_without_being_an_invalid_fraction(monkeypatch):
    import vcg_biomarker.core.biomarkers.loop_parametrisation_biomarkers as loop
    values = np.array([1., .001, .001, .001, .24, 1.])
    monkeypatch.setattr(loop, "_speed", lambda ctx: values)
    result = uniformity._slow_run_speed_cv(SimpleNamespace(fs=500))
    assert result.value > 1.
    spec = next(s for s in uniformity.CONDUCTION_UNIFORMITY_SPECS
                if s.name == "qrs_slow_run_speed_cv")
    assert spec.unit == "dimensionless"
