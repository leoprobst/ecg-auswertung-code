"""Temporal markers must not silently close gaps in the sampled trajectory."""

from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import loop_parametrisation_biomarkers as loop
from vcg_biomarker.core.biomarkers import propagation_sequence_biomarkers as propagation
from vcg_biomarker.core.biomarkers import thesis_derived_biomarkers as thesis
from vcg_biomarker.mimic.marker_plausibility import declared_range
from vcg_biomarker.core.biomarkers import ischemia_thesis_biomarkers as ischemia


@pytest.mark.parametrize("module", [thesis, ischemia])
@pytest.mark.parametrize("confidence", [np.nan, np.inf, -1., 1.1, "bad", None])
def test_invalid_fiducial_confidence_is_not_accepted(module, confidence):
    beat = SimpleNamespace(r_onset=SimpleNamespace(sample=5, confidence=confidence))
    assert module._beat_sample(beat, "r_onset", "qrs") is None


@pytest.mark.parametrize("module", [thesis, ischemia])
@pytest.mark.parametrize("sample", [-1, 2.5, np.nan, np.inf, "bad", None])
def test_invalid_sample_is_not_clamped_or_truncated(module, sample):
    beat = SimpleNamespace(r_onset=SimpleNamespace(sample=sample, confidence=1.))
    assert module._beat_sample(beat, "r_onset", "qrs") is None


@pytest.mark.parametrize("module", [thesis, ischemia])
def test_broken_wave_status_abstains(module):
    def broken_status(wave):
        raise RuntimeError("synthetic unavailable status")
    beat = SimpleNamespace(r_onset=SimpleNamespace(sample=5, confidence=1.),
                           resolved_wave_status=broken_status)
    assert module._beat_sample(beat, "r_onset", "qrs") is None


def test_small_shoulder_does_not_borrow_distant_peak_baseline():
    result = loop.landmarks_from_speed(np.array([0., 10., 9.4, 9.5, 9., 0.]))
    assert result is not None
    assert result["peaks"] == [1]


def test_prominent_second_burst_is_retained():
    result = loop.landmarks_from_speed(np.array([0., 10., 2., 8., 1., 0.]))
    assert result is not None
    assert result["peaks"] == [1, 3]


@pytest.mark.parametrize("reader", [loop._qrs_points, propagation._qrs_points,
                                   lambda ctx: thesis._segment_points(ctx, "t")])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", [0, 10, 20])
def test_missing_sample_abstains_without_compressing_time(reader, bad, position):
    points = np.column_stack([np.linspace(0, 1, 21)] * 3)
    points[position, 1] = bad
    original = points.copy()
    ctx = SimpleNamespace(vcg_segment_origin_corrected=lambda segment: points)
    assert reader(ctx) is None
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("reader", [loop._qrs_points, propagation._qrs_points,
                                   lambda ctx: thesis._segment_points(ctx, "qrs")])
def test_complete_trajectory_keeps_every_sample(reader):
    points = np.column_stack([np.linspace(0, 1, 21)] * 3)
    ctx = SimpleNamespace(vcg_segment_origin_corrected=lambda segment: points)
    np.testing.assert_array_equal(reader(ctx), points)


@pytest.mark.parametrize("marker", [
    "qrs_xy_cross_correlation_peak_lag_ms", "qrs_xz_cross_correlation_peak_lag_ms",
    "qrs_yz_cross_correlation_peak_lag_ms", "qrs_leftward_rightward_peak_order_ms",
    "qrs_posterior_anterior_peak_order_ms", "v6_v1_interpeak_interval_ms",
])
def test_signed_lags_keep_physical_bounds(marker):
    bounds = declared_range(marker, "ms")
    assert bounds is not None
    np.testing.assert_array_equal(
        bounds.violations(np.array([-10001., -174., 0., 174., 10001.])),
        [True, False, False, False, True],
    )


def test_ordinary_duration_still_cannot_be_negative():
    bounds = declared_range("classical_qrs_duration_ms", "ms")
    assert bounds is not None
    assert bounds.violations(np.array([-1.]))[0]


def context_for(points):
    return SimpleNamespace(fs=500., vcg_segment_origin_corrected=lambda segment: points)


def test_temporal_cancellation_does_not_require_simultaneous_opposing_sources():
    points = np.zeros((20, 3))
    points[:10, 0] = 1.
    points[10:, 0] = -1.
    assert propagation._qrs_cancellation_fraction(context_for(points)).value == pytest.approx(1.)


def test_arc_fraction_scalar_invariance_is_not_anisotropic_transfer_invariance():
    points = np.zeros((21, 3))
    points[:11, 0] = np.linspace(0., 1., 11)
    points[11:, 0] = 1.
    points[11:, 1] = np.linspace(.1, 1., 10)
    calculate = lambda x: loop._arc_length_fraction_at_half_time(context_for(x)).value
    assert calculate(points) == pytest.approx(.5)
    assert calculate(3 * points) == pytest.approx(.5)
    assert calculate(points * [2., 1., 1.]) == pytest.approx(2/3)


def test_half_arc_fraction_does_not_imply_uniform_speed():
    steps = np.tile([.1, .9], 10)
    points = np.zeros((21, 3))
    points[1:, 0] = np.cumsum(steps)
    result = loop._arc_length_fraction_at_half_time(context_for(points))
    assert result.value == pytest.approx(.5)
    assert np.std(steps) > .3


def test_arc_scatter_reports_unverified_quality_limits(monkeypatch):
    monkeypatch.setattr(thesis, "_beatwise_arc_fractions", lambda ctx: [.4, .5, .6])
    result = thesis._arc_fraction_beatwise_variability(SimpleNamespace())
    assert "contributing_beats=3" in result.notes
    assert "beat_class_homogeneity=not_verified" in result.notes
    assert "absolute_snr_gate=not_validated" in result.notes


def test_st_dispersion_reports_unverified_quality_limits(monkeypatch):
    monkeypatch.setattr(ischemia, "_beat_st_directions", lambda ctx: [np.array([1., 0., 0.])] * 3)
    result = ischemia._st_direction_dispersion(SimpleNamespace())
    assert result.value == 0.
    assert "contributing_beats=3" in result.notes
    assert "absolute_snr_gate=not_validated" in result.notes
