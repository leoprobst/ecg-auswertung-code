"""The VCG views read the same recording the markers read."""

from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core import views


FS = 500.0


def _helix(n=100, turns=1.0):
    t = np.linspace(0.0, 2 * np.pi * turns, n)
    return np.column_stack([np.cos(t) - 1.0, 0.5 * np.sin(t), 0.2 * t])


def _fast_slow_fast(n_fast=30, n_slow=40):
    """A loop that runs quickly, crawls, then runs quickly again."""

    a = np.linspace(0.0, 1.0, n_fast)[:, None] * np.array([1.0, 0.0, 0.0])
    b = a[-1] + np.linspace(0.0, 0.05, n_slow)[:, None] * np.array([0.0, 1.0, 0.0])
    c = b[-1] + np.linspace(0.0, 1.0, n_fast)[:, None] * np.array([0.0, 0.0, 1.0])
    return np.vstack([a, b, c])


def test_time_marks_are_placed_every_two_milliseconds():
    loop = views.time_marked_loop(_helix(200), FS, mark_ms=2.0)
    assert loop.points.shape == (200, 3)
    assert len(loop.times_ms) == 200 and loop.times_ms[-1] == pytest.approx(199 * 2.0)
    assert len(loop.speed) == 200
    assert np.all(np.diff(loop.mark_indices) == 1)  # 2 ms at 500 Hz is one sample


def test_isochronal_crowding_reads_the_slow_middle():
    crowd = views.isochronal_crowding(_fast_slow_fast(), FS, segments=5)
    assert crowd.arc_share.sum() == pytest.approx(1.0)
    assert crowd.time_share.sum() == pytest.approx(1.0)
    # the middle segments cover almost no arc in their time
    assert crowd.crowding[2] > 5.0
    assert crowd.crowding[0] < 1.0 and crowd.crowding[-1] < 1.0


def test_the_velocity_curve_places_landmarks_by_the_marker_rule():
    from vcg_biomarker.core.biomarkers.loop_parametrisation_biomarkers import landmarks_from_speed

    points = _fast_slow_fast()
    curve = views.spatial_velocity_curve(points, FS)
    expected = landmarks_from_speed(views.smoothed_speed(points, FS))
    assert tuple(expected["peaks"]) == curve.peak_indices
    assert tuple(expected["troughs"]) == curve.trough_indices
    assert curve.initial_burst_ms is not None and curve.initial_burst_ms > 0


def test_the_chronotopocardiogram_reads_direction_against_time():
    points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.001, 0.0, 0.0]])
    chrono = views.chronotopocardiogram(points, FS, minimum_share=0.05)
    assert chrono.azimuth_deg[0] == pytest.approx(0.0)      # +X, leftward
    assert chrono.azimuth_deg[1] == pytest.approx(90.0)     # +Z, posterior
    assert chrono.elevation_deg[2] == pytest.approx(0.0)    # +Y, inferior
    assert np.isnan(chrono.azimuth_deg[3])                  # too small to point


def test_the_arc_length_clock_matches_the_marker():
    symmetric = views.arc_length_clock(_fast_slow_fast())
    assert symmetric.arc_fraction[0] == pytest.approx(0.0) and symmetric.arc_fraction[-1] == pytest.approx(1.0)
    assert np.all(np.diff(symmetric.arc_fraction) >= -1e-12)
    assert symmetric.fraction_at_half_time == pytest.approx(0.5)
    # a loop that runs first and crawls afterwards has most of its arc behind
    # it at half time -- the early-fast, late-slow reading of the marker
    fast = np.linspace(0.0, 1.0, 30)[:, None] * np.array([1.0, 0.0, 0.0])
    crawl = fast[-1] + np.linspace(0.0, 0.05, 70)[:, None] * np.array([0.0, 1.0, 0.0])
    front_loaded = views.arc_length_clock(np.vstack([fast, crawl]))
    assert front_loaded.fraction_at_half_time > 0.9


def test_the_visibility_map_has_the_criterions_blind_spots():
    vis = views.criterion_visibility_map(st_vector=[0.0, 0.0, 0.3], azimuth_steps=36, elevation_steps=19)
    assert vis.visibility.shape == (19, 36)
    assert vis.visibility.max() <= 1.0 + 1e-9 and vis.visibility.max() > 0.95
    # posterior (azimuth +90, horizontal) is a blind spot; inferior (elevation 0) is bright
    posterior = vis.visibility[9, np.argmin(np.abs(vis.azimuth_deg - 90.0))]
    inferior = vis.visibility[0, :].max()
    assert posterior < 0.2 and inferior > 0.9
    assert vis.st_azimuth_deg == pytest.approx(90.0) and vis.st_visibility < 0.2


def test_gradient_axes_shares_sum_to_one():
    axes = views.gradient_axes(qrs_area=[1.0, 2.0, -2.0], t_area=[0.0, 0.0, 0.0], svg=[3.0, 0.0, 4.0])
    assert axes.shares["qrs_area"].sum() == pytest.approx(1.0)
    assert axes.magnitudes["svg"] == pytest.approx(5.0)
    assert np.isnan(axes.shares["t_area"]).all()


def test_the_discordance_split_matches_the_decision_marker():
    split = views.discordance_split([-0.3, 0.4, 0.0], [5.0, 0.0, 0.0])
    assert split.discordant_mv == pytest.approx(0.3)
    assert split.beyond_mv == pytest.approx(0.4)
    assert split.beyond_share == pytest.approx(0.8)
    assert views.discordance_split([0.1, 0.0, 0.0], [0.0, 0.0, 0.0]) is None


def test_render_views_produces_a_six_panel_figure():
    fig = views.render_views(_helix(120), _helix(80), FS, st_vector=[0.0, 0.0, 0.3], title="")
    try:
        assert len(fig.axes) >= 6
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_circular_plot_seam_preserves_samples_without_false_swing():
    times, angles = views.wrapped_angle_plot_data([0, 1, 2], [179, -179, -175])
    np.testing.assert_equal(angles, [179, np.nan, -179, -175])
    np.testing.assert_equal(times, [0, np.nan, 1, 2])


def test_research_labels_and_backend_are_preserved():
    import matplotlib
    backend = matplotlib.get_backend()
    fig = views.render_views(_helix(), None, FS)
    assert matplotlib.get_backend() == backend
    assert "Forschungsansicht" in fig._suptitle.get_text()
    assert "Dower-Geometrieproxy" in fig.axes[5].get_title()
