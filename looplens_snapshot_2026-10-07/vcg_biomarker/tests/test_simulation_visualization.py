import numpy as np
import pytest

from vcg_biomarker.simulation.visualization import graph_slice, process_summary, surface_segments


def test_graph_slice_interpolates_actual_edges():
    xyz = np.array([[-1., 0., 0.], [1., 0., 0.], [2., 1., 0.]])
    section = graph_slice(xyz, np.array([[0, 1], [1, 2]]), axis=0, coordinate=0.)
    np.testing.assert_allclose(section.xyz, [[0., 0., 0.]])
    np.testing.assert_allclose(section.interpolate([0., 10., 20.]), [5.])
    with pytest.raises(ValueError):
        graph_slice(xyz, np.array([[0, 9]]), 0, 0.)


def test_triangle_plane_intersection_is_two_points():
    vertices = np.array([[-1., 0., 0.], [1., -1., 0.], [1., 1., 0.]])
    lines = surface_segments(vertices, np.array([[0, 1, 2]]), 0, 0.)
    assert lines.shape == (1, 2, 3)
    np.testing.assert_allclose(lines[0, :, 0], [0., 0.])
    np.testing.assert_allclose(sorted(lines[0, :, 1]), [-.5, .5])


def test_process_summary_distinguishes_activation_from_recovery():
    times = [0., 2., 4., 6.]
    voltage = np.array([[0., 0., 0.], [.8, 0., 0.], [.8, .9, 0.], [.05, .8, 0.]])
    summary = process_summary(times, voltage, np.array([1.5, 3.5, np.nan]), np.array([1, 3, 5]))
    np.testing.assert_allclose(summary['ventricle']['active_fraction'], [0, 1, 1, 0])
    np.testing.assert_allclose(summary['ventricle']['recovered_fraction'], [0, 0, 0, 1])
    np.testing.assert_allclose(summary['ventricle']['new_activation_fraction'], [0, 1, 0, 0])
    np.testing.assert_allclose(summary['atrium']['new_activation_fraction'], [0, 0, 1, 0])
    assert summary['conduction_cable']['new_activation_fraction'].sum() == 0
