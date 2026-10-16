"""Closed-area circularity must use the same closed boundary, not an open arc."""
from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import catalog_biomarkers as catalog


def context(points):
    return SimpleNamespace(_loop2d={}, _loop3d={}, _primitive_cache={},
        vcg_segment=lambda segment: points,
        vcg_segment_origin_corrected=lambda segment: points,
        duration_s=lambda segment: .1)


@pytest.mark.parametrize("plane,axes", catalog.PLANES.items())
@pytest.mark.parametrize("scale,offset", [(1., 0.), (.01, 7.), (10., -2.)])
def test_square_circularity_is_scale_translation_and_traversal_invariant(plane, axes, scale, offset):
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])*scale+offset
    points = np.zeros((4, 3)); points[:, axes] = square
    forward = catalog._loop2d_features(context(points), "qrs", plane)
    reverse = catalog._loop2d_features(context(points[::-1]), "qrs", plane)
    closed = catalog._loop2d_features(context(np.vstack([points, points[0]])), "qrs", plane)
    assert forward["roundness"] == pytest.approx(np.pi/4, abs=1e-9)
    assert reverse["roundness"] == pytest.approx(forward["roundness"])
    assert closed["roundness"] == pytest.approx(forward["roundness"])
    assert forward["perimeter"] == pytest.approx(3*scale)  # observed open path remains unchanged
    assert forward["signed_area"] == pytest.approx(-reverse["signed_area"])


def test_best_fit_and_fixed_plane_circularity_agree_for_a_planar_triangle():
    points = np.array([[0., 0., 0.], [2., 0., 0.], [1., 1., 0.]])
    fixed = catalog._loop2d_features(context(points), "qrs", "frontal_xy")
    best = catalog._loop3d_features(context(points), "qrs")
    expected = 4*np.pi / (2+2*np.sqrt(2))**2
    assert fixed["roundness"] == pytest.approx(expected)
    assert best["roundness_best_fit_plane"] == pytest.approx(expected)


def test_zero_loop_has_no_circularity_instead_of_apparently_normal_zero():
    result = catalog._loop2d_features(context(np.zeros((5, 3))), "qrs", "frontal_xy")
    assert np.isnan(result["roundness"])


def test_open_sampled_circle_is_bounded_and_converges_to_one():
    angle = np.linspace(0, 2*np.pi, 100, endpoint=False)
    points = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(100)])
    result = catalog._loop2d_features(context(points), "t", "frontal_xy")
    assert .999 < result["roundness"] < 1.


def test_family_audit_catches_the_old_circularity_error_in_all_twenty_markers():
    from vcg_biomarker.mimic.biomarker_audit import _PROJECTED_CIRCULARITY, _range_issue
    assert len(_PROJECTED_CIRCULARITY) == 20
    for marker in _PROJECTED_CIRCULARITY:
        assert _range_issue(marker, 4*np.pi/9) == "expected_[0,1]"
        assert _range_issue(marker, np.pi/4) == ""
