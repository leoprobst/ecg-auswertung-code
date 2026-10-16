"""Physical counterexamples, not validation of COPD diagnosis."""
import numpy as np
import pytest
from vcg_biomarker.core.vcg_transform import loop_dominant_axis, axis_angle_deg


def angle(matrix):
    return axis_angle_deg(loop_dominant_axis(matrix, 0, 20),
                          loop_dominant_axis(matrix, 21, 41))


def loops():
    t = np.linspace(-1, 1, 21)[:, None]
    return np.vstack([t * [1., 0., 0.], t * [1., 1., 0.]])


def test_common_rotation_gain_and_baseline_preserve_relative_axis():
    rng = np.random.default_rng(17)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    original = loops()
    assert angle(original) == pytest.approx(45.)
    assert angle(3.2 * original @ rotation + [7., -4., 1.]) == pytest.approx(45.)


def test_anisotropic_conduction_can_change_angle_without_source_change():
    assert angle(loops() @ np.diag([2., 1., 1.])) == pytest.approx(np.degrees(np.arctan(.5)))


def test_axis_cannot_distinguish_polarity_reversal():
    original = loops()
    reversed_p = original.copy()
    reversed_p[:21] *= -1
    assert angle(reversed_p) == pytest.approx(angle(original))
