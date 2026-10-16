from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.biomarkers import invariant_biomarkers as module


def test_terminal_area_is_additive_not_cancelled_signed_area(monkeypatch):
    t = np.linspace(0., 2 * np.pi, 90)
    points = np.column_stack([np.sin(t), np.sin(2*t), np.zeros_like(t)])
    monkeypatch.setattr(module, "_segment_curve", lambda ctx, segment: points)
    actual = module._p_terminal_area_fraction(None).value
    centered = points - points.mean(axis=0)
    sectors = np.abs(np.cross(centered[:-1], centered[1:])[:, 2])
    expected = sectors[-29:].sum() / sectors.sum()
    assert actual == pytest.approx(expected)
    assert 0 < actual < 1


def test_terminal_area_common_rotation_translation_and_gain(monkeypatch):
    rng = np.random.default_rng(67)
    points = rng.normal(size=(60, 3))
    monkeypatch.setattr(module, "_segment_curve", lambda ctx, segment: points)
    expected = module._p_terminal_area_fraction(None).value
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    points = 12 * points @ rotation + [4, 2, -3]
    assert module._p_terminal_area_fraction(None).value == pytest.approx(expected)


def test_invariant_segments_do_not_bridge_nonfinite_gaps():
    signal = np.ones((20, 3))
    signal[8] = np.nan
    ctx = SimpleNamespace(_primitive_cache={}, origin_corrected_vcg=lambda: signal,
                          bounds=lambda segment: (0, 19),
                          is_segment_evaluable=lambda segment: True)
    assert module._segment_curve(ctx, "p") is None


@pytest.mark.parametrize("scales", [[1., 1., 1.], [2., 1., 1.], [2., 0., 0.]])
def test_nonidentifiable_plane_abstains_under_rotations(monkeypatch, scales):
    points = np.vstack([np.eye(3), -np.eye(3)]) * scales
    rng = np.random.default_rng(68)
    for _ in range(10):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotated = points @ rotation
        monkeypatch.setattr(module, "_segment_curve", lambda ctx, segment: rotated)
        result = module._p_terminal_area_fraction(None)
        assert result.value is None
        assert "non-identifiable" in result.notes


def test_isotropic_within_plane_does_not_require_unique_in_plane_axes(monkeypatch):
    t = np.linspace(0, 2*np.pi, 60, endpoint=False)
    points = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    monkeypatch.setattr(module, "_segment_curve", lambda ctx, segment: points)
    # Equal first/second singular values are fine: only the plane matters.
    assert module._p_terminal_area_fraction(None).value == pytest.approx(19/59)
