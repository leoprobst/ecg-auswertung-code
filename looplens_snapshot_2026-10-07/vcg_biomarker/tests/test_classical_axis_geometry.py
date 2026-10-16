import numpy as np
import pytest

from vcg_biomarker.mimic.classical import _frontal_axis_surrogate


def leads(x, y):
    return {'I': {'r_height': max(x, 0), 's_depth': max(-x, 0)},
            'aVF': {'r_height': max(y, 0), 's_depth': max(-y, 0)}}


@pytest.mark.parametrize('angle', [-150, -90, -45, -30, 0, 30, 60, 90, 150])
def test_einthoven_projected_vectors_recover_angle(angle):
    rad = np.radians(angle)
    qrs = leads(np.cos(rad), np.sqrt(3)/2*np.sin(rad))
    assert _frontal_axis_surrogate(qrs) == pytest.approx(angle)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -999, 999])
def test_invalid_machine_axis_not_silently_replaced(value):
    assert _frontal_axis_surrogate(leads(1, 1), value) is None


def test_zero_and_missing_are_not_normal_axis():
    assert _frontal_axis_surrogate(leads(0, 0)) is None
    qrs = leads(1, 1)
    qrs['I']['r_height'] = float('nan')
    assert _frontal_axis_surrogate(qrs) is None
    assert _frontal_axis_surrogate(qrs, 45) == 45
