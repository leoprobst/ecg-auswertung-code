import pytest
from vcg_biomarker.gui.lab_screen_calibration import pixels_per_mm


@pytest.mark.parametrize('length,expected', [(50, 4), (100, 2), (10, 20), (200, 1)])
def test_physical_calibration_uses_logical_not_device_pixels(length, expected):
    assert pixels_per_mm(length) == expected


@pytest.mark.parametrize('length', [0, -1, 201, float('nan'), float('inf'), True, '50'])
def test_invalid_measurements_rejected(length):
    with pytest.raises(ValueError):
        pixels_per_mm(length)
