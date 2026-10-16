import numpy as np
import pytest
from types import SimpleNamespace

from vcg_biomarker.core.biomarkers import invariant_biomarkers as markers
from vcg_biomarker.mimic.marker_plausibility import declared_range


def series(duration, dt=0.5, frequency=0.2):
    times = np.arange(0, duration, dt)
    angle = 0.2 * np.sin(2 * np.pi * frequency * times)
    return times, np.column_stack([np.cos(angle), np.sin(angle), np.zeros(len(times))])


@pytest.mark.parametrize("frequency", [0.04, 0.1, 0.2, 0.3, 0.6, 0.8])
def test_modulation_is_subset_integral_without_clipping(monkeypatch, frequency):
    times, vectors = series(60, frequency=frequency)
    monkeypatch.setattr(markers, "_beat_axis_series", lambda _: (times, vectors))
    result = markers._respiratory_modulation_depth(None)
    signal = vectors @ (vectors.mean(axis=0) / np.linalg.norm(vectors.mean(axis=0)))
    signal -= signal.mean()
    frequencies = np.unique(np.r_[np.linspace(0.02, 1.0, 192), 0.1, 0.6])
    phase = 2 * np.pi * np.outer(frequencies, times)
    power = (np.cos(phase) @ signal)**2 + (np.sin(phase) @ signal)**2
    areas = np.diff(frequencies) * (power[:-1] + power[1:]) / 2
    selected = (frequencies[:-1] >= 0.1) & (frequencies[1:] <= 0.6)
    assert result.value == pytest.approx(areas[selected].sum() / areas.sum())
    assert 0 <= result.value <= 1


@pytest.mark.parametrize("duration,dt", [(10, .5), (60, 1.0)])
def test_unsupported_duration_or_band_abstains(monkeypatch, duration, dt):
    monkeypatch.setattr(markers, "_beat_axis_series", lambda _: series(duration, dt))
    assert markers._respiratory_modulation_depth(None).value is None


@pytest.mark.parametrize("name,unit,value", [
    ("qrs_sagittal_yz_median_strip_slenderness", "dimensionless", 1.049354),
    ("strip_path_length_per_beat", "mV", 148.844665),
])
def test_accumulated_path_and_shape_ratio_not_amplitude_or_fraction(name, unit, value):
    bounds = declared_range(name, unit)
    assert bounds.lower == 0 and bounds.upper is None
    assert not bounds.violations(np.array([value]))[0]
    assert bounds.violations(np.array([-0.01]))[0]


def test_freeze_exposes_unknown_coverage_without_fabricated_pass():
    from vcg_biomarker.mimic.marker_plausibility import plausibility_contract
    assert plausibility_contract("unknown", None)["coverage"] == "not_defined"
    assert plausibility_contract("strip_path_length_per_beat", "mV")["measurement_kind"] == "accumulated_path_per_beat"
    from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
    rows = marker_freeze_payload()["markers"]
    assert rows and all("plausibility_coverage" in row for row in rows)
    assert all(row["plausibility_coverage"]["observed_values_checked"] is False for row in rows)


def test_constructed_96_point_geometry_has_strip_slenderness_above_one(monkeypatch):
    from vcg_biomarker.core.biomarkers import discovery_v3_biomarkers as geometry
    x = np.repeat(np.linspace(-.98, .98, 12), 2)
    y = np.tile([-1.2, 1.2], 12)
    points = np.column_stack([np.r_[x, np.tile([-1., 1.], 36)],
                              np.r_[y, np.zeros(72)], np.zeros(96)])
    monkeypatch.setattr(geometry, "_resampled", lambda *args: points)
    result = geometry._plane_metrics(SimpleNamespace(_primitive_cache={}), "qrs", "frontal_xy")
    assert result["median_strip_slenderness"] == pytest.approx(1.2)
