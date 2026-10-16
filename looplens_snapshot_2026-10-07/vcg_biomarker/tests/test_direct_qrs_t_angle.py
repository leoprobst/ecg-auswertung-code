from __future__ import annotations

import numpy as np

from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import (
    direct_qrs_t_angles_8lead,
)


QRS = (20, 39)
T_WAVE = (60, 89)
BASELINE = (5, 14)


def _signal(qrs_vector: np.ndarray, t_vector: np.ndarray) -> np.ndarray:
    signal = np.zeros((100, 12), dtype=float)
    independent = [0, 1, 6, 7, 8, 9, 10, 11]
    signal[QRS[0] : QRS[1] + 1, independent] = qrs_vector
    signal[T_WAVE[0] : T_WAVE[1] + 1, independent] = t_vector
    return signal


def _angles(signal: np.ndarray):
    return direct_qrs_t_angles_8lead(
        signal,
        qrs_bounds=QRS,
        t_bounds=T_WAVE,
        baseline_bounds=BASELINE,
    )


def test_direct_angles_recover_parallel_orthogonal_and_opposite_vectors():
    qrs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for t_vector, expected in (
        (qrs, 0.0),
        (np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 90.0),
        (-qrs, 180.0),
    ):
        area, integral = _angles(_signal(qrs, t_vector))
        assert area == pytest.approx(expected, abs=1e-10)
        assert integral == pytest.approx(expected, abs=1e-10)


def test_direct_angles_are_invariant_to_offset_and_positive_scale():
    qrs = np.array([1.0, 0.2, -0.1, 0.4, 0.3, -0.2, 0.1, 0.5])
    t_wave = np.array([0.3, 0.8, 0.2, -0.1, 0.5, 0.1, -0.2, 0.4])
    signal = _signal(qrs, t_wave)
    expected = _angles(signal)
    offsets = np.linspace(-2.0, 2.0, signal.shape[1])
    observed = _angles(signal * 3.5 + offsets)
    assert observed == pytest.approx(expected, abs=1e-10)


def test_integral_method_retains_morphology_lost_by_area_method():
    signal = _signal(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    # Add cancelling excursions in a second independent lead. They cancel in
    # the signed T area vector but increase the Integral denominator.
    signal[T_WAVE[0] : 75, 1] = 2.0
    signal[75 : T_WAVE[1] + 1, 1] = -2.0
    area, integral = _angles(signal)
    assert area is not None and area < 2.0
    assert integral is not None and integral > area + 30.0


def test_direct_angles_abstain_on_missing_lead_or_unmeasured_baseline():
    signal = _signal(np.ones(8), np.ones(8))
    assert direct_qrs_t_angles_8lead(
        signal[:, :11],
        qrs_bounds=QRS,
        t_bounds=T_WAVE,
        baseline_bounds=BASELINE,
    ) == (None, None)
    signal[BASELINE[0], 0] = np.nan
    assert _angles(signal) == (None, None)


# Local import keeps the production module import independent of pytest.
import pytest

