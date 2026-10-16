"""At fast rates the T end must not run into the next beat's P wave."""
from __future__ import annotations

import numpy as np

from vcg_biomarker.core.delineation.multilead_delineator import MultileadDelineator

FS = 500
RR_S = 0.52
T_END_S = 0.33
GAINS = np.asarray([0.75, 1.00, 0.65, -0.70, 0.55, 0.80, -0.45, 0.65, 0.85, 1.05, 1.10, 0.95])


def _wave(signal, center, offsets, amplitudes, gains):
    samples = center + np.rint(np.asarray(offsets) * FS).astype(int)
    start, end = max(0, samples[0]), min(len(signal), samples[-1] + 1)
    if end > start:
        support = np.arange(start, end)
        signal[start:end] += np.interp(support, samples, amplitudes)[:, None] * gains[None, :]


def _tachycardia():
    # 115/min: the T wave ends 330 ms after R and the next P wave (0.2 mV)
    # starts right there -- T and P touch without an isoelectric gap, as at
    # fast rates in the MIMIC and QTDB cases where the T end ran late.
    n = int(10 * FS)
    ecg = np.zeros((n, 12))
    peaks = np.arange(int(0.6 * FS), n - int(0.6 * FS), int(RR_S * FS))
    for r in peaks:
        _wave(ecg, r, (-0.045, -0.02, 0.0, 0.025, 0.045), (0.0, -0.15, 1.0, -0.25, 0.0), GAINS)
        _wave(ecg, r, (0.10, 0.22, T_END_S), (0.0, 0.30, 0.0), GAINS)
        _wave(ecg, r, (T_END_S - RR_S, T_END_S - RR_S + 0.045, T_END_S - RR_S + 0.09),
              (0.0, 0.2, 0.0), np.abs(GAINS))
    ecg += np.random.default_rng(5).normal(0.0, 0.004, ecg.shape)
    return ecg, peaks


def _t_end_errors_ms(margin):
    previous = MultileadDelineator.t_window_next_p_margin_s
    MultileadDelineator.t_window_next_p_margin_s = margin
    try:
        ecg, peaks = _tachycardia()
        beats = MultileadDelineator().delineate(ecg, peaks, FS)
    finally:
        MultileadDelineator.t_window_next_p_margin_s = previous
    return np.asarray([
        (beat.t_offset.sample - beat.r_peak.sample) * 1000 / FS - T_END_S * 1000
        for beat in beats[1:-1] if beat.t_offset is not None
    ])


def test_without_the_guard_the_t_end_runs_into_the_next_p_wave() -> None:
    errors = _t_end_errors_ms(None)
    assert errors.size >= 10
    assert np.median(errors) > 50.0


def test_the_guard_keeps_the_t_end_off_the_next_p_wave() -> None:
    errors = _t_end_errors_ms(MultileadDelineator.t_window_next_p_margin_s)
    assert errors.size >= 10
    # Where T and P touch, the T end is cut a little short at worst; it
    # never lands on the P wave (which starts at error 0).
    assert np.all(errors <= 10.0)
    assert np.median(errors) >= -40.0
