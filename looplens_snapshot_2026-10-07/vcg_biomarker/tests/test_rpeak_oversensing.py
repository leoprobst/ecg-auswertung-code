"""QRS detection and rate on rhythms where P or T waves compete with the QRS."""
from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.core.delineation.base import ventricular_rr_seconds
from vcg_biomarker.core.preprocessing import detect_r_peaks_multilead

FS = 500
GAINS = np.asarray([0.75, 1.00, 0.65, -0.70, 0.55, 0.80, -0.45, 0.65, 0.85, 1.05, 1.10, 0.95])


def _wave(signal, center, offsets, amplitudes, gains):
    samples = center + np.rint(np.asarray(offsets) * FS).astype(int)
    start, end = max(0, samples[0]), min(len(signal), samples[-1] + 1)
    if end <= start:
        return
    support = np.arange(start, end)
    signal[start:end] += np.interp(support, samples, amplitudes)[:, None] * gains[None, :]


def _qrs(signal, r, *, wide=False, height=1.0):
    offsets = (-0.105, -0.070, 0.0, 0.075, 0.135) if wide else (-0.050, -0.027, 0.0, 0.028, 0.055)
    _wave(signal, r, offsets, (0.0, -0.18 * height, height, -0.30 * height, 0.0), GAINS)
    _wave(signal, r, (0.135, 0.285, 0.485), (0.0, 0.32, 0.0), GAINS)


def _noise(ecg, seed=3):
    ecg += np.random.default_rng(seed).normal(0.0, 0.004, size=ecg.shape)
    return ecg


def _matched(detected, expected, tolerance=int(0.05 * FS)):
    return sum(bool(np.any(np.abs(detected - r) <= tolerance)) for r in expected)


def test_complete_av_block_does_not_count_p_waves_as_beats() -> None:
    # Junctional escape at 36/min, sinus P waves (0.2 mV, 110 ms) at 90/min
    # dissociated from it. Before the fix the P waves were accepted as QRS
    # complexes and the rate came out near the atrial rate.
    ecg = np.zeros((10 * FS, 12))
    qrs = np.arange(int(0.9 * FS), 10 * FS - int(0.6 * FS), int(1.65 * FS))
    for r in qrs:
        _qrs(ecg, r)
    for p in np.arange(int(0.4 * FS), 10 * FS - int(0.4 * FS), int(0.66 * FS)):
        if np.min(np.abs(qrs - p)) > int(0.2 * FS):
            _wave(ecg, p, (-0.055, -0.02, 0.02, 0.055), (0.0, 0.2, 0.2, 0.0), np.abs(GAINS))
    detected = detect_r_peaks_multilead(_noise(ecg), FS)
    inner = qrs[(qrs >= int(0.25 * FS)) & (qrs < 10 * FS - int(0.25 * FS))]
    assert _matched(detected, inner) == len(inner)
    assert len(detected) == len(inner)


def test_bigeminy_keeps_every_wide_ectopic_beat() -> None:
    # Wide premature beats have less slope energy than the narrow beats and
    # arrive 480 ms after them, inside the preceding T wave's reach; they
    # must not be taken for T waves.
    ecg = np.zeros((10 * FS, 12))
    expected = []
    r = int(0.8 * FS)
    while r < 10 * FS - int(0.8 * FS):
        _qrs(ecg, r)
        expected.append(r)
        ectopic = r + int(0.48 * FS)
        _wave(ecg, ectopic, (-0.08, -0.04, 0.0, 0.05, 0.10), (0.0, 0.4, -1.2, 0.3, 0.0), GAINS)
        _wave(ecg, ectopic, (0.12, 0.27, 0.44), (0.0, 0.45, 0.0), GAINS)
        expected.append(ectopic)
        r += int(1.30 * FS)
    detected = detect_r_peaks_multilead(_noise(ecg), FS)
    expected = np.asarray(expected)
    inner = expected[(expected >= int(0.25 * FS)) & (expected < 10 * FS - int(0.25 * FS))]
    assert _matched(detected, inner, tolerance=int(0.08 * FS)) == len(inner)


def test_rate_is_the_mean_rr_not_the_coupling_interval() -> None:
    # Bigeminy: coupling 420 ms, compensatory pause 880 ms, ten beats. The
    # median of the nine intervals is the coupling interval (143/min); the
    # rate is ten beats over their span.
    beats = [0]
    for index in range(9):
        beats.append(beats[-1] + (210 if index % 2 == 0 else 440))
    rr = ventricular_rr_seconds(beats, FS)
    assert 60.0 / rr == pytest.approx(60.0 * 9 / (beats[-1] / FS))
    assert 60.0 / (np.median(np.diff(beats)) / FS) > 140.0
    assert 60.0 / rr < 100.0
    assert ventricular_rr_seconds([100], FS) is None
    assert ventricular_rr_seconds([100, 100], FS) is None
