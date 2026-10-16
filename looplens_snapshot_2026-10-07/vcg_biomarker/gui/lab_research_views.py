"""Bounded descriptive displays. No production features or diagnostic thresholds."""
from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram


def beat_amplitude_map(signal, fs, r_seconds, *, before=.25, after=.5, max_beats=200):
    x = np.asarray(signal)
    peaks = np.asarray(r_seconds, dtype=float)
    if x.ndim != 1 or not np.isfinite(fs) or fs <= 0 or before <= 0 or after <= 0:
        raise ValueError('Invalid native signal/timebase')
    if peaks.ndim != 1 or not np.all(np.isfinite(peaks)) or np.any(np.diff(peaks) <= 0):
        raise ValueError('R times must be finite, unique and ordered')
    if not 1 <= max_beats <= 200 or fs > 10000:
        raise ValueError('Display budget exceeded')
    offsets = np.arange(-round(before*fs), round(after*fs)+1, dtype=int)
    centers = np.rint(peaks*fs).astype(int)
    valid = (centers+offsets[0] >= 0) & (centers+offsets[-1] < len(x))
    ids = np.flatnonzero(valid)
    eligible = len(ids)
    if len(ids) > max_beats:
        ids = ids[np.linspace(0, len(ids)-1, max_beats, dtype=int)]
    values = x[centers[ids, None]+offsets] if len(ids) else np.empty((0, len(offsets)))
    if np.any(np.isinf(values)):
        raise ValueError('Infinite signal amplitudes')
    return {'values_mv': values, 'relative_seconds': offsets/fs, 'beat_indices': ids,
            'r_seconds': peaks[ids], 'eligible_beats': eligible,
            'edge_excluded': int((~valid).sum()), 'sampled_for_display': eligible > len(ids)}


def native_spectrogram(signal, fs):
    """First <=60 seconds, Hann 256-ms windows, 75% overlap; no resampling."""
    if not np.isfinite(fs) or fs <= 0 or fs > 10000:
        raise ValueError('Unsupported native sampling rate')
    x = np.asarray(signal)
    if x.ndim != 1:
        raise ValueError('Expected one lead')
    x = np.asarray(x[:int(60*fs)], dtype=float)
    if len(x) < 32 or not np.all(np.isfinite(x)):
        raise ValueError('Signal too short or contains gaps; no interpolation performed')
    window = min(len(x), max(16, round(.256*fs)))
    frequencies, times, power = spectrogram(x, fs=fs, window='hann', nperseg=window,
                                            noverlap=window*3//4, detrend='constant', scaling='density')
    return {'frequencies_hz': frequencies, 'seconds': times, 'power_mv2_hz': power,
            'window_samples': window, 'duration_seconds': len(x)/fs}
