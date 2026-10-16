"""Exact centered running median with SciPy's half-sample reflection.

The installed pandas native rolling-median kernel maintains window order
statistics instead of selecting from scratch for every two-second window.
No JIT, approximate quantile or new dependency is used.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def centered_median(values, window, *, accelerated=True):
    signal = np.asarray(values)
    if not isinstance(window, (int, np.integer)) or window < 1 or window % 2 != 1:
        raise ValueError("Median window must be positive and odd")
    if signal.ndim != 2:
        raise ValueError("Expected samples by leads")
    if (signal.size and window > len(signal) and np.isfinite(signal).all()
            and (not accelerated or signal.dtype != np.float64)):
        # Avoid the installed SciPy kernel's unstable edge case when its
        # footprint exceeds an extremely short record. Preserve integer order
        # statistics exactly (including int64 values not representable in float).
        half = window // 2
        padded = np.pad(signal, ((half, half), (0, 0)), mode="symmetric")
        result = np.empty_like(signal)
        for index in range(len(signal)):
            result[index] = np.partition(padded[index:index + window], half, axis=0)[half]
        return result
    if (not accelerated or signal.dtype != np.float64
            or not np.isfinite(signal).all() or not signal.size):
        return median_filter(signal, size=(window, 1), mode="reflect")
    half = window // 2
    padded = np.pad(signal, ((half, half), (0, 0)), mode="symmetric")
    # Odd windows return an actual order statistic, not an interpolated median.
    # The trailing rolling windows are aligned after half-sample reflection.
    result = pd.DataFrame(padded).rolling(window, min_periods=window).median()
    return result.iloc[window - 1:].to_numpy(copy=True)
