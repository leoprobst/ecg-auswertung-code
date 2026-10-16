"""Conservative timing measurements for simple QRS morphology, lab only."""
import numpy as np


def simple_qrs_timing(segment, fs):
    """Baseline-referenced inclusive QRS window; reject ambiguous components.

    Single positive component with a unique R maximum; S is the following
    contiguous negative component with observed baseline crossings. No
    invented boundary at the window edge and no R/R-prime morphology diagnosis.
    """
    result = {'r_peak_time': np.nan, 's_duration': np.nan}
    x = np.asarray(segment, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(fs) or fs <= 0:
        return result
    positive = x > 0
    starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    if len(starts) != 1 or x.max() < .05:
        return result
    maxima = np.flatnonzero(x == x.max())
    if len(maxima) != 1:
        return result
    r = int(maxima[0])
    result['r_peak_time'] = r * 1000 / fs
    after = np.flatnonzero(x[r+1:] < 0)
    if not len(after):
        return result
    start = r+1+int(after[0])
    ends = np.flatnonzero(x[start:] >= 0)
    if not len(ends):
        return result
    end = start+int(ends[0])
    if -x[start:end].min() < .05 or np.any(x[end:] < 0):
        return result
    # Linear crossing interpolation; finer positioning is not higher accuracy.
    left = start-1 + x[start-1]/(x[start-1]-x[start])
    right = end-1 + (-x[end-1])/(x[end]-x[end-1])
    result['s_duration'] = (right-left)*1000/fs
    return result
