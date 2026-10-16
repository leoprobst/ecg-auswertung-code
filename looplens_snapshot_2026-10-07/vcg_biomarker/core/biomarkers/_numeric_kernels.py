"""Vectorized kernels for the record-level geometric hot paths.

Marker formulas remain inspectable Python. Segment-intersection counting and
dynamic time warping are different: they repeat the same tiny numeric
operation tens of thousands of times per ECG and dominated the full catalog
profile. NumPy/SciPy move their arithmetic into stable native wheels. The
executable scalar implementations remain the equivalence oracle.

Numba 0.66/llvmlite 0.48 was explicitly benchmarked here and was faster, but
reproducibly emitted Windows access violations during import/JIT shutdown.
An analysis preflight cannot accept that failure mode, even when the process
occasionally returns exit code zero, so these kernels intentionally avoid JIT
state and remain safe in spawned feature workers.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _self_intersection_loop(values: np.ndarray) -> int:
    count = 0
    segment_count = len(values) - 1
    for first in range(segment_count):
        ax = values[first, 0]
        ay = values[first, 1]
        bx = values[first + 1, 0]
        by = values[first + 1, 1]
        for second in range(first + 2, segment_count):
            if first == 0 and second == segment_count - 1:
                continue
            cx = values[second, 0]
            cy = values[second, 1]
            dx = values[second + 1, 0]
            dy = values[second + 1, 1]
            first_c = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            first_d = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
            second_a = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
            second_b = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
            if first_c * first_d < 0.0 and second_a * second_b < 0.0:
                count += 1
    return count


def self_intersection_count_reference(points: np.ndarray) -> int:
    """Count strict, non-adjacent 2-D segment intersections in Python."""

    values = np.ascontiguousarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 4:
        return 0
    return _self_intersection_loop(values)


def _dtw_loop(first: np.ndarray, second: np.ndarray) -> float:
    n = len(first)
    m = len(second)
    if n == 0 or m == 0:
        return float("inf")
    previous = np.full(m + 1, np.inf, dtype=np.float64)
    current = np.full(m + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for row in range(1, n + 1):
        current[0] = np.inf
        for column in range(1, m + 1):
            squared = 0.0
            for coordinate in range(first.shape[1]):
                delta = first[row - 1, coordinate] - second[
                    column - 1,
                    coordinate,
                ]
                squared += delta * delta
            predecessor = min(
                current[column - 1],
                previous[column],
                previous[column - 1],
            )
            current[column] = float(np.sqrt(squared)) + predecessor
        previous, current = current, previous
    return float(previous[m] / max(n + m, 1))


def dtw_distance_reference(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-row dynamic-time-warping reference implementation."""

    first = np.ascontiguousarray(a, dtype=np.float64)
    second = np.ascontiguousarray(b, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("DTW inputs must be two-dimensional")
    if first.shape[1] != second.shape[1]:
        raise ValueError("DTW inputs must have the same coordinate count")
    return _dtw_loop(first, second)


def self_intersection_count(points: np.ndarray) -> int:
    """Count intersections in bounded NumPy broadcast blocks."""

    values = np.ascontiguousarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 4:
        return 0
    starts = values[:-1]
    ends = values[1:]
    segment_count = len(starts)
    total = 0
    # A 256 x N block caps temporary memory for long global traces while
    # retaining large contiguous vector operations for ordinary wave loops.
    for block_start in range(0, segment_count, 256):
        block_stop = min(segment_count, block_start + 256)
        first_indices = np.arange(block_start, block_stop)[:, None]
        second_indices = np.arange(segment_count)[None, :]
        candidate = second_indices >= first_indices + 2
        if block_start == 0:
            candidate[0, segment_count - 1] = False
        if not np.any(candidate):
            continue

        a = starts[block_start:block_stop, None, :]
        b = ends[block_start:block_stop, None, :]
        c = starts[None, :, :]
        d = ends[None, :, :]
        ab = b - a
        cd = d - c
        first_c = ab[..., 0] * (c[..., 1] - a[..., 1]) - ab[
            ..., 1
        ] * (c[..., 0] - a[..., 0])
        first_d = ab[..., 0] * (d[..., 1] - a[..., 1]) - ab[
            ..., 1
        ] * (d[..., 0] - a[..., 0])
        second_a = cd[..., 0] * (a[..., 1] - c[..., 1]) - cd[
            ..., 1
        ] * (a[..., 0] - c[..., 0])
        second_b = cd[..., 0] * (b[..., 1] - c[..., 1]) - cd[
            ..., 1
        ] * (b[..., 0] - c[..., 0])
        total += int(
            np.count_nonzero(
                candidate
                & (first_c * first_d < 0.0)
                & (second_a * second_b < 0.0)
            )
        )
    return total


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute exact DTW by vectorizing independent anti-diagonals."""

    first = np.ascontiguousarray(a, dtype=np.float64)
    second = np.ascontiguousarray(b, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("DTW inputs must be two-dimensional")
    if first.shape[1] != second.shape[1]:
        raise ValueError("DTW inputs must have the same coordinate count")
    n = len(first)
    m = len(second)
    if n == 0 or m == 0:
        return float("inf")
    costs = cdist(first, second, metric="euclidean")
    accumulated = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    accumulated[0, 0] = 0.0
    # Up, left and diagonal predecessors belong to earlier anti-diagonals;
    # every cell on the current anti-diagonal can therefore be updated at
    # once without changing the dynamic-programming recurrence.
    for diagonal in range(2, n + m + 1):
        row_start = max(1, diagonal - m)
        row_stop = min(n, diagonal - 1)
        rows = np.arange(row_start, row_stop + 1)
        columns = diagonal - rows
        predecessor = np.minimum(
            np.minimum(
                accumulated[rows - 1, columns],
                accumulated[rows, columns - 1],
            ),
            accumulated[rows - 1, columns - 1],
        )
        accumulated[rows, columns] = (
            costs[rows - 1, columns - 1] + predecessor
        )
    return float(accumulated[n, m] / max(n + m, 1))


__all__ = [
    "dtw_distance",
    "dtw_distance_reference",
    "self_intersection_count",
    "self_intersection_count_reference",
]
