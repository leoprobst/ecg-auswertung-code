from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from vcg_biomarker.core.biomarkers._numeric_kernels import (
    dtw_distance,
    dtw_distance_reference,
    self_intersection_count,
    self_intersection_count_reference,
)
from vcg_biomarker.core.biomarkers.catalog_biomarkers import (
    _convex_hull_2d_metrics,
)


def _legacy_self_intersections(points: np.ndarray) -> int:
    def orientation(a, b, c):
        return np.sign(
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )

    count = 0
    for first in range(len(points) - 1):
        for second in range(first + 2, len(points) - 1):
            if first == 0 and second == len(points) - 2:
                continue
            a, b = points[first], points[first + 1]
            c, d = points[second], points[second + 1]
            if (
                orientation(a, b, c) * orientation(a, b, d) < 0
                and orientation(c, d, a) * orientation(c, d, b) < 0
            ):
                count += 1
    return count


@pytest.mark.parametrize("point_count", [4, 5, 17, 64, 129])
def test_native_intersection_kernel_is_exactly_legacy_equivalent(
    point_count: int,
) -> None:
    rng = np.random.default_rng(20260902 + point_count)
    for _ in range(8):
        points = rng.normal(size=(point_count, 2))
        expected = _legacy_self_intersections(points)
        assert self_intersection_count_reference(points) == expected
        assert self_intersection_count(points) == expected


def test_native_dtw_is_exactly_reference_equivalent() -> None:
    rng = np.random.default_rng(20260902)
    for first_count, second_count, dimensions in (
        (3, 7, 1),
        (16, 11, 2),
        (64, 64, 3),
        (101, 37, 8),
    ):
        first = rng.normal(size=(first_count, dimensions))
        second = rng.normal(size=(second_count, dimensions))
        assert dtw_distance(first, second) == pytest.approx(
            dtw_distance_reference(first, second),
            rel=0.0,
            abs=1e-14,
        )


def test_two_dimensional_hull_matches_qhull_without_tempfile_dependency() -> None:
    rng = np.random.default_rng(20260902)
    for count in (8, 31, 128):
        points = rng.normal(size=(count, 2))
        points = np.vstack((points, points[:3]))
        expected = ConvexHull(points)
        measured = _convex_hull_2d_metrics(points)
        assert measured is not None
        area, perimeter = measured
        assert area == pytest.approx(float(expected.volume), rel=1e-12)
        assert perimeter == pytest.approx(float(expected.area), rel=1e-12)

    assert _convex_hull_2d_metrics(
        np.column_stack((np.arange(8.0), np.arange(8.0)))
    ) is None


def test_profiled_numeric_kernels_retain_material_vectorized_speedup() -> None:
    rng = np.random.default_rng(42)
    points = rng.normal(size=(192, 2))
    first = rng.normal(size=(96, 3))
    second = rng.normal(size=(96, 3))

    # Warm native wheel dispatch and memory allocation before timing.
    self_intersection_count(points)
    dtw_distance(first, second)

    started = time.perf_counter()
    self_intersection_count_reference(points)
    dtw_distance_reference(first, second)
    reference_seconds = time.perf_counter() - started

    repetitions = 20
    started = time.perf_counter()
    for _ in range(repetitions):
        self_intersection_count(points)
        dtw_distance(first, second)
    native_seconds_per_call = (
        time.perf_counter() - started
    ) / repetitions

    # The local profile is substantially faster. A two-fold floor is loose
    # enough for loaded CI hosts while still detecting a silent regression to
    # the original nested Python loops.
    assert native_seconds_per_call * 2.0 < reference_seconds
