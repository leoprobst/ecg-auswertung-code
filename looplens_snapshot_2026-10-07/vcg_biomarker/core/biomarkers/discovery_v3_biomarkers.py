"""Large prospective reserve of visually and mechanistically motivated VCG features.

Discovery-v3 is deliberately isolated from Core/Sentinel.  Its constructs are
screening hypotheses, not diagnostic tests.  Expensive primitives are cached
once on the record-scoped :class:`FeatureContext`; every public marker is a
scalar and abstains when its geometric or fiducial prerequisites are absent.

Evidence grades are conservative: B = a related construct has human cohort
evidence; C = normative/small-cohort or transfer evidence; D = a mathematical
or pathophysiological hypothesis without independent clinical validation.
There are no clinical thresholds in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import ndimage
from scipy import signal as sp_signal
from scipy.spatial import ConvexHull, distance
from scipy.stats import wasserstein_distance

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from .catalog_biomarkers import EPS, FeatureContext, FeatureValue, _context, _resample_curve
from ..marker_bounds import bounded_ratio
from ..data_loader import LEAD_MAP
from ..delineation.base import (
    RecordDelineation,
    measurement_confidence_threshold,
)
from ..vcg_geometry import (
    VCG_AXIS_CONVENTION,
    VCG_AXIS_CONVENTION_VERSION,
    vcg_azimuth_deg,
    vcg_elevation_deg,
    vector_angle_deg,
)


DISCOVERY_V3_VERSION = "prospective-discovery-v3.0.0"
DISCOVERY_V3_SOURCE = "Prospective VCG morphology and mechanism audit, 2026-08"

REF_NORMAL_VCG = "https://doi.org/10.1111/j.1475-097x.1994.tb00420.x"
REF_LOOP_MORPHOLOGY = "https://doi.org/10.1016/j.jelectrocard.2015.12.014"
REF_LOOP_FOLDING = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5606210/"
REF_SYNTHESIS = "https://doi.org/10.1016/0022-0736(88)90113-6"
REF_RV_OVERLOAD = "https://doi.org/10.1016/S0022-0736(83)80159-9"
REF_RVH_COPD = "https://doi.org/10.1253/jcj.40.1301"
REF_ARVC = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5640940/"
REF_HCM = "https://doi.org/10.4044/joma1947.105.7-8_681"
REF_HCM_REVIEW = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6932007/"
REF_T_LOOP = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12147597/"
REF_T_LOOP_OUTCOME = "https://doi.org/10.1016/j.jelectrocard.2010.05.004"
REF_ST_OFFSETS = "https://doi.org/10.1016/j.jelectrocard.2008.08.039"
REF_DYNAMIC_VCG = "https://doi.org/10.1371/journal.pone.0057175"
REF_FRECHET = "https://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf"
REF_DTW = "https://doi.org/10.1109/TASSP.1978.1163055"
REF_ENERGY_DISTANCE = "https://doi.org/10.1214/009053607000000505"
REF_SLICED_WASSERSTEIN = "https://doi.org/10.1007/978-3-540-88688-4_26"
REF_PERSISTENCE = "https://doi.org/10.1007/s00454-002-2885-2"
REF_BOX_DIMENSION = "https://doi.org/10.1016/0167-2789(90)90081-L"

PLANES: dict[str, tuple[int, int]] = {
    "frontal_xy": (0, 1),
    "horizontal_xz": (0, 2),
    "sagittal_yz": (1, 2),
}


@dataclass(frozen=True)
class DiscoveryV3Spec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    dependencies: tuple[str, ...]
    evidence_level: str
    evidence_note: str
    references: tuple[str, ...]
    calculator: Callable[[FeatureContext], FeatureValue]
    visual_assessment: str = "mathematical_only"

    @property
    def visually_interpretable(self) -> bool:
        """True only when a clinician can judge the construct directly."""

        return self.visual_assessment == "direct"


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_discovery_v3_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_discovery_v3_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _finite_xyz(value: object, minimum: int = 3) -> Optional[np.ndarray]:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    return points if len(points) >= minimum else None


def _segment(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    """Return the centrally origin-corrected representative segment."""

    def build() -> Optional[np.ndarray]:
        return _finite_xyz(ctx.vcg_segment_origin_corrected(segment))

    return _memo(ctx, ("segment", segment), build)


def _resampled(ctx: FeatureContext, segment: str, n: int = 96) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        points = _segment(ctx, segment)
        if points is None:
            return None
        sampled = np.asarray(_resample_curve(points, n), dtype=float)
        return _finite_xyz(sampled, minimum=max(3, n // 2))

    return _memo(ctx, ("resampled", segment, n), build)


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """An amplitude ratio, abstaining below EPS and capped above it.

    Abstaining at ``EPS`` = 1e-12 bounds the divisor, not the quotient: the
    four ``rvh_*_force_ratio`` markers reached 5.618 to 39.469 in the current
    store on denominators just above the guard.
    """

    if abs(denominator) <= EPS:
        return None
    return bounded_ratio(numerator, denominator)


def _value_from_dict(
    primitive: Callable[[FeatureContext], Optional[dict[str, float]]],
    key: str,
    reason: str,
) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        values = primitive(ctx)
        if values is None or key not in values or not np.isfinite(values[key]):
            return _abstain(reason)
        return FeatureValue(float(values[key]))

    return calculate


def _oriented_normal(points: np.ndarray) -> Optional[np.ndarray]:
    centered = points - np.mean(points, axis=0)
    if len(centered) < 3 or float(np.linalg.norm(centered)) <= EPS:
        return None
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] <= EPS:
        return None
    normal = np.asarray(vh[-1], dtype=float)
    # Sign is deterministic in the fixed Frank frame.  Plane-to-plane twist is
    # nevertheless evaluated as an unsigned plane angle and is sign invariant.
    pivot = int(np.argmax(np.abs(normal)))
    if normal[pivot] < 0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def _unsigned_plane_angle(first: np.ndarray, second: np.ndarray) -> float:
    angle = vector_angle_deg(first, second)
    if angle is None:
        return float("nan")
    return float(min(angle, 180.0 - angle))


def _polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _smooth(points: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if len(points) < 7:
        return points.copy()
    return ndimage.gaussian_filter1d(points, sigma=sigma, axis=0, mode="nearest")


# --- visually readable plane and 3-D morphology ---------------------------


def _distance_to_polygon_edges(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    distances: list[np.ndarray] = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        length2 = float(np.dot(edge, edge))
        if length2 <= EPS:
            continue
        projection = np.clip(((points - start) @ edge) / length2, 0.0, 1.0)
        nearest = start + projection[:, None] * edge
        distances.append(np.linalg.norm(points - nearest, axis=1))
    if not distances:
        return np.full(len(points), np.nan)
    return np.min(np.vstack(distances), axis=0)


def _plane_metrics(
    ctx: FeatureContext,
    segment: str,
    plane: str,
) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points3 = _resampled(ctx, segment, 96)
        if points3 is None:
            return None
        indices = PLANES[plane]
        points = np.asarray(points3[:, indices], dtype=float)
        diameter = float(np.linalg.norm(np.ptp(points, axis=0)))
        if diameter <= EPS:
            return None

        # Polar inscription is referenced to the physiological VCG origin.
        radii = np.linalg.norm(points, axis=1)
        valid = radii > max(EPS, 0.05 * float(np.max(radii)))
        angle = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
        increments = np.diff(ndimage.gaussian_filter1d(angle, 1.0, mode="nearest"))
        step_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
        valid_step = valid[:-1] & valid[1:] & (np.abs(increments) > np.radians(0.5))
        if np.count_nonzero(valid_step) >= 2:
            signed_rotation = float(np.sum(increments[valid_step]) / (2.0 * np.pi))
            weight = step_length[valid_step]
            total_weight = float(np.sum(weight))
            clockwise = (
                float(np.sum(weight[increments[valid_step] < 0]) / total_weight)
                if total_weight > EPS
                else float("nan")
            )
            signs = np.sign(increments[valid_step])
            reversal_count = float(np.count_nonzero(signs[1:] != signs[:-1]))
            reversal_fraction = float(reversal_count / max(1, len(signs) - 1))
        else:
            signed_rotation = clockwise = reversal_count = reversal_fraction = float("nan")

        # Adaptive occupancy is scale/translation invariant, but tied to the
        # named anatomical projection.  It complements the legacy raw 10x10
        # frontal T count with normalized multi-plane density and entropy.
        minimum = np.min(points, axis=0)
        span = np.ptp(points, axis=0)
        normalized = (points - minimum) / np.maximum(span, EPS)
        cells = np.minimum((normalized * 12).astype(int), 11)
        flat = cells[:, 0] * 12 + cells[:, 1]
        counts = np.bincount(flat, minlength=144).astype(float)
        occupied = counts[counts > 0]
        probability = occupied / np.sum(occupied)
        occupancy = float(len(occupied) / 144.0)
        entropy = float(-np.sum(probability * np.log(probability)) / np.log(144.0))

        try:
            hull = ConvexHull(points)
            boundary = points[hull.vertices]
            inward = _distance_to_polygon_edges(points, boundary)
            bite = float(np.nanpercentile(inward, 95) / diameter)
        except Exception:
            bite = float("nan")

        # A cusp must survive two smoothing scales.  Prominence is expressed
        # in turning-angle degrees, so it is amplitude-scale invariant.
        cusp_counts: list[int] = []
        for sigma in (0.8, 1.6):
            curve = _smooth(points, sigma)
            tangent = np.diff(curve, axis=0)
            norm = np.linalg.norm(tangent, axis=1)
            usable = norm > EPS
            unit = tangent[usable] / norm[usable, None]
            if len(unit) < 5:
                cusp_counts.append(0)
                continue
            turning = np.degrees(
                np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0))
            )
            threshold = max(5.0, 0.5 * float(np.percentile(turning, 75)))
            peaks, _ = sp_signal.find_peaks(turning, prominence=threshold, distance=4)
            cusp_counts.append(int(len(peaks)))
        cusp_count = float(min(cusp_counts))

        centered = points - np.mean(points, axis=0)
        local_area = (
            centered[:-1, 0] * centered[1:, 1]
            - centered[:-1, 1] * centered[1:, 0]
        )
        cutoff = max(EPS, 0.02 * float(np.percentile(np.abs(local_area), 90)))
        area_signs = np.sign(local_area[np.abs(local_area) > cutoff])
        lobe_count = (
            float(1 + np.count_nonzero(area_signs[1:] != area_signs[:-1]))
            if len(area_signs)
            else float("nan")
        )

        # This is not the existing PCA eigenvalue/roundness ratio.  It asks
        # how wide the median occupied strip is along the loop's major axis.
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        principal = centered @ vh.T
        major_span = float(np.ptp(principal[:, 0]))
        strip_widths: list[float] = []
        if major_span > EPS:
            bins = np.linspace(np.min(principal[:, 0]), np.max(principal[:, 0]), 13)
            labels = np.clip(np.digitize(principal[:, 0], bins) - 1, 0, 11)
            for bin_index in range(12):
                selected = principal[labels == bin_index, 1]
                if len(selected) >= 2:
                    strip_widths.append(float(np.ptp(selected)))
        slenderness = (
            float(np.median(strip_widths) / major_span)
            if strip_widths and major_span > EPS
            else float("nan")
        )
        return {
            "signed_rotation_turns": signed_rotation,
            "clockwise_path_fraction": clockwise,
            "rotation_reversal_count": reversal_count,
            "rotation_reversal_fraction": reversal_fraction,
            "grid_occupancy_fraction": occupancy,
            "grid_entropy_fraction": entropy,
            "hull_bite_depth_fraction": bite,
            "persistent_cusp_count": cusp_count,
            "orientation_lobe_count": lobe_count,
            "median_strip_slenderness": slenderness,
        }

    return _memo(ctx, ("plane_metrics", segment, plane), build)


def _plane_metric_calculator(segment: str, plane: str, metric: str):
    return _value_from_dict(
        lambda ctx: _plane_metrics(ctx, segment, plane),
        metric,
        f"{segment} {plane} projection is degenerate",
    )


def _temporal_geometry(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 96)
        if points is None:
            return None
        scale = float(np.sqrt(np.mean(np.sum((points - np.mean(points, axis=0)) ** 2, axis=1))))
        if scale <= EPS:
            return None
        design = np.column_stack(((points - np.mean(points, axis=0)) / scale, np.ones(len(points))))
        time = np.linspace(0.0, 1.0, len(points))
        fitted = design @ np.linalg.lstsq(design, time, rcond=None)[0]
        denominator = float(np.sum((time - np.mean(time)) ** 2))
        r2 = 1.0 - float(np.sum((time - fitted) ** 2)) / denominator

        thirds = np.array_split(points, 3)
        normals = [_oriented_normal(part) for part in thirds]
        if any(normal is None for normal in normals):
            pair_angles = [float("nan")] * 3
        else:
            pair_angles = [
                _unsigned_plane_angle(normals[0], normals[1]),
                _unsigned_plane_angle(normals[1], normals[2]),
                _unsigned_plane_angle(normals[0], normals[2]),
            ]

        window = 24
        sliding = [
            _oriented_normal(points[start : start + window])
            for start in range(0, len(points) - window + 1, 6)
        ]
        sliding = [normal for normal in sliding if normal is not None]
        twist = [
            _unsigned_plane_angle(first, second)
            for first, second in zip(sliding[:-1], sliding[1:])
        ]
        return {
            "temporal_color_gradient_r2": float(np.clip(r2, 0.0, 1.0)),
            "initial_central_plane_twist_deg": pair_angles[0],
            "central_terminal_plane_twist_deg": pair_angles[1],
            "initial_terminal_plane_twist_deg": pair_angles[2],
            "integrated_sliding_plane_twist_deg": float(np.sum(twist)) if twist else float("nan"),
            "maximum_sliding_plane_twist_deg": float(np.max(twist)) if twist else float("nan"),
        }

    return _memo(ctx, ("temporal_geometry", segment), build)


def _temporal_calculator(segment: str, metric: str):
    return _value_from_dict(
        lambda ctx: _temporal_geometry(ctx, segment),
        metric,
        f"{segment} temporal geometry is degenerate",
    )


# --- QRS/T curve distances and local activation-repolarization coupling ----


def _discrete_frechet(cost: np.ndarray) -> float:
    values = np.asarray(cost, dtype=float)
    if values.ndim != 2 or not values.size or not np.all(np.isfinite(values)):
        return float("nan")
    rows, columns = values.shape
    table = np.empty_like(values)
    table[0, 0] = values[0, 0]
    for row in range(1, rows):
        table[row, 0] = max(table[row - 1, 0], values[row, 0])
    for column in range(1, columns):
        table[0, column] = max(table[0, column - 1], values[0, column])
    for row in range(1, rows):
        for column in range(1, columns):
            table[row, column] = max(
                values[row, column],
                min(table[row - 1, column], table[row - 1, column - 1], table[row, column - 1]),
            )
    return float(table[-1, -1])


def _dtw_mean_cost(cost: np.ndarray) -> float:
    values = np.asarray(cost, dtype=float)
    rows, columns = values.shape
    accumulated = np.full((rows + 1, columns + 1), np.inf)
    steps = np.zeros((rows + 1, columns + 1), dtype=int)
    accumulated[0, 0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            predecessors = (
                accumulated[row - 1, column],
                accumulated[row, column - 1],
                accumulated[row - 1, column - 1],
            )
            chosen = int(np.argmin(predecessors))
            previous = ((row - 1, column), (row, column - 1), (row - 1, column - 1))[chosen]
            accumulated[row, column] = values[row - 1, column - 1] + predecessors[chosen]
            steps[row, column] = steps[previous] + 1
    return float(accumulated[-1, -1] / max(1, steps[-1, -1]))


def _unit_rows(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1)
    return points[norms > EPS] / norms[norms > EPS, None]


def _fibonacci_directions(count: int = 32) -> np.ndarray:
    index = np.arange(count, dtype=float)
    y = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.column_stack((radius * np.cos(angle), y, radius * np.sin(angle)))


def _qrs_t_distances(ctx: FeatureContext) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        qrs = _resampled(ctx, "qrs", 64)
        t_wave = _resampled(ctx, "t", 64)
        if qrs is None or t_wave is None:
            return None
        euclidean_cost = distance.cdist(qrs, t_wave)
        frechet = _discrete_frechet(euclidean_cost)

        q_centered = qrs - np.mean(qrs, axis=0)
        t_centered = t_wave - np.mean(t_wave, axis=0)
        q_scale = float(np.sqrt(np.mean(np.sum(q_centered * q_centered, axis=1))))
        t_scale = float(np.sqrt(np.mean(np.sum(t_centered * t_centered, axis=1))))
        if q_scale <= EPS or t_scale <= EPS:
            return None
        q_normalized = q_centered / q_scale
        t_normalized = t_centered / t_scale
        normalized_cost = distance.cdist(q_normalized, t_normalized)

        q_direction = _unit_rows(qrs)
        t_direction = _unit_rows(t_wave)
        if len(q_direction) < 3 or len(t_direction) < 3:
            return None
        angular_cost = np.degrees(
            np.arccos(np.clip(q_direction @ t_direction.T, -1.0, 1.0))
        )

        q_velocity = _unit_rows(np.diff(_smooth(q_normalized, 1.0), axis=0))
        t_velocity = _unit_rows(np.diff(_smooth(t_normalized, 1.0), axis=0))
        if len(q_velocity) < 3 or len(t_velocity) < 3:
            derivative_dtw = float("nan")
        else:
            derivative_cost = np.arccos(np.clip(q_velocity @ t_velocity.T, -1.0, 1.0)) / np.pi
            derivative_dtw = _dtw_mean_cost(derivative_cost)

        cross_mean = float(np.mean(euclidean_cost))
        energy_squared = max(
            0.0,
            2.0 * cross_mean
            - float(np.mean(distance.cdist(qrs, qrs)))
            - float(np.mean(distance.cdist(t_wave, t_wave))),
        )
        normalized_energy_squared = max(
            0.0,
            2.0 * float(np.mean(normalized_cost))
            - float(np.mean(distance.cdist(q_normalized, q_normalized)))
            - float(np.mean(distance.cdist(t_normalized, t_normalized))),
        )

        sliced = np.asarray(
            [
                wasserstein_distance(qrs @ direction, t_wave @ direction)
                for direction in _fibonacci_directions()
            ],
            dtype=float,
        )
        nearest_angles = np.concatenate((np.min(angular_cost, axis=1), np.min(angular_cost, axis=0)))

        q_phase = np.asarray(_resample_curve(q_direction, 64), dtype=float)
        t_phase = np.asarray(_resample_curve(t_direction, 64), dtype=float)
        q_phase /= np.maximum(np.linalg.norm(q_phase, axis=1, keepdims=True), EPS)
        t_phase /= np.maximum(np.linalg.norm(t_phase, axis=1, keepdims=True), EPS)
        phase_cosine = np.sum(q_phase * t_phase, axis=1)
        phase_distance = np.linalg.norm(qrs - t_wave, axis=1)
        third = len(phase_distance) // 3
        return {
            "discrete_frechet_mv": frechet,
            "normalized_discrete_frechet": _discrete_frechet(normalized_cost),
            "directional_discrete_frechet_deg": _discrete_frechet(angular_cost),
            "derivative_direction_dtw": derivative_dtw,
            # The square-root energy statistic has sqrt(mV), not mV, units.
            "energy_distance_sqrt_mv": float(np.sqrt(energy_squared)),
            "normalized_energy_distance": float(np.sqrt(normalized_energy_squared)),
            "sliced_wasserstein_mean_mv": float(np.mean(sliced)),
            "sliced_wasserstein_p90_mv": float(np.percentile(sliced, 90)),
            "nearest_direction_angle_median_deg": float(np.median(nearest_angles)),
            "nearest_direction_angle_p90_deg": float(np.percentile(nearest_angles, 90)),
            "local_alignment_dwell_fraction": float(np.mean(phase_cosine >= 0.5)),
            "local_opposition_dwell_fraction": float(np.mean(phase_cosine <= -0.5)),
            "early_phase_rms_point_distance_mv": float(np.sqrt(np.mean(phase_distance[:third] ** 2))),
            "late_phase_rms_point_distance_mv": float(np.sqrt(np.mean(phase_distance[-third:] ** 2))),
        }

    return _memo(ctx, ("qrs_t_distances",), build)


def _distance_calculator(metric: str):
    return _value_from_dict(
        _qrs_t_distances,
        metric,
        "QRS/T curves are unavailable or degenerate",
    )


# --- time-local forces, ST vectors, and phase coupling --------------------


def _window_vector(
    ctx: FeatureContext,
    alignment: str,
    window_ms: int,
) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, "qrs")
        if points is None:
            return None
        count = max(1, int(round(window_ms * ctx.fs / 1000.0)))
        if len(points) < count:
            return None
        selected = points[:count] if alignment == "initial" else points[-count:]
        vector = np.mean(selected, axis=0)
        magnitude = float(np.linalg.norm(vector))
        azimuth = vcg_azimuth_deg(vector)
        elevation = vcg_elevation_deg(vector)
        return {
            "x": float(vector[0]),
            "y": float(vector[1]),
            "z": float(vector[2]),
            "magnitude": magnitude,
            "azimuth": float(azimuth) if azimuth is not None else float("nan"),
            "elevation": float(elevation) if elevation is not None else float("nan"),
        }

    return _memo(ctx, ("qrs_window_vector", alignment, window_ms), build)


def _window_vector_calculator(alignment: str, window_ms: int, component: str):
    return _value_from_dict(
        lambda ctx: _window_vector(ctx, alignment, window_ms),
        component,
        f"{alignment} {window_ms}-ms QRS force is unavailable or directionless",
    )


def _qrs_dynamics(ctx: FeatureContext) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, "qrs")
        if points is None or len(points) < 12:
            return None
        halves = {"early": points[: len(points) // 2 + 1], "late": points[len(points) // 2 :]}
        output: dict[str, float] = {}
        for label, part in halves.items():
            absolute_area = float(np.trapezoid(np.linalg.norm(part, axis=1), dx=1.0 / ctx.fs))
            net_area = np.trapezoid(part, dx=1.0 / ctx.fs, axis=0)
            output[f"{label}_cancellation_fraction"] = (
                1.0 - float(np.linalg.norm(net_area)) / absolute_area
                if absolute_area > EPS
                else float("nan")
            )
            speed = np.linalg.norm(np.diff(_smooth(part, 1.0), axis=0), axis=1) * ctx.fs
            positive = speed[speed > EPS]
            if len(positive) < 3:
                output[f"{label}_stall_fraction"] = float("nan")
                output[f"{label}_burst_index"] = float("nan")
                continue
            # Relative, record-internal thresholds avoid pretending that a
            # novel absolute velocity cutoff is clinically established.
            stall_cutoff = 0.20 * float(np.median(positive))
            output[f"{label}_stall_fraction"] = float(np.mean(speed <= stall_cutoff))
            output[f"{label}_burst_index"] = float(
                np.percentile(positive, 95) / max(float(np.median(positive)), EPS)
            )
        return output

    return _memo(ctx, ("qrs_dynamics",), build)


def _qrs_dynamic_calculator(metric: str):
    return _value_from_dict(_qrs_dynamics, metric, "QRS dynamics are unavailable")


def _terminal_sector_metrics(ctx: FeatureContext) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, "qrs")
        if points is None:
            return None
        count = max(3, int(round(0.040 * ctx.fs)))
        tail = points[-count:]
        radius = np.linalg.norm(tail, axis=1)
        valid = radius > max(EPS, 0.05 * float(np.max(radius)))
        tail = tail[valid]
        if len(tail) < 3:
            return None
        signs = (tail >= 0).astype(int)
        octants = signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]
        counts = np.bincount(octants, minlength=8)
        return {f"octant_{index:03b}_fraction": float(counts[index] / len(tail)) for index in range(8)}

    return _memo(ctx, ("terminal_sector_metrics",), build)


def _terminal_sector_calculator(metric: str):
    return _value_from_dict(
        _terminal_sector_metrics,
        metric,
        "terminal 40-ms QRS direction sector is unavailable",
    )


def _st_offset_vector(ctx: FeatureContext, offset_ms: int) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        signal = ctx.origin_corrected_vcg()
        j_point = ctx._st_start()
        t_end = ctx._sample("t_offset")
        if signal is None or j_point is None:
            return None
        index = int(j_point + round(offset_ms * ctx.fs / 1000.0))
        if index >= len(signal) or (t_end is not None and index > t_end):
            return None
        vector = np.asarray(signal[index], dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            return None
        azimuth = vcg_azimuth_deg(vector)
        elevation = vcg_elevation_deg(vector)
        return {
            "x": float(vector[0]),
            "y": float(vector[1]),
            "z": float(vector[2]),
            "azimuth": float(azimuth) if azimuth is not None else float("nan"),
            "elevation": float(elevation) if elevation is not None else float("nan"),
        }

    return _memo(ctx, ("st_offset_vector", offset_ms), build)


def _st_offset_calculator(offset_ms: int, component: str):
    return _value_from_dict(
        lambda ctx: _st_offset_vector(ctx, offset_ms),
        component,
        f"J+{offset_ms} vector is unavailable or directionless",
    )


def _t_phase_metrics(ctx: FeatureContext) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, "t")
        if points is None or len(points) < 8:
            return None
        midpoint = len(points) // 2
        vectors = {
            "early": np.mean(points[: midpoint + 1], axis=0),
            "late": np.mean(points[midpoint:], axis=0),
        }
        output: dict[str, float] = {}
        for label, vector in vectors.items():
            azimuth = vcg_azimuth_deg(vector)
            elevation = vcg_elevation_deg(vector)
            output[f"{label}_magnitude"] = float(np.linalg.norm(vector))
            output[f"{label}_azimuth"] = float(azimuth) if azimuth is not None else float("nan")
            output[f"{label}_elevation"] = float(elevation) if elevation is not None else float("nan")

        bounds = ctx.bounds("t")
        peak = ctx._sample("t_peak")
        if bounds is None or peak is None or not bounds[0] <= peak <= bounds[1]:
            tp_te = None
        else:
            tp_te = points[peak - bounds[0] :]
        if tp_te is None or len(tp_te) < 2:
            output["tp_te_path_length_mv"] = float("nan")
            output["tp_te_straightness_fraction"] = float("nan")
        else:
            path = _polyline_length(tp_te)
            output["tp_te_path_length_mv"] = path
            output["tp_te_straightness_fraction"] = (
                float(np.linalg.norm(tp_te[-1] - tp_te[0]) / path) if path > EPS else float("nan")
            )

        initial = _window_vector(ctx, "initial", 10)
        terminal = _window_vector(ctx, "terminal", 40)
        output["initial_qrs_early_t_angle_deg"] = (
            float(vector_angle_deg(np.asarray([initial[k] for k in ("x", "y", "z")]), vectors["early"]))
            if initial is not None
            and vector_angle_deg(np.asarray([initial[k] for k in ("x", "y", "z")]), vectors["early"]) is not None
            else float("nan")
        )
        output["terminal_qrs_late_t_angle_deg"] = (
            float(vector_angle_deg(np.asarray([terminal[k] for k in ("x", "y", "z")]), vectors["late"]))
            if terminal is not None
            and vector_angle_deg(np.asarray([terminal[k] for k in ("x", "y", "z")]), vectors["late"]) is not None
            else float("nan")
        )
        return output

    return _memo(ctx, ("t_phase_metrics",), build)


def _t_phase_calculator(metric: str):
    return _value_from_dict(_t_phase_metrics, metric, "T phase geometry is unavailable")


# --- published disease-oriented constructions ----------------------------


def _lead_baseline(ctx: FeatureContext, lead: str) -> float:
    pr = ctx.ecg_segment(lead, "pr")
    if pr is None or len(pr) < 3 or not np.all(np.isfinite(pr)):
        source = getattr(ctx, "_record_ecg_12lead", None)
        if source is None:
            source = ctx.ecg_12lead
        windows: list[np.ndarray] = []
        if source is not None and lead in LEAD_MAP:
            column = LEAD_MAP[lead]
            for first, second in zip(ctx.delineation.beats[:-1], ctx.delineation.beats[1:]):
                end_point = getattr(first, "t_offset", None)
                start_point = getattr(second, "p_onset", None) or getattr(second, "r_onset", None)
                if end_point is None or start_point is None:
                    continue
                start, end = int(end_point.sample), int(start_point.sample)
                if end - start >= 3 and 0 <= start < end <= len(source):
                    windows.append(np.asarray(source[start:end, column], dtype=float))
        if windows:
            return float(np.median(np.concatenate(windows)))
        return 0.0
    return float(np.median(pr))


def _signed_absolute_peak(values: np.ndarray) -> float:
    return float(values[int(np.argmax(np.abs(values)))])


def _arvc_rpd_metrics(ctx: FeatureContext) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        required = ("V5", "II", "V1")
        if ctx.ecg_12lead is None or any(lead not in LEAD_MAP for lead in required):
            return None
        qrs: dict[str, np.ndarray] = {}
        t_wave: dict[str, np.ndarray] = {}
        for lead in required:
            q_values = ctx.ecg_segment(lead, "qrs")
            t_values = ctx.ecg_segment(lead, "t")
            if q_values is None or t_values is None:
                return None
            baseline = _lead_baseline(ctx, lead)
            qrs[lead] = np.asarray(q_values, dtype=float) - baseline
            t_wave[lead] = np.asarray(t_values, dtype=float) - baseline
        # Cortez et al. Fig. 1: S(V5) only, signed largest R/S in II,
        # and -0.5*R(V1).  T uses V5, II and -0.5*V1 deviations.
        q_vector = np.asarray(
            [
                min(0.0, float(np.min(qrs["V5"]))),
                _signed_absolute_peak(qrs["II"]),
                -0.5 * max(0.0, float(np.max(qrs["V1"]))),
            ]
        )
        t_vector = np.asarray(
            [
                _signed_absolute_peak(t_wave["V5"]),
                _signed_absolute_peak(t_wave["II"]),
                -0.5 * _signed_absolute_peak(t_wave["V1"]),
            ]
        )
        magnitude = float(np.linalg.norm(q_vector))
        angle = vector_angle_deg(q_vector, t_vector)
        return {
            "rtrms_qrs_mv": magnitude,
            "rpd_qrs_t_angle_deg": float(angle) if angle is not None else float("nan"),
        }

    return _memo(ctx, ("arvc_rpd_metrics",), build)


def _arvc_calculator(metric: str):
    return _value_from_dict(
        _arvc_rpd_metrics,
        metric,
        "ARVC right-precordial lead construction is unavailable",
    )


def _directional_peak_force(points: np.ndarray, axis: int, positive: bool) -> float:
    values = points[:, axis]
    selected = np.maximum(values, 0.0) if positive else np.maximum(-values, 0.0)
    return float(np.max(selected))


def _disease_geometry(ctx: FeatureContext) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        qrs = _segment(ctx, "qrs")
        p_loop = _segment(ctx, "p")
        t_wave = _segment(ctx, "t")
        if qrs is None and p_loop is None and t_wave is None:
            return None
        output: dict[str, Optional[float]] = {}
        if qrs is not None:
            q_posterior = _directional_peak_force(qrs, 2, True)
            q_anterior = _directional_peak_force(qrs, 2, False)
            q_left = _directional_peak_force(qrs, 0, True)
            q_right = _directional_peak_force(qrs, 0, False)
            q_radius = np.linalg.norm(qrs, axis=1)
            total_voltage = float(np.sum(q_radius))
            initial_count = max(1, int(round(0.020 * ctx.fs)))
            initial = qrs[:initial_count]
            terminal = qrs[-initial_count:]
            output.update(
                {
                    "rvh_qrs_posterior_anterior_force_ratio": _safe_ratio(q_posterior, q_anterior),
                    "rvh_qrs_right_left_force_ratio": _safe_ratio(q_right, q_left),
                    "rvh_qrs_posterior_left_force_ratio": _safe_ratio(q_posterior, q_left),
                    "hcm_qrs_anterior_residence_fraction": float(np.mean(qrs[:, 2] < 0.0)),
                    "hcm_qrs_posterior_voltage_fraction": (
                        float(np.sum(q_radius[qrs[:, 2] > 0.0]) / total_voltage)
                        if total_voltage > EPS
                        else float("nan")
                    ),
                    "hcm_initial20_left_anterior_sector_fraction": float(
                        np.mean((initial[:, 0] > 0.0) & (initial[:, 2] < 0.0))
                    ),
                    "hcm_terminal20_right_posterior_sector_fraction": float(
                        np.mean((terminal[:, 0] < 0.0) & (terminal[:, 2] > 0.0))
                    ),
                }
            )
        if p_loop is not None:
            p_anterior = _directional_peak_force(p_loop, 2, False)
            p_posterior = _directional_peak_force(p_loop, 2, True)
            output["rvh_p_anterior_posterior_force_ratio"] = _safe_ratio(p_anterior, p_posterior)
        if t_wave is not None:
            t_centered = t_wave - np.mean(t_wave, axis=0)
            _, singular, vh = np.linalg.svd(t_centered, full_matrices=False)
            t_principal = t_centered @ vh.T
            major_span = float(np.ptp(t_principal[:, 0]))
            radial = np.linalg.norm(t_principal[:, 1:], axis=1)
            strip_widths: list[float] = []
            if major_span > EPS:
                bins = np.linspace(np.min(t_principal[:, 0]), np.max(t_principal[:, 0]), 13)
                labels = np.clip(np.digitize(t_principal[:, 0], bins) - 1, 0, 11)
                for bin_index in range(12):
                    selected = radial[labels == bin_index]
                    if len(selected) >= 2:
                        strip_widths.append(float(np.percentile(selected, 75) - np.percentile(selected, 25)))
            output["hcm_t_3d_median_tube_slenderness"] = (
                float(np.median(strip_widths) / major_span)
                if strip_widths and major_span > EPS and len(singular) >= 2
                else float("nan")
            )
            normal = _oriented_normal(t_wave)
            if normal is None:
                tavplan_mean = tavplan_rms = tavplan_ratio = float("nan")
            else:
                plane_distance = np.abs(t_centered @ normal)
                tavplan_mean = float(np.mean(plane_distance) * 1000.0)
                tavplan_rms = float(np.sqrt(np.mean(plane_distance**2)) * 1000.0)
                t_amplitude = float(np.max(np.linalg.norm(t_wave, axis=1)) * 1000.0)
                tavplan_ratio = tavplan_mean / t_amplitude if t_amplitude > EPS else float("nan")
            output.update(
                {
                    "tavplan_mean_uv": tavplan_mean,
                    "tavplan_rms_uv": tavplan_rms,
                    "tavplan_to_t_amplitude_ratio": tavplan_ratio,
                }
            )
        return {key: float(value) if value is not None else float("nan") for key, value in output.items()}

    return _memo(ctx, ("disease_geometry",), build)


def _disease_calculator(metric: str):
    return _value_from_dict(
        _disease_geometry,
        metric,
        "disease-oriented loop geometry is unavailable or has a zero denominator",
    )


# --- multiscale geometry and genuine radial lower-star persistence --------


def _radial_h0_persistence(radius: np.ndarray) -> np.ndarray:
    """Finite H0 lifetimes of a cyclic 1-D lower-star filtration.

    This is an actual union-find persistence computation.  It intentionally
    replaces neither the retired catalog placeholder nor a full point-cloud
    Vietoris-Rips persistence diagram; it answers the narrower, well-defined
    question of persistent radial valleys/lobes along the ordered loop.
    """

    values = np.asarray(radius, dtype=float).ravel()
    count = len(values)
    if count < 4 or not np.all(np.isfinite(values)):
        return np.asarray([], dtype=float)
    parent = np.arange(count)
    birth = values.copy()
    active = np.zeros(count, dtype=bool)
    lifetimes: list[float] = []

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            following = int(parent[index])
            parent[index] = root
            index = following
        return root

    for index in np.argsort(values, kind="stable"):
        index = int(index)
        active[index] = True
        for neighbour in ((index - 1) % count, (index + 1) % count):
            if not active[neighbour]:
                continue
            left, right = find(index), find(neighbour)
            if left == right:
                continue
            # Elder rule, deterministic tie break.
            if (birth[left], left) <= (birth[right], right):
                older, younger = left, right
            else:
                older, younger = right, left
            lifetimes.append(max(0.0, float(values[index] - birth[younger])))
            parent[younger] = older
    return np.asarray([value for value in lifetimes if value > 1e-8], dtype=float)


def _topology_metrics(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 128)
        if points is None:
            return None
        centered = points - np.mean(points, axis=0)
        rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        if rms <= EPS:
            return None

        # PCA alignment removes rigid 3-D rotation before axis-aligned box
        # counting.  Isotropic scaling preserves the curve's anisotropy.
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        aligned = centered @ vh.T
        scale = float(np.max(np.ptp(aligned, axis=0)))
        normalized = (aligned - np.min(aligned, axis=0)) / max(scale, EPS)
        subdivisions = np.asarray([2, 4, 8, 16, 32], dtype=float)
        occupied_counts: list[float] = []
        for subdivision in subdivisions.astype(int):
            cells = np.floor(normalized * subdivision).astype(int)
            cells = np.minimum(cells, subdivision - 1)
            occupied_counts.append(float(len(np.unique(cells, axis=0))))
        occupied_array = np.asarray(occupied_counts)
        usable = (occupied_array > 1) & (occupied_array < 0.95 * len(points))
        if np.count_nonzero(usable) >= 3:
            box_dimension = float(
                np.polyfit(np.log(subdivisions[usable]), np.log(occupied_array[usable]), 1)[0]
            )
        else:
            box_dimension = float("nan")

        residuals: list[float] = []
        scales = np.asarray([1.0, 2.0, 4.0, 8.0])
        for sigma in scales:
            smoothed = ndimage.gaussian_filter1d(points, sigma=sigma, axis=0, mode="nearest")
            residuals.append(float(np.sqrt(np.mean(np.sum((points - smoothed) ** 2, axis=1))) / rms))
        residual_array = np.asarray(residuals)
        usable_roughness = residual_array > EPS
        roughness_slope = (
            float(np.polyfit(np.log(scales[usable_roughness]), np.log(residual_array[usable_roughness]), 1)[0])
            if np.count_nonzero(usable_roughness) >= 3
            else float("nan")
        )

        radial = np.linalg.norm(points, axis=1)
        radial_scale = float(np.max(radial))
        if radial_scale <= EPS:
            persistence_max = persistence_entropy = float("nan")
        else:
            lifetimes = _radial_h0_persistence(radial / radial_scale)
            if len(lifetimes):
                probability = lifetimes / np.sum(lifetimes)
                persistence_max = float(np.max(lifetimes))
                persistence_entropy = (
                    float(-np.sum(probability * np.log(probability)) / np.log(len(probability)))
                    if len(probability) > 1
                    else 0.0
                )
            else:
                # No finite off-diagonal class is a valid topological result,
                # not missing data.
                persistence_max = 0.0
                persistence_entropy = 0.0
        return {
            "box_counting_dimension": box_dimension,
            "multiscale_roughness_slope": roughness_slope,
            "radial_h0_max_persistence_fraction": persistence_max,
            "radial_h0_persistence_entropy": persistence_entropy,
        }

    return _memo(ctx, ("topology_metrics", segment), build)


def _topology_calculator(segment: str, metric: str):
    return _value_from_dict(
        lambda ctx: _topology_metrics(ctx, segment),
        metric,
        f"{segment} multiscale topology is unavailable",
    )


# --- declarative scalar catalog ------------------------------------------


def _spec(
    name: str,
    family: str,
    description: str,
    formula: str,
    unit: str,
    dependencies: tuple[str, ...],
    evidence_level: str,
    evidence_note: str,
    references: tuple[str, ...],
    calculator: Callable[[FeatureContext], FeatureValue],
    *,
    visual: bool | str = False,
) -> DiscoveryV3Spec:
    visual_assessment = (
        "direct"
        if visual is True
        else "mathematical_only"
        if visual is False
        else str(visual)
    )
    if visual_assessment not in {"direct", "visual_correlate", "mathematical_only"}:
        raise ValueError(f"Unknown visual assessment: {visual_assessment}")
    return DiscoveryV3Spec(
        name=name,
        family=family,
        description=description,
        formula=formula,
        unit=unit,
        dependencies=dependencies,
        evidence_level=evidence_level,
        evidence_note=evidence_note,
        references=references,
        calculator=calculator,
        visual_assessment=visual_assessment,
    )


def _build_specs() -> tuple[DiscoveryV3Spec, ...]:
    specs: list[DiscoveryV3Spec] = []

    rotation_meta = {
        "signed_rotation_turns": (
            "signed_rotation_turns",
            "Net signed inscription of the projected loop; negative is clockwise in the named displayed axes.",
            "sum(wrap(delta atan2(v,u)))/(2*pi)",
            "turns",
        ),
        "clockwise_path_fraction": (
            "clockwise_path_fraction",
            "Projected path-length fraction inscribed clockwise.",
            "sum(ds where delta_theta<0)/sum(ds)",
            "fraction",
        ),
        "rotation_reversal_count": (
            "rotation_reversal_count",
            "Number of robust clockwise/counterclockwise inscription reversals.",
            "count(sign(delta_theta_i) != sign(delta_theta_{i-1}))",
            "count",
        ),
        "rotation_reversal_fraction": (
            "rotation_reversal_fraction",
            "Fraction of eligible angular transitions that reverse inscription.",
            "rotation_reversal_count/(N_angular_steps-1)",
            "fraction",
        ),
    }
    morphology_meta = {
        "grid_occupancy_fraction": (
            "grid_occupancy_fraction",
            "Fraction of adaptive 12 by 12 display cells visited by the loop.",
            "occupied_cells/144",
            "fraction",
        ),
        "grid_entropy_fraction": (
            "grid_entropy_fraction",
            "Normalized entropy of loop dwell across an adaptive 12 by 12 display grid.",
            "-sum(p_cell log(p_cell))/log(144)",
            "fraction",
        ),
        "hull_bite_depth_fraction": (
            "hull_bite_depth_fraction",
            "Robust depth of inward bites relative to projected loop diameter.",
            "p95(distance(curve, convex_hull_boundary))/diameter",
            "fraction",
        ),
        "persistent_cusp_count": (
            "persistent_cusp_count",
            "Count of visually sharp cusps surviving two smoothing scales.",
            "min_sigma count(prominent turning-angle peaks)",
            "count",
        ),
        "orientation_lobe_count": (
            "orientation_lobe_count",
            "Number of robust signed local-area runs, a visible multi-lobe count.",
            "1+count(sign(cross(r_i-c,r_{i+1}-c)) changes)",
            "count",
        ),
        "median_strip_slenderness": (
            "median_strip_slenderness",
            "Median occupied minor-axis strip width divided by major-axis span.",
            "median_b range(minor | major_bin=b)/range(major)",
            "dimensionless",
        ),
    }
    for segment in ("qrs", "t"):
        for plane in PLANES:
            for metric, (suffix, description, formula, unit) in rotation_meta.items():
                specs.append(
                    _spec(
                        f"{segment}_{plane}_{suffix}",
                        "Discovery-v3 visible plane inscription",
                        f"{description} Segment={segment.upper()}, plane={plane}.",
                        formula,
                        unit,
                        (segment,),
                        "B" if metric in {"signed_rotation_turns", "clockwise_path_fraction"} else "C",
                        (
                            "Directions of inscription are established VCG descriptors; the continuous scalar is exploratory."
                            if metric in {"signed_rotation_turns", "clockwise_path_fraction"}
                            else "Loop folding/reversal is described in small cohorts; this robust count/fraction needs validation."
                        ),
                        (REF_NORMAL_VCG, REF_LOOP_FOLDING, REF_SYNTHESIS),
                        _plane_metric_calculator(segment, plane, metric),
                        visual="direct",
                    )
                )
            for metric, (suffix, description, formula, unit) in morphology_meta.items():
                specs.append(
                    _spec(
                        f"{segment}_{plane}_{suffix}",
                        "Discovery-v3 visible plane morphology",
                        f"{description} Segment={segment.upper()}, plane={plane}.",
                        formula,
                        unit,
                        (segment,),
                        "C" if metric in {"persistent_cusp_count", "median_strip_slenderness"} else "D",
                        "Visible loop morphology is clinically plausible; this exact robust scalar has no validated cutoff.",
                        (REF_LOOP_MORPHOLOGY, REF_LOOP_FOLDING),
                        _plane_metric_calculator(segment, plane, metric),
                        visual=(
                            "visual_correlate"
                            if metric in {"grid_occupancy_fraction", "grid_entropy_fraction"}
                            else "direct"
                        ),
                    )
                )

    temporal_meta = {
        "temporal_color_gradient_r2": (
            "Linear recoverability of normalized time from 3-D loop position, a scalar surrogate for a clean color gradient.",
            "R2(t_norm ~ X_norm+Y_norm+Z_norm)",
            "fraction",
        ),
        "initial_central_plane_twist_deg": (
            "Unsigned angle between initial- and central-third preferential planes.",
            "acos(abs(n_initial dot n_central))",
            "degrees",
        ),
        "central_terminal_plane_twist_deg": (
            "Unsigned angle between central- and terminal-third preferential planes.",
            "acos(abs(n_central dot n_terminal))",
            "degrees",
        ),
        "initial_terminal_plane_twist_deg": (
            "Unsigned angle between initial- and terminal-third preferential planes.",
            "acos(abs(n_initial dot n_terminal))",
            "degrees",
        ),
        "integrated_sliding_plane_twist_deg": (
            "Accumulated local preferential-plane twist along the loop.",
            "sum_k acos(abs(n_k dot n_{k+1}))",
            "degrees",
        ),
        "maximum_sliding_plane_twist_deg": (
            "Largest local preferential-plane twist step along the loop.",
            "max_k acos(abs(n_k dot n_{k+1}))",
            "degrees",
        ),
    }
    for segment in ("qrs", "t"):
        for metric, (description, formula, unit) in temporal_meta.items():
            specs.append(
                _spec(
                    f"{segment}_{metric}",
                    "Discovery-v3 temporal 3-D loop readability",
                    f"{description} Segment={segment.upper()}.",
                    formula,
                    unit,
                    (segment,),
                    "C" if "plane_twist" in metric else "D",
                    "Plane alteration has small-cohort VCG support; exact temporal summaries are prospective.",
                    (REF_LOOP_MORPHOLOGY, REF_LOOP_FOLDING),
                    _temporal_calculator(segment, metric),
                    visual=True,
                )
            )

    distance_meta = {
        "discrete_frechet_mv": ("Discrete Fréchet leash distance between ordered QRS and T curves.", "d_F(Q,T)", "mV", REF_FRECHET),
        "normalized_discrete_frechet": ("Shape-only discrete Fréchet distance after separate centering and RMS scaling.", "d_F(norm(Q),norm(T))", "dimensionless", REF_FRECHET),
        "directional_discrete_frechet_deg": ("Discrete Fréchet distance between origin-referenced QRS and T direction paths.", "d_F^angle(unit(Q),unit(T))", "degrees", REF_FRECHET),
        "derivative_direction_dtw": ("Mean DTW cost between normalized tangent-direction sequences.", "DTW(versor(dQ),versor(dT))/path_length", "dimensionless", REF_DTW),
        "energy_distance_sqrt_mv": ("Multivariate energy distance between QRS and T point distributions.", "sqrt(2E||Q-T||-E||Q-Q'||-E||T-T'||)", "sqrt(mV)", REF_ENERGY_DISTANCE),
        "normalized_energy_distance": ("Energy distance after separate centering and RMS scaling.", "energy(norm(Q),norm(T))", "dimensionless", REF_ENERGY_DISTANCE),
        "sliced_wasserstein_mean_mv": ("Mean one-dimensional Wasserstein distance over 32 fixed sphere directions.", "mean_u W1(<Q,u>,<T,u>)", "mV", REF_SLICED_WASSERSTEIN),
        "sliced_wasserstein_p90_mv": ("90th percentile sliced Wasserstein distance over 32 sphere directions.", "p90_u W1(<Q,u>,<T,u>)", "mV", REF_SLICED_WASSERSTEIN),
        "nearest_direction_angle_median_deg": ("Median symmetric nearest-neighbour angle between QRS and T direction clouds.", "median(min angular cross-distance)", "degrees", REF_DYNAMIC_VCG),
        "nearest_direction_angle_p90_deg": ("90th percentile symmetric nearest-neighbour angle between QRS and T direction clouds.", "p90(min angular cross-distance)", "degrees", REF_DYNAMIC_VCG),
        "local_alignment_dwell_fraction": ("Phase-aligned dwell fraction with QRS/T directional angle at most 60 degrees.", "mean(cos(Q_i,T_i)>=0.5)", "fraction", REF_DYNAMIC_VCG),
        "local_opposition_dwell_fraction": ("Phase-aligned dwell fraction with QRS/T directional angle at least 120 degrees.", "mean(cos(Q_i,T_i)<=-0.5)", "fraction", REF_DYNAMIC_VCG),
        "early_phase_rms_point_distance_mv": ("RMS QRS/T point distance over the first phase third.", "RMS(||Q_i-T_i||), i in first third", "mV", REF_DYNAMIC_VCG),
        "late_phase_rms_point_distance_mv": ("RMS QRS/T point distance over the last phase third.", "RMS(||Q_i-T_i||), i in last third", "mV", REF_DYNAMIC_VCG),
    }
    for metric, (description, formula, unit, reference) in distance_meta.items():
        specs.append(
            _spec(
                f"qrs_t_{metric}",
                "Discovery-v3 QRS-T curve distance",
                description,
                formula,
                unit,
                ("qrs", "t"),
                "D",
                "The metric is mathematically established, but its use as a clinical VCG biomarker is unvalidated.",
                (reference, REF_DYNAMIC_VCG),
                _distance_calculator(metric),
                visual=(
                    "mathematical_only"
                    if metric in {
                        "energy_distance_sqrt_mv",
                        "normalized_energy_distance",
                        "sliced_wasserstein_mean_mv",
                        "sliced_wasserstein_p90_mv",
                    }
                    else "visual_correlate"
                ),
            )
        )

    for alignment in ("initial", "terminal"):
        for window_ms in (10, 20, 30, 40):
            for component in ("magnitude", "azimuth", "elevation"):
                if window_ms == 20 and component == "magnitude":
                    continue  # exact legacy scalar already exists
                angular = component in {"azimuth", "elevation"}
                specs.append(
                    _spec(
                        f"qrs_{alignment}_{window_ms}ms_force_{component}_{'deg' if angular else 'mv'}",
                        "Discovery-v3 time-local QRS force",
                        f"{component.capitalize()} of the mean origin-referenced {alignment} {window_ms}-ms QRS force.",
                        f"{component}(mean(r_QRS(t), {alignment} {window_ms} ms))",
                        "degrees" if angular else "mV",
                        ("qrs",),
                        "B" if window_ms in {20, 30} else "C",
                        "Initial 20/30-ms vectors and inscription have normative human data; other windows are interpolated hypotheses.",
                        (REF_NORMAL_VCG,),
                        _window_vector_calculator(alignment, window_ms, component),
                        visual=True,
                    )
                )
    for alignment, window_ms in (("initial", 10), ("terminal", 40)):
        for component in ("x", "y", "z"):
            specs.append(
                _spec(
                    f"qrs_{alignment}_{window_ms}ms_force_{component}_mv",
                    "Discovery-v3 time-local QRS force",
                    f"Frank-{component.upper()} component of the mean {alignment} {window_ms}-ms QRS force.",
                    f"mean({component}(t), {alignment} {window_ms} ms)",
                    "mV",
                    ("qrs",),
                    "C",
                    f"Coordinate dependent under {VCG_AXIS_CONVENTION}; no cutoff is claimed.",
                    (REF_NORMAL_VCG,),
                    _window_vector_calculator(alignment, window_ms, component),
                    visual="direct",
                )
            )

    for phase in ("early", "late"):
        for metric, unit, formula in (
            ("cancellation_fraction", "fraction", "1-||integral r dt||/integral ||r|| dt"),
            ("stall_fraction", "fraction", "mean(speed <= 0.2*median(nonzero speed))"),
            ("burst_index", "ratio", "p95(nonzero speed)/median(nonzero speed)"),
        ):
            key = f"{phase}_{metric}"
            specs.append(
                _spec(
                    f"qrs_{key}",
                    "Discovery-v3 QRS activation dynamics",
                    f"{metric.replace('_', ' ').capitalize()} during the {phase} QRS half.",
                    formula,
                    unit,
                    ("qrs",),
                    "D",
                    "Mechanistically motivated cancellation/stall/burst hypothesis without clinical validation.",
                    (REF_LOOP_FOLDING,),
                    _qrs_dynamic_calculator(key),
                    visual="visual_correlate",
                )
            )

    for octant in range(8):
        key = f"octant_{octant:03b}_fraction"
        signs = tuple("+" if bit == "1" else "-" for bit in f"{octant:03b}")
        specs.append(
            _spec(
                f"qrs_terminal40_xyz_{key}",
                "Discovery-v3 terminal activation sectors",
                f"Terminal 40-ms residence in Frank sector X{signs[0]} Y{signs[1]} Z{signs[2]}.",
                f"count_terminal40(octant={octant:03b})/N_terminal40",
                "fraction",
                ("qrs",),
                "D",
                f"Coordinate dependent under {VCG_AXIS_CONVENTION}; intended for sector discovery.",
                (REF_NORMAL_VCG,),
                _terminal_sector_calculator(key),
                visual="direct",
            )
        )

    for offset_ms in (20, 40, 60, 80):
        for component in ("x", "y", "z", "azimuth", "elevation"):
            angular = component in {"azimuth", "elevation"}
            specs.append(
                _spec(
                    f"st_j_plus_{offset_ms}ms_{component}_{'deg' if angular else 'mv'}",
                    "Discovery-v3 ST injury-vector trajectory",
                    f"{component.capitalize()} of the origin-referenced ST vector at J+{offset_ms} ms.",
                    f"{component}(r(J+{offset_ms} ms))",
                    "degrees" if angular else "mV",
                    ("st",),
                    "B" if offset_ms == 60 else "C",
                    "J+60-ms ST vector is established in human ischemia studies; adjacent time samples are prospective trajectory extensions.",
                    (REF_ST_OFFSETS, REF_T_LOOP),
                    _st_offset_calculator(offset_ms, component),
                    visual="direct",
                )
            )

    for phase in ("early", "late"):
        for component in ("magnitude", "azimuth", "elevation"):
            angular = component in {"azimuth", "elevation"}
            key = f"{phase}_{component}"
            specs.append(
                _spec(
                    f"t_{phase}_trajectory_{component}_{'deg' if angular else 'mv'}",
                    "Discovery-v3 T trajectory phases",
                    f"{component.capitalize()} of the mean {phase} T-loop trajectory.",
                    f"{component}(mean(r_T(t), {phase} half))",
                    "degrees" if angular else "mV",
                    ("t",),
                    "C",
                    "T-loop direction is clinically studied; the half-loop summary is an exploratory scalarization.",
                    (REF_T_LOOP, REF_DYNAMIC_VCG),
                    _t_phase_calculator(key),
                    visual=True,
                )
            )
    for metric, description, formula, unit in (
        ("tp_te_path_length_mv", "3-D path length from T peak to T end.", "sum_{Tp:Te} ||delta r||", "mV"),
        ("tp_te_straightness_fraction", "Straight-line displacement divided by path length from T peak to T end.", "||r_Te-r_Tp||/sum||delta r||", "fraction"),
        ("initial_qrs_early_t_angle_deg", "Angle between initial 10-ms QRS force and early T force.", "angle(QRS_initial10,T_early)", "degrees"),
        ("terminal_qrs_late_t_angle_deg", "Angle between terminal 40-ms QRS force and late T force.", "angle(QRS_terminal40,T_late)", "degrees"),
    ):
        specs.append(
            _spec(
                metric,
                "Discovery-v3 activation-repolarization trajectory coupling",
                description,
                formula,
                unit,
                ("qrs", "t") if "qrs" in metric else ("t",),
                "C" if metric.startswith("tp_te") else "D",
                "Tpeak-end and QRS-T relations have human evidence; this spatial trajectory formulation needs validation.",
                (REF_T_LOOP, REF_DYNAMIC_VCG),
                _t_phase_calculator(metric),
                visual="direct",
            )
        )

    literature_specs = (
        ("arvc_rtrms_qrs_mv", "ARVC right-precordial-directed QRS magnitude from S(V5), max-absolute II and -0.5R(V1).", "sqrt(S_V5^2 + peak_II^2 + (-0.5 R_V1)^2)", "mV", ("qrs",), "B", (REF_ARVC,), _arvc_calculator("rtrms_qrs_mv")),
        ("arvc_rpd_qrs_t_angle_deg", "ARVC right-precordial-directed angle between the published restricted QRS and T constructions.", "angle([S_V5,peak_II,-0.5R_V1],[T_V5,T_II,-0.5T_V1])", "degrees", ("qrs", "t"), "B", (REF_ARVC,), _arvc_calculator("rpd_qrs_t_angle_deg")),
        ("rvh_qrs_posterior_anterior_force_ratio", "Ratio of peak posterior to peak anterior QRS force.", "max(max(Z,0))/max(max(-Z,0))", "ratio", ("qrs",), "B", (REF_RVH_COPD,), _disease_calculator("rvh_qrs_posterior_anterior_force_ratio")),
        ("rvh_qrs_right_left_force_ratio", "Ratio of peak rightward to peak leftward QRS force.", "max(max(-X,0))/max(max(X,0))", "ratio", ("qrs",), "B", (REF_RVH_COPD, REF_RV_OVERLOAD), _disease_calculator("rvh_qrs_right_left_force_ratio")),
        ("rvh_qrs_posterior_left_force_ratio", "Ratio of peak posterior to peak leftward QRS force.", "max(max(Z,0))/max(max(X,0))", "ratio", ("qrs",), "B", (REF_RVH_COPD,), _disease_calculator("rvh_qrs_posterior_left_force_ratio")),
        ("rvh_p_anterior_posterior_force_ratio", "Ratio of peak anterior to peak posterior P-loop force.", "max(max(-Z,0))/max(max(Z,0))", "ratio", ("p",), "B", (REF_RVH_COPD,), _disease_calculator("rvh_p_anterior_posterior_force_ratio")),
        ("hcm_qrs_anterior_residence_fraction", "QRS sample fraction directed anteriorly.", "mean(Z_QRS<0)", "fraction", ("qrs",), "C", (REF_HCM, REF_HCM_REVIEW), _disease_calculator("hcm_qrs_anterior_residence_fraction")),
        ("hcm_qrs_posterior_voltage_fraction", "Voltage-weighted QRS fraction directed posteriorly.", "sum(||r|| where Z>0)/sum||r||", "fraction", ("qrs",), "C", (REF_HCM, REF_HCM_REVIEW), _disease_calculator("hcm_qrs_posterior_voltage_fraction")),
        ("hcm_initial20_left_anterior_sector_fraction", "Initial 20-ms QRS fraction in the left-anterior sector.", "mean(X>0 and Z<0), initial20", "fraction", ("qrs",), "C", (REF_HCM_REVIEW,), _disease_calculator("hcm_initial20_left_anterior_sector_fraction")),
        ("hcm_terminal20_right_posterior_sector_fraction", "Terminal 20-ms QRS fraction in the right-posterior sector.", "mean(X<0 and Z>0), terminal20", "fraction", ("qrs",), "C", (REF_HCM_REVIEW,), _disease_calculator("hcm_terminal20_right_posterior_sector_fraction")),
        ("hcm_t_3d_median_tube_slenderness", "Median 3-D T-loop tube thickness across major-axis bins, normalized by major span.", "median_b IQR(radial_minor | major_bin=b)/range(major)", "fraction", ("t",), "B", (REF_HCM,), _disease_calculator("hcm_t_3d_median_tube_slenderness")),
        ("tavplan_mean_uv", "Mean absolute T-loop distance from its preferential plane (Tavplan).", "mean(abs((r-mean(r)) dot n_plane))*1000", "uV", ("t",), "B", (REF_T_LOOP_OUTCOME, REF_T_LOOP), _disease_calculator("tavplan_mean_uv")),
        ("tavplan_rms_uv", "RMS T-loop distance from its preferential plane.", "RMS((r-mean(r)) dot n_plane)*1000", "uV", ("t",), "C", (REF_T_LOOP_OUTCOME, REF_T_LOOP), _disease_calculator("tavplan_rms_uv")),
        ("tavplan_to_t_amplitude_ratio", "Mean Tavplan divided by peak origin-referenced T magnitude.", "Tavplan_mean/max||T||", "fraction", ("t",), "D", (REF_T_LOOP_OUTCOME, REF_T_LOOP), _disease_calculator("tavplan_to_t_amplitude_ratio")),
    )
    for name, description, formula, unit, dependencies, evidence, references, calculator in literature_specs:
        specs.append(
            _spec(
                name,
                "Discovery-v3 literature-translated disease geometry",
                description,
                formula,
                unit,
                dependencies,
                evidence,
                "The cited disease association supports the construct; thresholds require exact-method external validation.",
                references,
                calculator,
                visual="visual_correlate",
            )
        )

    topology_meta = {
        "box_counting_dimension": ("PCA-aligned multiscale box-counting dimension of the ordered loop point cloud.", "slope(log N_boxes(epsilon),log(1/epsilon))", "dimensionless", REF_BOX_DIMENSION),
        "multiscale_roughness_slope": ("Log-log slope of normalized residual roughness across Gaussian scales.", "slope(log RMS(r-G_sigma*r),log sigma)", "dimensionless", REF_LOOP_MORPHOLOGY),
        "radial_h0_max_persistence_fraction": ("Largest finite H0 lifetime in the cyclic radial lower-star filtration.", "max finite(death-birth), radius/max(radius)", "fraction", REF_PERSISTENCE),
        "radial_h0_persistence_entropy": ("Normalized entropy of finite H0 lifetimes in the cyclic radial lower-star filtration.", "H(lifetime/sum lifetime)/log(n)", "fraction", REF_PERSISTENCE),
    }
    for segment in ("qrs", "t"):
        for metric, (description, formula, unit, reference) in topology_meta.items():
            specs.append(
                _spec(
                    f"{segment}_{metric}",
                    "Discovery-v3 multiscale topology",
                    f"{description} Segment={segment.upper()}.",
                    formula,
                    unit,
                    (segment,),
                    "D",
                    "The mathematical primitive is implemented directly; clinical VCG utility remains unvalidated.",
                    (reference, REF_LOOP_MORPHOLOGY),
                    _topology_calculator(segment, metric),
                    visual=(
                        "mathematical_only"
                        if metric.startswith("radial_h0_")
                        else "visual_correlate"
                    ),
                )
            )

    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"Duplicate Discovery-v3 biomarker names: {duplicates}")
    if len(specs) != 180:
        raise RuntimeError(f"Discovery-v3 must expose exactly 180 scalar hypotheses, got {len(specs)}")
    return tuple(specs)


DISCOVERY_V3_SPECS = _build_specs()


class DiscoveryV3Biomarker(BiomarkerBase):
    def __init__(self, spec: DiscoveryV3Spec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def unit(self) -> str:
        return self.spec.unit

    @property
    def category(self) -> str:
        return "vcg"

    @property
    def references(self) -> list[str]:
        return list(self.spec.references)

    def compute(
        self,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray] = None,
        vcg_xyz: Optional[np.ndarray] = None,
        fs: int = 500,
    ) -> BiomarkerResult:
        # Each dependency is judged against its own wave's floor: the
        # confidence scales are not comparable across waves. The reported
        # confidence stays the minimum across dependencies -- that is the
        # honest summary of how well the weakest one was measured.
        context = _context(delineation, ecg_12lead, vcg_xyz, fs)
        dependency_confidences = {
            item: context.segment_confidence(item)
            for item in self.spec.dependencies
        }
        dependency_confidence = min(
            dependency_confidences.values(),
            default=1.0,
        )
        insufficient = [
            item
            for item in self.spec.dependencies
            if dependency_confidences[item]
            < measurement_confidence_threshold(item)
        ]
        if insufficient:
            return self._result(
                None,
                0.0,
                "not_evaluable: segment confidence below the wave floor for "
                + ", ".join(sorted(insufficient)),
            )
        try:
            feature = self.spec.calculator(context)
        except Exception as exc:
            return self._result(None, 0.0, f"error: {type(exc).__name__}: {exc}")
        value = feature.value
        if value is not None and not np.isfinite(value):
            value = None
        confidence = min(float(feature.confidence), dependency_confidence) if value is not None else 0.0
        return self._result(value, confidence, feature.notes, feature.per_lead)

    def _result(
        self,
        value: Optional[float],
        confidence: float,
        notes: str,
        per_lead: Optional[dict[str, float]] = None,
    ) -> BiomarkerResult:
        visual = self.spec.visual_assessment
        return BiomarkerResult(
            name=self.name,
            value=None if value is None else float(value),
            unit=self.unit,
            category=self.category,
            confidence=float(confidence),
            per_lead=per_lead,
            description=self.description,
            references=self.references,
            computation_notes=(
                f"visual_assessment={visual}; {notes}"
                if notes
                else f"visual_assessment={visual}"
            ),
            family=self.spec.family,
            formula=self.spec.formula,
            implementation_status="implemented",
            pathological_direction="data_driven",
            pathological_note=(
                f"Prospective Discovery-v3 hypothesis; evidence {self.spec.evidence_level}. "
                f"{self.spec.evidence_note} Direction and threshold must be learned in grouped "
                "training data and independently replicated."
            ),
        )


def register_discovery_v3_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in DISCOVERY_V3_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), DiscoveryV3Biomarker):
                continue
            raise RuntimeError(f"Discovery-v3 marker collides with an existing biomarker: {spec.name}")
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=DISCOVERY_V3_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    f"Prospective Discovery-v3 hypothesis; evidence {spec.evidence_level}. "
                    f"{spec.evidence_note} No diagnostic cutoff is claimed."
                ),
                implementation_status="implemented",
                references=list(spec.references),
                notes=(
                    f"{DISCOVERY_V3_VERSION}; {VCG_AXIS_CONVENTION_VERSION}; "
                    f"visual_assessment={spec.visual_assessment}; "
                    "post-development hypothesis; Discovery Reserve only; no Core/Sentinel eligibility."
                ),
            )
        )
        BiomarkerRegistry.register(DiscoveryV3Biomarker(spec))
        existing.add(spec.name)


__all__ = [
    "DISCOVERY_V3_SOURCE",
    "DISCOVERY_V3_SPECS",
    "DISCOVERY_V3_VERSION",
    "DiscoveryV3Biomarker",
    "DiscoveryV3Spec",
    "register_discovery_v3_biomarkers",
]
