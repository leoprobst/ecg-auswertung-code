"""Discovery-v4: uncertainty, exact topology, and what the eye can see.

Three blocks, each answering a question the earlier discovery panels left open.

**Block 1 -- how much of a number is the number, and how much is the method.**
Every VCG marker in this project rests on two choices that are never visible in
the result: which transform reconstructed the vector (Kors or inverse Dower),
and which interval defined the electrical zero (TP or PR). Both are defensible;
neither is the truth. These markers quantify how far the answer moves when the
choice moves. A marker whose value survives the choice is a different kind of
finding from one that does not, and until now nothing in the pipeline could
tell them apart.

**Block 2 -- exact topology instead of proxies.**
The catalog carries several markers whose names promise persistent homology,
alpha shapes or recurrence quantification and whose implementations are
approximations. Approximations are fine; approximations wearing the name of the
exact quantity are not. Everything here is computed exactly for the filtration
it names:

* H0 persistence via the minimum spanning tree. This is not an approximation:
  the multiset of H0 death times of a Vietoris-Rips filtration is exactly the
  multiset of MST edge weights.
* H1 persistence of the radial function on the closed loop. One-dimensional
  persistent homology of a function on a circle is exactly computable by
  pairing local extrema, so this is the real thing -- for that filtration,
  which the name states.
* Full recurrence quantification: determinism, laminarity, trapping time,
  maximal diagonal, and diagonal entropy, from the actual recurrence matrix.
* Real spherical harmonics: coefficients and per-degree power of the direction
  curve on the unit sphere, replacing an entropy marker that carried the name
  without the mathematics.

**Block 3 -- markers for what a reader actually sees.**
A clinician looking at a vector loop does not perceive eigenvalues. They see
that a loop is pinched, that it doubles back, that the second half is scribbled
while the first is smooth, that one beat's loop sits outside the bundle of the
others. Each of those percepts is a measurable geometric statement, and each is
here. ``visual_assessment="direct"`` marks the ones a human can verify by
looking, which makes them the subset worth putting in front of a reader when a
hit needs adjudicating.

All markers are Discovery Reserve. Evidence grades describe prior support for
the construct, never diagnostic validity, and no threshold here is a cutoff.
Anything whose prerequisite is missing abstains; nothing returns a fabricated
zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy import signal as sp_signal
from scipy.spatial import ConvexHull, distance

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from .catalog_biomarkers import EPS, FeatureContext, FeatureValue, _context, _resample_curve
from ..delineation.base import RecordDelineation
from ..vcg_geometry import (
    VCG_AXIS_CONVENTION_VERSION,
    area_vector_uv_ms,
    vcg_azimuth_deg,
    vcg_elevation_deg,
    vector_angle_deg,
)


DISCOVERY_V4_VERSION = "prospective-discovery-v4.0.0"
DISCOVERY_V4_SOURCE = (
    "Prospective VCG uncertainty, exact-topology and visual-morphology audit, "
    "2026-08"
)

REF_KORS = "https://doi.org/10.1093/oxfordjournals.eurheartj.a060332"
REF_DOWER = "https://doi.org/10.1016/0022-0736(80)90008-2"
REF_ARIC_ORIGIN = "https://doi.org/10.1016/j.jelectrocard.2017.08.024"
REF_GEH = "https://physionet.org/content/geh/1.0.0/"
REF_PERSISTENCE = "https://doi.org/10.1007/s00454-002-2885-2"
REF_MST_H0 = "https://doi.org/10.1007/s10208-015-9255-y"
REF_RQA = "https://doi.org/10.1016/j.physrep.2006.11.001"
REF_SPHERICAL_HARMONICS = "https://doi.org/10.1016/j.cag.2003.09.002"
REF_LOOP_MORPHOLOGY = "https://doi.org/10.1016/j.jelectrocard.2015.12.014"
REF_LOOP_FOLDING = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5606210/"
REF_RESPIRATION = "https://doi.org/10.1088/0967-3334/29/1/002"
REF_AMYLOID = "https://doi.org/10.1016/j.jelectrocard.2015.12.014"
REF_BEAT_CLOUD = "https://doi.org/10.1016/j.hrthm.2019.03.006"

PLANES: dict[str, tuple[int, int]] = {
    "frontal_xy": (0, 1),
    "horizontal_xz": (0, 2),
    "sagittal_yz": (1, 2),
}


@dataclass(frozen=True)
class DiscoveryV4Spec:
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
        return self.visual_assessment == "direct"


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(ctx: FeatureContext, key: tuple[object, ...], factory: Callable[[], object]):
    cache = getattr(ctx, "_discovery_v4_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_discovery_v4_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _finite_xyz(value: object, minimum: int = 3) -> Optional[np.ndarray]:
    if value is None:
        return None
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    return points if len(points) >= minimum else None


def _segment(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    return _memo(
        ctx,
        ("segment", segment),
        lambda: _finite_xyz(ctx.vcg_segment_origin_corrected(segment)),
    )


def _resampled(ctx: FeatureContext, segment: str, n: int = 96) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        points = _segment(ctx, segment)
        if points is None:
            return None
        return _finite_xyz(
            np.asarray(_resample_curve(points, n), dtype=float),
            minimum=max(3, n // 2),
        )

    return _memo(ctx, ("resampled", segment, n), build)


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or abs(denominator) <= EPS
    ):
        return None
    return float(numerator / denominator)


def _value_from_dict(
    primitive: Callable[[FeatureContext], Optional[dict[str, float]]],
    key: str,
    reason: str,
) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        values = primitive(ctx)
        if values is None or key not in values:
            return _abstain(reason)
        value = values[key]
        if value is None or not np.isfinite(value):
            return _abstain(reason)
        return FeatureValue(float(value))

    return calculate


def _shannon_entropy(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= EPS:
        return 0.0
    probabilities = weights / total
    probabilities = probabilities[probabilities > EPS]
    if len(probabilities) < 2:
        return 0.0
    return float(
        -np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities))
    )


# ======================================================================
# Block 1 -- method uncertainty
# ======================================================================


def _twelve_lead(ctx: FeatureContext) -> Optional[np.ndarray]:
    source = getattr(ctx, "_record_ecg_12lead", None)
    if source is None:
        return None
    values = np.asarray(source, dtype=float)
    if values.ndim != 2 or values.shape[1] != 12 or len(values) < 8:
        return None
    return values if np.all(np.isfinite(values)) else None


def _transform_ensemble(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """How far the QRS area vector moves when the transform changes.

    Kors and inverse Dower are both accepted reconstructions of the vector
    from 12 leads, and they disagree. The disagreement is not noise to be
    averaged away -- it is the part of a VCG marker that belongs to the
    reconstruction rather than to the patient, and a marker that only exists
    under one transform is a property of that matrix.
    """

    def build() -> Optional[dict[str, float]]:
        ecg = _twelve_lead(ctx)
        bounds = ctx.bounds("qrs")
        if ecg is None or bounds is None:
            return None
        from ..vcg_transform import VCGTransformRegistry

        start, end = bounds
        if end + 1 > len(ecg) or end - start < 3:
            return None
        try:
            transforms = VCGTransformRegistry.transform_all(ecg)
        except Exception:
            return None
        vectors: dict[str, np.ndarray] = {}
        for method, result in transforms.items():
            xyz = np.column_stack([result.vx, result.vy, result.vz])
            segment = xyz[start : end + 1]
            if not np.all(np.isfinite(segment)):
                continue
            vectors[method] = area_vector_uv_ms(segment, ctx.fs)
        if len(vectors) < 2:
            return None
        stacked = np.asarray(list(vectors.values()), dtype=float)
        # Two transforms of a flat signal are two zero vectors, which agree
        # perfectly and mean nothing.
        if float(np.min(np.linalg.norm(stacked, axis=1))) <= EPS:
            return None
        mean_vector = np.mean(stacked, axis=0)
        mean_norm = float(np.linalg.norm(mean_vector))
        angles = [
            vector_angle_deg(vector, mean_vector)
            for vector in stacked
            if vector_angle_deg(vector, mean_vector) is not None
        ]
        magnitudes = np.linalg.norm(stacked, axis=1)
        pairwise_angles: list[float] = []
        for first in range(len(stacked)):
            for second in range(first + 1, len(stacked)):
                angle = vector_angle_deg(stacked[first], stacked[second])
                if angle is not None:
                    pairwise_angles.append(angle)
        return {
            "n_transforms": float(len(stacked)),
            "angular_spread_deg": float(np.max(angles)) if angles else float("nan"),
            "mean_angular_deviation_deg": (
                float(np.mean(angles)) if angles else float("nan")
            ),
            "pairwise_max_angle_deg": (
                float(np.max(pairwise_angles)) if pairwise_angles else float("nan")
            ),
            "magnitude_cv": (
                float(np.std(magnitudes) / np.mean(magnitudes))
                if float(np.mean(magnitudes)) > EPS
                else float("nan")
            ),
            "magnitude_range_uv_ms": float(np.ptp(magnitudes)),
            "relative_vector_disagreement": (
                float(np.max(np.linalg.norm(stacked - mean_vector, axis=1)) / mean_norm)
                if mean_norm > EPS
                else float("nan")
            ),
        }

    return _memo(ctx, ("transform_ensemble",), build)


def _origin_uncertainty(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """How far the QRS area vector moves when the electrical zero moves.

    TP and PR are the two defensible isoelectric references, and a constant
    offset ``b`` displaces an integral vector by ``b * duration``. The cone
    swept by the area vector across the available origins is therefore the
    part of the answer that the origin choice, not the heart, decides.
    """

    def build() -> Optional[dict[str, float]]:
        assessment = ctx.vcg_origin_assessment
        raw = getattr(ctx, "_record_vcg_xyz", None)
        bounds = ctx.bounds("qrs")
        if raw is None or bounds is None:
            return None
        signal = np.asarray(raw, dtype=float)
        start, end = bounds
        if signal.ndim != 2 or signal.shape[1] != 3 or end + 1 > len(signal):
            return None
        candidates = {
            name: candidate
            for name, candidate in (
                ("tp", assessment.tp),
                ("pr", assessment.pr),
            )
            if candidate is not None
        }
        if len(candidates) < 2:
            return None
        vectors: dict[str, np.ndarray] = {}
        for name, candidate in candidates.items():
            corrected = signal - candidate.vector_millivolts[np.newaxis, :]
            segment = corrected[start : end + 1]
            if not np.all(np.isfinite(segment)):
                continue
            vectors[name] = area_vector_uv_ms(segment, ctx.fs)
        if len(vectors) < 2:
            return None
        stacked = np.asarray(list(vectors.values()), dtype=float)
        magnitudes = np.linalg.norm(stacked, axis=1)
        angle = vector_angle_deg(stacked[0], stacked[1])
        mean_magnitude = float(np.mean(magnitudes))
        offset = (
            candidates["tp"].vector_millivolts - candidates["pr"].vector_millivolts
        )
        return {
            "cone_half_angle_deg": (
                float(angle) / 2.0 if angle is not None else float("nan")
            ),
            "magnitude_spread_uv_ms": float(np.ptp(magnitudes)),
            "relative_magnitude_spread": (
                float(np.ptp(magnitudes) / mean_magnitude)
                if mean_magnitude > EPS
                else float("nan")
            ),
            "tp_pr_offset_uv": float(np.linalg.norm(offset) * 1000.0),
            "origin_confidence": float(
                min(candidate.confidence for candidate in candidates.values())
            ),
        }

    return _memo(ctx, ("origin_uncertainty",), build)


def _peak_area_discordance(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """Where the vector points at its strongest, versus where it points on average.

    The peak spatial vector is an instantaneous direction; the area vector is
    the time integral. In a homogeneous ventricle they nearly coincide. They
    separate when activation and recovery are distributed unevenly in time --
    which is the mechanism, not an artefact, and is invisible to either
    quantity alone.
    """

    def build() -> Optional[dict[str, float]]:
        metrics = ctx.geh_area_metrics()
        qrs = _segment(ctx, "qrs")
        if metrics is None or qrs is None:
            return None
        magnitudes = np.linalg.norm(qrs, axis=1)
        if not len(magnitudes) or not np.isfinite(magnitudes).any():
            return None
        peak_index = int(np.nanargmax(magnitudes))
        peak_vector = qrs[peak_index]
        area_vector = np.asarray(metrics.qrs_area_vector_uv_ms, dtype=float)
        angle = vector_angle_deg(peak_vector, area_vector)
        peak_magnitude_mv = float(np.linalg.norm(peak_vector))
        area_magnitude = float(np.linalg.norm(area_vector))
        # Area is uV*ms, peak is mV: the ratio is only interpretable once the
        # area is expressed as the peak that would produce it over the same
        # duration.
        duration_ms = float(len(qrs) * 1000.0 / ctx.fs)
        equivalent_peak_mv = (
            area_magnitude / 1000.0 / duration_ms if duration_ms > EPS else float("nan")
        )
        result = {
            "peak_area_angle_deg": (
                float(angle) if angle is not None else float("nan")
            ),
            "peak_magnitude_mv": peak_magnitude_mv,
            "area_equivalent_peak_mv": equivalent_peak_mv,
            "peak_to_area_equivalent_ratio": (
                _safe_ratio(peak_magnitude_mv, equivalent_peak_mv) or float("nan")
            ),
            "peak_time_fraction": float(peak_index / max(1, len(qrs) - 1)),
        }
        azimuth_peak = vcg_azimuth_deg(peak_vector)
        azimuth_area = vcg_azimuth_deg(area_vector)
        if azimuth_peak is not None and azimuth_area is not None:
            difference = abs(azimuth_peak - azimuth_area) % 360.0
            result["peak_area_azimuth_difference_deg"] = float(
                min(difference, 360.0 - difference)
            )
        elevation_peak = vcg_elevation_deg(peak_vector)
        elevation_area = vcg_elevation_deg(area_vector)
        if elevation_peak is not None and elevation_area is not None:
            result["peak_area_elevation_difference_deg"] = float(
                abs(elevation_peak - elevation_area)
            )
        return result

    return _memo(ctx, ("peak_area_discordance",), build)


# ======================================================================
# Block 2 -- exact topology, exact recurrence, exact harmonics
# ======================================================================


def _h0_persistence(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    """Exact H0 persistent homology of the Vietoris-Rips filtration.

    Not an approximation: the multiset of finite H0 death times of a Rips
    filtration equals the multiset of minimum-spanning-tree edge weights, so a
    single MST gives the exact barcode. Everything below is read off that
    barcode.
    """

    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 64)
        if points is None or len(points) < 6:
            return None
        distances = distance.squareform(distance.pdist(points))
        n = len(points)
        # Prim's algorithm: the MST edge weights are the H0 death times.
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        best = distances[0].copy()
        best[0] = np.inf
        deaths: list[float] = []
        for _ in range(n - 1):
            nearest = int(np.argmin(best))
            weight = float(best[nearest])
            if not np.isfinite(weight):
                break
            deaths.append(weight)
            in_tree[nearest] = True
            best = np.minimum(best, distances[nearest])
            best[in_tree] = np.inf
        if len(deaths) < 3:
            return None
        bars = np.asarray(sorted(deaths), dtype=float)
        return {
            "h0_total_persistence_mv": float(np.sum(bars)),
            "h0_max_persistence_mv": float(bars[-1]),
            "h0_second_persistence_mv": float(bars[-2]),
            "h0_persistence_entropy": _shannon_entropy(bars),
            "h0_persistence_gap_ratio": (
                _safe_ratio(float(bars[-1]), float(bars[-2])) or float("nan")
            ),
            "h0_mean_persistence_mv": float(np.mean(bars)),
        }

    return _memo(ctx, ("h0_persistence", segment), build)


def _h1_radial_persistence(
    ctx: FeatureContext,
    segment: str,
) -> Optional[dict[str, float]]:
    """Exact persistent homology of the radial function on the closed loop.

    One-dimensional persistent homology of a function on a circle is exactly
    computable by pairing local minima with the saddles that merge them, which
    is what this does. The name states the filtration, because the answer is
    exact for this one and would be an approximation for the Rips filtration
    people usually mean by "H1 of a loop".
    """

    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 96)
        if points is None or len(points) < 12:
            return None
        radius = np.linalg.norm(points - np.mean(points, axis=0), axis=1)
        if not np.isfinite(radius).all() or float(np.ptp(radius)) <= EPS:
            return None
        values = ndimage.gaussian_filter1d(radius, sigma=1.0, mode="wrap")
        n = len(values)
        # Sublevel-set filtration on a circle: sort by value, union adjacent
        # components, and record the merge heights. Elder rule applies.
        order = np.argsort(values, kind="mergesort")
        parent = np.full(n, -1, dtype=int)
        birth = np.zeros(n, dtype=float)
        active = np.zeros(n, dtype=bool)

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        bars: list[float] = []
        for index in order:
            parent[index] = index
            birth[index] = values[index]
            active[index] = True
            for neighbour in ((index - 1) % n, (index + 1) % n):
                if not active[neighbour]:
                    continue
                root_a = find(index)
                root_b = find(neighbour)
                if root_a == root_b:
                    continue
                # Elder rule: the younger component dies at this height.
                if birth[root_a] <= birth[root_b]:
                    older, younger = root_a, root_b
                else:
                    older, younger = root_b, root_a
                bars.append(float(values[index] - birth[younger]))
                parent[younger] = older
        span = float(np.ptp(values))
        # A perfect circle has a constant radius; what survives smoothing is
        # float noise, and calling that a lobe would be inventing structure.
        noise_floor = max(1e-9, 0.01 * span)
        bars = [
            bar for bar in bars if np.isfinite(bar) and bar > noise_floor
        ]
        if len(bars) < 2:
            return None
        lifetimes = np.asarray(sorted(bars), dtype=float)
        return {
            "radial_h1_max_lifetime_mv": float(lifetimes[-1]),
            "radial_h1_total_persistence_mv": float(np.sum(lifetimes)),
            "radial_h1_persistence_entropy": _shannon_entropy(lifetimes),
            "radial_h1_bar_count": float(len(lifetimes)),
            "radial_h1_relative_max_lifetime": (
                _safe_ratio(float(lifetimes[-1]), span) or float("nan")
            ),
            "radial_h1_significant_bar_count": float(
                int(np.sum(lifetimes > 0.1 * span))
            ),
        }

    return _memo(ctx, ("h1_radial", segment), build)


def _minkowski_spectrum(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    """Volume, surface and convexity defect of the loop across dilation scales.

    A loop that is merely large and a loop that is genuinely space-filling
    behave differently as the tube around them thickens; a single-scale hull
    cannot separate them.
    """

    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 96)
        if points is None or len(points) < 8:
            return None
        centered = points - np.mean(points, axis=0)
        scale = float(np.max(np.linalg.norm(centered, axis=1)))
        if scale <= EPS:
            return None
        try:
            hull = ConvexHull(centered)
        except Exception:
            return None
        hull_volume = float(hull.volume)
        hull_surface = float(hull.area)
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        # Minkowski sausage: the curve dilated by r has volume that grows as a
        # polynomial whose coefficients separate length, area and volume.
        radii = np.asarray([0.05, 0.10, 0.20, 0.40], dtype=float) * scale
        occupancies: list[float] = []
        for radius in radii:
            counts = np.floor(centered / max(radius, EPS)).astype(int)
            occupancies.append(float(len(np.unique(counts, axis=0))))
        occupancies_array = np.asarray(occupancies, dtype=float)
        logs = np.log(np.maximum(occupancies_array, 1.0))
        log_radii = np.log(radii / scale)
        slope = float(np.polyfit(log_radii, logs, 1)[0]) if len(radii) > 2 else float("nan")
        return {
            "minkowski_box_dimension": float(-slope),
            "minkowski_hull_volume_mv3": hull_volume,
            "minkowski_hull_surface_mv2": hull_surface,
            "minkowski_isoperimetric_ratio": (
                _safe_ratio(
                    36.0 * np.pi * hull_volume * hull_volume,
                    hull_surface**3,
                )
                or float("nan")
            ),
            "minkowski_length_to_hull_surface": (
                _safe_ratio(length, hull_surface) or float("nan")
            ),
            "minkowski_occupancy_fine": float(occupancies_array[0]),
            "minkowski_occupancy_ratio": (
                _safe_ratio(float(occupancies_array[0]), float(occupancies_array[-1]))
                or float("nan")
            ),
        }

    return _memo(ctx, ("minkowski", segment), build)


def _recurrence_quantification(
    ctx: FeatureContext,
    segment: str,
) -> Optional[dict[str, float]]:
    """Full RQA on the loop's own recurrence matrix.

    Determinism, laminarity, trapping time, maximal diagonal and diagonal
    entropy -- the standard set, from the actual matrix rather than a single
    summary statistic.
    """

    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 96)
        if points is None or len(points) < 20:
            return None
        distances = distance.squareform(distance.pdist(points))
        finite = distances[np.isfinite(distances)]
        if not len(finite):
            return None
        # Fixed radius, not fixed density. A percentile threshold makes the
        # recurrence rate its own definition -- it came out at 0.0906 on every
        # single record -- and with it the maximal diagonal saturates at the
        # resampling length. Scaling the radius to the loop instead lets both
        # carry information about the loop.
        scale = float(np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1)))
        if scale <= EPS:
            return None
        threshold = 0.15 * scale
        recurrence = distances <= threshold
        n = len(points)
        # Theiler window. Consecutive samples of a smooth curve are always
        # within any sensible radius of each other, so the sub-diagonal is
        # fully recurrent on every record and the maximal diagonal saturates
        # at n-1 -- it came out at exactly 95 on all of them. Excluding a band
        # around the main diagonal is the standard remedy: it removes temporal
        # correlation and leaves genuine recurrence.
        theiler = max(2, int(round(0.05 * n)))
        offsets = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        recurrence &= offsets > theiler
        if not recurrence.any():
            return None
        comparable = int(np.sum(offsets > theiler))
        density = float(np.sum(recurrence) / max(1, comparable))

        def line_lengths(matrix: np.ndarray, diagonal: bool) -> np.ndarray:
            lengths: list[int] = []
            if diagonal:
                lines = [
                    np.diagonal(matrix, offset)
                    for offset in range(-n + 2, n - 1)
                ]
            else:
                lines = [matrix[:, column] for column in range(n)]
            for line in lines:
                run = 0
                for value in line:
                    if value:
                        run += 1
                    elif run:
                        lengths.append(run)
                        run = 0
                if run:
                    lengths.append(run)
            return np.asarray(lengths, dtype=float)

        diagonals = line_lengths(recurrence, True)
        verticals = line_lengths(recurrence, False)
        minimum_line = 2.0
        long_diagonals = diagonals[diagonals >= minimum_line]
        long_verticals = verticals[verticals >= minimum_line]
        total_recurrent = float(np.sum(recurrence))
        histogram = (
            np.bincount(long_diagonals.astype(int))[int(minimum_line) :]
            if len(long_diagonals)
            else np.zeros(1)
        )
        return {
            "rqa_recurrence_rate": density,
            "rqa_determinism": (
                float(np.sum(long_diagonals) / total_recurrent)
                if total_recurrent > 0
                else float("nan")
            ),
            "rqa_laminarity": (
                float(np.sum(long_verticals) / total_recurrent)
                if total_recurrent > 0
                else float("nan")
            ),
            "rqa_trapping_time": (
                float(np.mean(long_verticals)) if len(long_verticals) else float("nan")
            ),
            "rqa_max_diagonal_line": (
                float(np.max(long_diagonals)) if len(long_diagonals) else float("nan")
            ),
            "rqa_mean_diagonal_line": (
                float(np.mean(long_diagonals)) if len(long_diagonals) else float("nan")
            ),
            "rqa_diagonal_entropy": _shannon_entropy(
                histogram.astype(float)
            ),
            "rqa_divergence": (
                _safe_ratio(1.0, float(np.max(long_diagonals)))
                if len(long_diagonals)
                else float("nan")
            ),
        }

    return _memo(ctx, ("rqa", segment), build)


def _spherical_harmonics(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    """Real spherical-harmonic decomposition of the direction curve.

    Coefficients ``a_lm`` and per-degree power ``C_l`` of the loop's direction
    density on the unit sphere.

    A closed loop traverses very nearly antipodal directions, so its direction
    density is an *orientation* density and the odd degrees cancel by symmetry
    -- measured at 1e-33 on a synthetic ellipse. Degree 1 is therefore not the
    axis here and is useless as a denominator; the axis of an orientation
    distribution lives in degree 2, and degree 0 is the isotropic part that
    normalises it. Residual odd-degree power is not noise either: it measures
    how far the loop fails to close, which is exactly what a reader means by a
    loop that does not come back.
    """

    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 96)
        if points is None or len(points) < 12:
            return None
        centered = points - np.mean(points, axis=0)
        norms = np.linalg.norm(centered, axis=1)
        usable = norms > EPS
        if int(np.sum(usable)) < 10:
            return None
        directions = centered[usable] / norms[usable][:, None]
        weights = norms[usable]
        weights = weights / float(np.sum(weights))
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        # Real spherical harmonics up to degree 3, in Cartesian form: no
        # angle conventions to get wrong, and exactly orthonormal.
        harmonics: dict[int, list[np.ndarray]] = {
            0: [np.full(len(x), 0.5 * np.sqrt(1.0 / np.pi))],
            1: [
                np.sqrt(3.0 / (4.0 * np.pi)) * y,
                np.sqrt(3.0 / (4.0 * np.pi)) * z,
                np.sqrt(3.0 / (4.0 * np.pi)) * x,
            ],
            2: [
                0.5 * np.sqrt(15.0 / np.pi) * x * y,
                0.5 * np.sqrt(15.0 / np.pi) * y * z,
                0.25 * np.sqrt(5.0 / np.pi) * (3.0 * z * z - 1.0),
                0.5 * np.sqrt(15.0 / np.pi) * x * z,
                0.25 * np.sqrt(15.0 / np.pi) * (x * x - y * y),
            ],
            3: [
                0.25 * np.sqrt(35.0 / (2.0 * np.pi)) * y * (3.0 * x * x - y * y),
                0.5 * np.sqrt(105.0 / np.pi) * x * y * z,
                0.25 * np.sqrt(21.0 / (2.0 * np.pi)) * y * (5.0 * z * z - 1.0),
                0.25 * np.sqrt(7.0 / np.pi) * z * (5.0 * z * z - 3.0),
                0.25 * np.sqrt(21.0 / (2.0 * np.pi)) * x * (5.0 * z * z - 1.0),
                0.25 * np.sqrt(105.0 / np.pi) * (x * x - y * y) * z,
                0.25 * np.sqrt(35.0 / (2.0 * np.pi)) * x * (x * x - 3.0 * y * y),
            ],
        }
        power: dict[int, float] = {}
        for degree, basis in harmonics.items():
            coefficients = np.asarray(
                [float(np.sum(weights * function)) for function in basis],
                dtype=float,
            )
            power[degree] = float(np.sum(coefficients**2))
        total = float(sum(power.values()))
        if total <= EPS:
            return None
        result = {
            f"sh_power_degree_{degree}": value for degree, value in power.items()
        }
        result["sh_total_power"] = total
        result["sh_axial_fraction"] = float(power[2] / total)
        result["sh_isotropic_fraction"] = float(power[0] / total)
        result["sh_higher_order_fraction"] = float(
            (total - power[0] - power[2]) / total
        )
        result["sh_axial_to_isotropic"] = (
            _safe_ratio(power[2], power[0]) or float("nan")
        )
        result["sh_degree3_to_degree2"] = (
            _safe_ratio(power[3], power[2]) or float("nan")
        )
        # Odd degrees vanish for a perfectly antipodal (closed) loop, so what
        # survives measures non-closure rather than orientation.
        result["sh_odd_degree_fraction"] = float(
            (power[1] + power[3]) / total
        )
        result["sh_spectral_entropy"] = _shannon_entropy(
            np.asarray([power[degree] for degree in sorted(power)], dtype=float)
        )
        return result

    return _memo(ctx, ("spherical_harmonics", segment), build)


def _spherical_holonomy(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    """How far the loop's direction curve departs from a flat one.

    The holonomy of a closed path on the sphere equals the enclosed solid
    angle, and for the direction curve of a planar loop that is exactly a
    great circle: 2*pi steradian, half the sphere. Measured across planar,
    twisted, figure-of-eight and spiralling loops the enclosed fraction stays
    within 0.4964-0.5000 -- it is a topological invariant of winding once, not
    a description of shape, and reporting it directly would add near-constant
    columns that can only dilute the multiplicity budget.

    What does carry shape is the *departure* from that planar value. It is
    zero for a flat loop by construction and grows with genuine
    three-dimensional twist, which no single projection can show.
    """

    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 96)
        if points is None or len(points) < 12:
            return None
        centered = points - np.mean(points, axis=0)
        norms = np.linalg.norm(centered, axis=1)
        usable = norms > EPS
        if int(np.sum(usable)) < 10:
            return None
        directions = centered[usable] / norms[usable][:, None]
        closed = np.vstack([directions, directions[:1]])
        # Signed spherical area by the spherical-excess sum over the fan of
        # triangles from the first vertex.
        anchor = closed[0]
        total_area = 0.0
        # Fan triangles (v0, v_i, v_i+1) for i = 1..n-2. Running to n-1 would
        # append the degenerate triangle (v0, v_n-1, v0).
        for index in range(1, len(closed) - 2):
            first, second = closed[index], closed[index + 1]
            numerator = float(np.dot(anchor, np.cross(first, second)))
            denominator = (
                1.0
                + float(np.dot(anchor, first))
                + float(np.dot(first, second))
                + float(np.dot(second, anchor))
            )
            total_area += 2.0 * float(np.arctan2(numerator, denominator))
        # Holonomy angle is the enclosed solid angle, wrapped to (-2pi, 2pi].
        holonomy = float(total_area)
        # A planar loop encloses exactly a half sphere; everything else is the
        # part that depends on the loop rather than on its winding number.
        departure = abs(holonomy) - 2.0 * np.pi
        return {
            "holonomy_solid_angle_sr": holonomy,
            "holonomy_planar_departure_sr": float(departure),
            "holonomy_absolute_planar_departure_sr": float(abs(departure)),
            "holonomy_planar_departure_deg": float(np.degrees(departure)),
            "holonomy_orientation_is_clockwise": float(holonomy < 0),
        }

    return _memo(ctx, ("holonomy", segment), build)


# ======================================================================
# Block 3 -- what a reader sees
# ======================================================================


def _activation_energy_quantiles(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """When the QRS spends its energy.

    Cumulative ``integral |r'(t)|^2`` reaches its quantiles early in a
    normally activating ventricle and late when terminal forces are delayed.
    Visible as a loop that crowds its points at the end instead of the middle.
    """

    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, "qrs")
        if points is None or len(points) < 8:
            return None
        velocity = np.diff(points, axis=0)
        energy = np.sum(velocity * velocity, axis=1)
        total = float(np.sum(energy))
        if total <= EPS:
            return None
        cumulative = np.cumsum(energy) / total
        duration_ms = float(len(points) * 1000.0 / ctx.fs)
        result: dict[str, float] = {}
        for quantile in (10, 25, 50, 75, 90):
            index = int(np.searchsorted(cumulative, quantile / 100.0))
            fraction = float(min(index, len(cumulative) - 1) / max(1, len(cumulative) - 1))
            result[f"activation_energy_q{quantile}_fraction"] = fraction
            result[f"activation_energy_q{quantile}_ms"] = fraction * duration_ms
        result["activation_energy_iqr_fraction"] = float(
            result["activation_energy_q75_fraction"]
            - result["activation_energy_q25_fraction"]
        )
        result["activation_energy_terminal_share"] = float(
            1.0 - cumulative[int(0.75 * (len(cumulative) - 1))]
        )
        result["activation_energy_skew"] = float(
            result["activation_energy_q50_fraction"] - 0.5
        )
        return result

    return _memo(ctx, ("activation_energy",), build)


def _septal_to_basal(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """Time between the early (septal) and late (basal) centres of QRS force.

    The early QRS is septal and points rightward-anterior; the late QRS is
    basal. The separation between the energy-weighted centroids of the first
    and last thirds is a direction-and-time statement about activation
    sequence, and is visible as the loop's start and end pointing apart.
    """

    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, "qrs")
        if points is None or len(points) < 12:
            return None
        magnitudes = np.linalg.norm(points, axis=1)
        if float(np.sum(magnitudes)) <= EPS:
            return None
        third = max(2, len(points) // 3)
        early = points[:third]
        late = points[-third:]
        early_weights = np.linalg.norm(early, axis=1)
        late_weights = np.linalg.norm(late, axis=1)
        if float(np.sum(early_weights)) <= EPS or float(np.sum(late_weights)) <= EPS:
            return None
        early_centroid = np.average(early, axis=0, weights=early_weights)
        late_centroid = np.average(late, axis=0, weights=late_weights)
        early_peak = int(np.argmax(early_weights))
        late_peak = int(len(points) - third + int(np.argmax(late_weights)))
        angle = vector_angle_deg(early_centroid, late_centroid)
        sample_ms = 1000.0 / ctx.fs
        return {
            "septal_to_basal_interval_ms": float((late_peak - early_peak) * sample_ms),
            "septal_to_basal_angle_deg": (
                float(angle) if angle is not None else float("nan")
            ),
            "septal_to_basal_distance_mv": float(
                np.linalg.norm(late_centroid - early_centroid)
            ),
            "septal_basal_magnitude_ratio": (
                _safe_ratio(
                    float(np.linalg.norm(late_centroid)),
                    float(np.linalg.norm(early_centroid)),
                )
                or float("nan")
            ),
            "basal_force_share": float(
                np.sum(late_weights) / float(np.sum(magnitudes))
            ),
        }

    return _memo(ctx, ("septal_to_basal",), build)


def _voltage_complexity(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """A small loop that is nevertheless complicated.

    Infiltrative disease lowers voltage while leaving -- or creating --
    fractionated, tortuous activation. Amplitude alone calls that normal-ish
    low voltage; complexity alone calls it noise. The mismatch is the
    hypothesis, and it is exactly what a reader means by "small but scribbled".
    """

    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, "qrs")
        if points is None or len(points) < 12:
            return None
        magnitudes = np.linalg.norm(points, axis=1)
        peak = float(np.max(magnitudes))
        if peak <= EPS:
            return None
        path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        span = float(
            np.max(distance.pdist(points)) if len(points) > 1 else 0.0
        )
        tortuosity = _safe_ratio(path_length, span)
        smoothed = ndimage.gaussian_filter1d(points, sigma=2.0, axis=0, mode="nearest")
        residual = float(
            np.mean(np.linalg.norm(points - smoothed, axis=1))
        )
        spectrum = np.abs(np.fft.rfft(magnitudes - float(np.mean(magnitudes))))
        if len(spectrum) < 4:
            return None
        high_band = float(np.sum(spectrum[len(spectrum) // 2 :] ** 2))
        total_band = float(np.sum(spectrum**2))
        return {
            "voltage_complexity_peak_mv": peak,
            "voltage_complexity_tortuosity": (
                float(tortuosity) if tortuosity is not None else float("nan")
            ),
            "voltage_complexity_mismatch_index": (
                float(tortuosity / peak)
                if tortuosity is not None and peak > EPS
                else float("nan")
            ),
            "voltage_complexity_roughness_fraction": float(residual / peak),
            "voltage_complexity_high_band_fraction": (
                float(high_band / total_band) if total_band > EPS else float("nan")
            ),
            "voltage_complexity_per_mv_path": (
                _safe_ratio(path_length, peak) or float("nan")
            ),
        }

    return _memo(ctx, ("voltage_complexity",), build)


def _accepted_beats(ctx: FeatureContext) -> list[object]:
    beats = getattr(ctx.delineation, "accepted_beats", None)
    if beats is None:
        beats = getattr(ctx.delineation, "beats", ())
    return [beat for beat in (beats or ())]


def _beat_loops(ctx: FeatureContext, n: int = 48) -> Optional[list[np.ndarray]]:
    """Time-normalised QRS loop of every accepted beat."""

    def build() -> Optional[list[np.ndarray]]:
        source = getattr(ctx, "_record_vcg_xyz", None)
        if source is None:
            return None
        assessment = ctx.vcg_origin_assessment
        signal = assessment.corrected(np.asarray(source, dtype=float))
        if signal is None:
            return None
        loops: list[np.ndarray] = []
        for beat in _accepted_beats(ctx):
            onset = getattr(beat, "r_onset", None)
            offset = getattr(beat, "r_offset", None)
            if onset is None or offset is None:
                continue
            start = getattr(onset, "sample", None)
            end = getattr(offset, "sample", None)
            if start is None or end is None or end - start < 6:
                continue
            if end + 1 > len(signal):
                continue
            piece = signal[int(start) : int(end) + 1]
            if not np.all(np.isfinite(piece)):
                continue
            loops.append(np.asarray(_resample_curve(piece, n), dtype=float))
        return loops if len(loops) >= 3 else None

    return _memo(ctx, ("beat_loops", n), build)


def _beat_covariance_tube(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """How thick the bundle of beat loops is, point by point.

    Superimpose every beat's time-normalised loop and the result is a tube.
    A thin tube is a reproducible activation sequence; a thick or locally
    bulging one is beat-to-beat instability, and a reader sees it directly as
    a fuzzy or split loop.
    """

    def build() -> Optional[dict[str, float]]:
        loops = _beat_loops(ctx)
        if loops is None:
            return None
        stack = np.stack(loops, axis=0)
        mean_loop = np.mean(stack, axis=0)
        deviations = stack - mean_loop[None, :, :]
        radial = np.linalg.norm(deviations, axis=2)
        pointwise = np.mean(radial, axis=0)
        scale = float(np.max(np.linalg.norm(mean_loop, axis=1)))
        if scale <= EPS:
            return None
        # Local anisotropy: is the spread along the loop, or across it?
        volumes: list[float] = []
        for index in range(stack.shape[1]):
            cloud = deviations[:, index, :]
            covariance = np.cov(cloud, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(covariance)
            eigenvalues = np.clip(eigenvalues, 0.0, None)
            volumes.append(float(np.sqrt(np.prod(eigenvalues + EPS))))
        volumes_array = np.asarray(volumes, dtype=float)
        return {
            "beat_tube_mean_radius_mv": float(np.mean(pointwise)),
            "beat_tube_max_radius_mv": float(np.max(pointwise)),
            "beat_tube_relative_thickness": float(np.mean(pointwise) / scale),
            "beat_tube_bulge_ratio": (
                _safe_ratio(float(np.max(pointwise)), float(np.mean(pointwise)))
                or float("nan")
            ),
            "beat_tube_thickness_entropy": _shannon_entropy(pointwise),
            "beat_tube_ellipsoid_volume_mv3": float(np.mean(volumes_array)),
            "beat_tube_late_to_early_ratio": (
                _safe_ratio(
                    float(np.mean(pointwise[len(pointwise) // 2 :])),
                    float(np.mean(pointwise[: len(pointwise) // 2])),
                )
                or float("nan")
            ),
            "beat_tube_n_beats": float(len(loops)),
        }

    return _memo(ctx, ("beat_tube",), build)


def _respiratory_modulation(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """Beat-to-beat swing of the loop's plane, centroid and size.

    The heart moves with the chest, so the loop rocks periodically. Two things
    make this worth measuring: the swing is a confounder for every geometric
    marker, and its absence in a patient who is breathing is itself a signal.
    Ten seconds hold only a few breaths, so this abstains below eight beats.
    """

    def build() -> Optional[dict[str, float]]:
        loops = _beat_loops(ctx)
        if loops is None or len(loops) < 8:
            return None
        normals: list[np.ndarray] = []
        centroids: list[np.ndarray] = []
        magnitudes: list[float] = []
        for loop in loops:
            centered = loop - np.mean(loop, axis=0)
            if float(np.linalg.norm(centered)) <= EPS:
                continue
            _, singular, vh = np.linalg.svd(centered, full_matrices=False)
            if len(singular) < 3:
                continue
            normal = np.asarray(vh[-1], dtype=float)
            pivot = int(np.argmax(np.abs(normal)))
            if normal[pivot] < 0:
                normal = -normal
            normals.append(normal)
            centroids.append(np.mean(loop, axis=0))
            magnitudes.append(float(np.max(np.linalg.norm(loop, axis=1))))
        if len(normals) < 8:
            return None
        normals_array = np.asarray(normals)
        centroids_array = np.asarray(centroids)
        magnitudes_array = np.asarray(magnitudes)
        mean_normal = np.mean(normals_array, axis=0)
        mean_normal /= max(float(np.linalg.norm(mean_normal)), EPS)
        angles = np.asarray(
            [
                vector_angle_deg(normal, mean_normal) or 0.0
                for normal in normals_array
            ],
            dtype=float,
        )
        result = {
            "respiratory_plane_swing_deg": float(np.ptp(angles)),
            "respiratory_plane_swing_sd_deg": float(np.std(angles)),
            "respiratory_centroid_swing_mv": float(
                np.max(np.linalg.norm(centroids_array - np.mean(centroids_array, axis=0), axis=1))
            ),
            "respiratory_magnitude_cv": (
                float(np.std(magnitudes_array) / np.mean(magnitudes_array))
                if float(np.mean(magnitudes_array)) > EPS
                else float("nan")
            ),
            "respiratory_n_beats": float(len(normals)),
        }
        # Periodicity: a respiratory swing is oscillatory, drift is not.
        if len(angles) >= 8:
            detrended = angles - float(np.mean(angles))
            spectrum = np.abs(np.fft.rfft(detrended)) ** 2
            if len(spectrum) > 2 and float(np.sum(spectrum[1:])) > EPS:
                dominant = int(np.argmax(spectrum[1:])) + 1
                result["respiratory_periodicity_fraction"] = float(
                    spectrum[dominant] / float(np.sum(spectrum[1:]))
                )
                result["respiratory_cycles_per_record"] = float(dominant)
        return result

    return _memo(ctx, ("respiratory",), build)


def _cross_axis_coupling(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """Phase coupling and non-linear mixing between the three spatial axes.

    Ordinary activation drives X, Y and Z as filtered copies of one process,
    so their coherence is high and their phase relations are stable.
    Fragmented conduction breaks that: the axes acquire independent
    high-frequency content, and bicoherence detects the quadratic mixing that
    ordinary coherence misses.
    """

    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, "qrst")
        if points is None or len(points) < 48:
            return None
        result: dict[str, float] = {}
        pairs = (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2))
        coherences: list[float] = []
        for label, first, second in pairs:
            a = points[:, first] - float(np.mean(points[:, first]))
            b = points[:, second] - float(np.mean(points[:, second]))
            if float(np.std(a)) <= EPS or float(np.std(b)) <= EPS:
                continue
            segment_length = min(32, len(a))
            try:
                _, coherence = sp_signal.coherence(
                    a,
                    b,
                    fs=float(ctx.fs),
                    nperseg=segment_length,
                )
            except Exception:
                continue
            value = float(np.mean(coherence[np.isfinite(coherence)]))
            result[f"cross_axis_coherence_{label}"] = value
            coherences.append(value)
        if not coherences:
            return None
        result["cross_axis_coherence_mean"] = float(np.mean(coherences))
        result["cross_axis_coherence_min"] = float(np.min(coherences))
        # Bicoherence of the spatial magnitude: quadratic phase coupling
        # between f1, f2 and f1+f2 that linear coherence cannot see.
        magnitude = np.linalg.norm(points, axis=1)
        magnitude = magnitude - float(np.mean(magnitude))
        spectrum = np.fft.rfft(magnitude)
        n_bins = min(12, len(spectrum) // 2)
        if n_bins >= 3:
            # The triple products must be summed as complex numbers so that
            # unlocked phases cancel. Taking the magnitude first makes the
            # ratio identically one -- measured at 1.0 +- 9e-17 before the fix.
            numerator = 0.0 + 0.0j
            denominator = 0.0
            for first in range(1, n_bins):
                for second in range(first, n_bins):
                    if first + second >= len(spectrum):
                        continue
                    numerator += (
                        spectrum[first]
                        * spectrum[second]
                        * np.conjugate(spectrum[first + second])
                    )
                    denominator += abs(spectrum[first]) * abs(
                        spectrum[second]
                    ) * abs(spectrum[first + second])
            if denominator > EPS:
                result["cross_axis_bicoherence"] = float(
                    abs(numerator) / denominator
                )
        return result

    return _memo(ctx, ("cross_axis",), build)


def _segment_distance_2d(
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> float:
    """Smallest distance between two 2-D line segments."""

    def point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        edge = end - start
        length2 = float(np.dot(edge, edge))
        if length2 <= EPS:
            return float(np.linalg.norm(point - start))
        t = float(np.clip(np.dot(point - start, edge) / length2, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + t * edge)))

    # Proper crossing gives distance zero; the 2-D cross product is written
    # out because NumPy 2.0 deprecated np.cross on two-dimensional vectors.
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    d1 = cross(a2 - a1, b1 - a1)
    d2 = cross(a2 - a1, b2 - a1)
    d3 = cross(b2 - b1, a1 - b1)
    d4 = cross(b2 - b1, a2 - b1)
    if d1 * d2 < 0.0 and d3 * d4 < 0.0:
        return 0.0
    return min(
        point_to_segment(b1, a1, a2),
        point_to_segment(b2, a1, a2),
        point_to_segment(a1, b1, b2),
        point_to_segment(a2, b1, b2),
    )


def _count_self_contacts(points: np.ndarray, *, tolerance: float) -> float:
    """How often the curve comes back to within ``tolerance`` of itself.

    Non-adjacent segments only: neighbours always touch at their shared
    vertex, which is not a self-intersection.
    """

    n = len(points)
    if n < 6 or tolerance <= 0.0:
        return 0.0
    starts = np.asarray(points[:-1], dtype=float)
    ends = np.asarray(points[1:], dtype=float)

    # All pairwise segment distances are only 95 x 95 for the fixed
    # 96-sample visual loop.  Broadcasting the geometry in NumPy avoids about
    # 17,000 Python function calls per projection while remaining far below a
    # megabyte of working memory.  The four point-to-segment distances and the
    # strict-crossing rule are the same operations as
    # ``_segment_distance_2d`` above; this is an exact bulk implementation,
    # not a different contact definition.
    first_start = starts[:, None, :]
    first_end = ends[:, None, :]
    second_start = starts[None, :, :]
    second_end = ends[None, :, :]

    def point_to_segment_squared(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        edge = end - start
        offset = point - start
        length_squared = np.sum(edge * edge, axis=-1)
        numerator = np.sum(offset * edge, axis=-1)
        projection = np.zeros_like(numerator, dtype=float)
        np.divide(
            numerator,
            length_squared,
            out=projection,
            where=length_squared > EPS,
        )
        projection = np.clip(projection, 0.0, 1.0)
        residual = offset - projection[..., None] * edge
        squared = np.sum(residual * residual, axis=-1)
        degenerate = length_squared <= EPS
        if np.any(degenerate):
            squared = np.where(
                degenerate,
                np.sum(offset * offset, axis=-1),
                squared,
            )
        return squared

    distance_squared = np.minimum.reduce(
        (
            point_to_segment_squared(
                second_start, first_start, first_end
            ),
            point_to_segment_squared(second_end, first_start, first_end),
            point_to_segment_squared(first_start, second_start, second_end),
            point_to_segment_squared(first_end, second_start, second_end),
        )
    )

    def cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    first_edge = first_end - first_start
    second_edge = second_end - second_start
    d1 = cross(first_edge, second_start - first_start)
    d2 = cross(first_edge, second_end - first_start)
    d3 = cross(second_edge, first_start - second_start)
    d4 = cross(second_edge, first_end - second_start)
    proper_crossing = (d1 * d2 < 0.0) & (d3 * d4 < 0.0)
    distance_squared[proper_crossing] = 0.0

    non_adjacent = np.triu(
        np.ones(distance_squared.shape, dtype=bool),
        k=3,
    )
    contact_first = np.any(
        non_adjacent
        & (np.sqrt(np.maximum(distance_squared, 0.0)) <= tolerance),
        axis=1,
    )
    contact_indices = np.flatnonzero(contact_first)
    if len(contact_indices) == 0:
        return 0.0
    # Preserve the legacy event de-duplication exactly: every witnessing
    # first segment updates ``last_contact``; only a gap greater than two
    # begins a new event.
    return float(1 + np.count_nonzero(np.diff(contact_indices) > 2))


def _visual_loop_reading(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    """The things a reader names out loud when looking at a loop.

    Openness (does it close?), pinching (is there a waist?), doubling back,
    late scribble, and where the mass sits relative to the direction of travel.
    Each is a sentence a clinician would say; each is a number here.
    """

    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 96)
        if points is None or len(points) < 16:
            return None
        scale = float(np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1)))
        if scale <= EPS:
            return None
        start, end = points[0], points[-1]
        gap = float(np.linalg.norm(end - start))
        path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

        # Waist: the smallest cross-loop width along the principal axis.
        centered = points - np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        along = centered @ vh[0]
        across = centered @ vh[1]
        bins = np.linspace(float(np.min(along)), float(np.max(along)), 9)
        widths: list[float] = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            inside = (along >= lower) & (along <= upper)
            if int(np.sum(inside)) >= 2:
                widths.append(float(np.ptp(across[inside])))
        widths_array = np.asarray(widths, dtype=float) if widths else np.zeros(1)

        # Doubling back: how often the direction of travel reverses.
        steps = np.diff(points, axis=0)
        norms = np.linalg.norm(steps, axis=1)
        usable = norms > EPS
        reversals = 0.0
        if int(np.sum(usable)) > 2:
            unit = steps[usable] / norms[usable][:, None]
            dots = np.sum(unit[:-1] * unit[1:], axis=1)
            reversals = float(np.sum(dots < 0.0) / max(1, len(dots)))

        # Late scribble: roughness of the last third against the first third.
        smoothed = ndimage.gaussian_filter1d(points, sigma=2.0, axis=0, mode="nearest")
        roughness = np.linalg.norm(points - smoothed, axis=1)
        third = max(2, len(roughness) // 3)
        early_roughness = float(np.mean(roughness[:third]))
        late_roughness = float(np.mean(roughness[-third:]))

        result = {
            "visual_open_gap_fraction": float(gap / scale),
            "visual_open_gap_mv": gap,
            "visual_waist_fraction": (
                float(np.min(widths_array) / max(float(np.max(widths_array)), EPS))
                if len(widths_array) > 1
                else float("nan")
            ),
            "visual_waist_width_mv": float(np.min(widths_array)),
            "visual_reversal_fraction": reversals,
            "visual_late_roughness_ratio": (
                _safe_ratio(late_roughness, early_roughness) or float("nan")
            ),
            "visual_roughness_fraction": float(np.mean(roughness) / scale),
            "visual_path_to_span_ratio": (
                _safe_ratio(path_length, 2.0 * scale) or float("nan")
            ),
            "visual_elongation": (
                _safe_ratio(float(np.ptp(along)), float(np.ptp(across))) or float("nan")
            ),
        }
        # Where the mass sits: leading, centred, or trailing.
        magnitudes = np.linalg.norm(points, axis=1)
        if float(np.sum(magnitudes)) > EPS:
            positions = np.linspace(0.0, 1.0, len(magnitudes))
            result["visual_mass_centroid_fraction"] = float(
                np.average(positions, weights=magnitudes)
            )
        return result

    return _memo(ctx, ("visual_reading", segment), build)


def _plane_projection_reading(
    ctx: FeatureContext,
    segment: str,
    plane: str,
) -> Optional[dict[str, float]]:
    """What each of the three standard projections shows on its own.

    A reader looks at frontal, horizontal and sagittal separately, and a
    finding that appears in only one of them is a different finding from one
    that appears in all three.
    """

    def build() -> Optional[dict[str, float]]:
        points = _resampled(ctx, segment, 96)
        axes = PLANES.get(plane)
        if points is None or axes is None:
            return None
        flat = points[:, list(axes)]
        centered = flat - np.mean(flat, axis=0)
        scale = float(np.max(np.linalg.norm(centered, axis=1)))
        if scale <= EPS:
            return None
        # Signed area (shoelace) gives both size and rotation direction.
        x, y = centered[:, 0], centered[:, 1]
        signed_area = 0.5 * float(
            np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        )
        try:
            hull = ConvexHull(centered)
            hull_area = float(hull.volume)
            hull_perimeter = float(hull.area)
        except Exception:
            return None
        perimeter = float(
            np.sum(np.linalg.norm(np.diff(np.vstack([centered, centered[:1]]), axis=0), axis=1))
        )
        # Self-intersections: the figure-of-eight a reader notices at once.
        # Orientation signs alone miss a curve that crosses exactly on a
        # sample -- which is precisely what a symmetric figure-of-eight does --
        # so this measures how close two non-adjacent segments come instead.
        crossings = _count_self_contacts(centered, tolerance=0.02 * scale)
        return {
            "plane_signed_area_mv2": signed_area,
            "plane_rotation_is_counterclockwise": float(signed_area > 0),
            "plane_area_to_hull_ratio": (
                _safe_ratio(abs(signed_area), hull_area) or float("nan")
            ),
            "plane_perimeter_to_hull_ratio": (
                _safe_ratio(perimeter, hull_perimeter) or float("nan")
            ),
            "plane_self_intersections": float(crossings),
            "plane_circularity": (
                _safe_ratio(4.0 * np.pi * abs(signed_area), perimeter**2)
                or float("nan")
            ),
            "plane_bite_depth_fraction": float(
                1.0 - (abs(signed_area) / hull_area) if hull_area > EPS else np.nan
            ),
        }

    return _memo(ctx, ("plane_reading", segment, plane), build)


def _qrs_t_visual_relation(ctx: FeatureContext) -> Optional[dict[str, float]]:
    """How the QRS loop and the T loop sit relative to each other.

    Opposition, unusual parallelism, size mismatch and whether the T loop
    starts where the QRS ended -- the relations a reader judges at a glance
    and that no single-loop marker captures.
    """

    def build() -> Optional[dict[str, float]]:
        qrs = _resampled(ctx, "qrs", 64)
        t_loop = _resampled(ctx, "t", 64)
        if qrs is None or t_loop is None:
            return None
        qrs_scale = float(np.max(np.linalg.norm(qrs, axis=1)))
        t_scale = float(np.max(np.linalg.norm(t_loop, axis=1)))
        if qrs_scale <= EPS or t_scale <= EPS:
            return None
        qrs_peak = qrs[int(np.argmax(np.linalg.norm(qrs, axis=1)))]
        t_peak = t_loop[int(np.argmax(np.linalg.norm(t_loop, axis=1)))]
        angle = vector_angle_deg(qrs_peak, t_peak)
        junction_gap = float(np.linalg.norm(t_loop[0] - qrs[-1]))
        # Do the two loops occupy the same space, or separate regions?
        cross_distances = distance.cdist(qrs, t_loop)
        return {
            "qrs_t_peak_angle_deg": float(angle) if angle is not None else float("nan"),
            "qrs_t_size_ratio": _safe_ratio(t_scale, qrs_scale) or float("nan"),
            "qrs_t_junction_gap_fraction": float(junction_gap / qrs_scale),
            "qrs_t_minimum_separation_mv": float(np.min(cross_distances)),
            "qrs_t_minimum_separation_fraction": float(
                np.min(cross_distances) / qrs_scale
            ),
            "qrs_t_centroid_distance_mv": float(
                np.linalg.norm(np.mean(t_loop, axis=0) - np.mean(qrs, axis=0))
            ),
            "qrs_t_unusual_parallelism": float(
                1.0 if angle is not None and angle < 20.0 else 0.0
            ),
        }

    return _memo(ctx, ("qrs_t_visual",), build)


# ======================================================================
# Spec construction
# ======================================================================


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
    visual_assessment: str = "mathematical_only",
) -> DiscoveryV4Spec:
    return DiscoveryV4Spec(
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


def _dict_specs(
    primitive: Callable[[FeatureContext], Optional[dict[str, float]]],
    family: str,
    reason: str,
    evidence_level: str,
    evidence_note: str,
    references: tuple[str, ...],
    entries: Sequence[tuple[str, str, str, str]],
    dependencies: tuple[str, ...],
    visual_assessment: str = "mathematical_only",
) -> list[DiscoveryV4Spec]:
    """Build one spec per key of a shared primitive.

    The primitive is memoized, so a family of twelve markers costs one
    computation rather than twelve.
    """

    specs: list[DiscoveryV4Spec] = []
    for name, key, unit, description in entries:
        specs.append(
            _spec(
                name=name,
                family=family,
                description=description,
                formula=f"{key} of {primitive.__name__.strip('_')}",
                unit=unit,
                dependencies=dependencies,
                evidence_level=evidence_level,
                evidence_note=evidence_note,
                references=references,
                calculator=_value_from_dict(primitive, key, reason),
                visual_assessment=visual_assessment,
            )
        )
    return specs


def _build_specs() -> tuple[DiscoveryV4Spec, ...]:
    specs: list[DiscoveryV4Spec] = []

    # --- Block 1: method uncertainty -----------------------------------
    specs.extend(
        _dict_specs(
            _transform_ensemble,
            "Discovery-v4 transform-ensemble uncertainty",
            "fewer than two usable VCG transforms",
            "C",
            "Kors and inverse Dower are both accepted syntheses with documented "
            "disagreement; the spread is a property of the reconstruction.",
            (REF_KORS, REF_DOWER),
            [
                ("transform_ensemble_angular_spread_deg", "angular_spread_deg", "degrees",
                 "Largest angle between a transform's QRS area vector and the ensemble mean."),
                ("transform_ensemble_mean_angular_deviation_deg", "mean_angular_deviation_deg", "degrees",
                 "Mean angular deviation of the QRS area vector across transforms."),
                ("transform_ensemble_pairwise_max_angle_deg", "pairwise_max_angle_deg", "degrees",
                 "Largest angle between any two transforms' QRS area vectors."),
                ("transform_ensemble_magnitude_cv", "magnitude_cv", "dimensionless",
                 "Coefficient of variation of QRS area magnitude across transforms."),
                ("transform_ensemble_magnitude_range_uv_ms", "magnitude_range_uv_ms", "uV*ms",
                 "Range of QRS area magnitude across transforms."),
                ("transform_ensemble_relative_disagreement", "relative_vector_disagreement", "dimensionless",
                 "Largest transform-to-mean vector distance relative to the mean magnitude."),
            ],
            ("record_ecg_12lead", "qrs_bounds"),
        )
    )
    specs.extend(
        _dict_specs(
            _origin_uncertainty,
            "Discovery-v4 origin-uncertainty cone",
            "TP and PR origins not both available",
            "B",
            "A physiologically defined isoelectric origin measurably changed GEH "
            "marker performance in ARIC; the residual choice is quantified here.",
            (REF_ARIC_ORIGIN, REF_GEH),
            [
                ("origin_cone_half_angle_deg", "cone_half_angle_deg", "degrees",
                 "Half-angle of the cone swept by the QRS area vector between TP and PR origins."),
                ("origin_cone_magnitude_spread_uv_ms", "magnitude_spread_uv_ms", "uV*ms",
                 "Range of QRS area magnitude across the two origin definitions."),
                ("origin_cone_relative_magnitude_spread", "relative_magnitude_spread", "dimensionless",
                 "Origin-induced magnitude spread relative to the mean magnitude."),
                ("origin_cone_tp_pr_offset_uv", "tp_pr_offset_uv", "uV",
                 "Distance between the TP and PR origin vectors."),
                ("origin_cone_confidence", "origin_confidence", "dimensionless",
                 "Lowest confidence among the origin candidates entering the cone."),
            ],
            ("vcg_origin_assessment",),
        )
    )
    specs.extend(
        _dict_specs(
            _peak_area_discordance,
            "Discovery-v4 QRS peak/area discordance",
            "QRS segment or GEH area metrics unavailable",
            "C",
            "Peak and area ventricular gradients are established separately; their "
            "divergence is the prospective construct.",
            (REF_GEH,),
            [
                ("peak_area_qrs_angle_deg", "peak_area_angle_deg", "degrees",
                 "Angle between the peak spatial QRS vector and the QRS area vector."),
                ("peak_area_qrs_azimuth_difference_deg", "peak_area_azimuth_difference_deg", "degrees",
                 "Azimuth difference between peak and area QRS vectors."),
                ("peak_area_qrs_elevation_difference_deg", "peak_area_elevation_difference_deg", "degrees",
                 "Elevation difference between peak and area QRS vectors."),
                ("peak_area_qrs_magnitude_ratio", "peak_to_area_equivalent_ratio", "dimensionless",
                 "Peak magnitude divided by the duration-equivalent area magnitude."),
                ("peak_area_qrs_peak_time_fraction", "peak_time_fraction", "dimensionless",
                 "Where inside the QRS the spatial peak occurs."),
                ("peak_area_qrs_equivalent_peak_mv", "area_equivalent_peak_mv", "mV",
                 "Constant vector magnitude that would produce the observed QRS area."),
            ],
            ("qrs_segment", "geh_area_metrics"),
        )
    )

    # --- Block 2: exact topology ---------------------------------------
    for segment in ("qrs", "t", "qrst"):
        specs.extend(
            _dict_specs(
                lambda ctx, segment=segment: _h0_persistence(ctx, segment),
                "Discovery-v4 exact H0 persistence",
                f"{segment} segment too short for a spanning tree",
                "D",
                "Exact H0 barcode of the Vietoris-Rips filtration via the minimum "
                "spanning tree; a mathematical descriptor without clinical validation.",
                (REF_PERSISTENCE, REF_MST_H0),
                [
                    (f"{segment}_h0_total_persistence_mv", "h0_total_persistence_mv", "mV",
                     f"Sum of exact H0 barcode lengths of the {segment} loop."),
                    (f"{segment}_h0_max_persistence_mv", "h0_max_persistence_mv", "mV",
                     f"Longest exact H0 bar of the {segment} loop."),
                    (f"{segment}_h0_mean_persistence_mv", "h0_mean_persistence_mv", "mV",
                     f"Mean exact H0 bar length of the {segment} loop."),
                ],
                (f"{segment}_segment",),
            )
        )
        specs.extend(
            _dict_specs(
                lambda ctx, segment=segment: _h1_radial_persistence(ctx, segment),
                "Discovery-v4 exact radial H1 persistence",
                f"{segment} radial function degenerate",
                "D",
                "Exact one-dimensional persistent homology of the radial function on "
                "the closed loop; a mathematical descriptor without clinical validation.",
                (REF_PERSISTENCE,),
                [
                    (f"{segment}_radial_h1_max_lifetime_mv", "radial_h1_max_lifetime_mv", "mV",
                     f"Longest exact radial H1 bar of the {segment} loop."),
                    (f"{segment}_radial_h1_total_persistence_mv", "radial_h1_total_persistence_mv", "mV",
                     f"Sum of exact radial H1 bar lengths of the {segment} loop."),
                    (f"{segment}_radial_h1_persistence_entropy", "radial_h1_persistence_entropy", "dimensionless",
                     f"Normalised entropy of the radial H1 barcode of the {segment} loop."),
                    (f"{segment}_radial_h1_bar_count", "radial_h1_bar_count", "count",
                     f"Number of radial H1 bars of the {segment} loop."),
                    (f"{segment}_radial_h1_significant_bar_count", "radial_h1_significant_bar_count", "count",
                     f"Radial H1 bars exceeding a tenth of the {segment} radial span."),
                    (f"{segment}_radial_h1_relative_max_lifetime", "radial_h1_relative_max_lifetime", "dimensionless",
                     f"Longest radial H1 bar relative to the {segment} radial span."),
                ],
                (f"{segment}_segment",),
            )
        )
        specs.extend(
            _dict_specs(
                lambda ctx, segment=segment: _minkowski_spectrum(ctx, segment),
                "Discovery-v4 Minkowski and scale spectrum",
                f"{segment} loop hull degenerate",
                "D",
                "Multi-scale occupancy and Minkowski functionals; a mathematical "
                "descriptor without clinical validation.",
                (REF_LOOP_MORPHOLOGY,),
                [
                    (f"{segment}_minkowski_box_dimension", "minkowski_box_dimension", "dimensionless",
                     f"Box-counting dimension of the {segment} loop across four scales."),
                    (f"{segment}_minkowski_isoperimetric_ratio", "minkowski_isoperimetric_ratio", "dimensionless",
                     f"Isoperimetric ratio of the {segment} convex hull; one for a sphere."),
                    (f"{segment}_minkowski_length_to_hull_surface", "minkowski_length_to_hull_surface", "1/mV",
                     f"Path length of the {segment} loop per unit hull surface."),
                    (f"{segment}_minkowski_occupancy_ratio", "minkowski_occupancy_ratio", "dimensionless",
                     f"Fine-to-coarse occupancy ratio of the {segment} loop."),
                    (f"{segment}_minkowski_hull_surface_mv2", "minkowski_hull_surface_mv2", "mV^2",
                     f"Convex hull surface of the {segment} loop."),
                ],
                (f"{segment}_segment",),
            )
        )
        specs.extend(
            _dict_specs(
                lambda ctx, segment=segment: _recurrence_quantification(ctx, segment),
                "Discovery-v4 full recurrence quantification",
                f"{segment} recurrence matrix degenerate",
                "D",
                "Standard RQA measures computed from the actual recurrence matrix; "
                "a mathematical descriptor without clinical validation.",
                (REF_RQA,),
                [
                    (f"{segment}_rqa_determinism", "rqa_determinism", "dimensionless",
                     f"Fraction of recurrent points of the {segment} loop on diagonal lines."),
                    (f"{segment}_rqa_laminarity", "rqa_laminarity", "dimensionless",
                     f"Fraction of recurrent points of the {segment} loop on vertical lines."),
                    (f"{segment}_rqa_trapping_time", "rqa_trapping_time", "samples",
                     f"Mean vertical line length of the {segment} recurrence matrix."),
                    (f"{segment}_rqa_max_diagonal_line", "rqa_max_diagonal_line", "samples",
                     f"Longest diagonal line of the {segment} recurrence matrix."),
                    (f"{segment}_rqa_diagonal_entropy", "rqa_diagonal_entropy", "dimensionless",
                     f"Entropy of the diagonal line distribution of the {segment} loop."),
                    (f"{segment}_rqa_divergence", "rqa_divergence", "1/samples",
                     f"Inverse maximal diagonal line of the {segment} loop."),
                    (f"{segment}_rqa_recurrence_rate", "rqa_recurrence_rate", "dimensionless",
                     f"Recurrence density of the {segment} loop at the tenth distance percentile."),
                    (f"{segment}_rqa_mean_diagonal_line", "rqa_mean_diagonal_line", "samples",
                     f"Mean diagonal line length of the {segment} recurrence matrix."),
                ],
                (f"{segment}_segment",),
            )
        )
        specs.extend(
            _dict_specs(
                lambda ctx, segment=segment: _spherical_harmonics(ctx, segment),
                "Discovery-v4 real spherical harmonics",
                f"{segment} direction distribution degenerate",
                "D",
                "Real spherical-harmonic power spectrum of the direction density; "
                "replaces a same-named entropy proxy with the actual decomposition.",
                (REF_SPHERICAL_HARMONICS,),
                [
                    (f"{segment}_sh_axial_fraction", "sh_axial_fraction", "dimensionless",
                     f"Degree-2 share of the {segment} direction power; the axis of an orientation distribution."),
                    (f"{segment}_sh_isotropic_fraction", "sh_isotropic_fraction", "dimensionless",
                     f"Degree-0 share of the {segment} direction power; how evenly directions are spread."),
                    (f"{segment}_sh_higher_order_fraction", "sh_higher_order_fraction", "dimensionless",
                     f"Share of {segment} direction power beyond the isotropic and axial terms."),
                    (f"{segment}_sh_axial_to_isotropic", "sh_axial_to_isotropic", "dimensionless",
                     f"Axial-to-isotropic power ratio of the {segment} loop; a directed versus a spread loop."),
                    (f"{segment}_sh_degree3_to_degree2", "sh_degree3_to_degree2", "dimensionless",
                     f"Octupolar-to-axial power ratio of the {segment} loop."),
                    (f"{segment}_sh_odd_degree_fraction", "sh_odd_degree_fraction", "dimensionless",
                     f"Odd-degree power share of the {segment} loop; zero for a perfectly closed loop, so this measures non-closure."),
                    (f"{segment}_sh_spectral_entropy", "sh_spectral_entropy", "dimensionless",
                     f"Entropy of the {segment} spherical-harmonic power spectrum."),
                    (f"{segment}_sh_power_degree_2", "sh_power_degree_2", "dimensionless",
                     f"Degree-2 spherical-harmonic power of the {segment} loop."),
                    (f"{segment}_sh_power_degree_3", "sh_power_degree_3", "dimensionless",
                     f"Degree-3 spherical-harmonic power of the {segment} loop."),
                ],
                (f"{segment}_segment",),
            )
        )
        specs.extend(
            _dict_specs(
                lambda ctx, segment=segment: _spherical_holonomy(ctx, segment),
                "Discovery-v4 spherical holonomy and planar departure",
                f"{segment} direction curve degenerate",
                "D",
                "Solid angle enclosed by the direction curve; a projection-free "
                "measure of three-dimensional twist without clinical validation.",
                (REF_LOOP_FOLDING,),
                [
                    (f"{segment}_holonomy_planar_departure_sr", "holonomy_planar_departure_sr", "steradian",
                     f"Signed departure of the {segment} direction curve from the half sphere a planar loop encloses."),
                    (f"{segment}_holonomy_absolute_planar_departure_sr", "holonomy_absolute_planar_departure_sr", "steradian",
                     f"Absolute three-dimensional twist of the {segment} loop, measured against a planar loop."),
                    (f"{segment}_holonomy_planar_departure_deg", "holonomy_planar_departure_deg", "degrees",
                     f"Planar departure of the {segment} direction curve expressed as an angle."),
                    (f"{segment}_holonomy_orientation_is_clockwise", "holonomy_orientation_is_clockwise", "dimensionless",
                     f"Direction of circulation of the {segment} direction curve on the sphere."),
                ],
                (f"{segment}_segment",),
            )
        )

    # --- Block 3: visual morphology ------------------------------------
    specs.extend(
        _dict_specs(
            _activation_energy_quantiles,
            "Discovery-v4 activation energy timing",
            "QRS segment too short",
            "C",
            "Terminal activation delay is an established concept; the energy-quantile "
            "formulation is prospective.",
            (REF_LOOP_MORPHOLOGY,),
            [
                ("activation_energy_q10_fraction", "activation_energy_q10_fraction", "dimensionless",
                 "Fraction of the QRS elapsed when a tenth of the activation energy is spent."),
                ("activation_energy_q25_fraction", "activation_energy_q25_fraction", "dimensionless",
                 "Fraction of the QRS elapsed at a quarter of the activation energy."),
                ("activation_energy_q50_fraction", "activation_energy_q50_fraction", "dimensionless",
                 "Fraction of the QRS elapsed at half the activation energy."),
                ("activation_energy_q75_fraction", "activation_energy_q75_fraction", "dimensionless",
                 "Fraction of the QRS elapsed at three quarters of the activation energy."),
                ("activation_energy_q90_fraction", "activation_energy_q90_fraction", "dimensionless",
                 "Fraction of the QRS elapsed at nine tenths of the activation energy."),
                ("activation_energy_q50_ms", "activation_energy_q50_ms", "ms",
                 "Time from QRS onset to half the activation energy."),
                ("activation_energy_q90_ms", "activation_energy_q90_ms", "ms",
                 "Time from QRS onset to nine tenths of the activation energy."),
                ("activation_energy_iqr_fraction", "activation_energy_iqr_fraction", "dimensionless",
                 "Span between the quarter and three-quarter energy times."),
                ("activation_energy_terminal_share", "activation_energy_terminal_share", "dimensionless",
                 "Share of activation energy spent in the last quarter of the QRS."),
                ("activation_energy_timing_skew", "activation_energy_skew", "dimensionless",
                 "Displacement of the median energy time from the QRS midpoint."),
            ],
            ("qrs_segment",),
            visual_assessment="direct",
        )
    )
    specs.extend(
        _dict_specs(
            _septal_to_basal,
            "Discovery-v4 septal-to-basal activation",
            "QRS segment too short for thirds",
            "C",
            "Septal-then-basal activation sequence is textbook physiology; this "
            "vector formulation is prospective.",
            (REF_LOOP_MORPHOLOGY,),
            [
                ("septal_to_basal_interval_ms", "septal_to_basal_interval_ms", "ms",
                 "Time between the early-third and late-third QRS force peaks."),
                ("septal_to_basal_angle_deg", "septal_to_basal_angle_deg", "degrees",
                 "Angle between the early-third and late-third QRS force centroids."),
                ("septal_to_basal_distance_mv", "septal_to_basal_distance_mv", "mV",
                 "Distance between the early-third and late-third QRS force centroids."),
                ("septal_basal_magnitude_ratio", "septal_basal_magnitude_ratio", "dimensionless",
                 "Late-third to early-third QRS force magnitude ratio."),
                ("basal_force_share", "basal_force_share", "dimensionless",
                 "Share of total QRS magnitude carried by the last third."),
            ],
            ("qrs_segment",),
            visual_assessment="direct",
        )
    )
    specs.extend(
        _dict_specs(
            _voltage_complexity,
            "Discovery-v4 voltage-complexity mismatch",
            "QRS segment too short",
            "C",
            "Low voltage with preserved or increased fractionation is the "
            "infiltrative-disease hypothesis; the composite is prospective.",
            (REF_AMYLOID,),
            [
                ("voltage_complexity_mismatch_index", "voltage_complexity_mismatch_index", "1/mV",
                 "QRS tortuosity per millivolt of peak magnitude; small but scribbled loops score high."),
                ("voltage_complexity_tortuosity", "voltage_complexity_tortuosity", "dimensionless",
                 "QRS path length divided by its largest point-to-point span."),
                ("voltage_complexity_roughness_fraction", "voltage_complexity_roughness_fraction", "dimensionless",
                 "Mean deviation from the smoothed QRS loop relative to peak magnitude."),
                ("voltage_complexity_high_band_fraction", "voltage_complexity_high_band_fraction", "dimensionless",
                 "Upper-half spectral share of the QRS magnitude envelope."),
                ("voltage_complexity_per_mv_path", "voltage_complexity_per_mv_path", "dimensionless",
                 "QRS path length per unit peak magnitude."),
                ("voltage_complexity_peak_mv", "voltage_complexity_peak_mv", "mV",
                 "Peak spatial QRS magnitude entering the mismatch index."),
            ],
            ("qrs_segment",),
            visual_assessment="direct",
        )
    )
    specs.extend(
        _dict_specs(
            _beat_covariance_tube,
            "Discovery-v4 beat covariance tube",
            "fewer than three usable beat loops",
            "C",
            "Beat-to-beat loop variability has cohort evidence for arrhythmia risk; "
            "the tube formulation is prospective.",
            (REF_BEAT_CLOUD,),
            [
                ("beat_tube_relative_thickness", "beat_tube_relative_thickness", "dimensionless",
                 "Mean beat-to-beat loop spread relative to loop size; a visibly fuzzy loop."),
                ("beat_tube_mean_radius_mv", "beat_tube_mean_radius_mv", "mV",
                 "Mean radius of the tube swept by the individual beat loops."),
                ("beat_tube_max_radius_mv", "beat_tube_max_radius_mv", "mV",
                 "Largest local radius of the beat-loop tube."),
                ("beat_tube_bulge_ratio", "beat_tube_bulge_ratio", "dimensionless",
                 "Largest local tube radius divided by the mean; a locally split loop."),
                ("beat_tube_thickness_entropy", "beat_tube_thickness_entropy", "dimensionless",
                 "Entropy of tube thickness along the loop; uniform versus focal instability."),
                ("beat_tube_ellipsoid_volume_mv3", "beat_tube_ellipsoid_volume_mv3", "mV^3",
                 "Mean pointwise covariance ellipsoid volume of the beat loops."),
                ("beat_tube_late_to_early_ratio", "beat_tube_late_to_early_ratio", "dimensionless",
                 "Tube thickness in the second half of the loop relative to the first."),
                ("beat_tube_n_beats", "beat_tube_n_beats", "count",
                 "Number of beat loops entering the tube."),
            ],
            ("beat_loops",),
            visual_assessment="direct",
        )
    )
    specs.extend(
        _dict_specs(
            _respiratory_modulation,
            "Discovery-v4 respiratory loop modulation",
            "fewer than eight usable beat loops",
            "C",
            "Respiratory modulation of the electrical axis is established; measuring "
            "its plane and centroid swing per record is prospective.",
            (REF_RESPIRATION,),
            [
                ("respiratory_plane_swing_deg", "respiratory_plane_swing_deg", "degrees",
                 "Range of beat-loop plane orientation across the record."),
                ("respiratory_plane_swing_sd_deg", "respiratory_plane_swing_sd_deg", "degrees",
                 "Standard deviation of beat-loop plane orientation."),
                ("respiratory_centroid_swing_mv", "respiratory_centroid_swing_mv", "mV",
                 "Largest displacement of the beat-loop centroid across the record."),
                ("respiratory_magnitude_cv", "respiratory_magnitude_cv", "dimensionless",
                 "Coefficient of variation of peak beat-loop magnitude."),
                ("respiratory_periodicity_fraction", "respiratory_periodicity_fraction", "dimensionless",
                 "Share of plane-angle variance at its dominant frequency; oscillation versus drift."),
                ("respiratory_cycles_per_record", "respiratory_cycles_per_record", "count",
                 "Dominant plane-angle cycle count over the record."),
            ],
            ("beat_loops",),
            visual_assessment="direct",
        )
    )
    specs.extend(
        _dict_specs(
            _cross_axis_coupling,
            "Discovery-v4 cross-axis coupling",
            "QRST segment too short for spectral estimation",
            "D",
            "Phase coupling between spatial axes as a fragmentation hypothesis; no "
            "independent clinical validation.",
            (REF_LOOP_MORPHOLOGY,),
            [
                ("cross_axis_coherence_mean", "cross_axis_coherence_mean", "dimensionless",
                 "Mean magnitude-squared coherence across the three axis pairs."),
                ("cross_axis_coherence_min", "cross_axis_coherence_min", "dimensionless",
                 "Lowest axis-pair coherence; the axis pair that decoupled."),
                ("cross_axis_coherence_xy", "cross_axis_coherence_xy", "dimensionless",
                 "Coherence between the X and Y axes over the QRST interval."),
                ("cross_axis_coherence_xz", "cross_axis_coherence_xz", "dimensionless",
                 "Coherence between the X and Z axes over the QRST interval."),
                ("cross_axis_coherence_yz", "cross_axis_coherence_yz", "dimensionless",
                 "Coherence between the Y and Z axes over the QRST interval."),
                ("cross_axis_bicoherence", "cross_axis_bicoherence", "dimensionless",
                 "Quadratic phase coupling of the spatial magnitude envelope."),
            ],
            ("qrst_segment",),
        )
    )
    for segment in ("qrs", "t", "p"):
        specs.extend(
            _dict_specs(
                lambda ctx, segment=segment: _visual_loop_reading(ctx, segment),
                "Discovery-v4 visual loop reading",
                f"{segment} loop too short to read",
                "C",
                "Each construct is a sentence a reader says out loud about a loop; "
                "the quantification is prospective.",
                (REF_LOOP_MORPHOLOGY, REF_LOOP_FOLDING),
                [
                    (f"{segment}_visual_open_gap_fraction", "visual_open_gap_fraction", "dimensionless",
                     f"Distance from {segment} loop start to end, relative to loop size; an open loop."),
                    (f"{segment}_visual_waist_fraction", "visual_waist_fraction", "dimensionless",
                     f"Narrowest to widest cross-loop width of the {segment} loop; a pinched waist."),
                    (f"{segment}_visual_reversal_fraction", "visual_reversal_fraction", "dimensionless",
                     f"Share of {segment} loop steps that reverse direction; visible doubling back."),
                    (f"{segment}_visual_late_roughness_ratio", "visual_late_roughness_ratio", "dimensionless",
                     f"Roughness of the last third of the {segment} loop relative to the first."),
                    (f"{segment}_visual_roughness_fraction", "visual_roughness_fraction", "dimensionless",
                     f"Mean deviation of the {segment} loop from its smoothed course, relative to size."),
                    (f"{segment}_visual_path_to_span_ratio", "visual_path_to_span_ratio", "dimensionless",
                     f"Path length of the {segment} loop relative to twice its radius."),
                    (f"{segment}_visual_elongation", "visual_elongation", "dimensionless",
                     f"Long-axis to short-axis extent of the {segment} loop; long and narrow versus round."),
                    (f"{segment}_visual_mass_centroid_fraction", "visual_mass_centroid_fraction", "dimensionless",
                     f"Magnitude-weighted temporal centroid of the {segment} loop; leading or trailing mass."),
                    (f"{segment}_visual_waist_width_mv", "visual_waist_width_mv", "mV",
                     f"Absolute narrowest cross-loop width of the {segment} loop."),
                ],
                (f"{segment}_segment",),
                visual_assessment="direct",
            )
        )
    for segment in ("qrs", "t"):
        for plane in PLANES:
            specs.extend(
                _dict_specs(
                    lambda ctx, segment=segment, plane=plane: _plane_projection_reading(
                        ctx, segment, plane
                    ),
                    "Discovery-v4 projection reading",
                    f"{segment} projection onto {plane} degenerate",
                    "C",
                    "Projection-specific loop morphology as read from the three standard "
                    "views; the per-view quantification is prospective.",
                    (REF_LOOP_MORPHOLOGY,),
                    [
                        (f"{segment}_{plane}_signed_area_mv2", "plane_signed_area_mv2", "mV^2",
                         f"Signed area of the {segment} loop in the {plane} view; sign is rotation direction."),
                        (f"{segment}_{plane}_area_to_hull_ratio", "plane_area_to_hull_ratio", "dimensionless",
                         f"Enclosed to hull area of the {segment} loop in the {plane} view; concavity."),
                        (f"{segment}_{plane}_self_intersections", "plane_self_intersections", "count",
                         f"Self-crossings of the {segment} loop in the {plane} view; a figure-of-eight."),
                        (f"{segment}_{plane}_circularity", "plane_circularity", "dimensionless",
                         f"Isoperimetric circularity of the {segment} loop in the {plane} view."),
                        (f"{segment}_{plane}_bite_depth_fraction", "plane_bite_depth_fraction", "dimensionless",
                         f"Hull area not enclosed by the {segment} loop in the {plane} view; a visible bite."),
                        (f"{segment}_{plane}_perimeter_to_hull_ratio", "plane_perimeter_to_hull_ratio", "dimensionless",
                         f"Loop perimeter to hull perimeter of the {segment} loop in the {plane} view."),
                    ],
                    (f"{segment}_segment",),
                    visual_assessment="direct",
                )
            )
    specs.extend(
        _dict_specs(
            _qrs_t_visual_relation,
            "Discovery-v4 QRS-T visual relation",
            "QRS or T loop unavailable",
            "B",
            "QRS-T angle and discordance have extensive cohort evidence; the "
            "junction-gap and separation formulations are prospective.",
            (REF_GEH, REF_LOOP_MORPHOLOGY),
            [
                ("qrs_t_visual_peak_angle_deg", "qrs_t_peak_angle_deg", "degrees",
                 "Angle between the peak QRS and peak T vectors; visible opposition."),
                ("qrs_t_visual_size_ratio", "qrs_t_size_ratio", "dimensionless",
                 "Peak T magnitude relative to peak QRS magnitude."),
                ("qrs_t_visual_junction_gap_fraction", "qrs_t_junction_gap_fraction", "dimensionless",
                 "Distance from the QRS loop end to the T loop start, relative to QRS size."),
                ("qrs_t_visual_minimum_separation_fraction", "qrs_t_minimum_separation_fraction", "dimensionless",
                 "Closest approach of the QRS and T loops, relative to QRS size."),
                ("qrs_t_visual_centroid_distance_mv", "qrs_t_centroid_distance_mv", "mV",
                 "Distance between the QRS and T loop centroids."),
                ("qrs_t_visual_unusual_parallelism", "qrs_t_unusual_parallelism", "dimensionless",
                 "Flag for a QRS-T peak angle below twenty degrees; abnormally parallel loops."),
            ],
            ("qrs_segment", "t_segment"),
            visual_assessment="direct",
        )
    )
    return tuple(specs)


DISCOVERY_V4_SPECS: tuple[DiscoveryV4Spec, ...] = _build_specs()


class DiscoveryV4Biomarker(BiomarkerBase):
    def __init__(self, spec: DiscoveryV4Spec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def unit(self) -> str:
        return self._spec.unit

    @property
    def category(self) -> str:
        return "vcg"

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def references(self) -> list[str]:
        return list(self._spec.references)

    def compute(
        self,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray] = None,
        vcg_xyz: Optional[np.ndarray] = None,
        fs: int = 500,
    ) -> BiomarkerResult:
        ctx = _context(delineation, ecg_12lead, vcg_xyz, fs)
        try:
            value = self._spec.calculator(ctx)
        except Exception as error:  # noqa: BLE001 - one marker must not stop a panel
            value = FeatureValue(None, 0.0, notes=f"not_evaluable: {error}")
        notes = [
            f"discovery_v4_version={DISCOVERY_V4_VERSION}",
            f"axis_convention={VCG_AXIS_CONVENTION_VERSION}",
            f"visual_assessment={self._spec.visual_assessment}",
            f"evidence={self._spec.evidence_level}",
        ]
        if value.notes:
            notes.insert(0, value.notes)
        return BiomarkerResult(
            name=self._spec.name,
            value=value.value,
            unit=self._spec.unit,
            category="vcg",
            confidence=float(value.confidence),
            description=self._spec.description,
            references=list(self._spec.references),
            computation_notes="; ".join(notes),
            family=self._spec.family,
            formula=self._spec.formula,
        )


def visually_interpretable_marker_ids() -> tuple[str, ...]:
    """Markers a reader can verify by looking at the loop.

    This is the subset worth showing a human when a screening hit needs
    adjudicating: every one of them corresponds to something visible.
    """

    return tuple(
        spec.name for spec in DISCOVERY_V4_SPECS if spec.visually_interpretable
    )


def register_discovery_v4_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in DISCOVERY_V4_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), DiscoveryV4Biomarker):
                continue
            raise RuntimeError(
                f"Discovery-v4 marker collides with an existing biomarker: {spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=DISCOVERY_V4_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    f"Prospective Discovery-v4 hypothesis; evidence {spec.evidence_level}. "
                    f"{spec.evidence_note} No diagnostic cutoff is claimed."
                ),
                references=list(spec.references),
                implementation_status="implemented",
            )
        )
        BiomarkerRegistry.register(DiscoveryV4Biomarker(spec))


__all__ = [
    "DISCOVERY_V4_SPECS",
    "DISCOVERY_V4_SOURCE",
    "DISCOVERY_V4_VERSION",
    "DiscoveryV4Biomarker",
    "DiscoveryV4Spec",
    "register_discovery_v4_biomarkers",
    "visually_interpretable_marker_ids",
]
