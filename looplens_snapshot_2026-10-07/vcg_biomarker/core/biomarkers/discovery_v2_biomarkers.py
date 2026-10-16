"""Prospective, explicitly exploratory VCG biomarker block.

This module contains hypotheses identified during the August 2026 scientific
audit.  They are deliberately isolated from the historical catalog and from
the frozen Core/Sentinel panels.  Every output is a scalar with an explicit
unit; vector-valued foundations are exposed as separate component markers.

The implementation follows four rules which are important for a brute-force
screening panel:

* expensive segment, SVD, cloud and geometry primitives are memoized on the
  record-scoped :class:`FeatureContext` and shared by all markers;
* coordinate-dependent quantities say so (X/Y/Z, azimuth/elevation), while
  shape descriptors are translation/rotation/scale invariant where stated;
* an unavailable or degenerate prerequisite produces an abstention, never a
  fabricated zero;
* evidence grades describe prior support, not diagnostic validity.  Grade D
  is a mathematical/pathophysiological hypothesis only.

Evidence grades in this file are intentionally conservative: A = replicated
human outcome/method evidence; B = clinical cohort evidence for the construct;
C = limited, transfer, or normative-cohort evidence; D = unvalidated
prospective hypothesis.  No grade implies a transportable cutoff.

No threshold in this module is a clinical cutoff.  All markers were conceived
after outcome-facing project development and are Discovery Reserve only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.spatial import ConvexHull, distance

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from .catalog_biomarkers import (
    EPS,
    FeatureContext,
    FeatureValue,
    _angle_between,
    _azimuth,
    _context,
    _elevation,
    _resample_curve,
    _vector_for,
)
from ..data_loader import LEAD_MAP
from ..delineation.base import (
    RecordDelineation,
    measurement_confidence_threshold,
)


DISCOVERY_V2_VERSION = "prospective-discovery-v2.0.0"
DISCOVERY_V2_SOURCE = "Prospective VCG scientific gap audit, 2026-08"

REF_QRS_AREA = "https://doi.org/10.1016/j.hrthm.2015.07.033"
REF_QRS_AREA_CRT = "https://doi.org/10.1016/j.jelectrocard.2014.10.003"
REF_GEH_METHOD = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8627471/"
REF_TWR_METHOD = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6932246/"
REF_NONDIPOLAR = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6932570/"
REF_TT_PRIME = "https://doi.org/10.1161/JAHA.114.001357"
REF_DYNAMIC_VCG = "https://doi.org/10.1371/journal.pone.0057175"
REF_PEAK_CLOUD = "https://doi.org/10.1016/j.hrthm.2010.08.022"
REF_VG_RVPO = "https://doi.org/10.1038/s41598-021-04186-8"
REF_VTI = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12674847/"
REF_LOOP_MORPHOLOGY = "https://doi.org/10.1016/j.jelectrocard.2015.12.014"
REF_BROAD_VCG = "https://doi.org/10.1016/j.jelectrocard.2023.12.004"


@dataclass(frozen=True)
class DiscoveryV2Spec:
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


def _abstain(reason: str) -> FeatureValue:
    return FeatureValue(None, 0.0, notes=f"not_evaluable: {reason}")


def _memo(
    ctx: FeatureContext,
    key: tuple[object, ...],
    factory: Callable[[], object],
):
    """Record-scoped memoization without a second global/thread-local cache."""

    cache = getattr(ctx, "_discovery_v2_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_discovery_v2_cache", cache)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _finite_rows(points: object, *, columns: Optional[int] = 3) -> Optional[np.ndarray]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or (columns is not None and values.shape[1] != columns):
        return None
    values = values[np.all(np.isfinite(values), axis=1)]
    return values if len(values) else None


def _segment(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        # Origin-dependent vectors (peak direction, VTI, octants, spherical
        # paths) and origin-independent shape metrics must share one
        # physiological reference.  In particular, mixing a raw peak vector
        # with an origin-corrected area vector would make peak/area angles
        # depend on arbitrary acquisition DC offsets.
        values = ctx.vcg_segment_origin_corrected(segment)
        return None if values is None else _finite_rows(values)

    return _memo(ctx, ("segment", segment), build)


def _area_vector_mv_ms(
    ctx: FeatureContext,
    segment: str,
) -> Optional[tuple[np.ndarray, dict[str, float]]]:
    """Baseline-referenced area vector, converted from mV*s to mV*ms.

    ``_vector_for`` is intentionally reused: it is the single cached area
    primitive shared with the historical catalog and with the central GEH
    origin policy.  Discovery-v2 therefore cannot silently choose a different
    isoelectric origin.
    """

    def build() -> Optional[tuple[np.ndarray, dict[str, float]]]:
        result = _vector_for(ctx, segment, "area")
        if result is None:
            return None
        vector = np.asarray(result[0], dtype=float) * 1000.0
        magnitude = float(np.linalg.norm(vector))
        azimuth = _azimuth(vector)
        elevation = _elevation(vector)
        details = {
            "x": float(vector[0]),
            "y": float(vector[1]),
            "z": float(vector[2]),
            "magnitude": magnitude,
            "azimuth": float(azimuth) if azimuth is not None else np.nan,
            "elevation": float(elevation) if elevation is not None else np.nan,
        }
        return vector, details

    return _memo(ctx, ("area_vector_mv_ms", segment), build)


def _area_component(segment: str, component: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _area_vector_mv_ms(ctx, segment)
        if result is None:
            return _abstain(f"{segment} segment unavailable")
        value = result[1][component]
        if not np.isfinite(value):
            return _abstain(f"{segment} {component} undefined for a zero vector")
        return FeatureValue(float(value))

    return calculate


def _peak_vector(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        points = _segment(ctx, segment)
        if points is None or len(points) < 2:
            return None
        magnitude = np.linalg.norm(points, axis=1)
        if not np.any(np.isfinite(magnitude)) or float(np.max(magnitude)) <= EPS:
            return None
        return np.asarray(points[int(np.nanargmax(magnitude))], dtype=float)

    return _memo(ctx, ("peak_vector", segment), build)


def _peak_svg(ctx: FeatureContext) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        qrs = _peak_vector(ctx, "qrs")
        t_wave = _peak_vector(ctx, "t")
        if qrs is None or t_wave is None:
            return None
        return qrs + t_wave

    return _memo(ctx, ("peak_svg",), build)


def _peak_svg_component(component: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        vector = _peak_svg(ctx)
        if vector is None:
            return _abstain("QRS-peak or T-peak vector unavailable")
        magnitude = float(np.linalg.norm(vector))
        if component == "magnitude":
            return FeatureValue(magnitude)
        if component == "azimuth":
            value = _azimuth(vector)
        elif component == "elevation":
            value = _elevation(vector)
        else:
            value = float(vector[{"x": 0, "y": 1, "z": 2}[component]])
        if value is None or not np.isfinite(value):
            return _abstain("peak SVG direction is undefined for a zero vector")
        return FeatureValue(float(value))

    return calculate


def _peak_area_svg_angle(ctx: FeatureContext) -> FeatureValue:
    peak = _peak_svg(ctx)
    qrs = _area_vector_mv_ms(ctx, "qrs")
    t_wave = _area_vector_mv_ms(ctx, "t")
    if peak is None or qrs is None or t_wave is None:
        return _abstain("peak or area SVG unavailable")
    angle = _angle_between(peak, qrs[0] + t_wave[0])
    return _abstain("peak or area SVG is degenerate") if angle is None else FeatureValue(angle)


def _peak_area_svg_magnitude_ratio(ctx: FeatureContext) -> FeatureValue:
    peak = _peak_svg(ctx)
    qrs = _area_vector_mv_ms(ctx, "qrs")
    t_wave = _area_vector_mv_ms(ctx, "t")
    if peak is None or qrs is None or t_wave is None:
        return _abstain("peak or area SVG unavailable")
    # Divide the area SVG by its QRST duration to compare voltage with voltage.
    duration_s = ctx.duration_s("qrst")
    if duration_s is None or duration_s <= 0:
        return _abstain("QRST duration unavailable")
    area_mean_voltage = np.linalg.norm(qrs[0] + t_wave[0]) / (duration_s * 1000.0)
    if area_mean_voltage <= EPS:
        return _abstain("area SVG mean voltage is zero")
    return FeatureValue(float(np.linalg.norm(peak) / area_mean_voltage))


INDEPENDENT_8_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def _lead8_spectrum(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    """Eight-lead mean-square singular spectrum in mV^2.

    III/aVR/aVL/aVF are algebraic derivatives and are deliberately excluded.
    Channel means are removed before SVD, making the spectrum invariant to a
    lead-wise DC offset.  No variance-standardisation is applied because the
    relative physical lead amplitudes carry the morphology of interest.
    """

    def build() -> Optional[np.ndarray]:
        if ctx.ecg_12lead is None:
            return None
        bounds = ctx.bounds(segment)
        if bounds is None or not ctx.is_segment_evaluable(segment, 0.35):
            return None
        indices = [LEAD_MAP[name] for name in INDEPENDENT_8_LEADS]
        if max(indices) >= ctx.ecg_12lead.shape[1]:
            return None
        matrix = np.asarray(
            ctx.ecg_12lead[bounds[0] : bounds[1] + 1, indices],
            dtype=float,
        )
        matrix = _finite_rows(matrix, columns=8)
        if matrix is None or len(matrix) < 8:
            return None
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
        singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        energy = np.zeros(8, dtype=float)
        energy[: len(singular)] = singular**2 / float(len(matrix))
        return energy

    return _memo(ctx, ("lead8_spectrum", segment), build)


def _lead8_metric(segment: str, metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        energy = _lead8_spectrum(ctx, segment)
        if energy is None:
            return _abstain(f"eight independent leads or {segment} bounds unavailable")
        total = float(np.sum(energy))
        if total <= EPS:
            return _abstain(f"zero {segment} energy in the eight independent leads")
        if metric == "nondipolar_fraction":
            value = float(np.sum(energy[3:]) / total)
        elif metric == "nondipolar_energy":
            value = float(np.sum(energy[3:]))
        elif metric == "sigma2_sigma1":
            value = float(np.sqrt(energy[1] / energy[0])) if energy[0] > EPS else np.nan
        elif metric == "effective_rank":
            shares = energy / total
            positive = shares[shares > EPS]
            value = float(np.exp(-np.sum(positive * np.log(positive))))
        else:  # pragma: no cover - specs constrain this
            raise ValueError(metric)
        return _abstain(f"{segment} {metric} undefined") if not np.isfinite(value) else FeatureValue(value)

    return calculate


def _vti_3d(ctx: FeatureContext, segment: str) -> FeatureValue:
    points = _segment(ctx, segment)
    if points is None or len(points) < 2 or ctx.fs <= 0:
        return _abstain(f"{segment} segment unavailable")
    value = np.trapezoid(np.linalg.norm(points, axis=1), dx=1000.0 / ctx.fs)
    return FeatureValue(float(value))


def _vti_calculator(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    return lambda ctx: _vti_3d(ctx, segment)


def _vg_rvpo(ctx: FeatureContext) -> FeatureValue:
    qrs = _area_vector_mv_ms(ctx, "qrs")
    t_wave = _area_vector_mv_ms(ctx, "t")
    if qrs is None or t_wave is None:
        return _abstain("QRS or T area vector unavailable")
    vg = qrs[0] + t_wave[0]
    # Published inverse-Dower XYZ projection (elevation 27 deg, azimuth 155 deg).
    coefficients = np.asarray(
        [-0.807526151172, 0.404508497187, 0.376555628452],
        dtype=float,
    )
    return FeatureValue(float(np.dot(coefficients, vg)))


def _accepted_beats(ctx: FeatureContext) -> list[object]:
    beats = getattr(ctx.delineation, "accepted_beats", None)
    if beats is None:
        beats = getattr(ctx.delineation, "beats", ())
    return sorted(
        (
            beat
            for beat in beats
            if not bool(getattr(beat, "morphology_outlier", False))
            and getattr(beat, "morphology_cluster_id", 0) in (None, 0)
        ),
        key=lambda beat: int(getattr(getattr(beat, "r_peak", None), "sample", 0)),
    )


def _beat_peak_vectors(ctx: FeatureContext, marker: str) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        segment_index = 0 if marker == "r_peak" else 1
        vectors: list[np.ndarray] = []
        for beat in _accepted_beats(ctx):
            geometry = _dynamic_beat_geometry(ctx, beat)
            points = None if geometry is None else geometry[segment_index]
            if points is None or len(points) < 2:
                continue
            # Dynamic-VCG papers define the spatial R/T peak as the point
            # furthest from that beat's translated QRS-T crosspoint, not
            # necessarily the scalar lead's annotated r_peak/t_peak sample.
            magnitude = np.linalg.norm(points, axis=1)
            if float(np.max(magnitude)) <= EPS:
                continue
            vectors.append(np.asarray(points[int(np.argmax(magnitude))], dtype=float))
        return np.asarray(vectors, dtype=float) if vectors else None

    return _memo(ctx, ("beat_peak_vectors", marker), build)


def _tt_prime_angles(ctx: FeatureContext) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        vectors = _beat_peak_vectors(ctx, "t_peak")
        if vectors is None or len(vectors) < 3:
            return None
        first = vectors[:-1]
        second = vectors[1:]
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        usable = denominator > EPS
        if np.sum(usable) < 2:
            return None
        cosine = np.sum(first[usable] * second[usable], axis=1) / denominator[usable]
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return _memo(ctx, ("tt_prime_angles",), build)


def _tt_prime_metric(metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        angles = _tt_prime_angles(ctx)
        if angles is None or not len(angles):
            return _abstain("at least three evaluable sinus-morphology T peaks required")
        if metric == "mean":
            value = np.mean(angles)
        elif metric == "median":
            value = np.median(angles)
        elif metric == "p90":
            value = np.percentile(angles, 90)
        elif metric == "max":
            value = np.max(angles)
        else:  # resultant length of the ordered unit T vectors
            vectors = _beat_peak_vectors(ctx, "t_peak")
            norms = np.linalg.norm(vectors, axis=1)
            unit = vectors[norms > EPS] / norms[norms > EPS, None]
            value = np.linalg.norm(np.mean(unit, axis=0))
        # Outcome work used routine 10-s recordings; retain short records for
        # discovery, but expose their reduced sampling support in confidence.
        confidence = 0.65 * min(1.0, len(angles) / 8.0)
        return FeatureValue(
            float(value),
            confidence=float(confidence),
            notes=f"n_consecutive_t_vector_pairs={len(angles)}; beat-local crosspoint translation",
        )

    return calculate


def _cloud_metrics(ctx: FeatureContext, marker: str) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _beat_peak_vectors(ctx, marker)
        if points is None or len(points) < 4:
            return None
        centered = points - np.mean(points, axis=0, keepdims=True)
        covariance = np.cov(centered.T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
        rms_radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        max_radius = float(np.max(np.linalg.norm(centered, axis=1)))
        ellipsoid = float(4.0 * np.pi / 3.0 * np.sqrt(np.prod(eigenvalues)))
        anisotropy = (
            float(1.0 - eigenvalues[-1] / eigenvalues[0])
            if eigenvalues[0] > EPS
            else 0.0
        )
        if np.linalg.matrix_rank(centered, tol=1e-10) < 3:
            hull_volume = 0.0
        else:
            try:
                hull_volume = float(ConvexHull(points).volume)
            except Exception:
                hull_volume = np.nan
        return {
            "rms_radius": rms_radius,
            "max_radius": max_radius,
            "ellipsoid_volume": ellipsoid,
            "convex_hull_volume": hull_volume,
            "anisotropy": anisotropy,
            "n_vectors": float(len(points)),
        }

    return _memo(ctx, ("cloud_metrics", marker), build)


def _cloud_metric(marker: str, metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        metrics = _cloud_metrics(ctx, marker)
        if metrics is None:
            return _abstain("at least four evaluable sinus-morphology peak vectors required")
        value = metrics[metric]
        if not np.isfinite(value):
            return _abstain("peak cloud is numerically degenerate")
        n_vectors = int(metrics["n_vectors"])
        return FeatureValue(
            value,
            confidence=0.65 * min(1.0, n_vectors / 30.0),
            notes=f"n_peak_vectors={n_vectors}; published peak-cloud protocol used 30 sinus beats",
        )

    return calculate


def _cloud_ratio(metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        r_cloud = _cloud_metrics(ctx, "r_peak")
        t_cloud = _cloud_metrics(ctx, "t_peak")
        if r_cloud is None or t_cloud is None:
            return _abstain("both R and T peak clouds require at least four vectors")
        denominator = float(r_cloud[metric])
        if denominator <= EPS:
            return _abstain(f"R peak cloud {metric} is zero")
        n_vectors = int(min(r_cloud["n_vectors"], t_cloud["n_vectors"]))
        return FeatureValue(
            float(t_cloud[metric] / denominator),
            confidence=0.65 * min(1.0, n_vectors / 30.0),
            notes=f"n_paired_peak_vectors={n_vectors}; published peak-cloud protocol used 30 sinus beats",
        )

    return calculate


def _beat_segment(ctx: FeatureContext, beat: object, segment: str) -> Optional[np.ndarray]:
    # Dynamic-VCG beat alignment has its own published, beat-local origin
    # (the QRS/T crosspoint).  Read the record array here; the common offset is
    # removed by ``_dynamic_beat_geometry`` below.
    source = ctx._record_vcg_xyz
    if source is None:
        return None
    if segment == "qrs":
        first, last = "r_onset", "r_offset"
    elif segment == "t":
        first, last = "t_onset", "t_offset"
    else:
        return None
    start_point = getattr(beat, first, None)
    end_point = getattr(beat, last, None)
    if (
        start_point is None
        or end_point is None
        or min(
            float(getattr(start_point, "confidence", 0.0)),
            float(getattr(end_point, "confidence", 0.0)),
        ) < 0.35
    ):
        return None
    start, end = int(start_point.sample), int(end_point.sample)
    if start < 0 or end >= len(source) or end <= start:
        return None
    return _finite_rows(source[start : end + 1])


def _dynamic_beat_geometry(
    ctx: FeatureContext,
    beat: object,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """QRS/T loops translated to the beat-local dynamic-VCG crosspoint.

    Published TT-prime processing aligns each beat by translating the QRS/T
    crosspoint to (0, 0, 0), which suppresses baseline wander without erasing
    the directional repolarization variation of interest.  With already
    delineated disjoint QRS and T samples, the midpoint of their closest pair
    is the explicit discrete crosspoint estimator.
    """

    def build() -> Optional[tuple[np.ndarray, np.ndarray]]:
        qrs = _beat_segment(ctx, beat, "qrs")
        t_wave = _beat_segment(ctx, beat, "t")
        if qrs is None or t_wave is None:
            return None
        pairwise = distance.cdist(qrs, t_wave)
        if not pairwise.size or not np.any(np.isfinite(pairwise)):
            return None
        row, column = np.unravel_index(int(np.nanargmin(pairwise)), pairwise.shape)
        origin = 0.5 * (qrs[row] + t_wave[column])
        return qrs - origin, t_wave - origin

    return _memo(ctx, ("dynamic_beat_geometry", id(beat)), build)


def _minimum_interloop_distance(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    if first is None or second is None or len(first) < 2 or len(second) < 2:
        return None
    matrix = distance.cdist(first, second)
    return float(np.min(matrix)) if matrix.size else None


def _interloop_distances(ctx: FeatureContext) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        values = []
        for beat in _accepted_beats(ctx):
            geometry = _dynamic_beat_geometry(ctx, beat)
            qrs, t_wave = geometry if geometry is not None else (None, None)
            value = _minimum_interloop_distance(qrs, t_wave)
            if value is not None and np.isfinite(value):
                values.append(value)
        return np.asarray(values, dtype=float) if values else None

    return _memo(ctx, ("interloop_distances",), build)


def _representative_interloop(ctx: FeatureContext) -> FeatureValue:
    value = _minimum_interloop_distance(_segment(ctx, "qrs"), _segment(ctx, "t"))
    return _abstain("QRS or T loop unavailable") if value is None else FeatureValue(value)


def _interloop_metric(metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        values = _interloop_distances(ctx)
        if values is None or len(values) < (2 if metric == "sd" else 1):
            return _abstain("insufficient beatwise QRS/T loop pairs")
        return FeatureValue(float(np.mean(values) if metric == "mean" else np.std(values, ddof=1)))

    return calculate


def _interloop_geometry(ctx: FeatureContext) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        qrs = _segment(ctx, "qrs")
        t_wave = _segment(ctx, "t")
        if qrs is None or t_wave is None:
            return None
        qrs_r = _resample_curve(qrs, 64)
        t_r = _resample_curve(t_wave, 64)
        if qrs_r is None or t_r is None:
            return None
        matrix = distance.cdist(qrs_r, t_r)
        chamfer = 0.5 * (float(np.mean(np.min(matrix, axis=0))) + float(np.mean(np.min(matrix, axis=1))))
        centroid = float(np.linalg.norm(np.mean(qrs_r, axis=0) - np.mean(t_r, axis=0)))
        q_scale = np.sqrt(np.mean(np.sum((qrs_r - np.mean(qrs_r, axis=0)) ** 2, axis=1)))
        t_scale = np.sqrt(np.mean(np.sum((t_r - np.mean(t_r, axis=0)) ** 2, axis=1)))
        scale = float(np.sqrt(q_scale * q_scale + t_scale * t_scale))
        q_peak = _peak_vector(ctx, "qrs")
        t_peak = _peak_vector(ctx, "t")
        peak_distance = (
            float(np.linalg.norm(q_peak - t_peak))
            if q_peak is not None and t_peak is not None
            else np.nan
        )
        return {
            "centroid_distance": centroid,
            "chamfer_distance": chamfer,
            "normalized_chamfer": chamfer / scale if scale > EPS else np.nan,
            "peak_distance": peak_distance,
        }

    return _memo(ctx, ("interloop_geometry",), build)


def _interloop_geometry_metric(metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _interloop_geometry(ctx)
        if result is None or not np.isfinite(result[metric]):
            return _abstain("representative QRS/T loop geometry unavailable")
        return FeatureValue(float(result[metric]))

    return calculate


def _closure_ratio(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        points = _segment(ctx, segment)
        if points is None or len(points) < 3:
            return _abstain(f"{segment} loop unavailable")
        path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if path <= EPS:
            return _abstain(f"{segment} loop has zero path length")
        return FeatureValue(float(np.linalg.norm(points[-1] - points[0]) / path))

    return calculate


PLANES = {"frontal_xy": (0, 1), "horizontal_xz": (0, 2), "sagittal_yz": (1, 2)}


def _projected_orientation_metrics(
    ctx: FeatureContext,
    segment: str,
    plane: str,
) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, segment)
        if points is None or len(points) < 4:
            return None
        projection = points[:, PLANES[plane]]
        center = np.mean(projection, axis=0)
        closed = np.vstack((projection, projection[0]))
        a = closed[:-1] - center
        b = closed[1:] - center
        triangles = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        absolute = float(np.sum(np.abs(triangles)))
        if absolute <= EPS:
            return None
        signed = float(np.sum(triangles))
        balance = float(np.clip(signed / absolute, -1.0, 1.0))
        return {
            "orientation_balance": balance,
            "signed_area_cancellation": float(1.0 - abs(balance)),
        }

    return _memo(ctx, ("projected_orientation", segment, plane), build)


def _projected_orientation_calculator(
    segment: str,
    plane: str,
    metric: str,
) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _projected_orientation_metrics(ctx, segment, plane)
        if result is None:
            return _abstain(f"{segment} {plane} projected area is degenerate")
        return FeatureValue(result[metric])

    return calculate


def _direction_solid_angles(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    """Signed and absolute area swept on the unit sphere, in steradians."""

    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, segment)
        if points is None or len(points) < 4:
            return None
        magnitude = np.linalg.norm(points, axis=1)
        peak = float(np.max(magnitude))
        keep = magnitude >= max(EPS, 0.05 * peak)
        unit = points[keep] / magnitude[keep, None]
        if len(unit) < 4:
            return None
        # Arc-length resampling prevents sample-rate/dwell-time dependence.
        unit = _resample_curve(unit, 64)
        if unit is None:
            return None
        norm = np.linalg.norm(unit, axis=1)
        unit = unit[norm > EPS] / norm[norm > EPS, None]
        if len(unit) < 4:
            return None
        anchor = unit[0]
        triangles = []
        for index in range(1, len(unit) - 1):
            b, c = unit[index], unit[index + 1]
            numerator = float(np.dot(anchor, np.cross(b, c)))
            denominator = float(
                1.0 + np.dot(anchor, b) + np.dot(b, c) + np.dot(c, anchor)
            )
            triangles.append(2.0 * np.arctan2(numerator, denominator))
        if not triangles:
            return None
        values = np.asarray(triangles, dtype=float)
        return {
            "signed": float(np.sum(values)),
            "absolute": float(np.sum(np.abs(values))),
        }

    return _memo(ctx, ("direction_solid_angles", segment), build)


def _solid_angle_metric(segment: str, metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _direction_solid_angles(ctx, segment)
        if result is None:
            return _abstain(f"{segment} direction path unavailable or crosses a degenerate origin")
        return FeatureValue(result[metric])

    return calculate


def _closed_resampled(points: Optional[np.ndarray], n: int = 48) -> Optional[np.ndarray]:
    if points is None or len(points) < 4:
        return None
    sampled = _resample_curve(np.asarray(points, dtype=float), n)
    if sampled is None:
        return None
    return np.vstack((sampled, sampled[0]))


def _gauss_pair_integrals(
    first: np.ndarray,
    second: np.ndarray,
    *,
    self_pair: bool,
) -> Optional[dict[str, float]]:
    a = _closed_resampled(first)
    b = a if self_pair else _closed_resampled(second)
    if a is None or b is None:
        return None
    da, db = np.diff(a, axis=0), np.diff(b, axis=0)
    ma = 0.5 * (a[:-1] + a[1:])
    mb = 0.5 * (b[:-1] + b[1:])
    # The quadrature is a Cartesian product of at most 48 x 48 segment
    # midpoints.  Broadcasting it executes the identical Gauss integrand in
    # NumPy's compiled loops instead of making several thousand tiny
    # ``norm``/``cross``/``dot`` calls for every ECG.
    delta = ma[:, None, :] - mb[None, :, :]
    norms = np.linalg.norm(delta, axis=2)
    cross_products = np.cross(da[:, None, :], db[None, :, :], axis=2)
    numerators = np.sum(delta * cross_products, axis=2)
    usable = norms > 1e-6
    if self_pair:
        usable &= np.triu(
            np.ones((len(da), len(db)), dtype=bool),
            k=2,
        )
        usable[0, len(db) - 1] = False
    if not np.any(usable):
        return None
    terms = numerators[usable] / norms[usable] ** 3
    signed_total = float(np.sum(terms))
    absolute_total = float(np.sum(np.abs(terms)))
    multiplier = 2.0 if self_pair else 1.0
    scale = multiplier / (4.0 * np.pi)
    return {
        "signed": float(scale * signed_total),
        "absolute": float(scale * absolute_total),
    }


def _gauss_metric(
    first_segment: str,
    second_segment: str,
    *,
    self_pair: bool,
    absolute: bool,
) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        key = ("gauss", first_segment, second_segment, self_pair)
        values = _memo(
            ctx,
            key,
            lambda: _gauss_pair_integrals(
                _segment(ctx, first_segment),
                _segment(ctx, second_segment),
                self_pair=self_pair,
            ),
        )
        if values is None:
            return _abstain("loop geometry is degenerate for Gauss quadrature")
        return FeatureValue(values["absolute" if absolute else "signed"])

    return calculate


def _graph_spectrum(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    """Rotation/translation/scale-invariant 4-NN loop graph spectrum."""

    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, segment)
        if points is None or len(points) < 8:
            return None
        points = points - np.mean(points, axis=0, keepdims=True)
        scale = float(np.sqrt(np.mean(np.sum(points**2, axis=1))))
        if scale <= EPS:
            return None
        points = _resample_curve(points / scale, 64)
        if points is None:
            return None
        pairwise = distance.squareform(distance.pdist(points))
        n = len(points)
        adjacency = np.zeros((n, n), dtype=float)
        neighbours = np.argsort(pairwise, axis=1)[:, 1:5]
        local = pairwise[np.arange(n)[:, None], neighbours]
        sigma = float(np.median(local[local > EPS])) if np.any(local > EPS) else 1.0
        for row in range(n):
            columns = neighbours[row]
            adjacency[row, columns] = np.exp(-0.5 * (pairwise[row, columns] / sigma) ** 2)
        adjacency = np.maximum(adjacency, adjacency.T)
        degree = np.sum(adjacency, axis=1)
        if np.any(degree <= EPS):
            return None
        inverse = 1.0 / np.sqrt(degree)
        laplacian = np.eye(n) - inverse[:, None] * adjacency * inverse[None, :]
        eigenvalues = np.maximum(np.linalg.eigvalsh(laplacian), 0.0)
        positive = eigenvalues[eigenvalues > 1e-9]
        if len(positive) < 2:
            return None
        shares = positive / np.sum(positive)
        entropy = float(-np.sum(shares * np.log(shares)) / np.log(len(shares)))
        return {"fiedler": float(eigenvalues[1]), "spectral_entropy": entropy}

    return _memo(ctx, ("graph_spectrum", segment), build)


def _graph_metric(segment: str, metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _graph_spectrum(ctx, segment)
        if result is None:
            return _abstain(f"{segment} loop graph unavailable")
        return FeatureValue(result[metric])

    return calculate


def _activation_energy_metrics(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, segment)
        if points is None or len(points) < 6 or ctx.fs <= 0:
            return None
        velocity = np.gradient(points, 1.0 / ctx.fs, axis=0)
        energy = np.sum(velocity**2, axis=1)
        total = float(np.sum(energy))
        if total <= EPS:
            return None
        cumulative = np.cumsum(energy) / total
        quantiles = {}
        for quantile in (0.25, 0.50, 0.75):
            index = int(np.searchsorted(cumulative, quantile, side="left"))
            quantiles[f"t{int(100 * quantile)}"] = index / max(len(energy) - 1, 1)
        midpoint = len(energy) // 2
        early = float(np.sum(energy[:midpoint]))
        late = float(np.sum(energy[midpoint:]))
        quantiles["early_late_balance"] = (late - early) / total
        return quantiles

    return _memo(ctx, ("activation_energy", segment), build)


def _activation_energy_metric(segment: str, metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _activation_energy_metrics(ctx, segment)
        if result is None:
            return _abstain(f"{segment} activation energy unavailable")
        return FeatureValue(float(result[metric]))

    return calculate


def _octant_metrics(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    def build() -> Optional[dict[str, float]]:
        points = _segment(ctx, segment)
        if points is None or len(points) < 6:
            return None
        magnitude = np.linalg.norm(points, axis=1)
        keep = magnitude >= max(EPS, 0.05 * float(np.max(magnitude)))
        points = points[keep]
        if len(points) < 6:
            return None
        octants = (
            (points[:, 0] >= 0).astype(int)
            + 2 * (points[:, 1] >= 0).astype(int)
            + 4 * (points[:, 2] >= 0).astype(int)
        )
        counts = np.bincount(octants, minlength=8).astype(float)
        shares = counts[counts > 0] / np.sum(counts)
        occupancy_entropy = float(-np.sum(shares * np.log2(shares)) / 3.0)
        dominant_fraction = float(np.max(counts) / np.sum(counts))
        transitions = octants[:-1] * 8 + octants[1:]
        transition_counts = np.bincount(transitions, minlength=64).astype(float)
        transition_shares = transition_counts[transition_counts > 0] / np.sum(transition_counts)
        transition_entropy = (
            float(-np.sum(transition_shares * np.log2(transition_shares)) / 6.0)
            if len(transition_shares)
            else 0.0
        )
        return {
            "occupancy_entropy": occupancy_entropy,
            "dominant_fraction": dominant_fraction,
            "transition_entropy": transition_entropy,
        }

    return _memo(ctx, ("octant_metrics", segment), build)


def _octant_metric(segment: str, metric: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        result = _octant_metrics(ctx, segment)
        if result is None:
            return _abstain(f"{segment} directional octants unavailable")
        return FeatureValue(result[metric])

    return calculate


def _plane_normal_flip_fraction(ctx: FeatureContext, segment: str) -> FeatureValue:
    points = _segment(ctx, segment)
    if points is None or len(points) < 8:
        return _abstain(f"{segment} loop unavailable")
    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= EPS:
        return _abstain(f"{segment} loop has zero variance")
    normal = vectors[:, 0]
    tangents = np.diff(points, axis=0)
    local_normals = np.cross(tangents[:-1], tangents[1:])
    norms = np.linalg.norm(local_normals, axis=1)
    valid = norms > max(EPS, 1e-4 * float(np.max(norms)))
    signs = np.sign(local_normals[valid] @ normal)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return _abstain(f"{segment} local plane normals are degenerate")
    return FeatureValue(float(np.mean(signs[1:] != signs[:-1])))


def _multiscale_curvature_count(ctx: FeatureContext, segment: str) -> FeatureValue:
    points = _segment(ctx, segment)
    if points is None or len(points) < 10:
        return _abstain(f"{segment} loop unavailable")
    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    sampled = _resample_curve(centered / scale, 96) if scale > EPS else None
    if sampled is None:
        return _abstain(f"{segment} loop is degenerate")
    counts = []
    for window in (5, 9, 17):
        smooth = sp_signal.savgol_filter(sampled, window, 2, axis=0, mode="interp")
        tangent = np.diff(smooth, axis=0)
        norms = np.linalg.norm(tangent, axis=1)
        valid = norms > EPS
        unit = tangent[valid] / norms[valid, None]
        if len(unit) < 4:
            counts.append(0)
            continue
        angles = np.degrees(
            np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0))
        )
        prominence = max(2.0, 0.25 * float(np.percentile(angles, 75)))
        peaks, _ = sp_signal.find_peaks(angles, prominence=prominence, distance=3)
        counts.append(int(len(peaks)))
    return FeatureValue(float(min(counts)))


def _peak_area_angle(segment: str) -> Callable[[FeatureContext], FeatureValue]:
    def calculate(ctx: FeatureContext) -> FeatureValue:
        peak = _peak_vector(ctx, segment)
        area = _area_vector_mv_ms(ctx, segment)
        angle = _angle_between(peak, None if area is None else area[0])
        return _abstain(f"{segment} peak or area vector unavailable") if angle is None else FeatureValue(angle)

    return calculate


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
) -> DiscoveryV2Spec:
    return DiscoveryV2Spec(
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
    )


def _build_specs() -> tuple[DiscoveryV2Spec, ...]:
    specs: list[DiscoveryV2Spec] = []

    # Scalarization of already-cached physiologic vector foundations.
    for segment in ("qrs", "p", "t", "st"):
        refs = (REF_QRS_AREA, REF_QRS_AREA_CRT) if segment == "qrs" else (REF_GEH_METHOD,)
        evidence = "B" if segment in {"qrs", "t"} else "D"
        for component in ("x", "y", "z", "magnitude", "azimuth", "elevation"):
            angular = component in {"azimuth", "elevation"}
            specs.append(_spec(
                f"{segment}_area_vector_{component}_{'deg' if angular else 'mv_ms'}",
                "Discovery-v2 area-vector scalars",
                f"Scalar {component} of the baseline-referenced {segment.upper()} area vector.",
                (
                    f"{component}(A_{segment}), A_{segment}=integral_{segment} r(t) dt"
                    if angular
                    else (
                        f"||A_{segment}||, A_{segment}=integral_{segment} r(t) dt"
                        if component == "magnitude"
                        else f"A_{segment},{component}=integral_{segment} {component}(t) dt"
                    )
                ),
                "degrees" if angular else "mV*ms",
                (segment,),
                evidence,
                (
                    "Direct activation/CRT validation exists for QRS area; other segment/component outputs are prospective extensions."
                    if segment == "qrs"
                    else "Component-wise scalarization is physiologically interpretable but not independently validated as a clinical marker."
                ),
                refs,
                _area_component(segment, component),
            ))

    for component in ("x", "y", "z", "magnitude", "azimuth", "elevation"):
        angular = component in {"azimuth", "elevation"}
        specs.append(_spec(
            f"peak_svg_{component}_{'deg' if angular else 'mv'}",
            "Discovery-v2 peak spatial ventricular gradient",
            f"{component} of peak SVG, the sum of peak QRS and peak T vectors.",
            f"{component}(SVG_peak), SVG_peak=QRS_peak+T_peak",
            "degrees" if angular else "mV",
            ("qrs", "t"),
            "B",
            "Established GEH construction; this exact implementation still requires transform-specific external replication.",
            (REF_GEH_METHOD,),
            _peak_svg_component(component),
        ))
    specs.extend((
        _spec(
            "peak_area_svg_angle_deg",
            "Discovery-v2 peak/area discordance",
            "Angle between peak SVG and area SVG.",
            "acos(SVG_peak dot SVG_area / (||SVG_peak|| ||SVG_area||))",
            "degrees", ("qrs", "t"), "D",
            "Prospective comparison of two established GEH constructions.",
            (REF_GEH_METHOD,), _peak_area_svg_angle,
        ),
        _spec(
            "peak_to_area_mean_svg_magnitude_ratio",
            "Discovery-v2 peak/area discordance",
            "Peak SVG magnitude divided by QRST-duration-normalized area SVG magnitude.",
            "||SVG_peak|| / (||SVG_area|| / duration_QRST)",
            "dimensionless", ("qrs", "t"), "D",
            "Prospective amplitude-shape discordance; no validated clinical threshold.",
            (REF_GEH_METHOD,), _peak_area_svg_magnitude_ratio,
        ),
    ))

    # True eight-independent-lead SVD markers.
    for segment, prefix in (("t", "lead8_t_wave_residuum"), ("qrs", "lead8_qrs_nondipolar")):
        dependencies = (segment,)
        descriptions = {
            "nondipolar_fraction": ("fraction", "sum(lambda4..lambda8) / sum(lambda1..lambda8)"),
            "nondipolar_energy": ("mV^2", "sum(lambda4..lambda8)"),
            "sigma2_sigma1": ("dimensionless", "sqrt(lambda2/lambda1)"),
            "effective_rank": ("dimensionless", "exp(-sum(p_i log(p_i))), p_i=lambda_i/sum(lambda)"),
        }
        for metric, (unit, formula) in descriptions.items():
            suffix = {
                "nondipolar_fraction": "fraction",
                "nondipolar_energy": "energy_mv2",
                "sigma2_sigma1": "sigma2_sigma1",
                "effective_rank": "effective_rank",
            }[metric]
            specs.append(_spec(
                f"{prefix}_{suffix}",
                "Discovery-v2 true 8-lead nondipolarity",
                f"{segment.upper()} {metric.replace('_', ' ')} from SVD of I, II and V1-V6 only.",
                formula, unit, dependencies, "B" if segment == "t" else "C",
                "T-wave residuum has prior clinical literature; the homologous QRS measure is less established and remains exploratory.",
                (REF_TWR_METHOD, REF_NONDIPOLAR),
                _lead8_metric(segment, metric),
            ))

    # VTI additions only; historical magnitude_area_qrs/qrst already provide
    # the QRS and QRST 3D VTIs, so aliases are intentionally not duplicated.
    for segment in ("p", "t", "st"):
        specs.append(_spec(
            f"{segment}_vti_3d_mv_ms",
            "Discovery-v2 3D voltage-time integral",
            f"Integral of instantaneous 3D vector magnitude over {segment.upper()}.",
            f"integral_{segment} sqrt(X(t)^2+Y(t)^2+Z(t)^2) dt",
            "mV*ms", (segment,), "C" if segment == "t" else "D",
            "QRS VTI has cohort evidence; P/T/ST extensions are prospective and intentionally separate from SAI.",
            (REF_VTI,), _vti_calculator(segment),
        ))

    specs.append(_spec(
        "vg_rvpo_mv_ms",
        "Discovery-v2 ventricular-gradient RV pressure-overload projection",
        "Published ventricular-gradient projection optimized for right-ventricular pressure overload.",
        "-0.807526151172*VGx + 0.404508497187*VGy + 0.376555628452*VGz",
        "mV*ms", ("qrs", "t"), "B",
        "Validated in selected RV-pressure-load cohorts; transform and population dependence preclude a universal cutoff.",
        (REF_VG_RVPO,), _vg_rvpo,
    ))

    for metric in ("mean", "median", "p90", "max", "resultant_length"):
        tt_evidence = "A" if metric == "mean" else ("C" if metric != "resultant_length" else "D")
        tt_note = (
            "Published mean spatial TT-prime has reproducibility plus ICD and community outcome evidence; "
            "this implementation performs beat-local crosspoint translation but no respiration-derived rotation."
            if metric == "mean"
            else (
                "The underlying TT-prime angles are established, but this distributional summary has not been independently validated."
                if metric != "resultant_length"
                else "Prospective directional-concentration companion to TT-prime; no independent clinical validation."
            )
        )
        specs.append(_spec(
            f"tt_prime_{metric}_{'deg' if metric != 'resultant_length' else 'fraction'}",
            "Discovery-v2 beat-to-beat repolarization direction",
            f"{metric.replace('_', ' ')} of consecutive beat T-peak vector directions.",
            (
                "summary_i acos(T_i dot T_{i+1}/(||T_i|| ||T_{i+1}||))"
                if metric != "resultant_length"
                else "||mean_i(T_i/||T_i||)||"
            ),
            "degrees" if metric != "resultant_length" else "fraction",
            ("t",), tt_evidence,
            tt_note,
            (REF_TT_PRIME, REF_DYNAMIC_VCG), _tt_prime_metric(metric),
        ))

    cloud_units = {
        "rms_radius": "mV", "max_radius": "mV",
        "ellipsoid_volume": "mV^3", "convex_hull_volume": "mV^3",
        "anisotropy": "fraction",
    }
    cloud_name_units = {
        "rms_radius": "mv", "max_radius": "mv",
        "ellipsoid_volume": "mv3", "convex_hull_volume": "mv3",
        "anisotropy": "fraction",
    }
    for marker, label in (("r_peak", "r"), ("t_peak", "t")):
        for metric, unit in cloud_units.items():
            specs.append(_spec(
                f"{label}_peak_cloud_{metric}_{cloud_name_units[metric]}",
                "Discovery-v2 R/T peak clouds",
                f"{metric.replace('_', ' ')} of beatwise {label.upper()}-peak vectors.",
                {
                    "rms_radius": "sqrt(mean_i ||p_i-centroid(P)||^2)",
                    "max_radius": "max_i ||p_i-centroid(P)||",
                    "ellipsoid_volume": "4*pi/3*sqrt(lambda1*lambda2*lambda3)",
                    "convex_hull_volume": "volume(convex_hull(P))",
                    "anisotropy": "1-lambda3/lambda1",
                }[metric],
                unit, ("qrs",) if marker == "r_peak" else ("t",), "B" if metric == "convex_hull_volume" else "D",
                "Convex-hull peak-cloud volume has prospective ICD-cohort evidence; auxiliary cloud descriptors are new robustness/visualization hypotheses.",
                (REF_PEAK_CLOUD, REF_DYNAMIC_VCG), _cloud_metric(marker, metric),
            ))
    for metric, suffix in (("convex_hull_volume", "convex_hull_volume_ratio"), ("ellipsoid_volume", "ellipsoid_volume_ratio"), ("rms_radius", "rms_radius_ratio")):
        specs.append(_spec(
            f"t_to_r_peak_cloud_{suffix}",
            "Discovery-v2 R/T peak clouds",
            f"Ratio of T-cloud to R-cloud {metric.replace('_', ' ')}.",
            f"T_cloud_{metric}/R_cloud_{metric}", "dimensionless", ("qrs", "t"),
            "B" if metric == "convex_hull_volume" else "D",
            "T/R convex-hull volume ratio predicted ventricular arrhythmia in an ICD cohort; alternatives are exploratory.",
            (REF_PEAK_CLOUD,), _cloud_ratio(metric),
        ))

    specs.extend((
        _spec(
            "qrs_t_interloop_distance_representative_mv",
            "Discovery-v2 QRS-T interloop distance",
            "Minimum Euclidean separation of representative QRS and T loops.",
            "min_{q in QRS,t in T} ||q-t||", "mV", ("qrs", "t"), "C",
            "Dynamic VCG literature defines interloop distance in a normative cohort; clinical outcome validation remains limited.",
            (REF_DYNAMIC_VCG,), _representative_interloop,
        ),
        _spec(
            "qrs_t_interloop_distance_mean_mv",
            "Discovery-v2 QRS-T interloop distance",
            "Mean beatwise minimum QRS-T interloop distance.",
            "mean_b min_{q in QRS_b,t in T_b} ||q-t||", "mV", ("qrs", "t"), "C",
            "Published beatwise dynamic-VCG measure with normative-cohort support but no transportable clinical cutoff.",
            (REF_DYNAMIC_VCG,), _interloop_metric("mean"),
        ),
        _spec(
            "qrs_t_interloop_distance_sd_mv",
            "Discovery-v2 QRS-T interloop distance",
            "Sample SD of beatwise minimum QRS-T interloop distance.",
            "sd_b min_{q in QRS_b,t in T_b} ||q-t||", "mV", ("qrs", "t"), "D",
            "Variability extension; no independent prognostic validation is claimed.",
            (REF_DYNAMIC_VCG,), _interloop_metric("sd"),
        ),
    ))
    for metric, unit, name_unit in (("centroid_distance", "mV", "mv"), ("chamfer_distance", "mV", "mv"), ("normalized_chamfer", "dimensionless", "dimensionless"), ("peak_distance", "mV", "mv")):
        specs.append(_spec(
            f"qrs_t_{metric}_{name_unit}",
            "Discovery-v2 QRS-T interloop geometry",
            f"{metric.replace('_', ' ')} between representative QRS and T loops.",
            {
                "centroid_distance": "||centroid(QRS)-centroid(T)||",
                "chamfer_distance": "0.5*(mean_q min_t ||q-t|| + mean_t min_q ||t-q||)",
                "normalized_chamfer": "Chamfer(QRS,T)/sqrt(rms_radius_QRS^2+rms_radius_T^2)",
                "peak_distance": "||QRS_peak-T_peak||",
            }[metric], unit, ("qrs", "t"), "D",
            "Geometrically explicit visual hypothesis; no clinical threshold or independent validation.",
            (REF_DYNAMIC_VCG, REF_BROAD_VCG), _interloop_geometry_metric(metric),
        ))

    for segment in ("p", "qrs", "t", "st"):
        specs.append(_spec(
            f"{segment}_normalized_open_loop_gap",
            "Discovery-v2 visually open loops",
            f"Start-to-end gap as a fraction of {segment.upper()} path length.",
            "||r_end-r_start|| / sum_i ||r_i-r_{i-1}||", "fraction", (segment,), "D",
            "Directly visible closure defect; normalization removes amplitude and sampling-density scale.",
            (REF_BROAD_VCG,), _closure_ratio(segment),
        ))

    for segment in ("qrs", "t"):
        for plane in PLANES:
            for metric in ("orientation_balance", "signed_area_cancellation"):
                specs.append(_spec(
                    f"{segment}_{plane}_{metric}",
                    "Discovery-v2 visible lobes and figure-eight loops",
                    f"{metric.replace('_', ' ')} of the {segment.upper()} loop in {plane}.",
                    (
                        "sum(signed triangle areas)/sum(abs triangle areas)"
                        if metric == "orientation_balance"
                        else "1-abs(sum(signed triangle areas))/sum(abs triangle areas)"
                    ),
                    "fraction", (segment,), "D",
                    "Rotation reversals and figure-eight lobes are visually apparent; this exact normalized scalar is prospective.",
                    (REF_LOOP_MORPHOLOGY, REF_BROAD_VCG),
                    _projected_orientation_calculator(segment, plane, metric),
                ))
        for metric in ("signed", "absolute"):
            specs.append(_spec(
                f"{segment}_direction_{metric}_solid_angle_sr",
                "Discovery-v2 spherical direction path",
                f"{metric} solid angle swept by the {segment.upper()} vector direction on the unit sphere.",
                "sum spherical_triangle_area(unit(r(t)))",
                "steradian", (segment,), "D",
                "Rotation- and amplitude-scale invariant mathematical hypothesis; no clinical validation.",
                (REF_BROAD_VCG,), _solid_angle_metric(segment, metric),
            ))
        specs.extend((
            _spec(
                f"{segment}_gauss_writhe_proxy",
                "Discovery-v2 3D folding and entanglement",
                f"Signed discrete Gauss self-integral of the closed {segment.upper()} trajectory.",
                "(1/4pi) double_integral ((r-r') dot (dr x dr'))/||r-r'||^3",
                "dimensionless", (segment,), "D",
                "Rotation/translation/scale-invariant midpoint quadrature proxy, not a validated topological diagnosis.",
                (REF_LOOP_MORPHOLOGY,), _gauss_metric(segment, segment, self_pair=True, absolute=False),
            ),
            _spec(
                f"{segment}_gauss_absolute_entanglement",
                "Discovery-v2 3D folding and entanglement",
                f"Absolute discrete Gauss self-integral of the closed {segment.upper()} trajectory.",
                "(1/4pi) double_integral abs(((r-r') dot (dr x dr'))/||r-r'||^3)",
                "dimensionless", (segment,), "D",
                "Mathematical visual-complexity hypothesis; sensitive to close approaches and closure convention.",
                (REF_LOOP_MORPHOLOGY,), _gauss_metric(segment, segment, self_pair=True, absolute=True),
            ),
            _spec(
                f"{segment}_loop_graph_fiedler_value",
                "Discovery-v2 loop graph spectrum",
                f"Second eigenvalue of the normalized 4-NN graph Laplacian for {segment.upper()}.",
                "lambda2(I-D^-1/2 W D^-1/2)", "dimensionless", (segment,), "D",
                "Rotation/translation/scale-invariant shape hypothesis; no clinical validation.",
                (REF_LOOP_MORPHOLOGY,), _graph_metric(segment, "fiedler"),
            ),
            _spec(
                f"{segment}_loop_graph_spectral_entropy",
                "Discovery-v2 loop graph spectrum",
                f"Normalized entropy of positive normalized-Laplacian eigenvalues for {segment.upper()}.",
                "-sum(p_i log p_i)/log(n_positive)", "fraction", (segment,), "D",
                "Rotation/translation/scale-invariant shape-complexity hypothesis.",
                (REF_LOOP_MORPHOLOGY,), _graph_metric(segment, "spectral_entropy"),
            ),
            _spec(
                f"{segment}_plane_normal_flip_fraction",
                "Discovery-v2 visible loop folding",
                f"Fraction of local binormal sign flips relative to the global {segment.upper()} best-fit plane.",
                "mean(sign(b_i dot n_global) != sign(b_{i-1} dot n_global))",
                "fraction", (segment,), "C",
                "Loop folding/plane changes have small-cohort VT evidence; this exact robust fraction is exploratory.",
                (REF_LOOP_MORPHOLOGY,), lambda ctx, s=segment: _plane_normal_flip_fraction(ctx, s),
            ),
            _spec(
                f"{segment}_multiscale_curvature_peak_count",
                "Discovery-v2 persistent visible notches",
                f"Minimum curvature-peak count retained across three smoothing scales for {segment.upper()}.",
                "min_scale count(prominent local maxima of turning angle)",
                "count", (segment,), "C",
                "Notches/folding have prior morphology evidence; the scale-persistent count is a new robustness definition.",
                (REF_LOOP_MORPHOLOGY,), lambda ctx, s=segment: _multiscale_curvature_count(ctx, s),
            ),
            _spec(
                f"{segment}_peak_area_vector_angle_deg",
                "Discovery-v2 within-loop vector discordance",
                f"Angle between peak and area vectors of {segment.upper()}.",
                "acos(v_peak dot A / (||v_peak|| ||A||))",
                "degrees", (segment,), "D",
                "Visually interpretable peak-versus-net-force discordance; no independent validation.",
                (REF_GEH_METHOD,), _peak_area_angle(segment),
            ),
        ))
        for metric in ("t25", "t50", "t75", "early_late_balance"):
            specs.append(_spec(
                f"{segment}_activation_energy_{metric}_fraction",
                "Discovery-v2 activation-energy timing",
                f"{metric.replace('_', ' ')} of {segment.upper()} spatial-velocity-squared energy.",
                (
                    f"first normalized time where cumulative sum(||dr/dt||^2) reaches {metric[1:]}%"
                    if metric.startswith("t")
                    else "(E_late-E_early)/(E_late+E_early)"
                ),
                "fraction", (segment,), "D",
                "Rotation- and amplitude-scale invariant timing hypothesis; no clinical cutoff.",
                (REF_BROAD_VCG,), _activation_energy_metric(segment, metric),
            ))
        for metric in ("occupancy_entropy", "dominant_fraction", "transition_entropy"):
            specs.append(_spec(
                f"{segment}_octant_{metric}",
                "Discovery-v2 visual XYZ octant trajectory",
                f"{metric.replace('_', ' ')} of {segment.upper()} vector octants.",
                {
                    "occupancy_entropy": "H(octant(r(t)))/log2(8)",
                    "dominant_fraction": "max_o count(o)/N",
                    "transition_entropy": "H(octant_t -> octant_{t+1})/log2(64)",
                }[metric],
                "fraction", (segment,), "D",
                "Coordinate-frame-dependent but amplitude-scale-invariant visual trajectory hypothesis.",
                (REF_BROAD_VCG,), _octant_metric(segment, metric),
            ))

    specs.extend((
        _spec(
            "qrs_t_gauss_linking_proxy",
            "Discovery-v2 3D interloop entanglement",
            "Signed discrete Gauss linking integral between linearly closed QRS and T loops.",
            "(1/4pi) double_integral ((q-t) dot (dq x dt))/||q-t||^3",
            "dimensionless", ("qrs", "t"), "D",
            "Mathematical 3D relationship hypothesis; closure-dependent and not a topological invariant for native open loops.",
            (REF_LOOP_MORPHOLOGY,), _gauss_metric("qrs", "t", self_pair=False, absolute=False),
        ),
        _spec(
            "qrs_t_gauss_absolute_entanglement",
            "Discovery-v2 3D interloop entanglement",
            "Absolute discrete Gauss interaction between linearly closed QRS and T loops.",
            "(1/4pi) double_integral abs(((q-t) dot (dq x dt))/||q-t||^3)",
            "dimensionless", ("qrs", "t"), "D",
            "Continuous visual proximity/folding hypothesis; not an integer linking number.",
            (REF_LOOP_MORPHOLOGY,), _gauss_metric("qrs", "t", self_pair=False, absolute=True),
        ),
    ))

    names = [item.name for item in specs]
    if len(names) != len(set(names)):  # fail during import, not after a run
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"Duplicate Discovery-v2 biomarker names: {duplicates}")
    return tuple(specs)


DISCOVERY_V2_SPECS = _build_specs()


class DiscoveryV2Biomarker(BiomarkerBase):
    def __init__(self, spec: DiscoveryV2Spec):
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
        except Exception as exc:  # a failed hypothesis must not abort the panel
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
        return BiomarkerResult(
            name=self.name,
            value=None if value is None else float(value),
            unit=self.unit,
            category=self.category,
            confidence=float(confidence),
            per_lead=per_lead,
            description=self.description,
            references=self.references,
            computation_notes=notes,
            family=self.spec.family,
            formula=self.spec.formula,
            implementation_status="implemented",
            pathological_direction="data_driven",
            pathological_note=(
                f"Prospective Discovery-v2 hypothesis; evidence {self.spec.evidence_level}. "
                f"{self.spec.evidence_note} Direction and threshold must be learned in grouped "
                "training data and independently replicated."
            ),
        )


def register_discovery_v2_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in DISCOVERY_V2_SPECS:
        if spec.name in existing:
            if isinstance(BiomarkerRegistry.get(spec.name), DiscoveryV2Biomarker):
                continue
            raise RuntimeError(f"Discovery-v2 marker collides with an existing biomarker: {spec.name}")
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=DISCOVERY_V2_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    f"Prospective Discovery-v2 hypothesis; evidence {spec.evidence_level}. "
                    f"{spec.evidence_note} No diagnostic cutoff is claimed."
                ),
                implementation_status="implemented",
                references=list(spec.references),
                notes=(
                    f"{DISCOVERY_V2_VERSION}; post-development prospective hypothesis; "
                    "Discovery Reserve only; no Core/Sentinel eligibility."
                ),
            )
        )
        BiomarkerRegistry.register(DiscoveryV2Biomarker(spec))
        existing.add(spec.name)


__all__ = [
    "DISCOVERY_V2_SOURCE",
    "DISCOVERY_V2_SPECS",
    "DISCOVERY_V2_VERSION",
    "DiscoveryV2Biomarker",
    "DiscoveryV2Spec",
    "INDEPENDENT_8_LEADS",
    "register_discovery_v2_biomarkers",
]
