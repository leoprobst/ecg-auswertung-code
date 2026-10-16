from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.spatial import distance

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
    _context,
    _dtw_distance,
    _memoized_primitive,
    _path_length,
    _pca,
    _resample_curve,
    _shape_normalize,
    _trapz,
)
from ..data_loader import LEAD_MAP
from ..delineation.base import (
    RecordDelineation,
    RhythmClass,
    WaveStatus,
    fiducial_confidence_threshold,
    measurement_confidence_threshold,
)
from ..vcg_transform import (
    VCGTransformRegistry,
    aggregate_loop_axes,
    axis_angle_deg,
    loop_dominant_axis,
)
from ..vcg_geometry import sum_absolute_integral_mv_ms


ADVANCED_VCG_VERSION = "post-development-vcg-v1.5.0-atrial-axis-quality"
ADVANCED_VCG_SOURCE = "Project physiology audit, 2026-07"
ADVANCED_VCG_REFERENCES = (
    "https://pubmed.ncbi.nlm.nih.gov/22516167/",
    "https://pubmed.ncbi.nlm.nih.gov/23166683/",
    "https://pubmed.ncbi.nlm.nih.gov/33537064/",
    "https://pubmed.ncbi.nlm.nih.gov/39322999/",
)


@dataclass(frozen=True)
class _AdvancedSpec:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    dependencies: tuple[str, ...]
    calculator: Callable[[FeatureContext], FeatureValue]
    references: tuple[str, ...] = ()


DIRECT_QRS_T_REFERENCE = "https://doi.org/10.1038/s41598-026-37361-w"
DIRECT_QRS_T_LEADS: tuple[str, ...] = (
    "I",
    "II",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


def direct_qrs_t_angles_8lead(
    ecg_12lead: np.ndarray,
    *,
    qrs_bounds: tuple[int, int],
    t_bounds: tuple[int, int],
    baseline_bounds: tuple[int, int],
) -> tuple[Optional[float], Optional[float]]:
    """Direct Area and Integral QRS-T angles in the native 8-lead space.

    The 2026 method treats I, II and V1-V6 as an eight-dimensional signal.
    ``Area`` is the angle between the signed per-lead QRS and T integrals.
    ``Integral`` is the magnitude-weighted mean angle over every QRS/T sample
    pair. The latter keeps within-wave morphology instead of reducing each
    wave to one area vector.

    Bounds are inclusive and a measured isoelectric window is mandatory. A
    guessed zero baseline would make the result dependent on acquisition
    offset and is therefore rejected rather than silently substituted.
    """

    signal = np.asarray(ecg_12lead, dtype=float)
    if signal.ndim != 2 or signal.shape[1] < len(LEAD_MAP):
        return None, None
    indices = [LEAD_MAP[lead] for lead in DIRECT_QRS_T_LEADS]
    q0, q1 = (int(value) for value in qrs_bounds)
    t0, t1 = (int(value) for value in t_bounds)
    b0, b1 = (int(value) for value in baseline_bounds)
    if (
        q0 < 0
        or t0 < 0
        or b0 < 0
        or q1 < q0
        or t1 < t0
        or b1 < b0
        or max(q1, t1, b1) >= len(signal)
    ):
        return None, None
    baseline_window = signal[b0 : b1 + 1, indices]
    qrs = signal[q0 : q1 + 1, indices]
    t_wave = signal[t0 : t1 + 1, indices]
    if (
        len(baseline_window) < 2
        or len(qrs) < 2
        or len(t_wave) < 2
        or not np.isfinite(baseline_window).all()
        or not np.isfinite(qrs).all()
        or not np.isfinite(t_wave).all()
    ):
        return None, None
    baseline = np.mean(baseline_window, axis=0)
    qrs = qrs - baseline
    t_wave = t_wave - baseline

    # Uniform sampling means the common dt factors cancel from both cosines.
    qrs_area = np.trapezoid(qrs, axis=0)
    t_area = np.trapezoid(t_wave, axis=0)
    numerator = float(np.dot(qrs_area, t_area))
    area_denominator = float(np.linalg.norm(qrs_area) * np.linalg.norm(t_area))
    if area_denominator <= EPS:
        area_angle = None
    else:
        area_angle = float(
            np.degrees(np.arccos(np.clip(numerator / area_denominator, -1.0, 1.0)))
        )

    # Rehor et al.'s Integral method averages the *pairwise angles*, weighted
    # by both vector magnitudes.  ``acos(weighted mean cosine)`` is not the
    # same quantity because acos is nonlinear; the previous implementation
    # made precisely that substitution.  Trapezoidal sample weights implement
    # the published dual integral.  Matrix multiplication keeps the expensive
    # eight-dimensional dot products in compiled code.
    qrs_magnitude = np.linalg.norm(qrs, axis=1)
    t_magnitude = np.linalg.norm(t_wave, axis=1)
    qrs_weights = qrs_magnitude.copy()
    t_weights = t_magnitude.copy()
    qrs_weights[[0, -1]] *= 0.5
    t_weights[[0, -1]] *= 0.5
    integral_denominator = float(qrs_weights.sum() * t_weights.sum())
    if integral_denominator <= EPS:
        integral_angle = None
    else:
        qrs_unit = np.divide(
            qrs,
            qrs_magnitude[:, None],
            out=np.zeros_like(qrs),
            where=qrs_magnitude[:, None] > EPS,
        )
        t_unit = np.divide(
            t_wave,
            t_magnitude[:, None],
            out=np.zeros_like(t_wave),
            where=t_magnitude[:, None] > EPS,
        )
        pairwise_angles = np.arccos(
            np.clip(qrs_unit @ t_unit.T, -1.0, 1.0)
        )
        weighted_angle_sum = float(
            qrs_weights @ pairwise_angles @ t_weights
        )
        integral_angle = float(
            np.degrees(weighted_angle_sum / integral_denominator)
        )
    return area_angle, integral_angle


def _direct_qrs_t_angles(ctx: FeatureContext) -> tuple[Optional[float], Optional[float]]:
    cache_key = ("direct_qrs_t_angles_8lead",)
    cached = ctx._primitive_cache.get(cache_key)
    if cached is not None:
        return cached
    if ctx.ecg_12lead is None:
        return None, None
    qrs_bounds = ctx.bounds("qrs")
    t_bounds = ctx.bounds("t")
    baseline_bounds = ctx.bounds("pr")
    if qrs_bounds is None or t_bounds is None or baseline_bounds is None:
        return None, None
    # The publication integrates the repolarisation vector from J (QRS
    # offset), not merely from the visually delineated T-wave onset.
    result = direct_qrs_t_angles_8lead(
        ctx.ecg_12lead,
        qrs_bounds=qrs_bounds,
        t_bounds=(qrs_bounds[1], t_bounds[1]),
        baseline_bounds=baseline_bounds,
    )
    ctx._primitive_cache[cache_key] = result
    return result


def _direct_qrs_t_abstention_reason(ctx: FeatureContext) -> str:
    if ctx.delineation.rhythm.discrete_p_not_evaluable:
        return "rhythm_restricts_pr_baseline"
    signal = ctx.ecg_12lead
    if signal is None:
        return "signal_unavailable"
    if signal.ndim != 2 or signal.shape[1] < len(LEAD_MAP):
        return "required_leads_missing"
    for segment, reason in (("qrs", "qrs_boundaries"), ("t", "t_boundaries"), ("pr", "pr_baseline_boundaries")):
        bounds = ctx.bounds(segment)
        if bounds is None:
            return reason
        if not ctx.is_segment_evaluable(segment):
            return f"{segment}_quality"
        start, end = bounds
        if not 0 <= start < end < len(signal):
            return reason
        if not np.isfinite(signal[start:end + 1, [LEAD_MAP[x] for x in DIRECT_QRS_T_LEADS]]).all():
            return "nonfinite_required_lead_samples"
    return "numerical_vector_degeneration"


def _qrs_t_area_angle_direct_8lead(ctx: FeatureContext) -> FeatureValue:
    area, _ = _direct_qrs_t_angles(ctx)
    return FeatureValue(
        area,
        confidence=0.9 if area is not None else 0.0,
        notes=(
            "direct native-lead Area method; I, II, V1-V6; PR baseline"
            if area is not None
            else f"not_evaluable: reason_code={_direct_qrs_t_abstention_reason(ctx)}; direct 8-lead QRS-T Area angle requires finite "
            "I, II, V1-V6 and evaluable QRS, T and PR baseline windows"
        ),
    )


def _qrs_t_integral_angle_direct_8lead(ctx: FeatureContext) -> FeatureValue:
    _, integral = _direct_qrs_t_angles(ctx)
    return FeatureValue(
        integral,
        confidence=0.9 if integral is not None else 0.0,
        notes=(
            "direct native-lead Integral method; I, II, V1-V6; PR baseline"
            if integral is not None
            else f"not_evaluable: reason_code={_direct_qrs_t_abstention_reason(ctx)}; direct 8-lead QRS-T Integral angle requires "
            "finite I, II, V1-V6 and evaluable QRS, T and PR baseline windows"
        ),
    )


def _smooth_resampled(
    points: Optional[np.ndarray],
    *,
    n: int = 64,
) -> Optional[np.ndarray]:
    if points is None or len(points) < 4:
        return None
    values = np.asarray(points, dtype=float)
    window = min(11, len(values) if len(values) % 2 else len(values) - 1)
    if window >= 5:
        values = sp_signal.savgol_filter(
            values,
            window_length=window,
            polyorder=min(2, window - 1),
            axis=0,
            mode="interp",
        )
    return _resample_curve(values, n)


def _smooth_resampled_segment(
    ctx: FeatureContext,
    segment: str,
    *,
    n: int = 64,
) -> Optional[np.ndarray]:
    return _memoized_primitive(
        ctx,
        ("advanced_smooth_resampled", segment, int(n)),
        lambda: _smooth_resampled(ctx.vcg_segment(segment), n=n),
    )


def _turning_angles(
    ctx: FeatureContext,
    segment: str,
) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        curve = _smooth_resampled_segment(ctx, segment, n=64)
        if curve is None:
            return None
        tangent = np.diff(curve, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        valid = norm > EPS
        tangent = tangent[valid]
        norm = norm[valid]
        if len(tangent) < 3:
            return None
        unit = tangent / norm[:, None]
        cosine = np.clip(
            np.sum(unit[:-1] * unit[1:], axis=1),
            -1.0,
            1.0,
        )
        return np.degrees(np.arccos(cosine))

    return _memoized_primitive(
        ctx,
        ("advanced_turning_angles", segment),
        build,
    )


def _qrs_total_turning(ctx: FeatureContext) -> FeatureValue:
    angles = _turning_angles(ctx, "qrs")
    return (
        FeatureValue(None, 0.0)
        if angles is None
        else FeatureValue(float(np.sum(angles)))
    )


def _qrs_turning_p95(ctx: FeatureContext) -> FeatureValue:
    angles = _turning_angles(ctx, "qrs")
    return (
        FeatureValue(None, 0.0)
        if angles is None
        else FeatureValue(float(np.percentile(angles, 95)))
    )


def _window_energy_fraction(
    ctx: FeatureContext,
    *,
    segment: str,
    length_ms: float,
    end_aligned: bool,
) -> Optional[float]:
    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 2:
        return None
    samples = max(2, int(round(length_ms * ctx.fs / 1000.0)))
    window = points[-samples:] if end_aligned else points[:samples]
    total = float(_trapz(np.sum(points ** 2, axis=1), 1.0 / ctx.fs))
    partial = float(_trapz(np.sum(window ** 2, axis=1), 1.0 / ctx.fs))
    if total <= EPS:
        return None
    return float(np.clip(partial / total, 0.0, 1.0))


def _qrs_terminal_energy_fraction(ctx: FeatureContext) -> FeatureValue:
    return FeatureValue(
        _window_energy_fraction(
            ctx,
            segment="qrs",
            length_ms=20.0,
            end_aligned=True,
        )
    )


def _qrs_initial_terminal_energy_balance(ctx: FeatureContext) -> FeatureValue:
    initial = _window_energy_fraction(
        ctx,
        segment="qrs",
        length_ms=20.0,
        end_aligned=False,
    )
    terminal = _window_energy_fraction(
        ctx,
        segment="qrs",
        length_ms=20.0,
        end_aligned=True,
    )
    if initial is None or terminal is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(
        float((terminal - initial) / (terminal + initial + EPS))
    )


def _normalized_curves(
    ctx: FeatureContext,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    def build() -> Optional[tuple[np.ndarray, np.ndarray]]:
        qrs = _smooth_resampled_segment(ctx, "qrs")
        t_wave = _smooth_resampled_segment(ctx, "t")
        if qrs is None or t_wave is None:
            return None
        qrs = _shape_normalize(qrs)
        t_wave = _shape_normalize(t_wave)
        if qrs is None or t_wave is None:
            return None
        return qrs, t_wave

    return _memoized_primitive(ctx, ("advanced_normalized_curves",), build)


def _qrs_t_normalized_hausdorff(ctx: FeatureContext) -> FeatureValue:
    curves = _normalized_curves(ctx)
    if curves is None:
        return FeatureValue(None, 0.0)
    qrs, t = curves
    value = max(
        distance.directed_hausdorff(qrs, t)[0],
        distance.directed_hausdorff(t, qrs)[0],
    )
    return FeatureValue(float(value))


def _qrs_t_normalized_dtw(ctx: FeatureContext) -> FeatureValue:
    curves = _normalized_curves(ctx)
    if curves is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(_dtw_distance(*curves)))


def _mirror_phase_angles(ctx: FeatureContext) -> Optional[np.ndarray]:
    def build() -> Optional[np.ndarray]:
        qrs = _smooth_resampled_segment(ctx, "qrs")
        t_wave = _smooth_resampled_segment(ctx, "t")
        if qrs is None or t_wave is None:
            return None
        qrs_norm = np.linalg.norm(qrs, axis=1)
        t_norm = np.linalg.norm(t_wave, axis=1)
        valid = (
            (qrs_norm > max(EPS, 0.05 * float(np.max(qrs_norm))))
            & (t_norm > max(EPS, 0.05 * float(np.max(t_norm))))
        )
        if np.sum(valid) < 8:
            return None
        q_unit = qrs[valid] / qrs_norm[valid, None]
        mirrored_t_unit = -t_wave[valid] / t_norm[valid, None]
        cosine = np.clip(
            np.sum(q_unit * mirrored_t_unit, axis=1),
            -1.0,
            1.0,
        )
        return np.degrees(np.arccos(cosine))

    return _memoized_primitive(ctx, ("advanced_mirror_phase_angles",), build)


def _mirror_angle_median(ctx: FeatureContext) -> FeatureValue:
    values = _mirror_phase_angles(ctx)
    return (
        FeatureValue(None, 0.0)
        if values is None
        else FeatureValue(float(np.median(values)))
    )


def _mirror_angle_p90(ctx: FeatureContext) -> FeatureValue:
    values = _mirror_phase_angles(ctx)
    return (
        FeatureValue(None, 0.0)
        if values is None
        else FeatureValue(float(np.percentile(values, 90)))
    )


def _mirror_discordant_fraction(ctx: FeatureContext) -> FeatureValue:
    values = _mirror_phase_angles(ctx)
    return (
        FeatureValue(None, 0.0)
        if values is None
        else FeatureValue(float(np.mean(values > 90.0)))
    )


def _area_vector(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 2:
        return None
    return np.asarray(_trapz(points, 1.0 / ctx.fs, axis=0), dtype=float)


def _svg_cancellation(ctx: FeatureContext) -> FeatureValue:
    qrs = _area_vector(ctx, "qrs")
    t = _area_vector(ctx, "t")
    if qrs is None or t is None:
        return FeatureValue(None, 0.0)
    denominator = float(np.linalg.norm(qrs) + np.linalg.norm(t))
    if denominator <= EPS:
        return FeatureValue(None, 0.0)
    cancellation = 1.0 - float(np.linalg.norm(qrs + t)) / denominator
    return FeatureValue(float(np.clip(cancellation, 0.0, 1.0)))


def _t_terminal_path_fraction(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment("t")
    if points is None or len(points) < 5:
        return FeatureValue(None, 0.0)
    split = max(1, int(np.floor(0.80 * (len(points) - 1))))
    total = _path_length(points)
    if total <= EPS:
        return FeatureValue(None, 0.0)
    return FeatureValue(
        float(np.clip(_path_length(points[split:]) / total, 0.0, 1.0))
    )


def _direction_dispersion(
    points: Optional[np.ndarray],
) -> Optional[float]:
    curve = _smooth_resampled(points, n=32)
    if curve is None:
        return None
    tangent = np.diff(curve, axis=0)
    norms = np.linalg.norm(tangent, axis=1)
    tangent = tangent[norms > EPS]
    norms = norms[norms > EPS]
    if len(tangent) < 3:
        return None
    unit = tangent / norms[:, None]
    resultant = np.sum(unit, axis=0)
    if np.linalg.norm(resultant) <= EPS:
        return 180.0
    resultant /= np.linalg.norm(resultant)
    angles = np.degrees(
        np.arccos(np.clip(unit @ resultant, -1.0, 1.0))
    )
    return float(np.sqrt(np.mean(angles ** 2)))


def _t_terminal_direction_dispersion(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment("t")
    if points is None or len(points) < 5:
        return FeatureValue(None, 0.0)
    split = max(1, int(np.floor(0.80 * (len(points) - 1))))
    return FeatureValue(_direction_dispersion(points[split:]))


def _edge_direction(
    points: Optional[np.ndarray],
    *,
    from_start: bool,
    samples: int,
) -> Optional[np.ndarray]:
    if points is None or len(points) < 3:
        return None
    width = min(len(points), max(3, samples))
    window = points[:width] if from_start else points[-width:]
    vector = window[-1] - window[0]
    return vector if np.linalg.norm(vector) > EPS else None


def _st_t_transition_angle(ctx: FeatureContext) -> FeatureValue:
    support = max(3, int(round(20.0 * ctx.fs / 1000.0)))
    st_direction = _edge_direction(
        ctx.vcg_segment("st"),
        from_start=False,
        samples=support,
    )
    t_direction = _edge_direction(
        ctx.vcg_segment("t"),
        from_start=True,
        samples=support,
    )
    if st_direction is None or t_direction is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(_angle_between(st_direction, t_direction))


def _beat_bounds(
    ctx: FeatureContext,
    beat,
    segment: str,
) -> Optional[tuple[int, int]]:
    def sample(attribute: str) -> Optional[int]:
        point = getattr(beat, attribute, None)
        if point is None or getattr(point, "confidence", 1.0) < 0.35:
            return None
        return int(point.sample)

    if segment == "qrs":
        start = sample("r_onset")
        end = sample("r_offset")
    elif segment == "t":
        start = sample("t_onset")
        end = sample("t_offset")
    elif segment == "jt":
        # Published GEH T-area is the complete JT interval, not only the
        # visually delineated T wave.
        start = sample("r_offset")
        end = sample("t_offset")
    else:
        return None
    if (
        start is None
        or end is None
        or ctx._record_vcg_xyz is None
        or start < 0
        or end >= len(ctx._record_vcg_xyz)
        or end <= start
    ):
        return None
    return start, end


def _beat_area_vectors(
    ctx: FeatureContext,
    segment: str,
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return vectors
    for beat in ctx.delineation.beats:
        bounds = _beat_bounds(ctx, beat, segment)
        if bounds is None:
            continue
        values = record_vcg[bounds[0] : bounds[1] + 1]
        vector = np.asarray(
            _trapz(values, 1.0 / ctx.fs, axis=0),
            dtype=float,
        )
        if np.linalg.norm(vector) > EPS:
            vectors.append(vector)
    return vectors


def _beat_area_vectors_by_identity(
    ctx: FeatureContext,
    segment: str,
) -> dict[tuple[int, int], np.ndarray]:
    """Beat-keyed area vectors for safe cross-wave pairing.

    Independent QRS and T/JT lists cannot be zipped after invalid beats have
    been skipped: one missing T boundary shifts every subsequent pair.  The
    list index plus R-peak sample is a record-local immutable identity and
    makes the join explicit.  SVG directions are origin-dependent, so this
    path also fails closed unless the physiological origin is available.
    """

    record_vcg = ctx._record_vcg_xyz
    assessment = ctx.vcg_origin_assessment
    if record_vcg is None or assessment.primary is None:
        return {}
    vectors: dict[tuple[int, int], np.ndarray] = {}
    for beat_index, beat in enumerate(ctx.delineation.beats):
        bounds = _beat_bounds(ctx, beat, segment)
        if bounds is None:
            continue
        values = (
            record_vcg[bounds[0] : bounds[1] + 1]
            - assessment.primary.vector_millivolts[np.newaxis, :]
        )
        vector = np.asarray(
            _trapz(values, 1.0 / ctx.fs, axis=0),
            dtype=float,
        )
        if np.linalg.norm(vector) <= EPS:
            continue
        identity = (beat_index, int(beat.r_peak.sample))
        vectors[identity] = vector
    return vectors


def _axis_coherence(vectors: list[np.ndarray]) -> Optional[float]:
    if len(vectors) < 3:
        return None
    unit = np.asarray(
        [vector / np.linalg.norm(vector) for vector in vectors],
        dtype=float,
    )
    return float(np.clip(np.linalg.norm(np.mean(unit, axis=0)), 0.0, 1.0))


def _qrs_beat_area_axis_coherence(ctx: FeatureContext) -> FeatureValue:
    return FeatureValue(_axis_coherence(_beat_area_vectors(ctx, "qrs")))


def _t_beat_area_axis_coherence(ctx: FeatureContext) -> FeatureValue:
    return FeatureValue(_axis_coherence(_beat_area_vectors(ctx, "t")))


def _cluster_qrs_area_vectors(
    ctx: FeatureContext,
    cluster_id: int,
) -> list[np.ndarray]:
    """QRS area vectors restricted to one FusionDelineator morphology cluster.

    Mirrors ``_beat_area_vectors`` but filters on
    ``BeatDelineation.morphology_cluster_id`` (0 = primary/dominant, 1/2 =
    secondary/tertiary), so a distinct secondary population -- bigeminy,
    multifocal ectopy, intermittent aberrancy -- can be characterized on its
    own instead of only ever being excluded as an outlier.
    """
    vectors: list[np.ndarray] = []
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return vectors
    for beat in ctx.delineation.beats:
        if getattr(beat, "morphology_cluster_id", None) != cluster_id:
            continue
        bounds = _beat_bounds(ctx, beat, "qrs")
        if bounds is None:
            continue
        values = record_vcg[bounds[0] : bounds[1] + 1]
        vector = np.asarray(_trapz(values, 1.0 / ctx.fs, axis=0), dtype=float)
        if np.linalg.norm(vector) > EPS:
            vectors.append(vector)
    return vectors


def _secondary_morphology_beat_fraction(ctx: FeatureContext) -> FeatureValue:
    value = ctx.delineation.diagnostics.get("morphology_secondary_fraction")
    return FeatureValue(None, 0.0) if value is None else FeatureValue(float(value))


def _secondary_morphology_centroid_distance(ctx: FeatureContext) -> FeatureValue:
    value = ctx.delineation.diagnostics.get(
        "morphology_secondary_centroid_distance_mv"
    )
    return FeatureValue(None, 0.0) if value is None else FeatureValue(float(value))


def _secondary_morphology_mean_coupling_ratio(ctx: FeatureContext) -> FeatureValue:
    value = ctx.delineation.diagnostics.get(
        "morphology_secondary_mean_coupling_ratio"
    )
    return FeatureValue(None, 0.0) if value is None else FeatureValue(float(value))


def _secondary_morphology_frontal_axis_delta(ctx: FeatureContext) -> FeatureValue:
    primary = [vector[:2] for vector in _cluster_qrs_area_vectors(ctx, 0)]
    secondary = [vector[:2] for vector in _cluster_qrs_area_vectors(ctx, 1)]
    if len(primary) < 3 or len(secondary) < 3:
        return FeatureValue(None, 0.0)
    primary_mean = np.mean(primary, axis=0)
    secondary_mean = np.mean(secondary, axis=0)
    if (
        np.linalg.norm(primary_mean) <= EPS
        or np.linalg.norm(secondary_mean) <= EPS
    ):
        return FeatureValue(None, 0.0)
    return FeatureValue(_angle_between(primary_mean, secondary_mean))


def _secondary_morphology_frontal_axis(ctx: FeatureContext) -> FeatureValue:
    vectors = [vector[:2] for vector in _cluster_qrs_area_vectors(ctx, 1)]
    if len(vectors) < 3:
        return FeatureValue(None, 0.0)
    mean_vector = np.mean(vectors, axis=0)
    if np.linalg.norm(mean_vector) <= EPS:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.degrees(np.arctan2(mean_vector[1], mean_vector[0]))))


def _secondary_morphology_compensatory_pause_ratio(ctx: FeatureContext) -> FeatureValue:
    """Full-vs-incomplete compensatory pause after a candidate ectopic beat.

    A classic PVC does not reset the sinus node, so the RR interval before
    plus after it approximates two undisturbed sinus cycles (ratio ~1.0). A
    PAC typically resets the sinus node early, giving a distinctly shorter
    combined interval. Computed per qualifying ectopic beat from RR context
    already assigned in FusionDelineator._assign_beat_coupling_context, no
    new delineation fields needed. Exploratory hypothesis marker, not a
    PVC/PAC classifier (known exceptions exist, e.g. retrograde conduction).
    """
    beats = ctx.delineation.beats
    ratios: list[float] = []
    for index, beat in enumerate(beats):
        if getattr(beat, "morphology_cluster_id", None) not in (1, 2):
            continue
        coupling_ratio = beat.coupling_ratio
        rr_before = beat.rr_interval_to_previous_ms
        if (
            coupling_ratio is None
            or rr_before is None
            or coupling_ratio >= 0.85
            or index + 1 >= len(beats)
        ):
            continue
        rr_after = beats[index + 1].rr_interval_to_previous_ms
        if rr_after is None or coupling_ratio <= 0:
            continue
        baseline_rr = rr_before / coupling_ratio
        if baseline_rr <= 0:
            continue
        ratios.append((rr_before + rr_after) / (2.0 * baseline_rr))
    if not ratios:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.median(ratios)))


def _secondary_morphology_qt_delta(ctx: FeatureContext) -> FeatureValue:
    primary = _cluster_beats_or_accepted(ctx, 0)
    secondary = _cluster_beats_or_accepted(ctx, 1)
    fs = ctx.fs
    primary_qt = [
        beat.qt_interval_samples / fs * 1000.0
        for beat in primary
        if beat.qt_interval_samples is not None
    ]
    secondary_qt = [
        beat.qt_interval_samples / fs * 1000.0
        for beat in secondary
        if beat.qt_interval_samples is not None
    ]
    if len(primary_qt) < 3 or len(secondary_qt) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.mean(secondary_qt) - np.mean(primary_qt)))


def _secondary_morphology_qrs_duration_delta(ctx: FeatureContext) -> FeatureValue:
    primary = _cluster_beats_or_accepted(ctx, 0)
    secondary = _cluster_beats_or_accepted(ctx, 1)
    fs = ctx.fs
    primary_qrs = [
        beat.qrs_duration_samples / fs * 1000.0
        for beat in primary
        if beat.qrs_duration_samples is not None
    ]
    secondary_qrs = [
        beat.qrs_duration_samples / fs * 1000.0
        for beat in secondary
        if beat.qrs_duration_samples is not None
    ]
    if len(primary_qrs) < 3 or len(secondary_qrs) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.mean(secondary_qrs) - np.mean(primary_qrs)))


def _secondary_morphology_qrs_area_vector_magnitude_delta(
    ctx: FeatureContext,
) -> FeatureValue:
    primary = _cluster_qrs_area_vectors(ctx, 0)
    secondary = _cluster_qrs_area_vectors(ctx, 1)
    if len(primary) < 3 or len(secondary) < 3:
        return FeatureValue(None, 0.0)
    primary_mag = float(np.mean([np.linalg.norm(v) for v in primary]))
    secondary_mag = float(np.mean([np.linalg.norm(v) for v in secondary]))
    return FeatureValue(secondary_mag - primary_mag)


def _post_ectopic_primary_beat_indices(beats: list) -> set[int]:
    indices: set[int] = set()
    for index, beat in enumerate(beats):
        if getattr(beat, "morphology_cluster_id", None) not in (1, 2):
            continue
        if (
            index + 1 < len(beats)
            and getattr(beats[index + 1], "morphology_cluster_id", None) == 0
        ):
            indices.add(index + 1)
    return indices


def _beat_t_peak_vector(
    beat,
    record_vcg: np.ndarray,
) -> Optional[np.ndarray]:
    t_peak = getattr(beat, "t_peak", None)
    if (
        t_peak is None
        or t_peak.confidence < fiducial_confidence_threshold("t_peak")
        or beat.resolved_wave_status("t").value != "present"
        or not 0 <= t_peak.sample < len(record_vcg)
    ):
        return None
    vector = record_vcg[t_peak.sample]
    return vector if np.linalg.norm(vector) > EPS else None


def _post_ectopic_t_amplitude_delta(ctx: FeatureContext) -> FeatureValue:
    """Cardiac memory: T-wave amplitude change in the sinus beat right
    after an ectopic beat, versus non-post-ectopic primary-cluster beats.

    Rosenbaum et al. (Am J Cardiol 1982) described electrotonic T-wave
    modulation following abnormal ventricular activation. Needs only one
    captured ectopic-then-sinus pair; small-N is an inherent limit of a
    single 10s strip, made explicit via the low beat count rather than
    hidden behind averaging.
    """
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    beats = ctx.delineation.beats
    post_indices = _post_ectopic_primary_beat_indices(beats)
    if not post_indices:
        return FeatureValue(None, 0.0)
    post_values = [
        float(np.linalg.norm(vector))
        for vector in (
            _beat_t_peak_vector(beats[index], record_vcg)
            for index in sorted(post_indices)
        )
        if vector is not None
    ]
    baseline_values = [
        float(np.linalg.norm(vector))
        for index, beat in enumerate(beats)
        if getattr(beat, "morphology_cluster_id", None) == 0
        and index not in post_indices
        for vector in (_beat_t_peak_vector(beat, record_vcg),)
        if vector is not None
    ]
    if not post_values or len(baseline_values) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.mean(post_values) - np.mean(baseline_values)))


def _post_ectopic_t_axis_delta(ctx: FeatureContext) -> FeatureValue:
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    beats = ctx.delineation.beats
    post_indices = _post_ectopic_primary_beat_indices(beats)
    if not post_indices:
        return FeatureValue(None, 0.0)
    post_vectors = [
        vector
        for vector in (
            _beat_t_peak_vector(beats[index], record_vcg)
            for index in sorted(post_indices)
        )
        if vector is not None
    ]
    baseline_vectors = [
        vector
        for index, beat in enumerate(beats)
        if getattr(beat, "morphology_cluster_id", None) == 0
        and index not in post_indices
        for vector in (_beat_t_peak_vector(beat, record_vcg),)
        if vector is not None
    ]
    if not post_vectors or len(baseline_vectors) < 3:
        return FeatureValue(None, 0.0)
    post_mean = np.mean(post_vectors, axis=0)
    baseline_mean = np.mean(baseline_vectors, axis=0)
    if np.linalg.norm(post_mean) <= EPS or np.linalg.norm(baseline_mean) <= EPS:
        return FeatureValue(None, 0.0)
    return FeatureValue(_angle_between(post_mean, baseline_mean))


def _secondary_morphology_coupling_variability(ctx: FeatureContext) -> FeatureValue:
    """Fixed- vs. variable-coupling consistency of the secondary cluster.

    Classical reentrant extrasystoles couple to the preceding sinus beat at
    a near-fixed interval; a parasystolic focus (protected by entrance
    block) shows variable coupling instead (Pick, Circulation 1953). A high
    coefficient of variation is more consistent with a variable/
    parasystolic pattern, a low one with fixed reentrant coupling.
    """
    beats = _cluster_beats_or_accepted(ctx, 1)
    ratios = [beat.coupling_ratio for beat in beats if beat.coupling_ratio is not None]
    if len(ratios) < 3:
        return FeatureValue(None, 0.0)
    mean = float(np.mean(ratios))
    if mean <= 0:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.std(ratios, ddof=1) / mean))


def _secondary_morphology_common_denominator_fit(ctx: FeatureContext) -> FeatureValue:
    """How well inter-ectopic intervals fit integer multiples of a base cycle.

    Parasystole's "rule of common denominator": successive ectopic-to-
    ectopic intervals are approximately integer multiples of the
    parasystolic focus's own, sinus-independent cycle length. Uses the
    shortest observed inter-ectopic interval as the candidate base cycle,
    rounds every interval to its nearest integer multiple, and reports
    1 minus the mean relative residual (1.0 = perfect fit). This does not
    by itself rule out fixed-coupling reentry, which can also yield fairly
    regular inter-ectopic spacing under a regular sinus rate -- an
    explicitly exploratory marker, not a parasystole diagnosis.
    """
    beats = sorted(_cluster_beats_or_accepted(ctx, 1), key=lambda beat: beat.r_peak.sample)
    if len(beats) < 3:
        return FeatureValue(None, 0.0)
    fs = ctx.fs
    inter_ectopic_ms = [
        (beats[index + 1].r_peak.sample - beats[index].r_peak.sample) / fs * 1000.0
        for index in range(len(beats) - 1)
    ]
    base = min(inter_ectopic_ms)
    if base <= 0:
        return FeatureValue(None, 0.0)
    residuals = [
        abs(interval - max(1, round(interval / base)) * base) / base
        for interval in inter_ectopic_ms
    ]
    return FeatureValue(float(max(0.0, 1.0 - np.mean(residuals))))


def _octant_id_for_cluster(ctx: FeatureContext, cluster_id: int) -> Optional[float]:
    """Coarse 3D sign-octant (0-7) of a cluster's mean QRS area vector.

    Uses the same X/Y/Z sign bit-packing already used internally by
    <seg>_spherical_direction_entropy, but exposes it as a standalone
    coarse direction descriptor rather than folding it into an entropy
    calculation. Purely descriptive -- no established clinical mapping
    from octant to anatomic origin exists; its value, if any, is meant to
    be discovered through the outcome-linked discovery pipeline.
    """
    vectors = _cluster_qrs_area_vectors(ctx, cluster_id)
    if len(vectors) < 3:
        return None
    mean_vector = np.mean(vectors, axis=0)
    if np.linalg.norm(mean_vector) <= EPS:
        return None
    return float((mean_vector > 0).astype(int) @ np.array([1, 2, 4]))


def _primary_morphology_qrs_octant_id(ctx: FeatureContext) -> FeatureValue:
    value = _octant_id_for_cluster(ctx, 0)
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _secondary_morphology_qrs_octant_id(ctx: FeatureContext) -> FeatureValue:
    value = _octant_id_for_cluster(ctx, 1)
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _poincare_sd1_sd2(values: list[float]) -> Optional[tuple[float, float]]:
    """Standard Poincare-plot short-/long-term dispersion (Brennan et al.,
    IEEE Trans Biomed Eng 2001), for any successive-pair value sequence."""
    if len(values) < 4:
        return None
    arr = np.asarray(values, dtype=float)
    diff = arr[1:] - arr[:-1]
    sd1 = float(np.sqrt(np.var(diff, ddof=1) / 2.0))
    sd2_arg = max(0.0, 2.0 * float(np.var(arr, ddof=1)) - sd1**2)
    return sd1, float(np.sqrt(sd2_arg))


def _rr_series_ms(ctx: FeatureContext) -> list[float]:
    beats = sorted(ctx.delineation.beats, key=lambda beat: beat.r_peak.sample)
    return [
        (beats[index].r_peak.sample - beats[index - 1].r_peak.sample)
        / ctx.fs
        * 1000.0
        for index in range(1, len(beats))
    ]


def _rr_poincare_sd1(ctx: FeatureContext) -> FeatureValue:
    result = _poincare_sd1_sd2(_rr_series_ms(ctx))
    return FeatureValue(None, 0.0) if result is None else FeatureValue(result[0])


def _rr_poincare_sd2(ctx: FeatureContext) -> FeatureValue:
    result = _poincare_sd1_sd2(_rr_series_ms(ctx))
    return FeatureValue(None, 0.0) if result is None else FeatureValue(result[1])


def _beat_peak_magnitude_sequence(
    ctx: FeatureContext,
    segment: str,
) -> list[float]:
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return []
    beats = sorted(
        _cluster_beats_or_accepted(ctx, 0), key=lambda beat: beat.r_peak.sample
    )
    magnitudes: list[float] = []
    for beat in beats:
        bounds = _beat_bounds(ctx, beat, segment)
        if bounds is None:
            continue
        values = record_vcg[bounds[0] : bounds[1] + 1]
        magnitudes.append(float(np.max(np.linalg.norm(values, axis=1))))
    return magnitudes


def _qrs_magnitude_poincare_sd1(ctx: FeatureContext) -> FeatureValue:
    result = _poincare_sd1_sd2(_beat_peak_magnitude_sequence(ctx, "qrs"))
    return FeatureValue(None, 0.0) if result is None else FeatureValue(result[0])


def _qrs_magnitude_poincare_sd2(ctx: FeatureContext) -> FeatureValue:
    result = _poincare_sd1_sd2(_beat_peak_magnitude_sequence(ctx, "qrs"))
    return FeatureValue(None, 0.0) if result is None else FeatureValue(result[1])


def _t_magnitude_poincare_sd1(ctx: FeatureContext) -> FeatureValue:
    result = _poincare_sd1_sd2(_beat_peak_magnitude_sequence(ctx, "t"))
    return FeatureValue(None, 0.0) if result is None else FeatureValue(result[0])


def _t_magnitude_poincare_sd2(ctx: FeatureContext) -> FeatureValue:
    result = _poincare_sd1_sd2(_beat_peak_magnitude_sequence(ctx, "t"))
    return FeatureValue(None, 0.0) if result is None else FeatureValue(result[1])


def _split_beats_into_halves(beats: list) -> Optional[tuple[list, list]]:
    ordered = sorted(beats, key=lambda beat: beat.r_peak.sample)
    if len(ordered) < 6:
        return None
    midpoint = len(ordered) // 2
    return ordered[:midpoint], ordered[midpoint:]


def _within_record_qrs_axis_drift(ctx: FeatureContext) -> FeatureValue:
    """First- vs. second-half-of-recording QRS axis drift.

    No prior marker compares the two halves of the ~10s recording itself
    (existing 'drift' markers operate within a single beat or loop plane).
    Interpretation is deliberately left open -- lead repositioning,
    electrode contact change, autonomic modulation are all plausible and
    not distinguishable from this marker alone.
    """
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    halves = _split_beats_into_halves(_cluster_beats_or_accepted(ctx, 0))
    if halves is None:
        return FeatureValue(None, 0.0)

    def mean_qrs_vector(group: list) -> Optional[np.ndarray]:
        vectors = []
        for beat in group:
            bounds = _beat_bounds(ctx, beat, "qrs")
            if bounds is None:
                continue
            values = record_vcg[bounds[0] : bounds[1] + 1]
            vector = _trapz(values, 1.0 / ctx.fs, axis=0)
            if np.linalg.norm(vector) > EPS:
                vectors.append(vector)
        return np.mean(vectors, axis=0) if len(vectors) >= 2 else None

    first_vector = mean_qrs_vector(halves[0])
    second_vector = mean_qrs_vector(halves[1])
    if first_vector is None or second_vector is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(_angle_between(first_vector, second_vector))


def _within_record_qrs_amplitude_drift(ctx: FeatureContext) -> FeatureValue:
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    halves = _split_beats_into_halves(_cluster_beats_or_accepted(ctx, 0))
    if halves is None:
        return FeatureValue(None, 0.0)

    def mean_magnitude(group: list) -> Optional[float]:
        magnitudes = []
        for beat in group:
            bounds = _beat_bounds(ctx, beat, "qrs")
            if bounds is None:
                continue
            values = record_vcg[bounds[0] : bounds[1] + 1]
            magnitudes.append(float(np.max(np.linalg.norm(values, axis=1))))
        return float(np.mean(magnitudes)) if magnitudes else None

    first_magnitude = mean_magnitude(halves[0])
    second_magnitude = mean_magnitude(halves[1])
    if first_magnitude is None or second_magnitude is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(second_magnitude - first_magnitude)


def _within_record_qtc_drift(ctx: FeatureContext) -> FeatureValue:
    measurements = ctx.delineation.qt_rr_measurements()
    if len(measurements) < 6:
        return FeatureValue(None, 0.0)
    beats = ctx.delineation.beats
    primary_ids = {
        id(beat)
        for beat in beats
        if getattr(beat, "morphology_cluster_id", None) == 0
    }
    if len(primary_ids) >= 3:
        filtered = [
            item
            for item in measurements
            if 0 <= item.beat_index < len(beats)
            and id(beats[item.beat_index]) in primary_ids
        ]
    else:
        filtered = list(measurements)
    qtc_values = [
        item.qt_ms / float(np.sqrt(item.preceding_rr_seconds))
        for item in filtered
        if item.preceding_rr_seconds > 0
    ]
    if len(qtc_values) < 6:
        return FeatureValue(None, 0.0)
    midpoint = len(qtc_values) // 2
    return FeatureValue(
        float(np.mean(qtc_values[midpoint:]) - np.mean(qtc_values[:midpoint]))
    )


def _kors_vs_dower_axis_delta(ctx: FeatureContext, segment: str) -> FeatureValue:
    """Sensitivity of a derived VCG loop's axis to the transform used.

    This project derives the VCG from the 12-lead ECG (Kors regression by
    default) rather than measuring it directly; Jaros et al. (Sensors
    2019) found Kors the most accurate against true Frank leads, with the
    inverse-Dower transform a systematically different alternative. A
    large delta flags a record where other VCG-derived markers should be
    read with more caution -- a per-record confidence signal, not a
    clinical marker in its own right.
    """
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return FeatureValue(None, 0.0)
    kors_points = ctx.vcg_segment(segment)
    bounds = ctx.bounds(segment)
    if kors_points is None or len(kors_points) < 3 or bounds is None:
        return FeatureValue(None, 0.0)
    try:
        dower_result = VCGTransformRegistry.transform(ecg, "inverse_dower")
    except Exception:
        return FeatureValue(None, 0.0)
    start, end = bounds
    dower_xyz = dower_result.xyz
    if end >= len(dower_xyz):
        return FeatureValue(None, 0.0)
    dower_points = dower_xyz[start : end + 1]
    kors_vector = _trapz(kors_points, 1.0 / ctx.fs, axis=0)
    dower_vector = _trapz(dower_points, 1.0 / ctx.fs, axis=0)
    if np.linalg.norm(kors_vector) <= EPS or np.linalg.norm(dower_vector) <= EPS:
        return FeatureValue(None, 0.0)
    return FeatureValue(_angle_between(kors_vector, dower_vector))


def _qrs_axis_kors_vs_dower_delta(ctx: FeatureContext) -> FeatureValue:
    return _kors_vs_dower_axis_delta(ctx, "qrs")


def _t_axis_kors_vs_dower_delta(ctx: FeatureContext) -> FeatureValue:
    return _kors_vs_dower_axis_delta(ctx, "t")


def _sample_entropy(series: np.ndarray, m: int = 2, r_factor: float = 0.2) -> Optional[float]:
    """Sample entropy (Richman & Moorman), one scale of multiscale entropy."""
    n = len(series)
    if n < (m + 1) * 4:
        return None
    tolerance = r_factor * float(np.std(series))
    if tolerance <= 0:
        return None

    def _template_match_count(length: int) -> int:
        templates = np.array([series[i : i + length] for i in range(n - length + 1)])
        count = 0
        for index in range(len(templates)):
            distances = np.max(np.abs(templates - templates[index]), axis=1)
            count += int(np.sum(distances <= tolerance)) - 1
        return count

    matches_m = _template_match_count(m)
    matches_m1 = _template_match_count(m + 1)
    if matches_m <= 0 or matches_m1 <= 0:
        return None
    return float(-np.log(matches_m1 / matches_m))


def _coarse_grain(series: np.ndarray, scale: int) -> np.ndarray:
    length = len(series) // scale
    if length < 1:
        return np.array([])
    return series[: length * scale].reshape(length, scale).mean(axis=1)


def _multiscale_entropy(series: np.ndarray, scales: tuple[int, ...] = (1, 2, 3)) -> Optional[float]:
    """Costa/Goldberger/Peng (Phys Rev Lett 2002) multiscale entropy, run
    over 1-3 coarse-graining scales -- the window is a single ~QRS/T
    complex (hundreds of samples), so more scales than that are not
    meaningful. Exploratory application of an established technique;
    no prior use of this specific segment/window is known."""
    values = [
        entropy
        for scale in scales
        for entropy in (_sample_entropy(_coarse_grain(series, scale)),)
        if entropy is not None
    ]
    return float(np.mean(values)) if values else None


def _multiscale_entropy_for_segment(ctx: FeatureContext, segment: str) -> Optional[float]:
    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 24:
        return None
    return _multiscale_entropy(np.linalg.norm(points, axis=1))


def _qrs_multiscale_entropy(ctx: FeatureContext) -> FeatureValue:
    value = _multiscale_entropy_for_segment(ctx, "qrs")
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _t_multiscale_entropy(ctx: FeatureContext) -> FeatureValue:
    value = _multiscale_entropy_for_segment(ctx, "t")
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _higuchi_fractal_dimension(series: np.ndarray, k_max: int = 6) -> Optional[float]:
    """Higuchi (Physica D 1988) fractal dimension via multi-k curve-length
    scaling -- mathematically distinct from the already-registered
    <segment>_katz_fractal_dimension (single-scale L/d formula)."""
    n = len(series)
    if n < k_max * 4:
        return None
    log_lengths: list[float] = []
    log_inverse_k: list[float] = []
    for k in range(1, k_max + 1):
        lengths = []
        for m in range(k):
            indices = np.arange(m, n, k)
            if len(indices) < 2:
                continue
            diffs = np.abs(np.diff(series[indices]))
            normalization = (n - 1) / (len(indices) - 1) / k
            lengths.append(float(np.sum(diffs)) * normalization)
        if lengths:
            log_lengths.append(float(np.log(np.mean(lengths) + EPS)))
            log_inverse_k.append(float(np.log(1.0 / k)))
    if len(log_lengths) < 2:
        return None
    slope, _ = np.polyfit(log_inverse_k, log_lengths, 1)
    return float(slope)


def _higuchi_fractal_dimension_for_segment(
    ctx: FeatureContext,
    segment: str,
) -> Optional[float]:
    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 24:
        return None
    return _higuchi_fractal_dimension(np.linalg.norm(points, axis=1))


def _qrs_higuchi_fractal_dimension(ctx: FeatureContext) -> FeatureValue:
    value = _higuchi_fractal_dimension_for_segment(ctx, "qrs")
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _t_higuchi_fractal_dimension(ctx: FeatureContext) -> FeatureValue:
    value = _higuchi_fractal_dimension_for_segment(ctx, "t")
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _cross_correlation_peak(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
) -> Optional[tuple[int, float]]:
    x = x - np.mean(x)
    y = y - np.mean(y)
    if np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    n = len(x)
    best_lag = 0
    best_value = -2.0
    for lag in range(-max_lag, max_lag + 1):
        xs, ys = (x[lag:], y[: n - lag]) if lag >= 0 else (x[: n + lag], y[-lag:])
        if len(xs) < 5:
            continue
        denominator = float(np.std(xs) * np.std(ys))
        if denominator <= EPS:
            continue
        value = float(np.mean(xs * ys) / denominator)
        if value > best_value:
            best_value = value
            best_lag = lag
    return (best_lag, best_value) if best_value > -1.5 else None


def _qrs_xy_cross_correlation(ctx: FeatureContext) -> Optional[tuple[int, float]]:
    points = ctx.vcg_segment("qrs")
    if points is None or len(points) < 20:
        return None
    return _cross_correlation_peak(
        points[:, 0], points[:, 1], max_lag=max(1, min(20, len(points) // 4))
    )


def _qrs_xy_cross_correlation_peak_lag(ctx: FeatureContext) -> FeatureValue:
    """Lag (ms) of peak X-Y cross-correlation within the QRS window: does
    one VCG axis lead the other in the activation sequence? No prior
    marker computes a pairwise per-axis lag-correlation function (only an
    aggregate phase-spread exists, loop_fourier_phase_shift_xyz)."""
    result = _qrs_xy_cross_correlation(ctx)
    if result is None:
        return FeatureValue(None, 0.0)
    lag, _ = result
    return FeatureValue(float(lag) / ctx.fs * 1000.0)


def _qrs_xy_cross_correlation_peak_value(ctx: FeatureContext) -> FeatureValue:
    result = _qrs_xy_cross_correlation(ctx)
    if result is None:
        return FeatureValue(None, 0.0)
    _, value = result
    return FeatureValue(value)


def _time_reversal_asymmetry_for_segment(
    ctx: FeatureContext,
    segment: str,
) -> Optional[float]:
    """RMS distance between a loop trajectory and its own time-reversed
    version, scale-normalized -- a forward/backward playback symmetry
    check, conceptually distinct from the already-registered rotational
    chirality index (cross-product handedness). No literature precedent
    found; a mathematically-motivated exploratory construct."""
    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 8:
        return None
    resampled = _resample_curve(points, 64)
    if resampled is None:
        return None
    reversed_curve = resampled[::-1]
    centered = resampled - np.mean(resampled, axis=0)
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale <= EPS:
        return None
    rms = float(np.sqrt(np.mean(np.sum((resampled - reversed_curve) ** 2, axis=1))))
    return rms / scale


def _qrs_time_reversal_asymmetry(ctx: FeatureContext) -> FeatureValue:
    value = _time_reversal_asymmetry_for_segment(ctx, "qrs")
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _t_time_reversal_asymmetry(ctx: FeatureContext) -> FeatureValue:
    value = _time_reversal_asymmetry_for_segment(ctx, "t")
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _requires_wide_complex_tachycardia_pattern(ctx: FeatureContext) -> bool:
    return "wide_complex_tachycardia_pattern" in ctx.delineation.rhythm.flags


def _requires_narrow_complex_tachycardia_pattern(ctx: FeatureContext) -> bool:
    return "narrow_complex_tachycardia_pattern" in ctx.delineation.rhythm.flags


def _lead_rs_duration_ms(segment: np.ndarray, fs: int) -> Optional[float]:
    """R-to-S-nadir duration if `segment` shows a genuine RS pattern
    (positive R peak followed by a negative S trough), else None."""
    if len(segment) < 3:
        return None
    r_index = int(np.argmax(segment))
    if r_index >= len(segment) - 1 or segment[r_index] <= 0:
        return None
    tail = segment[r_index:]
    s_index = int(np.argmin(tail))
    if s_index <= 0 or tail[s_index] >= 0:
        return None
    return float(s_index) / fs * 1000.0


def _precordial_rs_durations(ctx: FeatureContext) -> Optional[dict[str, float]]:
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return None
    beats = _cluster_beats_or_accepted(ctx, 0)
    lead_values: dict[str, list[float]] = {lead: [] for lead in _PRECORDIAL_LEADS}
    for beat in beats:
        r_onset = getattr(beat, "r_onset", None)
        r_offset = getattr(beat, "r_offset", None)
        if r_onset is None or r_offset is None:
            continue
        start, end = int(r_onset.sample), int(r_offset.sample)
        if end - start < 6 or start < 0 or end >= ecg.shape[0]:
            continue
        for lead in _PRECORDIAL_LEADS:
            index = LEAD_MAP.get(lead)
            if index is None or index >= ecg.shape[1]:
                continue
            duration = _lead_rs_duration_ms(ecg[start : end + 1, index], ctx.fs)
            if duration is not None:
                lead_values[lead].append(duration)
    return {
        lead: float(np.median(values))
        for lead, values in lead_values.items()
        if values
    }


def _precordial_rs_absent_flag(ctx: FeatureContext) -> FeatureValue:
    """Brugada criterion step 1: absence of an RS complex in ALL
    precordial leads favors VT (Brugada et al., Circulation 1991)."""
    if not _requires_wide_complex_tachycardia_pattern(ctx):
        return FeatureValue(None, 0.0)
    durations = _precordial_rs_durations(ctx)
    if durations is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(0.0 if durations else 1.0)


def _precordial_rs_duration_max_ms(ctx: FeatureContext) -> FeatureValue:
    """Brugada criterion step 2: RS duration >100ms in any precordial
    lead favors VT."""
    if not _requires_wide_complex_tachycardia_pattern(ctx):
        return FeatureValue(None, 0.0)
    durations = _precordial_rs_durations(ctx)
    if not durations:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(max(durations.values())))


def _precordial_concordance_flag(ctx: FeatureContext) -> FeatureValue:
    """All V1-V6 QRS complexes uniformly monophasic positive or negative
    -- high-specificity, low-sensitivity VT indicator."""
    if not _requires_wide_complex_tachycardia_pattern(ctx):
        return FeatureValue(None, 0.0)
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return FeatureValue(None, 0.0)
    beats = _cluster_beats_or_accepted(ctx, 0)
    lead_net: dict[str, list[float]] = {lead: [] for lead in _PRECORDIAL_LEADS}
    for beat in beats:
        r_onset = getattr(beat, "r_onset", None)
        r_offset = getattr(beat, "r_offset", None)
        if r_onset is None or r_offset is None:
            continue
        start, end = int(r_onset.sample), int(r_offset.sample)
        if end - start < 6 or start < 0 or end >= ecg.shape[0]:
            continue
        for lead in _PRECORDIAL_LEADS:
            index = LEAD_MAP.get(lead)
            if index is None or index >= ecg.shape[1]:
                continue
            lead_net[lead].append(float(np.sum(ecg[start : end + 1, index])))
    medians = {
        lead: float(np.median(values))
        for lead, values in lead_net.items()
        if values
    }
    if len(medians) < 4:
        return FeatureValue(None, 0.0)
    all_positive = all(value > 0 for value in medians.values())
    all_negative = all(value < 0 for value in medians.values())
    return FeatureValue(1.0 if (all_positive or all_negative) else 0.0)


def _avr_vi_vt_ratio(ctx: FeatureContext) -> FeatureValue:
    """Vereckei aVR algorithm: ratio of vertical excursion during the
    initial vs. terminal 40ms of the QRS in lead aVR (Vereckei et al.,
    Eur Heart J 2007). Ratio <=1 favors VT."""
    if not _requires_wide_complex_tachycardia_pattern(ctx):
        return FeatureValue(None, 0.0)
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return FeatureValue(None, 0.0)
    avr_index = LEAD_MAP.get("aVR")
    if avr_index is None or avr_index >= ecg.shape[1]:
        return FeatureValue(None, 0.0)
    window_samples = max(2, int(round(0.040 * ctx.fs)))
    ratios: list[float] = []
    for beat in _cluster_beats_or_accepted(ctx, 0):
        r_onset = getattr(beat, "r_onset", None)
        r_offset = getattr(beat, "r_offset", None)
        if r_onset is None or r_offset is None:
            continue
        start, end = int(r_onset.sample), int(r_offset.sample)
        if end - start < 2 * window_samples or start < 0 or end >= ecg.shape[0]:
            continue
        segment = ecg[start : end + 1, avr_index]
        initial = segment[:window_samples]
        terminal = segment[-window_samples:]
        vi = float(np.sum(np.abs(np.diff(initial))))
        vt = float(np.sum(np.abs(np.diff(terminal))))
        if vt <= EPS:
            continue
        ratios.append(vi / vt)
    if len(ratios) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.median(ratios)))


def _av_dissociation_supports_vt_flag(ctx: FeatureContext) -> FeatureValue:
    """AV dissociation during a wide-complex tachycardia -- an
    independently periodic atrial rate that does not track the
    ventricular rate and is not explained by a simple N:1 conducted
    ratio (FusionDelineator._atrial_spectrum / RhythmAssessment flag
    'av_dissociation_compatible') -- is one of the most specific classic
    criteria for VT over SVT-with-aberrancy (Wellens et al., Am J Med
    1978). This is deliberately a rate-comparison surrogate, not an
    attempt to resolve a P wave that visually coincides with the QRS:
    at typical P amplitude (~0.1-0.25 mV) against a wide/polymorphic
    QRS several times larger, direct waveform detection of a P wave
    buried inside the QRS is not a reliable target, regardless of lead
    selection or VCG magnitude fusion."""
    if not _requires_wide_complex_tachycardia_pattern(ctx):
        return FeatureValue(None, 0.0)
    rhythm = ctx.delineation.rhythm
    if rhythm.independent_atrial_rate_hz is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(1.0 if "av_dissociation_compatible" in rhythm.flags else 0.0)


def _independent_atrial_rate_bpm(ctx: FeatureContext) -> FeatureValue:
    """Continuous exposure of the independent (QRS-masked spectral)
    atrial-rate estimate that feeds the av_dissociation_compatible /
    third_degree_av_block_pattern rhythm flags (see
    FusionDelineator._atrial_spectrum). Exposed as a raw continuous
    value, not just the binary flags, so the platform's own statistical
    discovery pipeline can locate a data-driven cutoff rather than
    relying solely on the hand-picked thresholds baked into those
    flags."""
    rhythm = ctx.delineation.rhythm
    if rhythm.independent_atrial_rate_hz is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(rhythm.independent_atrial_rate_hz) * 60.0)


def _independent_atrial_ventricular_rate_ratio(ctx: FeatureContext) -> FeatureValue:
    """Continuous ratio of the independent atrial-rate estimate to the
    ventricular (RR-median-derived) rate -- the same signal underlying
    av_dissociation_compatible, exposed as a raw ratio rather than the
    fixed 22%-mismatch / 5%-integer-tolerance rules used there, again
    for data-driven threshold discovery."""
    rhythm = ctx.delineation.rhythm
    if rhythm.independent_atrial_rate_hz is None or not rhythm.rr_median_ms:
        return FeatureValue(None, 0.0)
    heart_rate_bpm = 60000.0 / rhythm.rr_median_ms
    if heart_rate_bpm <= 0:
        return FeatureValue(None, 0.0)
    return FeatureValue(
        float(rhythm.independent_atrial_rate_hz) * 60.0 / heart_rate_bpm
    )


def _rp_prime_interval_ms(ctx: FeatureContext) -> FeatureValue:
    """R-to-retrograde-P' interval (the clinical "RP'" measurement): time
    from the R peak to a retrograde P candidate found after this beat's
    own QRS (MultileadDelineator._search_retrograde_p_candidates). A
    short RP' (<~70ms) is typical for typical (slow-fast) AVNRT, where
    the retrograde P is buried in or immediately follows the QRS; a
    longer RP' points toward atypical AVNRT or another retrograde-
    conduction SVT (Katritsis & Josephson, Europace 2016). The value
    itself is a candidate feature for the discovery pipeline, not a
    diagnosis."""
    intervals_ms: list[float] = []
    for beat in _cluster_beats_or_accepted(ctx, 0):
        if beat.retrograde_p_status != WaveStatus.PRESENT:
            continue
        if beat.retrograde_p_peak is None:
            continue
        interval = (
            (beat.retrograde_p_peak.sample - beat.r_peak.sample)
            / ctx.fs
            * 1000.0
        )
        if interval > 0:
            intervals_ms.append(interval)
    if len(intervals_ms) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.median(intervals_ms)))


def _retrograde_p_qrs_axis_angle_deg(ctx: FeatureContext) -> FeatureValue:
    """Spatial angle between a retrograde P candidate's dominant VCG
    loop axis and this beat's own QRS-loop axis (see
    core.vcg_transform.loop_dominant_axis). A genuinely retrograde
    atrial wavefront -- originating near the AV node and propagating
    backward/upward through the atria -- is a physically different
    activation direction than antegrade ventricular depolarization, so
    a materially non-parallel axis is exploratory supporting evidence
    for a retrograde-P candidate being real atrial activity rather than
    a misclassified QRS/T fragment; it is not itself a detection
    criterion (the candidate is already found by
    _search_retrograde_p_candidates before this runs)."""
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    angles: list[float] = []
    for beat in _cluster_beats_or_accepted(ctx, 0):
        if beat.retrograde_p_status != WaveStatus.PRESENT:
            continue
        if beat.retrograde_p_onset is None or beat.retrograde_p_offset is None:
            continue
        if beat.r_onset is None or beat.r_offset is None:
            continue
        p_axis = loop_dominant_axis(
            record_vcg,
            beat.retrograde_p_onset.sample,
            beat.retrograde_p_offset.sample,
        )
        qrs_axis = loop_dominant_axis(
            record_vcg, beat.r_onset.sample, beat.r_offset.sample
        )
        angle = axis_angle_deg(p_axis, qrs_axis)
        if angle is not None:
            angles.append(angle)
    if len(angles) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.median(angles)))


def _qualified_axis_segments(beat, *segments: str) -> bool:
    """Require finite qualified boundaries for every axis in a comparison."""
    for segment in segments:
        confidence = beat.segment_confidence(segment)
        if not np.isfinite(confidence) or not measurement_confidence_threshold(segment) <= confidence <= 1:
            return False
    return True


def _p_loop_qrs_loop_axis_angle_deg(ctx: FeatureContext) -> FeatureValue:
    """Spatial angle between the (forward) P-loop's dominant VCG axis
    and this beat's QRS-loop axis -- a direction-based spatial
    companion to the amplitude/duration-based p_terminal_force_v1 and
    p_duration_ge_120 classical criteria for atrial abnormality/
    interatrial block. Uses loop DIRECTION (SVD dominant axis) rather
    than amplitude, unlike every existing pre-development VCG loop
    biomarker, which characterizes a single segment's shape but never
    compares directions ACROSS segments."""
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    angles: list[float] = []
    for beat in _cluster_beats_or_accepted(ctx, 0):
        if beat.resolved_wave_status("p") != WaveStatus.PRESENT:
            continue
        if beat.p_onset is None or beat.p_offset is None:
            continue
        if beat.r_onset is None or beat.r_offset is None:
            continue
        if not _qualified_axis_segments(beat, "p", "qrs"):
            continue
        p_axis = loop_dominant_axis(
            record_vcg, beat.p_onset.sample, beat.p_offset.sample
        )
        qrs_axis = loop_dominant_axis(
            record_vcg, beat.r_onset.sample, beat.r_offset.sample
        )
        angle = axis_angle_deg(p_axis, qrs_axis)
        if angle is not None:
            angles.append(angle)
    if len(angles) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.median(angles)))


def _p_loop_t_loop_axis_angle_deg(ctx: FeatureContext) -> FeatureValue:
    """Spatial angle between the P-loop's dominant VCG axis and the
    T-loop's dominant VCG axis for the same beat -- an atrial-to-
    ventricular-repolarization directional counterpart to the
    established spatial QRS-T angle, which compares depolarization to
    repolarization but has no atrial analogue in the existing catalog."""
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    angles: list[float] = []
    for beat in _cluster_beats_or_accepted(ctx, 0):
        if beat.resolved_wave_status("p") != WaveStatus.PRESENT:
            continue
        if beat.resolved_wave_status("t") != WaveStatus.PRESENT:
            continue
        if beat.p_onset is None or beat.p_offset is None:
            continue
        if beat.t_onset is None or beat.t_offset is None:
            continue
        if not _qualified_axis_segments(beat, "p", "t"):
            continue
        p_axis = loop_dominant_axis(
            record_vcg, beat.p_onset.sample, beat.p_offset.sample
        )
        t_axis = loop_dominant_axis(
            record_vcg, beat.t_onset.sample, beat.t_offset.sample
        )
        angle = axis_angle_deg(p_axis, t_axis)
        if angle is not None:
            angles.append(angle)
    if len(angles) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.median(angles)))


def _beatwise_p_loop_axes(ctx: FeatureContext) -> list[np.ndarray]:
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return []
    axes: list[np.ndarray] = []
    for beat in sorted(ctx.delineation.beats, key=lambda beat: beat.r_peak.sample):
        if beat.resolved_wave_status("p") != WaveStatus.PRESENT:
            continue
        if beat.p_onset is None or beat.p_offset is None:
            continue
        axis = loop_dominant_axis(
            record_vcg, beat.p_onset.sample, beat.p_offset.sample
        )
        if axis is not None:
            axes.append(axis)
    return axes


def _beatwise_p_loop_axis_stability_deg(ctx: FeatureContext) -> FeatureValue:
    """Beat-to-beat angular dispersion of the P-loop's dominant spatial
    axis across the whole record (all beats, not cluster-filtered --
    axis wandering may not form a stable morphology cluster in the
    first place, the same rationale beatwise_qrs_axis_alternation_score
    uses). High dispersion (a wandering P-axis) is a candidate spatial
    signature of multifocal atrial activity / wandering atrial
    pacemaker, complementing multifocal_atrial_tachycardia_pattern_flag
    (G11), which is a timing/morphology-count criterion and does not
    look at loop direction at all."""
    axes = _beatwise_p_loop_axes(ctx)
    if len(axes) < 4:
        return FeatureValue(None, 0.0)
    reference = axes[0]
    deviations_deg: list[float] = []
    for axis in axes:
        aligned = -axis if np.dot(axis, reference) < 0 else axis
        cos_similarity = float(np.clip(np.dot(aligned, reference), -1.0, 1.0))
        deviations_deg.append(float(np.degrees(np.arccos(cos_similarity))))
    return FeatureValue(float(np.std(deviations_deg)))


def _terminal_window_has_extra_deflection(segment: np.ndarray, positive: bool) -> bool:
    """An interior local extremum in the terminal QRS window (not merely
    the window's own boundary sample) -- a candidate extra terminal
    deflection (pseudo-r'/pseudo-s), independent of the sign convention
    requested."""
    from scipy.signal import find_peaks

    if len(segment) < 4:
        return False
    values = segment if positive else -segment
    prominence = max(0.02, float(np.std(values)) * 0.3)
    peaks, _ = find_peaks(values, prominence=prominence)
    interior = [index for index in peaks if 0 < index < len(segment) - 1]
    return len(interior) >= 1


def _terminal_qrs_notch_flag(
    ctx: FeatureContext,
    leads: tuple[str, ...],
    positive: bool,
    terminal_ms: float = 50.0,
) -> FeatureValue:
    if not _requires_narrow_complex_tachycardia_pattern(ctx):
        return FeatureValue(None, 0.0)
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return FeatureValue(None, 0.0)
    terminal_samples = max(2, int(round(terminal_ms / 1000.0 * ctx.fs)))
    votes: list[bool] = []
    for beat in _cluster_beats_or_accepted(ctx, 0):
        r_onset = getattr(beat, "r_onset", None)
        r_offset = getattr(beat, "r_offset", None)
        if r_onset is None or r_offset is None:
            continue
        start, end = int(r_onset.sample), int(r_offset.sample)
        if end - start < terminal_samples + 2 or start < 0 or end >= ecg.shape[0]:
            continue
        terminal_start = max(start, end - terminal_samples)
        found = False
        for lead in leads:
            index = LEAD_MAP.get(lead)
            if index is None or index >= ecg.shape[1]:
                continue
            segment = ecg[terminal_start : end + 1, index]
            if _terminal_window_has_extra_deflection(segment, positive):
                found = True
                break
        votes.append(found)
    if len(votes) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(1.0 if (sum(votes) / len(votes)) > 0.5 else 0.0)


def _terminal_qrs_pseudo_r_prime_v1_flag(ctx: FeatureContext) -> FeatureValue:
    """Extra positive terminal deflection in V1 during a narrow-complex
    tachycardia -- compatible with (not diagnostic of) a retrograde P
    wave fused into the terminal QRS, the typical-AVNRT pseudo-r' sign
    (Kalbfleisch et al., JACC 1993). No sinus-rhythm baseline comparison
    is available here, unlike the clinical criterion."""
    return _terminal_qrs_notch_flag(ctx, ("V1",), positive=True)


def _terminal_qrs_pseudo_s_inferior_flag(ctx: FeatureContext) -> FeatureValue:
    """Extra negative terminal deflection in the inferior leads during a
    narrow-complex tachycardia -- the pseudo-s counterpart."""
    return _terminal_qrs_notch_flag(ctx, ("II", "III", "aVF"), positive=False)


def _beatwise_frontal_qrs_axes(ctx: FeatureContext) -> list[float]:
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return []
    axes: list[float] = []
    for beat in sorted(ctx.delineation.beats, key=lambda beat: beat.r_peak.sample):
        bounds = _beat_bounds(ctx, beat, "qrs")
        if bounds is None:
            continue
        values = record_vcg[bounds[0] : bounds[1] + 1]
        vector = _trapz(values, 1.0 / ctx.fs, axis=0)
        if np.linalg.norm(vector[:2]) <= EPS:
            continue
        axes.append(float(np.degrees(np.arctan2(vector[1], vector[0]))))
    return axes


def _angular_distance_deg(first: float, second: float) -> float:
    delta = abs(first - second) % 360.0
    return min(delta, 360.0 - delta)


def _beatwise_qrs_axis_alternation_score(ctx: FeatureContext) -> FeatureValue:
    """Bidirectional-VT/CPVT candidate: strict period-2 QRS-axis
    alternation across the beat sequence (axis(N) ~= axis(N+2), axis(N)
    vs axis(N+1) ~180 degrees apart) -- Leenhardt et al., Circulation
    1995. Runs over ALL beats, not cluster-filtered, since alternating
    beats may not form a stable shape cluster under E3's similarity
    metric in the first place."""
    axes = _beatwise_frontal_qrs_axes(ctx)
    if len(axes) < 4:
        return FeatureValue(None, 0.0)
    same_parity = [
        _angular_distance_deg(axes[index], axes[index + 2])
        for index in range(len(axes) - 2)
    ]
    alternating = [
        _angular_distance_deg(axes[index], axes[index + 1])
        for index in range(len(axes) - 1)
    ]
    same_parity_score = 1.0 - float(np.mean(same_parity)) / 180.0
    alternating_score = float(np.mean(alternating)) / 180.0
    return FeatureValue(
        float(np.clip((same_parity_score + alternating_score) / 2.0, 0.0, 1.0))
    )


def _qrs_axis_progressive_rotation(ctx: FeatureContext) -> FeatureValue:
    """Torsades-de-pointes candidate: mean sign-consistent angular step
    between successive beats' frontal QRS axis -- monotonic drift rather
    than G9's strict alternation. Purely descriptive; no QTc precondition
    is built in so the discovery pipeline can test that correlation
    itself rather than the marker definition assuming it."""
    axes = _beatwise_frontal_qrs_axes(ctx)
    if len(axes) < 6:
        return FeatureValue(None, 0.0)
    steps = [
        (axes[index + 1] - axes[index] + 180.0) % 360.0 - 180.0
        for index in range(len(axes) - 1)
    ]
    return FeatureValue(float(np.mean(steps)))


def _distinct_p_morphology_cluster_count_raw(ctx: FeatureContext) -> Optional[int]:
    """Number of distinct P-wave shape populations across the whole beat
    sequence, via the same medoid + cosine-similarity clustering E3 uses
    for QRS -- but read per-beat P-loops directly (not gated on
    is_segment_evaluable("p")), since MAT's very definition is the
    absence of one stable P consensus that gate would require.
    """
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return None
    vectors: list[np.ndarray] = []
    for beat in ctx.delineation.beats:
        p_onset = getattr(beat, "p_onset", None)
        p_offset = getattr(beat, "p_offset", None)
        if p_onset is None or p_offset is None:
            continue
        start, end = int(p_onset.sample), int(p_offset.sample)
        if end - start < 4 or start < 0 or end >= len(record_vcg):
            continue
        curve = _resample_curve(record_vcg[start : end + 1], 32)
        if curve is None:
            continue
        vector = curve.ravel()
        scale = float(np.linalg.norm(vector))
        if scale <= EPS:
            continue
        vectors.append(vector / scale)
    if len(vectors) < 3:
        return None
    matrix = np.asarray(vectors)
    remaining = np.arange(len(matrix))
    cluster_count = 0
    minimum_cluster_beats = 2  # P waves are smaller/noisier than QRS;
    # relaxed from E3's 3-beat minimum for the secondary QRS cluster.
    max_clusters = 4
    while len(remaining) >= minimum_cluster_beats and cluster_count < max_clusters:
        sub = matrix[remaining]
        similarities = np.clip(sub @ sub.T, -1.0, 1.0)
        medoid = int(np.argmax(np.median(similarities, axis=1)))
        members = similarities[medoid] >= 0.70
        if int(np.sum(members)) < minimum_cluster_beats:
            break
        cluster_count += 1
        remaining = remaining[~members]
    return cluster_count


def _distinct_p_morphology_cluster_count(ctx: FeatureContext) -> FeatureValue:
    count = _distinct_p_morphology_cluster_count_raw(ctx)
    return FeatureValue(None, 0.0) if count is None else FeatureValue(float(count))


def _multifocal_atrial_tachycardia_pattern_flag(ctx: FeatureContext) -> FeatureValue:
    """Kastor's MAT criteria (NEJM 1990): >=3 distinct P-wave
    morphologies, irregular rate >100bpm, no flatter waves. Fills a real
    gap -- the existing multiple_atrial_activity RhythmClass counts beats
    with MULTIPLE atrial deflections before one QRS, which is a different
    concept from P morphology varying from beat to beat."""
    count = _distinct_p_morphology_cluster_count_raw(ctx)
    if count is None:
        return FeatureValue(None, 0.0)
    rhythm = ctx.delineation.rhythm
    heart_rate_bpm = (
        60000.0 / rhythm.rr_median_ms if rhythm.rr_median_ms else None
    )
    if heart_rate_bpm is None:
        return FeatureValue(None, 0.0)
    is_irregular = rhythm.rhythm_class in (
        RhythmClass.IRREGULAR,
        RhythmClass.MIXED,
    ) or (rhythm.rr_cv is not None and rhythm.rr_cv >= 0.12)
    no_flutter = rhythm.rhythm_class != RhythmClass.FLUTTER_COMPATIBLE
    present = count >= 3 and heart_rate_bpm > 100.0 and is_irregular and no_flutter
    return FeatureValue(1.0 if present else 0.0)


def _beat_delta_wave_slope_ratio(
    beat,
    ecg: np.ndarray,
    fs: int,
    lead_index: int,
) -> Optional[float]:
    r_onset = getattr(beat, "r_onset", None)
    r_offset = getattr(beat, "r_offset", None)
    if r_onset is None or r_offset is None:
        return None
    start, end = int(r_onset.sample), int(r_offset.sample)
    initial_samples = max(2, int(round(0.040 * fs)))
    if end - start < initial_samples * 2 or start < 0 or end >= ecg.shape[0]:
        return None
    segment = ecg[start : end + 1, lead_index]
    initial_slope = float(np.mean(np.abs(np.diff(segment[:initial_samples]))))
    overall_slope = float(np.mean(np.abs(np.diff(segment))))
    if overall_slope <= EPS:
        return None
    return initial_slope / overall_slope


_SHORT_PR_INTERVAL_MS = 120.0


def _delta_wave_present_flag(ctx: FeatureContext) -> FeatureValue:
    """Delta wave = an unusually SLOW initial QRS upstroke (slurred
    onset) rather than the sharp onset a normally-conducted beat shows --
    the inverse of F8's Maximum Deflection Index intent. Uses whichever
    standard lead has the largest QRS amplitude per beat.

    A real LUDB spot-check found the slope heuristic alone fires on
    records with unrelated causes of a slurred/altered initial QRS
    deflection (LVH/scar conduction delay; pathological Q waves from a
    prior MI), neither of which is WPW. When a P wave is reliably
    present, a short PR interval (<120ms) -- the second classic
    Wolff-Parkinson-White sign, since pre-excitation partially bypasses
    the AV node -- is required to corroborate the slope signal. Without
    a measurable P/PR (e.g. during the tachycardia itself), the slope
    signal alone is used since there is nothing to corroborate it with.
    Residual false-positive risk remains even with this corroboration
    (a coincidentally borderline-short PR next to an MI-related Q wave
    can still pass both checks) -- this stays a coarse screening
    heuristic, not a WPW diagnosis.
    """
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return FeatureValue(None, 0.0)
    ratios: list[float] = []
    pr_intervals_ms: list[float] = []
    for beat in _cluster_beats_or_accepted(ctx, 0):
        r_onset = getattr(beat, "r_onset", None)
        r_offset = getattr(beat, "r_offset", None)
        if r_onset is None or r_offset is None:
            continue
        start, end = int(r_onset.sample), int(r_offset.sample)
        if end - start < 10 or start < 0 or end >= ecg.shape[0]:
            continue
        best_ratio: Optional[float] = None
        best_amplitude = -1.0
        for lead in _ALL_STANDARD_LEADS:
            index = LEAD_MAP.get(lead)
            if index is None or index >= ecg.shape[1]:
                continue
            segment = ecg[start : end + 1, index]
            amplitude = float(np.max(segment) - np.min(segment))
            if amplitude > best_amplitude:
                ratio = _beat_delta_wave_slope_ratio(beat, ecg, ctx.fs, index)
                if ratio is not None:
                    best_amplitude = amplitude
                    best_ratio = ratio
        if best_ratio is not None:
            ratios.append(best_ratio)
        pr_interval = beat.pr_interval_ms(ctx.fs)
        if pr_interval is not None:
            pr_intervals_ms.append(pr_interval)
    if len(ratios) < 3:
        return FeatureValue(None, 0.0)
    slope_positive = float(np.median(ratios)) < 0.5
    if len(pr_intervals_ms) >= 3:
        slope_positive = (
            slope_positive
            and float(np.median(pr_intervals_ms)) < _SHORT_PR_INTERVAL_MS
        )
    return FeatureValue(1.0 if slope_positive else 0.0)


def _delta_wave_lead_initial_amplitude(
    ctx: FeatureContext,
    lead: str,
) -> Optional[float]:
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return None
    index = LEAD_MAP.get(lead)
    if index is None or index >= ecg.shape[1]:
        return None
    initial_samples = max(2, int(round(0.040 * ctx.fs)))
    values: list[float] = []
    for beat in _cluster_beats_or_accepted(ctx, 0):
        r_onset = getattr(beat, "r_onset", None)
        if r_onset is None:
            continue
        start = int(r_onset.sample)
        end = start + initial_samples
        if start < 0 or end >= ecg.shape[0]:
            continue
        values.append(float(np.mean(ecg[start:end, index])))
    return float(np.median(values)) if len(values) >= 3 else None


def _delta_wave_frontal_axis(ctx: FeatureContext) -> FeatureValue:
    """Frontal-plane axis of the initial (delta-wave) QRS forces, via
    lead I / aVF sign (standard hexaxial convention: I = 0deg, aVF =
    +90deg), following the Arruda-style delta-axis step of accessory-
    pathway localization (Arruda et al. 1998). Only meaningful once a
    delta wave is actually present."""
    if _delta_wave_present_flag(ctx).value != 1.0:
        return FeatureValue(None, 0.0)
    lead_i = _delta_wave_lead_initial_amplitude(ctx, "I")
    lead_avf = _delta_wave_lead_initial_amplitude(ctx, "aVF")
    if lead_i is None or lead_avf is None:
        return FeatureValue(None, 0.0)
    if abs(lead_i) <= EPS and abs(lead_avf) <= EPS:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.degrees(np.arctan2(lead_avf, lead_i))))


def _accessory_pathway_quadrant_estimate(ctx: FeatureContext) -> FeatureValue:
    """Coarse, unvalidated accessory-pathway quadrant hypothesis: delta-
    wave frontal axis (anterolateral vs. posterolateral) combined with V1
    initial polarity (left- vs. right-sided), a deliberately simplified
    stand-in for the much more elaborate validated Arruda/EASY-WPW/
    SMART-WPW algorithms. 0=right/anterolateral, 1=right/posterolateral,
    2=left/anterolateral, 3=left/posterolateral."""
    axis = _delta_wave_frontal_axis(ctx)
    if axis.value is None:
        return FeatureValue(None, 0.0)
    v1_initial = _delta_wave_lead_initial_amplitude(ctx, "V1")
    if v1_initial is None:
        return FeatureValue(None, 0.0)
    left_sided = v1_initial > 0
    anterolateral = axis.value >= 0
    quadrant = (2 if left_sided else 0) + (0 if anterolateral else 1)
    return FeatureValue(float(quadrant))


def _flutter_av_block_ratio_estimate(ctx: FeatureContext) -> FeatureValue:
    """Atrial:ventricular rate ratio for flutter-compatible rhythms
    (e.g. ~2.0 for classic 2:1 block), from already-available rhythm
    fields -- no new signal processing."""
    rhythm = ctx.delineation.rhythm
    if rhythm.rhythm_class != RhythmClass.FLUTTER_COMPATIBLE:
        return FeatureValue(None, 0.0)
    if not rhythm.atrial_frequency_hz or not rhythm.rr_median_ms:
        return FeatureValue(None, 0.0)
    ventricular_rate_hz = 1000.0 / rhythm.rr_median_ms
    if ventricular_rate_hz <= EPS:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(rhythm.atrial_frequency_hz / ventricular_rate_hz))


_ALL_STANDARD_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

_QRS_CONTIGUOUS_LEAD_GROUPS = (
    ("II", "III"),
    ("II", "aVF"),
    ("III", "aVF"),
    ("I", "aVL"),
    ("I", "V5"),
    ("I", "V6"),
    ("aVL", "V5"),
    ("aVL", "V6"),
    ("V5", "V6"),
    ("V1", "V2"),
    ("V2", "V3"),
    ("V3", "V4"),
    ("V4", "V5"),
)  # Anatomically contiguous coronary-territory lead pairs; mirrors
   # mimic/classical.py's CONTIGUOUS_GROUPS, duplicated here (rather than
   # imported) to keep core/biomarkers independent of the mimic layer.


def _lead_is_fragmented(segment: np.ndarray) -> bool:
    """More than the two extrema (e.g. R and S) a clean QRS deflection
    normally shows in a single lead -- an extra notch or R' wave, the
    signature Das et al. (Circulation 2006, 2008) scored as fragmented."""
    from scipy.signal import find_peaks

    prominence = max(0.02, float(np.std(segment)) * 0.3)
    positive_peaks, _ = find_peaks(segment, prominence=prominence)
    negative_peaks, _ = find_peaks(-segment, prominence=prominence)
    return len(positive_peaks) + len(negative_peaks) > 2


def _cluster_fragmented_lead_votes(
    ctx: FeatureContext,
    cluster_id: int,
) -> Optional[dict[str, bool]]:
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return None
    beats = _cluster_beats_or_accepted(ctx, cluster_id)
    beat_flags: list[dict[str, bool]] = []
    for beat in beats:
        r_onset = getattr(beat, "r_onset", None)
        r_offset = getattr(beat, "r_offset", None)
        if r_onset is None or r_offset is None:
            continue
        start, end = int(r_onset.sample), int(r_offset.sample)
        if end - start < 6 or start < 0 or end >= ecg.shape[0]:
            continue
        flags: dict[str, bool] = {}
        for lead in _ALL_STANDARD_LEADS:
            index = LEAD_MAP.get(lead)
            if index is None or index >= ecg.shape[1]:
                continue
            flags[lead] = _lead_is_fragmented(ecg[start : end + 1, index])
        if flags:
            beat_flags.append(flags)
    if len(beat_flags) < 3:
        return None
    votes: dict[str, bool] = {}
    for lead in _ALL_STANDARD_LEADS:
        observed = [flags[lead] for flags in beat_flags if lead in flags]
        if observed:
            votes[lead] = (sum(observed) / len(observed)) > 0.5
    return votes


def _fragmented_qrs_excluded_by_wide_qrs(ctx: FeatureContext) -> bool:
    representative = ctx.delineation.representative_beat
    return bool(representative is not None) and (
        bool(getattr(representative, "wide_qrs", False))
        or bool(getattr(representative, "bundle_branch_block", False))
    )


def _fragmented_qrs_present_flag(ctx: FeatureContext) -> FeatureValue:
    """Classic Das-criteria fQRS: >=2 anatomically contiguous leads with a
    notch/extra R' wave, excluding typical wide-QRS/bundle-branch-block
    morphology (for which this narrow-QRS criterion was not validated)."""
    if _fragmented_qrs_excluded_by_wide_qrs(ctx):
        return FeatureValue(None, 0.0)
    votes = _cluster_fragmented_lead_votes(ctx, 0)
    if votes is None:
        return FeatureValue(None, 0.0)
    flagged = {lead for lead, fragmented in votes.items() if fragmented}
    present = any(
        all(lead in flagged for lead in group)
        for group in _QRS_CONTIGUOUS_LEAD_GROUPS
    )
    return FeatureValue(1.0 if present else 0.0)


def _fragmented_qrs_lead_count(ctx: FeatureContext) -> FeatureValue:
    if _fragmented_qrs_excluded_by_wide_qrs(ctx):
        return FeatureValue(None, 0.0)
    votes = _cluster_fragmented_lead_votes(ctx, 0)
    if votes is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(sum(1 for fragmented in votes.values() if fragmented)))


def _rhythm_flag(ctx: FeatureContext, flag: str) -> FeatureValue:
    """Reads a rate-aware pattern flag set by FusionDelineator._assess_rhythm
    (e.g. narrow_complex_tachycardia_pattern) as a 0/1 Discovery marker.
    RhythmClass itself has no rate dimension, so these flags are the only
    place a fast vs. normal-rate distinction is recorded."""
    return FeatureValue(1.0 if flag in ctx.delineation.rhythm.flags else 0.0)


def _narrow_complex_tachycardia_pattern_flag(ctx: FeatureContext) -> FeatureValue:
    return _rhythm_flag(ctx, "narrow_complex_tachycardia_pattern")


def _wide_complex_tachycardia_pattern_flag(ctx: FeatureContext) -> FeatureValue:
    return _rhythm_flag(ctx, "wide_complex_tachycardia_pattern")


def _accelerated_idioventricular_pattern_flag(ctx: FeatureContext) -> FeatureValue:
    return _rhythm_flag(ctx, "accelerated_idioventricular_pattern")


def _junctional_rhythm_pattern_flag(ctx: FeatureContext) -> FeatureValue:
    present = (
        "junctional_rhythm_pattern" in ctx.delineation.rhythm.flags
        or "junctional_tachycardia_pattern" in ctx.delineation.rhythm.flags
    )
    return FeatureValue(1.0 if present else 0.0)


def _sai_geh_feature(ctx: FeatureContext, interval: str) -> FeatureValue:
    """Published SAI definition: integrate each absolute XYZ lead first.

    This is deliberately *not* ``sum(abs(area_vector))``.  That expression
    permits positive and negative samples within an axis to cancel before
    the absolute value and is therefore a different quantity.  All three
    intervals share the cached, origin-corrected GEH computation.
    """

    metrics = ctx.geh_area_metrics()
    if metrics is None:
        # SAI-QRS and SAI-JT have fewer fiducial requirements than the full
        # QRST/GEH bundle.  Preserve those legitimate partial measurements
        # while still using the same origin-corrected cached segments.
        segment_name = {"qrs": "qrs", "jt": "st_t"}.get(interval)
        segment = (
            ctx.vcg_segment_origin_corrected(segment_name)
            if segment_name is not None
            else None
        )
        if segment is not None:
            return FeatureValue(
                sum_absolute_integral_mv_ms(segment, ctx.fs),
                per_lead=ctx.vcg_origin_assessment.numeric_metadata(),
                notes=(
                    ctx.vcg_origin_assessment.provenance_note()
                    + "; SAI=integral(|X|+|Y|+|Z|)dt; "
                    + f"interval={interval.upper()}"
                ),
            )
        return FeatureValue(
            None,
            0.0,
            notes=ctx.vcg_origin_assessment.provenance_note(),
        )
    values = {
        "qrs": metrics.sai_qrs_mv_ms,
        "jt": metrics.sai_jt_mv_ms,
        "qrst": metrics.sai_qrst_mv_ms,
    }
    value = values[interval]
    metadata = ctx.vcg_origin_assessment.numeric_metadata()
    metadata.update(
        {
            "sai_qrs_mv_ms": metrics.sai_qrs_mv_ms,
            "sai_jt_mv_ms": metrics.sai_jt_mv_ms,
            "sai_qrst_mv_ms": metrics.sai_qrst_mv_ms,
        }
    )
    return FeatureValue(
        float(value),
        per_lead=metadata,
        notes=(
            ctx.vcg_origin_assessment.provenance_note()
            + "; SAI=integral(|X|+|Y|+|Z|)dt"
        ),
    )


def _sum_absolute_qrs_integral(ctx: FeatureContext) -> FeatureValue:
    return _sai_geh_feature(ctx, "qrs")


def _sum_absolute_jt_integral(ctx: FeatureContext) -> FeatureValue:
    return _sai_geh_feature(ctx, "jt")


def _sum_absolute_qrst_integral(ctx: FeatureContext) -> FeatureValue:
    return _sai_geh_feature(ctx, "qrst")


def _origin_tp_pr_distance(ctx: FeatureContext) -> FeatureValue:
    assessment = ctx.vcg_origin_assessment
    if assessment.tp is None or assessment.pr is None:
        return FeatureValue(
            None,
            0.0,
            notes=(
                assessment.provenance_note()
                + "; TP-versus-PR sensitivity requires both qualified intervals"
            ),
        )
    value = float(
        np.linalg.norm(
            assessment.tp.vector_millivolts
            - assessment.pr.vector_millivolts
        )
        * 1000.0
    )
    return FeatureValue(
        value,
        confidence=min(assessment.tp.confidence, assessment.pr.confidence),
        per_lead=assessment.numeric_metadata(),
        notes=assessment.provenance_note(),
    )


def _svg_origin_sensitivity(ctx: FeatureContext) -> FeatureValue:
    assessment = ctx.vcg_origin_assessment
    metrics = ctx.geh_area_metrics()
    if metrics is None or assessment.tp is None or assessment.pr is None:
        return FeatureValue(
            None,
            0.0,
            notes=(
                assessment.provenance_note()
                + "; SVG origin sensitivity requires GEH plus qualified TP and PR"
            ),
        )
    delta_mv = assessment.tp.vector_millivolts - assessment.pr.vector_millivolts
    # Changing a constant origin by delta changes the QRST area vector by
    # delta * duration.  This norm is the conservative vector-area shift,
    # not a claim that the SVG magnitude itself shifts by exactly this much.
    value = float(
        np.linalg.norm(delta_mv)
        * metrics.qrst_duration_ms
        * 1000.0
    )
    return FeatureValue(
        value,
        confidence=min(assessment.tp.confidence, assessment.pr.confidence),
        per_lead=assessment.numeric_metadata(),
        notes=(
            assessment.provenance_note()
            + "; norm((origin_TP-origin_PR)*QRST_duration)"
        ),
    )


def _t_loop_biphasic_index(ctx: FeatureContext) -> FeatureValue:
    """T-loop analog of the existing p_loop_biphasic_index.

    Projects the T-loop onto its dominant PCA axis and reports the
    balance between positive and negative mass beyond a 10% noise-floor
    threshold (0 = fully monophasic, 1 = perfectly balanced biphasic).
    Exploratory geometry, not a repolarization-abnormality diagnosis.
    """
    points = ctx.vcg_segment("t")
    if points is None or len(points) < 4:
        return FeatureValue(None, 0.0)
    baseline = 0.5 * (points[0] + points[-1])
    vectors = points - baseline
    _, values, axes = _pca(vectors)
    if values[0] <= EPS:
        return FeatureValue(None, 0.0)
    projection = vectors @ axes[:, 0]
    threshold = 0.10 * float(np.max(np.abs(projection)))
    projection = projection[np.abs(projection) >= threshold]
    if len(projection) < 3:
        return FeatureValue(None, 0.0)
    positive_mass = float(np.sum(np.abs(projection[projection > 0])))
    negative_mass = float(np.sum(np.abs(projection[projection < 0])))
    total_mass = positive_mass + negative_mass
    return FeatureValue(float(2.0 * min(positive_mass, negative_mass) / (total_mass + EPS)))


_PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")


def _beat_maximum_deflection_index(
    beat,
    ecg_12lead: np.ndarray,
) -> Optional[float]:
    """Time-to-earliest-maximal-deflection / QRS duration, in the
    precordial lead of maximum QRS amplitude for this beat.

    Daniels et al. (Circulation 2006) defined this ratio (>=0.55 in their
    cohort) to separate epicardial from endocardial ventricular ectopy
    origin, building on Berruezo et al.'s (Circulation 2004) intrinsicoid-
    deflection-timing observation.
    """
    r_onset = getattr(beat, "r_onset", None)
    r_offset = getattr(beat, "r_offset", None)
    if r_onset is None or r_offset is None:
        return None
    start, end = int(r_onset.sample), int(r_offset.sample)
    duration = end - start
    if duration < 4 or start < 0 or end >= ecg_12lead.shape[0]:
        return None
    best_lead_index: Optional[int] = None
    best_amplitude = -1.0
    for lead in _PRECORDIAL_LEADS:
        index = LEAD_MAP.get(lead)
        if index is None or index >= ecg_12lead.shape[1]:
            continue
        segment = ecg_12lead[start : end + 1, index]
        amplitude = float(np.max(segment) - np.min(segment))
        if amplitude > best_amplitude:
            best_amplitude = amplitude
            best_lead_index = index
    if best_lead_index is None or best_amplitude <= EPS:
        return None
    segment = ecg_12lead[start : end + 1, best_lead_index]
    peak_offset = int(np.argmax(np.abs(segment)))
    return float(peak_offset / duration)


def _maximum_deflection_index_for_cluster(
    ctx: FeatureContext,
    cluster_id: int,
) -> Optional[float]:
    ecg = ctx._record_ecg_12lead
    if ecg is None:
        return None
    beats = _cluster_beats_or_accepted(ctx, cluster_id)
    values = [
        value
        for value in (_beat_maximum_deflection_index(beat, ecg) for beat in beats)
        if value is not None
    ]
    if len(values) < 3:
        return None
    return float(np.median(values))


def _primary_morphology_maximum_deflection_index(ctx: FeatureContext) -> FeatureValue:
    value = _maximum_deflection_index_for_cluster(ctx, 0)
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _secondary_morphology_maximum_deflection_index(ctx: FeatureContext) -> FeatureValue:
    value = _maximum_deflection_index_for_cluster(ctx, 1)
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _secondary_morphology_maximum_deflection_index_delta(
    ctx: FeatureContext,
) -> FeatureValue:
    primary = _maximum_deflection_index_for_cluster(ctx, 0)
    secondary = _maximum_deflection_index_for_cluster(ctx, 1)
    if primary is None or secondary is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(secondary - primary)


def _cluster_beats_or_accepted(ctx: FeatureContext, cluster_id: int = 0) -> list:
    """Beats belonging to one FusionDelineator morphology cluster.

    For cluster 0 (primary/dominant), prefers the explicit cluster
    assignment but falls back to the legacy accepted/non-outlier set when
    cluster information was not established (too few beats, or no coherent
    cluster at all) -- preserving the original behaviour this helper had
    before secondary-cluster support existed. For cluster 1/2
    (secondary/tertiary), no fallback is used: an absent or too-small
    cluster simply yields an empty/short list, and callers abstain via
    their own minimum-count checks rather than silently substituting a
    different population.
    """
    members = [
        beat
        for beat in ctx.delineation.beats
        if getattr(beat, "morphology_cluster_id", None) == cluster_id
    ]
    if cluster_id != 0 or len(members) >= 3:
        return members
    return [
        beat
        for beat in ctx.delineation.accepted_beats
        if not bool(getattr(beat, "morphology_outlier", False))
    ]


def _primary_cluster_or_accepted_beats(ctx: FeatureContext) -> list:
    return _cluster_beats_or_accepted(ctx, 0)


def _beat_to_beat_mean_absolute_difference(
    values: list[float],
) -> Optional[float]:
    if len(values) < 3:
        return None
    diffs = np.abs(np.diff(np.asarray(values, dtype=float)))
    return float(np.mean(diffs))


def _t_amplitude_variability_for_cluster(
    ctx: FeatureContext,
    cluster_id: int,
) -> FeatureValue:
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    beats = sorted(
        _cluster_beats_or_accepted(ctx, cluster_id),
        key=lambda beat: beat.r_peak.sample,
    )
    amplitudes: list[float] = []
    for beat in beats:
        t_peak = getattr(beat, "t_peak", None)
        if (
            t_peak is None
            or t_peak.confidence < fiducial_confidence_threshold("t_peak")
            or beat.resolved_wave_status("t").value != "present"
            or not 0 <= t_peak.sample < len(record_vcg)
        ):
            continue
        amplitudes.append(float(np.linalg.norm(record_vcg[t_peak.sample])))
    value = _beat_to_beat_mean_absolute_difference(amplitudes)
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _t_amplitude_beat_to_beat_variability(ctx: FeatureContext) -> FeatureValue:
    return _t_amplitude_variability_for_cluster(ctx, 0)


def _secondary_t_amplitude_beat_to_beat_variability(
    ctx: FeatureContext,
) -> FeatureValue:
    return _t_amplitude_variability_for_cluster(ctx, 1)


def _st_level_variability_for_cluster(
    ctx: FeatureContext,
    cluster_id: int,
) -> FeatureValue:
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return FeatureValue(None, 0.0)
    beats = sorted(
        _cluster_beats_or_accepted(ctx, cluster_id),
        key=lambda beat: beat.r_peak.sample,
    )
    levels: list[float] = []
    for beat in beats:
        j_point = getattr(beat, "j_point", None) or getattr(beat, "r_offset", None)
        if j_point is None or j_point.confidence < fiducial_confidence_threshold(
            "j_point"
        ):
            continue
        st_sample = int(j_point.sample) + int(round(0.060 * ctx.fs))
        if not 0 <= st_sample < len(record_vcg):
            continue
        levels.append(float(np.linalg.norm(record_vcg[st_sample])))
    value = _beat_to_beat_mean_absolute_difference(levels)
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _st_level_beat_to_beat_variability(ctx: FeatureContext) -> FeatureValue:
    return _st_level_variability_for_cluster(ctx, 0)


def _secondary_st_level_beat_to_beat_variability(
    ctx: FeatureContext,
) -> FeatureValue:
    return _st_level_variability_for_cluster(ctx, 1)


def _qt_variability_for_cluster(
    ctx: FeatureContext,
    cluster_id: int,
) -> FeatureValue:
    beats = ctx.delineation.beats
    # The producer sorts by R time but keeps original beat_index identities.
    # Index adjacency is temporal adjacency only for a strictly ordered input.
    peaks = [getattr(beat, "r_peak", None) for beat in beats]
    if any(peak is None or not np.isfinite(peak.sample) for peak in peaks):
        return FeatureValue(None, 0.0, notes="missing chronological beat identity")
    if len(peaks) > 1 and np.any(np.diff([peak.sample for peak in peaks]) <= 0):
        return FeatureValue(None, 0.0, notes="nonmonotonic chronological beat identity")
    measurements = ctx.delineation.qt_rr_measurements()
    if not measurements:
        return FeatureValue(None, 0.0)
    indices = [item.beat_index for item in measurements]
    if len(indices) != len(set(indices)):
        return FeatureValue(None, 0.0, notes="duplicate QT measurement identity")
    qualified = {
        item.beat_index: item.qt_ms for item in measurements
        if 0 <= item.beat_index < len(beats)
        and getattr(beats[item.beat_index], "morphology_cluster_id", None) == cluster_id
        and np.isfinite(item.qt_ms) and item.qt_ms > 0
        and np.isfinite(item.confidence) and item.confidence >= 0.35
    }
    # Missing measurements or intervening morphologies are not adjacent beats.
    diffs = [abs(value - qualified[index - 1]) for index, value in qualified.items()
             if index - 1 in qualified]
    value = float(np.mean(diffs)) if len(diffs) >= 2 else None
    return FeatureValue(None, 0.0) if value is None else FeatureValue(value)


def _qt_beat_to_beat_variability(ctx: FeatureContext) -> FeatureValue:
    return _qt_variability_for_cluster(ctx, 0)


def _secondary_qt_beat_to_beat_variability(ctx: FeatureContext) -> FeatureValue:
    return _qt_variability_for_cluster(ctx, 1)


def _beatwise_svg_direction_dispersion(ctx: FeatureContext) -> FeatureValue:
    qrs_vectors = _beat_area_vectors_by_identity(ctx, "qrs")
    # The area-T component of GEH/SVG is JT (QRS offset to T offset).
    jt_vectors = _beat_area_vectors_by_identity(ctx, "jt")
    paired = sorted(set(qrs_vectors).intersection(jt_vectors))
    if len(paired) < 3:
        return FeatureValue(
            None,
            0.0,
            notes=(
                "not_evaluable: fewer than three beat-identity-matched "
                "QRS/JT area-vector pairs"
            ),
        )
    svg = [
        qrs_vectors[identity] + jt_vectors[identity]
        for identity in paired
        if np.linalg.norm(
            qrs_vectors[identity] + jt_vectors[identity]
        ) > EPS
    ]
    if len(svg) < 3:
        return FeatureValue(None, 0.0)
    unit = np.asarray([vector / np.linalg.norm(vector) for vector in svg])
    reference = np.sum(unit, axis=0)
    if np.linalg.norm(reference) <= EPS:
        return FeatureValue(180.0)
    reference /= np.linalg.norm(reference)
    angles = np.degrees(
        np.arccos(np.clip(unit @ reference, -1.0, 1.0))
    )
    return FeatureValue(
        float(np.sqrt(np.mean(angles ** 2))),
        per_lead={"paired_beats": float(len(svg))},
        notes="Beat-identity-matched QRS+JT SVG directions.",
    )


def _u_peak_vector_magnitude(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment("u")
    if points is None or len(points) < 3:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.max(np.linalg.norm(points, axis=1))))


def _loop_area_vector(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points is None or len(points) < 4:
        return None
    centered = np.asarray(points, dtype=float) - np.mean(points, axis=0)
    closed = np.vstack((centered, centered[0]))
    return 0.5 * np.sum(np.cross(closed[:-1], closed[1:]), axis=0)


def _u_loop_area_vector_magnitude(ctx: FeatureContext) -> FeatureValue:
    vector = _loop_area_vector(ctx.vcg_segment("u"))
    return FeatureValue(
        None if vector is None else float(np.linalg.norm(vector))
    )


def _u_to_t_peak_magnitude_ratio(ctx: FeatureContext) -> FeatureValue:
    u = ctx.vcg_segment("u")
    t = ctx.vcg_segment("t")
    if u is None or t is None or len(u) < 3 or len(t) < 3:
        return FeatureValue(None, 0.0)
    denominator = float(np.max(np.linalg.norm(t, axis=1)))
    if denominator <= EPS:
        return FeatureValue(None, 0.0)
    return FeatureValue(
        float(np.max(np.linalg.norm(u, axis=1)) / denominator)
    )


def _t_u_area_vector_angle(ctx: FeatureContext) -> FeatureValue:
    t_vector = _loop_area_vector(ctx.vcg_segment("t"))
    u_vector = _loop_area_vector(ctx.vcg_segment("u"))
    if t_vector is None or u_vector is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(_angle_between(t_vector, u_vector))


def _u_loop_planarity(ctx: FeatureContext) -> FeatureValue:
    points = ctx.vcg_segment("u")
    if points is None or len(points) < 5:
        return FeatureValue(None, 0.0)
    centered = points - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = float(np.sum(energy))
    if total <= EPS:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.clip(1.0 - energy[-1] / total, 0.0, 1.0)))


def _t_u_transition_angle(ctx: FeatureContext) -> FeatureValue:
    support = max(3, int(round(20.0 * ctx.fs / 1000.0)))
    t_direction = _edge_direction(
        ctx.vcg_segment("t"),
        from_start=False,
        samples=support,
    )
    u_direction = _edge_direction(
        ctx.vcg_segment("u"),
        from_start=True,
        samples=support,
    )
    if t_direction is None or u_direction is None:
        return FeatureValue(None, 0.0)
    return FeatureValue(_angle_between(t_direction, u_direction))


ADVANCED_SPECS: tuple[_AdvancedSpec, ...] = (
    _AdvancedSpec(
        "qrs_t_area_angle_direct_8lead_deg",
        "Direct native-lead QRS-T geometry",
        "Transformation-free QRS-T angle between signed QRS and T area "
        "vectors in the eight algebraically independent ECG leads.",
        "angle(integral(QRS_[I,II,V1..V6]), integral(T_[I,II,V1..V6]))",
        "degrees",
        ("qrs", "t"),
        _qrs_t_area_angle_direct_8lead,
        (DIRECT_QRS_T_REFERENCE,),
    ),
    _AdvancedSpec(
        "qrs_t_integral_angle_direct_8lead_deg",
        "Direct native-lead QRS-T geometry",
        "Transformation-free magnitude-weighted integrative QRS-T angle "
        "that retains within-wave morphology in the eight independent leads.",
        "weighted mean of angle(q(t),t(u)) over QRS x J-to-Tend; "
        "weights=norm(q(t))*norm(t(u))",
        "degrees",
        ("qrs", "t"),
        _qrs_t_integral_angle_direct_8lead,
        (DIRECT_QRS_T_REFERENCE,),
    ),
    _AdvancedSpec(
        "qrs_total_turning_angle_deg",
        "Post-development QRS geometry",
        "Total robust turning of the spatial QRS trajectory.",
        "sum angular changes of equal-arc, smoothed QRS tangents",
        "degrees",
        ("qrs",),
        _qrs_total_turning,
    ),
    _AdvancedSpec(
        "qrs_turning_angle_p95_deg",
        "Post-development QRS geometry",
        "Upper-tail local turn severity of the spatial QRS trajectory.",
        "95th percentile of equal-arc QRS tangent changes",
        "degrees",
        ("qrs",),
        _qrs_turning_p95,
    ),
    _AdvancedSpec(
        "qrs_terminal_energy_fraction_20ms",
        "Post-development terminal QRS",
        "Fraction of QRS vector energy concentrated in the final 20 ms.",
        "integral(final20ms ||r||^2) / integral(QRS ||r||^2)",
        "fraction",
        ("qrs",),
        _qrs_terminal_energy_fraction,
    ),
    _AdvancedSpec(
        "qrs_initial_terminal_energy_balance",
        "Post-development terminal QRS",
        "Signed balance of terminal versus initial 20-ms QRS energy.",
        "(terminal20 energy fraction - initial20) / (terminal20 + initial20)",
        "dimensionless",
        ("qrs",),
        _qrs_initial_terminal_energy_balance,
    ),
    _AdvancedSpec(
        "qrs_t_normalized_hausdorff_distance",
        "Post-development QRS-T coupling",
        "Amplitude-normalized extreme geometric QRS-T separation.",
        "Hausdorff(centered RMS-normalized QRS, centered RMS-normalized T)",
        "dimensionless",
        ("qrs", "t"),
        _qrs_t_normalized_hausdorff,
    ),
    _AdvancedSpec(
        "qrs_t_normalized_dtw_distance",
        "Post-development QRS-T coupling",
        "Amplitude-normalized time-warped QRS-T shape separation.",
        "DTW(centered RMS-normalized QRS, centered RMS-normalized T)",
        "dimensionless",
        ("qrs", "t"),
        _qrs_t_normalized_dtw,
    ),
    _AdvancedSpec(
        "qrs_t_mirror_phase_angle_median_deg",
        "Post-development QRS-T coupling",
        "Median phase-wise angle between QRS and sign-inverted T vectors.",
        "median angle(qrs_phase, -t_phase)",
        "degrees",
        ("qrs", "t"),
        _mirror_angle_median,
    ),
    _AdvancedSpec(
        "qrs_t_mirror_phase_angle_p90_deg",
        "Post-development QRS-T coupling",
        "Upper-tail phase-wise QRS versus sign-inverted T discordance.",
        "90th percentile angle(qrs_phase, -t_phase)",
        "degrees",
        ("qrs", "t"),
        _mirror_angle_p90,
    ),
    _AdvancedSpec(
        "qrs_t_mirror_discordant_fraction",
        "Post-development QRS-T coupling",
        "Fraction of phases more than 90 degrees from QRS/T mirror alignment.",
        "mean(angle(qrs_phase, -t_phase) > 90 degrees)",
        "fraction",
        ("qrs", "t"),
        _mirror_discordant_fraction,
    ),
    _AdvancedSpec(
        "svg_cancellation_index",
        "Post-development ventricular gradient",
        "Cancellation between QRS and T area vectors.",
        "1 - ||QRS_area + T_area||/(||QRS_area|| + ||T_area||)",
        "fraction",
        ("qrs", "t"),
        _svg_cancellation,
    ),
    _AdvancedSpec(
        "t_terminal_path_fraction",
        "Post-development T geometry",
        "Fraction of T-loop path traversed in the final 20% of T duration.",
        "path(final temporal 20% T) / path(T)",
        "fraction",
        ("t",),
        _t_terminal_path_fraction,
    ),
    _AdvancedSpec(
        "t_terminal_direction_dispersion_deg",
        "Post-development T geometry",
        "Directional dispersion of the terminal T-loop trajectory.",
        "RMS angular spread of final-20%-T tangents around their resultant",
        "degrees",
        ("t",),
        _t_terminal_direction_dispersion,
    ),
    _AdvancedSpec(
        "st_t_transition_angle_deg",
        "Post-development ST-T coupling",
        "Angular continuity break from terminal ST to initial T trajectory.",
        "angle(last-20ms ST tangent, first-20ms T tangent)",
        "degrees",
        ("st", "t"),
        _st_t_transition_angle,
    ),
    _AdvancedSpec(
        "qrs_beat_area_axis_coherence",
        "Post-development beat stability",
        "Beat-to-beat directional coherence of QRS area vectors.",
        "norm(mean(unit QRS area vectors across evaluable beats))",
        "fraction",
        ("qrs",),
        _qrs_beat_area_axis_coherence,
    ),
    _AdvancedSpec(
        "t_beat_area_axis_coherence",
        "Post-development beat stability",
        "Beat-to-beat directional coherence of T area vectors.",
        "norm(mean(unit T area vectors across evaluable beats))",
        "fraction",
        ("t",),
        _t_beat_area_axis_coherence,
    ),
    _AdvancedSpec(
        "beatwise_svg_direction_dispersion_deg",
        "Post-development beat stability",
        "Beat-to-beat directional dispersion of identity-matched Wilson SVGs.",
        "RMS angle of beat-wise (QRS area + JT area) unit vectors around their resultant",
        "degrees",
        ("qrs", "st_t"),
        _beatwise_svg_direction_dispersion,
    ),
    _AdvancedSpec(
        "u_peak_vector_magnitude_mv",
        "Post-development U-wave geometry",
        "Peak spatial magnitude of a separately delineated U wave.",
        "max(||VCG_U(t)||)",
        "mV",
        ("u",),
        _u_peak_vector_magnitude,
    ),
    _AdvancedSpec(
        "u_loop_area_vector_magnitude_mv2",
        "Post-development U-wave geometry",
        "Magnitude of the oriented spatial U-loop area vector.",
        "0.5 * ||sum((u_i-c) cross (u_(i+1)-c))||",
        "mV^2",
        ("u",),
        _u_loop_area_vector_magnitude,
    ),
    _AdvancedSpec(
        "u_to_t_peak_magnitude_ratio",
        "Post-development T-U coupling",
        "Relative peak vector magnitude of U compared with T.",
        "max(||U||) / max(||T||)",
        "ratio",
        ("t", "u"),
        _u_to_t_peak_magnitude_ratio,
    ),
    _AdvancedSpec(
        "t_u_area_vector_angle_deg",
        "Post-development T-U coupling",
        "Angle between oriented T- and U-loop area vectors.",
        "angle(T loop area vector, U loop area vector)",
        "degrees",
        ("t", "u"),
        _t_u_area_vector_angle,
    ),
    _AdvancedSpec(
        "u_loop_planarity",
        "Post-development U-wave geometry",
        "Fraction of U-loop energy explained by its best spatial plane.",
        "1 - smallest SVD energy / total SVD energy",
        "fraction",
        ("u",),
        _u_loop_planarity,
    ),
    _AdvancedSpec(
        "t_u_transition_angle_deg",
        "Post-development T-U coupling",
        "Directional change from terminal T to initial U trajectory.",
        "angle(last-20ms T tangent, first-20ms U tangent)",
        "degrees",
        ("t", "u"),
        _t_u_transition_angle,
    ),
    _AdvancedSpec(
        "secondary_morphology_beat_fraction",
        "Post-development multi-morphology clustering",
        "Fraction of candidate beats explained by a coherent secondary QRS "
        "population distinct from the record's dominant template.",
        "secondary_cluster_beat_count / candidate_beat_count",
        "fraction",
        ("qrs",),
        _secondary_morphology_beat_fraction,
    ),
    _AdvancedSpec(
        "secondary_morphology_centroid_distance_mv",
        "Post-development multi-morphology clustering",
        "Waveform-amplitude distance between the secondary and primary "
        "morphology cluster templates.",
        "RMS(median(raw secondary-cluster QRS window) "
        "- median(raw primary-cluster QRS window))",
        "mV",
        ("qrs",),
        _secondary_morphology_centroid_distance,
    ),
    _AdvancedSpec(
        "secondary_morphology_mean_coupling_ratio",
        "Post-development multi-morphology clustering",
        "Mean RR-coupling ratio of the secondary morphology cluster's beats.",
        "mean(coupling_ratio) over secondary-cluster beats",
        "ratio",
        ("qrs",),
        _secondary_morphology_mean_coupling_ratio,
    ),
    _AdvancedSpec(
        "secondary_morphology_frontal_axis_delta_deg",
        "Post-development multi-morphology clustering",
        "Frontal-plane QRS axis shift of the secondary morphology cluster "
        "relative to the primary/dominant cluster.",
        "angle(mean frontal-plane QRS area vector[primary], "
        "mean frontal-plane QRS area vector[secondary])",
        "degrees",
        ("qrs",),
        _secondary_morphology_frontal_axis_delta,
    ),
    _AdvancedSpec(
        "t_amplitude_beat_to_beat_variability",
        "Post-development beat-to-beat repolarization variability",
        "Beat-to-beat variability of repolarization (BVR): successive "
        "T-peak vector-magnitude change across the record's dominant, "
        "non-ectopic beat population.",
        "mean(|diff(T-peak vector magnitude)|) over primary-cluster beats",
        "mV",
        ("t",),
        _t_amplitude_beat_to_beat_variability,
    ),
    _AdvancedSpec(
        "st_level_beat_to_beat_variability",
        "Post-development beat-to-beat repolarization variability",
        "Beat-to-beat variability of repolarization (BVR): successive "
        "ST-level (J+60ms) vector-magnitude change across the record's "
        "dominant, non-ectopic beat population.",
        "mean(|diff(vector magnitude at J+60ms)|) over primary-cluster beats",
        "mV",
        ("st", "t"),
        _st_level_beat_to_beat_variability,
    ),
    _AdvancedSpec(
        "qt_beat_to_beat_variability_ms",
        "Post-development beat-to-beat repolarization variability",
        "Beat-to-beat variability of repolarization (BVR): successive "
        "RR-paired QT change across the record's dominant, non-ectopic "
        "beat population.",
        "mean(|QT[i]-QT[i-1]|) for adjacent qualified primary-cluster beats; at least two pairs",
        "ms",
        ("qrs", "t"),
        _qt_beat_to_beat_variability,
    ),
    _AdvancedSpec(
        "secondary_morphology_frontal_axis_deg",
        "Post-development multi-morphology clustering",
        "Absolute frontal-plane QRS axis of the secondary morphology "
        "cluster's mean area vector.",
        "atan2(mean frontal-plane QRS area vector[secondary])",
        "degrees",
        ("qrs",),
        _secondary_morphology_frontal_axis,
    ),
    _AdvancedSpec(
        "secondary_morphology_compensatory_pause_ratio",
        "Post-development multi-morphology clustering",
        "Completeness of the post-ectopic compensatory pause: near 1.0 is "
        "consistent with a classic PVC (undisturbed sinus node), "
        "distinctly below 1.0 with an early sinus reset (PAC-like).",
        "median((rr_before + rr_after) / (2 * local_baseline_rr)) over "
        "qualifying secondary-cluster beats",
        "ratio",
        ("qrs",),
        _secondary_morphology_compensatory_pause_ratio,
    ),
    _AdvancedSpec(
        "secondary_morphology_t_amplitude_beat_to_beat_variability",
        "Post-development multi-morphology clustering",
        "Beat-to-beat variability of T-peak vector magnitude within the "
        "secondary (ectopic) morphology cluster itself.",
        "mean(|diff(T-peak vector magnitude)|) over secondary-cluster beats",
        "mV",
        ("qrs", "t"),
        _secondary_t_amplitude_beat_to_beat_variability,
    ),
    _AdvancedSpec(
        "secondary_morphology_qt_beat_to_beat_variability_ms",
        "Post-development multi-morphology clustering",
        "Beat-to-beat variability of RR-paired QT within the secondary "
        "(ectopic) morphology cluster itself.",
        "mean(|QT[i]-QT[i-1]|) for adjacent qualified secondary-cluster beats; at least two pairs",
        "ms",
        ("qrs", "t"),
        _secondary_qt_beat_to_beat_variability,
    ),
    _AdvancedSpec(
        "secondary_morphology_st_level_beat_to_beat_variability",
        "Post-development multi-morphology clustering",
        "Beat-to-beat variability of ST-level (J+60ms) vector magnitude "
        "within the secondary (ectopic) morphology cluster itself.",
        "mean(|diff(vector magnitude at J+60ms)|) over secondary-cluster beats",
        "mV",
        ("qrs", "st", "t"),
        _secondary_st_level_beat_to_beat_variability,
    ),
    _AdvancedSpec(
        "secondary_morphology_qt_delta_ms",
        "Post-development multi-morphology clustering",
        "Difference in mean QT interval between the secondary and primary "
        "morphology cluster's beats.",
        "mean(qt_ms[secondary]) - mean(qt_ms[primary])",
        "ms",
        ("qrs", "t"),
        _secondary_morphology_qt_delta,
    ),
    _AdvancedSpec(
        "secondary_morphology_qrs_duration_delta_ms",
        "Post-development multi-morphology clustering",
        "Difference in mean QRS duration between the secondary and primary "
        "morphology cluster's beats.",
        "mean(qrs_duration_ms[secondary]) - mean(qrs_duration_ms[primary])",
        "ms",
        ("qrs",),
        _secondary_morphology_qrs_duration_delta,
    ),
    _AdvancedSpec(
        "secondary_morphology_qrs_area_vector_magnitude_delta",
        "Post-development multi-morphology clustering",
        "Difference in mean QRS area-vector magnitude (a loop-size proxy) "
        "between the secondary and primary morphology cluster's beats.",
        "mean(||qrs_area_vector[secondary]||) - mean(||qrs_area_vector[primary]||)",
        "mV*s",
        ("qrs",),
        _secondary_morphology_qrs_area_vector_magnitude_delta,
    ),
    _AdvancedSpec(
        "post_ectopic_t_amplitude_delta_mv",
        "Post-development cardiac memory",
        "Cardiac-memory candidate: T-peak vector-magnitude change in the "
        "sinus beat immediately after an ectopic beat, relative to "
        "non-post-ectopic primary-cluster beats.",
        "mean(T-peak magnitude[post-ectopic]) - mean(T-peak magnitude[baseline])",
        "mV",
        ("qrs", "t"),
        _post_ectopic_t_amplitude_delta,
    ),
    _AdvancedSpec(
        "post_ectopic_t_axis_delta_deg",
        "Post-development cardiac memory",
        "Cardiac-memory candidate: T-peak vector direction change in the "
        "sinus beat immediately after an ectopic beat, relative to "
        "non-post-ectopic primary-cluster beats.",
        "angle(mean T-peak vector[post-ectopic], mean T-peak vector[baseline])",
        "degrees",
        ("qrs", "t"),
        _post_ectopic_t_axis_delta,
    ),
    _AdvancedSpec(
        "secondary_morphology_coupling_variability_cv",
        "Post-development parasystole consistency",
        "Coefficient of variation of the coupling ratio across the "
        "secondary morphology cluster's beats -- high is more consistent "
        "with variable/parasystolic coupling, low with fixed reentry.",
        "std(coupling_ratio, ddof=1) / mean(coupling_ratio) over "
        "secondary-cluster beats",
        "ratio",
        ("qrs",),
        _secondary_morphology_coupling_variability,
    ),
    _AdvancedSpec(
        "secondary_morphology_common_denominator_fit",
        "Post-development parasystole consistency",
        "How well inter-ectopic intervals fit integer multiples of a "
        "common base cycle (parasystole's rule of common denominator).",
        "1 - mean(|interval - round(interval/base)*base| / base) over "
        "inter-ectopic intervals, base = shortest observed interval",
        "fraction",
        ("qrs",),
        _secondary_morphology_common_denominator_fit,
    ),
    _AdvancedSpec(
        "primary_morphology_maximum_deflection_index",
        "Post-development activation origin",
        "Maximum Deflection Index of the primary/dominant morphology "
        "cluster (time to earliest maximal QRS deflection / QRS duration, "
        "in the precordial lead of maximum amplitude).",
        "median(argmax(|segment|)/duration) over primary-cluster beats, "
        "precordial lead of max peak-to-peak amplitude",
        "ratio",
        ("qrs",),
        _primary_morphology_maximum_deflection_index,
    ),
    _AdvancedSpec(
        "secondary_morphology_maximum_deflection_index",
        "Post-development activation origin",
        "Maximum Deflection Index of the secondary morphology cluster -- "
        "a markedly higher value than the primary cluster's is consistent "
        "with an epicardial/more distant activation origin for the "
        "ectopic beats (Daniels et al., Circulation 2006), a hypothesis "
        "not a diagnosis.",
        "median(argmax(|segment|)/duration) over secondary-cluster beats, "
        "precordial lead of max peak-to-peak amplitude",
        "ratio",
        ("qrs",),
        _secondary_morphology_maximum_deflection_index,
    ),
    _AdvancedSpec(
        "secondary_morphology_maximum_deflection_index_delta",
        "Post-development activation origin",
        "Difference in Maximum Deflection Index between the secondary and "
        "primary morphology cluster.",
        "secondary_morphology_maximum_deflection_index - "
        "primary_morphology_maximum_deflection_index",
        "ratio",
        ("qrs",),
        _secondary_morphology_maximum_deflection_index_delta,
    ),
    _AdvancedSpec(
        "sum_absolute_qrs_integral_mv_ms",
        "Post-development ventricular gradient",
        "Sum Absolute QRS Integral (SAI QRS), with samplewise absolute "
        "values on all three VCG axes.",
        "integral_QRS (|X(t)| + |Y(t)| + |Z(t)|) dt",
        "mV*ms",
        ("qrs",),
        _sum_absolute_qrs_integral,
    ),
    _AdvancedSpec(
        "sum_absolute_jt_integral_mv_ms",
        "Post-development ventricular gradient",
        "Sum Absolute JT Integral (SAI JT), including the complete interval "
        "from QRS offset through T offset.",
        "integral_JT (|X(t)| + |Y(t)| + |Z(t)|) dt",
        "mV*ms",
        ("st_t",),
        _sum_absolute_jt_integral,
    ),
    _AdvancedSpec(
        "sum_absolute_qrst_integral_mv_ms",
        "Post-development ventricular gradient",
        "Sum Absolute QRST Integral (SAI QRST): samplewise L1 integral, "
        "distinct from the L1 norm of the signed SVG area vector.",
        "integral_QRS-onset:T-offset (|X(t)| + |Y(t)| + |Z(t)|) dt",
        "mV*ms",
        ("qrs", "st_t"),
        _sum_absolute_qrst_integral,
    ),
    _AdvancedSpec(
        "vcg_origin_tp_pr_distance_uv",
        "Post-development origin robustness",
        "Distance between independently qualified TP and PR VCG origins; "
        "abstains unless both intervals are available.",
        "1000 * norm(origin_TP - origin_PR)",
        "uV",
        ("pr",),
        _origin_tp_pr_distance,
    ),
    _AdvancedSpec(
        "svg_origin_sensitivity_uv_ms",
        "Post-development origin robustness",
        "Conservative SVG-vector area shift implied by choosing the qualified "
        "PR origin instead of the preferred TP origin.",
        "1000 * QRST_duration_ms * norm(origin_TP - origin_PR)",
        "uV*ms",
        ("qrs", "st_t", "pr"),
        _svg_origin_sensitivity,
    ),
    _AdvancedSpec(
        "t_loop_biphasic_index",
        "Post-development T geometry",
        "Balance between positive and negative T-loop mass along its "
        "dominant PCA axis (0 = monophasic, 1 = balanced biphasic).",
        "2*min(positive_mass, negative_mass) / total_mass, "
        "PCA axis-1 projection beyond a 10% noise floor",
        "fraction",
        ("t",),
        _t_loop_biphasic_index,
    ),
    _AdvancedSpec(
        "primary_morphology_qrs_octant_id",
        "Post-development activation origin",
        "Coarse 3D sign-octant (0-7) of the primary morphology cluster's "
        "mean QRS area vector.",
        "bit_pack(sign(x)>0, sign(y)>0, sign(z)>0) of mean QRS area vector",
        "index",
        ("qrs",),
        _primary_morphology_qrs_octant_id,
    ),
    _AdvancedSpec(
        "secondary_morphology_qrs_octant_id",
        "Post-development activation origin",
        "Coarse 3D sign-octant (0-7) of the secondary morphology "
        "cluster's mean QRS area vector.",
        "bit_pack(sign(x)>0, sign(y)>0, sign(z)>0) of mean QRS area vector",
        "index",
        ("qrs",),
        _secondary_morphology_qrs_octant_id,
    ),
    _AdvancedSpec(
        "qrs_axis_kors_vs_dower_delta_deg",
        "Post-development transform robustness",
        "Angle between the QRS area vector derived via the Kors transform "
        "and via the inverse-Dower transform, from the same raw 12-lead "
        "signal -- a per-record VCG-derivation sensitivity indicator.",
        "angle(qrs_area_vector[kors], qrs_area_vector[inverse_dower])",
        "degrees",
        ("qrs",),
        _qrs_axis_kors_vs_dower_delta,
    ),
    _AdvancedSpec(
        "t_axis_kors_vs_dower_delta_deg",
        "Post-development transform robustness",
        "Angle between the T area vector derived via the Kors transform "
        "and via the inverse-Dower transform, from the same raw 12-lead "
        "signal -- a per-record VCG-derivation sensitivity indicator.",
        "angle(t_area_vector[kors], t_area_vector[inverse_dower])",
        "degrees",
        ("t",),
        _t_axis_kors_vs_dower_delta,
    ),
    _AdvancedSpec(
        "rr_poincare_sd1_ms",
        "Post-development Poincare dispersion",
        "Short-term (beat-to-beat) RR-interval Poincare dispersion.",
        "sqrt(var(diff(rr_ms)) / 2)",
        "ms",
        ("qrs",),
        _rr_poincare_sd1,
    ),
    _AdvancedSpec(
        "rr_poincare_sd2_ms",
        "Post-development Poincare dispersion",
        "Longer-term RR-interval Poincare dispersion.",
        "sqrt(2*var(rr_ms) - sd1^2)",
        "ms",
        ("qrs",),
        _rr_poincare_sd2,
    ),
    _AdvancedSpec(
        "qrs_magnitude_poincare_sd1_mv",
        "Post-development Poincare dispersion",
        "Short-term Poincare dispersion of successive QRS peak vector "
        "magnitudes -- a novel repurposing of the standard RR-interval "
        "Poincare method onto beat morphology magnitude.",
        "sqrt(var(diff(qrs_peak_magnitude)) / 2), primary-cluster beats",
        "mV",
        ("qrs",),
        _qrs_magnitude_poincare_sd1,
    ),
    _AdvancedSpec(
        "qrs_magnitude_poincare_sd2_mv",
        "Post-development Poincare dispersion",
        "Longer-term Poincare dispersion of successive QRS peak vector "
        "magnitudes.",
        "sqrt(2*var(qrs_peak_magnitude) - sd1^2), primary-cluster beats",
        "mV",
        ("qrs",),
        _qrs_magnitude_poincare_sd2,
    ),
    _AdvancedSpec(
        "t_magnitude_poincare_sd1_mv",
        "Post-development Poincare dispersion",
        "Short-term Poincare dispersion of successive T peak vector "
        "magnitudes.",
        "sqrt(var(diff(t_peak_magnitude)) / 2), primary-cluster beats",
        "mV",
        ("t",),
        _t_magnitude_poincare_sd1,
    ),
    _AdvancedSpec(
        "t_magnitude_poincare_sd2_mv",
        "Post-development Poincare dispersion",
        "Longer-term Poincare dispersion of successive T peak vector "
        "magnitudes.",
        "sqrt(2*var(t_peak_magnitude) - sd1^2), primary-cluster beats",
        "mV",
        ("t",),
        _t_magnitude_poincare_sd2,
    ),
    _AdvancedSpec(
        "within_record_qrs_axis_drift_deg",
        "Post-development within-record drift",
        "QRS axis change between the first and second half of the "
        "recording's primary-cluster beats.",
        "angle(mean QRS area vector[first half], mean QRS area vector[second half])",
        "degrees",
        ("qrs",),
        _within_record_qrs_axis_drift,
    ),
    _AdvancedSpec(
        "within_record_qrs_amplitude_drift_mv",
        "Post-development within-record drift",
        "QRS peak vector-magnitude change between the first and second "
        "half of the recording's primary-cluster beats.",
        "mean(QRS peak magnitude[second half]) - mean(QRS peak magnitude[first half])",
        "mV",
        ("qrs",),
        _within_record_qrs_amplitude_drift,
    ),
    _AdvancedSpec(
        "within_record_qtc_drift_ms",
        "Post-development within-record drift",
        "Bazett-corrected QT change between the first and second half of "
        "the recording's primary-cluster beats.",
        "mean(qtc_ms[second half]) - mean(qtc_ms[first half])",
        "ms",
        ("qrs", "t"),
        _within_record_qtc_drift,
    ),
    _AdvancedSpec(
        "narrow_complex_tachycardia_pattern_flag",
        "Post-development rate-aware rhythm pattern",
        "Regular, narrow-QRS rhythm at >=150bpm -- a technical SVT-compatible "
        "rate/QRS-width pattern, not an SVT diagnosis.",
        "rhythm_class==REGULAR and not majority_wide_qrs and heart_rate_bpm>=150",
        "binary",
        ("qrs",),
        _narrow_complex_tachycardia_pattern_flag,
    ),
    _AdvancedSpec(
        "wide_complex_tachycardia_pattern_flag",
        "Post-development rate-aware rhythm pattern",
        "Majority-wide-QRS rhythm at >=120bpm -- a technical VT-compatible "
        "rate/QRS-width pattern, not a VT diagnosis.",
        "majority_wide_qrs and heart_rate_bpm>=120",
        "binary",
        ("qrs",),
        _wide_complex_tachycardia_pattern_flag,
    ),
    _AdvancedSpec(
        "accelerated_idioventricular_pattern_flag",
        "Post-development rate-aware rhythm pattern",
        "Majority-wide-QRS rhythm at 60-110bpm -- distinguishes the benign "
        "accelerated idioventricular rhythm from faster VT-compatible rates, "
        "both otherwise indistinguishable under RhythmClass.WIDE_QRS alone.",
        "majority_wide_qrs and 60<=heart_rate_bpm<110",
        "binary",
        ("qrs",),
        _accelerated_idioventricular_pattern_flag,
    ),
    _AdvancedSpec(
        "junctional_rhythm_pattern_flag",
        "Post-development rate-aware rhythm pattern",
        "Regular narrow-QRS rhythm with very low discrete-P coverage but no "
        "AF/flutter signature -- a technical pattern compatible with a "
        "junctional rhythm/tachycardia, not a diagnosis.",
        "rhythm_class==REGULAR and not majority_wide_qrs and p_coverage<0.15",
        "binary",
        ("qrs",),
        _junctional_rhythm_pattern_flag,
    ),
    _AdvancedSpec(
        "fragmented_qrs_present_flag",
        "Post-development scar/fragmentation",
        "Classic fragmented-QRS (fQRS) criterion: a notch or extra R' "
        "wave present in >=2 anatomically contiguous leads, excluding "
        "typical wide-QRS/bundle-branch-block morphology.",
        "any(all(lead fragmented for lead in contiguous_group) for "
        "contiguous_group in coronary-territory lead pairs)",
        "binary",
        ("qrs",),
        _fragmented_qrs_present_flag,
    ),
    _AdvancedSpec(
        "fragmented_qrs_lead_count",
        "Post-development scar/fragmentation",
        "Total number of the 12 standard leads showing a notch/extra R' "
        "wave, a continuous fQRS severity proxy alongside the "
        "contiguity-gated present/absent flag.",
        "count(lead fragmented) over all 12 standard leads",
        "count",
        ("qrs",),
        _fragmented_qrs_lead_count,
    ),
    _AdvancedSpec(
        "qrs_multiscale_entropy",
        "Post-development within-beat complexity (exploratory)",
        "Multiscale sample entropy of the QRS-window VCG magnitude "
        "trajectory, averaged over 1-3 coarse-graining scales.",
        "mean(sample_entropy(coarse_grain(||r(t)||, scale))) for scale in (1,2,3)",
        "dimensionless",
        ("qrs",),
        _qrs_multiscale_entropy,
    ),
    _AdvancedSpec(
        "t_multiscale_entropy",
        "Post-development within-beat complexity (exploratory)",
        "Multiscale sample entropy of the T-window VCG magnitude "
        "trajectory, averaged over 1-3 coarse-graining scales.",
        "mean(sample_entropy(coarse_grain(||r(t)||, scale))) for scale in (1,2,3)",
        "dimensionless",
        ("t",),
        _t_multiscale_entropy,
    ),
    _AdvancedSpec(
        "qrs_higuchi_fractal_dimension",
        "Post-development within-beat complexity (exploratory)",
        "Higuchi fractal dimension of the QRS-window VCG magnitude "
        "trajectory.",
        "slope of log(curve_length(k)) vs log(1/k), k=1..6",
        "dimensionless",
        ("qrs",),
        _qrs_higuchi_fractal_dimension,
    ),
    _AdvancedSpec(
        "t_higuchi_fractal_dimension",
        "Post-development within-beat complexity (exploratory)",
        "Higuchi fractal dimension of the T-window VCG magnitude "
        "trajectory.",
        "slope of log(curve_length(k)) vs log(1/k), k=1..6",
        "dimensionless",
        ("t",),
        _t_higuchi_fractal_dimension,
    ),
    _AdvancedSpec(
        "qrs_xy_cross_correlation_peak_lag_ms",
        "Post-development within-beat complexity (exploratory)",
        "Time lag of peak cross-correlation between the X and Y VCG "
        "channels within the QRS window.",
        "argmax_lag(cross_correlation(x(t), y(t))) / fs",
        "ms",
        ("qrs",),
        _qrs_xy_cross_correlation_peak_lag,
    ),
    _AdvancedSpec(
        "qrs_xy_cross_correlation_peak_value",
        "Post-development within-beat complexity (exploratory)",
        "Peak cross-correlation value between the X and Y VCG channels "
        "within the QRS window.",
        "max_lag(cross_correlation(x(t), y(t)))",
        "dimensionless",
        ("qrs",),
        _qrs_xy_cross_correlation_peak_value,
    ),
    _AdvancedSpec(
        "qrs_time_reversal_asymmetry",
        "Post-development within-beat complexity (exploratory)",
        "Scale-normalized RMS distance between the QRS loop trajectory "
        "and its own time-reversed version.",
        "RMS(curve(t) - curve(reversed t)) / max(||curve - centroid||)",
        "dimensionless",
        ("qrs",),
        _qrs_time_reversal_asymmetry,
    ),
    _AdvancedSpec(
        "t_time_reversal_asymmetry",
        "Post-development within-beat complexity (exploratory)",
        "Scale-normalized RMS distance between the T loop trajectory and "
        "its own time-reversed version.",
        "RMS(curve(t) - curve(reversed t)) / max(||curve - centroid||)",
        "dimensionless",
        ("t",),
        _t_time_reversal_asymmetry,
    ),
    _AdvancedSpec(
        "precordial_rs_absent_flag",
        "Post-development wide-complex tachycardia differentiation",
        "Brugada criterion step 1: absence of a genuine RS complex in all "
        "precordial leads favors ventricular origin. Computed only when "
        "wide_complex_tachycardia_pattern is set.",
        "no precordial lead (V1-V6) shows R-peak-then-S-trough morphology",
        "binary",
        ("qrs",),
        _precordial_rs_absent_flag,
    ),
    _AdvancedSpec(
        "precordial_rs_duration_max_ms",
        "Post-development wide-complex tachycardia differentiation",
        "Brugada criterion step 2: R-to-S-nadir duration >100ms in any "
        "precordial lead favors ventricular origin.",
        "max over V1-V6 of median(R-peak-to-S-trough duration)",
        "ms",
        ("qrs",),
        _precordial_rs_duration_max_ms,
    ),
    _AdvancedSpec(
        "precordial_concordance_flag",
        "Post-development wide-complex tachycardia differentiation",
        "All V1-V6 QRS complexes uniformly monophasic positive or "
        "negative -- high-specificity, low-sensitivity VT indicator.",
        "sign(median net QRS deflection) identical across all of V1-V6",
        "binary",
        ("qrs",),
        _precordial_concordance_flag,
    ),
    _AdvancedSpec(
        "avr_vi_vt_ratio",
        "Post-development wide-complex tachycardia differentiation",
        "Vereckei aVR algorithm: ratio of vertical excursion during the "
        "initial vs. terminal 40ms of the QRS in lead aVR. Ratio <=1 "
        "favors VT (Vereckei et al., Eur Heart J 2007).",
        "median(sum(|diff(aVR[first 40ms])|) / sum(|diff(aVR[last 40ms])|))",
        "ratio",
        ("qrs",),
        _avr_vi_vt_ratio,
    ),
    _AdvancedSpec(
        "av_dissociation_supports_vt_flag",
        "Post-development wide-complex tachycardia differentiation",
        "AV dissociation: an independently periodic atrial rate (spectral "
        "search on the QRS-masked residual) that neither tracks the "
        "ventricular rate nor sits in a simple N:1 conducted ratio to it "
        "-- one of the most specific classic VT-vs-SVT criteria. A rate "
        "comparison, not direct detection of a P wave buried inside the "
        "QRS, which is not reliably visible by amplitude.",
        "RhythmAssessment flag 'av_dissociation_compatible' during "
        "wide_complex_tachycardia_pattern",
        "binary",
        ("qrs",),
        _av_dissociation_supports_vt_flag,
    ),
    _AdvancedSpec(
        "independent_atrial_rate_bpm",
        "Post-development AV dissociation / rate-independence (exploratory)",
        "Continuous exposure of the QRS-masked spectral atrial-rate "
        "estimate feeding av_dissociation_compatible / "
        "third_degree_av_block_pattern, for data-driven cutoff discovery "
        "instead of the fixed rule thresholds baked into those flags.",
        "independent_atrial_rate_hz * 60",
        "bpm",
        ("qrs",),
        _independent_atrial_rate_bpm,
    ),
    _AdvancedSpec(
        "independent_atrial_ventricular_rate_ratio",
        "Post-development AV dissociation / rate-independence (exploratory)",
        "Continuous ratio of the independent atrial-rate estimate to the "
        "RR-derived ventricular rate -- the same signal underlying "
        "av_dissociation_compatible, exposed as a raw ratio.",
        "independent_atrial_rate_bpm / (60000 / rr_median_ms)",
        "ratio",
        ("qrs",),
        _independent_atrial_ventricular_rate_ratio,
    ),
    _AdvancedSpec(
        "rp_prime_interval_ms",
        "Post-development retrograde conduction signature (exploratory)",
        "R-to-retrograde-P' interval: time from the R peak to a "
        "retrograde P candidate found after this beat's own QRS "
        "(MultileadDelineator._search_retrograde_p_candidates). Short "
        "RP' is the classic typical-AVNRT signature (Katritsis & "
        "Josephson, Europace 2016); a candidate feature, not a "
        "diagnosis.",
        "median(retrograde_p_peak.sample - r_peak.sample) / fs * 1000",
        "ms",
        ("qrs",),
        _rp_prime_interval_ms,
    ),
    _AdvancedSpec(
        "retrograde_p_qrs_axis_angle_deg",
        "Post-development retrograde conduction signature (exploratory)",
        "Spatial angle between a retrograde-P candidate's dominant VCG "
        "loop axis and this beat's QRS-loop axis -- supporting evidence "
        "that a retrograde-P candidate is real atrial activity (a "
        "different activation direction than the QRS) rather than a "
        "misclassified QRS/T fragment.",
        "median(angle(loop_axis(retrograde_p), loop_axis(qrs)))",
        "degrees",
        ("qrs",),
        _retrograde_p_qrs_axis_angle_deg,
    ),
    _AdvancedSpec(
        "p_loop_qrs_loop_axis_angle_deg",
        "Post-development atrial-loop spatial geometry (exploratory)",
        "Spatial angle between the forward P-loop's dominant VCG axis "
        "and the QRS-loop axis -- a direction-based spatial companion "
        "to the amplitude/duration-based p_terminal_force_v1 and "
        "p_duration_ge_120 classical criteria for atrial abnormality.",
        "median(angle(loop_axis(p), loop_axis(qrs)))",
        "degrees",
        ("p", "qrs"),
        _p_loop_qrs_loop_axis_angle_deg,
    ),
    _AdvancedSpec(
        "p_loop_t_loop_axis_angle_deg",
        "Post-development atrial-loop spatial geometry (exploratory)",
        "Spatial angle between the P-loop's dominant VCG axis and the "
        "T-loop's dominant VCG axis for the same beat -- an atrial-to-"
        "ventricular-repolarization directional counterpart to the "
        "established spatial QRS-T angle, which has no atrial analogue "
        "in the existing catalog.",
        "median(angle(loop_axis(p), loop_axis(t)))",
        "degrees",
        ("p", "t"),
        _p_loop_t_loop_axis_angle_deg,
    ),
    _AdvancedSpec(
        "beatwise_p_loop_axis_stability_deg",
        "Post-development atrial-loop spatial geometry (exploratory)",
        "Beat-to-beat angular dispersion of the P-loop's dominant "
        "spatial axis across the record -- a candidate spatial "
        "signature of multifocal atrial activity / wandering atrial "
        "pacemaker, complementing the timing/morphology-count-based "
        "multifocal_atrial_tachycardia_pattern_flag (G11).",
        "std(angle(loop_axis(p, beat_n), loop_axis(p, beat_0)))",
        "degrees",
        ("p",),
        _beatwise_p_loop_axis_stability_deg,
    ),
    _AdvancedSpec(
        "terminal_qrs_pseudo_r_prime_v1_flag",
        "Post-development reentry signature (exploratory)",
        "Extra positive deflection in the terminal ~50ms of the QRS in "
        "V1 during a narrow-complex tachycardia -- compatible with a "
        "retrograde P wave fused into the terminal QRS (typical-AVNRT "
        "pseudo-r', Kalbfleisch et al., JACC 1993). No sinus-baseline "
        "comparison available; a hypothesis flag, not an AVNRT diagnosis.",
        "interior local maximum in the terminal 50ms QRS window in V1, "
        "majority vote over primary-cluster beats",
        "binary",
        ("qrs",),
        _terminal_qrs_pseudo_r_prime_v1_flag,
    ),
    _AdvancedSpec(
        "terminal_qrs_pseudo_s_inferior_flag",
        "Post-development reentry signature (exploratory)",
        "Extra negative deflection in the terminal ~50ms of the QRS in "
        "II/III/aVF during a narrow-complex tachycardia -- the pseudo-s "
        "counterpart of terminal_qrs_pseudo_r_prime_v1_flag.",
        "interior local minimum in the terminal 50ms QRS window in "
        "II/III/aVF, majority vote over primary-cluster beats",
        "binary",
        ("qrs",),
        _terminal_qrs_pseudo_s_inferior_flag,
    ),
    _AdvancedSpec(
        "beatwise_qrs_axis_alternation_score",
        "Post-development beat-sequence rhythm signature (exploratory)",
        "Strict period-2 QRS-axis alternation across the beat sequence "
        "(1.0 = perfect bidirectional alternation) -- CPVT candidate "
        "(Leenhardt et al., Circulation 1995).",
        "(1 - mean(angle(axis[N], axis[N+2]))/180 "
        "+ mean(angle(axis[N], axis[N+1]))/180) / 2",
        "fraction",
        ("qrs",),
        _beatwise_qrs_axis_alternation_score,
    ),
    _AdvancedSpec(
        "qrs_axis_progressive_rotation_deg_per_beat",
        "Post-development beat-sequence rhythm signature (exploratory)",
        "Mean sign-consistent angular step in QRS axis between "
        "successive beats -- Torsades-de-pointes-pattern candidate "
        "(monotonic rotation, distinct from the strict alternation of "
        "beatwise_qrs_axis_alternation_score).",
        "mean(signed angular step between successive beats' QRS axis)",
        "degrees",
        ("qrs",),
        _qrs_axis_progressive_rotation,
    ),
    _AdvancedSpec(
        "distinct_p_morphology_cluster_count",
        "Post-development atrial focus multiplicity",
        "Number of distinct P-wave shape populations across the beat "
        "sequence (medoid + cosine-similarity clustering, E3's method "
        "applied to P instead of QRS).",
        "count of coherent P-loop shape clusters (similarity>=0.70, "
        "min 2 beats each) over all beats",
        "count",
        ("qrs",),
        _distinct_p_morphology_cluster_count,
    ),
    _AdvancedSpec(
        "multifocal_atrial_tachycardia_pattern_flag",
        "Post-development atrial focus multiplicity",
        "Kastor's MAT criteria (NEJM 1990): >=3 distinct P-wave "
        "morphologies, irregular rate >100bpm, no flutter pattern.",
        "distinct_p_morphology_cluster_count>=3 and heart_rate_bpm>100 "
        "and irregular and not flutter_compatible",
        "binary",
        ("qrs",),
        _multifocal_atrial_tachycardia_pattern_flag,
    ),
    _AdvancedSpec(
        "delta_wave_present_flag",
        "Post-development pre-excitation localization (exploratory)",
        "Unusually slow initial QRS upstroke (slurred onset) in the "
        "beat's largest-amplitude lead, corroborated by a short PR "
        "interval (<120ms) when P waves are reliably present -- the two "
        "classic Wolff-Parkinson-White signs together. A real LUDB "
        "spot-check found the slope signal alone also fires on LVH/scar "
        "conduction delay and post-MI pathological Q waves; the PR "
        "corroboration reduces but does not eliminate that risk. "
        "Surface morphology cannot distinguish atrioventricular from "
        "fasciculoventricular pathways, including in Danon/PRKAG2 disease; "
        "this is not a genotype or pathway-risk diagnosis. "
        "The fixed PR threshold is not age-normalized and is not validated "
        "for infants or children; do not interpret it as pediatric short PR.",
        "median over primary-cluster beats of "
        "(mean|diff(QRS[first 40ms])| / mean|diff(QRS)|) < 0.5, AND "
        "median(PR interval) < 120ms when >=3 beats have a measurable PR",
        "binary",
        ("qrs",),
        _delta_wave_present_flag,
    ),
    _AdvancedSpec(
        "delta_wave_frontal_axis_deg",
        "Post-development pre-excitation localization (exploratory)",
        "Frontal-plane axis of the initial (delta-wave) QRS forces from "
        "lead I / aVF sign, the first step of Arruda-style accessory-"
        "pathway localization (Arruda et al., PACE 1998). Only computed "
        "when delta_wave_present_flag is set.",
        "atan2(median(aVF[first 40ms]), median(I[first 40ms])) in degrees",
        "degrees",
        ("qrs",),
        _delta_wave_frontal_axis,
    ),
    _AdvancedSpec(
        "accessory_pathway_quadrant_estimate",
        "Post-development pre-excitation localization (exploratory)",
        "Coarse, unvalidated accessory-pathway quadrant hypothesis "
        "combining delta_wave_frontal_axis_deg with V1 initial polarity "
        "-- a deliberately simplified stand-in for the more elaborate "
        "validated Arruda/EASY-WPW/SMART-WPW algorithms, not a "
        "diagnostic localization. Does not establish an atrioventricular "
        "accessory pathway: fasciculoventricular pathways can coexist or "
        "produce preexcitation. Not an ablation target or treatment indication.",
        "2*(V1[first 40ms] median>0) + (delta_wave_frontal_axis_deg<0): "
        "0=right/anterolateral, 1=right/posterolateral, "
        "2=left/anterolateral, 3=left/posterolateral",
        "quadrant_code",
        ("qrs",),
        _accessory_pathway_quadrant_estimate,
    ),
    _AdvancedSpec(
        "flutter_av_block_ratio_estimate",
        "Post-development flutter conduction ratio",
        "Atrial:ventricular rate ratio for flutter-compatible rhythms "
        "(e.g. ~2.0 for classic 2:1 block, ~4.0 for 4:1) -- purely a "
        "derived ratio of already-available rhythm-assessment fields.",
        "atrial_frequency_hz / (1000/rr_median_ms), "
        "only when rhythm_class==FLUTTER_COMPATIBLE",
        "ratio",
        ("qrs",),
        _flutter_av_block_ratio_estimate,
    ),
)


from .a108_robustness import specifications as _a108_specifications

ADVANCED_SPECS += _a108_specifications(_AdvancedSpec)


class AdvancedVCGBiomarker(BiomarkerBase):
    def __init__(self, spec: _AdvancedSpec):
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
        return list(self.spec.references or ADVANCED_VCG_REFERENCES)

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
                "not_evaluable: reason_code=segment_quality; segment confidence below the wave floor for "
                + ", ".join(sorted(insufficient)),
            )
        feature = self.spec.calculator(context)
        value = feature.value
        if value is not None and not np.isfinite(value):
            value = None
        confidence = (
            min(feature.confidence, dependency_confidence)
            if value is not None
            else 0.0
        )
        return self._result(value, confidence, feature.notes, feature.per_lead, feature.measurement_windows)

    def _result(
        self,
        value: Optional[float],
        confidence: float,
        notes: str,
        per_lead: Optional[dict[str, float]] = None,
        measurement_windows: Optional[dict] = None,
    ) -> BiomarkerResult:
        return BiomarkerResult(
            name=self.name,
            value=value,
            unit=self.unit,
            category=self.category,
            confidence=float(confidence),
            per_lead=per_lead,
            description=self.description,
            references=self.references,
            computation_notes=notes,
            measurement_windows=measurement_windows,
            family=self.spec.family,
            formula=self.spec.formula,
            implementation_status="implemented",
            pathological_direction="data_driven",
            pathological_note=(
                "Post-development MIMIC hypothesis. Direction and threshold "
                "must be learned in grouped Train data and independently replicated."
            ),
        )


def register_advanced_vcg_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in ADVANCED_SPECS:
        if spec.name in existing:
            continue
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=spec.description,
                formula=spec.formula,
                unit=spec.unit,
                category="vcg",
                source=ADVANCED_VCG_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "Post-development MIMIC hypothesis; Discovery Reserve only "
                    "until independent replication."
                ),
                implementation_status="implemented",
                references=list(spec.references or ADVANCED_VCG_REFERENCES),
                notes=(
                    f"{ADVANCED_VCG_VERSION}; conceived after inspection of "
                    "development results and therefore excluded from frozen Core/Sentinel."
                ),
            )
        )
        BiomarkerRegistry.register(AdvancedVCGBiomarker(spec))
        existing.add(spec.name)


__all__ = [
    "ADVANCED_SPECS",
    "ADVANCED_VCG_VERSION",
    "DIRECT_QRS_T_LEADS",
    "DIRECT_QRS_T_REFERENCE",
    "AdvancedVCGBiomarker",
    "direct_qrs_t_angles_8lead",
    "register_advanced_vcg_biomarkers",
]
