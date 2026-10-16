from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


VCG_AXIS_CONVENTION = (
    "Frank-compatible XYZ: +X left, +Y inferior, +Z posterior; "
    "azimuth=atan2(Z,X) in [-180,180] degrees; elevation is the angle "
    "from +Y in [0,180] degrees (0=inferior, 90=horizontal, 180=superior)."
)
VCG_AXIS_CONVENTION_VERSION = "frank-xyz-geh-v1.0.0"
PHYSIONET_GEH_REFERENCE = (
    "https://physionet.org/content/geh/1.0.0/GEH%20calculation.pdf"
)
ORIGIN_METHOD_VERSION = "physiological-vcg-origin-v1.1.0"
GEH_FORMULA_VERSION = "physionet-geh-v1.0.0"
_EPS = 1e-12


@dataclass(frozen=True)
class VCGOriginCandidate:
    vector_millivolts: np.ndarray
    method: str
    start_sample: int
    end_sample: int
    confidence: float
    robust_mad_millivolts: float

    @property
    def duration_samples(self) -> int:
        return int(self.end_sample - self.start_sample + 1)


@dataclass(frozen=True)
class VCGOriginAssessment:
    primary: Optional[VCGOriginCandidate]
    tp: Optional[VCGOriginCandidate]
    pr: Optional[VCGOriginCandidate]
    status: str
    reason: str
    tq: Optional[VCGOriginCandidate] = None

    @property
    def evaluable(self) -> bool:
        return self.primary is not None and self.status == "ok"

    def corrected(self, vcg_xyz: np.ndarray) -> Optional[np.ndarray]:
        if self.primary is None:
            return None
        signal = np.asarray(vcg_xyz, dtype=float)
        return signal - self.primary.vector_millivolts[np.newaxis, :]

    def provenance_note(self) -> str:
        if self.primary is None:
            return (
                "not_evaluable: physiological VCG origin unavailable; "
                f"reason={self.reason}; origin_version={ORIGIN_METHOD_VERSION}"
            )
        candidate = self.primary
        alternative = (
            "available"
            if self.tp is not None and self.pr is not None
            else "unavailable"
        )
        tier = (
            "primary_tp"
            if candidate.method.startswith("tp")
            else "secondary_pr"
            if candidate.method == "pr_fallback"
            else "tertiary_tq_no_p_wave_required"
        )
        return (
            f"origin={candidate.method}; interval_samples="
            f"{candidate.start_sample}:{candidate.end_sample}; "
            f"origin_confidence={candidate.confidence:.3f}; "
            f"tp_pr_sensitivity={alternative}; "
            f"origin_tier={tier}; "
            f"origin_version={ORIGIN_METHOD_VERSION}"
        )

    def numeric_metadata(self) -> dict[str, float]:
        if self.primary is None:
            return {"origin_evaluable": 0.0}
        candidate = self.primary
        metadata = {
            "origin_evaluable": 1.0,
            "origin_x_mv": float(candidate.vector_millivolts[0]),
            "origin_y_mv": float(candidate.vector_millivolts[1]),
            "origin_z_mv": float(candidate.vector_millivolts[2]),
            "origin_confidence": float(candidate.confidence),
            "origin_interval_start_sample": float(candidate.start_sample),
            "origin_interval_end_sample": float(candidate.end_sample),
            "origin_interval_samples": float(candidate.duration_samples),
            "origin_robust_mad_mv": float(candidate.robust_mad_millivolts),
        }
        if self.tp is not None and self.pr is not None:
            metadata["origin_tp_pr_distance_uv"] = float(
                np.linalg.norm(
                    self.tp.vector_millivolts - self.pr.vector_millivolts
                )
                * 1000.0
            )
        return metadata


@dataclass(frozen=True)
class GEHAreaMetrics:
    qrs_area_vector_uv_ms: np.ndarray
    jt_area_vector_uv_ms: np.ndarray
    qrst_area_vector_uv_ms: np.ndarray
    svg_magnitude_uv_ms: float
    qrs_jt_angle_deg: Optional[float]
    sai_qrs_mv_ms: float
    sai_jt_mv_ms: float
    sai_qrst_mv_ms: float
    qrst_duration_ms: float


def vcg_azimuth_deg(vector: np.ndarray) -> Optional[float]:
    """GEH azimuth in the transverse X-Z plane."""

    values = _validated_vector(vector)
    if values is None or np.hypot(values[0], values[2]) <= _EPS:
        return None
    return float(np.degrees(np.arctan2(values[2], values[0])))


def vcg_elevation_deg(vector: np.ndarray) -> Optional[float]:
    """Published GEH elevation: angle from the positive (inferior) Y axis."""

    values = _validated_vector(vector)
    if values is None:
        return None
    magnitude = float(np.linalg.norm(values))
    if magnitude <= _EPS:
        return None
    cosine = float(np.clip(values[1] / magnitude, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def vector_angle_deg(
    first: np.ndarray,
    second: np.ndarray,
) -> Optional[float]:
    left = _validated_vector(first)
    right = _validated_vector(second)
    if left is None or right is None:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPS:
        return None
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def area_vector_uv_ms(segment_millivolts: np.ndarray, fs: int) -> np.ndarray:
    values = _validated_xyz(segment_millivolts)
    if fs <= 0:
        raise ValueError("sampling rate must be positive")
    if len(values) < 2:
        return np.zeros(3, dtype=float)
    # mV * ms -> uV * ms
    return (
        np.trapezoid(values, dx=1000.0 / float(fs), axis=0)
        * 1000.0
    )


def sum_absolute_integral_mv_ms(
    segment_millivolts: np.ndarray,
    fs: int,
) -> float:
    """SAI = integral(|X|)+integral(|Y|)+integral(|Z|), in mV*ms."""

    values = _validated_xyz(segment_millivolts)
    if fs <= 0:
        raise ValueError("sampling rate must be positive")
    if len(values) < 2:
        return 0.0
    per_axis = np.trapezoid(
        np.abs(values),
        dx=1000.0 / float(fs),
        axis=0,
    )
    return float(np.sum(per_axis))


def compute_geh_area_metrics(
    origin_corrected_vcg_xyz: np.ndarray,
    *,
    qrs_onset: int,
    qrs_offset: int,
    t_offset: int,
    fs: int,
) -> GEHAreaMetrics:
    """Compute the area GEH quantities using the official PhysioNet bounds.

    QRS area is integrated from QRS onset through QRS offset.  The published
    T-area vector is the JT integral from QRS offset through T offset, not the
    visually delineated T-wave alone.  Wilson's SVG is independently
    integrated over the entire QRST interval as an audit against QRS + JT.
    """

    values = _validated_xyz(origin_corrected_vcg_xyz)
    if fs <= 0:
        raise ValueError("sampling rate must be positive")
    start = int(qrs_onset)
    junction = int(qrs_offset)
    end = int(t_offset)
    if not 0 <= start < junction < end < len(values):
        raise ValueError(
            "GEH bounds must satisfy 0 <= qrs_onset < qrs_offset "
            "< t_offset < n_samples"
        )
    qrs = values[start : junction + 1]
    jt = values[junction : end + 1]
    qrst = values[start : end + 1]
    qrs_area = area_vector_uv_ms(qrs, fs)
    jt_area = area_vector_uv_ms(jt, fs)
    qrst_area = area_vector_uv_ms(qrst, fs)
    return GEHAreaMetrics(
        qrs_area_vector_uv_ms=qrs_area,
        jt_area_vector_uv_ms=jt_area,
        qrst_area_vector_uv_ms=qrst_area,
        svg_magnitude_uv_ms=float(np.linalg.norm(qrst_area)),
        qrs_jt_angle_deg=vector_angle_deg(qrs_area, jt_area),
        sai_qrs_mv_ms=sum_absolute_integral_mv_ms(qrs, fs),
        sai_jt_mv_ms=sum_absolute_integral_mv_ms(jt, fs),
        sai_qrst_mv_ms=sum_absolute_integral_mv_ms(qrst, fs),
        qrst_duration_ms=float((end - start) * 1000.0 / fs),
    )


def estimate_physiological_vcg_origin(
    vcg_xyz: np.ndarray,
    delineation,
    fs: int,
    *,
    minimum_confidence: float = 0.35,
) -> VCGOriginAssessment:
    """Estimate a physiological isoelectric VCG origin.

    A quiet TP/UP interval adjacent to the representative beat is preferred.
    A PR interval is accepted only when discrete P and QRS boundaries are
    established, the rhythm permits a discrete P wave, and safety margins
    leave a sufficiently long, stable segment.

    Both of those require a detected P wave, and P detection is the weakest
    part of the delineator.  Measured on 38 consecutive MIMIC records, TP or
    PR resolved on 39 % -- so every origin-corrected marker, which is the
    whole corrected GEH/SVG stack, silently abstained on the other 61 %.

    The third tier removes that dependency.  The early TQ interval -- from T
    offset to the next QRS onset, restricted to its first 60 % -- is
    electrical diastole and needs no P detection at all.  It is deliberately
    the *last* choice, because the full TQ interval contains the P wave and
    only the early part is safely atrial-quiet; the restriction is what makes
    it usable, and the tier is recorded so a sensitivity analysis can compare
    the three.  Measured spatial scatter inside that window was 18 uV median,
    against a 150 uV trust threshold, and it resolved on 100 % of the same
    records.

    There is still deliberately no zero-vector fallback: when even the TQ
    interval is unusable, origin-dependent biomarkers abstain.
    """

    try:
        signal = _validated_xyz(vcg_xyz)
    except ValueError as exc:
        return VCGOriginAssessment(None, None, None, "not_evaluable", str(exc))
    if fs <= 0:
        return VCGOriginAssessment(
            None, None, None, "not_evaluable", "invalid_sampling_rate"
        )
    representative = getattr(delineation, "representative_beat", None)
    if representative is None:
        return VCGOriginAssessment(
            None, None, None, "not_evaluable", "representative_beat_unavailable"
        )
    beats = list(getattr(delineation, "accepted_beats", ()) or ())
    if not beats:
        beats = list(getattr(delineation, "beats", ()) or ())
    beats = sorted(beats, key=lambda item: int(item.r_peak.sample))
    if not beats:
        return VCGOriginAssessment(
            None, None, None, "not_evaluable", "accepted_beats_unavailable"
        )
    target_index = int(
        np.argmin(
            [
                abs(int(beat.r_peak.sample) - int(representative.r_peak.sample))
                for beat in beats
            ]
        )
    )
    rhythm = getattr(delineation, "rhythm", None)
    discrete_p_permitted = not bool(
        getattr(rhythm, "discrete_p_not_evaluable", False)
    )

    tp_candidates: list[VCGOriginCandidate] = []
    if target_index + 1 < len(beats):
        post = _tp_candidate(
            signal,
            left_beat=representative,
            right_beat=beats[target_index + 1],
            fs=fs,
            method="tp_post",
            minimum_confidence=minimum_confidence,
            discrete_p_permitted=discrete_p_permitted,
        )
        if post is not None:
            tp_candidates.append(post)
    if target_index > 0:
        previous = _tp_candidate(
            signal,
            left_beat=beats[target_index - 1],
            right_beat=representative,
            fs=fs,
            method="tp_pre",
            minimum_confidence=minimum_confidence,
            discrete_p_permitted=discrete_p_permitted,
        )
        if previous is not None:
            tp_candidates.append(previous)
    tp = (
        max(
            tp_candidates,
            key=lambda item: (item.confidence, item.duration_samples),
        )
        if tp_candidates
        else None
    )
    pr = _pr_candidate(
        signal,
        representative,
        fs=fs,
        minimum_confidence=max(0.50, float(minimum_confidence)),
        discrete_p_permitted=discrete_p_permitted,
    )
    tq = None
    if tp is None and pr is None:
        tq = _tq_candidate(
            signal,
            beats=beats,
            target_index=target_index,
            fs=fs,
            minimum_confidence=minimum_confidence,
        )
    primary = tp if tp is not None else (pr if pr is not None else tq)
    if primary is None:
        return VCGOriginAssessment(
            None,
            tp,
            pr,
            "not_evaluable",
            "no_qualified_tp_pr_or_tq_interval",
            tq,
        )
    return VCGOriginAssessment(primary, tp, pr, "ok", "", tq)


def origin_sensitivity_metadata(
    assessment: VCGOriginAssessment,
    *,
    qrst_duration_ms: Optional[float] = None,
) -> dict[str, float]:
    metadata = assessment.numeric_metadata()
    if assessment.tp is None or assessment.pr is None:
        return metadata
    delta_mv = assessment.tp.vector_millivolts - assessment.pr.vector_millivolts
    distance_uv = float(np.linalg.norm(delta_mv) * 1000.0)
    metadata["origin_tp_pr_distance_uv"] = distance_uv
    if qrst_duration_ms is not None and np.isfinite(qrst_duration_ms):
        # (mV origin difference) * ms * 1000 = uV*ms
        metadata["svg_origin_sensitivity_uv_ms"] = float(
            np.linalg.norm(delta_mv) * float(qrst_duration_ms) * 1000.0
        )
    return metadata


def _tp_candidate(
    signal: np.ndarray,
    *,
    left_beat,
    right_beat,
    fs: int,
    method: str,
    minimum_confidence: float,
    discrete_p_permitted: bool,
) -> Optional[VCGOriginCandidate]:
    if not discrete_p_permitted:
        return None
    if not _wave_present(left_beat, "t") or not _wave_present(right_beat, "p"):
        return None
    if bool(getattr(right_beat, "p_t_fusion_risk", False)):
        return None
    left_marker = getattr(left_beat, "t_offset", None)
    u_offset = getattr(left_beat, "u_offset", None)
    if _wave_present(left_beat, "u") and _qualified_marker(
        u_offset, minimum_confidence
    ):
        left_marker = u_offset
    right_marker = getattr(right_beat, "p_onset", None)
    if not _qualified_marker(left_marker, minimum_confidence):
        return None
    if not _qualified_marker(right_marker, minimum_confidence):
        return None
    left = int(left_marker.sample)
    right = int(right_marker.sample)
    if right <= left:
        return None
    gap = right - left
    margin = max(int(round(0.008 * fs)), int(round(0.10 * gap)))
    start = left + margin
    end = right - margin
    confidence = min(float(left_marker.confidence), float(right_marker.confidence))
    return _origin_candidate(
        signal,
        start=start,
        end=end,
        fs=fs,
        method=method,
        endpoint_confidence=confidence,
        minimum_duration_ms=20.0,
    )


def _pr_candidate(
    signal: np.ndarray,
    beat,
    *,
    fs: int,
    minimum_confidence: float,
    discrete_p_permitted: bool,
) -> Optional[VCGOriginCandidate]:
    if not discrete_p_permitted or bool(getattr(beat, "p_t_fusion_risk", False)):
        return None
    if not _wave_present(beat, "p") or not _wave_present(beat, "qrs"):
        return None
    p_offset = getattr(beat, "p_offset", None)
    qrs_onset = getattr(beat, "r_onset", None)
    if not _qualified_marker(p_offset, minimum_confidence):
        return None
    if not _qualified_marker(qrs_onset, minimum_confidence):
        return None
    start = int(p_offset.sample) + int(round(0.004 * fs))
    end = int(qrs_onset.sample) - int(round(0.008 * fs))
    return _origin_candidate(
        signal,
        start=start,
        end=end,
        fs=fs,
        method="pr_fallback",
        endpoint_confidence=min(
            float(p_offset.confidence), float(qrs_onset.confidence)
        ),
        minimum_duration_ms=12.0,
    )


def _tq_candidate(
    signal: np.ndarray,
    *,
    beats: list,
    target_index: int,
    fs: int,
    minimum_confidence: float,
) -> Optional[VCGOriginCandidate]:
    """Early electrical diastole, without needing a P wave.

    Restricted to the first 60 % of the T-offset-to-next-QRS-onset interval:
    at ordinary rates atrial activation falls in the later part, so the early
    portion is atrial-quiet even when the P wave itself was never detected.
    The 150 uV scatter gate still applies, so a noisy diastole is rejected
    rather than trusted.
    """

    # Every consecutive pair is a candidate, ordered by distance from the
    # representative beat. Restricting to the two neighbours -- as the TP tier
    # does -- left a quarter of records without any origin because the nearest
    # T offset happened to be unqualified, and a diastolic interval three
    # beats away is still this record's diastole.
    pairs = sorted(
        (
            (min(abs(index - target_index), abs(index + 1 - target_index)),
             beats[index],
             beats[index + 1])
            for index in range(len(beats) - 1)
        ),
        key=lambda item: item[0],
    )
    best: Optional[VCGOriginCandidate] = None
    for distance_from_target, left_beat, right_beat in pairs:
        left_marker = getattr(left_beat, "t_offset", None)
        right_marker = getattr(right_beat, "r_onset", None)
        if not _qualified_marker(left_marker, minimum_confidence):
            continue
        if not _qualified_marker(right_marker, minimum_confidence):
            continue
        left = int(left_marker.sample)
        right = int(right_marker.sample)
        gap = right - left
        if gap < int(round(0.060 * fs)):
            continue
        start = left + int(round(0.020 * fs))
        end = left + int(round(0.60 * gap))
        candidate = _origin_candidate(
            signal,
            start=start,
            end=end,
            fs=fs,
            method="tq_early_fallback",
            # A third-tier origin must not present itself as a first-tier one,
            # and one taken further from the representative beat must not
            # present itself as an adjacent one.
            endpoint_confidence=0.60
            * (0.90 ** int(distance_from_target))
            * min(float(left_marker.confidence), float(right_marker.confidence)),
            minimum_duration_ms=20.0,
        )
        if candidate is None:
            continue
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return best


def _origin_candidate(
    signal: np.ndarray,
    *,
    start: int,
    end: int,
    fs: int,
    method: str,
    endpoint_confidence: float,
    minimum_duration_ms: float,
) -> Optional[VCGOriginCandidate]:
    start = max(0, int(start))
    end = min(len(signal) - 1, int(end))
    minimum_samples = max(3, int(round(minimum_duration_ms / 1000.0 * fs)))
    if end - start + 1 < minimum_samples:
        return None
    segment = signal[start : end + 1]
    if not np.all(np.isfinite(segment)):
        return None
    vector = np.median(segment, axis=0)
    residual_norm = np.linalg.norm(segment - vector[np.newaxis, :], axis=1)
    robust_mad = float(np.median(residual_norm))
    # More than 150 uV median spatial scatter is not a trustworthy
    # isoelectric interval.  Lower scatter smoothly increases confidence.
    if robust_mad > 0.150:
        return None
    stability = float(np.clip(1.0 - robust_mad / 0.150, 0.25, 1.0))
    return VCGOriginCandidate(
        vector_millivolts=np.asarray(vector, dtype=float),
        method=method,
        start_sample=start,
        end_sample=end,
        confidence=float(np.clip(endpoint_confidence * stability, 0.0, 1.0)),
        robust_mad_millivolts=robust_mad,
    )


def _qualified_marker(marker, minimum_confidence: float) -> bool:
    return (
        marker is not None
        and float(getattr(marker, "confidence", 0.0)) >= minimum_confidence
        and float(getattr(marker, "uncertainty_ms", 0.0)) <= 60.0
    )


def _wave_present(beat, wave: str) -> bool:
    try:
        return beat.resolved_wave_status(wave).value == "present"
    except (AttributeError, ValueError):
        return False


def _validated_xyz(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"VCG must be samples-by-3, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("VCG contains non-finite samples")
    return array


def _validated_vector(values: np.ndarray) -> Optional[np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return array


__all__ = [
    "GEHAreaMetrics",
    "GEH_FORMULA_VERSION",
    "ORIGIN_METHOD_VERSION",
    "PHYSIONET_GEH_REFERENCE",
    "VCG_AXIS_CONVENTION",
    "VCG_AXIS_CONVENTION_VERSION",
    "VCGOriginAssessment",
    "VCGOriginCandidate",
    "area_vector_uv_ms",
    "compute_geh_area_metrics",
    "estimate_physiological_vcg_origin",
    "origin_sensitivity_metadata",
    "sum_absolute_integral_mv_ms",
    "vcg_azimuth_deg",
    "vcg_elevation_deg",
    "vector_angle_deg",
]
