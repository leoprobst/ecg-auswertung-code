from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from threading import local
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import signal as sp_signal
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import ConvexHull, distance, procrustes
from scipy.stats import kurtosis, skew

from ._numeric_kernels import dtw_distance, self_intersection_count
from .base import BiomarkerBase, BiomarkerDefinition, BiomarkerRegistry, BiomarkerResult
from .catalog import candidate_definitions
from ..data_loader import LEAD_MAP
from ..delineation.base import (
    RecordDelineation,
    fiducial_confidence_threshold,
    measurement_confidence_threshold,
    ventricular_rr_seconds,
)
from ..vcg_geometry import (
    GEHAreaMetrics,
    GEH_FORMULA_VERSION,
    VCGOriginAssessment,
    area_vector_uv_ms,
    compute_geh_area_metrics,
    estimate_physiological_vcg_origin,
    origin_sensitivity_metadata,
    vcg_azimuth_deg,
    vcg_elevation_deg,
)


EPS = 1e-10
# Signals reach the feature layer in millivolts (see external_loader's
# signal_millivolts). Any marker the catalog declares in uV must apply this
# explicitly -- the two that did not were off by 1e3 and 1e6 respectively.
MILLIVOLT_TO_MICROVOLT = 1000.0
SEGMENTS = (
    "global_vcg", "qrs_early", "qrs_mid", "qrs_late",
    "t_early", "t_mid", "t_late", "t_upstroke", "t_downstroke", "beat", "qrst", "st_t",
    "qrs", "pr", "st", "p", "t",
)
PLANES = {
    "frontal_xy": (0, 1),
    "horizontal_xz": (0, 2),
    "sagittal_yz": (1, 2),
}


@dataclass
class FeatureValue:
    value: Optional[float]
    confidence: float = 0.65
    per_lead: Optional[dict[str, float]] = None
    notes: str = ""
    measurement_windows: Optional[dict] = None


@dataclass
class FeatureContext:
    delineation: RecordDelineation
    ecg_12lead: Optional[np.ndarray]
    vcg_xyz: Optional[np.ndarray]
    fs: int
    _segments: dict[str, Optional[tuple[int, int]]] = field(default_factory=dict)
    _segment_confidences: dict[str, float] = field(default_factory=dict)
    _segment_evaluability: dict[tuple[str, Optional[float]], bool] = field(
        default_factory=dict
    )
    _kinematics: dict[str, dict[str, float]] = field(default_factory=dict)
    _loop3d: dict[str, dict[str, float]] = field(default_factory=dict)
    _loop2d: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    _vectors: dict[tuple[str, str], tuple[np.ndarray, dict[str, float]]] = field(default_factory=dict)
    # Record-scoped primitive DAG. Scalar catalog markers are intentionally
    # numerous, but smoothing/PCA/FFT/distance matrices must be computed once
    # per segment and parameter set rather than once per scalar leaf.
    _primitive_cache: dict[tuple[object, ...], object] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _representative_beat: object = field(default=None, init=False, repr=False)
    _record_ecg_12lead: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )
    _record_raw_ecg_12lead: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _record_acquisition: dict = field(default_factory=dict, init=False, repr=False)
    _record_vcg_xyz: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )
    _record_vcg_wideband_xyz: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )
    _vcg_origin_assessment: Optional[VCGOriginAssessment] = field(
        default=None,
        init=False,
        repr=False,
    )
    _vcg_origin_assessment_attempted: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _origin_corrected_vcg_xyz: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )
    _origin_correction_attempted: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _geh_area_metrics: Optional[GEHAreaMetrics] = field(
        default=None,
        init=False,
        repr=False,
    )
    _geh_area_attempted: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        # No fallback: morphology/delineation inputs are not raw recordings.
        self._record_raw_ecg_12lead = getattr(_CONTEXT_STATE, "record_raw_ecg_12lead", None)
        self._record_acquisition = dict(getattr(_CONTEXT_STATE, "record_acquisition", None) or {})
        self._representative_beat = self.delineation.representative_beat
        self._record_ecg_12lead = getattr(
            _CONTEXT_STATE,
            "record_ecg_12lead",
            None,
        )
        self._record_vcg_xyz = getattr(
            _CONTEXT_STATE,
            "record_vcg_xyz",
            None,
        )
        self._record_vcg_wideband_xyz = getattr(
            _CONTEXT_STATE,
            "record_vcg_wideband_xyz",
            None,
        )
        if self._record_ecg_12lead is None:
            self._record_ecg_12lead = self.ecg_12lead
        if self._record_vcg_xyz is None:
            self._record_vcg_xyz = self.vcg_xyz

    @property
    def beat(self):
        return self._representative_beat

    @property
    def n_samples(self) -> int:
        if self.vcg_xyz is not None:
            return int(self.vcg_xyz.shape[0])
        if self.ecg_12lead is not None:
            return int(self.ecg_12lead.shape[0])
        return int(self.delineation.n_samples)

    def _sample(
        self,
        attr: str,
        minimum_confidence: Optional[float] = None,
    ) -> Optional[int]:
        """Resolve one fiducial point, at that wave's calibrated strictness.

        ``None`` means "use the threshold calibrated for this landmark's
        wave".  It used to mean 0.35 for every landmark, which quietly
        overrode the per-wave calibration applied to the segment gates: a
        beat admitted at the P segment's 0.15 lost its P samples here and the
        marker returned nothing anyway.
        """

        beat = self.beat
        if beat is None:
            return None
        threshold = (
            fiducial_confidence_threshold(attr)
            if minimum_confidence is None
            else float(minimum_confidence)
        )
        point = getattr(beat, attr, None)
        if point is None or point.confidence < threshold:
            return None
        if attr.startswith("p_") and beat.resolved_wave_status("p").value != "present":
            return None
        if attr.startswith("t_") and beat.resolved_wave_status("t").value != "present":
            return None
        if attr.startswith("u_") and beat.resolved_wave_status("u").value != "present":
            return None
        sample = int(point.sample)
        if sample < 0 or sample >= self.n_samples:
            return None
        return sample

    def _qrs_start(self) -> Optional[int]:
        return self._sample("r_onset") if self._sample("r_onset") is not None else self._sample("q_peak")

    def _qrs_end(self) -> Optional[int]:
        return self._sample("r_offset") if self._sample("r_offset") is not None else self._sample("s_peak")

    def _st_start(self) -> Optional[int]:
        return self._sample("j_point") if self._sample("j_point") is not None else self._qrs_end()

    def _next_qrs_onset(self, after: int) -> Optional[int]:
        """The depolarisation that closes the atrial window.

        Taken from the record's own beats rather than from the consensus beat,
        because the consensus carries one set of fiducials and the window ends
        at the *next* beat. The earliest QRS onset after ``after`` is that
        beat: the list is not guaranteed sorted, and a beat whose onset the
        delineator was not confident about would give a window that ends in
        the middle of a QRS.
        """

        beats = getattr(self.delineation, "beats", None) or ()
        threshold = fiducial_confidence_threshold("r_onset")
        limit = self.n_samples
        best: Optional[int] = None
        for beat in beats:
            point = getattr(beat, "r_onset", None)
            if point is None or float(point.confidence) < threshold:
                continue
            sample = int(point.sample)
            if sample <= int(after) or sample >= limit:
                continue
            if best is None or sample < best:
                best = sample
        return best

    def bounds(self, segment: str) -> Optional[tuple[int, int]]:
        if segment in self._segments:
            return self._segments[segment]

        n = self.n_samples
        qrs_start = self._qrs_start()
        qrs_end = self._qrs_end()
        st_start = self._st_start()
        t_on = self._sample("t_onset")
        t_peak = self._sample("t_peak")
        t_end = self._sample("t_offset")
        u_on = self._sample("u_onset")
        u_end = self._sample("u_offset")
        p_on = self._sample("p_onset")
        p_off = self._sample("p_offset")
        r_peak = self._sample("r_peak")

        start: Optional[int]
        end: Optional[int]
        if segment == "global_vcg":
            start, end = 0, n - 1
        elif segment == "beat":
            start = p_on if p_on is not None else (qrs_start if qrs_start is not None else r_peak)
            end = t_end if t_end is not None else (qrs_end if qrs_end is not None else r_peak)
        elif segment == "p":
            start, end = p_on, p_off
        elif segment == "pr":
            start, end = p_off, qrs_start
        elif segment == "qrs":
            start, end = qrs_start, qrs_end
        elif segment in ("qrs_early", "qrs_mid", "qrs_late"):
            if qrs_start is None or qrs_end is None:
                start, end = None, None
            else:
                one = int(round(qrs_start + (qrs_end - qrs_start) / 3.0))
                two = int(round(qrs_start + 2.0 * (qrs_end - qrs_start) / 3.0))
                if segment == "qrs_early":
                    start, end = qrs_start, one
                elif segment == "qrs_mid":
                    start, end = one, two
                else:
                    start, end = two, qrs_end
        elif segment == "qrst":
            start, end = qrs_start, t_end
        elif segment == "st":
            start = st_start
            end = t_on if t_on is not None else t_peak
        elif segment == "st_t":
            start, end = st_start, t_end
        elif segment == "t":
            start, end = t_on, t_end
        elif segment in ("t_early", "t_mid", "t_late"):
            if t_on is None or t_end is None:
                start, end = None, None
            else:
                one = int(round(t_on + (t_end - t_on) / 3.0))
                two = int(round(t_on + 2.0 * (t_end - t_on) / 3.0))
                if segment == "t_early":
                    start, end = t_on, one
                elif segment == "t_mid":
                    start, end = one, two
                else:
                    start, end = two, t_end
        elif segment == "t_upstroke":
            start, end = t_on, t_peak
        elif segment == "t_downstroke":
            start, end = t_peak, t_end
        elif segment == "u":
            start, end = u_on, u_end
        elif segment == "tq":
            # T end to the next QRS onset: the atrial window that does not ask
            # whether a P wave exists. See ``segment_confidence`` in the
            # delineation for why this window is here at all.
            start = t_end
            end = self._next_qrs_onset(t_end) if t_end is not None else None
        elif segment == "tu":
            start, end = t_end, u_on
        elif segment == "qu":
            start, end = qrs_start, u_end
        else:
            start, end = None, None

        if start is None or end is None:
            self._segments[segment] = None
            return None
        start = max(0, min(int(start), n - 1))
        end = max(0, min(int(end), n - 1))
        if end <= start:
            self._segments[segment] = None
            return None
        self._segments[segment] = (start, end)
        return self._segments[segment]

    def vcg_segment(self, segment: str) -> Optional[np.ndarray]:
        source = (
            self._record_vcg_xyz
            if segment == "global_vcg"
            else self.vcg_xyz
        )
        if source is None:
            return None
        if segment != "global_vcg" and not self.is_segment_evaluable(segment):
            return None
        bounds = self.bounds(segment)
        if bounds is None:
            return None
        start, end = bounds
        data = np.asarray(source[start:end + 1, :], dtype=float)
        return data if data.ndim == 2 and data.shape[0] > 0 else None

    @property
    def vcg_origin_assessment(self) -> VCGOriginAssessment:
        if not self._vcg_origin_assessment_attempted:
            self._vcg_origin_assessment_attempted = True
            origin_source = (
                self._record_vcg_xyz
                if self._record_vcg_xyz is not None
                else self.vcg_xyz
            )
            if origin_source is not None:
                self._vcg_origin_assessment = (
                    estimate_physiological_vcg_origin(
                        origin_source,
                        self.delineation,
                        self.fs,
                    )
                )
        if self._vcg_origin_assessment is None:
            return VCGOriginAssessment(
                primary=None,
                tp=None,
                pr=None,
                status="not_evaluable",
                reason="vcg_signal_unavailable",
            )
        return self._vcg_origin_assessment

    def vcg_segment_origin_corrected(
        self,
        segment: str,
    ) -> Optional[np.ndarray]:
        if segment == "global_vcg":
            data = self.vcg_segment(segment)
            assessment = self.vcg_origin_assessment
            if data is None or assessment.primary is None:
                return None
            return data - assessment.primary.vector_millivolts[np.newaxis, :]
        centered = self.origin_corrected_vcg()
        if centered is None:
            return None
        bounds = self.bounds(segment)
        if bounds is None:
            return None
        start, end = bounds
        return centered[start : end + 1]

    def origin_corrected_vcg(self) -> Optional[np.ndarray]:
        if not self._origin_correction_attempted:
            self._origin_correction_attempted = True
            if self.vcg_xyz is not None:
                self._origin_corrected_vcg_xyz = (
                    self.vcg_origin_assessment.corrected(self.vcg_xyz)
                )
        return self._origin_corrected_vcg_xyz

    def geh_area_metrics(self) -> Optional[GEHAreaMetrics]:
        if self._geh_area_attempted:
            return self._geh_area_metrics
        self._geh_area_attempted = True
        centered = self.origin_corrected_vcg()
        qrs_bounds = self.bounds("qrs")
        qrst_bounds = self.bounds("qrst")
        if centered is None or qrs_bounds is None or qrst_bounds is None:
            return None
        try:
            self._geh_area_metrics = compute_geh_area_metrics(
                centered,
                qrs_onset=qrs_bounds[0],
                qrs_offset=qrs_bounds[1],
                t_offset=qrst_bounds[1],
                fs=self.fs,
            )
        except ValueError:
            self._geh_area_metrics = None
        return self._geh_area_metrics

    def vcg_segment_wideband(self, segment: str) -> Optional[np.ndarray]:
        """Acquisition-qualified 0.05--150 Hz VCG segment, if available.

        High-frequency markers use a fixed 80--150 Hz band.  They must
        abstain unless source metadata establish that the analogue front end
        actually supports 150 Hz; digital sampling rate alone is not proof of
        usable high-frequency physiology.
        """

        source = self._record_vcg_wideband_xyz
        if source is None:
            return None
        if not self.is_segment_evaluable(segment):
            return None
        bounds = self.bounds(segment)
        if bounds is None:
            return None
        start, end = bounds
        if end + 1 > len(source):
            return None
        data = np.asarray(source[start:end + 1, :], dtype=float)
        return data if data.ndim == 2 and data.shape[0] > 0 else None

    def ecg_segment(self, lead: str, segment: str) -> Optional[np.ndarray]:
        source = (
            self._record_ecg_12lead
            if segment == "global_vcg"
            else self.ecg_12lead
        )
        if source is None or lead not in LEAD_MAP:
            return None
        if segment != "global_vcg" and not self.is_segment_evaluable(segment):
            return None
        bounds = self.bounds(segment)
        if bounds is None:
            return None
        start, end = bounds
        idx = LEAD_MAP[lead]
        if idx >= source.shape[1]:
            return None
        return np.asarray(source[start:end + 1, idx], dtype=float)

    def duration_s(self, segment: str) -> Optional[float]:
        bounds = self.bounds(segment)
        if bounds is None:
            return None
        return (bounds[1] - bounds[0]) / float(self.fs)

    def segment_confidence(self, segment: str) -> float:
        """Cache immutable delineation confidence for this computation."""

        if segment not in self._segment_confidences:
            self._segment_confidences[segment] = float(
                self.delineation.segment_confidence(segment)
            )
        return self._segment_confidences[segment]

    def is_segment_evaluable(
        self,
        segment: str,
        minimum_confidence: Optional[float] = None,
    ) -> bool:
        """Evaluate each record-scoped segment gate only once."""

        threshold = (
            None
            if minimum_confidence is None
            else float(minimum_confidence)
        )
        key = (segment, threshold)
        if key not in self._segment_evaluability:
            base = segment.casefold().split("_", 1)[0]
            severe_incoherence = bool(
                self.delineation.diagnostics.get(
                    "severe_signal_incoherence",
                    False,
                )
            )
            if severe_incoherence and base in {
                "p",
                "pr",
                "qrs",
                "st",
                "t",
                "u",
                "qt",
                "jt",
                "qrst",
                "tq",
            }:
                evaluable = False
            else:
                effective_threshold = (
                    measurement_confidence_threshold(segment)
                    if threshold is None
                    else threshold
                )
                evaluable = (
                    self.segment_confidence(segment)
                    >= effective_threshold
                )
            self._segment_evaluability[key] = bool(evaluable)
        return self._segment_evaluability[key]

    def marker_index(self, marker: str) -> Optional[int]:
        mapping = {
            "j_point": "j_point",
            "p_onset": "p_onset",
            "qrs_onset": "r_onset",
            "r_peak": "r_peak",
            "t_end": "t_offset",
            "t_peak": "t_peak",
        }
        return self._sample(mapping.get(marker, marker))


_CONTEXT_STATE = local()


def reset_feature_context() -> None:
    _CONTEXT_STATE.key = None
    _CONTEXT_STATE.context = None
    _CONTEXT_STATE.record_ecg_12lead = None
    _CONTEXT_STATE.record_raw_ecg_12lead = None
    _CONTEXT_STATE.record_acquisition = None
    _CONTEXT_STATE.record_vcg_xyz = None
    _CONTEXT_STATE.record_vcg_wideband_xyz = None


def set_feature_context_record_sources(
    ecg_12lead: Optional[np.ndarray],
    vcg_xyz: Optional[np.ndarray],
    vcg_wideband_xyz: Optional[np.ndarray] = None,
    raw_ecg_12lead: Optional[np.ndarray] = None,
    acquisition: Optional[dict] = None,
) -> None:
    _CONTEXT_STATE.record_ecg_12lead = ecg_12lead
    _CONTEXT_STATE.record_raw_ecg_12lead = raw_ecg_12lead
    _CONTEXT_STATE.record_acquisition = dict(acquisition or {})
    _CONTEXT_STATE.record_vcg_xyz = vcg_xyz
    _CONTEXT_STATE.record_vcg_wideband_xyz = vcg_wideband_xyz


def _context(delineation, ecg_12lead, vcg_xyz, fs) -> FeatureContext:
    # Identity, not id(). CPython recycles the address of a collected object,
    # so a plain id() tuple can match a previous, unrelated record and hand it
    # that record's cached segments, origin and primitive DAG -- a wrong
    # number rather than a crash. Keeping the objects referenced in the key
    # makes the addresses unrecyclable for as long as the cache is live.
    previous = getattr(_CONTEXT_STATE, "key", None)
    if (
        previous is None
        or previous[0] is not delineation
        or previous[1] is not ecg_12lead
        or previous[2] is not vcg_xyz
        or previous[3] != fs
    ):
        _CONTEXT_STATE.key = (delineation, ecg_12lead, vcg_xyz, fs)
        _CONTEXT_STATE.context = FeatureContext(
            delineation,
            ecg_12lead,
            vcg_xyz,
            fs,
        )
    return _CONTEXT_STATE.context


def _memoized_primitive(
    ctx: FeatureContext,
    key: tuple[object, ...],
    factory: Callable[[], object],
):
    """Evaluate one primitive node once for the current ECG context."""

    if key not in ctx._primitive_cache:
        ctx._primitive_cache[key] = factory()
    return ctx._primitive_cache[key]


def _safe_div(a: float, b: float) -> float:
    return float(a / (b + EPS))


def _safe_corr(a: np.ndarray, b: np.ndarray, default: Optional[float] = 0.0) -> Optional[float]:
    """Pearson correlation with finite-value and zero-variance guards."""
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    n = min(len(aa), len(bb))
    if n < 2:
        return default
    aa = aa[:n]
    bb = bb[:n]
    mask = np.isfinite(aa) & np.isfinite(bb)
    if np.sum(mask) < 2:
        return default
    aa = aa[mask]
    bb = bb[mask]
    if np.std(aa) < EPS or np.std(bb) < EPS:
        return default
    corr = float(np.corrcoef(aa, bb)[0, 1])
    return corr if np.isfinite(corr) else default


def _finite(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[np.all(np.isfinite(x), axis=1)] if x.ndim == 2 else np.asarray(x, dtype=float)[np.isfinite(x)]


def _trapz(y: np.ndarray, dx: float, axis: int = 0) -> np.ndarray:
    if len(y) < 2:
        return np.zeros(y.shape[1] if y.ndim == 2 else 1)
    return np.trapezoid(y, dx=dx, axis=axis)


def _entropy(values: np.ndarray, bins: int = 16) -> float:
    values = _finite(np.asarray(values, dtype=float))
    if len(values) < 2 or np.nanmax(values) - np.nanmin(values) < EPS:
        return 0.0
    hist, _ = np.histogram(values, bins=bins)
    p = hist.astype(float) / max(float(np.sum(hist)), EPS)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(bins))


def _angle_between(
    a: Optional[np.ndarray],
    b: Optional[np.ndarray],
) -> Optional[float]:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if (
        a.size == 0
        or b.size == 0
        or a.shape != b.shape
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0))))


def _azimuth(v: np.ndarray) -> Optional[float]:
    return vcg_azimuth_deg(v)


def _elevation(v: np.ndarray) -> Optional[float]:
    return vcg_elevation_deg(v)


def _shoelace(points: np.ndarray) -> float:
    if points is None or len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1]))


def _path_length(points: np.ndarray) -> float:
    if points is None or len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points - np.mean(points, axis=0)
    if len(points) < 3:
        return centered, np.zeros(3), np.eye(3)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    return centered, vals, vecs


def _segment_pca_bundle(
    ctx: FeatureContext,
    segment: str,
) -> Optional[dict[str, np.ndarray]]:
    def build() -> Optional[dict[str, np.ndarray]]:
        points = ctx.vcg_segment(segment)
        if points is None or len(points) < 3:
            return None
        centered, values, vectors = _pca(points)
        return {
            "points": points,
            "centered": centered,
            "values": values,
            "vectors": vectors,
            "scores": centered @ vectors,
        }

    return _memoized_primitive(ctx, ("segment_pca", segment), build)


def _convex_hull_2d_metrics(
    points: np.ndarray,
) -> Optional[tuple[float, float]]:
    """Return area and perimeter without Qhull's per-call temp file.

    SciPy redirects Qhull diagnostics through ``mkstemp`` on Windows. The
    complete marker panel made roughly 26 two-dimensional Qhull calls per ECG,
    which would mean millions of avoidable filesystem operations. Andrew's
    monotone chain is exact for this 2-D task and needs only one native sort.
    """

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        return None
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < 3:
        return None
    ordered = values[np.lexsort((values[:, 1], values[:, 0]))]
    ordered = np.unique(ordered, axis=0)
    if len(ordered) < 3:
        return None

    def cross(origin, first, second) -> float:
        return float(
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[np.ndarray] = []
    for point in ordered:
        while (
            len(lower) >= 2
            and cross(lower[-2], lower[-1], point) <= 0.0
        ):
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while (
            len(upper) >= 2
            and cross(upper[-2], upper[-1], point) <= 0.0
        ):
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        return None
    following = np.roll(hull, -1, axis=0)
    signed_twice_area = np.sum(
        hull[:, 0] * following[:, 1]
        - hull[:, 1] * following[:, 0]
    )
    area = float(abs(signed_twice_area) * 0.5)
    if not np.isfinite(area) or area <= EPS:
        return None
    perimeter = float(
        np.sum(np.linalg.norm(following - hull, axis=1))
    )
    return area, perimeter


def _convex_area_2d(points: np.ndarray) -> Optional[float]:
    metrics = _convex_hull_2d_metrics(points)
    return None if metrics is None else metrics[0]


def _convex_perimeter_2d(points: np.ndarray) -> Optional[float]:
    metrics = _convex_hull_2d_metrics(points)
    return None if metrics is None else metrics[1]


def _convex_volume_3d(points: np.ndarray) -> Optional[float]:
    if len(points) < 4:
        return None
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return None


def _pairwise_diameter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.max(distance.pdist(points)))


def _segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _self_intersections(points: np.ndarray) -> int:
    return self_intersection_count(points)


def _resample_curve(points: np.ndarray, n: int = 64) -> Optional[np.ndarray]:
    if points is None or len(points) < 2:
        return None
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    if s[-1] < EPS:
        return None
    target = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(target, s, points[:, i]) for i in range(points.shape[1])])


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    return dtw_distance(a, b)


def _shape_normalize(points: np.ndarray) -> Optional[np.ndarray]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        return None
    centered = values - np.mean(values, axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if not np.isfinite(scale) or scale < EPS:
        return None
    return centered / scale


def _amplitude_weighted_directions(
    points: np.ndarray,
    *,
    n: int = 24,
) -> Optional[np.ndarray]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        return None
    magnitude = np.linalg.norm(values, axis=1)
    finite = np.all(np.isfinite(values), axis=1) & np.isfinite(magnitude)
    if np.sum(finite) < 3:
        return None
    values = values[finite]
    magnitude = magnitude[finite]
    peak = float(np.max(magnitude))
    keep = magnitude >= max(EPS, 0.05 * peak)
    if np.sum(keep) < 3:
        return None
    values = values[keep]
    magnitude = magnitude[keep]
    cumulative = np.cumsum(magnitude)
    cumulative = (cumulative - 0.5 * magnitude) / max(
        float(np.sum(magnitude)),
        EPS,
    )
    quantiles = (np.arange(n, dtype=float) + 0.5) / n
    sampled = np.column_stack(
        [
            np.interp(quantiles, cumulative, values[:, axis])
            for axis in range(3)
        ]
    )
    sampled_norm = np.linalg.norm(sampled, axis=1, keepdims=True)
    valid = sampled_norm[:, 0] > EPS
    if np.sum(valid) < max(3, n // 2):
        return None
    sampled = sampled[valid] / sampled_norm[valid]
    return sampled


def _directional_wasserstein_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> Optional[float]:
    """Empirical angular 1-Wasserstein distance on the unit sphere.

    Both trajectories are represented by amplitude-weighted directional
    quantiles. The optimal one-to-one assignment uses geodesic angular cost,
    normalized by pi, so the result is bounded to [0, 1].
    """

    a = _amplitude_weighted_directions(first)
    b = _amplitude_weighted_directions(second)
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    cosine = np.clip(a @ b.T, -1.0, 1.0)
    cost = np.arccos(cosine) / np.pi
    rows, columns = linear_sum_assignment(cost)
    return float(np.mean(cost[rows, columns]))


def _recurrence_metrics(points: np.ndarray) -> Optional[tuple[float, float]]:
    """Return fixed-scale recurrence rate and diagonal-line determinism."""

    sampled = _resample_curve(np.asarray(points, dtype=float), 64)
    if sampled is None or len(sampled) < 8:
        return None
    diameter = _pairwise_diameter(sampled)
    if diameter < EPS:
        return None
    matrix = distance.squareform(distance.pdist(sampled))
    n = len(sampled)
    indices = np.arange(n)
    valid = np.abs(indices[:, None] - indices[None, :]) > 2
    recurrent = (matrix <= 0.10 * diameter) & valid
    denominator = int(np.sum(valid))
    if denominator == 0:
        return None
    recurrence_rate = float(np.sum(recurrent) / denominator)

    diagonal_points = 0
    total_recurrent = int(np.sum(recurrent))
    for offset in range(-(n - 1), n):
        if abs(offset) <= 2:
            continue
        diagonal = np.diag(recurrent, k=offset).astype(np.int8)
        if len(diagonal) < 3:
            continue
        padded = np.pad(diagonal, (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        lengths = ends - starts
        diagonal_points += int(np.sum(lengths[lengths >= 3]))
    determinism = (
        float(np.clip(diagonal_points / total_recurrent, 0.0, 1.0))
        if total_recurrent
        else 0.0
    )
    return recurrence_rate, determinism


def _ricker(points: int, width: float) -> np.ndarray:
    points = max(3, int(points))
    x = np.arange(points) - (points - 1.0) / 2.0
    a2 = float(width) ** 2
    wavelet = (1.0 - (x ** 2) / a2) * np.exp(-(x ** 2) / (2.0 * a2))
    norm = np.sqrt(np.sum(wavelet ** 2))
    return wavelet / (norm + EPS)


def _vector_for(ctx: FeatureContext, segment: str, mode: str) -> Optional[tuple[np.ndarray, dict[str, float]]]:
    key = (segment, mode)
    if key in ctx._vectors:
        return ctx._vectors[key]
    data = ctx.vcg_segment_origin_corrected(segment)
    if data is None:
        return None
    if mode == "area":
        vector = _trapz(data, dx=1.0 / ctx.fs, axis=0)
    elif mode == "mean":
        vector = np.mean(data, axis=0)
    elif mode == "max":
        idx = int(np.argmax(np.linalg.norm(data, axis=1)))
        vector = data[idx]
    else:
        return None
    per = {
        "x": float(vector[0]),
        "y": float(vector[1]),
        "z": float(vector[2]),
        "magnitude": float(np.linalg.norm(vector)),
        "axis": float(np.degrees(np.arctan2(vector[1], vector[0]))) if np.linalg.norm(vector[:2]) >= EPS else np.nan,
        "azimuth": _azimuth(vector) if _azimuth(vector) is not None else np.nan,
        "elevation": _elevation(vector) if _elevation(vector) is not None else np.nan,
        **ctx.vcg_origin_assessment.numeric_metadata(),
    }
    ctx._vectors[key] = (vector, per)
    return ctx._vectors[key]


def _differential_bundle(
    ctx: FeatureContext,
    segment: str,
) -> Optional[dict[str, object]]:
    def build() -> Optional[dict[str, object]]:
        data = ctx.vcg_segment(segment)
        duration = ctx.duration_s(segment)
        if data is None or duration is None or len(data) < 4:
            return None
        dt = 1.0 / ctx.fs
        derivative_input = _smooth_derivative_input(data, ctx.fs)
        velocity = np.gradient(derivative_input, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        jerk_vector = np.gradient(acceleration, dt, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(acceleration, axis=1)
        jerk = np.linalg.norm(jerk_vector, axis=1)
        cross_va = np.cross(velocity, acceleration)
        curvature = np.linalg.norm(cross_va, axis=1) / (speed ** 3 + EPS)
        torsion = np.einsum("ij,ij->i", cross_va, jerk_vector) / (
            np.linalg.norm(cross_va, axis=1) ** 2 + EPS
        )
        angular_velocity = np.linalg.norm(
            np.cross(data, velocity), axis=1
        ) / (np.linalg.norm(data, axis=1) ** 2 + EPS)
        return {
            "data": data,
            "duration": duration,
            "dt": dt,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk_vector": jerk_vector,
            "speed": speed,
            "accel": accel,
            "jerk": jerk,
            "cross_va": cross_va,
            "curvature": curvature,
            "torsion": torsion,
            "angular_velocity": angular_velocity,
            "ds": speed * dt,
            "path": _path_length(data),
        }

    return _memoized_primitive(ctx, ("differential", segment), build)


def _kinematic_features(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    if segment in ctx._kinematics:
        return ctx._kinematics[segment]
    bundle = _differential_bundle(ctx, segment)
    if bundle is None:
        return None
    data = bundle["data"]
    duration = float(bundle["duration"])
    dt = float(bundle["dt"])
    speed = bundle["speed"]
    accel = bundle["accel"]
    jerk = bundle["jerk"]
    curvature = bundle["curvature"]
    torsion = bundle["torsion"]
    angular_velocity = bundle["angular_velocity"]
    path = float(bundle["path"])
    jerk_sq = float(_trapz(jerk ** 2, dt))
    dimensionless_jerk = jerk_sq * duration ** 5 / (path ** 2 + EPS)
    features = {
        "velocity_max": float(np.max(speed)),
        "velocity_mean": float(np.mean(speed)),
        "velocity_median": float(np.median(speed)),
        "velocity_rms": float(np.sqrt(np.mean(speed ** 2))),
        "velocity_integral": path,
        "velocity_energy": float(_trapz(speed ** 2, dt)),
        "velocity_entropy": _entropy(speed),
        "velocity_peak_count": float(len(sp_signal.find_peaks(speed)[0])),
        "acceleration_max": float(np.max(accel)),
        "acceleration_mean": float(np.mean(accel)),
        "acceleration_energy": float(_trapz(accel, dt)),
        "jerk_max": float(np.max(jerk)),
        "jerk_mean": float(np.mean(jerk)),
        "jerk_energy": float(_trapz(jerk, dt)),
        "angular_velocity_max": float(np.max(angular_velocity)),
        "angular_velocity_mean": float(np.mean(angular_velocity)),
        "curvature_max": float(np.max(curvature)),
        "curvature_mean": float(np.mean(curvature)),
        "curvature_integral": float(_trapz(curvature, dt)),
        "torsion_max_abs": float(np.max(np.abs(torsion))),
        "torsion_integral_abs": float(_trapz(np.abs(torsion), dt)),
        "path_length_per_time": _safe_div(path, duration),
        "rotation_entropy": _entropy(angular_velocity),
        "sharpness_index": _safe_div(float(np.max(accel)), float(np.mean(speed))),
        "smoothness_index": float(1.0 / (1.0 + dimensionless_jerk)),
    }
    ctx._kinematics[segment] = features
    return features


def _smooth_derivative_input(
    values: np.ndarray,
    fs: int,
    window_ms: float = 15.0,
) -> np.ndarray:
    """Use a fixed, outcome-independent smoother before numerical derivatives."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or len(data) < 5 or fs <= 0:
        return data
    requested = max(5, int(round(window_ms / 1000.0 * fs)))
    if requested % 2 == 0:
        requested += 1
    maximum = len(data) if len(data) % 2 else len(data) - 1
    window = min(requested, maximum)
    if window < 5:
        return data
    return sp_signal.savgol_filter(
        data,
        window_length=window,
        polyorder=min(3, window - 2),
        axis=0,
        mode="interp",
    )


def _oriented_plane_normal(centered: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Resolve PCA's arbitrary normal sign from temporal loop circulation.

    Eigenvectors are axes, not directed vectors: LAPACK may return ``n`` or
    ``-n`` for the same data.  The oriented vector area of the closed temporal
    trajectory supplies a reproducible, rotation-equivariant direction.  A
    nearly self-cancelling loop falls back to a deterministic hemisphere.
    """

    points = np.asarray(centered, dtype=float)
    oriented = np.asarray(normal, dtype=float).copy()
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return oriented
    closed_next = np.roll(points, -1, axis=0)
    vector_area = 0.5 * np.sum(np.cross(points, closed_next), axis=0)
    area_norm = float(np.linalg.norm(vector_area))
    if area_norm > EPS:
        if float(np.dot(oriented, vector_area)) < 0.0:
            oriented = -oriented
        return oriented
    dominant = int(np.argmax(np.abs(oriented)))
    return -oriented if oriented[dominant] < 0.0 else oriented


def _loop3d_features(ctx: FeatureContext, segment: str) -> Optional[dict[str, float]]:
    if segment in ctx._loop3d:
        return ctx._loop3d[segment]
    pca = _segment_pca_bundle(ctx, segment)
    duration = ctx.duration_s(segment)
    if pca is None or duration is None:
        return None
    data = pca["points"]
    centered = pca["centered"]
    vals = pca["values"]
    vecs = pca["vectors"]
    scores = pca["scores"]
    area = abs(_shoelace(scores[:, :2]))
    hull_metrics = _convex_hull_2d_metrics(scores[:, :2])
    hull_area = None if hull_metrics is None else hull_metrics[0]
    hull_perimeter = None if hull_metrics is None else hull_metrics[1]
    path = _path_length(data)
    projected_path = _path_length(scores[:, :2])
    projected_closed_path = projected_path + float(np.linalg.norm(scores[-1, :2] - scores[0, :2]))
    start_end = float(np.linalg.norm(data[-1] - data[0]))
    closed_path = path + start_end
    centroid = np.mean(data, axis=0)
    origin_corrected = ctx.vcg_segment_origin_corrected(segment)
    physiological_centroid = (
        np.mean(origin_corrected, axis=0)
        if origin_corrected is not None and len(origin_corrected)
        else None
    )
    radial = np.linalg.norm(data - centroid, axis=1)
    widths = np.ptp(scores, axis=0)
    total_var = float(np.sum(vals))
    normal = _oriented_plane_normal(centered, vecs[:, 2])
    convex_volume = _convex_volume_3d(data)
    diameter = _pairwise_diameter(data)
    normal_azimuth = _azimuth(normal)
    normal_elevation = _elevation(normal)
    features = {
        "actual_area_to_convex_hull_ratio": _safe_div(area, hull_area or 0.0),
        "area_best_fit_plane": area,
        "bounding_box_volume": float(np.prod(np.ptp(data, axis=0))),
        "bounding_ellipsoid_volume_approx": float(4.0 / 3.0 * np.pi * np.prod(2.0 * np.sqrt(np.maximum(vals, 0.0)))),
        # This is the only origin-dependent scalar in the generic loop-shape
        # bundle.  It must use a physiological TP/UP (PR fallback) origin;
        # otherwise an arbitrary acquisition DC offset is reported as biology.
        "centroid_shift_magnitude": (
            float(np.linalg.norm(physiological_centroid))
            if physiological_centroid is not None
            else np.nan
        ),
        "convex_hull_area_best_fit_plane": float(hull_area) if hull_area is not None else np.nan,
        "convex_hull_perimeter_best_fit_plane": (
            float(hull_perimeter) if hull_perimeter is not None else np.nan
        ),
        "convex_hull_volume_3d": convex_volume if convex_volume is not None else np.nan,
        "diameter": diameter,
        "eccentricity": float(np.sqrt(max(0.0, 1.0 - vals[1] / (vals[0] + EPS)))) if vals[0] > EPS else 0.0,
        "elongation": _safe_div(widths[0], widths[1]),
        "flatness": _safe_div(vals[2], vals[1]),
        "lambda1": float(vals[0]),
        "lambda2": float(vals[1]),
        "lambda3": float(vals[2]),
        "linearity": _safe_div(vals[0], total_var),
        "max_point_distance": diameter,
        "mean_distance_centroid": float(np.mean(radial)),
        "nonplanarity": _safe_div(vals[2], total_var),
        "path_length": path,
        "path_length_per_ms": _safe_div(path, duration * 1000.0),
        "perimeter_best_fit_plane": projected_path,
        "planarity": _safe_div(vals[0] + vals[1], total_var),
        "plane_normal_x": float(normal[0]),
        "plane_normal_y": float(normal[1]),
        "plane_normal_z": float(normal[2]),
        "plane_normal_azimuth": normal_azimuth if normal_azimuth is not None else np.nan,
        "plane_normal_elevation": normal_elevation if normal_elevation is not None else np.nan,
        "radial_skewness": float(skew(radial)) if len(radial) > 2 else 0.0,
        "radial_kurtosis": float(kurtosis(radial)) if len(radial) > 3 else 0.0,
        "return_to_baseline_distance": start_end,
        "rms_distance_centroid": float(np.sqrt(np.mean(radial ** 2))),
        "roundness_best_fit_plane": float(4.0 * np.pi * area / projected_closed_path ** 2) if projected_closed_path > EPS else np.nan,
        "sphericity": _safe_div(vals[2], vals[0]),
        "start_end_distance": start_end,
        "thickness": float(widths[2]),
        "tortuosity": (
            _safe_div(closed_path, hull_perimeter)
            if hull_perimeter is not None and hull_perimeter > EPS
            else np.nan
        ),
        "width_major": float(widths[0]),
        "width_minor": float(widths[1]),
    }
    ctx._loop3d[segment] = features
    return features


def _loop2d_features(ctx: FeatureContext, segment: str, plane: str) -> Optional[dict[str, float]]:
    key = (segment, plane)
    if key in ctx._loop2d:
        return ctx._loop2d[key]
    data = ctx.vcg_segment(segment)
    if data is None or len(data) < 3 or plane not in PLANES:
        return None
    pts = data[:, PLANES[plane]]
    centered = pts - np.mean(pts, axis=0)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    scores = centered @ vecs
    signed_area = _shoelace(pts)
    area = abs(signed_area)
    perimeter = _path_length(pts)
    # Shoelace closes the polygon. Circularity must use that same boundary;
    # the separate perimeter marker intentionally remains the observed path.
    closed_perimeter = perimeter + float(np.linalg.norm(pts[-1] - pts[0]))
    hull_area = _convex_area_2d(pts)
    angles = np.unwrap(np.arctan2(centered[:, 1], centered[:, 0]))
    turns = np.diff(np.unwrap(np.arctan2(np.diff(pts, axis=0)[:, 1], np.diff(pts, axis=0)[:, 0]))) if len(pts) > 2 else np.array([])
    features = {
        "area": area,
        "convex_hull_area": float(hull_area) if hull_area is not None else np.nan,
        "eccentricity": float(np.sqrt(max(0.0, 1.0 - vals[1] / (vals[0] + EPS)))) if vals[0] > EPS else 0.0,
        "max_width": float(np.ptp(scores[:, 0])),
        "min_width": float(np.ptp(scores[:, 1])),
        "perimeter": perimeter,
        "rotation_change_count": float(np.sum(np.diff(np.sign(turns)) != 0)) if len(turns) > 1 else 0.0,
        "rotation_direction": float(np.sign(signed_area)),
        "roundness": float(4.0 * np.pi * area / closed_perimeter ** 2) if closed_perimeter > EPS else np.nan,
        "self_intersection_count": float(_self_intersections(pts)),
        "signed_area": signed_area,
        "winding_number": float((angles[-1] - angles[0]) / (2.0 * np.pi)) if len(angles) > 1 else 0.0,
    }
    ctx._loop2d[key] = features
    return features


_CATALOG_DISPATCH_CACHE: dict[
    str,
    Callable[[FeatureContext, str], Optional[FeatureValue]],
] = {}


class _InputUnavailable:
    """A dispatcher could not read its input, and claims no name by saying so.

    The dispatch protocol had two return values and needed three. ``None``
    meant "not my name"; anything else meant "mine, here is the answer" -- and
    the loop below cached the first non-``None`` responder as the permanent
    owner of that name.

    Two dispatchers answered before looking at the name at all:

        def _experimental_feature(ctx, name):
            points = ctx.vcg_segment("qrs")
            if points is None:
                return FeatureValue(None, 0.0)      # for *every* name

    ``_morph_signal_feature`` did the same. So a record whose QRS segment is
    unusable -- about one in forty -- made ``_experimental_feature`` claim
    every name that reached it, and the claim was then cached for the life of
    the worker process. The fifteen panel markers owned by the five
    dispatchers after it came back empty from that worker for every subsequent
    recording, among them ``spatial_peak_qrs_t_angle_deg``,
    ``microvolt_t_wave_alternans_vector`` and
    ``qrs_fragmentation_high_frequency_energy``.

    Nothing failed. The columns were merely emptier in some workers than in
    others, which also means the same recording could yield different values
    depending on which worker took it and what that worker happened to see
    first -- reproducibility lost to a cache.

    The third answer says "the name may be mine, but I cannot compute it from
    this record". The loop keeps searching, caches nothing, and falls back to
    the abstention only if no later dispatcher owns the name.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<input unavailable>"


_INPUT_UNAVAILABLE = _InputUnavailable()


def _catalog_value(ctx: FeatureContext, name: str) -> FeatureValue:
    cached = _CATALOG_DISPATCH_CACHE.get(name)
    if cached is not None:
        value = cached(ctx, name)
        if isinstance(value, _InputUnavailable):
            return FeatureValue(
                None,
                confidence=0.0,
                notes="Required inputs are unavailable for this catalog candidate.",
            )
        if value is not None:
            return value
        # A supported calculator may return None when its required signal is
        # unavailable.  Do not pay for probing every unrelated family again.
        return FeatureValue(
            None,
            confidence=0.0,
            notes="Required inputs are unavailable for this catalog candidate.",
        )

    # Resolve the name to its calculator once per worker process.  The first
    # evaluable record pays the legacy dispatcher search; all subsequent ECGs
    # call the compiled leaf directly.
    dispatchers = (
        _axis_feature,
        _vector_feature,
        _kinematic_feature,
        _kinematic_landmark_feature,
        _loop_geometry_feature,
        _loop_projection_feature,
        _ecg_projection_feature,
        _interval_feature,
        _spatial_magnitude_feature,
        _qrst_feature,
        _segment_specific_feature,
        _angle_relation_feature,
        _st_timepoint_feature,
        _pathophysiology_feature,
        _loop_extra_feature,
        _math_loop_feature,
        _quality_feature,
        _pca_svd_feature,
        _r_progression_feature,
        _additional_vcg_idea_feature,
        _beat_variability_feature,
        _experimental_feature,
        _literature_feature,
        _robustness_feature,
        _review_feature,
        _morph_signal_feature,
        _template_feature,
    )
    unavailable = False
    for calculator in dispatchers:
        value = calculator(ctx, name)
        if isinstance(value, _InputUnavailable):
            # A dispatcher that cannot read its input has not identified
            # itself as the owner of this name. Keep looking, and above all do
            # not cache it: that is how fifteen markers were emptied for the
            # life of a worker process.
            unavailable = True
            continue
        if value is not None:
            _CATALOG_DISPATCH_CACHE[name] = calculator
            return value
    if unavailable:
        return FeatureValue(
            None,
            confidence=0.0,
            notes="Required inputs are unavailable for this catalog candidate.",
        )
    return FeatureValue(None, confidence=0.0, notes="No scientifically defined single-record calculator is available for this candidate.")


def _axis_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if ctx.vcg_xyz is None:
        return None
    axis_map = {"x": 0, "y": 1, "z": 2}
    if name in ("vcg_x_max/min", "vcg_y_max/min", "vcg_z_max/min"):
        axis = name.split("_")[1]
        vals = ctx.vcg_xyz[:, axis_map[axis]]
        return FeatureValue(float(np.max(np.abs(vals))), per_lead={"max": float(np.max(vals)), "min": float(np.min(vals))})
    if name in ("vcg_x_peak_to_peak", "vcg_y_peak_to_peak", "vcg_z_peak_to_peak"):
        axis = name.split("_")[1]
        vals = ctx.vcg_xyz[:, axis_map[axis]]
        return FeatureValue(float(np.ptp(vals)))
    if name in ("vcg_x_negative_area", "vcg_x_positive_area", "vcg_y_negative_area", "vcg_y_positive_area", "vcg_z_negative_area", "vcg_z_positive_area"):
        axis = name.split("_")[1]
        sign = name.split("_")[2]
        vals = ctx.vcg_xyz[:, axis_map[axis]]
        selected = np.minimum(vals, 0.0) if sign == "negative" else np.maximum(vals, 0.0)
        return FeatureValue(float(_trapz(selected, 1.0 / ctx.fs)))
    if name in ("vcg_x_y_ptp_ratio", "vcg_x_z_ptp_ratio", "vcg_y_z_ptp_ratio"):
        parts = name.split("_")
        a = np.ptp(ctx.vcg_xyz[:, axis_map[parts[1]]])
        b = np.ptp(ctx.vcg_xyz[:, axis_map[parts[2]]])
        return FeatureValue(_safe_div(a, b))
    if name == "axis_anisotropy_std_max_min_ratio":
        stds = np.std(ctx.vcg_xyz, axis=0)
        return FeatureValue(_safe_div(float(np.max(stds)), float(np.min(stds))))
    if name == "axis_compression_index_std_min_max":
        stds = np.std(ctx.vcg_xyz, axis=0)
        return FeatureValue(_safe_div(float(np.min(stds)), float(np.max(stds))))
    if name in ("frontal_plane_dominance_xy", "horizontal_plane_dominance_xz", "sagittal_plane_dominance_yz", "x/y/z_dominance_variance_fraction"):
        var = np.var(ctx.vcg_xyz, axis=0)
        total = float(np.sum(var))
        if name == "frontal_plane_dominance_xy":
            return FeatureValue(_safe_div(var[0] + var[1], total))
        if name == "horizontal_plane_dominance_xz":
            return FeatureValue(_safe_div(var[0] + var[2], total))
        if name == "sagittal_plane_dominance_yz":
            return FeatureValue(_safe_div(var[1] + var[2], total))
        per = {"x": _safe_div(var[0], total), "y": _safe_div(var[1], total), "z": _safe_div(var[2], total)}
        return FeatureValue(float(max(per.values())), per_lead=per)
    return None


def _vector_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    for segment in ("p", "qrs", "qrst", "st", "st_t", "t"):
        for mode in ("area", "max", "mean"):
            prefix = f"{segment}_{mode}_vector"
            if name == f"{prefix}_x/y/z/magnitude/axis":
                result = _vector_for(ctx, segment, mode)
                if result is None:
                    return FeatureValue(None, confidence=0.0, notes=f"Missing {segment} segment")
                _, per = result
                return FeatureValue(float(per["magnitude"]), per_lead=per)
    if name == "qrs_initial_20ms_vector":
        return _window_mean_vector(ctx, "qrs", start_ms=0, length_ms=20)
    if name == "qrs_terminal_20ms_vector":
        return _window_mean_vector(ctx, "qrs", end_aligned=True, length_ms=20)
    if name == "tpeak_vector / tend_vector / j_point_vector":
        per = {}
        vectors = []
        for label, marker in (("tpeak", "t_peak"), ("tend", "t_offset"), ("j_point", "j_point")):
            idx = ctx.marker_index(marker)
            if idx is not None and ctx.vcg_xyz is not None and 0 <= idx < len(ctx.vcg_xyz):
                vec = ctx.vcg_xyz[idx]
                vectors.append(vec)
                per[f"{label}_magnitude"] = float(np.linalg.norm(vec))
        if not vectors:
            return FeatureValue(None, confidence=0.0, notes="No requested landmark vector is available")
        return FeatureValue(float(np.mean([np.linalg.norm(v) for v in vectors])), per_lead=per)
    return None


def _window_mean_vector(ctx: FeatureContext, segment: str, start_ms: int = 0, length_ms: int = 20, end_aligned: bool = False) -> FeatureValue:
    bounds = ctx.bounds(segment)
    if bounds is None or ctx.vcg_xyz is None:
        return FeatureValue(None, confidence=0.0, notes=f"Missing {segment} segment")
    samples = max(1, int(round(length_ms * ctx.fs / 1000.0)))
    start, end = bounds
    if end_aligned:
        lo, hi = max(start, end - samples + 1), end + 1
    else:
        lo, hi = start + int(round(start_ms * ctx.fs / 1000.0)), min(end + 1, start + samples)
    if hi <= lo:
        return FeatureValue(None, confidence=0.0, notes="Window is outside segment")
    vec = np.mean(ctx.vcg_xyz[lo:hi], axis=0)
    per = {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2]), "magnitude": float(np.linalg.norm(vec))}
    return FeatureValue(float(per["magnitude"]), per_lead=per)


def _kinematic_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    for segment in SEGMENTS:
        prefix = f"{segment}_"
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        features = _kinematic_features(ctx, segment)
        if features and suffix in features:
            return FeatureValue(features[suffix])
    return None


def _kinematic_landmark_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name in ("late_qrs_slowing_index", "qrs_velocity_t_velocity_ratio", "t_wave_deceleration_index"):
        if name == "late_qrs_slowing_index":
            early = _subsegment_kinematic(ctx, "qrs", 0.0, 1.0 / 3.0)
            late = _subsegment_kinematic(ctx, "qrs", 2.0 / 3.0, 1.0)
            return FeatureValue(_safe_div(late or 0.0, early or 0.0)) if early is not None and late is not None else FeatureValue(None, 0.0)
        if name == "qrs_velocity_t_velocity_ratio":
            qrs = _kinematic_features(ctx, "qrs")
            t = _kinematic_features(ctx, "t")
            return FeatureValue(_safe_div(qrs["velocity_mean"], t["velocity_mean"])) if qrs and t else FeatureValue(None, 0.0)
        up = _subsegment_bounds_by_marker(ctx, "t", ctx._sample("t_onset"), ctx._sample("t_peak"))
        down = _subsegment_bounds_by_marker(ctx, "t", ctx._sample("t_peak"), ctx._sample("t_offset"))
        up_v = _mean_speed_for_bounds(ctx, up)
        down_v = _mean_speed_for_bounds(ctx, down)
        return FeatureValue(_safe_div(down_v or 0.0, up_v or 0.0)) if up_v is not None and down_v is not None else FeatureValue(None, 0.0)
    if name.startswith("velocity_at_"):
        marker = name[len("velocity_at_"):]
        idx = ctx.marker_index(marker)
        if idx is None or ctx.vcg_xyz is None or idx <= 0 or idx >= len(ctx.vcg_xyz):
            return FeatureValue(None, confidence=0.0, notes=f"Missing {marker}")
        v = (ctx.vcg_xyz[idx] - ctx.vcg_xyz[idx - 1]) * ctx.fs
        return FeatureValue(float(np.linalg.norm(v)))
    return None


def _subsegment_kinematic(ctx: FeatureContext, segment: str, a: float, b: float) -> Optional[float]:
    bounds = ctx.bounds(segment)
    if bounds is None:
        return None
    start, end = bounds
    lo = int(round(start + (end - start) * a))
    hi = int(round(start + (end - start) * b))
    return _mean_speed_for_bounds(ctx, (lo, hi))


def _subsegment_bounds_by_marker(ctx: FeatureContext, segment: str, start: Optional[int], end: Optional[int]) -> Optional[tuple[int, int]]:
    if start is None or end is None:
        return None
    bounds = ctx.bounds(segment)
    if bounds is None:
        return None
    lo = max(bounds[0], start)
    hi = min(bounds[1], end)
    return (lo, hi) if hi > lo else None


def _mean_speed_for_bounds(ctx: FeatureContext, bounds: Optional[tuple[int, int]]) -> Optional[float]:
    if bounds is None or ctx.vcg_xyz is None:
        return None
    lo, hi = bounds
    data = ctx.vcg_xyz[lo:hi + 1]
    if len(data) < 2:
        return None
    return float(np.mean(np.linalg.norm(np.diff(data, axis=0) * ctx.fs, axis=1)))


def _loop_geometry_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    for segment in ("p", "qrs", "qrst", "st", "t"):
        prefix = f"{segment}_loop_"
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if any(suffix.startswith(plane) for plane in PLANES):
            continue
        features = _loop3d_features(ctx, segment)
        if not features:
            return FeatureValue(None, confidence=0.0, notes=f"Missing {segment} loop")
        if suffix == "lambda1/lambda2/lambda3":
            return FeatureValue(features["lambda1"], per_lead={k: features[k] for k in ("lambda1", "lambda2", "lambda3")})
        if suffix == "plane_normal_x/y/z/azimuth/elevation":
            keys = ["plane_normal_x", "plane_normal_y", "plane_normal_z", "plane_normal_azimuth", "plane_normal_elevation"]
            return FeatureValue(features["plane_normal_azimuth"], per_lead={k: features[k] for k in keys})
        if suffix == "radial_skewness/kurtosis":
            return FeatureValue(features["radial_skewness"], per_lead={"skewness": features["radial_skewness"], "kurtosis": features["radial_kurtosis"]})
        if suffix in features:
            return FeatureValue(features[suffix])
    return None


def _loop_projection_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    for segment in ("p", "qrs", "qrst", "st", "t"):
        for plane in PLANES:
            prefix = f"{segment}_loop_{plane}_"
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            features = _loop2d_features(ctx, segment, plane)
            if not features:
                return FeatureValue(None, confidence=0.0, notes=f"Missing {segment} loop projection")
            if suffix in features:
                return FeatureValue(features[suffix])
    return None


def _ecg_projection_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    lead_lookup = {"II": "II", "I": "I", "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6", "aVF": "aVF"}
    for token, lead in lead_lookup.items():
        if name == f"lead_{token}_qrs_peak_positive/negative/abs_peak":
            data = ctx.ecg_segment(lead, "qrs")
            if data is None or len(data) == 0:
                return FeatureValue(None, confidence=0.0, notes=f"Missing lead {lead} QRS segment")
            pos = float(np.max(data) * 1000.0)
            neg = float(np.min(data) * 1000.0)
            abs_peak = float(np.max(np.abs(data)) * 1000.0)
            return FeatureValue(abs_peak, per_lead={"positive": pos, "negative": neg, "abs_peak": abs_peak})
    if name == "cornell_voltage_surrogate":
        avl = ctx.ecg_segment("aVL", "qrs")
        v3 = ctx.ecg_segment("V3", "qrs")
        if avl is None or v3 is None:
            return FeatureValue(None, confidence=0.0, notes="Missing aVL or V3")
        return FeatureValue(float((np.max(avl) + abs(np.min(v3))) * 1000.0))
    if name == "sokolow_lyon_voltage_surrogate":
        v1 = ctx.ecg_segment("V1", "qrs")
        v5 = ctx.ecg_segment("V5", "qrs")
        v6 = ctx.ecg_segment("V6", "qrs")
        if v1 is None or v5 is None or v6 is None:
            return FeatureValue(None, confidence=0.0, notes="Missing V1, V5 or V6")
        return FeatureValue(float((abs(np.min(v1)) + max(np.max(v5), np.max(v6))) * 1000.0))
    if name == "peguero_lo_presti_voltage_surrogate":
        if ctx.ecg_12lead is None:
            return FeatureValue(None, confidence=0.0, notes="Missing 12-lead ECG")
        v4 = ctx.ecg_segment("V4", "qrs")
        if v4 is None:
            return FeatureValue(None, confidence=0.0, notes="Missing V4")
        lead_s_depths = []
        for lead in LEAD_MAP:
            segment = ctx.ecg_segment(lead, "qrs")
            if segment is not None and len(segment):
                lead_s_depths.append(abs(float(np.min(segment))))
        if not lead_s_depths:
            return FeatureValue(None, confidence=0.0, notes="Missing QRS segments")
        deepest_s = max(lead_s_depths)
        return FeatureValue(float((deepest_s + abs(np.min(v4))) * 1000.0))
    return None


def _interval_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    beat = ctx.beat
    if beat is None:
        return None
    rr = ventricular_rr_seconds(
        [b.r_peak.sample for b in ctx.delineation.beats],
        ctx.fs,
    )
    qrs_start = ctx._qrs_start()
    qrs_end = ctx._qrs_end()
    t_end = ctx._sample("t_offset")
    t_on = ctx._sample("t_onset")
    t_peak = ctx._sample("t_peak")
    p_on = ctx._sample("p_onset")
    p_off = ctx._sample("p_offset")

    def ms(a, b):
        """Duration between two fiducials of the same beat, or nothing.

        A negative duration is not a short interval, it is two landmarks in
        the wrong order: measured on the store, ``st_duration_ms`` reached
        -26 ms and ``tpeak_tend_ms`` -10 ms, which say that T onset was placed
        before the J point and T end before the T peak. Returning the negative
        number let an inconsistent delineation enter the distribution as an
        extreme value; abstaining sends it where the other unavailable
        fiducials already go.

        ``electric_diastole_ms`` is deliberately not routed through here. It
        is a difference rather than an interval, and it is legitimately
        negative when repolarisation outlasts the cycle length.
        """

        if a is None or b is None:
            return None
        duration = (b - a) / ctx.fs * 1000.0
        return None if duration < 0.0 else duration

    qt = ms(qrs_start, t_end)
    qrs = ms(qrs_start, qrs_end)
    mapping = {
        "heart_rate_bpm": None if rr is None or rr <= 0 else 60.0 / rr,
        "rr_interval_ms": None if rr is None else rr * 1000.0,
        "rr_variance_ms2": None if len(ctx.delineation.beats) < 3 else float(np.var(np.diff([b.r_peak.sample for b in ctx.delineation.beats]) / ctx.fs * 1000.0)),
        "p_duration_ms": ms(p_on, p_off),
        "pr_interval_ms": ms(p_on, qrs_start),
        "qrs_duration_ms": qrs,
        "qt_interval_ms": qt,
        "jt_interval_ms": ms(qrs_end, t_end),
        "p_onset_to_t_end_ms": ms(p_on, t_end),
        "st_duration_ms": ms(ctx._st_start(), t_on),
        "t_duration_ms": ms(t_on, t_end),
        "tpeak_tend_ms": ms(t_peak, t_end),
        "electric_systole_ms": qt,
        "electric_diastole_ms": None if rr is None or qt is None else rr * 1000.0 - qt,
    }
    if name in mapping:
        if mapping[name] is None:
            return FeatureValue(
                None,
                0.0,
                notes="Required fiducials unavailable or out of order",
            )
        dependencies = _feature_segment_dependencies(
            name,
            "Zeitintervalle",
        )
        confidence = (
            min(
                ctx.segment_confidence(segment)
                for segment in dependencies
            )
            if dependencies
            else max(0.35, ctx.delineation.quality_score)
        )
        return FeatureValue(float(mapping[name]), confidence=confidence)
    if name in ("qtc_bazett_ms", "qtc_fridericia_ms", "qtc_framingham_ms", "jtc_bazett/fridericia/framingham_ms"):
        measurements = ctx.delineation.qt_rr_measurements()
        required = (
            5
            if ctx.delineation.rhythm.requires_beatwise_rate_correction
            else 1
        )
        if len(measurements) < required:
            return FeatureValue(
                None,
                0.0,
                notes=(
                    "QTc not evaluable: "
                    f"{len(measurements)}/{required} valid beatwise QT-RR pairs"
                ),
            )
        qt_values = np.asarray([item.qt_ms for item in measurements], dtype=float)
        rr_values = np.asarray(
            [item.preceding_rr_seconds for item in measurements],
            dtype=float,
        )
        qtc_b_values = qt_values / np.sqrt(rr_values)
        qtc_f_values = qt_values / np.cbrt(rr_values)
        qtc_fr_values = qt_values + 154.0 * (1.0 - rr_values)
        qtc_b = float(np.median(qtc_b_values))
        qtc_f = float(np.median(qtc_f_values))
        qtc_fr = float(np.median(qtc_fr_values))
        confidence = float(
            np.median([item.confidence for item in measurements])
            * min(1.0, 0.55 + 0.09 * len(measurements))
        )
        details = {
            "n_qt_rr_pairs": float(len(measurements)),
            "rr_median_ms": float(np.median(rr_values) * 1000.0),
            "qt_median_ms": float(np.median(qt_values)),
            "qtc_iqr_ms": float(
                np.percentile(qtc_f_values, 75)
                - np.percentile(qtc_f_values, 25)
            ),
        }
        notes = (
            "Robust median of QT values corrected with each beat's "
            f"preceding RR (n={len(measurements)})"
        )
        if name == "qtc_bazett_ms":
            return FeatureValue(qtc_b, confidence, details, notes)
        if name == "qtc_fridericia_ms":
            return FeatureValue(qtc_f, confidence, details, notes)
        if name == "qtc_framingham_ms":
            return FeatureValue(qtc_fr, confidence, details, notes)
        qrs_values = np.asarray(
            [
                item.qrs_ms
                for item in measurements
                if item.qrs_ms is not None
            ],
            dtype=float,
        )
        if len(qrs_values) != len(measurements):
            return FeatureValue(None, 0.0, notes="JTC requires QRS duration per beat")
        jtc_b = qtc_b_values - qrs_values
        jtc_f = qtc_f_values - qrs_values
        jtc_fr = qtc_fr_values - qrs_values
        return FeatureValue(
            float(np.median(jtc_f)),
            confidence,
            {
                **details,
                "bazett": float(np.median(jtc_b)),
                "fridericia": float(np.median(jtc_f)),
                "framingham": float(np.median(jtc_fr)),
            },
            notes,
        )
    if name == "qrs_early/mid/late_duration_ms":
        if qrs is None:
            return FeatureValue(None, 0.0)
        part = qrs / 3.0
        return FeatureValue(float(part), per_lead={"early": float(part), "mid": float(part), "late": float(part)})
    if name == "t_upstroke/downstroke_duration_ms":
        up = ms(t_on, t_peak)
        down = ms(t_peak, t_end)
        if up is None or down is None:
            return FeatureValue(None, 0.0)
        return FeatureValue(float(up + down), per_lead={"upstroke": float(up), "downstroke": float(down)})
    return None


def _spatial_magnitude_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    supported = {
        "spatial_magnitude_m_t",
        "spatial_magnitude_integral",
        "spatial_magnitude_max",
        "spatial_magnitude_mean",
        "spatial_magnitude_median",
        "spatial_magnitude_p95",
        "spatial_magnitude_rms",
        "spatial_low_voltage_qrs_peak_magnitude",
        "magnitude_area_qrs",
        "magnitude_area_qrst",
        "magnitude_area_st_t",
    }
    if name not in supported:
        return None
    if ctx.vcg_xyz is None:
        return None
    centered = ctx.origin_corrected_vcg()
    if centered is None:
        return FeatureValue(
            None,
            0.0,
            notes=ctx.vcg_origin_assessment.provenance_note(),
        )
    origin_metadata = ctx.vcg_origin_assessment.numeric_metadata()
    mag = np.linalg.norm(centered, axis=1)
    if name == "spatial_magnitude_m_t":
        return FeatureValue(
            float(np.max(mag)),
            per_lead={
                "mean": float(np.mean(mag)),
                "max": float(np.max(mag)),
                **origin_metadata,
            },
            notes=ctx.vcg_origin_assessment.provenance_note(),
        )
    if name == "spatial_magnitude_integral":
        return FeatureValue(
            float(_trapz(mag, 1.0 / ctx.fs)),
            per_lead=origin_metadata,
            notes=ctx.vcg_origin_assessment.provenance_note(),
        )
    if name == "spatial_magnitude_max":
        return FeatureValue(float(np.max(mag)), per_lead=origin_metadata)
    if name == "spatial_magnitude_mean":
        return FeatureValue(float(np.mean(mag)), per_lead=origin_metadata)
    if name == "spatial_magnitude_median":
        return FeatureValue(float(np.median(mag)), per_lead=origin_metadata)
    if name == "spatial_magnitude_p95":
        return FeatureValue(float(np.percentile(mag, 95)), per_lead=origin_metadata)
    if name == "spatial_magnitude_rms":
        return FeatureValue(
            float(np.sqrt(np.mean(mag ** 2))),
            per_lead=origin_metadata,
        )
    if name == "spatial_low_voltage_qrs_peak_magnitude":
        data = ctx.vcg_segment_origin_corrected("qrs")
        return (
            FeatureValue(None, 0.0, notes=ctx.vcg_origin_assessment.provenance_note())
            if data is None
            else FeatureValue(
                float(np.max(np.linalg.norm(data, axis=1))),
                per_lead=origin_metadata,
                notes=ctx.vcg_origin_assessment.provenance_note(),
            )
        )
    for segment_name, segment in (("magnitude_area_qrs", "qrs"), ("magnitude_area_qrst", "qrst"), ("magnitude_area_st_t", "st_t")):
        if name == segment_name:
            data = ctx.vcg_segment_origin_corrected(segment)
            return (
                FeatureValue(None, 0.0, notes=ctx.vcg_origin_assessment.provenance_note())
                if data is None
                else FeatureValue(
                    float(_trapz(np.linalg.norm(data, axis=1), 1.0 / ctx.fs)),
                    per_lead=origin_metadata,
                    notes=ctx.vcg_origin_assessment.provenance_note(),
                )
            )
    return None


def _qrst_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name in ("qrs_area_to_t_area_angle_deg", "spatial_qrs_t_angle_deg", "svg_magnitude", "spatial_ventricular_gradient_vector", "svg_angle_to_qrs_area_deg", "svg_angle_to_t_area_deg", "svg_qrs_area_ratio", "svg_t_area_ratio"):
        metrics = ctx.geh_area_metrics()
        if metrics is None:
            return FeatureValue(
                None,
                0.0,
                notes=ctx.vcg_origin_assessment.provenance_note(),
            )
        qrs_area = metrics.qrs_area_vector_uv_ms
        # In published GEH terminology the T-area vector is the complete JT
        # integral (QRS offset through T offset), including the ST segment.
        t_area = metrics.jt_area_vector_uv_ms
        svg = metrics.qrst_area_vector_uv_ms
        metadata = {
            "qrs_area_x_uv_ms": float(qrs_area[0]),
            "qrs_area_y_uv_ms": float(qrs_area[1]),
            "qrs_area_z_uv_ms": float(qrs_area[2]),
            "jt_area_x_uv_ms": float(t_area[0]),
            "jt_area_y_uv_ms": float(t_area[1]),
            "jt_area_z_uv_ms": float(t_area[2]),
            "svg_x_uv_ms": float(svg[0]),
            "svg_y_uv_ms": float(svg[1]),
            "svg_z_uv_ms": float(svg[2]),
            **origin_sensitivity_metadata(
                ctx.vcg_origin_assessment,
                qrst_duration_ms=metrics.qrst_duration_ms,
            ),
        }
        note = (
            ctx.vcg_origin_assessment.provenance_note()
            + f"; geh_formula_version={GEH_FORMULA_VERSION}; "
            "T-area=JT(QRS-offset:T-offset)"
        )
        if name == "qrs_area_to_t_area_angle_deg":
            return FeatureValue(
                metrics.qrs_jt_angle_deg,
                per_lead=metadata,
                notes=note,
            )
        if name == "spatial_qrs_t_angle_deg":
            return FeatureValue(
                metrics.qrs_jt_angle_deg,
                per_lead=metadata,
                notes=note,
            )
        if name in ("svg_magnitude", "spatial_ventricular_gradient_vector"):
            metadata["magnitude_uv_ms"] = metrics.svg_magnitude_uv_ms
            return FeatureValue(
                metrics.svg_magnitude_uv_ms,
                per_lead=metadata,
                notes=note,
            )
        if name == "svg_angle_to_qrs_area_deg":
            return FeatureValue(_angle_between(svg, qrs_area), per_lead=metadata, notes=note)
        if name == "svg_angle_to_t_area_deg":
            return FeatureValue(_angle_between(svg, t_area), per_lead=metadata, notes=note)
        if name == "svg_qrs_area_ratio":
            return FeatureValue(
                _safe_div(metrics.svg_magnitude_uv_ms, float(np.linalg.norm(qrs_area))),
                per_lead=metadata,
                notes=note,
            )
        if name == "svg_t_area_ratio":
            return FeatureValue(
                _safe_div(metrics.svg_magnitude_uv_ms, float(np.linalg.norm(t_area))),
                per_lead=metadata,
                notes=note,
            )
    if name == "terminal_repolarization_burden":
        bounds = ctx.bounds("t")
        centered = ctx.origin_corrected_vcg()
        if bounds is None or centered is None:
            return FeatureValue(None, 0.0)
        start, end = bounds
        late_start = int(round(start + 2.0 * (end - start) / 3.0))
        vec = np.mean(centered[late_start:end + 1], axis=0)
        return FeatureValue(
            float(np.linalg.norm(vec)),
            per_lead=ctx.vcg_origin_assessment.numeric_metadata(),
            notes=ctx.vcg_origin_assessment.provenance_note(),
        )
    qrs = ctx.vcg_segment("qrs")
    t = ctx.vcg_segment("t")
    if qrs is None or t is None:
        return None
    if name in ("qrs_loop_area_t_loop_area_ratio", "qrs_loop_volume_t_loop_volume_ratio"):
        qrs_f = _loop3d_features(ctx, "qrs")
        t_f = _loop3d_features(ctx, "t")
        if not qrs_f or not t_f:
            return FeatureValue(None, 0.0)
        if name == "qrs_loop_area_t_loop_area_ratio":
            return FeatureValue(_safe_div(qrs_f["area_best_fit_plane"], t_f["area_best_fit_plane"]))
        return FeatureValue(_safe_div(qrs_f["bounding_ellipsoid_volume_approx"], t_f["bounding_ellipsoid_volume_approx"]))
    if name.startswith("qrs_t_"):
        q = _resample_curve(qrs)
        tt = _resample_curve(t)
        if q is None or tt is None:
            return FeatureValue(None, 0.0)
        q_flat = q.ravel()
        t_flat = tt.ravel()
        if name == "qrs_t_cosine_distance":
            return FeatureValue(float(distance.cosine(q_flat, t_flat)) if np.linalg.norm(q_flat) > EPS and np.linalg.norm(t_flat) > EPS else None)
        if name == "qrs_t_vector_correlation_distance":
            corr = _safe_corr(q_flat, t_flat, default=None)
            return FeatureValue(float(1.0 - corr) if corr is not None else None)
        if name == "qrs_t_dtw_distance":
            return FeatureValue(_dtw_distance(q, tt))
        if name == "qrs_t_hausdorff_distance":
            return FeatureValue(float(max(distance.directed_hausdorff(q, tt)[0], distance.directed_hausdorff(tt, q)[0])))
        if name == "qrs_t_procrustes_distance":
            try:
                _, _, disparity = procrustes(q, tt)
                return FeatureValue(float(disparity))
            except Exception:
                return FeatureValue(None, 0.0)
        if name == "qrs_t_morphological_discordance":
            q_normalized = _shape_normalize(q)
            t_normalized = _shape_normalize(tt)
            if q_normalized is None or t_normalized is None:
                return FeatureValue(None, 0.0)
            q_flat = q_normalized.ravel()
            t_flat = t_normalized.ravel()
            cosine_component = float(
                np.clip(distance.cosine(q_flat, t_flat) / 2.0, 0.0, 1.0)
            )
            dtw = _dtw_distance(q_normalized, t_normalized)
            dtw_component = float(dtw / (1.0 + dtw))
            try:
                _, _, disparity = procrustes(q_normalized, t_normalized)
                procrustes_component = float(
                    disparity / (1.0 + disparity)
                )
            except Exception:
                procrustes_component = np.nan
            vals = np.array(
                [
                    cosine_component,
                    dtw_component,
                    procrustes_component,
                ],
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            return FeatureValue(
                float(np.mean(vals)) if len(vals) == 3 else None,
                confidence=0.65 if len(vals) == 3 else 0.0,
                notes=(
                    "Mean of bounded cosine, normalized-DTW and Procrustes "
                    "shape components; no dimensional distances are mixed."
                ),
            )
    return None


def _segment_specific_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "initial_5/10/20/40ms_qrs_vector":
        per = {}
        vals = []
        for ms_len in (5, 10, 20, 40):
            fv = _window_mean_vector(ctx, "qrs", length_ms=ms_len)
            if fv.value is not None:
                per[f"{ms_len}ms"] = fv.value
                vals.append(fv.value)
        return FeatureValue(float(vals[-1]), per_lead=per) if vals else FeatureValue(None, 0.0)
    if name == "terminal_5/10/20/40ms_qrs_vector":
        per = {}
        vals = []
        for ms_len in (5, 10, 20, 40):
            fv = _window_mean_vector(ctx, "qrs", length_ms=ms_len, end_aligned=True)
            if fv.value is not None:
                per[f"{ms_len}ms"] = fv.value
                vals.append(fv.value)
        return FeatureValue(float(vals[-1]), per_lead=per) if vals else FeatureValue(None, 0.0)
    if name in ("j_point_vector", "tpeak_to_tend_vector"):
        centered = ctx.origin_corrected_vcg()
        if centered is None:
            return FeatureValue(None, 0.0)
        if name == "j_point_vector":
            idx = ctx.marker_index("j_point")
            if idx is None:
                idx = ctx._qrs_end()
            if idx is None or not (0 <= idx < len(centered)):
                return FeatureValue(None, 0.0)
            vec = centered[idx]
        else:
            a = ctx.marker_index("t_peak")
            b = ctx.marker_index("t_end")
            if a is None or b is None:
                return FeatureValue(None, 0.0)
            vec = centered[b] - centered[a]
        return FeatureValue(float(np.linalg.norm(vec)), per_lead={"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])})
    if name in ("p_area_vector", "p_mean_vector", "terminal_t_vector", "late_qrs_vector_magnitude", "p_vector_magnitude"):
        if name == "p_area_vector":
            res = _vector_for(ctx, "p", "area")
        elif name in ("p_mean_vector", "p_vector_magnitude"):
            res = _vector_for(ctx, "p", "mean")
        elif name == "terminal_t_vector":
            return _window_mean_vector(ctx, "t", length_ms=40, end_aligned=True)
        else:
            return _window_mean_vector(ctx, "qrs", length_ms=40, end_aligned=True)
        return FeatureValue(None, 0.0) if res is None else FeatureValue(float(res[1]["magnitude"]), per_lead=res[1])
    if name == "p_low_voltage_index":
        res = _vector_for(ctx, "p", "mean")
        return FeatureValue(None, 0.0) if res is None else FeatureValue(_safe_div(1.0, res[1]["magnitude"]))
    if name == "pr_drift_magnitude":
        return _segment_drift(ctx, "pr")
    if name == "pr_segment_slope_magnitude":
        drift = _segment_drift(ctx, "pr")
        dur = ctx.duration_s("pr")
        return FeatureValue(None, 0.0) if drift.value is None or not dur else FeatureValue(_safe_div(drift.value, dur))
    if name == "pr_baseline_instability":
        pr = ctx.vcg_segment("pr")
        return FeatureValue(None, 0.0) if pr is None else FeatureValue(float(np.mean(np.std(pr, axis=0))))
    if name == "pr_vector":
        vec = _segment_delta_vector(ctx, "pr")
        return FeatureValue(None, 0.0) if vec is None else FeatureValue(float(np.linalg.norm(vec)), per_lead={"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])})
    if name in ("pr_qrs_angle_deg", "pr_st_angle_deg", "pr_t_angle_deg", "terminal_qrs_angle_relative_to_main_qrs_deg"):
        if name == "terminal_qrs_angle_relative_to_main_qrs_deg":
            a = _window_vector(ctx, "qrs", length_ms=40, end_aligned=True)
            b = _vector_array(ctx, "qrs", "mean")
        else:
            _, seg_b, _ = name.split("_", 2)
            a = _segment_delta_vector(ctx, "pr")
            b = _vector_array(ctx, seg_b, "mean")
        return FeatureValue(_angle_between(a, b) if a is not None and b is not None else None)
    if name == "u_wave_candidate_magnitude":
        bounds = ctx.bounds("t")
        centered = ctx.origin_corrected_vcg()
        if bounds is None or centered is None:
            return FeatureValue(None, 0.0)
        start = bounds[1] + 1
        end = min(len(centered), start + int(round(160 * ctx.fs / 1000.0)))
        if end <= start:
            return FeatureValue(None, 0.0)
        return FeatureValue(float(np.max(np.linalg.norm(centered[start:end], axis=1))))
    if name == "u_wave_detected_flag/confidence":
        mag = _segment_specific_feature(ctx, "u_wave_candidate_magnitude")
        if mag is None or mag.value is None:
            return FeatureValue(0.0, per_lead={"confidence": 0.0})
        conf = float(min(1.0, mag.value / 0.05))
        return FeatureValue(1.0 if conf >= 0.5 else 0.0, per_lead={"confidence": conf})
    if name in ("st_qrs_discordance_angle_deg", "st_qrs_discordance_deg"):
        return FeatureValue(_angle_between(_vector_array(ctx, "st", "mean"), _vector_array(ctx, "qrs", "mean")))
    if name == "st_vector_magnitude":
        vec = _vector_array(ctx, "st", "mean")
        return FeatureValue(None, 0.0) if vec is None else FeatureValue(float(np.linalg.norm(vec)))
    if name == "terminal_qrs_rightward/leftward/posterior/anterior_index":
        vec = _window_vector(ctx, "qrs", 40, True)
        if vec is None:
            return FeatureValue(None, 0.0)
        per = {
            "rightward": float(max(vec[0], 0.0)),
            "leftward": float(max(-vec[0], 0.0)),
            "posterior": float(max(vec[2], 0.0)),
            "anterior": float(max(-vec[2], 0.0)),
        }
        return FeatureValue(float(max(per.values())), per_lead=per)
    if name == "qrs_early/mid/late_mean_vector":
        per = {}
        vals = []
        for segment in ("qrs_early", "qrs_mid", "qrs_late"):
            vec = _vector_array(ctx, segment, "mean")
            if vec is not None:
                per[segment] = float(np.linalg.norm(vec))
                vals.append(per[segment])
        return FeatureValue(float(vals[-1]), per_lead=per) if vals else FeatureValue(None, 0.0)
    if name == "st20/st40/st60/st80_vector":
        centered = ctx.origin_corrected_vcg()
        if centered is None or ctx._st_start() is None:
            return FeatureValue(None, 0.0)
        per = {}
        for ms in (20, 40, 60, 80):
            idx = ctx._st_start() + int(round(ms * ctx.fs / 1000.0))
            if 0 <= idx < len(centered):
                per[f"st{ms}"] = float(np.linalg.norm(centered[idx]))
        return FeatureValue(float(np.mean(list(per.values()))), per_lead=per) if per else FeatureValue(None, 0.0)
    if name in ("st_area_vector", "t_area_vector"):
        # Published GEH T area is the complete JT area, beginning at QRS
        # offset; it must not discard the ST portion before visual T onset.
        segment = "st" if name.startswith("st_") else "st_t"
        res = _vector_for(ctx, segment, "area")
        return (
            FeatureValue(
                None,
                0.0,
                notes=ctx.vcg_origin_assessment.provenance_note(),
            )
            if res is None
            else FeatureValue(
                float(res[1]["magnitude"]),
                per_lead=res[1],
                notes=ctx.vcg_origin_assessment.provenance_note(),
            )
        )
    if name == "st_loop_drift":
        return _segment_drift(ctx, "st")
    if name == "st_slope_vector":
        drift = _segment_drift(ctx, "st")
        dur = ctx.duration_s("st")
        return FeatureValue(None, 0.0) if drift.value is None or not dur else FeatureValue(_safe_div(drift.value, dur))
    if name == "st_t_continuity_index":
        return FeatureValue(_angle_between(_window_vector(ctx, "st", 20, True), _window_vector(ctx, "t", 20)))
    if name in ("t_upstroke_vector", "t_downstroke_vector"):
        segment = "t_upstroke" if name.startswith("t_upstroke") else "t_downstroke"
        vec = _vector_array(ctx, segment, "mean")
        return FeatureValue(None, 0.0) if vec is None else FeatureValue(float(np.linalg.norm(vec)), per_lead={"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])})
    if name == "t_upstroke_downstroke_angle_deg":
        return FeatureValue(_angle_between(_vector_array(ctx, "t_upstroke", "mean"), _vector_array(ctx, "t_downstroke", "mean")))
    return None


def _segment_drift(ctx: FeatureContext, segment: str) -> FeatureValue:
    data = ctx.vcg_segment(segment)
    if data is None or len(data) < 2:
        return FeatureValue(None, 0.0)
    return FeatureValue(float(np.linalg.norm(data[-1] - data[0])))


def _segment_delta_vector(ctx: FeatureContext, segment: str) -> Optional[np.ndarray]:
    data = ctx.vcg_segment(segment)
    if data is None or len(data) < 2:
        return None
    return data[-1] - data[0]


def _vector_array(ctx: FeatureContext, segment: str, mode: str) -> Optional[np.ndarray]:
    res = _vector_for(ctx, segment, mode)
    return None if res is None else res[0]


def _window_vector(ctx: FeatureContext, segment: str, length_ms: int = 20, end_aligned: bool = False) -> Optional[np.ndarray]:
    bounds = ctx.bounds(segment)
    centered = ctx.origin_corrected_vcg()
    if bounds is None or centered is None:
        return None
    samples = max(1, int(round(length_ms * ctx.fs / 1000.0)))
    start, end = bounds
    if end_aligned:
        lo, hi = max(start, end - samples + 1), end + 1
    else:
        lo, hi = start, min(end + 1, start + samples)
    return np.mean(centered[lo:hi], axis=0) if hi > lo else None


def _angle_relation_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "early_late_qrs_angle_deg":
        return FeatureValue(_angle_between(_window_vector(ctx, "qrs", 40), _window_vector(ctx, "qrs", 40, True)))
    if name == "early_late_t_angle_deg":
        return FeatureValue(_angle_between(_window_vector(ctx, "t", 40), _window_vector(ctx, "t", 40, True)))
    if name == "initial_terminal_qrs_angle_deg":
        return FeatureValue(_angle_between(_window_vector(ctx, "qrs", 20), _window_vector(ctx, "qrs", 20, True)))
    if name in ("frontal_xy_qrs_t_angle_deg", "horizontal_xz_qrs_t_angle_deg", "sagittal_yz_qrs_t_angle_deg"):
        plane = name.split("_qrs_t_angle")[0]
        q = _vector_array(ctx, "qrs", "mean")
        t = _vector_array(ctx, "t", "mean")
        if q is None or t is None or plane not in PLANES:
            return FeatureValue(None, 0.0)
        idx = PLANES[plane]
        return FeatureValue(_angle_between(q[list(idx)], t[list(idx)]))
    if name == "j_vector_to_qrs_axis_angle_deg":
        j = _segment_specific_feature(ctx, "j_point_vector")
        q = _vector_array(ctx, "qrs", "mean")
        if j is None or j.per_lead is None or q is None:
            return FeatureValue(None, 0.0)
        return FeatureValue(_angle_between(np.array([j.per_lead["x"], j.per_lead["y"], j.per_lead["z"]]), q))
    if name == "max_qrs_to_max_t_angle_deg":
        return FeatureValue(_angle_between(_vector_array(ctx, "qrs", "max"), _vector_array(ctx, "t", "max")))
    if name in ("p_qrs_angle_deg", "p_t_angle_deg", "qrs_axis_to_t_axis_angle_deg", "qrs_st_angle_deg", "st_t_angle_deg"):
        parts = name.replace("_angle_deg", "").split("_")
        if parts[0] == "qrs" and parts[1] == "axis":
            a, b = "qrs", "t"
        else:
            a, b = parts[0], parts[1]
        return FeatureValue(_angle_between(_vector_array(ctx, a, "mean"), _vector_array(ctx, b, "mean")))
    if name in ("p_qrs_loop_plane_angle_deg", "p_t_loop_plane_angle_deg", "qrs_t_loop_plane_angle_deg", "qrs_area_to_t_area_angle_deg_axis_relation"):
        if name == "qrs_area_to_t_area_angle_deg_axis_relation":
            return _qrst_feature(ctx, "qrs_area_to_t_area_angle_deg")
        parts = name.replace("_loop_plane_angle_deg", "").split("_")
        a, b = parts[0], parts[1]
        return FeatureValue(_plane_angle(ctx, a, b))
    if name == "spatial_qrs_t_angle_deg_catalog":
        return _qrst_feature(ctx, "spatial_qrs_t_angle_deg")
    if name == "st_vector_to_tpeak_angle_deg":
        st = _vector_array(ctx, "st", "mean")
        tpeak = None
        idx = ctx.marker_index("t_peak")
        if idx is not None and ctx.vcg_xyz is not None and 0 <= idx < len(ctx.vcg_xyz):
            tpeak = ctx.vcg_xyz[idx]
        return FeatureValue(_angle_between(st, tpeak))
    return None


def _plane_angle(ctx: FeatureContext, a: str, b: str) -> Optional[float]:
    fa = _loop3d_features(ctx, a)
    fb = _loop3d_features(ctx, b)
    if not fa or not fb:
        return None
    va = np.array([fa["plane_normal_x"], fa["plane_normal_y"], fa["plane_normal_z"]])
    vb = np.array([fb["plane_normal_x"], fb["plane_normal_y"], fb["plane_normal_z"]])
    angle = _angle_between(va, vb)
    return None if angle is None else min(angle, 180.0 - angle)


def _st_timepoint_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if not name.startswith("st") or "deviation" not in name and "elevation" not in name:
        return None
    if ctx.ecg_12lead is None:
        return FeatureValue(None, 0.0)
    start = ctx._st_start()
    if start is None:
        return FeatureValue(None, 0.0)
    try:
        ms = int(name[2:4])
    except Exception:
        return None
    idx = start + int(round(ms * ctx.fs / 1000.0))
    if idx < 0 or idx >= len(ctx.ecg_12lead):
        return FeatureValue(None, 0.0)
    baseline = np.zeros(ctx.ecg_12lead.shape[1])
    pr = ctx.bounds("pr")
    if pr is not None:
        baseline = np.mean(ctx.ecg_12lead[pr[0]:pr[1] + 1], axis=0)
    values = (ctx.ecg_12lead[idx] - baseline) * 1000.0
    if "deviation_mean_abs" in name:
        return FeatureValue(float(np.mean(np.abs(values))))
    if "elevation_max_lead" in name:
        return FeatureValue(float(np.max(values)), per_lead={"depression_min": float(np.min(values))})
    return None


def _pathophysiology_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    qrs_f = _loop3d_features(ctx, "qrs")
    t_f = _loop3d_features(ctx, "t")
    if name in ("activation_inefficiency", "conduction_detour_index"):
        if not qrs_f:
            return FeatureValue(None, 0.0)
        displacement = qrs_f["start_end_distance"]
        if name == "activation_inefficiency":
            return FeatureValue(_safe_div(qrs_f["path_length"], max(qrs_f["diameter"], displacement)))
        minimum_out_and_back = 2.0 * qrs_f["diameter"]
        closed_path = qrs_f["path_length"] + displacement
        return FeatureValue(
            float(
                max(
                    0.0,
                    _safe_div(
                        closed_path - minimum_out_and_back,
                        minimum_out_and_back,
                    ),
                )
            ),
            notes=(
                "Closed QRS path excess relative to the minimum out-and-back "
                "path spanning the loop diameter."
            ),
        )
    if name == "depolarization_repolarization_mismatch":
        return _qrst_feature(ctx, "spatial_qrs_t_angle_deg")
    if name == "initial_qrs_loss_index":
        early = _window_mean_vector(ctx, "qrs", length_ms=20)
        qrs = ctx.vcg_segment("qrs")
        if early.value is None or qrs is None:
            return FeatureValue(None, 0.0)
        peak = float(np.max(np.linalg.norm(qrs, axis=1)))
        return FeatureValue(float(1.0 - _safe_div(early.value, peak)))
    if name == "late_qrs_vector_magnitude":
        return _segment_specific_feature(ctx, "late_qrs_vector_magnitude")
    if name in ("lbbb_like_vector_score", "rbbb_like_terminal_vector_score", "lvh_vector_score / rvh_vector_score", "acute_ischemia_vector_score", "scar_vector_index", "qt_prolongation/shortening_surrogate"):
        qrs_duration = _interval_feature(ctx, "qrs_duration_ms")
        qrs_dur = (
            qrs_duration.value
            if qrs_duration is not None
            else None
        )
        qrs = ctx.vcg_segment("qrs")
        peak = (
            float(np.max(np.linalg.norm(qrs, axis=1)))
            if qrs is not None
            else None
        )
        if name == "lbbb_like_vector_score":
            late_result = _segment_specific_feature(
                ctx,
                "late_qrs_vector_magnitude",
            )
            late = late_result.value if late_result is not None else None
            if qrs_dur is None or peak is None or late is None:
                return FeatureValue(
                    None,
                    0.0,
                    notes="Complete QRS duration and terminal vector required",
                )
            return FeatureValue(float(min(1.0, qrs_dur / 160.0 + _safe_div(late, peak)) / 2.0))
        if name == "rbbb_like_terminal_vector_score":
            term = _window_vector(ctx, "qrs", 40, True)
            if qrs_dur is None or peak is None or term is None:
                return FeatureValue(
                    None,
                    0.0,
                    notes="Complete QRS duration and terminal vector required",
                )
            ap = abs(float(term[2]))
            return FeatureValue(float(min(1.0, qrs_dur / 140.0 + _safe_div(ap, peak)) / 2.0))
        if name == "lvh_vector_score / rvh_vector_score":
            if qrs is None or peak is None:
                return FeatureValue(
                    None,
                    0.0,
                    notes="Complete QRS segment required",
                )
            mag = peak
            xcomp = float(np.max(np.abs(qrs[:, 0])))
            return FeatureValue(float(min(1.0, mag / 3.5)), per_lead={"lvh": float(min(1.0, mag / 3.5)), "rvh": float(min(1.0, xcomp / 2.0))})
        if name == "acute_ischemia_vector_score":
            st = ctx.vcg_segment("st")
            angle_result = _angle_relation_feature(ctx, "st_t_angle_deg")
            angle = (
                angle_result.value
                if angle_result is not None
                else None
            )
            if st is None or peak is None or angle is None:
                return FeatureValue(
                    None,
                    0.0,
                    notes="Complete QRS, ST and T segments required",
                )
            st_mag = float(np.mean(np.linalg.norm(st, axis=1)))
            return FeatureValue(float(min(1.0, _safe_div(st_mag, peak) + angle / 180.0) / 2.0))
        if name == "scar_vector_index":
            loss_result = _pathophysiology_feature(
                ctx,
                "initial_qrs_loss_index",
            )
            notch_result = _pathophysiology_feature(
                ctx,
                "qrs_loop_notch_burden",
            )
            loss = loss_result.value if loss_result is not None else None
            notch = notch_result.value if notch_result is not None else None
            if loss is None or notch is None or not qrs_f:
                return FeatureValue(
                    None,
                    0.0,
                    notes="Complete QRS morphology required",
                )
            deform = qrs_f["nonplanarity"]
            return FeatureValue(float(np.mean([loss, min(1.0, notch / 5.0), deform])))
        qt_result = _interval_feature(ctx, "qt_interval_ms")
        qt = qt_result.value if qt_result is not None else None
        return FeatureValue(None, 0.0) if qt is None else FeatureValue(float(max(qt / 450.0, 320.0 / max(qt, EPS))))
    if name == "qrs_loop_notch_burden":
        kin = _kinematic_features(ctx, "qrs")
        return FeatureValue(None, 0.0) if not kin else FeatureValue(kin["velocity_peak_count"])
    return None


def _loop_extra_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name in ("qrs_t_peak_magnitude_ratio", "p_loop_to_qrs_loop_area_ratio", "p_qrs_loop_plane_angle", "p_t_loop_plane_angle", "p_qrs_t_plane_triad_volume", "p_qrs_t_axis_consistency", "qrs_t_loop_shape_distance", "t_loop_to_qrs_loop_complexity_ratio"):
        if name == "qrs_t_peak_magnitude_ratio":
            qrs = ctx.vcg_segment("qrs")
            t = ctx.vcg_segment("t")
            return FeatureValue(None, 0.0) if qrs is None or t is None else FeatureValue(_safe_div(float(np.max(np.linalg.norm(qrs, axis=1))), float(np.max(np.linalg.norm(t, axis=1)))))
        pf, qf, tf = _loop3d_features(ctx, "p"), _loop3d_features(ctx, "qrs"), _loop3d_features(ctx, "t")
        if name == "p_loop_to_qrs_loop_area_ratio":
            return FeatureValue(None, 0.0) if not pf or not qf else FeatureValue(_safe_div(pf["area_best_fit_plane"], qf["area_best_fit_plane"]))
        if name == "p_qrs_loop_plane_angle":
            return FeatureValue(_plane_angle(ctx, "p", "qrs"))
        if name == "p_t_loop_plane_angle":
            return FeatureValue(_plane_angle(ctx, "p", "t"))
        if name == "p_qrs_t_plane_triad_volume":
            if not pf or not qf or not tf:
                return FeatureValue(None, 0.0)
            normals = np.array([[pf["plane_normal_x"], pf["plane_normal_y"], pf["plane_normal_z"]], [qf["plane_normal_x"], qf["plane_normal_y"], qf["plane_normal_z"]], [tf["plane_normal_x"], tf["plane_normal_y"], tf["plane_normal_z"]]])
            return FeatureValue(float(abs(np.linalg.det(normals))))
        if name == "p_qrs_t_axis_consistency":
            vectors = [_vector_array(ctx, s, "mean") for s in ("p", "qrs", "t")]
            vectors = [v / (np.linalg.norm(v) + EPS) for v in vectors if v is not None]
            return FeatureValue(None, 0.0) if len(vectors) < 2 else FeatureValue(float(1.0 - np.linalg.norm(np.mean(vectors, axis=0))))
        if name == "qrs_t_loop_shape_distance":
            return _qrst_feature(ctx, "qrs_t_procrustes_distance")
        if name == "t_loop_to_qrs_loop_complexity_ratio":
            qk, tk = _kinematic_features(ctx, "qrs"), _kinematic_features(ctx, "t")
            return FeatureValue(None, 0.0) if not qk or not tk else FeatureValue(_safe_div(tk["curvature_integral"], qk["curvature_integral"]))
    if name.startswith("p_loop_"):
        return _p_loop_extra(ctx, name)
    if name.startswith("t_loop_"):
        return _t_loop_extra(ctx, name)
    return None


def _p_loop_extra(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    p = ctx.vcg_segment("p")
    if p is None:
        return FeatureValue(None, 0.0)
    if name == "p_loop_anterior_posterior_balance":
        return FeatureValue(float(abs(np.mean(p[:, 2] >= 0) - np.mean(p[:, 2] < 0))))
    if name == "p_loop_left_right_balance":
        return FeatureValue(float(abs(np.mean(p[:, 0] >= 0) - np.mean(p[:, 0] < 0))))
    if name == "p_loop_biphasic_index":
        baseline = 0.5 * (p[0] + p[-1])
        vectors = p - baseline
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
        return FeatureValue(
            float(2.0 * min(positive_mass, negative_mass) / (total_mass + EPS))
        )
    if name == "p_loop_double_rotation_index":
        return FeatureValue(_loop2d_features(ctx, "p", "frontal_xy")["rotation_change_count"])
    if name == "p_loop_fractionated_motion_index":
        kin = _kinematic_features(ctx, "p")
        dur = ctx.duration_s("p")
        return FeatureValue(None, 0.0) if not kin or not dur else FeatureValue(_safe_div(kin["velocity_peak_count"], dur))
    if name == "p_loop_plane_instability":
        return _math_loop_feature_for_segment(ctx, "loop_sliding_plane_angle_std", "p")
    if name == "p_loop_interbeat_axis_variability":
        return _beat_variability_feature(
            ctx,
            "<seg>_axis_beat_to_beat_variability_deg",
            segment_override="p",
        )
    if name == "p_loop_template_instability":
        curves = _beat_segment_curves(ctx, "p")
        if len(curves) < 2:
            return FeatureValue(None, 0.0)
        median = np.median(np.stack(curves), axis=0)
        vals = [float(np.sqrt(np.mean((c - median) ** 2))) for c in curves]
        return FeatureValue(float(np.median(vals)))
    if name == "p_loop_terminal_force_vector":
        return _window_mean_vector(ctx, "p", length_ms=20, end_aligned=True)
    if name == "p_loop_terminal_to_initial_angle":
        return FeatureValue(_angle_between(_window_vector(ctx, "p", 20), _window_vector(ctx, "p", 20, True)))
    return None


def _t_loop_extra(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    t = ctx.vcg_segment("t")
    if t is None:
        return FeatureValue(None, 0.0)
    if name == "t_loop_apex_sharpness":
        bounds = ctx.bounds("t")
        peak = ctx.marker_index("t_peak")
        if bounds is None or peak is None:
            return FeatureValue(None, 0.0)
        local_peak = int(peak - bounds[0])
        radius = max(1, int(round(8.0 * ctx.fs / 1000.0)))
        first = max(0, local_peak - radius)
        third = min(len(t) - 1, local_peak + radius)
        if not (first < local_peak < third):
            return FeatureValue(None, 0.0)
        a = t[first]
        b = t[local_peak]
        c = t[third]
        ab = float(np.linalg.norm(a - b))
        bc = float(np.linalg.norm(b - c))
        ac = float(np.linalg.norm(a - c))
        denominator = ab * bc * ac
        if denominator < EPS:
            return FeatureValue(None, 0.0)
        curvature = float(2.0 * np.linalg.norm(np.cross(b - a, c - a)) / denominator)
        return FeatureValue(curvature)
    if name == "t_loop_bulging_index":
        line = t[-1] - t[0]
        denom = np.linalg.norm(line)
        if denom < EPS:
            return FeatureValue(0.0)
        dist = np.linalg.norm(np.cross(t - t[0], line), axis=1) / denom
        return FeatureValue(float(np.max(dist)))
    if name == "t_loop_late_area_fraction":
        bounds = ctx.bounds("t")
        if bounds is None:
            return FeatureValue(None, 0.0)
        start, end = bounds
        split = int(round(start + 2.0 * (end - start) / 3.0))
        centered, _, vecs = _pca(t)
        full_scores = centered @ vecs
        projected = full_scores[:, :2]
        if len(projected) < 3:
            return FeatureValue(None, 0.0)
        # A shoelace area of an open late sub-path can exceed the signed area
        # of a self-intersecting full loop through cancellation (A017 reached
        # 125).  Partition the absolute areal sweep about the same centroid
        # instead: every temporal edge contributes once, so the terminal third
        # is mathematically guaranteed to be a subset of the total.
        edge_sweep = 0.5 * np.abs(
            projected[:-1, 0] * projected[1:, 1]
            - projected[1:, 0] * projected[:-1, 1]
        )
        total_sweep = float(np.sum(edge_sweep))
        if not np.isfinite(total_sweep) or total_sweep <= EPS:
            return FeatureValue(
                None,
                0.0,
                notes="not_evaluable: T loop has no areal sweep in its own plane",
            )
        local_split = max(0, min(len(projected) - 1, split - start))
        late_sweep = float(np.sum(edge_sweep[local_split:]))
        return FeatureValue(float(np.clip(late_sweep / total_sweep, 0.0, 1.0)))
    if name == "t_loop_qrs_plane_escape_angle":
        return FeatureValue(_plane_angle(ctx, "t", "qrs"))
    if name == "t_loop_repolarization_tail_index":
        if len(t) < 4:
            return FeatureValue(None, 0.0)
        split = max(1, int(np.floor(2.0 * len(t) / 3.0)))
        total_path = _path_length(t)
        terminal_path = _path_length(t[split:])
        return FeatureValue(
            _safe_div(terminal_path, total_path),
            notes="Path in the final temporal third divided by total T-loop path.",
        )
    if name == "t_loop_roundness_vs_planarity_ratio":
        tf = _loop3d_features(ctx, "t")
        return FeatureValue(None, 0.0) if not tf else FeatureValue(_safe_div(tf["roundness_best_fit_plane"], tf["planarity"]))
    if name == "t_loop_terminal_dispersion":
        last = t[int(len(t) * 2 / 3):]
        return FeatureValue(float(np.mean(np.linalg.norm(last - np.mean(last, axis=0), axis=1))))
    if name == "t_loop_upstroke_downstroke_angle":
        return FeatureValue(_angle_between(_window_vector(ctx, "t", 40), _window_vector(ctx, "t", 40, True)))
    if name == "t_loop_upstroke_downstroke_area_ratio":
        peak = ctx.marker_index("t_peak")
        bounds = ctx.bounds("t")
        if peak is None or bounds is None:
            return FeatureValue(None, 0.0)
        start, end = bounds
        centered, _, vecs = _pca(t)
        up = (ctx.vcg_xyz[start:peak + 1] - np.mean(t, axis=0)) @ vecs
        down = (ctx.vcg_xyz[peak:end + 1] - np.mean(t, axis=0)) @ vecs
        return FeatureValue(_safe_div(abs(_shoelace(up[:, :2])), abs(_shoelace(down[:, :2]))))
    if name == "t_loop_vector_reversal_index":
        return _backtracking_fraction(t)
    return None


def _backtracking_fraction(points: np.ndarray) -> FeatureValue:
    if points is None or len(points) < 3:
        return FeatureValue(None, 0.0)
    tangent = np.diff(points, axis=0)
    mean_dir = np.mean(tangent, axis=0)
    if np.linalg.norm(mean_dir) < EPS:
        mean_dir = points[-1] - points[0]
    if np.linalg.norm(mean_dir) < EPS:
        return FeatureValue(0.0)
    dots = tangent @ (mean_dir / np.linalg.norm(mean_dir))
    return FeatureValue(float(np.mean(dots < 0)))


def _math_loop_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    try:
        family = BiomarkerRegistry.definition(name).family
    except ValueError:
        return None
    if not family.startswith("Mathematische Loop-Morphologie"):
        return None
    return _math_loop_feature_for_segment(ctx, name, "qrs")


def _nonlocal_distance_bundle(
    ctx: FeatureContext,
    segment: str,
    exclusion: int = 5,
) -> np.ndarray:
    return _memoized_primitive(
        ctx,
        ("nonlocal_distances", segment, int(exclusion)),
        lambda: _nonlocal_distances(ctx.vcg_segment(segment), exclusion=exclusion),
    )


def _fourier_loop_bundle(
    ctx: FeatureContext,
    segment: str,
) -> Optional[dict[str, object]]:
    def build() -> Optional[dict[str, object]]:
        points = ctx.vcg_segment(segment)
        resampled = _resample_curve(points, 128)
        if resampled is None:
            return None
        centered = resampled - np.mean(resampled, axis=0)
        coefficients = np.fft.rfft(centered, axis=0)
        energy = np.sum(np.abs(coefficients) ** 2, axis=1)
        return {
            "centered": centered,
            "coefficients": coefficients,
            "energy": energy,
            "total_non_dc_energy": float(np.sum(energy[1:])),
        }

    return _memoized_primitive(ctx, ("fourier_loop", segment, 128), build)


def _angular_loop_bundle(
    ctx: FeatureContext,
    segment: str,
) -> Optional[dict[str, object]]:
    def build() -> Optional[dict[str, object]]:
        pca = _segment_pca_bundle(ctx, segment)
        if pca is None:
            return None
        points = pca["points"]
        scores = pca["scores"]
        angles = np.mod(np.arctan2(scores[:, 1], scores[:, 0]), 2.0 * np.pi)
        angles_sorted = np.sort(angles)
        gaps = np.diff(
            np.concatenate([angles_sorted, [angles_sorted[0] + 2.0 * np.pi]])
        )
        projected = scores[:, :2]
        return {
            "angles": angles,
            "gaps": gaps,
            "radial": np.linalg.norm(projected, axis=1),
            "hull_area": _convex_area_2d(projected),
            "signed_area": _shoelace(projected),
            "hull_depths": _hull_depths(projected),
            "convex_volume": _convex_volume_3d(points),
        }

    return _memoized_primitive(ctx, ("angular_loop", segment), build)


def _rotation_loop_bundle(
    ctx: FeatureContext,
    segment: str,
) -> dict[str, np.ndarray]:
    def build() -> dict[str, np.ndarray]:
        points = ctx.vcg_segment(segment)
        return {
            "local_angles": _local_rotation_angles(points),
            "dihedral_angles": _dihedral_angles(points),
        }

    return _memoized_primitive(ctx, ("rotation_loop", segment), build)


def _sliding_plane_bundle(
    ctx: FeatureContext,
    segment: str,
    window: int = 9,
) -> Optional[dict[str, object]]:
    def build() -> Optional[dict[str, object]]:
        pca = _segment_pca_bundle(ctx, segment)
        loop = _loop3d_features(ctx, segment)
        if pca is None or loop is None:
            return None
        points = pca["points"]
        centered = pca["centered"]
        scores = pca["scores"]
        global_normal = np.array(
            [
                loop["plane_normal_x"],
                loop["plane_normal_y"],
                loop["plane_normal_z"],
            ],
            dtype=float,
        )
        normals = _sliding_normals(points, window=window)
        if len(normals):
            # Local PCA normals also have an arbitrary sign. Orient them to
            # the global loop normal before computing temporal twist.
            normals = normals.copy()
            flip = (normals @ global_normal) < 0.0
            normals[flip] *= -1.0
            cosine = np.clip(normals @ global_normal, -1.0, 1.0)
            global_angles = np.degrees(np.arccos(cosine))
            if len(normals) > 1:
                adjacent_cosine = np.clip(
                    np.sum(normals[:-1] * normals[1:], axis=1), -1.0, 1.0
                )
                adjacent_angles = np.degrees(np.arccos(adjacent_cosine))
            else:
                adjacent_angles = np.array([], dtype=float)
        else:
            global_angles = np.array([], dtype=float)
            adjacent_angles = np.array([], dtype=float)
        return {
            "normals": normals,
            "global_angles": global_angles,
            "adjacent_angles": adjacent_angles,
            "plane_residual": np.abs(scores[:, 2]),
            "signed_distance": scores[:, 2],
        }

    return _memoized_primitive(
        ctx,
        ("sliding_plane", segment, int(window)),
        build,
    )


def _graph_loop_bundle(
    ctx: FeatureContext,
    segment: str,
) -> dict[str, object]:
    def build() -> dict[str, object]:
        points = ctx.vcg_segment(segment)
        path = _path_length(points)
        direct = float(np.linalg.norm(points[-1] - points[0]))
        pairwise = (
            distance.squareform(distance.pdist(points))
            if len(points) > 1
            else np.zeros((len(points), len(points)))
        )
        nonlocal_values = _nonlocal_distance_bundle(ctx, segment)
        radius = (
            float(np.percentile(nonlocal_values, 10))
            if len(nonlocal_values)
            else 0.0
        )
        adjacency = (pairwise < radius) & (pairwise > 0)
        return {
            "path": path,
            "direct": direct,
            "pairwise": pairwise,
            "nonlocal": nonlocal_values,
            "radius": radius,
            "adjacency": adjacency,
            "degree": np.sum(adjacency, axis=1),
        }

    return _memoized_primitive(ctx, ("graph_loop", segment), build)


def _multiscale_loop_bundle(
    ctx: FeatureContext,
    segment: str,
) -> Optional[dict[str, np.ndarray]]:
    def build() -> Optional[dict[str, np.ndarray]]:
        points = ctx.vcg_segment(segment)
        if points is None:
            return None
        areas: list[float] = []
        peaks: list[float] = []
        intersections: list[float] = []
        for step in (1, 2, 4):
            sampled = points[::step]
            if len(sampled) < 4:
                continue
            centered, _, vectors = _pca(sampled)
            scores = centered @ vectors
            areas.append(abs(_shoelace(scores[:, :2])))
            radius = np.linalg.norm(scores[:, :2], axis=1)
            peak_indices = sp_signal.find_peaks(radius)[0]
            peaks.append(
                float(np.mean(sp_signal.peak_prominences(radius, peak_indices)[0]))
                if len(peak_indices)
                else 0.0
            )
            intersections.append(float(sum(_loop2d_self_counts(sampled).values())))
        if not areas:
            return None
        return {
            "areas": np.asarray(areas, dtype=float),
            "peaks": np.asarray(peaks, dtype=float),
            "intersections": np.asarray(intersections, dtype=float),
        }

    return _memoized_primitive(ctx, ("multiscale_loop", segment), build)


def _math_loop_feature_for_segment(ctx: FeatureContext, name: str, segment: str) -> Optional[FeatureValue]:
    points = ctx.vcg_segment(segment)
    if points is None or len(points) < 4:
        return FeatureValue(None, 0.0)
    kin = _kinematic_features(ctx, segment)
    loop = _loop3d_features(ctx, segment)
    if name in ("loop_bending_energy", "loop_curvature_entropy", "loop_curvature_peak_count", "loop_curvature_torsion_ratio", "loop_mean_curvature", "loop_torsion_sign_changes", "loop_total_curvature", "loop_total_torsion"):
        differential = _differential_bundle(ctx, segment)
        if not kin or differential is None:
            return FeatureValue(None, 0.0)
        curvature = differential["curvature"]
        torsion = differential["torsion"]
        ds = differential["ds"]
        if name == "loop_bending_energy":
            return FeatureValue(float(np.sum(curvature ** 2 * ds)))
        if name == "loop_curvature_entropy":
            return FeatureValue(_entropy(curvature))
        if name == "loop_curvature_peak_count":
            return FeatureValue(float(len(sp_signal.find_peaks(curvature)[0])))
        if name == "loop_curvature_torsion_ratio":
            return FeatureValue(_safe_div(float(np.sum(np.abs(curvature) * ds)), float(np.sum(np.abs(torsion) * ds))))
        if name == "loop_mean_curvature":
            return FeatureValue(float(np.mean(curvature)))
        if name == "loop_torsion_sign_changes":
            signs = np.sign(torsion[np.abs(torsion) > EPS])
            return FeatureValue(float(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0.0)
        if name == "loop_total_curvature":
            return FeatureValue(float(np.sum(np.abs(curvature) * ds)))
        if name == "loop_total_torsion":
            return FeatureValue(float(np.sum(np.abs(torsion) * ds)))
    if name.startswith("loop_ellipse") or name in ("loop_actual_to_ellipse_area_ratio", "loop_best_fit_ellipse_error", "loop_superellipse_exponent"):
        if not loop:
            return FeatureValue(None, 0.0)
        major = loop["width_major"] / 2.0
        minor = loop["width_minor"] / 2.0
        ellipse_area = float(np.pi * major * minor)
        if name == "loop_ellipse_area":
            return FeatureValue(ellipse_area)
        if name == "loop_ellipse_axis_ratio":
            return FeatureValue(_safe_div(major, minor))
        if name == "loop_ellipse_major_axis":
            return FeatureValue(float(2.0 * major))
        if name == "loop_ellipse_minor_axis":
            return FeatureValue(float(2.0 * minor))
        if name == "loop_ellipse_orientation_deg":
            pca = _segment_pca_bundle(ctx, segment)
            if pca is None:
                return FeatureValue(None, 0.0)
            vecs = pca["vectors"]
            return FeatureValue(float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))))
        if name == "loop_actual_to_ellipse_area_ratio":
            return FeatureValue(_safe_div(loop["area_best_fit_plane"], ellipse_area))
        if name == "loop_best_fit_ellipse_error":
            pca = _segment_pca_bundle(ctx, segment)
            if pca is None:
                return FeatureValue(None, 0.0)
            score = pca["scores"]
            r = np.sqrt((score[:, 0] / (major + EPS)) ** 2 + (score[:, 1] / (minor + EPS)) ** 2)
            return FeatureValue(float(np.sqrt(np.mean((r - 1.0) ** 2))))
        if name == "loop_superellipse_exponent":
            exponent, fit_error = _fit_superellipse_exponent(points)
            if exponent is None:
                return FeatureValue(None, 0.0)
            confidence = float(
                np.clip(1.0 / (1.0 + 3.0 * fit_error), 0.15, 0.9)
            )
            return FeatureValue(
                exponent,
                confidence=confidence,
                notes=(
                    "Robust bounded implicit superellipse fit in the QRS "
                    "PCA plane; n=2 is elliptical, n<2 diamond-like and "
                    "n>2 increasingly rectangular."
                ),
            )
    if name in ("loop_backtracking_fraction", "trajectory_reversal_fraction"):
        return _backtracking_fraction(points)
    if name.startswith("loop_self_intersection") or name in ("loop_total_self_intersections", "loop_fold_density", "loop_near_self_approach_count", "loop_min_nonlocal_distance"):
        counts = {plane: _loop2d_features(ctx, segment, plane)["self_intersection_count"] for plane in PLANES}
        if name == "loop_self_intersection_xy":
            return FeatureValue(counts["frontal_xy"])
        if name == "loop_self_intersection_xz":
            return FeatureValue(counts["horizontal_xz"])
        if name == "loop_self_intersection_yz":
            return FeatureValue(counts["sagittal_yz"])
        if name == "loop_total_self_intersections":
            return FeatureValue(float(sum(counts.values())))
        if name == "loop_fold_density":
            return FeatureValue(_safe_div(float(sum(counts.values())), _path_length(points)))
        nonlocal_dist = _nonlocal_distance_bundle(ctx, segment)
        if name == "loop_min_nonlocal_distance":
            return FeatureValue(float(np.min(nonlocal_dist)) if len(nonlocal_dist) else None)
        threshold = np.percentile(nonlocal_dist, 5) if len(nonlocal_dist) else 0.0
        return FeatureValue(float(np.sum(nonlocal_dist <= threshold)) if len(nonlocal_dist) else 0.0)
    if name.startswith("loop_fourier_"):
        return _fourier_loop_feature(ctx, segment, name)
    if name in ("loop_angular_coverage_deg", "loop_concavity_count", "loop_concavity_depth_max", "loop_convexity_ratio", "loop_hull_fill_ratio_3d", "loop_missing_sector_angle", "loop_radial_uniformity", "loop_sector_entropy", "loop_hole_radius_proxy"):
        angular = _angular_loop_bundle(ctx, segment)
        if angular is None:
            return FeatureValue(None, 0.0)
        angles = angular["angles"]
        gaps = angular["gaps"]
        radial = angular["radial"]
        if name == "loop_angular_coverage_deg":
            return FeatureValue(float(360.0 - np.degrees(np.max(gaps))))
        if name == "loop_missing_sector_angle":
            return FeatureValue(float(np.degrees(np.max(gaps))))
        if name == "loop_sector_entropy":
            return FeatureValue(_entropy(angles))
        if name == "loop_radial_uniformity":
            return FeatureValue(_safe_div(float(np.std(radial)), float(np.mean(radial))))
        if name == "loop_hole_radius_proxy":
            return FeatureValue(float(np.median(radial) - np.percentile(radial, 10)))
        hull = angular["hull_area"]
        area = abs(float(angular["signed_area"]))
        if name == "loop_convexity_ratio":
            return FeatureValue(_safe_div(area, hull or 0.0))
        if name == "loop_hull_fill_ratio_3d":
            vol = angular["convex_volume"]
            return FeatureValue(None, 0.0) if not loop or not vol or not np.isfinite(loop["bounding_ellipsoid_volume_approx"]) else FeatureValue(_safe_div(loop["bounding_ellipsoid_volume_approx"], vol))
        depths = angular["hull_depths"]
        if name == "loop_concavity_count":
            return FeatureValue(float(np.sum(depths > np.percentile(depths, 90))) if len(depths) else 0.0)
        return FeatureValue(float(np.max(depths)) if len(depths) else 0.0)
    if name.startswith("loop_dihedral") or name in ("loop_plane_flip_count", "loop_rotation_consistency", "loop_rotational_angle_mean", "loop_rotational_angle_rmssd", "loop_rotational_angle_std"):
        rotation = _rotation_loop_bundle(ctx, segment)
        angles = rotation["local_angles"]
        dihedral = rotation["dihedral_angles"]
        if name == "loop_dihedral_angle_entropy":
            return FeatureValue(_entropy(dihedral))
        if name == "loop_dihedral_angle_mean":
            return FeatureValue(float(np.mean(dihedral)) if len(dihedral) else 0.0)
        if name == "loop_dihedral_angle_std":
            return FeatureValue(float(np.std(dihedral)) if len(dihedral) else 0.0)
        if name == "loop_plane_flip_count":
            return FeatureValue(float(np.sum(dihedral > 60.0)))
        if name == "loop_rotation_consistency":
            return FeatureValue(float(1.0 - _safe_div(float(np.std(angles)), float(np.mean(np.abs(angles)))))) if len(angles) else FeatureValue(0.0)
        if name == "loop_rotational_angle_mean":
            return FeatureValue(float(np.mean(np.abs(angles))) if len(angles) else 0.0)
        if name == "loop_rotational_angle_rmssd":
            return FeatureValue(float(np.sqrt(np.mean(np.diff(angles) ** 2))) if len(angles) > 1 else 0.0)
        if name == "loop_rotational_angle_std":
            return FeatureValue(float(np.std(angles)) if len(angles) else 0.0)
    if name.startswith("loop_sliding") or name in ("loop_best_plane_residual_p95", "loop_plane_drift_start_end", "loop_plane_outlier_fraction", "loop_plane_twist_integral", "loop_signed_distance_to_plane_skewness", "local_plane_drift_index"):
        sliding = _sliding_plane_bundle(ctx, segment)
        if sliding is None:
            return FeatureValue(None, 0.0)
        normals = sliding["normals"]
        angles = sliding["global_angles"]
        if name == "loop_best_plane_residual_p95":
            residual = sliding["plane_residual"]
            return FeatureValue(float(np.percentile(residual, 95)))
        if name == "loop_plane_drift_start_end":
            return FeatureValue(_angle_between(normals[0], normals[-1]) if len(normals) >= 2 else 0.0)
        if name == "loop_plane_outlier_fraction":
            return FeatureValue(float(np.mean(angles > 30.0)) if len(angles) else 0.0)
        if name == "loop_plane_twist_integral":
            return FeatureValue(float(np.sum(sliding["adjacent_angles"])))
        if name == "loop_signed_distance_to_plane_skewness":
            return FeatureValue(float(skew(sliding["signed_distance"])))
        if name in ("loop_sliding_plane_angle_mean", "local_plane_drift_index"):
            return FeatureValue(float(np.mean(angles)) if len(angles) else 0.0)
        if name == "loop_sliding_plane_angle_std":
            return FeatureValue(float(np.std(angles)) if len(angles) else 0.0)
    if name in ("loop_geodesic_euclidean_ratio", "loop_graph_bottleneck_distance", "loop_graph_branchiness", "loop_graph_cycle_count_proxy", "loop_knn_graph_length", "loop_local_density_variance", "loop_revisit_index"):
        return _graph_loop_feature(ctx, segment, name)
    if name in ("loop_acceleration_burst_count", "loop_curvature_speed_correlation", "loop_late_slowing_fraction", "loop_shape_length_normalized", "loop_speed_shape_correlation", "loop_stop_go_index", "loop_time_warping_energy"):
        return _time_normalized_loop_feature(ctx, segment, name)
    if name.startswith("loop_multiscale") or name in ("loop_alpha_shape_area_proxy", "loop_radial_peak_persistence"):
        return _multiscale_loop_feature(ctx, segment, name)
    return None


def _nonlocal_distances(points: np.ndarray, exclusion: int = 5) -> np.ndarray:
    vals = []
    for i in range(len(points)):
        for j in range(i + exclusion, len(points)):
            vals.append(np.linalg.norm(points[i] - points[j]))
    return np.asarray(vals, dtype=float)


def _hull_depths(points: np.ndarray) -> np.ndarray:
    try:
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]
    except Exception:
        return np.array([])
    depths = []
    for p in points:
        depths.append(np.min(np.linalg.norm(hull_pts - p, axis=1)))
    return np.asarray(depths)


def _fourier_loop_feature(
    ctx: FeatureContext,
    segment: str,
    name: str,
) -> FeatureValue:
    bundle = _fourier_loop_bundle(ctx, segment)
    if bundle is None:
        return FeatureValue(None, 0.0)
    centered = bundle["centered"]
    fft = bundle["coefficients"]
    energy = bundle["energy"]
    total = float(bundle["total_non_dc_energy"])
    if name == "loop_fourier_harmonic1_energy":
        return FeatureValue(float(energy[1]))
    if name == "loop_fourier_harmonic2_energy":
        return FeatureValue(float(energy[2]) if len(energy) > 2 else 0.0)
    if name == "loop_fourier_harmonic3_energy":
        return FeatureValue(float(energy[3]) if len(energy) > 3 else 0.0)
    if name == "loop_fourier_high_order_energy":
        return FeatureValue(float(np.sum(energy[4:])) if len(energy) > 4 else 0.0)
    if name == "loop_fourier_complexity_ratio":
        return FeatureValue(_safe_div(float(np.sum(energy[4:])), total))
    if name == "loop_fourier_ellipticity":
        h1 = fft[1]
        return FeatureValue(_safe_div(float(np.max(np.abs(h1))), float(np.min(np.abs(h1)))))
    if name == "loop_fourier_phase_shift_xyz":
        phases = np.angle(fft[1])
        return FeatureValue(float(np.degrees(np.std(phases))))
    if name == "loop_fourier_reconstruction_error_k3":
        filtered = np.zeros_like(fft)
        filtered[:min(4, len(fft))] = fft[:min(4, len(fft))]
        recon = np.fft.irfft(filtered, n=len(centered), axis=0)
        return FeatureValue(float(np.sqrt(np.mean((centered - recon) ** 2))))
    return FeatureValue(None, 0.0)


def _local_rotation_angles(points: np.ndarray) -> np.ndarray:
    v = np.diff(points, axis=0)
    norms = np.linalg.norm(v, axis=1)
    valid = norms > EPS
    v = v[valid] / norms[valid, None]
    if len(v) < 2:
        return np.array([])
    dots = np.sum(v[:-1] * v[1:], axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def _dihedral_angles(points: np.ndarray) -> np.ndarray:
    normals = []
    for i in range(1, len(points) - 1):
        n = np.cross(points[i] - points[i - 1], points[i + 1] - points[i])
        norm = np.linalg.norm(n)
        if norm > EPS:
            normals.append(n / norm)
    return np.array([_angle_between(normals[i], normals[i + 1]) or 0.0 for i in range(len(normals) - 1)])


def _sliding_normals(points: np.ndarray, window: int = 9) -> np.ndarray:
    normals = []
    window = max(5, min(window, len(points)))
    for i in range(0, len(points) - window + 1):
        _, _, vecs = _pca(points[i:i + window])
        normals.append(vecs[:, 2])
    return np.asarray(normals)


def _graph_loop_feature(
    ctx: FeatureContext,
    segment: str,
    name: str,
) -> FeatureValue:
    if name == "loop_geodesic_euclidean_ratio":
        points = ctx.vcg_segment(segment)
        path = _path_length(points)
        direct = float(np.linalg.norm(points[-1] - points[0]))
        return FeatureValue(_safe_div(path, direct))
    bundle = _graph_loop_bundle(ctx, segment)
    dists = bundle["pairwise"]
    nonlocal_d = bundle["nonlocal"]
    if name == "loop_graph_bottleneck_distance":
        return FeatureValue(float(np.min(nonlocal_d)) if len(nonlocal_d) else 0.0)
    adjacency = bundle["adjacency"]
    degree = bundle["degree"]
    if name == "loop_graph_branchiness":
        return FeatureValue(float(np.sum(degree > 2)))
    if name == "loop_graph_cycle_count_proxy":
        edges = np.sum(adjacency) / 2.0
        nodes = len(dists)
        return FeatureValue(float(max(0.0, edges - nodes + 1)))
    if name == "loop_knn_graph_length":
        k = min(3, max(1, len(dists) - 1))
        d_for_sort = dists.copy()
        np.fill_diagonal(d_for_sort, np.inf)
        sorted_d = np.sort(d_for_sort, axis=1)
        return FeatureValue(float(np.sum(sorted_d[:, :k]) / 2.0))
    if name == "loop_local_density_variance":
        return FeatureValue(float(np.var(np.sum(adjacency, axis=1))))
    if name == "loop_revisit_index":
        return FeatureValue(float(np.mean(np.sum(adjacency, axis=1) > 2)))
    return FeatureValue(None, 0.0)


def _time_normalized_loop_feature(ctx: FeatureContext, segment: str, name: str) -> FeatureValue:
    differential = _differential_bundle(ctx, segment)
    pca = _segment_pca_bundle(ctx, segment)
    kin = _kinematic_features(ctx, segment)
    if differential is None or pca is None or not kin:
        return FeatureValue(None, 0.0)
    points = differential["data"]
    speed = differential["speed"]
    accel = differential["accel"]
    centered = pca["centered"]
    radius = np.linalg.norm(centered, axis=1)
    if name == "loop_acceleration_burst_count":
        return FeatureValue(float(len(sp_signal.find_peaks(accel)[0])))
    if name == "loop_curvature_speed_correlation":
        curvature = differential["curvature"]
        return FeatureValue(_safe_corr(curvature, speed, default=0.0))
    if name == "loop_late_slowing_fraction":
        late = speed[int(2 * len(speed) / 3):]
        return FeatureValue(float(np.mean(late < np.percentile(speed, 25))))
    if name == "loop_shape_length_normalized":
        return FeatureValue(_safe_div(_path_length(points), float(np.mean(radius))))
    if name == "loop_speed_shape_correlation":
        return FeatureValue(_safe_corr(radius, speed, default=0.0))
    if name == "loop_stop_go_index":
        return FeatureValue(float(np.mean(speed < np.percentile(speed, 10)) + np.mean(speed > np.percentile(speed, 90))))
    if name == "loop_time_warping_energy":
        res = _resample_curve(points, 64)
        if res is None:
            return FeatureValue(None, 0.0)
        line = np.linspace(res[0], res[-1], len(res))
        return FeatureValue(_dtw_distance(res, line))
    return FeatureValue(None, 0.0)


def _multiscale_loop_feature(ctx: FeatureContext, segment: str, name: str) -> FeatureValue:
    bundle = _multiscale_loop_bundle(ctx, segment)
    if bundle is None:
        return FeatureValue(None, 0.0)
    areas = bundle["areas"]
    peaks = bundle["peaks"]
    intersections = bundle["intersections"]
    if name == "loop_alpha_shape_area_proxy":
        return FeatureValue(float(np.mean(areas)))
    if name == "loop_multiscale_area_stability":
        return FeatureValue(_safe_div(float(np.std(areas)), float(np.mean(areas))))
    if name == "loop_multiscale_curvature_slope":
        return FeatureValue(float(np.polyfit(np.arange(len(areas)), areas, 1)[0]) if len(areas) > 1 else 0.0)
    if name == "loop_multiscale_intersection_count":
        return FeatureValue(float(np.mean(intersections)))
    if name == "loop_radial_peak_persistence":
        return FeatureValue(float(np.mean(peaks)))
    return FeatureValue(None, 0.0)


def _loop2d_self_counts(points: np.ndarray) -> dict[str, float]:
    return {plane: float(_self_intersections(points[:, idx])) for plane, idx in PLANES.items()}


def _quality_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "einthoven_relation_error_rms":
        if ctx.ecg_12lead is None:
            return FeatureValue(None, 0.0)
        i, ii, iii = LEAD_MAP["I"], LEAD_MAP["II"], LEAD_MAP["III"]
        err = ctx.ecg_12lead[:, iii] - (ctx.ecg_12lead[:, ii] - ctx.ecg_12lead[:, i])
        return FeatureValue(float(np.sqrt(np.mean(err ** 2)) * 1000.0))
    if name == "chest_lead_misplacement_probability_proxy":
        slope = _r_progression_feature(ctx, "r_wave_progression_slope_v1_v6")
        return FeatureValue(None, 0.0) if slope is None or slope.value is None else FeatureValue(float(1.0 / (1.0 + np.exp(5.0 * slope.value / 1000.0))))
    if name == "kors_vs_pca_axis_difference_deg":
        if ctx.vcg_xyz is None:
            return FeatureValue(None, 0.0)
        mean_v = np.mean(ctx.vcg_xyz, axis=0)
        _, _, vecs = _pca(ctx.vcg_xyz)
        return FeatureValue(_angle_between(mean_v, vecs[:, 0]))
    if name in ("residual_per_lead_<lead>", "residual_rms_<lead>"):
        return _pca_svd_feature(ctx, "residual_rms_<lead>")
    if name == "vcg_trust_score":
        qual = _signal_basis_feature(ctx, "quality_score")
        ein = _quality_feature(ctx, "einthoven_relation_error_rms")
        penalty = min(1.0, (ein.value or 0.0) / 200.0) if ein else 0.0
        return FeatureValue(float(max(0.0, (qual.value if qual and qual.value is not None else 0.5) - 0.5 * penalty)))
    value = _signal_basis_feature(ctx, name)
    return value


def _lead8_svd_bundle(ctx: FeatureContext) -> Optional[dict[str, object]]:
    def build() -> Optional[dict[str, object]]:
        if ctx.ecg_12lead is None:
            return None
        bounds = ctx.bounds("qrs")
        if bounds is None:
            return None
        leads = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
        indices = [LEAD_MAP[lead] for lead in leads]
        matrix = np.asarray(
            ctx.ecg_12lead[bounds[0]:bounds[1] + 1, indices],
            dtype=float,
        )
        if matrix.shape[0] < 3:
            return None
        matrix = matrix - np.mean(matrix, axis=0)
        _, singular_values, right_vectors = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        eigenvalues = (singular_values ** 2) / max(matrix.shape[0] - 1, 1)
        shares = eigenvalues / (np.sum(eigenvalues) + EPS)
        rank = min(3, len(singular_values))
        reconstruction = (
            matrix @ right_vectors[:rank].T
        ) @ right_vectors[:rank]
        residual = matrix - reconstruction
        residual_rms_uv = (
            np.sqrt(np.mean(residual ** 2, axis=0)) * MILLIVOLT_TO_MICROVOLT
        )
        return {
            "leads": leads,
            "matrix": matrix,
            "singular_values": singular_values,
            "right_vectors": right_vectors,
            "eigenvalues": eigenvalues,
            "shares": shares,
            "residual": residual,
            "residual_rms_uv": residual_rms_uv,
        }

    return _memoized_primitive(ctx, ("lead8_qrs_svd", 3), build)


def _pca_svd_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    try:
        family = BiomarkerRegistry.definition(name).family
    except ValueError:
        return None
    if family != "PCA/SVD":
        return None
    bundle = _lead8_svd_bundle(ctx)
    if bundle is None:
        return FeatureValue(None, 0.0)
    matrix = bundle["matrix"]
    lambdas = bundle["eigenvalues"]
    shares = bundle["shares"]
    leads = bundle["leads"]
    if name == "<segment>_lead8_effective_rank":
        p = shares[shares > 0]
        return FeatureValue(float(np.exp(-np.sum(p * np.log(p)))))
    if name == "<segment>_lead8_lambda1..lambda8":
        per = {f"lambda{i+1}": float(lambdas[i]) for i in range(min(8, len(lambdas)))}
        return FeatureValue(float(lambdas[0]), per_lead=per)
    if name == "<segment>_lead8_nondipolar_fraction":
        return FeatureValue(float(np.sum(shares[3:])))
    if name == "residual_rms_<lead>":
        rms = bundle["residual_rms_uv"]
        per = {lead: float(rms[idx]) for idx, lead in enumerate(leads)}
        return FeatureValue(float(np.mean(rms)), per_lead=per)
    if name == "jackknife_influence_<lead>":
        # Refit the rank-3 subspace after excluding each lead.  The reported
        # influence is the change in RMS residual of the remaining leads,
        # relative to their residual under the full eight-lead fit.
        full_residual = bundle["residual"]
        influence: dict[str, float] = {}
        for excluded, lead in enumerate(leads):
            keep = np.arange(len(leads)) != excluded
            reduced = matrix[:, keep]
            _, reduced_s, reduced_vt = np.linalg.svd(
                reduced,
                full_matrices=False,
            )
            reduced_rank = min(3, len(reduced_s))
            reduced_reconstruction = (
                reduced @ reduced_vt[:reduced_rank].T
            ) @ reduced_vt[:reduced_rank]
            refit_rms = float(
                np.sqrt(np.mean((reduced - reduced_reconstruction) ** 2))
                * MILLIVOLT_TO_MICROVOLT
            )
            full_rms = float(
                np.sqrt(np.mean(full_residual[:, keep] ** 2))
                * MILLIVOLT_TO_MICROVOLT
            )
            influence[lead] = full_rms - refit_rms
        return FeatureValue(
            float(np.mean(list(influence.values()))),
            per_lead=influence,
            notes="Lead-8 rank-3 residual improvement after leave-one-lead-out refit (uV).",
        )
    return None


def _r_progression_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name not in {
        "poor_r_wave_progression_surrogate",
        "r_wave_progression_slope_v1_v6",
    }:
        return None
    if ctx.ecg_12lead is None:
        return None
    amps = []
    for lead in ("V1", "V2", "V3", "V4", "V5", "V6"):
        seg = ctx.ecg_segment(lead, "qrs")
        if seg is None:
            return FeatureValue(None, 0.0)
        amps.append(float(np.max(seg) * 1000.0))
    x = np.arange(1, 7, dtype=float)
    slope = float(np.polyfit(x, np.asarray(amps), 1)[0])
    if name == "r_wave_progression_slope_v1_v6":
        return FeatureValue(slope)
    if name == "poor_r_wave_progression_surrogate":
        return FeatureValue(float(max(0.0, -slope)))
    return None


def _signal_basis_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "delineation_lead":
        return FeatureValue(1.0, notes="Encoded categorical value: 1.0 = lead II/default fused delineation reference.")
    if name == "delineation_method":
        return FeatureValue(1.0, notes=f"Encoded categorical value: 1.0 = {ctx.delineation.source_method}.")
    if name == "fiducial_confidence":
        beat = ctx.beat
        if beat is None:
            return FeatureValue(0.0)
        vals = [getattr(beat, attr).confidence for attr in ("p_onset", "p_offset", "r_onset", "r_offset", "t_onset", "t_offset") if getattr(beat, attr, None) is not None]
        return FeatureValue(float(np.mean(vals)) if vals else 0.0)
    if name == "quality_score":
        return FeatureValue(float(ctx.delineation.quality_score))
    if name == "spatial_magnitude_m_t":
        return _spatial_magnitude_feature(ctx, name)
    if name == "vcg_axes_xyz":
        if ctx.vcg_xyz is None:
            return FeatureValue(None, 0.0)
        per = {"x_rms": float(np.sqrt(np.mean(ctx.vcg_xyz[:, 0] ** 2))), "y_rms": float(np.sqrt(np.mean(ctx.vcg_xyz[:, 1] ** 2))), "z_rms": float(np.sqrt(np.mean(ctx.vcg_xyz[:, 2] ** 2)))}
        return FeatureValue(float(np.sqrt(np.mean(ctx.vcg_xyz ** 2))), per_lead=per)
    return None


def _additional_vcg_idea_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "activation_front_spread_velocity_proxy":
        f = _loop3d_features(ctx, "qrs")
        dur = ctx.duration_s("qrs")
        return FeatureValue(None, 0.0) if not f or not dur else FeatureValue(_safe_div(f["path_length"], dur))
    if name == "repolarization_front_spread_velocity_proxy":
        f = _loop3d_features(ctx, "t")
        dur = ctx.duration_s("t")
        return FeatureValue(None, 0.0) if not f or not dur else FeatureValue(_safe_div(f["path_length"], dur))
    if name == "axis_crossing_sequence_entropy":
        if ctx.vcg_xyz is None:
            return FeatureValue(None, 0.0)
        signs = np.sign(ctx.vcg_xyz)
        crossings = np.sum(np.abs(np.diff(signs, axis=0)) > 0, axis=1)
        return FeatureValue(_entropy(crossings))
    if name == "curvature_torsion_phase_coupling":
        return _math_loop_feature_for_segment(ctx, "loop_curvature_torsion_ratio", "qrs")
    if name == "curvature_velocity_coupling":
        return _time_normalized_loop_feature(ctx, "qrs", "loop_curvature_speed_correlation")
    if name == "directional_concentration_kappa_proxy":
        pts = ctx.vcg_segment("qrs")
        if pts is None:
            return FeatureValue(None, 0.0)
        unit = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + EPS)
        return FeatureValue(float(np.linalg.norm(np.mean(unit, axis=0))))
    if name == "beatwise_loop_barycenter_motion":
        centers = []
        for curve in _beat_segment_curves(ctx, "qrs"):
            centers.append(np.mean(curve, axis=0))
        return FeatureValue(float(_path_length(np.asarray(centers))) if len(centers) >= 2 else None)
    if name == "breathing_baseline_vector_swing":
        centers = []
        record_vcg = ctx._record_vcg_xyz
        if record_vcg is None:
            return FeatureValue(None, 0.0)
        for beat in ctx.delineation.beats:
            bounds = _beat_bounds(ctx, beat, "p")
            if bounds is not None:
                lo = max(0, bounds[0] - int(0.08 * ctx.fs))
                hi = bounds[0]
                if hi > lo:
                    centers.append(np.mean(record_vcg[lo:hi], axis=0))
        return FeatureValue(float(np.mean(np.linalg.norm(np.asarray(centers) - np.mean(centers, axis=0), axis=1))) if len(centers) >= 2 else None)
    if name == "ectopic_beat_vector_deviation":
        return _ectopic_beat_vector_deviation(ctx)
    if name == "electrical_action_density_map":
        pts = ctx.vcg_segment("qrs")
        if pts is None:
            return FeatureValue(None, 0.0)
        mag = np.linalg.norm(pts, axis=1)
        hist, _ = np.histogramdd(pts, bins=4, weights=mag)
        return FeatureValue(float(np.max(hist)))
    if name == "geodesic_direction_path_length":
        pts = ctx.vcg_segment("qrs")
        if pts is None:
            return FeatureValue(None, 0.0)
        unit = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + EPS)
        dots = np.sum(unit[:-1] * unit[1:], axis=1)
        return FeatureValue(float(np.sum(np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))))))
    if name == "lead_to_vcg_consistency_score":
        if (
            ctx._record_ecg_12lead is None
            or ctx._record_vcg_xyz is None
        ):
            return FeatureValue(None, 0.0)
        x = np.asarray(ctx._record_ecg_12lead, dtype=float)
        y = np.asarray(ctx._record_vcg_xyz, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] != 3:
            return FeatureValue(None, 0.0)
        x_centered = x - np.mean(x, axis=0, keepdims=True)
        y_centered = y - np.mean(y, axis=0, keepdims=True)
        ss_tot_by_lead = np.sum(x_centered ** 2, axis=0)
        if float(np.sum(ss_tot_by_lead)) <= EPS:
            return FeatureValue(None, 0.0)

        # A forward ECG->VCG fit is tautological because the VCG is itself a
        # linear ECG transform. Quantify instead how much of the original
        # 12-lead signal can be reconstructed from its three VCG axes.
        beta_back, *_ = np.linalg.lstsq(y_centered, x_centered, rcond=None)
        reconstructed = y_centered @ beta_back
        ss_res_by_lead = np.sum((x_centered - reconstructed) ** 2, axis=0)
        total_r2 = 1.0 - _safe_div(
            float(np.sum(ss_res_by_lead)),
            float(np.sum(ss_tot_by_lead)),
        )
        per_lead = {}
        lead_names = sorted(LEAD_MAP, key=LEAD_MAP.get)
        for index in range(x.shape[1]):
            denominator = float(ss_tot_by_lead[index])
            if denominator <= EPS:
                continue
            lead_r2 = 1.0 - float(ss_res_by_lead[index]) / denominator
            lead_name = lead_names[index] if index < len(lead_names) else f"lead_{index}"
            per_lead[lead_name] = float(np.clip(lead_r2, 0.0, 1.0))
        return FeatureValue(
            float(np.clip(total_r2, 0.0, 1.0)),
            per_lead=per_lead,
            notes="Record-fitted VCG-to-12-lead back-projection R-squared; QC only.",
        )
    if name == "local_plane_drift_index":
        return _math_loop_feature_for_segment(ctx, "local_plane_drift_index", "qrs")
    if name == "loop_area_density":
        f = _loop3d_features(ctx, "qrs")
        return FeatureValue(None, 0.0) if not f else FeatureValue(_safe_div(f["area_best_fit_plane"], f["path_length"]))
    if name == "loop_chirality_index_3d":
        pts = ctx.vcg_segment("qrs")
        if pts is None or len(pts) < 4:
            return FeatureValue(None, 0.0)
        dt = 1.0 / ctx.fs
        derivative_input = _smooth_derivative_input(pts, ctx.fs)
        v = np.gradient(derivative_input, dt, axis=0)
        a = np.gradient(v, dt, axis=0)
        return FeatureValue(float(np.mean(np.einsum("ij,ij->i", pts, np.cross(v, a)))))
    if name == "loop_energy_density":
        pts = ctx.vcg_segment("qrs")
        f = _loop3d_features(ctx, "qrs")
        return FeatureValue(None, 0.0) if pts is None or not f else FeatureValue(_safe_div(float(_trapz(np.linalg.norm(pts, axis=1) ** 2, 1.0 / ctx.fs)), f["path_length"]))
    if name == "medication_sensitive_repolarization_score":
        qtc = _interval_feature(ctx, "qtc_fridericia_ms")
        tpt = _interval_feature(ctx, "tpeak_tend_ms")
        svg = _qrst_feature(ctx, "svg_angle_to_t_area_deg")
        vals = [
            min(1.0, (qtc.value or 0.0) / 500.0) if qtc and qtc.value is not None else None,
            min(1.0, (tpt.value or 0.0) / 120.0) if tpt and tpt.value is not None else None,
            min(1.0, (svg.value or 0.0) / 180.0) if svg and svg.value is not None else None,
        ]
        vals = [v for v in vals if v is not None]
        return FeatureValue(float(np.mean(vals)) if vals else None)
    if name == "multi_resolution_qrs_shape_signature":
        return _multiscale_loop_feature(ctx, "qrs", "loop_multiscale_area_stability")
    if name == "pacemaker_capture_vector_signature":
        qrs_dur = _interval_feature(ctx, "qrs_duration_ms")
        init_v = _subsegment_kinematic(ctx, "qrs", 0.0, 0.15)
        return FeatureValue(None, 0.0) if qrs_dur is None or qrs_dur.value is None or init_v is None else FeatureValue(float(min(1.0, qrs_dur.value / 180.0 + init_v / 10.0) / 2.0))
    if name == "persistent_homology_loop_lifetime":
        return FeatureValue(
            None,
            0.0,
            notes=(
                "Retired: the historical calculator was a radial hole proxy, "
                "not persistent homology. A real persistence-diagram engine "
                "must be validated before this name can be reintroduced."
            ),
        )
    if name == "qrs_t_directional_wasserstein":
        q, t = ctx.vcg_segment("qrs"), ctx.vcg_segment("t")
        if q is None or t is None:
            return FeatureValue(None, 0.0)
        return FeatureValue(
            _directional_wasserstein_distance(q, t),
            notes=(
                "Amplitude-weighted empirical angular 1-Wasserstein distance "
                "on the unit sphere; geodesic cost normalized by pi."
            ),
        )
    if name == "qrs_t_loop_mirror_symmetry":
        q, t = _resample_curve(ctx.vcg_segment("qrs")), _resample_curve(ctx.vcg_segment("t"))
        if q is None or t is None:
            return FeatureValue(None, 0.0)
        q_normalized = _shape_normalize(q)
        t_normalized = _shape_normalize(t)
        if q_normalized is None or t_normalized is None:
            return FeatureValue(None, 0.0)
        distance_value = float(
            np.sqrt(np.mean((q_normalized + t_normalized) ** 2))
        )
        return FeatureValue(
            float(distance_value / (1.0 + distance_value)),
            notes="Centered and RMS-normalized fixed-axis mirror distance in [0, 1).",
        )
    if name == "repolarization_return_efficiency":
        t = ctx.vcg_segment("t")
        bounds = ctx.bounds("t")
        peak = ctx.marker_index("t_peak")
        if t is None or bounds is None or peak is None:
            return FeatureValue(None, 0.0)
        local_peak = int(np.clip(peak - bounds[0], 0, len(t) - 1))
        terminal = t[local_peak:]
        if len(terminal) < 2:
            return FeatureValue(None, 0.0)
        direct = float(np.linalg.norm(terminal[-1] - terminal[0]))
        path = _path_length(terminal)
        return FeatureValue(
            float(np.clip(_safe_div(direct, path), 0.0, 1.0)),
            notes="Direct T-peak-to-end return divided by terminal path length.",
        )
    if name == "segmental_directional_memory":
        qrs_terminal = _window_vector(ctx, "qrs", 20, True)
        t_initial = _window_vector(ctx, "t", 20)
        if qrs_terminal is None or t_initial is None:
            return FeatureValue(None, 0.0)
        denominator = float(
            np.linalg.norm(qrs_terminal) * np.linalg.norm(t_initial)
        )
        if denominator < EPS:
            return FeatureValue(None, 0.0)
        cosine = float(
            np.clip(
                np.dot(qrs_terminal, t_initial) / denominator,
                -1.0,
                1.0,
            )
        )
        return FeatureValue(cosine)
    if name == "sex_age_adjusted_residual_candidate":
        return FeatureValue(
            None,
            0.0,
            notes=(
                "not_evaluable: no train-fitted age/sex normative model is "
                "configured; a zero placeholder would be scientifically invalid."
            ),
        )
    if name == "spherical_harmonic_vcg_signature":
        pts = ctx.vcg_segment("qrs")
        if pts is None:
            return FeatureValue(None, 0.0)
        unit = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + EPS)
        theta = np.arctan2(unit[:, 1], unit[:, 0])
        phi = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
        return FeatureValue(float(_entropy(theta) + _entropy(phi)))
    if name == "st_t_continuity_breakpoint_score":
        stt = ctx.vcg_segment("st_t")
        if stt is None or len(stt) < 4:
            return FeatureValue(None, 0.0)
        mag = np.linalg.norm(stt, axis=1)
        return FeatureValue(float(np.max(np.abs(np.diff(mag, n=2)))))
    if name == "trajectory_reversal_fraction":
        return _backtracking_fraction(ctx.vcg_segment("qrs"))
    if name == "topological_signature_distance_qrs_t":
        return FeatureValue(
            None,
            0.0,
            notes=(
                "Retired: the historical calculator returned Hausdorff "
                "distance rather than persistence-diagram distance."
            ),
        )
    if name == "vcg_phase_space_recurrence_rate":
        pts = ctx.vcg_segment("qrs")
        if pts is None:
            return FeatureValue(None, 0.0)
        metrics = _recurrence_metrics(pts)
        return FeatureValue(None, 0.0) if metrics is None else FeatureValue(
            metrics[0],
            notes=(
                "Recurrence radius is fixed at 10% of loop diameter; "
                "a two-sample Theiler window excludes trivial neighbors."
            ),
        )
    if name == "vcg_phase_space_determinism":
        pts = ctx.vcg_segment("qrs")
        if pts is None:
            return FeatureValue(None, 0.0)
        metrics = _recurrence_metrics(pts)
        return FeatureValue(None, 0.0) if metrics is None else FeatureValue(
            metrics[1],
            notes=(
                "Fraction of recurrent points belonging to diagonal runs of "
                "at least three samples."
            ),
        )
    if name == "vcg_vector_field_curl_proxy":
        pts = ctx.vcg_segment("qrs")
        if pts is None or len(pts) < 3:
            return FeatureValue(None, 0.0)
        v = np.diff(pts, axis=0)
        return FeatureValue(float(np.mean(np.linalg.norm(np.cross(v[:-1], v[1:]), axis=1))))
    if name == "vcg_vector_field_divergence_proxy":
        pts = ctx.vcg_segment("qrs")
        if pts is None:
            return FeatureValue(None, 0.0)
        r = np.linalg.norm(pts - np.mean(pts, axis=0), axis=1)
        return FeatureValue(float(r[-1] - r[0]))
    return None


def _beat_variability_feature(
    ctx: FeatureContext,
    name: str,
    *,
    segment_override: Optional[str] = None,
) -> Optional[FeatureValue]:
    rr = np.diff([b.r_peak.sample for b in ctx.delineation.beats]) / ctx.fs if len(ctx.delineation.beats) >= 2 else np.array([])
    if name == "rr_entropy/irregularity_index":
        if len(rr) < 2:
            return FeatureValue(None, 0.0)
        per = {"entropy": _entropy(rr), "irregularity_index": _safe_div(float(np.std(rr)), float(np.mean(rr)))}
        return FeatureValue(per["irregularity_index"], per_lead=per)
    if name == "rr_mean_ms/sd_ms/rmssd_ms/pnn50":
        if len(rr) < 2:
            return FeatureValue(None, 0.0)
        diff = np.diff(rr)
        per = {
            "rr_mean_ms": float(np.mean(rr) * 1000.0),
            "rr_sd_ms": float(np.std(rr, ddof=1) * 1000.0),
            "rr_rmssd_ms": float(np.sqrt(np.mean(diff ** 2)) * 1000.0) if len(diff) else 0.0,
            "rr_pnn50": float(np.mean(np.abs(diff) > 0.05)) if len(diff) else 0.0,
        }
        return FeatureValue(per["rr_mean_ms"], per_lead=per)
    if name == "atrial_fibrillation_rr_surrogate":
        if len(rr) < 3:
            return FeatureValue(
                None,
                0.0,
                notes="At least four R peaks are required for rhythm context",
            )
        rr_median = float(np.median(rr))
        if rr_median <= 0:
            return FeatureValue(None, 0.0, notes="Invalid RR intervals")
        rr_cv = float(np.std(rr, ddof=1) / np.mean(rr))
        successive = float(np.median(np.abs(np.diff(rr))) / rr_median)
        rr_component = float(
            np.clip(
                max((rr_cv - 0.04) / 0.16, (successive - 0.06) / 0.24),
                0.0,
                1.0,
            )
        )
        rhythm = ctx.delineation.rhythm
        p_coverage = rhythm.p_wave_coverage
        if p_coverage is None:
            complete_p = sum(
                beat.segment_confidence("p") >= measurement_confidence_threshold("p")
                for beat in ctx.delineation.beats
            )
            p_coverage = complete_p / max(1, len(ctx.delineation.beats))
        score = float(
            np.clip(0.72 * rr_component + 0.28 * (1.0 - p_coverage), 0.0, 1.0)
        )
        if rhythm.rhythm_class.value == "af_compatible_pattern":
            score = max(score, 0.75)
        confidence = float(
            min(0.9, 0.35 + 0.05 * len(rr))
            * (0.65 + 0.35 * rhythm.confidence)
            * max(0.35, ctx.delineation.quality_score)
        )
        return FeatureValue(
            score,
            confidence,
            {
                "rr_cv": rr_cv,
                "successive_irregularity": successive,
                "p_wave_coverage": float(p_coverage),
                "n_rr_intervals": float(len(rr)),
            },
            "Technical AF-compatible rhythm surrogate; not an AF diagnosis",
        )
    if name in ("qrs_alternans_vector", "t_wave_alternans_vector"):
        segment = "qrs" if name.startswith("qrs") else "t"
        curves = _alternans_contiguous_curves(ctx, segment)
        if curves is None:
            return FeatureValue(None, 0.0, notes="not_evaluable: incompatible rhythm or fewer than four uninterrupted valid beats; no parity compression")
        mags = np.asarray([np.max(np.linalg.norm(curve, axis=1)) for curve in curves])
        return FeatureValue(float(abs(np.mean(mags[::2]) - np.mean(mags[1::2]))),
                            notes=_alternans_interpretation_notes(ctx, len(curves)))
    if name == "qrs_t_angle_variability":
        angles = _beat_qrst_angles(ctx)
        return FeatureValue(float(np.std(angles)) if len(angles) >= 2 else None)
    if name == "<seg>_vector_variability":
        vals = _beat_segment_magnitudes(ctx, segment_override or "qrs")
        return FeatureValue(float(np.std(vals)) if len(vals) >= 2 else None)
    if name == "<seg>_loop_area_variability":
        vals = _beat_segment_areas(ctx, segment_override or "qrs")
        return FeatureValue(float(np.std(vals)) if len(vals) >= 2 else None)
    if name == "<seg>_axis_beat_to_beat_variability_deg":
        vectors = _beat_segment_vectors(ctx, segment_override or "qrs")
        if len(vectors) < 2:
            return FeatureValue(None, 0.0)
        reference = np.sum(vectors, axis=0)
        if np.linalg.norm(reference) < EPS:
            reference = vectors[0]
        angles = [_angle_between(vector, reference) for vector in vectors]
        angles = [angle for angle in angles if angle is not None]
        return FeatureValue(
            float(np.std(angles, ddof=1)) if len(angles) >= 2 else None
        )
    if name == "<seg>_beat_to_beat_*_distance":
        curves = _beat_segment_curves(ctx, segment_override or "qrs")
        if len(curves) < 2:
            return FeatureValue(None, 0.0)
        median = np.median(np.stack(curves), axis=0)
        vals = [float(np.sqrt(np.mean((c - median) ** 2))) for c in curves]
        return FeatureValue(float(np.median(vals)))
    return None


def _beat_bounds(ctx: FeatureContext, beat, segment: str) -> Optional[tuple[int, int]]:
    def sample(attr):
        p = getattr(beat, attr, None)
        return None if p is None else int(p.sample)
    qrs_start = sample("r_onset") if sample("r_onset") is not None else sample("q_peak")
    qrs_end = sample("r_offset") if sample("r_offset") is not None else sample("s_peak")
    if segment == "qrs":
        start, end = qrs_start, qrs_end
    elif segment == "t":
        start, end = sample("t_onset"), sample("t_offset")
    elif segment == "p":
        start, end = sample("p_onset"), sample("p_offset")
    else:
        start, end = qrs_start, sample("t_offset")
    if start is None or end is None or ctx._record_vcg_xyz is None:
        return None
    start = max(0, min(start, len(ctx._record_vcg_xyz) - 1))
    end = max(0, min(end, len(ctx._record_vcg_xyz) - 1))
    return (start, end) if end > start else None


def _beat_segment_curves(ctx: FeatureContext, segment: str, *, time_normalized: bool = False) -> list[np.ndarray]:
    curves = []
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return curves
    for beat in ctx.delineation.beats:
        if time_normalized:
            # Runtime import avoids the catalog/thesis module import cycle.
            from .thesis_derived_biomarkers import _beat_sample
            attributes = {"qrs": ("r_onset", "r_offset"), "t": ("t_onset", "t_offset")}
            if segment not in attributes:
                return []
            start, end = (_beat_sample(beat, attr, segment) for attr in attributes[segment])
            bounds = (start, end) if (start is not None and end is not None
                      and 0 <= start < end < len(record_vcg)) else None
        else:
            bounds = _beat_bounds(ctx, beat, segment)
        if bounds is None:
            continue
        samples = record_vcg[bounds[0]:bounds[1] + 1]
        if time_normalized:
            if samples.ndim != 2 or samples.shape[1] != 3 or not np.isfinite(samples).all():
                continue
            source = np.linspace(0., 1., len(samples))
            target = np.linspace(0., 1., 64)
            curve = np.column_stack([np.interp(target, source, samples[:, axis]) for axis in range(3)])
        else:
            curve = _resample_curve(samples, 64)
        if curve is not None:
            curves.append(curve)
    return curves


def _alternans_interpretation_notes(ctx: FeatureContext, count: int) -> str:
    rhythm = getattr(ctx.delineation, "rhythm", None)
    rhythm_class = getattr(rhythm, "rhythm_class", None)
    status = getattr(rhythm_class, "value", "unknown")
    return (f"exploratory_odd_even_difference; n_beats={count}; rhythm_pattern={status}; "
            "time_normalized_segments; duration_alternans_not_measured; "
            "undetected_beats_and_local_rhythm_changes_not_excluded; "
            "not_validated_clinical_TWA")


def _alternans_contiguous_curves(ctx: FeatureContext, segment: str):
    """Do not reinterpret a filtered beat sequence as consecutive odd/even beats.

    Conservative abstention when any detected beat lacks a usable segment.
    This protects observed beat parity, not undetected beats or clinical TWA
    validity. Other variability features may still use available segments.
    """
    rhythm = getattr(ctx.delineation, "rhythm", None)
    # Known irregular/mixed patterns cannot be interpreted by this simple
    # unadjusted odd/even estimator. This does not certify regularity when
    # rhythm metadata are unavailable; the estimator remains exploratory.
    if rhythm is not None and rhythm.requires_beatwise_rate_correction:
        return None
    curves = _beat_segment_curves(ctx, segment, time_normalized=True)
    if len(curves) < 4 or len(curves) != len(ctx.delineation.beats):
        return None
    if any(curve.ndim != 2 or curve.shape[1] != 3 or not np.isfinite(curve).all()
           for curve in curves):
        return None
    return curves


def _beat_segment_magnitudes(ctx: FeatureContext, segment: str) -> np.ndarray:
    vals = []
    for curve in _beat_segment_curves(ctx, segment):
        vals.append(float(np.max(np.linalg.norm(curve, axis=1))))
    return np.asarray(vals)


def _cluster_beat_area_vectors(
    ctx: FeatureContext,
    segment: str,
    cluster_id: int,
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return vectors
    for beat in ctx.delineation.beats:
        if getattr(beat, "morphology_cluster_id", None) != cluster_id:
            continue
        bounds = _beat_bounds(ctx, beat, segment)
        if bounds is None:
            continue
        values = record_vcg[bounds[0]:bounds[1] + 1]
        vector = np.asarray(_trapz(values, 1.0 / ctx.fs, axis=0), dtype=float)
        if np.linalg.norm(vector) > EPS:
            vectors.append(vector)
    return vectors


def _ectopic_beat_vector_deviation(ctx: FeatureContext) -> FeatureValue:
    """Spatial QRS-area-vector distance of a secondary beat population.

    Uses FusionDelineator's morphology clustering (cluster 0 = primary/
    dominant, cluster 1 = secondary) instead of the record's overall beat
    dispersion, so a genuine second population -- bigeminy, multifocal
    ectopy -- is measured against the sinus/dominant template it actually
    deviates from, per this feature's own catalog note ("analyze
    extrasystoles separately; do not mix them as pathology").
    """
    primary = _cluster_beat_area_vectors(ctx, "qrs", 0)
    secondary = _cluster_beat_area_vectors(ctx, "qrs", 1)
    if len(primary) < 3 or len(secondary) < 3:
        return FeatureValue(None, 0.0)
    primary_mean = np.mean(primary, axis=0)
    secondary_mean = np.mean(secondary, axis=0)
    return FeatureValue(float(np.linalg.norm(secondary_mean - primary_mean)))


def _beat_area_vectors_uv_ms(
    ctx: FeatureContext,
    segment: str,
) -> list[np.ndarray]:
    """Per-beat area vector of one segment, in the GEH unit.

    Integrated on the origin-corrected VCG through ``area_vector_uv_ms``, so a
    beat-level vector is the same quantity as the record-level one in
    ``geh_area_metrics`` and the two can be compared without a conversion.
    """

    centered = ctx.origin_corrected_vcg()
    if centered is None:
        return []
    vectors: list[np.ndarray] = []
    for beat in ctx.delineation.beats:
        bounds = _beat_bounds(ctx, beat, segment)
        if bounds is None or bounds[1] + 1 > len(centered):
            continue
        window = centered[bounds[0]:bounds[1] + 1]
        if len(window) < 2:
            continue
        vector = np.asarray(area_vector_uv_ms(window, ctx.fs), dtype=float)
        if np.all(np.isfinite(vector)):
            vectors.append(vector)
    return vectors


def _vector_scatter(vectors: Sequence[np.ndarray]) -> Optional[float]:
    """Spread of a set of 3D vectors, independent of the axes it is read in.

    The square root of the total variance -- the trace of the covariance
    matrix, equivalently the root sum of the three per-axis variances. It is
    invariant under rotation of the coordinate frame, which the per-axis
    standard deviation is not, and it collapses to the ordinary standard
    deviation when the scatter lies along one direction.

    Two vectors are enough for a sample variance and are what a ten-second
    strip sometimes offers; fewer abstains.
    """

    if len(vectors) < 2:
        return None
    stacked = np.vstack([np.asarray(vector, dtype=float) for vector in vectors])
    if not np.all(np.isfinite(stacked)):
        return None
    variances = np.var(stacked, axis=0, ddof=1)
    return float(np.sqrt(float(np.sum(variances))))


def _beat_tpeak_tend_vectors(ctx: FeatureContext) -> list[np.ndarray]:
    """Per-beat displacement from the T peak to the T end, in microvolts."""

    centered = ctx.origin_corrected_vcg()
    if centered is None:
        return []
    vectors: list[np.ndarray] = []
    for beat in ctx.delineation.beats:
        peak = getattr(beat, "t_peak", None)
        end = getattr(beat, "t_offset", None)
        if peak is None or end is None:
            continue
        start_index = int(peak.sample)
        end_index = int(end.sample)
        if not (0 <= start_index < end_index < len(centered)):
            continue
        vector = (centered[end_index] - centered[start_index]) * 1000.0
        if np.all(np.isfinite(vector)):
            vectors.append(np.asarray(vector, dtype=float))
    return vectors


def _beat_segment_vectors(ctx: FeatureContext, segment: str) -> list[np.ndarray]:
    return [np.mean(c, axis=0) for c in _beat_segment_curves(ctx, segment)]


def _beat_segment_areas(ctx: FeatureContext, segment: str) -> np.ndarray:
    vals = []
    for curve in _beat_segment_curves(ctx, segment):
        centered, _, vecs = _pca(curve)
        vals.append(abs(_shoelace((centered @ vecs)[:, :2])))
    return np.asarray(vals)


def _beat_qrst_angles(ctx: FeatureContext) -> np.ndarray:
    vals = []
    record_vcg = ctx._record_vcg_xyz
    if record_vcg is None:
        return np.asarray(vals)
    for beat in ctx.delineation.beats:
        qb = _beat_bounds(ctx, beat, "qrs")
        tb = _beat_bounds(ctx, beat, "t")
        if qb is None or tb is None:
            continue
        q = np.mean(record_vcg[qb[0]:qb[1] + 1], axis=0)
        t = np.mean(record_vcg[tb[0]:tb[1] + 1], axis=0)
        angle = _angle_between(q, t)
        if angle is not None:
            vals.append(angle)
    return np.asarray(vals)


def _experimental_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    points = ctx.vcg_segment("qrs")
    if points is None:
        # Not FeatureValue: this fires before any name has been looked at, so
        # returning one claimed every name that reached here. See
        # ``_InputUnavailable``.
        return _INPUT_UNAVAILABLE
    if name == "<seg>_cov_*":
        vals = _loop3d_features(ctx, "qrs")
        if not vals:
            return FeatureValue(None, 0.0)
        lambdas = np.array([vals["lambda1"], vals["lambda2"], vals["lambda3"]])
        shares = lambdas / (np.sum(lambdas) + EPS)
        per = {
            "trace": float(np.sum(lambdas)),
            "determinant": float(np.prod(lambdas)),
            "condition": _safe_div(float(lambdas[0]), float(lambdas[-1])),
            "entropy": float(-np.sum(shares * np.log2(shares + EPS))),
        }
        return FeatureValue(per["trace"], per_lead=per)
    if name == "<seg>_directional_hysteresis_angle_deg":
        mid = len(points) // 2
        return FeatureValue(_angle_between(np.mean(points[:mid], axis=0), np.mean(points[mid:], axis=0)))
    if name == "<seg>_electrical_work":
        speed = np.linalg.norm(np.diff(points, axis=0) * ctx.fs, axis=1)
        mag = np.linalg.norm(points[:-1], axis=1)
        return FeatureValue(float(_trapz(mag * speed, 1.0 / ctx.fs)))
    if name == "<seg>_loop_angular_momentum":
        v = np.diff(points, axis=0) * ctx.fs
        return FeatureValue(float(_trapz(np.linalg.norm(np.cross(points[:-1], v), axis=1), 1.0 / ctx.fs)))
    if name == "<seg>_radial_peak_count/persistence_proxy":
        radial = np.linalg.norm(points - np.mean(points, axis=0), axis=1)
        peaks, _ = sp_signal.find_peaks(radial)
        prom = sp_signal.peak_prominences(radial, peaks)[0] if len(peaks) else np.array([])
        return FeatureValue(float(len(peaks)), per_lead={"persistence_proxy": float(np.mean(prom)) if len(prom) else 0.0})
    if name == "<seg>_self_crossings_*":
        return FeatureValue(float(sum(_loop2d_self_counts(points).values())), per_lead=_loop2d_self_counts(points))
    if name == "<seg>_spherical_direction_entropy":
        unit = points / (np.linalg.norm(points, axis=1, keepdims=True) + EPS)
        classes = (unit > 0).astype(int) @ np.array([1, 2, 4])
        return FeatureValue(_entropy(classes, bins=8))
    if name == "<seg>_symbolic_*":
        axis = np.argmax(np.abs(points), axis=1)
        return FeatureValue(_entropy(axis, bins=3))
    if name == "qrs_t_energy_transfer_proxy":
        q = _spatial_magnitude_feature(ctx, "magnitude_area_qrs")
        t = _spatial_magnitude_feature(ctx, "magnitude_area_st_t")
        return FeatureValue(None, 0.0) if q is None or t is None or q.value is None or t.value is None else FeatureValue(_safe_div(t.value, q.value))
    if name == "qrs_t_wasserstein_radial_proxy":
        q, t = ctx.vcg_segment("qrs"), ctx.vcg_segment("t")
        if q is None or t is None:
            return FeatureValue(None, 0.0)
        qrad = np.sort(np.linalg.norm(q - np.mean(q, axis=0), axis=1))
        trad = np.sort(np.linalg.norm(t - np.mean(t, axis=0), axis=1))
        n = min(len(qrad), len(trad))
        return FeatureValue(float(np.mean(np.abs(qrad[:n] - trad[:n]))))
    return None


def _literature_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "hf_qrs_vector_notching_score":
        return _pathophysiology_feature(ctx, "qrs_loop_notch_burden")
    if name == "microvolt_t_wave_alternans_vector":
        # Catalog contract: "even-odd difference of resampled T loops,
        # reported as RMS", unit uV. It previously delegated to
        # t_wave_alternans_vector, which is a different quantity (difference
        # of mean loop magnitudes) and is declared in mV -- so the value was
        # both the wrong statistic and a factor 1000 off its own unit. TWA is
        # discussed clinically at single-digit microvolts, which makes that
        # mislabel the kind that survives straight into a claim.
        curves = _alternans_contiguous_curves(ctx, "t")
        if curves is None:
            return FeatureValue(
                None,
                0.0,
                notes="not_evaluable: incompatible rhythm or fewer than four uninterrupted valid beats; no parity compression",
            )
        length = int(np.median([len(curve) for curve in curves]))
        if length < 4:
            return FeatureValue(
                None,
                0.0,
                notes="not_evaluable: T loops are too short to resample",
            )
        grid = np.linspace(0.0, 1.0, length)
        resampled = []
        for curve in curves:
            array = np.asarray(curve, dtype=float)
            if array.ndim != 2 or len(array) < 2:
                continue
            if len(array) == length:
                # The production loader already uses a common temporal grid.
                # Avoid three identity interpolations and a copy per beat.
                resampled.append(array)
                continue
            source = np.linspace(0.0, 1.0, len(array))
            resampled.append(
                np.column_stack(
                    [
                        np.interp(grid, source, array[:, axis])
                        for axis in range(array.shape[1])
                    ]
                )
            )
        if len(resampled) < 4:
            return FeatureValue(
                None,
                0.0,
                notes="not_evaluable: fewer than four resampled T loops",
            )
        even = np.mean(np.stack(resampled[::2]), axis=0)
        odd = np.mean(np.stack(resampled[1::2]), axis=0)
        difference = np.linalg.norm(even - odd, axis=1)
        return FeatureValue(
            float(np.sqrt(np.mean(difference ** 2)) * MILLIVOLT_TO_MICROVOLT),
            notes=_alternans_interpretation_notes(ctx, len(resampled)),
        )
    if name == "p_wave_mvp_vector_score":
        dur = _interval_feature(ctx, "p_duration_ms")
        mag = _segment_specific_feature(ctx, "p_vector_magnitude")
        comp = _p_loop_extra(ctx, "p_loop_fractionated_motion_index")
        vals = [v.value for v in (dur, mag, comp) if v is not None and v.value is not None]
        return FeatureValue(float(np.mean(vals)) if vals else None)
    if name == "p_wave_terminal_force_vector_vcg":
        return _p_loop_extra(ctx, "p_loop_terminal_force_vector")
    if name == "qrs_area_t_area_crt_response_surrogate":
        return _qrst_feature(ctx, "qrs_loop_area_t_loop_area_ratio")
    if name == "qrs_fragmentation_high_frequency_energy":
        # Must read the metadata-qualified wideband path.  The source analogue
        # frontend is documented only through 150 Hz, so using 150-250 Hz
        # would quantify frontend/filter artefact rather than myocardium.
        # No silent fallback -- without a wideband source the marker abstains.
        qrs = ctx.vcg_segment_wideband("qrs")
        if qrs is None:
            return FeatureValue(
                None,
                0.0,
                notes=(
                    "not_evaluable: no wideband signal source for the "
                    "high-frequency QRS band"
                ),
            )
        if len(qrs) < 4:
            return FeatureValue(
                None,
                0.0,
                notes="not_evaluable: QRS segment is too short",
            )
        nyq = ctx.fs / 2.0
        low = 80.0
        high = 150.0
        if nyq <= high:
            return FeatureValue(
                None,
                0.0,
                notes=(
                    "not_evaluable: sampling rate does not support the fixed "
                    "80-150 Hz HF-QRS band"
                ),
            )
        sos = sp_signal.butter(3, [low / nyq, high / nyq], btype="bandpass", output="sos")
        # scipy's zero-phase SOS filter requires a padding interval that can
        # exceed a physiologically valid but very short delineated QRS. Such a
        # segment is unavailable for this marker, not a calculation failure.
        default_padlen = 3 * (
            2 * len(sos)
            + 1
            - min(
                int(np.sum(sos[:, 2] == 0.0)),
                int(np.sum(sos[:, 5] == 0.0)),
            )
        )
        if len(qrs) <= default_padlen:
            return FeatureValue(
                None,
                0.0,
                notes=(
                    "not_evaluable: QRS segment is shorter than the "
                    f"zero-phase filter requirement ({len(qrs)} <= {default_padlen})"
                ),
            )
        # Filtering ||VCG|| is a nonlinear envelope operation: opposite-sign
        # XYZ components can create or erase HF content before the bandpass.
        # Filter each physical axis first, then aggregate spatial energy.
        filtered_xyz = sp_signal.sosfiltfilt(sos, qrs, axis=0)
        # Squared amplitude: mV^2 -> uV^2 is a factor 1e6, not 1e3.
        return FeatureValue(
            float(np.mean(np.sum(filtered_xyz ** 2, axis=1)))
            * MILLIVOLT_TO_MICROVOLT ** 2,
            per_lead={
                "x_uv2": float(np.mean(filtered_xyz[:, 0] ** 2))
                * MILLIVOLT_TO_MICROVOLT ** 2,
                "y_uv2": float(np.mean(filtered_xyz[:, 1] ** 2))
                * MILLIVOLT_TO_MICROVOLT ** 2,
                "z_uv2": float(np.mean(filtered_xyz[:, 2] ** 2))
                * MILLIVOLT_TO_MICROVOLT ** 2,
            },
            notes=(
                "Axiswise XYZ 80-150 Hz bandpass followed by spatial-energy "
                "aggregation; analogue bandwidth qualification is required."
            ),
        )
    if name == "qrs_loop_dihedral_angle_deg":
        return _math_loop_feature(ctx, "loop_dihedral_angle_mean")
    if name == "qrs_loop_folding_index":
        return _math_loop_feature(ctx, "loop_total_self_intersections")
    if name == "qrs_loop_roundness_planarity_composite":
        f = _loop3d_features(ctx, "qrs")
        return FeatureValue(None, 0.0) if not f else FeatureValue(float(np.mean([f["roundness_best_fit_plane"], f["planarity"], 1.0 / (1.0 + f["thickness"])])))
    if name == "qrs_t_angle_lability_index":
        return _beat_variability_feature(ctx, "qrs_t_angle_variability")
    if name in ("qt_rr_hysteresis_surrogate", "qt_variability_index_vector"):
        pairs = [item for item in ctx.delineation.qt_rr_measurements()
                 if np.isfinite(item.confidence) and item.confidence >= 0.35
                 and np.isfinite(item.qt_ms) and item.qt_ms > 0
                 and np.isfinite(item.preceding_rr_seconds) and item.preceding_rr_seconds > 0]
        if len(pairs) < 4:
            return FeatureValue(None, 0.0, notes="fewer than four qualified QT/preceding-RR pairs")
        qt = np.asarray([item.qt_ms / 1000.0 for item in pairs])
        rr = np.asarray([item.preceding_rr_seconds for item in pairs])
        if np.var(qt) <= EPS or np.var(rr) <= EPS:
            return FeatureValue(None, 0.0, notes="QT or RR variance not resolvable")
        if name == "qt_rr_hysteresis_surrogate":
            return FeatureValue(float(np.corrcoef(qt, rr)[0, 1]))
        qt_norm = np.var(qt) / np.mean(qt) ** 2
        rr_norm = np.var(rr) / np.mean(rr) ** 2
        return FeatureValue(float(np.log10(qt_norm / rr_norm)))
    if name == "spatial_mean_qrs_t_angle_deg":
        return _qrst_feature(ctx, "spatial_qrs_t_angle_deg")
    if name == "spatial_peak_qrs_t_angle_deg":
        return FeatureValue(_angle_between(_vector_array(ctx, "qrs", "max"), _vector_array(ctx, "t", "max")))
    if name in ("spatial_ventricular_gradient_dispersion", "t_wave_area_dispersion_12lead", "tpeak_tend_vector_dispersion", "vcg_t_area_dispersion_xyz"):
        if name == "spatial_ventricular_gradient_dispersion":
            # Two defects here, and the second hid the first.
            #
            # This read ``per_lead["x"]`` while the producing branch writes
            # ``svg_x_uv_ms``, so it raised KeyError on every recording where
            # the GEH metrics were computable at all: 328.358 of the 630.044
            # extracted records, practically the whole of the run's biomarker
            # failure count, all under this one name.
            #
            # It also computed the wrong quantity. The catalog declares
            # "standard deviation or MAD of beat-level SVG vectors"; taking
            # the standard deviation across the three spatial components of a
            # single vector is not a dispersion of anything -- it changes when
            # the coordinate frame is rotated, which no physiological quantity
            # may do.
            vectors = _beat_area_vectors_uv_ms(ctx, "qrst")
            scatter = _vector_scatter(vectors)
            if scatter is None:
                return FeatureValue(
                    None,
                    0.0,
                    notes="not_evaluable: fewer than two delineated QRST beats",
                )
            return FeatureValue(
                scatter,
                per_lead={"beat_count": float(len(vectors))},
                notes="beat-to-beat SVG scatter; sqrt(trace(cov)), uV*ms",
            )
        if name == "t_wave_area_dispersion_12lead":
            if ctx.ecg_12lead is None:
                return FeatureValue(None, 0.0)
            bounds = ctx.bounds("t")
            if bounds is None:
                return FeatureValue(None, 0.0)
            area = _trapz(ctx.ecg_12lead[bounds[0]:bounds[1] + 1], 1.0 / ctx.fs, axis=0)
            return FeatureValue(float(np.std(area) * 1000.0))
        if name == "tpeak_tend_vector_dispersion":
            # Same defect as the SVG dispersion above, without the KeyError:
            # the catalog says "beat-to-beat variation of the vector from T
            # peak to T end" and the code took the standard deviation over the
            # three components of one such vector. It returned a number, which
            # is why nothing noticed.
            vectors = _beat_tpeak_tend_vectors(ctx)
            scatter = _vector_scatter(vectors)
            if scatter is None:
                return FeatureValue(
                    None,
                    0.0,
                    notes="not_evaluable: fewer than two beats with T peak and T end",
                )
            return FeatureValue(
                scatter,
                per_lead={"beat_count": float(len(vectors))},
                notes="beat-to-beat Tpeak-Tend vector scatter; sqrt(trace(cov)), uV",
            )
        # vcg_t_area_dispersion_xyz. Coordinate-wise by declaration -- the
        # catalog asks for "standard deviation of T-area components" -- so
        # unlike the two above it is doing what it says. It is still frame
        # dependent, and ``t_area_vector_scatter`` is the invariant sibling
        # added beside it rather than a silent redefinition of this one.
        vec = _vector_array(ctx, "t", "area")
        return FeatureValue(None, 0.0) if vec is None else FeatureValue(float(np.std(vec)))
    return None


def _fit_superellipse_exponent(
    points: np.ndarray,
) -> tuple[Optional[float], float]:
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] < 8
        or points.shape[1] < 2
        or not np.all(np.isfinite(points))
    ):
        return None, np.inf
    centered, _, vectors = _pca(points)
    coordinates = centered @ vectors[:, :2]
    axis_scale = np.quantile(np.abs(coordinates), 0.98, axis=0)
    if (
        axis_scale.shape != (2,)
        or not np.all(np.isfinite(axis_scale))
        or np.min(axis_scale) <= EPS
    ):
        return None, np.inf

    lower_axes = np.maximum(axis_scale * 0.35, EPS)
    upper_axes = np.maximum(axis_scale * 2.5, lower_axes * 1.01)

    def residual(parameters: np.ndarray) -> np.ndarray:
        axes = np.exp(parameters[:2])
        exponent = float(parameters[2])
        normalized = np.abs(coordinates) / axes
        implicit = (
            normalized[:, 0] ** exponent
            + normalized[:, 1] ** exponent
        )
        return np.log(np.clip(implicit, EPS, None)) / exponent

    try:
        fitted = least_squares(
            residual,
            x0=np.asarray(
                [
                    np.log(axis_scale[0]),
                    np.log(axis_scale[1]),
                    2.0,
                ],
                dtype=float,
            ),
            bounds=(
                np.asarray(
                    [
                        np.log(lower_axes[0]),
                        np.log(lower_axes[1]),
                        0.6,
                    ]
                ),
                np.asarray(
                    [
                        np.log(upper_axes[0]),
                        np.log(upper_axes[1]),
                        8.0,
                    ]
                ),
            ),
            loss="soft_l1",
            f_scale=0.12,
            max_nfev=60,
        )
    except (FloatingPointError, ValueError):
        return None, np.inf
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return None, np.inf
    exponent = float(fitted.x[2])
    fit_residual = residual(fitted.x)
    fit_error = float(np.sqrt(np.mean(np.square(fit_residual))))
    if not np.isfinite(fit_error):
        return None, np.inf
    return exponent, fit_error


def _robustness_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "loop_missingness_risk":
        beat = ctx.beat
        if beat is None:
            return FeatureValue(1.0)
        required = ("p_onset", "p_offset", "r_onset", "r_offset", "t_onset", "t_offset")
        missing = sum(getattr(beat, attr, None) is None for attr in required)
        return FeatureValue(float(missing / len(required)))
    if name == "loop_fiducial_margin":
        margins = []
        for segment, marker in (("p", "p_peak"), ("qrs", "r_peak"), ("t", "t_peak")):
            bounds = ctx.bounds(segment)
            idx = ctx.marker_index(marker)
            if bounds is not None and idx is not None:
                margins.append(min(idx - bounds[0], bounds[1] - idx) / max(bounds[1] - bounds[0], 1))
        return FeatureValue(float(np.mean(margins)) if margins else None)
    if name == "loop_snr_proxy":
        qrs = ctx.vcg_segment("qrs")
        pr = ctx.vcg_segment("pr")
        if qrs is None:
            return FeatureValue(None, 0.0)
        signal = float(np.ptp(np.linalg.norm(qrs, axis=1)))
        noise = float(np.std(np.linalg.norm(pr, axis=1))) if pr is not None and len(pr) > 2 else EPS
        return FeatureValue(_safe_div(signal, noise))
    if name == "loop_confidence_weighted_value":
        qual = _signal_basis_feature(ctx, "fiducial_confidence")
        mag = _spatial_magnitude_feature(ctx, "spatial_magnitude_rms")
        return FeatureValue(None, 0.0) if qual is None or mag is None or mag.value is None else FeatureValue(float(mag.value * (qual.value or 0.0)))
    if name == "loop_interbeat_reproducibility":
        return _beat_variability_feature(ctx, "<seg>_beat_to_beat_*_distance")
    if name == "loop_outlier_beat_fraction":
        curves = _beat_segment_curves(ctx, "qrs")
        if len(curves) < 3:
            return FeatureValue(None, 0.0)
        median = np.median(np.stack(curves), axis=0)
        distv = np.array([float(np.sqrt(np.mean((c - median) ** 2))) for c in curves])
        q1, q3 = np.percentile(distv, [25, 75])
        return FeatureValue(float(np.mean(distv > q3 + 2.0 * (q3 - q1))))
    if name == "loop_quality_stratified_delta":
        return FeatureValue(None, 0.0, notes="Requires cohort-level high/low quality strata, not a single-record value.")
    return None


def _review_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "landmark_*_index_median":
        vals = [ctx.marker_index(m) for m in ("p_onset", "r_onset", "r_peak", "t_peak", "t_end")]
        vals = [v for v in vals if v is not None]
        return FeatureValue(float(np.median(vals)) if vals else None)
    if name == "qrs_loop_abnormal_score":
        f = _loop3d_features(ctx, "qrs")
        return FeatureValue(None, 0.0) if not f else FeatureValue(float(min(1.0, f["nonplanarity"] + f["tortuosity"] / 10.0)))
    if name == "t_loop_abnormal_score":
        f = _loop3d_features(ctx, "t")
        return FeatureValue(None, 0.0) if not f else FeatureValue(float(min(1.0, f["nonplanarity"] + f["tortuosity"] / 10.0)))
    if name == "manual_review_required":
        miss = _robustness_feature(ctx, "loop_missingness_risk")
        qrs = _review_feature(ctx, "qrs_loop_abnormal_score")
        t = _review_feature(ctx, "t_loop_abnormal_score")
        score = max(miss.value or 0.0, qrs.value or 0.0, t.value or 0.0)
        return FeatureValue(1.0 if score > 0.5 else 0.0, per_lead={"score": float(score)})
    return None


def _morph_signal_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    data = ctx.vcg_segment("qrs")
    if data is None:
        # As in ``_experimental_feature``: before any name test, so it must
        # not answer on behalf of a name it does not own.
        return _INPUT_UNAVAILABLE
    mag = np.linalg.norm(data, axis=1)
    if name == "<segment>_fragmentation_index":
        peaks = len(sp_signal.find_peaks(mag)[0]) + len(sp_signal.find_peaks(-mag)[0])
        dur = ctx.duration_s("qrs") or 0.0
        return FeatureValue(_safe_div(float(peaks), dur))
    if name == "<segment>_hilbert_amplitude/phase/frequency":
        analytic = sp_signal.hilbert(mag - np.mean(mag))
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.diff(phase) / (2.0 * np.pi) * ctx.fs
        return FeatureValue(float(np.mean(amp)), per_lead={"phase_span": float(phase[-1] - phase[0]), "frequency_mean": float(np.mean(freq)) if len(freq) else 0.0})
    if name == "<segment>_hjorth_activity/mobility/complexity":
        dx = np.diff(mag)
        ddx = np.diff(dx)
        activity = float(np.var(mag))
        mobility = float(np.sqrt(np.var(dx) / (activity + EPS)))
        complexity = float(np.sqrt(np.var(ddx) / (np.var(dx) + EPS)) / (mobility + EPS)) if len(ddx) else 0.0
        return FeatureValue(activity, per_lead={"mobility": mobility, "complexity": complexity})
    if name == "<segment>_katz_fractal_dimension":
        n = len(mag)
        length = np.sum(np.abs(np.diff(mag)))
        d = np.max(np.abs(mag - mag[0]))
        return FeatureValue(float(np.log10(n) / (np.log10(d / (length + EPS)) + np.log10(n) + EPS)) if n > 1 else None)
    if name == "<segment>_magnitude_mean/std/rms":
        return FeatureValue(float(np.mean(mag)), per_lead={"std": float(np.std(mag)), "rms": float(np.sqrt(np.mean(mag ** 2)))})
    if name == "<segment>_notching_index":
        second = np.diff(mag, n=2)
        return FeatureValue(float(len(sp_signal.find_peaks(np.abs(second))[0])))
    if name == "<segment>_permutation_entropy_order3":
        if len(mag) < 3:
            return FeatureValue(None, 0.0)
        patterns = []
        for i in range(len(mag) - 2):
            patterns.append(tuple(np.argsort(mag[i:i + 3])))
        _, counts = np.unique(patterns, return_counts=True, axis=0)
        p = counts / np.sum(counts)
        return FeatureValue(float(-np.sum(p * np.log2(p)) / np.log2(6)))
    if name == "<segment>_total_power/centroid/bandwidth":
        f, pxx = sp_signal.welch(mag, fs=ctx.fs, nperseg=min(128, len(mag)))
        total = float(np.trapezoid(pxx, f))
        centroid = float(np.sum(f * pxx) / (np.sum(pxx) + EPS))
        bandwidth = float(np.sqrt(np.sum(((f - centroid) ** 2) * pxx) / (np.sum(pxx) + EPS)))
        return FeatureValue(total, per_lead={"centroid": centroid, "bandwidth": bandwidth})
    if name == "<segment>_wavelet_scale*_energy":
        energies = []
        for width in (2, 4, 8):
            wavelet = _ricker(min(len(mag), 10 * width), width)
            conv = np.convolve(mag - np.mean(mag), wavelet, mode="same")
            energies.append(float(np.mean(conv ** 2)))
        return FeatureValue(float(np.mean(energies)), per_lead={f"scale_{s}": e for s, e in zip((2, 4, 8), energies)})
    return None


def _template_feature(ctx: FeatureContext, name: str) -> Optional[FeatureValue]:
    if name == "external_norm/pathology_template_bank_available_flag":
        return FeatureValue(0.0, notes="No external normal/pathology template bank is configured.")
    if name == "<segment>_record_template_*_distance":
        return _beat_variability_feature(ctx, "<seg>_beat_to_beat_*_distance")
    if name == "qrs_internal_cluster_distance_median":
        return _beat_variability_feature(ctx, "<seg>_beat_to_beat_*_distance")
    return None


class CatalogFeatureBiomarker(BiomarkerBase):
    def __init__(self, definition: BiomarkerDefinition):
        self.definition = definition

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def description(self) -> str:
        return self.definition.description

    @property
    def unit(self) -> str:
        return self.definition.unit

    @property
    def category(self) -> str:
        return self.definition.category

    @property
    def references(self) -> list[str]:
        return self.definition.references

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        ctx = _context(delineation, ecg_12lead, vcg_xyz, fs)
        dependencies = _feature_segment_dependencies(
            self.name,
            self.definition.family,
        )
        dependency_confidences = {
            dependency: ctx.segment_confidence(dependency)
            for dependency in dependencies
        }
        dependency_confidence = (
            min(dependency_confidences.values())
            if dependency_confidences
            else 1.0
        )
        failed_dependencies = tuple(
            dependency
            for dependency, confidence in dependency_confidences.items()
            if confidence < measurement_confidence_threshold(dependency)
        )
        if failed_dependencies:
            return BiomarkerResult(
                name=self.name,
                value=None,
                unit=self.unit,
                category=self.category,
                confidence=0.0,
                description=self.description,
                references=self.references,
                computation_notes=(
                    "not_evaluable: "
                    + ", ".join(failed_dependencies)
                    + " fiducials below the pre-specified confidence "
                    "threshold (0.35)"
                ),
                family=self.definition.family,
                formula=self.definition.formula,
                implementation_status="implemented",
                pathological_direction=self.definition.pathological_direction,
                pathological_note=self.definition.pathological_note,
            )
        fv = _catalog_value(ctx, self.name)
        value = fv.value
        notes = fv.notes
        if value is not None and not np.isfinite(value):
            value = None
            notes = (notes + " " if notes else "") + "Computed value was non-finite."
        return BiomarkerResult(
            name=self.name,
            value=value,
            unit=self.unit,
            category=self.category,
            confidence=(
                min(fv.confidence, dependency_confidence)
                if value is not None
                else 0.0
            ),
            per_lead=fv.per_lead,
            description=self.description,
            references=self.references,
            computation_notes=notes,
            measurement_windows=fv.measurement_windows,
            family=self.definition.family,
            formula=self.definition.formula,
            implementation_status="implemented",
            pathological_direction=self.definition.pathological_direction,
            pathological_note=self.definition.pathological_note,
        )


@lru_cache(maxsize=None)
def _feature_segment_dependencies(
    name: str,
    family: str,
) -> tuple[str, ...]:
    """Return every fiducial segment required by a scalar feature.

    Unknown catalog features fail conservatively through the representative
    beat gate. Only explicitly record-global or QC features are allowed to
    bypass fiducial gating.
    """

    normalized = name.casefold()
    family_normalized = family.casefold()
    no_fiducial_tokens = (
        "quality_score",
        "fiducial_confidence",
        "einthoven",
        "lead_to_vcg_consistency",
        "heart_rate",
        "rr_",
        "atrial_fibrillation_rr_surrogate",
        "global_",
        "vcg_axes_xyz",
        "spatial_magnitude_max",
        "spatial_magnitude_mean",
        "spatial_magnitude_median",
        "spatial_magnitude_p95",
        "spatial_magnitude_rms",
        "spatial_magnitude_integral",
        "spatial_magnitude_m_t",
        "kors_vs_pca_axis_difference",
        "axis_crossing_sequence_entropy",
        "axis_anisotropy",
        "axis_compression",
        "plane_dominance",
        "chest_lead_misplacement",
    )
    if any(token in normalized for token in no_fiducial_tokens):
        return ()
    if normalized.startswith(
        (
            "vcg_x_",
            "vcg_y_",
            "vcg_z_",
            "global_vcg_",
        )
    ):
        return ()
    if normalized in {
        "loop_missingness_risk",
        "loop_fiducial_margin",
        "loop_confidence_weighted_value",
        "manual_review_required",
    }:
        return ()

    if normalized.startswith("p_qrs_t") or "p_qrs_t_" in normalized:
        return ("p", "qrs", "t")
    if normalized.startswith("p_qrs") or "p_qrs_" in normalized:
        return ("p", "qrs")
    if normalized.startswith("p_t_") or "p_t_loop" in normalized:
        return ("p", "t")
    if normalized.startswith("pr_st") or "pr_st_" in normalized:
        return ("pr", "st")
    if normalized.startswith("pr_t") or "pr_t_" in normalized:
        return ("pr", "t")
    geh_area_features = {
        "qrs_area_to_t_area_angle_deg",
        "qrs_area_to_t_area_angle_deg_axis_relation",
        "spatial_qrs_t_angle_deg",
        "spatial_qrs_t_angle_deg_catalog",
        "spatial_ventricular_gradient_vector",
        "svg_magnitude",
        "svg_angle_to_qrs_area_deg",
        "svg_angle_to_t_area_deg",
        "svg_qrs_area_ratio",
        "svg_t_area_ratio",
        "t_area_vector",
    }
    if normalized in geh_area_features:
        # Area T is JT and therefore needs QRS offset + T offset, not T onset.
        return ("qrs", "st_t")
    if (
        "qrs_t" in normalized
        or "qrs-to-t" in normalized
        or "qrs_to_t" in normalized
        or "qrs_area_t_area" in normalized
        or "depolarization_repolarization" in normalized
        or normalized.startswith("svg_")
        or "ventricular_gradient" in family_normalized
    ):
        return ("qrs", "t")
    if "st_qrs" in normalized or "qrs_st" in normalized:
        return ("qrs", "st")
    if normalized == "acute_ischemia_vector_score":
        return ("qrs", "st", "t")
    if (
        "st_t" in normalized
        or "st_vector_to_t" in normalized
    ):
        return ("st", "t")
    if normalized in {
        "electric_systole_ms",
        "electric_diastole_ms",
        "qt_rr_hysteresis_surrogate",
        "qt_variability_index_vector",
    }:
        return ("qrst",)
    if normalized in {
        "scar_vector_index",
        "lbbb_like_vector_score",
        "rbbb_like_terminal_vector_score",
        "activation_inefficiency",
        "conduction_detour_index",
        "directional_concentration_kappa_proxy",
        "beatwise_loop_barycenter_motion",
        "ectopic_beat_vector_deviation",
        "electrical_action_density_map",
        "geodesic_direction_path_length",
        "local_plane_drift_index",
        "loop_interbeat_reproducibility",
        "loop_outlier_beat_fraction",
        "spherical_harmonic_vcg_signature",
        "vcg_phase_space_determinism",
        "vcg_phase_space_recurrence_rate",
    }:
        return ("qrs",)
    if normalized in {
        "terminal_repolarization_burden",
        "early_late_t_angle_deg",
        "microvolt_t_wave_alternans_vector",
        "vcg_t_area_dispersion_xyz",
    }:
        return ("t",)
    if normalized == "breathing_baseline_vector_swing":
        return ("p",)

    if (
        normalized.startswith("p_")
        or "p_loop" in normalized
        or "p_wave" in normalized
        or family_normalized.startswith("p-loop")
    ):
        return ("p",)
    if (
        "qrst" in normalized
        or normalized.startswith(("qt_", "qtc_", "jt_"))
        or "tpeak_tend" in normalized
    ):
        return ("qrst",)
    if normalized.startswith("st_") or "st_loop" in normalized:
        return ("st",)
    if (
        normalized.startswith("qrs_")
        or "qrs_loop" in normalized
        or normalized.endswith("_qrs")
        or "initial_qrs" in normalized
        or "terminal_qrs" in normalized
        or "early_late_qrs" in normalized
        or "hf_qrs" in normalized
        or "qrs_fragmentation" in normalized
        or "voltage_surrogate" in normalized
    ):
        return ("qrs",)
    if (
        normalized.startswith("t_")
        or "t_loop" in normalized
        or "repolarization" in normalized
        or family_normalized.startswith("t-loop")
    ):
        return ("t",)
    if normalized.startswith("pr_") or normalized == "pr_interval_ms":
        return ("pr",)
    if normalized.startswith("beat_"):
        return ("beat",)
    if (
        family_normalized.startswith("mathematische loop-morphologie")
        or normalized.startswith("loop_")
    ):
        return ("qrs",)
    return ("beat",)


def _feature_segment_dependency(
    name: str,
    family: str,
) -> Optional[str]:
    """Compatibility helper for callers that can consume one dependency."""

    dependencies = _feature_segment_dependencies(name, family)
    return dependencies[0] if dependencies else None


SUPPORTED_FAMILIES = {
    "Achsenamplituden",
    "Achsenflaechen",
    "Achsenverhaeltnisse",
    "Anisotropie/Kompression",
    "Hauptvektoren",
    "Hauptvektoren/Integrale",
    "Kinematik",
    "Kinematische Landmarken",
    "Loop-Geometrie",
    "Loop-Projektionen",
    "Klassische ECG-Projektionen",
    "Klassische ECG-Surrogate",
    "Raeumliche Magnitude",
    "Segment-Magnitudenflaechen",
    "Zeitintervalle",
    "QRS-T/Repolarisation",
    "Ventricular Gradient",
    "Segment-spezifisch",
    "Winkel/Achsenbeziehungen",
    "Pathophysiologische Surrogate",
    "P-Loop Morphologie extra",
    "T-Loop Morphologie extra",
    "P-QRS-T Loop-Beziehungen",
    "Magnitude-Verhaeltnis",
    "Mathematische Loop-Morphologie - Differentialgeometrie",
    "Mathematische Loop-Morphologie - Ellipse und Superellipse",
    "Mathematische Loop-Morphologie - Folding und Selbstannaeherung",
    "Mathematische Loop-Morphologie - Fourier-Deskriptoren",
    "Mathematische Loop-Morphologie - Konvexitaet und Sektoren",
    "Mathematische Loop-Morphologie - Rotation und Dihedralwinkel",
    "Mathematische Loop-Morphologie - Gleitende Ebenen",
    "Mathematische Loop-Morphologie - Graphfeatures",
    "Mathematische Loop-Morphologie - Zeitnormalisierte Form",
    "Mathematische Loop-Morphologie - Topologie light",
    "ST-Zeitpunkte",
    "PCA/SVD",
    "Qualitaet/Elektroden",
    "R-Wellen-Progression",
    "Zusaetzliche VCG-Idee",
    "Beat-Variabilitaet",
    "Experimentell/Topologisch",
    "Literaturbasierte Ergaenzung",
    "Loop-Robustheit und Reliabilitaet",
    "Manuelle Review",
    "Morphologische Signalverarbeitung",
    "Template/Distanz",
    "Signalbasis",
}


def _supported(definition: BiomarkerDefinition) -> bool:
    if definition.family not in SUPPORTED_FAMILIES:
        return False
    return True


def register_catalog_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for definition in candidate_definitions():
        if definition.name in existing or not _supported(definition):
            continue
        BiomarkerRegistry.register(CatalogFeatureBiomarker(definition))
        existing.add(definition.name)


__all__ = ["register_catalog_biomarkers", "CatalogFeatureBiomarker"]
