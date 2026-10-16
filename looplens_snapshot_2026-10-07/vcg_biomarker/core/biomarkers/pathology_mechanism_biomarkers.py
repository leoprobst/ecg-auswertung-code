"""Post-pilot mechanistic hypotheses: global geometry, never local tissue maps.

Round 2026-09-13, chapter 01: QRS/ST second-moment geometry. Neither
commutator nor overlap is a published clinical ischemia marker. Both are
deliberately polarity-blind; the existing signed vector angles are comparators.
Chapter 02 adds early/late QRS overlap; chapter 03 adds signed T energy
opposition, which deliberately is NOT invariant to isolated T polarity reversal.
Chapter 04 adds energy-weighted directional time moments of the T wave.
Chapter 05 adds fixed-duration QRS temporal cancellation, not hidden-source cancellation.
"""
from dataclasses import dataclass

import numpy as np

from .base import BiomarkerBase, BiomarkerDefinition, BiomarkerRegistry, BiomarkerResult
from .catalog_biomarkers import _context
from .thesis_derived_biomarkers import _beat_sample

VERSION = "1.6.0"
SOURCE = "pathology_mechanism_round_20260913"


def directional_operator(points):
    """Trapezoidal second moment, trace-normalized; no mean subtraction."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("invalid_or_nonfinite_segment")
    scale = np.max(np.abs(points))
    if scale <= 1e-12:
        raise ValueError("zero_segment_energy")
    values = points / scale
    weights = np.ones(len(values))
    weights[[0, -1]] = .5
    gram = values.T @ (values * weights[:, None])
    return gram / np.trace(gram)


def operator_pair(qrs, st):
    q, s = directional_operator(qrs), directional_operator(st)
    commutator = q @ s - s @ q
    return {"commutator": float(np.linalg.norm(commutator, "fro")),
            "overlap": float(np.trace(q @ s))}


def _metrics(ctx):
    key = (SOURCE, VERSION)
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    beat = ctx.beat
    if beat is None or not np.isfinite(ctx.fs) or ctx.fs <= 0:
        raise ValueError("missing_beat_or_invalid_sampling_rate")
    attributes = (("r_onset", "qrs"), ("r_offset", "qrs"), ("j_point", "qrs"), ("t_onset", "t"))
    bounds = [_beat_sample(beat, attr, wave) for attr, wave in attributes]
    if any(b is None for b in bounds):
        raise ValueError("uncertain_or_missing_fiducials")
    start, end, j, ton = bounds
    signal = ctx.origin_corrected_vcg()
    if signal is None or not 0 <= start < end <= j < ton < len(signal):
        raise ValueError("invalid_boundaries_or_origin")
    qrs, st = signal[start:end + 1], signal[j:ton]
    if len(qrs) < 12 or len(st) < 4 or (len(st) - 1) / ctx.fs < .008:
        raise ValueError("insufficient_segment_duration")
    if not np.isfinite(qrs).all() or not np.isfinite(st).all():
        raise ValueError("nonfinite_samples_no_gap_compression")
    # Common scaling preserves the relative gate without squaring raw amplitudes
    # (which can overflow even for finite input). Keep the absolute QRS floor.
    scale = max(float(np.max(np.abs(qrs))), float(np.max(np.abs(st))))
    if scale <= 1e-12:
        raise ValueError("ST_energy_below_exploratory_relative_quality_gate")
    qrs_rms = float(np.sqrt(np.mean((qrs / scale) ** 2)))
    st_rms = float(np.sqrt(np.mean((st / scale) ** 2)))
    if qrs_rms <= 1e-12 / scale or st_rms < .05 * qrs_rms:
        raise ValueError("ST_energy_below_exploratory_relative_quality_gate")
    metrics = operator_pair(qrs, st)
    ctx._primitive_cache[key] = metrics
    return metrics


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    component: str
    formula: str
    description: str
    chapter: str = "01_ISCHEMIA.md"
    polarity_blind: bool = True


def opposed_t_energy_fraction(qrs, t):
    """Signed T energy against the QRS area axis; not a fibrosis estimate.

    Uniform sampling; trapezoidal endpoints. Segment gain cancels, except
    the explicit absolute/relative quality gates. No time warping or centering.
    """
    arrays = [np.asarray(part, dtype=float) for part in (qrs, t)]
    for part in arrays:
        if part.ndim != 2 or part.shape[1] != 3 or len(part) < 12 or not np.isfinite(part).all():
            raise ValueError("invalid_or_nonfinite_qrs_t_segment")
    scales = [float(np.max(np.abs(part))) for part in arrays]
    if min(scales) <= 1e-12:
        raise ValueError("zero_qrs_or_t_energy")
    q, repol = [part / scale for part, scale in zip(arrays, scales)]
    q_weights, t_weights = np.ones(len(q)), np.ones(len(repol))
    q_weights[[0, -1]] = .5
    t_weights[[0, -1]] = .5
    area = q_weights @ q
    area_norm = float(np.linalg.norm(area))
    unsigned_area = float(q_weights @ np.linalg.norm(q, axis=1))
    if area_norm < .05 * unsigned_area:
        raise ValueError("unstable_qrs_area_direction")
    # Compare RMS after a common rescaling to avoid overflow in raw units.
    common = max(scales)
    q_rms = scales[0] / common * np.sqrt(np.mean(q ** 2))
    t_rms = scales[1] / common * np.sqrt(np.mean(repol ** 2))
    if t_rms < .02 * q_rms:
        raise ValueError("weak_t_relative_energy")
    projection = repol @ (area / area_norm)
    opposite = np.minimum(projection, 0.)
    denominator = float(t_weights @ np.sum(repol ** 2, axis=1))
    return float(np.clip((t_weights @ (opposite ** 2)) / denominator, 0., 1.))


def _opposed_t_energy(ctx):
    key = (SOURCE, VERSION, "opposed_t_energy")
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    if ctx.beat is None or not np.isfinite(ctx.fs) or ctx.fs <= 0:
        raise ValueError("missing_beat_or_invalid_sampling_rate")
    bounds = [_beat_sample(ctx.beat, attr, wave) for attr, wave in
              (("r_onset", "qrs"), ("r_offset", "qrs"), ("t_onset", "t"), ("t_offset", "t"))]
    signal = ctx.origin_corrected_vcg()
    if signal is None or any(b is None for b in bounds):
        raise ValueError("missing_qrs_t_boundaries_or_origin")
    start, end, ton, toff = bounds
    if not 0 <= start < end <= ton < toff < len(signal):
        raise ValueError("invalid_qrs_t_boundaries")
    if min(end - start, toff - ton) / ctx.fs < .04:
        raise ValueError("insufficient_qrs_t_duration")
    value = opposed_t_energy_fraction(signal[start:end + 1], signal[ton:toff + 1])
    ctx._primitive_cache[key] = value
    return value


def _early_late_overlap(ctx, segment="qrs"):
    key = (SOURCE, VERSION, "early_late", segment)
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    if ctx.beat is None or not np.isfinite(ctx.fs) or ctx.fs <= 0:
        raise ValueError("missing_beat_or_invalid_sampling_rate")
    if segment == "p":
        from ..delineation.base import RhythmClass, WaveStatus
        if (ctx.delineation.rhythm.rhythm_class != RhythmClass.REGULAR
                or ctx.beat.resolved_wave_status("p") != WaveStatus.PRESENT):
            raise ValueError("P_requires_regular_pattern_and_present_wave_not_AF_history")
    elif segment != "qrs":
        raise ValueError("unsupported_segment")
    prefix = "p" if segment == "p" else "r"
    start = _beat_sample(ctx.beat, prefix + "_onset", segment)
    end = _beat_sample(ctx.beat, prefix + "_offset", segment)
    signal = ctx.origin_corrected_vcg()
    if signal is None or start is None or end is None or not 0 <= start < end < len(signal):
        raise ValueError("invalid_qrs_boundaries_or_origin")
    qrs = signal[start:end + 1]
    if len(qrs) < 24 or (len(qrs) - 1) / ctx.fs < .04:
        raise ValueError("insufficient_qrs_support")
    middle = len(qrs) // 2
    first, last = qrs[:middle], qrs[middle:]
    # Every half must carry energy, otherwise trace normalization amplifies noise.
    if not np.isfinite(qrs).all():
        raise ValueError("nonfinite_qrs")
    scale = float(np.max(np.abs(qrs)))
    if scale <= 1e-12:
        raise ValueError("zero_qrs_energy")
    energy = [float(np.mean((part / scale) ** 2)) for part in (first, last)]
    if min(energy) < .0025 * max(energy):
        raise ValueError("weak_half_relative_energy")
    result = operator_pair(first, last)["overlap"]
    ctx._primitive_cache[key] = result
    return result


def direction_time_coupling(points):
    """First two weighted orthonormal time moments of vector direction.

    Amplitude-squared weights suppress zero-amplitude direction singularities.
    This is surface chronology, not local repolarization time or APD dispersion.
    No division by individual sample magnitudes is required.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 12 or not np.isfinite(points).all():
        raise ValueError("invalid_or_nonfinite_t_segment")
    scale = float(np.max(np.abs(points)))
    if scale <= 1e-12:
        raise ValueError("zero_t_energy")
    values = points / scale
    magnitude = np.linalg.norm(values, axis=1)
    quadrature = np.ones(len(values))
    quadrature[[0, -1]] = .5
    energy = quadrature * magnitude ** 2
    total = float(energy.sum())
    weights = energy / total
    if 1. / float(weights @ weights) < 8.:
        raise ValueError("insufficient_energy_weighted_support")
    phase = np.linspace(0., 1., len(values))
    centered = phase - weights @ phase
    variance = float(weights @ (centered ** 2))
    if variance <= 1e-12:
        raise ValueError("degenerate_energy_time_variance")
    linear = centered / np.sqrt(variance)
    quadratic = centered ** 2 - variance - float(weights @ (centered ** 3)) / variance * centered
    quadratic_variance = float(weights @ (quadratic ** 2))
    if quadratic_variance <= 1e-12:
        raise ValueError("degenerate_quadratic_time_basis")
    quadratic /= np.sqrt(quadratic_variance)
    # weight * unit_direction = quadrature * magnitude * values / total.
    weighted_direction = quadrature[:, None] * magnitude[:, None] * values / total
    moments = np.stack((linear, quadratic)) @ weighted_direction
    norms = np.linalg.norm(moments, axis=1)
    return {"time_linear": float(np.clip(norms[0], 0., 1.)),
            "time_quadratic": float(np.clip(norms[1], 0., 1.))}


def _t_direction_time(ctx):
    key = (SOURCE, VERSION, "t_direction_time")
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    if ctx.beat is None or not np.isfinite(ctx.fs) or ctx.fs <= 0:
        raise ValueError("missing_beat_or_invalid_sampling_rate")
    start = _beat_sample(ctx.beat, "t_onset", "t")
    end = _beat_sample(ctx.beat, "t_offset", "t")
    signal = ctx.origin_corrected_vcg()
    if signal is None or start is None or end is None or not 0 <= start < end < len(signal):
        raise ValueError("invalid_t_boundaries_or_origin")
    if (end - start) / ctx.fs < .04:
        raise ValueError("insufficient_t_duration")
    result = direction_time_coupling(signal[start:end + 1])
    ctx._primitive_cache[key] = result
    return result


def local_qrs_cancellation(points, fs, window_ms=24.):
    """Aggregate temporal cancellation over all complete sample-start windows.

    Prefix trapezoids give O(n) complexity. Fractional endpoints integrate
    piecewise-linear XYZ and piecewise-linear sampled magnitude separately.
    This does not reconstruct simultaneously cancelling myocardial sources.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 12 or not np.isfinite(points).all():
        raise ValueError("invalid_or_nonfinite_qrs_segment")
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("invalid_time_scale")
    span = fs * (window_ms / 1000.)
    if not np.isfinite(span) or span < 4 or span > len(points) - 3:
        raise ValueError("insufficient_window_sampling_or_support")
    if abs(span - round(span)) < 1e-10:
        span = float(round(span))
    scale = float(np.max(np.abs(points)))
    if scale <= 1e-12:
        raise ValueError("zero_qrs_energy")
    values = points / scale
    sampled = np.column_stack((values, np.linalg.norm(values, axis=1)))
    prefix = np.vstack((np.zeros(4), np.cumsum((sampled[:-1] + sampled[1:]) * .5, axis=0)))
    starts = np.arange(int(np.floor(len(points) - 1 - span)) + 1)
    endpoints = starts + span
    lower = np.floor(endpoints).astype(int)
    fraction = (endpoints - lower)[:, None]
    upper = np.minimum(lower + 1, len(points) - 1)
    end_integrals = prefix[lower] + fraction * sampled[lower] + .5 * fraction ** 2 * (sampled[upper] - sampled[lower])
    integrals = end_integrals - prefix[starts]
    denominator = float(integrals[:, 3].sum())
    if denominator <= 1e-12:
        raise ValueError("zero_integrated_qrs_magnitude")
    result = 1. - float(np.linalg.norm(integrals[:, :3], axis=1).sum()) / denominator
    return float(np.clip(result, 0., 1.))


def _local_qrs_cancellation(ctx):
    key = (SOURCE, VERSION, "qrs_local24ms")
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    if ctx.beat is None or not np.isfinite(ctx.fs) or ctx.fs <= 0:
        raise ValueError("missing_beat_or_invalid_sampling_rate")
    start = _beat_sample(ctx.beat, "r_onset", "qrs")
    end = _beat_sample(ctx.beat, "r_offset", "qrs")
    signal = ctx.origin_corrected_vcg()
    if signal is None or start is None or end is None or not 0 <= start < end < len(signal):
        raise ValueError("invalid_qrs_boundaries_or_origin")
    if (end - start) / ctx.fs < .04:
        raise ValueError("insufficient_qrs_duration")
    result = local_qrs_cancellation(signal[start:end + 1], ctx.fs)
    ctx._primitive_cache[key] = result
    return result


def t_energy_concentration(points):
    """Inverse temporal participation of uncentered vector energy, >= 1.

    Unlike centered radial kurtosis, this measures temporal energy occupancy.
    Scaling before powers avoids overflow. No missing-sample compression.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 12 or not np.isfinite(x).all():
        raise ValueError("invalid_t_segment")
    scale = float(np.max(np.abs(x)))
    if scale <= 1e-12:
        raise ValueError("zero_t_energy")
    energy = np.sum((x / scale) ** 2, axis=1)
    w = np.ones(len(x))
    w[[0, -1]] = .5
    w /= w.sum()
    return float((w @ energy ** 2) / (w @ energy) ** 2)


def _t_energy_concentration(ctx):
    key = (SOURCE, VERSION, "t_energy_concentration")
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    if ctx.beat is None or not np.isfinite(ctx.fs) or ctx.fs <= 0:
        raise ValueError("missing_beat_or_invalid_sampling_rate")
    start = _beat_sample(ctx.beat, "t_onset", "t")
    end = _beat_sample(ctx.beat, "t_offset", "t")
    signal = ctx.origin_corrected_vcg()
    if signal is None or start is None or end is None or not 0 <= start < end < len(signal):
        raise ValueError("invalid_t_boundaries_or_origin")
    if (end - start) / ctx.fs < .04:
        raise ValueError("insufficient_t_duration")
    value = t_energy_concentration(signal[start:end + 1])
    ctx._primitive_cache[key] = value
    return value


SPECS = (
    MechanismSpec("p_early_late_directional_operator_overlap", "p_early_late",
                  "tr(P_early P_late); trace-normalized second moments of nonoverlapping P-wave halves",
                  "Early/late atrial energy-axis organization in regular technical rhythm; not LA fibrosis, sinus diagnosis or AF prediction.",
                  "15_ATRIAL_CARDIOMYOPATHY.md"),
    MechanismSpec("t_energy_temporal_concentration", "t_energy_concentration",
                  "duration * integral ||T||^4 dt / (integral ||T||^2 dt)^2; trapezoidal quadrature",
                  "Temporal concentration of T-vector energy; exploratory peaking morphology, not potassium concentration or a diagnostic threshold.",
                  "PRIORITY_MARKER_IMPLEMENTATION.md"),
    MechanismSpec("qrs_local24ms_cancellation_fraction", "local_qrs_cancellation",
                  "1-sum_s ||integral_[s,s+24ms] QRS(t)dt|| / sum_s integral_[s,s+24ms] ||QRS(t)||dt; complete sample-start windows",
                  "Temporal QRS cancellation within fixed 24 ms windows; exploratory directional organization, not amyloid burden or simultaneously hidden myocardial sources.",
                  "05_AMYLOID_INFILTRATION.md"),
    MechanismSpec("t_direction_time_linear_coupling", "time_linear",
                  "||E_w[p1(u) n(u)]||; w proportional to ||T||^2 dt; p1=(u-E_w[u])/SD_w(u); n=T/||T||",
                  "Linear energy-weighted temporal segregation of T-vector directions; exploratory chronology, not regional APD.",
                  "04_HCM.md"),
    MechanismSpec("t_direction_time_quadratic_coupling", "time_quadratic",
                  "||E_w[p2(u) n(u)]||; p2 is degree-two time polynomial orthonormal to 1,p1 under T-energy weights",
                  "Quadratic energy-weighted T-direction chronology; captures center-versus-ends patterns missed by the linear moment, not HCM diagnosis.",
                  "04_HCM.md"),
    MechanismSpec("qrs_t_opposed_energy_fraction", "opposed_t_energy",
                  "integral min(T(t).qhat,0)^2 dt / integral ||T(t)||^2 dt; qhat=integral QRS/||integral QRS||",
                  "T-wave energy opposing the QRS area axis; exploratory repolarization distribution, not hypertrophy or fibrosis diagnosis.",
                  "03_HYPERTENSIVE_LVH.md", False),
    MechanismSpec("qrs_early_late_directional_operator_overlap", "early_late",
                  "tr(Q_early Q_late); trace-normalized second moments of nonoverlapping temporal QRS halves",
                  "Early/late QRS energy-axis overlap; exploratory global morphology, not scar location or a reentry circuit.",
                  "02_SCAR_REENTRY.md"),
    MechanismSpec("qrs_st_directional_operator_commutator", "commutator",
                  "||Q S - S Q||_F; Q=int(q q^T dt)/int(||q||^2 dt), S analogously for J..T_on",
                  "Non-commutation of global QRS/ST second-moment operators; own exploratory hypothesis, not local injury severity."),
    MechanismSpec("qrs_st_directional_operator_overlap", "overlap", "tr(Q S)",
                  "Overlap of normalized global QRS/ST second moments; distinguishes aligned and orthogonal commuting operators, not primary/secondary ischemia."),
)


class MechanismBiomarker(BiomarkerBase):
    def __init__(self, spec):
        self.spec = spec

    @property
    def name(self):
        return self.spec.name

    @property
    def unit(self):
        return "dimensionless"

    @property
    def category(self):
        return "vcg"

    @property
    def description(self):
        return self.spec.description

    def compute(self, delineation, ecg_12lead=None, vcg_xyz=None, fs=500):
        ctx = _context(delineation, ecg_12lead, vcg_xyz, fs)
        reason = ""
        try:
            if self.spec.component == "t_energy_concentration":
                value = _t_energy_concentration(ctx)
            elif self.spec.component == "local_qrs_cancellation":
                value = _local_qrs_cancellation(ctx)
            elif self.spec.component in ("time_linear", "time_quadratic"):
                value = _t_direction_time(ctx)[self.spec.component]
            elif self.spec.component == "opposed_t_energy":
                value = _opposed_t_energy(ctx)
            elif self.spec.component == "early_late":
                value = _early_late_overlap(ctx)
            elif self.spec.component == "p_early_late":
                value = _early_late_overlap(ctx, "p")
            else:
                value = _metrics(ctx)[self.spec.component]
        except ValueError as exc:
            value, reason = None, f"not_evaluable: {exc}; "
        return BiomarkerResult(name=self.name, value=value, unit=self.unit, category=self.category,
            confidence=0.0 if value is None else 1.0, description=self.description,
            formula=self.spec.formula, family="Pathologie-Mechanismushypothesen",
            computation_notes=reason + f"version={VERSION}; post_outcome_hypothesis=True; absolute_snr_gate=not_validated; polarity_blind={self.spec.polarity_blind}; clinical_claim_allowed=False")


def register_pathology_mechanism_biomarkers():
    for spec in SPECS:
        if spec.name in BiomarkerRegistry.available():
            if not isinstance(BiomarkerRegistry.get(spec.name), MechanismBiomarker):
                raise RuntimeError(f"Marker collision: {spec.name}")
            continue
        BiomarkerRegistry.register_definition(BiomarkerDefinition(
            name=spec.name, family="Pathologie-Mechanismushypothesen", description=spec.description,
            formula=spec.formula, unit="dimensionless", source=SOURCE,
            references=[f"research_pathology_round_20260913/{spec.chapter}: own mechanistic hypothesis, not a published clinical marker"],
            pathological_note="Exploratory post-pilot hypothesis. Global surface geometry, not tissue localization. No clinical cutoff.",
        ))
        BiomarkerRegistry.register(MechanismBiomarker(spec))
