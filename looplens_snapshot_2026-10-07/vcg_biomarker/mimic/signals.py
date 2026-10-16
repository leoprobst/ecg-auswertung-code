from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb
from scipy import signal as scipy_signal
from scipy.linalg import orthogonal_procrustes

from ..core.data_loader import LEAD_NAMES_12
from ..core.preprocessing import prepare_signal_paths, resample_signal
from ..core.vcg_transform import VCGResult, VCGTransformRegistry


class WaveformValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SignalQualityReport:
    usable: bool
    tier: str
    score: float
    duration_seconds: float
    sampling_rate_hz: float
    finite_fraction: float
    flatline_lead_count: int
    clipping_fraction: float
    high_frequency_noise_ratio: float
    baseline_wander_ratio: float
    powerline_ratio: float
    powerline_ratio_50hz: float
    powerline_ratio_60hz: float
    detected_powerline_hz: Optional[float]
    einthoven_residual_ratio: float
    amplitude_outlier_lead_count: int
    limb_lead_reversal_suspected: bool
    precordial_misplacement_suspected: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SignalRepairReport:
    """What, if anything, was filled in before the record was judged."""

    repaired_samples: int
    repaired_leads: int
    longest_gap_ms: float
    repaired_fraction: float
    status: str

    def to_dict(self) -> dict:
        return {
            "signal_repair_status": self.status,
            "signal_repair_samples": int(self.repaired_samples),
            "signal_repair_leads": int(self.repaired_leads),
            "signal_repair_longest_gap_ms": float(self.longest_gap_ms),
            "signal_repair_fraction": float(self.repaired_fraction),
        }


@dataclass(frozen=True)
class WaveformLoadResult:
    signal_millivolts: np.ndarray
    sampling_rate: int
    original_sampling_rate: float
    lead_names: tuple[str, ...]
    source_units: tuple[str, ...]
    record_path: str
    record_datetime: Optional[str]
    quality: SignalQualityReport
    repair: "SignalRepairReport" = None  # type: ignore[assignment]


class MimicWaveformLoader:
    def __init__(self, target_sampling_rate: int = 500):
        self.target_sampling_rate = int(target_sampling_rate)

    def load(
        self,
        record_path: Path | str,
        *,
        expected_subject_id: Optional[int] = None,
        expected_study_id: Optional[int] = None,
    ) -> WaveformLoadResult:
        path = Path(record_path)
        path_without_suffix = path.with_suffix("") if path.suffix in {".hea", ".dat"} else path
        if not path_without_suffix.with_suffix(".hea").exists():
            raise FileNotFoundError(
                f"WFDB header is missing: {path_without_suffix.with_suffix('.hea')}"
            )
        record = wfdb.rdrecord(str(path_without_suffix))
        if record.p_signal is None:
            raise WaveformValidationError(
                f"WFDB record has no physical signal: {path_without_suffix}"
            )
        signal = np.asarray(record.p_signal, dtype=float)
        if signal.ndim != 2:
            raise WaveformValidationError(
                f"Expected a two-dimensional ECG, got shape {signal.shape}"
            )

        indices = _strict_lead_indices(record.sig_name)
        source_units = tuple(str(record.units[index] or "") for index in indices)
        ordered = signal[:, indices]
        ordered = _convert_to_millivolts(ordered, source_units)
        original_fs = float(record.fs)
        if original_fs <= 0:
            raise WaveformValidationError(f"Invalid sampling rate: {record.fs}")
        if not np.isclose(
            original_fs,
            float(self.target_sampling_rate),
            rtol=0.0,
            atol=1e-9,
        ):
            ordered = resample_signal(
                ordered,
                original_fs,
                self.target_sampling_rate,
            )
        fs = self.target_sampling_rate
        # Repair before judging: a converter dropout of a few milliseconds is
        # not a reason to discard a tracing, and it was the single largest one.
        ordered, repair = repair_short_gaps(ordered, fs)
        quality = assess_signal_quality(ordered, fs)
        record_datetime = _record_datetime(record)

        if expected_study_id is not None:
            stem = path_without_suffix.name
            if stem.isdigit() and int(stem) != int(expected_study_id):
                raise WaveformValidationError(
                    f"Waveform study mismatch: expected {expected_study_id}, got {stem}"
                )
        if expected_subject_id is not None:
            subject_token = f"p{int(expected_subject_id)}"
            if subject_token not in {
                part.casefold() for part in path_without_suffix.parts
            }:
                raise WaveformValidationError(
                    f"Waveform path does not contain expected subject {expected_subject_id}"
                )

        return WaveformLoadResult(
            signal_millivolts=ordered,
            sampling_rate=fs,
            original_sampling_rate=original_fs,
            lead_names=tuple(LEAD_NAMES_12),
            source_units=source_units,
            record_path=str(path_without_suffix),
            record_datetime=record_datetime,
            quality=quality,
            repair=repair,
        )

    def prepare_paths(
        self,
        loaded: WaveformLoadResult,
        powerline_hz: float = 60.0,
        *,
        device_filtering: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        return prepare_signal_paths(
            loaded.signal_millivolts,
            loaded.sampling_rate,
            powerline=powerline_hz,
            notch=not device_notched_at(device_filtering, powerline_hz),
        )


def device_notched_at(filtering: Optional[str], powerline_hz: float) -> bool:
    """Whether the recorder states a notch at exactly this frequency.

    MIMIC-IV-ECG ``filtering``: "60 Hz notch Baseline filter" (677,415),
    "<not specified>" (107,398), "Baseline filter" (12,505), "50 Hz notch
    Baseline filter" (2,717). Only the first case is already notched at the
    60-Hz mains; unknown or different statements keep the software notch.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*hz\s+notch", str(filtering or "").lower())
    return bool(match) and abs(float(match.group(1)) - float(powerline_hz)) < 0.5


# The longest run of missing samples that may be filled in. A P wave is about
# 100 ms and the narrowest QRS about 70; at 40 ms an interpolated stretch
# cannot contain a wave, so the repair can restore a dropout but cannot invent
# a deflection. Anything longer is a real loss of signal and still fails.
MAXIMUM_REPAIRABLE_GAP_MS = 40.0

# And a ceiling on the total, so a record riddled with short holes is not
# reassembled out of straight lines: half a per cent of a ten-second strip is
# 50 ms spread over twelve leads.
MAXIMUM_REPAIRED_FRACTION = 0.005


def repair_short_gaps(
    ecg_millivolts: np.ndarray,
    fs: int,
    *,
    maximum_gap_ms: float = MAXIMUM_REPAIRABLE_GAP_MS,
    maximum_fraction: float = MAXIMUM_REPAIRED_FRACTION,
) -> tuple[np.ndarray, SignalRepairReport]:
    """Fill short runs of missing samples, or leave the record alone.

    The quality gate rejects a recording when a single sample is not finite
    (``finite_fraction < 1.0``). In the 2026-08-20 run that discarded **7.108**
    of the 10.060 recordings excluded at extraction -- by far the largest
    single reason, and most of them are a converter dropout of a few
    milliseconds rather than an unusable tracing.

    Interpolating across a short gap is the ordinary remedy, but it is also a
    way to invent physiology, so the two limits above are declared rather than
    tuned, the result is marked, and a record that needs more than they allow
    is still rejected. A repaired record is never silently identical to one
    that arrived intact: ``signal_repair_status`` says which it is.
    """

    ecg = np.asarray(ecg_millivolts, dtype=float)
    if ecg.ndim != 2:
        raise WaveformValidationError(
            f"Signal repair requires a two-dimensional ECG, got {ecg.shape}"
        )
    missing = ~np.isfinite(ecg)
    if not missing.any():
        return ecg, SignalRepairReport(0, 0, 0.0, 0.0, "not_needed")

    maximum_gap_samples = int(round(float(maximum_gap_ms) * float(fs) / 1000.0))
    repaired = ecg.copy()
    repaired_samples = 0
    repaired_leads = 0
    longest_gap = 0
    for lead in range(ecg.shape[1]):
        column = repaired[:, lead]
        holes = ~np.isfinite(column)
        if not holes.any():
            continue
        # Runs of consecutive missing samples.
        edges = np.flatnonzero(np.diff(holes.astype(np.int8)))
        starts = ([0] if holes[0] else []) + list(edges[holes[edges + 1]] + 1)
        ends = list(edges[~holes[edges + 1]] + 1) + (
            [len(column)] if holes[-1] else []
        )
        finite_positions = np.flatnonzero(~holes)
        if finite_positions.size < 2:
            longest_gap = max(longest_gap, int(holes.sum()))
            continue
        lead_repaired = 0
        for start, end in zip(starts, ends):
            length = int(end - start)
            longest_gap = max(longest_gap, length)
            if length > maximum_gap_samples:
                continue
            # Never extrapolate: a gap at either edge has no second anchor.
            if start == 0 or end >= len(column):
                continue
            column[start:end] = np.interp(
                np.arange(start, end, dtype=float),
                finite_positions.astype(float),
                column[finite_positions],
            )
            lead_repaired += length
        if lead_repaired:
            repaired_leads += 1
            repaired_samples += lead_repaired

    fraction = float(repaired_samples) / float(ecg.size) if ecg.size else 0.0
    longest_gap_ms = float(longest_gap) * 1000.0 / float(fs)
    if fraction > float(maximum_fraction):
        return ecg, SignalRepairReport(
            0,
            0,
            longest_gap_ms,
            fraction,
            "refused_too_much_missing",
        )
    if np.isfinite(repaired).all():
        status = "repaired" if repaired_samples else "not_needed"
    else:
        status = "partially_repaired"
    return repaired, SignalRepairReport(
        repaired_samples,
        repaired_leads,
        longest_gap_ms,
        fraction,
        status,
    )


def assess_signal_quality(
    ecg_millivolts: np.ndarray,
    fs: int,
    *,
    expected_powerline_hz: Optional[float] = None,
    minimum_duration_seconds: float = 8.0,
) -> SignalQualityReport:
    ecg = np.asarray(ecg_millivolts, dtype=float)
    reasons: list[str] = []
    if ecg.ndim != 2 or ecg.shape[1] != 12:
        raise WaveformValidationError(
            f"Signal quality requires 12 leads, got shape {ecg.shape}"
        )
    duration = ecg.shape[0] / float(fs)
    finite_fraction = float(np.mean(np.isfinite(ecg)))
    finite_ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

    lead_std = np.std(finite_ecg, axis=0)
    flatline_leads = int(np.sum(lead_std < 0.01))
    amplitude_outliers = int(
        np.sum((np.ptp(finite_ecg, axis=0) < 0.05) | (np.ptp(finite_ecg, axis=0) > 20.0))
    )
    clipping_fraction = _clipping_fraction(finite_ecg)
    (
        high_frequency_ratio,
        baseline_ratio,
        powerline_ratio_50,
        powerline_ratio_60,
    ) = _spectral_quality(
        finite_ecg,
        fs,
    )
    detected_powerline: Optional[float]
    if expected_powerline_hz is not None:
        powerline_ratio = (
            powerline_ratio_50
            if abs(float(expected_powerline_hz) - 50.0)
            <= abs(float(expected_powerline_hz) - 60.0)
            else powerline_ratio_60
        )
        detected_powerline = float(expected_powerline_hz)
    else:
        powerline_ratio = max(powerline_ratio_50, powerline_ratio_60)
        detected_powerline = (
            50.0
            if powerline_ratio_50 > powerline_ratio_60
            else 60.0
            if powerline_ratio_60 > powerline_ratio_50
            else None
        )
    (
        limb_reversal_suspected,
        precordial_misplacement_suspected,
    ) = _lead_configuration_qc(finite_ecg, fs)
    einthoven = finite_ecg[:, 0] + finite_ecg[:, 2] - finite_ecg[:, 1]
    limb_scale = np.sqrt(
        np.mean(finite_ecg[:, :3] ** 2)
    )
    einthoven_ratio = float(
        np.sqrt(np.mean(einthoven**2)) / max(limb_scale, 1e-9)
    )

    if duration < float(minimum_duration_seconds):
        reasons.append(
            f"duration_below_{float(minimum_duration_seconds):g}_seconds"
        )
    if finite_fraction < 1.0:
        reasons.append("non_finite_samples")
    if flatline_leads:
        reasons.append(f"flatline_leads={flatline_leads}")
    if clipping_fraction > 0.01:
        reasons.append("clipping")
    if high_frequency_ratio > 0.25:
        reasons.append("high_frequency_noise")
    if baseline_ratio > 0.35:
        reasons.append("baseline_wander")
    if powerline_ratio > 0.08:
        reasons.append(
            f"powerline_interference_{int(detected_powerline or 0)}hz"
        )
    if einthoven_ratio > 0.25:
        reasons.append("limb_lead_inconsistency")
    if amplitude_outliers:
        reasons.append(f"amplitude_outlier_leads={amplitude_outliers}")
    if limb_reversal_suspected:
        reasons.append("limb_lead_reversal_or_dextrocardia_suspected")
    if precordial_misplacement_suspected:
        reasons.append("precordial_misplacement_or_reverse_progression_suspected")

    penalties = (
        min(0.35, flatline_leads * 0.12)
        + min(0.20, clipping_fraction * 5)
        + min(0.15, high_frequency_ratio * 0.5)
        + min(0.10, baseline_ratio * 0.25)
        + min(0.10, powerline_ratio * 0.75)
        + min(0.15, einthoven_ratio * 0.4)
        + (0.10 if limb_reversal_suspected else 0.0)
        + (0.05 if precordial_misplacement_suspected else 0.0)
        + (0.20 if finite_fraction < 1.0 else 0.0)
        + (0.20 if duration < float(minimum_duration_seconds) else 0.0)
    )
    score = float(np.clip(1.0 - penalties, 0.0, 1.0))
    usable = (
        duration >= float(minimum_duration_seconds)
        and finite_fraction == 1.0
        and flatline_leads == 0
        and amplitude_outliers <= 1
        and score >= 0.50
    )
    tier = "high" if score >= 0.85 else "medium" if score >= 0.65 else "low"
    if not usable:
        tier = "unusable"

    return SignalQualityReport(
        usable=usable,
        tier=tier,
        score=score,
        duration_seconds=duration,
        sampling_rate_hz=float(fs),
        finite_fraction=finite_fraction,
        flatline_lead_count=flatline_leads,
        clipping_fraction=clipping_fraction,
        high_frequency_noise_ratio=high_frequency_ratio,
        baseline_wander_ratio=baseline_ratio,
        powerline_ratio=powerline_ratio,
        powerline_ratio_50hz=powerline_ratio_50,
        powerline_ratio_60hz=powerline_ratio_60,
        detected_powerline_hz=detected_powerline,
        einthoven_residual_ratio=einthoven_ratio,
        amplitude_outlier_lead_count=amplitude_outliers,
        limb_lead_reversal_suspected=limb_reversal_suspected,
        precordial_misplacement_suspected=(
            precordial_misplacement_suspected
        ),
        reasons=tuple(reasons),
    )


def compute_vcg_sensitivity_transforms(
    morphology_signal: np.ndarray,
    methods: tuple[str, ...] = ("inverse_dower", "pca_kors_aligned"),
    *,
    kors_result: Optional[VCGResult] = None,
) -> dict[str, VCGResult]:
    supported = {"inverse_dower", "pca_kors_aligned"}
    unknown = set(methods) - supported
    if unknown:
        raise ValueError(
            f"Unsupported VCG sensitivity transform(s): {sorted(unknown)}"
        )
    results: dict[str, VCGResult] = {}
    if "inverse_dower" in methods:
        results["inverse_dower"] = VCGTransformRegistry.transform(
            morphology_signal,
            "inverse_dower",
        )
    if "pca_kors_aligned" in methods:
        # The production extraction path already needs Kors for the primary
        # feature panel.  Reuse that exact transform instead of repeating the
        # same matrix multiplication for every sensitivity pass.
        kors = kors_result
        if kors is None:
            kors = VCGTransformRegistry.transform(morphology_signal, "kors")
        elif (
            kors.method_name != "kors"
            or kors.xyz.shape[0] != np.asarray(morphology_signal).shape[0]
        ):
            raise ValueError(
                "kors_result must be the Kors transform of morphology_signal"
            )
        pca_xyz = _kors_aligned_pca(morphology_signal, kors.xyz)
        results["pca_kors_aligned"] = VCGResult(
            vx=pca_xyz[:, 0],
            vy=pca_xyz[:, 1],
            vz=pca_xyz[:, 2],
            method_name="pca_kors_aligned",
            channels_used=list(LEAD_NAMES_12),
            references=[
                "PCA sensitivity transform aligned to Kors by orthogonal "
                "Procrustes; directional PCA features remain sensitivity-only."
            ],
        )
    return results


def _strict_lead_indices(signal_names: list[str]) -> list[int]:
    normalized = [_normalize_lead_name(name) for name in signal_names]
    duplicates = {
        name for name in normalized if normalized.count(name) > 1
    }
    if duplicates:
        raise WaveformValidationError(
            f"Duplicate lead names: {sorted(duplicates)}"
        )
    missing = [
        lead for lead in LEAD_NAMES_12 if _normalize_lead_name(lead) not in normalized
    ]
    if missing:
        raise WaveformValidationError(
            f"Missing required ECG leads: {', '.join(missing)}"
        )
    return [
        normalized.index(_normalize_lead_name(lead))
        for lead in LEAD_NAMES_12
    ]


def _normalize_lead_name(name: str) -> str:
    text = str(name).strip().replace(" ", "").casefold()
    aliases = {
        "avr": "avr",
        "avl": "avl",
        "avf": "avf",
    }
    return aliases.get(text, text)


def _convert_to_millivolts(
    signal: np.ndarray,
    units: tuple[str, ...],
) -> np.ndarray:
    output = np.asarray(signal, dtype=float).copy()
    for column, unit in enumerate(units):
        normalized = (
            unit.strip()
            .replace("\u00b5", "u")
            .replace("\u03bc", "u")
            .casefold()
        )
        if normalized in {"mv", "millivolt", "millivolts"}:
            factor = 1.0
        elif normalized in {"v", "volt", "volts"}:
            factor = 1000.0
        elif normalized in {"uv", "microvolt", "microvolts"}:
            factor = 0.001
        else:
            raise WaveformValidationError(
                f"Unsupported or missing physical unit for lead "
                f"{LEAD_NAMES_12[column]}: {unit!r}"
            )
        output[:, column] *= factor
    return output


def _spectral_quality(
    ecg: np.ndarray,
    fs: int,
) -> tuple[float, float, float, float]:
    nperseg = min(ecg.shape[0], max(256, fs * 2))
    frequencies, psd = scipy_signal.welch(
        ecg,
        fs=fs,
        axis=0,
        nperseg=nperseg,
    )
    total = np.trapezoid(psd, frequencies, axis=0)
    total = np.maximum(total, 1e-12)

    def band_ratio(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies <= high)
        if not np.any(mask):
            return 0.0
        power = np.trapezoid(psd[mask], frequencies[mask], axis=0)
        return float(np.median(power / total))

    high_frequency = band_ratio(40.0, min(fs / 2 - 1, 150.0))
    baseline = band_ratio(0.0, 0.5)
    powerline_50 = band_ratio(48.5, 51.5) if fs / 2 > 52 else 0.0
    powerline_60 = band_ratio(58.5, 61.5) if fs / 2 > 62 else 0.0
    return high_frequency, baseline, powerline_50, powerline_60


def _lead_configuration_qc(ecg: np.ndarray, fs: int) -> tuple[bool, bool]:
    """Conservative physiological flags; never used as a diagnostic label."""

    if ecg.shape[0] < max(20, int(fs)):
        return False, False
    nyquist = 0.5 * float(fs)
    sos = scipy_signal.butter(
        3,
        [5.0 / nyquist, min(40.0, 0.90 * nyquist) / nyquist],
        btype="bandpass",
        output="sos",
    )
    qrs_band = scipy_signal.sosfiltfilt(sos, ecg, axis=0)
    upper = np.percentile(qrs_band, 99.0, axis=0)
    lower = -np.percentile(qrs_band, 1.0, axis=0)
    scale = np.maximum(upper + lower, 1e-9)
    polarity = (upper - lower) / scale
    # Lead I predominantly negative together with upright aVR is the classic
    # high-specificity pattern for RA/LA reversal, but can also be true
    # dextrocardia; the reason string therefore preserves both explanations.
    limb_reversal = bool(polarity[0] < -0.25 and polarity[3] > 0.25)

    r_fraction = upper / scale
    early = float(np.median(r_fraction[6:8]))
    late = float(np.median(r_fraction[10:12]))
    precordial_reverse = bool(early - late > 0.30)
    return limb_reversal, precordial_reverse


def _clipping_fraction(ecg: np.ndarray) -> float:
    fractions: list[float] = []
    for column in range(ecg.shape[1]):
        lead = ecg[:, column]
        minimum: float = float(np.min(lead))
        maximum: float = float(np.max(lead))
        fractions.append(
            float(max(np.mean(lead == minimum), np.mean(lead == maximum)))
        )
    return float(max(fractions))


def _record_datetime(record) -> Optional[str]:
    date = getattr(record, "base_date", None)
    time = getattr(record, "base_time", None)
    if date is None and time is None:
        return None
    return f"{date or ''} {time or ''}".strip()


def _kors_aligned_pca(
    ecg_12lead: np.ndarray,
    kors_xyz: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(ecg_12lead, dtype=float)
    centered = centered - np.mean(centered, axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :3] * singular_values[:3]
    target = kors_xyz - np.mean(kors_xyz, axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(scores, target)
    aligned = scores @ rotation
    aligned_rms = np.sqrt(np.mean(aligned**2))
    target_rms = np.sqrt(np.mean(target**2))
    if aligned_rms > 0 and target_rms > 0:
        aligned *= target_rms / aligned_rms
    return aligned
