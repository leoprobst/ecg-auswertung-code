from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BiomarkerBase, BiomarkerResult
from ..delineation.base import RecordDelineation, BeatDelineation
from ..delineation.base import (
    fiducial_confidence_threshold,
    measurement_confidence_threshold,
)


def _corrected_qt_from_paired_beats(
    delineation: RecordDelineation,
    method: str,
) -> tuple[Optional[float], float, str]:
    measurements = delineation.qt_rr_measurements()
    required = 5 if delineation.rhythm.requires_beatwise_rate_correction else 1
    if len(measurements) < required:
        return (
            None,
            0.0,
            f"QTc requires {required} valid beatwise QT-RR pairs; "
            f"{len(measurements)} available",
        )
    qt = np.asarray([item.qt_ms for item in measurements], dtype=float)
    rr = np.asarray(
        [item.preceding_rr_seconds for item in measurements],
        dtype=float,
    )
    if method == "bazett":
        corrected = qt / np.sqrt(rr)
    elif method == "fridericia":
        corrected = qt / np.cbrt(rr)
    else:
        raise ValueError(f"Unsupported QT correction: {method}")
    confidence = float(
        np.median([item.confidence for item in measurements])
        * min(1.0, 0.55 + 0.09 * len(measurements))
    )
    return (
        float(np.median(corrected)),
        confidence,
        f"Median of {len(measurements)} beatwise QT-RR pairs",
    )


class HeartRateBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "heart_rate"

    @property
    def description(self) -> str:
        return "Heart rate derived from R-R intervals"

    @property
    def unit(self) -> str:
        return "bpm"

    @property
    def category(self) -> str:
        return "ecg"

    def normal_range(self, age=None, sex=None):
        return (60.0, 100.0)

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        beats = delineation.beats
        if len(beats) < 2:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        rr_samples = np.diff([b.r_peak.sample for b in beats])
        rr_seconds = rr_samples / fs
        rr_seconds = np.where(rr_seconds > 0, rr_seconds, np.nan)
        hr_values = 60.0 / rr_seconds
        hr_mean = float(np.nanmean(hr_values))
        if np.isnan(hr_mean):
            hr_mean = None
        return BiomarkerResult(name=self.name, value=hr_mean, unit=self.unit, category=self.category, confidence=0.95)


class PRIntervalBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "pr_interval"

    @property
    def description(self) -> str:
        return "PR interval: from P onset to QRS onset"

    @property
    def unit(self) -> str:
        return "ms"

    @property
    def category(self) -> str:
        return "ecg"

    def normal_range(self, age=None, sex=None):
        return (120.0, 200.0)

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        confidence = delineation.segment_confidence("pr")
        if confidence < measurement_confidence_threshold("pr"):
            return BiomarkerResult(
                name=self.name, value=None, unit=self.unit, category=self.category,
                confidence=0.0,
                computation_notes="PR boundaries are not reliably evaluable",
            )

        pr_samples = median_beat.r_onset.sample - median_beat.p_onset.sample
        pr_ms = (pr_samples / fs) * 1000.0

        return BiomarkerResult(
            name=self.name,
            value=pr_ms,
            unit=self.unit,
            category=self.category,
            confidence=confidence,
        )


class QRSDurationBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "qrs_duration"

    @property
    def description(self) -> str:
        return "QRS complex duration"

    @property
    def unit(self) -> str:
        return "ms"

    @property
    def category(self) -> str:
        return "ecg"

    def normal_range(self, age=None, sex=None):
        return (60.0, 120.0)

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        confidence = delineation.segment_confidence("qrs")
        if (
            confidence >= measurement_confidence_threshold("qrs")
            and median_beat.r_onset is not None
            and median_beat.r_offset is not None
        ):
            dur_samples = median_beat.r_offset.sample - median_beat.r_onset.sample
        else:
            return BiomarkerResult(
                name=self.name, value=None, unit=self.unit, category=self.category,
                confidence=0.0,
                computation_notes="QRS boundaries are not reliably evaluable",
            )

        dur_ms = (dur_samples / fs) * 1000.0
        return BiomarkerResult(
            name=self.name,
            value=dur_ms,
            unit=self.unit,
            category=self.category,
            confidence=confidence,
        )


class QTIntervalBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "qt_interval"

    @property
    def description(self) -> str:
        return "QT interval: from QRS onset to T offset"

    @property
    def unit(self) -> str:
        return "ms"

    @property
    def category(self) -> str:
        return "ecg"

    def normal_range(self, age=None, sex=None):
        return (350.0, 450.0)

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        if median_beat.t_offset is None:
            return BiomarkerResult(
                name=self.name, value=None, unit=self.unit, category=self.category,
                confidence=0.0, computation_notes="T offset not detected",
            )

        if median_beat.r_onset is None:
            return BiomarkerResult(
                name=self.name, value=None, unit=self.unit, category=self.category,
                confidence=0.0, computation_notes="QRS onset not detected",
            )

        qt_samples = median_beat.t_offset.sample - median_beat.r_onset.sample
        qt_ms = (qt_samples / fs) * 1000.0
        confidence = delineation.segment_confidence("qrst")
        if confidence < measurement_confidence_threshold("qrst"):
            return BiomarkerResult(
                name=self.name,
                value=None,
                unit=self.unit,
                category=self.category,
                confidence=0.0,
                computation_notes="QRS-T boundaries below confidence threshold",
            )
        return BiomarkerResult(
            name=self.name,
            value=qt_ms,
            unit=self.unit,
            category=self.category,
            confidence=confidence,
        )


class QTCBazettBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "qtc_bazett"

    @property
    def description(self) -> str:
        return "Corrected QT interval using Bazett's formula: QTc = QT / sqrt(RR)"

    @property
    def unit(self) -> str:
        return "ms"

    @property
    def category(self) -> str:
        return "ecg"

    def normal_range(self, age=None, sex=None):
        upper = 460.0 if sex == "F" else 440.0
        return (350.0, upper)

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        value, confidence, notes = _corrected_qt_from_paired_beats(
            delineation,
            "bazett",
        )
        return BiomarkerResult(
            name=self.name,
            value=value,
            unit=self.unit,
            category=self.category,
            confidence=confidence,
            computation_notes=notes,
        )


class QTCFridericiaBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "qtc_fridericia"

    @property
    def description(self) -> str:
        return "Corrected QT interval using Fridericia's formula: QTc = QT / RR^(1/3)"

    @property
    def unit(self) -> str:
        return "ms"

    @property
    def category(self) -> str:
        return "ecg"

    def normal_range(self, age=None, sex=None):
        upper = 470.0 if sex == "F" else 450.0
        return (350.0, upper)

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        value, confidence, notes = _corrected_qt_from_paired_beats(
            delineation,
            "fridericia",
        )
        return BiomarkerResult(
            name=self.name,
            value=value,
            unit=self.unit,
            category=self.category,
            confidence=confidence,
            computation_notes=notes,
        )


class PDurationBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "p_duration"

    @property
    def description(self) -> str:
        return "P wave duration"

    @property
    def unit(self) -> str:
        return "ms"

    @property
    def category(self) -> str:
        return "ecg"

    def normal_range(self, age=None, sex=None):
        return (80.0, 120.0)

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        confidence = delineation.segment_confidence("p")
        if confidence < measurement_confidence_threshold("p"):
            return BiomarkerResult(
                name=self.name, value=None, unit=self.unit, category=self.category,
                confidence=0.0,
                computation_notes="P-wave boundaries are not reliably evaluable",
            )

        dur_samples = median_beat.p_offset.sample - median_beat.p_onset.sample
        dur_ms = (dur_samples / fs) * 1000.0

        return BiomarkerResult(
            name=self.name,
            value=dur_ms,
            unit=self.unit,
            category=self.category,
            confidence=confidence,
        )


class PAmplitudeBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "p_amplitude_ii"

    @property
    def description(self) -> str:
        return "P wave amplitude in lead II"

    @property
    def unit(self) -> str:
        return "uV"

    @property
    def category(self) -> str:
        return "ecg"

    def normal_range(self, age=None, sex=None):
        return (50.0, 250.0)

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None or ecg_12lead is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        if (
            median_beat.p_peak is None
            or median_beat.resolved_wave_status("p").value != "present"
            or median_beat.p_peak.confidence
            < fiducial_confidence_threshold("p_peak")
            or delineation.segment_confidence("p") < measurement_confidence_threshold("p")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        lead_ii = ecg_12lead[:, 1]
        p_idx = median_beat.p_peak.sample
        if p_idx >= len(lead_ii):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        amplitude_uv = lead_ii[p_idx] * 1000.0
        return BiomarkerResult(name=self.name, value=amplitude_uv, unit=self.unit, category=self.category, confidence=median_beat.p_peak.confidence)


class RAmplitudeBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "r_amplitudes"

    @property
    def description(self) -> str:
        return "R wave amplitude per lead"

    @property
    def unit(self) -> str:
        return "uV"

    @property
    def category(self) -> str:
        return "ecg"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None or ecg_12lead is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        if median_beat.r_peak.confidence < fiducial_confidence_threshold(
            "r_peak"
        ):
            return BiomarkerResult(
                name=self.name,
                value=None,
                unit=self.unit,
                category=self.category,
                confidence=0.0,
                computation_notes="R peak is below the confidence threshold",
            )
        r_idx = median_beat.r_peak.sample
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        per_lead = {}

        for i, name in enumerate(lead_names):
            if i < ecg_12lead.shape[1] and r_idx < ecg_12lead.shape[0]:
                per_lead[name] = ecg_12lead[r_idx, i] * 1000.0

        max_amp = max(per_lead.values()) if per_lead else None
        return BiomarkerResult(
            name=self.name, value=max_amp, unit=self.unit, category=self.category,
            confidence=median_beat.r_peak.confidence, per_lead=per_lead,
        )


class SDepthBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "s_depths"

    @property
    def description(self) -> str:
        return "S wave depth per lead"

    @property
    def unit(self) -> str:
        return "uV"

    @property
    def category(self) -> str:
        return "ecg"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None or ecg_12lead is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        if (
            median_beat.s_peak is None
            or delineation.segment_confidence("qrs") < measurement_confidence_threshold("qrs")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        s_idx = median_beat.s_peak.sample
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        per_lead = {}

        for i, name in enumerate(lead_names):
            if i < ecg_12lead.shape[1] and s_idx < ecg_12lead.shape[0]:
                val = ecg_12lead[s_idx, i] * 1000.0
                if val < 0:
                    per_lead[name] = abs(val)

        max_depth = max(per_lead.values()) if per_lead else None
        return BiomarkerResult(
            name=self.name, value=max_depth, unit=self.unit, category=self.category,
            confidence=median_beat.s_peak.confidence, per_lead=per_lead,
        )


class STElevationBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "st_elevation"

    @property
    def description(self) -> str:
        return "ST segment elevation at J+60ms per lead"

    @property
    def unit(self) -> str:
        return "uV"

    @property
    def category(self) -> str:
        return "ecg"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None or ecg_12lead is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        confidence = delineation.segment_confidence("st")
        if confidence < measurement_confidence_threshold("st"):
            return BiomarkerResult(
                name=self.name,
                value=None,
                unit=self.unit,
                category=self.category,
                confidence=0.0,
                computation_notes="ST boundaries not reliably evaluable",
            )
        j_point = median_beat.j_point
        if j_point is None:
            r_offset = median_beat.r_offset
            if r_offset is not None:
                j_point = r_offset
            else:
                return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        j60_sample = j_point.sample + int(0.06 * fs)
        if j60_sample >= ecg_12lead.shape[0]:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        per_lead = {}
        for i, name in enumerate(lead_names):
            if i < ecg_12lead.shape[1]:
                val = ecg_12lead[j60_sample, i] * 1000.0
                if val > 0:
                    per_lead[name] = val

        max_elev = max(per_lead.values()) if per_lead else 0.0
        return BiomarkerResult(
            name=self.name, value=max_elev, unit=self.unit, category=self.category,
            confidence=confidence, per_lead=per_lead,
        )


class STDepressionBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "st_depression"

    @property
    def description(self) -> str:
        return "ST segment depression at J+60ms per lead"

    @property
    def unit(self) -> str:
        return "uV"

    @property
    def category(self) -> str:
        return "ecg"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        median_beat = delineation.representative_beat
        if median_beat is None or ecg_12lead is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        confidence = delineation.segment_confidence("st")
        if confidence < measurement_confidence_threshold("st"):
            return BiomarkerResult(
                name=self.name,
                value=None,
                unit=self.unit,
                category=self.category,
                confidence=0.0,
                computation_notes="ST boundaries not reliably evaluable",
            )
        j_point = median_beat.j_point
        if j_point is None:
            r_offset = median_beat.r_offset
            if r_offset is not None:
                j_point = r_offset
            else:
                return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        j60_sample = j_point.sample + int(0.06 * fs)
        if j60_sample >= ecg_12lead.shape[0]:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        per_lead = {}
        for i, name in enumerate(lead_names):
            if i < ecg_12lead.shape[1]:
                val = ecg_12lead[j60_sample, i] * 1000.0
                if val < 0:
                    per_lead[name] = abs(val)

        max_depr = max(per_lead.values()) if per_lead else 0.0
        return BiomarkerResult(
            name=self.name, value=max_depr, unit=self.unit, category=self.category,
            confidence=confidence, per_lead=per_lead,
        )


def register_ecg_biomarkers():
    from .base import BiomarkerRegistry
    for cls in [
        HeartRateBiomarker, PRIntervalBiomarker, QRSDurationBiomarker,
        QTIntervalBiomarker, QTCBazettBiomarker, QTCFridericiaBiomarker,
        PDurationBiomarker, PAmplitudeBiomarker, RAmplitudeBiomarker,
        SDepthBiomarker, STElevationBiomarker, STDepressionBiomarker,
    ]:
        BiomarkerRegistry.register(cls())
