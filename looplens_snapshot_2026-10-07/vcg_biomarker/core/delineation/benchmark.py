from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import wfdb

from ..data_loader import LEAD_NAMES_12
from ..preprocessing import (
    detect_r_peaks_multilead,
    prepare_signal_paths,
)
from ..vcg_transform import VCGTransformRegistry
from .base import (
    RecordDelineation,
    RhythmClass,
    WaveStatus,
    measurement_confidence_threshold,
)
from .fusion_delineator import FusionDelineator


LUDB_ANNOTATORS = (
    "i",
    "ii",
    "iii",
    "avr",
    "avl",
    "avf",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
)

# Fixed before benchmark evaluation. Matching tolerances identify the same
# physiological event; boundary tolerances are only descriptive pass bands.
# Raw signed and absolute errors are always reported as the primary audit.
EVENT_CLUSTER_TOLERANCE_MS = {"p": 65.0, "qrs": 120.0, "t": 90.0}
EVENT_MATCH_TOLERANCE_MS = {"p": 120.0, "qrs": 100.0, "t": 160.0}
BOUNDARY_TOLERANCE_MS = {
    "p_onset": 40.0,
    "p_peak": 40.0,
    "p_offset": 40.0,
    "qrs_onset": 30.0,
    "qrs_peak": 30.0,
    "qrs_offset": 30.0,
    "t_onset": 60.0,
    "t_peak": 60.0,
    "t_offset": 60.0,
}


@dataclass(frozen=True)
class AnnotatedWave:
    wave: str
    onset: int
    peak: int
    offset: int
    lead: str


@dataclass(frozen=True)
class GlobalWaveReference:
    wave: str
    onset: int
    peak: int
    offset: int
    lead_count: int
    leads: tuple[str, ...]
    onset_spread_ms: float
    peak_spread_ms: float
    offset_spread_ms: float


@dataclass(frozen=True)
class PredictedWave:
    wave: str
    onset: Optional[int]
    peak: int
    offset: Optional[int]
    confidence: float


@dataclass
class _WaveMetricAccumulator:
    reference_events: int = 0
    predicted_events: int = 0
    matched_events: int = 0
    errors_ms: dict[str, list[float]] = field(
        default_factory=lambda: {
            "onset": [],
            "peak": [],
            "offset": [],
        }
    )

    def update(
        self,
        references: list[GlobalWaveReference],
        predictions: list[PredictedWave],
        fs: int,
        match_tolerance_ms: float,
    ) -> None:
        self.reference_events += len(references)
        self.predicted_events += len(predictions)
        matches = _match_wave_events(
            references,
            predictions,
            fs,
            match_tolerance_ms,
        )
        self.matched_events += len(matches)
        for reference, prediction in matches:
            self.errors_ms["peak"].append(
                (prediction.peak - reference.peak) / fs * 1000.0
            )
            if prediction.onset is not None:
                self.errors_ms["onset"].append(
                    (prediction.onset - reference.onset) / fs * 1000.0
                )
            if prediction.offset is not None:
                self.errors_ms["offset"].append(
                    (prediction.offset - reference.offset) / fs * 1000.0
                )

    def to_dict(self, wave: str) -> dict:
        false_negative = self.reference_events - self.matched_events
        false_positive = self.predicted_events - self.matched_events
        sensitivity = (
            self.matched_events / self.reference_events
            if self.reference_events
            else None
        )
        ppv = (
            self.matched_events / self.predicted_events
            if self.predicted_events
            else None
        )
        return {
            "reference_events": self.reference_events,
            "predicted_events": self.predicted_events,
            "matched_events": self.matched_events,
            "false_negative_events": false_negative,
            "false_positive_events": false_positive,
            "sensitivity": sensitivity,
            "positive_predictive_value": ppv,
            "boundary_errors_ms": {
                location: _error_summary(
                    values,
                    BOUNDARY_TOLERANCE_MS[f"{wave}_{location}"],
                )
                for location, values in self.errors_ms.items()
            },
        }


@dataclass
class _GroupAccumulator:
    record_count: int = 0
    quality_scores: list[float] = field(default_factory=list)
    review_required_count: int = 0
    p_confidence: list[float] = field(default_factory=list)
    qrs_confidence: list[float] = field(default_factory=list)
    t_confidence: list[float] = field(default_factory=list)
    record_evaluable_count: dict[str, int] = field(
        default_factory=lambda: {"p": 0, "qrs": 0, "t": 0}
    )
    waves: dict[str, _WaveMetricAccumulator] = field(
        default_factory=lambda: {
            "p": _WaveMetricAccumulator(),
            "qrs": _WaveMetricAccumulator(),
            "t": _WaveMetricAccumulator(),
        }
    )
    evaluable_waves: dict[str, _WaveMetricAccumulator] = field(
        default_factory=lambda: {
            "p": _WaveMetricAccumulator(),
            "qrs": _WaveMetricAccumulator(),
            "t": _WaveMetricAccumulator(),
        }
    )

    def update(
        self,
        references: dict[str, list[GlobalWaveReference]],
        predictions: dict[str, list[PredictedWave]],
        evaluable_predictions: dict[str, list[PredictedWave]],
        delineation: RecordDelineation,
        fs: int,
    ) -> None:
        self.record_count += 1
        self.quality_scores.append(float(delineation.quality_score))
        self.review_required_count += int(delineation.review_required)
        self.p_confidence.append(delineation.segment_confidence("p"))
        self.qrs_confidence.append(delineation.segment_confidence("qrs"))
        self.t_confidence.append(delineation.segment_confidence("t"))
        for wave in ("p", "qrs", "t"):
            self.record_evaluable_count[wave] += int(
                delineation.is_segment_evaluable(wave)
            )
        for wave in ("p", "qrs", "t"):
            self.waves[wave].update(
                references[wave],
                predictions[wave],
                fs,
                EVENT_MATCH_TOLERANCE_MS[wave],
            )
            self.evaluable_waves[wave].update(
                references[wave],
                evaluable_predictions[wave],
                fs,
                EVENT_MATCH_TOLERANCE_MS[wave],
            )

    def to_dict(self) -> dict:
        return {
            "record_count": self.record_count,
            "review_required_rate": (
                self.review_required_count / self.record_count
                if self.record_count
                else None
            ),
            "quality_score": _distribution_summary(self.quality_scores),
            "segment_confidence": {
                "p": _distribution_summary(self.p_confidence),
                "qrs": _distribution_summary(self.qrs_confidence),
                "t": _distribution_summary(self.t_confidence),
            },
            "record_evaluable_rate": {
                wave: _safe_ratio(count, self.record_count)
                for wave, count in self.record_evaluable_count.items()
            },
            "waves": {
                wave: accumulator.to_dict(wave)
                for wave, accumulator in self.waves.items()
            },
            "evaluable_waves": {
                wave: accumulator.to_dict(wave)
                for wave, accumulator in self.evaluable_waves.items()
            },
        }


@dataclass
class _RhythmAudit:
    atrial_guard_expected_records: int = 0
    atrial_guard_applied_records: int = 0
    af_records: int = 0
    af_compatible_records: int = 0
    flutter_records: int = 0
    flutter_or_multiple_activity_records: int = 0
    guarded_beats: int = 0
    discrete_p_beats_in_guarded_records: int = 0
    total_beats_in_guarded_records: int = 0
    non_af_flutter_records: int = 0
    unexpected_af_flutter_pattern_records: int = 0
    unexpected_atrial_guard_records: int = 0

    def update(self, rhythm_label: str, delineation: RecordDelineation) -> None:
        normalized = _normalize_text(rhythm_label)
        is_af = "atrial_fibrillation" in normalized
        is_flutter = "atrial_flutter" in normalized
        expects_guard = is_af or is_flutter
        if not expects_guard:
            self.non_af_flutter_records += 1
            self.unexpected_af_flutter_pattern_records += int(
                delineation.rhythm.rhythm_class
                in {
                    RhythmClass.AF_COMPATIBLE,
                    RhythmClass.FLUTTER_COMPATIBLE,
                }
            )
            self.unexpected_atrial_guard_records += int(
                delineation.rhythm.discrete_p_not_evaluable
            )
        if is_af:
            self.af_records += 1
            self.af_compatible_records += int(
                delineation.rhythm.rhythm_class.value
                == "af_compatible_pattern"
            )
        if is_flutter:
            self.flutter_records += 1
            self.flutter_or_multiple_activity_records += int(
                delineation.rhythm.rhythm_class.value
                in {
                    "flutter_compatible_pattern",
                    "multiple_atrial_activity",
                }
            )
        if not expects_guard:
            return
        self.atrial_guard_expected_records += 1
        self.atrial_guard_applied_records += int(
            not delineation.is_segment_evaluable("p")
        )
        self.total_beats_in_guarded_records += len(delineation.beats)
        for beat in delineation.beats:
            if beat.resolved_wave_status("p") in {
                WaveStatus.REPLACED_BY_ATRIAL_ACTIVITY,
                WaveStatus.NOT_ASSESSABLE,
                WaveStatus.FUSED,
            }:
                self.guarded_beats += 1
            if beat.resolved_wave_status("p") == WaveStatus.PRESENT:
                self.discrete_p_beats_in_guarded_records += 1

    def to_dict(self) -> dict:
        return {
            "atrial_guard_expected_records": (
                self.atrial_guard_expected_records
            ),
            "atrial_guard_applied_records": self.atrial_guard_applied_records,
            "atrial_guard_record_sensitivity": _safe_ratio(
                self.atrial_guard_applied_records,
                self.atrial_guard_expected_records,
            ),
            "af_records": self.af_records,
            "af_compatible_records": self.af_compatible_records,
            "af_compatible_record_rate": _safe_ratio(
                self.af_compatible_records,
                self.af_records,
            ),
            "flutter_records": self.flutter_records,
            "flutter_or_multiple_activity_records": (
                self.flutter_or_multiple_activity_records
            ),
            "flutter_or_multiple_activity_record_rate": _safe_ratio(
                self.flutter_or_multiple_activity_records,
                self.flutter_records,
            ),
            "total_beats_in_guarded_records": (
                self.total_beats_in_guarded_records
            ),
            "guarded_beats": self.guarded_beats,
            "discrete_p_beats_in_guarded_records": (
                self.discrete_p_beats_in_guarded_records
            ),
            "false_discrete_p_beat_rate": _safe_ratio(
                self.discrete_p_beats_in_guarded_records,
                self.total_beats_in_guarded_records,
            ),
            "non_af_flutter_records": self.non_af_flutter_records,
            "unexpected_af_flutter_pattern_records": (
                self.unexpected_af_flutter_pattern_records
            ),
            "af_flutter_pattern_specificity": _safe_ratio(
                self.non_af_flutter_records
                - self.unexpected_af_flutter_pattern_records,
                self.non_af_flutter_records,
            ),
            "unexpected_atrial_guard_records": (
                self.unexpected_atrial_guard_records
            ),
            "atrial_guard_specificity": _safe_ratio(
                self.non_af_flutter_records
                - self.unexpected_atrial_guard_records,
                self.non_af_flutter_records,
            ),
        }


def run_ludb_benchmark(
    database_root: str | Path,
    output_path: str | Path,
    *,
    max_records: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run the production delineation path against cardiologist LUDB marks.

    LUDB is a development quality-control benchmark, not a locked clinical
    validation cohort. The function never modifies source data and writes one
    atomic JSON result after all selected records have completed.
    """

    root = resolve_ludb_root(database_root)
    records = [
        line.strip()
        for line in (root / "RECORDS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        records = records[: int(max_records)]
    metadata = _load_ludb_metadata(root / "ludb.csv")

    groups: dict[str, _GroupAccumulator] = {}
    rhythm_audit = _RhythmAudit()
    record_results: list[dict] = []
    quarantined: list[dict] = []
    delineator = FusionDelineator()
    for index, relative_record in enumerate(records, start=1):
        record_id = Path(relative_record).name
        if progress is not None:
            progress(
                f"LUDB {index}/{len(records)}: record {record_id}"
            )
        try:
            signal, fs = _load_ludb_signal(root / relative_record)
            references, annotation_qc = load_ludb_global_references(
                root / relative_record,
                fs,
            )
            paths = prepare_signal_paths(signal, fs, powerline=50.0)
            detected_r = detect_r_peaks_multilead(
                paths["delineation"],
                fs,
            )
            vcg = VCGTransformRegistry.transform(
                paths["delineation"],
                "kors",
            )
            delineation = delineator.delineate_record(
                paths["delineation"],
                detected_r,
                fs,
                ecg_id=int(record_id),
                vcg_xyz=vcg.xyz,
            )
            references = _filter_reference_edges(
                references,
                len(signal),
                int(round(0.25 * fs)),
            )
            predictions = _predicted_waves(
                delineation,
                edge_margin_samples=int(round(0.25 * fs)),
            )
            predictions = _filter_prediction_to_reference_coverage(
                predictions,
                references,
                fs,
            )
            evaluable_predictions = _predicted_waves(
                delineation,
                edge_margin_samples=int(round(0.25 * fs)),
                minimum_confidence=None,
            )
            evaluable_predictions = (
                _filter_prediction_to_reference_coverage(
                    evaluable_predictions,
                    references,
                    fs,
                )
            )
            row_metadata = metadata.get(record_id, {})
            record_groups = _record_groups(row_metadata)
            for group_name in record_groups:
                accumulator = groups.setdefault(
                    group_name,
                    _GroupAccumulator(),
                )
                accumulator.update(
                    references,
                    predictions,
                    evaluable_predictions,
                    delineation,
                    fs,
                )
            rhythm_label = row_metadata.get("Rhythms", "")
            rhythm_audit.update(rhythm_label, delineation)
            record_results.append(
                {
                    "record_id": record_id,
                    "groups": record_groups,
                    "reference_event_count": {
                        wave: len(values)
                        for wave, values in references.items()
                    },
                    "predicted_event_count": {
                        wave: len(values)
                        for wave, values in predictions.items()
                    },
                    "evaluable_event_count": {
                        wave: len(values)
                        for wave, values in evaluable_predictions.items()
                    },
                    "detected_r_peaks": int(len(detected_r)),
                    "quality_score": float(delineation.quality_score),
                    "review_required": delineation.review_required,
                    "review_reasons": list(delineation.review_reasons),
                    "predicted_rhythm_pattern": (
                        delineation.rhythm.rhythm_class.value
                    ),
                    "predicted_rhythm_confidence": float(
                        delineation.rhythm.confidence
                    ),
                    "rhythm_evidence": {
                        "rr_cv": delineation.rhythm.rr_cv,
                        "rr_successive_irregularity": (
                            delineation.rhythm.rr_successive_irregularity
                        ),
                        "rr_lag1_correlation": (
                            delineation.rhythm.rr_lag1_correlation
                        ),
                        "rr_turning_point_fraction": (
                            delineation.rhythm.rr_turning_point_fraction
                        ),
                        "p_wave_coverage": (
                            delineation.rhythm.p_wave_coverage
                        ),
                        "p_timing_mad_ms": (
                            delineation.rhythm.p_timing_mad_ms
                        ),
                        "p_candidate_coverage": (
                            delineation.rhythm.p_candidate_coverage
                        ),
                        "p_candidate_timing_mad_ms": (
                            delineation.rhythm.p_candidate_timing_mad_ms
                        ),
                        "p_candidate_timing_p10_p90_ms": (
                            delineation.rhythm.p_candidate_timing_p10_p90_ms
                        ),
                        "atrial_frequency_hz": (
                            delineation.rhythm.atrial_frequency_hz
                        ),
                        "atrial_spectral_concentration": (
                            delineation.rhythm.atrial_spectral_concentration
                        ),
                        "atrial_bandpower_ratio": (
                            delineation.rhythm.atrial_bandpower_ratio
                        ),
                        "flags": list(delineation.rhythm.flags),
                    },
                    "annotation_qc": annotation_qc,
                }
            )
        except Exception as exc:
            quarantined.append(
                {
                    "record_id": record_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    result = {
        "schema_version": "1.2.0",
        "benchmark": "LUDB",
        "benchmark_version": "1.0.1",
        "benchmark_role": (
            "development_quality_control_not_locked_clinical_validation"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root.resolve()),
        "dataset_manifest_sha256": _sha256_file(root / "SHA256SUMS.txt"),
        "implementation_sha256": _implementation_hash(),
        "configuration": {
            "powerline_hz": 50.0,
            "vcg_transform": "kors",
            "global_reference_rule": (
                "earliest multilead onset, median peak, latest multilead "
                "offset within a time-coherent wave cluster"
            ),
            "annotation_coverage_rule": (
                "predictions outside the lead-annotation event span plus "
                "half a median cycle are excluded"
            ),
            "minimum_reference_leads": 2,
            "edge_exclusion_ms": 250.0,
            "event_cluster_tolerance_ms": EVENT_CLUSTER_TOLERANCE_MS,
            "event_match_tolerance_ms": EVENT_MATCH_TOLERANCE_MS,
            "descriptive_boundary_tolerance_ms": BOUNDARY_TOLERANCE_MS,
            "measurement_confidence_threshold": {
                wave: measurement_confidence_threshold(wave)
                for wave in ("p", "qrs", "t")
            },
            "rhythm_guard_thresholds": {
                "rr_cv_irregular": 0.12,
                "successive_rr_irregularity": 0.10,
                "stable_p_timing_mad_ms": 20.0,
                "flutter_search_band_hz": [4.0, 8.5],
                "minimum_spectral_concentration": 0.32,
                "weak_high_frequency_minimum_concentration": 0.42,
                "strong_atrial_bandpower_ratio": 0.45,
                "atrial_bandpower_numerator_hz": [3.5, 9.0],
                "atrial_bandpower_denominator_hz": [0.5, 40.0],
            },
            "fiducial_uncertainty_limits_ms": {
                "p_boundary": 40.0,
                "qrs_boundary": 30.0,
                "t_boundary": 60.0,
            },
            "tolerance_policy": (
                "fixed before evaluation; raw errors remain primary"
            ),
        },
        "records_requested": len(records),
        "records_completed": len(record_results),
        "records_quarantined": len(quarantined),
        "quarantine": quarantined,
        "groups": {
            name: accumulator.to_dict()
            for name, accumulator in sorted(groups.items())
        },
        "rhythm_guard_audit": rhythm_audit.to_dict(),
        "records": record_results,
        "interpretation_guardrails": [
            "No ECG diagnosis is inferred from this engineering benchmark.",
            "Missing or uncertain boundaries are abstentions, not normal findings.",
            "AF/flutter P-wave performance is reported as a rhythm-guard task.",
            "LUDB was not used as a locked MIMIC or PTB test cohort.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return result


def resolve_ludb_root(database_root: str | Path) -> Path:
    root = Path(database_root).expanduser()
    candidates = [root]
    candidates.extend(
        path.parent
        for path in root.rglob("RECORDS")
        if path.is_file()
    )
    for candidate in candidates:
        if (
            (candidate / "RECORDS").is_file()
            and (candidate / "ludb.csv").is_file()
            and (candidate / "data").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "LUDB root not found. Expected RECORDS, ludb.csv and data/ under "
        f"{root}."
    )


def load_ludb_global_references(
    record_path: str | Path,
    fs: int,
) -> tuple[dict[str, list[GlobalWaveReference]], dict]:
    path = Path(record_path)
    waves: list[AnnotatedWave] = []
    malformed = 0
    available_leads = 0
    for annotator in LUDB_ANNOTATORS:
        annotation_file = path.with_suffix(f".{annotator}")
        if not annotation_file.is_file():
            continue
        available_leads += 1
        annotation = wfdb.rdann(str(path), annotator)
        parsed, invalid = _parse_ludb_annotation(
            annotation.sample,
            annotation.symbol,
            annotator,
        )
        waves.extend(parsed)
        malformed += invalid
    if available_leads < 2:
        raise ValueError(
            f"Only {available_leads} LUDB lead annotation files are available"
        )
    references = {
        wave: _cluster_global_references(
            [item for item in waves if item.wave == wave],
            fs,
            EVENT_CLUSTER_TOLERANCE_MS[wave],
        )
        for wave in ("p", "qrs", "t")
    }
    return references, {
        "available_annotation_leads": available_leads,
        "parsed_lead_waves": len(waves),
        "malformed_annotation_sequences": malformed,
    }


def _parse_ludb_annotation(
    samples: Iterable[int],
    symbols: Iterable[str],
    lead: str,
) -> tuple[list[AnnotatedWave], int]:
    waves: list[AnnotatedWave] = []
    onset: Optional[int] = None
    peak: Optional[int] = None
    wave: Optional[str] = None
    malformed = 0
    symbol_to_wave = {"p": "p", "N": "qrs", "t": "t"}
    for raw_sample, symbol in zip(samples, symbols):
        sample = int(raw_sample)
        if symbol == "(":
            if onset is not None:
                malformed += 1
            onset, peak, wave = sample, None, None
            continue
        if symbol in symbol_to_wave:
            if onset is None or peak is not None:
                malformed += 1
                onset, peak, wave = None, None, None
                continue
            peak = sample
            wave = symbol_to_wave[symbol]
            continue
        if symbol == ")":
            if (
                onset is None
                or peak is None
                or wave is None
                or not onset < peak < sample
            ):
                malformed += 1
            else:
                waves.append(
                    AnnotatedWave(
                        wave=wave,
                        onset=onset,
                        peak=peak,
                        offset=sample,
                        lead=lead,
                    )
                )
            onset, peak, wave = None, None, None
    if onset is not None:
        malformed += 1
    return waves, malformed


def _cluster_global_references(
    waves: list[AnnotatedWave],
    fs: int,
    tolerance_ms: float,
) -> list[GlobalWaveReference]:
    if not waves:
        return []
    tolerance = max(1, int(round(tolerance_ms / 1000.0 * fs)))
    clusters: list[list[AnnotatedWave]] = []
    for wave in sorted(waves, key=lambda item: item.peak):
        if (
            not clusters
            or abs(
                wave.peak
                - float(np.median([item.peak for item in clusters[-1]]))
            )
            > tolerance
        ):
            clusters.append([wave])
        else:
            clusters[-1].append(wave)

    references: list[GlobalWaveReference] = []
    for cluster in clusters:
        by_lead: dict[str, AnnotatedWave] = {}
        median_peak = float(np.median([item.peak for item in cluster]))
        for item in cluster:
            previous = by_lead.get(item.lead)
            if (
                previous is None
                or abs(item.peak - median_peak)
                < abs(previous.peak - median_peak)
            ):
                by_lead[item.lead] = item
        selected = list(by_lead.values())
        if len(selected) < 2:
            continue
        onsets = np.asarray([item.onset for item in selected], dtype=float)
        peaks = np.asarray([item.peak for item in selected], dtype=float)
        offsets = np.asarray([item.offset for item in selected], dtype=float)
        references.append(
            GlobalWaveReference(
                wave=selected[0].wave,
                onset=int(np.min(onsets)),
                peak=int(round(float(np.median(peaks)))),
                offset=int(np.max(offsets)),
                lead_count=len(selected),
                leads=tuple(sorted(by_lead)),
                onset_spread_ms=_robust_spread_ms(onsets, fs),
                peak_spread_ms=_robust_spread_ms(peaks, fs),
                offset_spread_ms=_robust_spread_ms(offsets, fs),
            )
        )
    return references


def _predicted_waves(
    delineation: RecordDelineation,
    *,
    edge_margin_samples: int,
    minimum_confidence: Optional[float] = 0.0,
) -> dict[str, list[PredictedWave]]:
    """Predicted waves, optionally restricted to what the markers would use.

    ``minimum_confidence=None`` applies the production floor of each wave (see
    ``measurement_confidence_threshold``); a number applies that same number to
    every wave, which is only meaningful for a threshold sweep.
    """

    def floor(wave: str) -> float:
        return (
            measurement_confidence_threshold(wave)
            if minimum_confidence is None
            else minimum_confidence
        )

    predictions = {"p": [], "qrs": [], "t": []}
    maximum = delineation.n_samples - edge_margin_samples
    for beat in delineation.beats:
        if not edge_margin_samples <= beat.r_peak.sample < maximum:
            continue
        if (
            beat.resolved_wave_status("qrs") == WaveStatus.PRESENT
            and beat.segment_confidence("qrs") >= floor("qrs")
        ):
            predictions["qrs"].append(
                PredictedWave(
                    wave="qrs",
                    onset=(
                        beat.r_onset.sample
                        if beat.r_onset is not None
                        else None
                    ),
                    peak=beat.r_peak.sample,
                    offset=(
                        beat.r_offset.sample
                        if beat.r_offset is not None
                        else None
                    ),
                    confidence=beat.segment_confidence("qrs"),
                )
            )
        if (
            beat.resolved_wave_status("p") == WaveStatus.PRESENT
            and beat.p_peak is not None
            and beat.segment_confidence("p") >= floor("p")
        ):
            predictions["p"].append(
                PredictedWave(
                    wave="p",
                    onset=(
                        beat.p_onset.sample
                        if beat.p_onset is not None
                        else None
                    ),
                    peak=beat.p_peak.sample,
                    offset=(
                        beat.p_offset.sample
                        if beat.p_offset is not None
                        else None
                    ),
                    confidence=beat.segment_confidence("p"),
                )
            )
        if (
            beat.resolved_wave_status("t") == WaveStatus.PRESENT
            and beat.t_peak is not None
            and beat.segment_confidence("t") >= floor("t")
        ):
            predictions["t"].append(
                PredictedWave(
                    wave="t",
                    onset=(
                        beat.t_onset.sample
                        if beat.t_onset is not None
                        else None
                    ),
                    peak=beat.t_peak.sample,
                    offset=(
                        beat.t_offset.sample
                        if beat.t_offset is not None
                        else None
                    ),
                    confidence=beat.segment_confidence("t"),
                )
            )
    return predictions


def _match_wave_events(
    references: list[GlobalWaveReference],
    predictions: list[PredictedWave],
    fs: int,
    tolerance_ms: float,
) -> list[tuple[GlobalWaveReference, PredictedWave]]:
    tolerance = max(1, int(round(tolerance_ms / 1000.0 * fs)))
    candidates = sorted(
        (
            (abs(reference.peak - prediction.peak), ref_index, pred_index)
            for ref_index, reference in enumerate(references)
            for pred_index, prediction in enumerate(predictions)
            if abs(reference.peak - prediction.peak) <= tolerance
        ),
        key=lambda item: item[0],
    )
    used_references: set[int] = set()
    used_predictions: set[int] = set()
    matches = []
    for _, ref_index, pred_index in candidates:
        if (
            ref_index in used_references
            or pred_index in used_predictions
        ):
            continue
        used_references.add(ref_index)
        used_predictions.add(pred_index)
        matches.append((references[ref_index], predictions[pred_index]))
    return matches


def _filter_reference_edges(
    references: dict[str, list[GlobalWaveReference]],
    n_samples: int,
    margin: int,
) -> dict[str, list[GlobalWaveReference]]:
    return {
        wave: [
            reference
            for reference in values
            if margin <= reference.peak < n_samples - margin
        ]
        for wave, values in references.items()
    }


def _filter_prediction_to_reference_coverage(
    predictions: dict[str, list[PredictedWave]],
    references: dict[str, list[GlobalWaveReference]],
    fs: int,
) -> dict[str, list[PredictedWave]]:
    filtered: dict[str, list[PredictedWave]] = {}
    for wave, values in predictions.items():
        wave_references = references.get(wave, [])
        if not wave_references:
            filtered[wave] = []
            continue
        peaks = np.asarray(
            [reference.peak for reference in wave_references],
            dtype=float,
        )
        if len(peaks) >= 2:
            typical_cycle = float(np.median(np.diff(peaks)))
            margin = max(1.0, 0.5 * typical_cycle)
        else:
            margin = EVENT_MATCH_TOLERANCE_MS[wave] / 1000.0 * fs
        lower = float(peaks[0] - margin)
        upper = float(peaks[-1] + margin)
        filtered[wave] = [
            prediction
            for prediction in values
            if lower <= prediction.peak <= upper
        ]
    return filtered


def _load_ludb_signal(record_path: Path) -> tuple[np.ndarray, int]:
    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None:
        raise ValueError("LUDB record has no physical signal")
    fs = int(round(float(record.fs)))
    if fs <= 0:
        raise ValueError(f"Invalid sampling rate {record.fs}")
    normalized_names = [_normalize_lead(name) for name in record.sig_name]
    required = [_normalize_lead(name) for name in LEAD_NAMES_12]
    missing = [
        name
        for name, normalized in zip(LEAD_NAMES_12, required)
        if normalized not in normalized_names
    ]
    if missing:
        raise ValueError(f"Missing required LUDB leads: {missing}")
    indices = [normalized_names.index(name) for name in required]
    signal = np.asarray(record.p_signal[:, indices], dtype=float)
    units = [str(record.units[index]).strip().lower() for index in indices]
    for index, unit in enumerate(units):
        if unit == "mv":
            continue
        if unit in {"v", "volt", "volts"}:
            signal[:, index] *= 1000.0
        elif unit in {"uv", "µv", "microvolt", "microvolts"}:
            signal[:, index] /= 1000.0
        else:
            raise ValueError(
                f"Unsupported unit {record.units[indices[index]]!r} "
                f"for lead {LEAD_NAMES_12[index]}"
            )
    if not np.all(np.isfinite(signal)):
        raise ValueError("LUDB signal contains non-finite samples")
    return signal, fs


def _load_ludb_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for raw in reader:
            row = {
                str(key).strip(): (
                    "" if value is None else str(value).strip()
                )
                for key, value in raw.items()
            }
            record_id = row.get("ID", "")
            if record_id:
                rows[str(int(float(record_id)))] = row
        return rows


def _record_groups(metadata: dict[str, str]) -> list[str]:
    rhythm = metadata.get("Rhythms", "") or "unknown"
    normalized_rhythm = _normalize_text(rhythm) or "unknown"
    groups = ["all", f"rhythm:{normalized_rhythm}"]
    combined = " ".join(metadata.values()).lower()
    challenges = []
    if "atrial fibrillation" in combined or "atrial flutter" in combined:
        challenges.append("af_or_flutter")
    if metadata.get("Cardiac pacing", ""):
        challenges.append("pacing")
    if metadata.get("Conduction abnormalities", ""):
        challenges.append("conduction")
    if metadata.get("Extrasystolies", ""):
        challenges.append("extrasystole")
    if (
        metadata.get("Ischemia", "")
        or metadata.get(
            "Non-specific repolarization abnormalities",
            "",
        )
    ):
        challenges.append("ischemia_or_repolarization")
    if "bradycardia" in combined or "tachycardia" in combined:
        challenges.append("brady_or_tachy")
    groups.extend(f"challenge:{name}" for name in challenges)
    if challenges:
        groups.append("challenge:any")
    return list(dict.fromkeys(groups))


def _error_summary(values: list[float], tolerance_ms: float) -> dict:
    if not values:
        return {
            "n": 0,
            "mean_signed": None,
            "standard_deviation": None,
            "median_signed": None,
            "mean_absolute": None,
            "median_absolute": None,
            "p95_absolute": None,
            "tolerance_ms": tolerance_ms,
            "within_tolerance_rate": None,
        }
    array = np.asarray(values, dtype=float)
    absolute = np.abs(array)
    return {
        "n": int(len(array)),
        "mean_signed": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1))
        if len(array) > 1
        else 0.0,
        "median_signed": float(np.median(array)),
        "mean_absolute": float(np.mean(absolute)),
        "median_absolute": float(np.median(absolute)),
        "p95_absolute": float(np.percentile(absolute, 95)),
        "tolerance_ms": tolerance_ms,
        "within_tolerance_rate": float(np.mean(absolute <= tolerance_ms)),
    }


def _distribution_summary(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "median": None,
            "p05": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "median": None,
            "p05": None,
            "p95": None,
        }
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def _implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "preprocessing.py",
        root / "running_median.py",
        root / "vcg_transform.py",
        *sorted((root / "delineation").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robust_spread_ms(values: np.ndarray, fs: int) -> float:
    if values.size < 2:
        return 0.0
    return float(
        (np.percentile(values, 90) - np.percentile(values, 10))
        / fs
        * 1000.0
    )


def _normalize_lead(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_")


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None
