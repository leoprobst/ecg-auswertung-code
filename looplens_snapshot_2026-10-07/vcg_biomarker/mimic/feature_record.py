"""Everything that turns one recording into marker values.

Split out of :mod:`vcg_biomarker.mimic.feature_extraction` so that
``feature_implementation_hash`` covers what a marker *is* and not how the stage
that computes it is scheduled.  Before the split, editing the partitioning or
the resume bookkeeping moved the marker hash and invalidated every extracted
recording -- 630,044 of them on the real project -- although no marker value
would have changed.

Nothing here may import the orchestrator: the dependency runs one way, and that
is what keeps the hash meaningful.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Optional

import numpy as np

from ..core.biomarkers import BiomarkerRegistry
from ..core.delineation.base import ventricular_rr_seconds
from ..core.delineation.fusion_delineator import FusionDelineator
from ..core.preprocessing import (
    construct_robust_morphology_record,
    detect_r_peaks_multilead,
)
from ..core.vcg_transform import VCGTransformRegistry
from .classical import ClassicalECGCriteria
from .report_parser import MachineReportParser, ReportMentionState
from .signals import MimicWaveformLoader, compute_vcg_sensitivity_transforms


_FEATURE_WORKER_OPTIONS: Optional[dict] = None
_FEATURE_WORKER_LOADER: Optional[MimicWaveformLoader] = None
_FEATURE_WORKER_DELINEATOR: Optional[FusionDelineator] = None
_REPORT_PARSER = MachineReportParser()
HF_QRS_LOWER_HZ = 80.0
LEAD_PLACEMENT_EXCLUDED_STATUS = "lead_placement_excluded"
HF_QRS_UPPER_HZ = 150.0


def _acquisition_upper_bandwidth_hz(value: object) -> Optional[float]:
    """Parse the recorder's analogue upper corner, never the ADC Nyquist.

    MIMIC stores values such as ``0.005-150 Hz``.  Missing or ambiguous
    metadata must not silently license a high-frequency measurement.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    numbers = []
    for token in re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", text):
        try:
            parsed = float(token.replace(",", "."))
        except ValueError:
            continue
        if np.isfinite(parsed) and parsed > 0:
            numbers.append(parsed)
    if not numbers:
        return None
    return float(max(numbers))


def _wideband_acquisition_support(value: object) -> tuple[bool, Optional[float], str]:
    upper = _acquisition_upper_bandwidth_hz(value)
    if upper is None:
        return False, None, "missing_or_unparseable_acquisition_bandwidth"
    if upper + 1e-9 < HF_QRS_UPPER_HZ:
        return False, upper, "acquisition_bandwidth_below_150_hz"
    return True, upper, "supported_80_150_hz"


def _initialize_feature_worker(task_config: dict) -> None:
    """Initialize immutable extraction state once per worker process."""

    from .resources import apply_worker_thread_limits

    apply_worker_thread_limits()
    global _FEATURE_WORKER_OPTIONS
    global _FEATURE_WORKER_LOADER
    global _FEATURE_WORKER_DELINEATOR
    _FEATURE_WORKER_OPTIONS = dict(task_config)
    _FEATURE_WORKER_LOADER = MimicWaveformLoader(target_sampling_rate=500)
    _FEATURE_WORKER_DELINEATOR = FusionDelineator(
        weights=task_config["delineation_weights"],
        conflict_threshold=task_config["delineation_conflict_threshold_ms"],
    )


def _extract_record_batch_worker(rows: list[dict]) -> list[dict]:
    """Return a small batch so repeated result keys are pickled only once."""

    return [_extract_record_worker(row) for row in rows]


def _extract_record_worker(task: dict | tuple[dict, dict]) -> dict:
    explicit_options = isinstance(task, tuple)
    if explicit_options:
        row, options = task
    else:
        row = task
        options = _FEATURE_WORKER_OPTIONS
        if options is None:
            raise RuntimeError("Feature worker was not initialized")
    base = {
        "subject_id": int(row["subject_id"]),
        "study_id": int(row["study_id"]),
        "split": row["split"],
        "ecg_time": row["ecg_time"],
        "extraction_status": "error",
        "error_type": "",
        "error_message": "",
        "biomarker_calculation_failure_count": 0,
        "biomarker_calculation_failure_names": "",
        "biomarker_calculation_failure_details": "[]",
        "context_temporal_research_json": "",
        "context_measurement_capsule_json": "",
        "context_measurement_sensitivity_json": "",
    }
    calculation_failures: list[dict[str, str]] = []
    try:
        loader = (
            MimicWaveformLoader(target_sampling_rate=500)
            if explicit_options or _FEATURE_WORKER_LOADER is None
            else _FEATURE_WORKER_LOADER
        )
        loaded = loader.load(
            row["waveform_path"],
            expected_subject_id=int(row["subject_id"]),
            expected_study_id=int(row["study_id"]),
        )
        # Audit X8: a content hash of the loaded waveform. MIMIC holds stored
        # duplicates under consecutive study_ids (5 of 115 same-timestamp
        # pairs were identical in every value); only the signal can tell a
        # duplicate from a genuine second recording.
        import hashlib

        base["signal_sha256"] = hashlib.sha256(
            np.ascontiguousarray(loaded.signal_millivolts, dtype=np.float64).tobytes()
        ).hexdigest()
        base.update(
            {
                "signal_quality_score": loaded.quality.score,
                "signal_quality_tier": loaded.quality.tier,
                "signal_quality_usable": loaded.quality.usable,
                "signal_quality_reasons": "|".join(loaded.quality.reasons),
                "original_sampling_rate": loaded.original_sampling_rate,
            }
        )
        # A repaired recording must never be indistinguishable from one that
        # arrived intact. Whether short gaps were filled travels with the row,
        # so any analysis can exclude them and any reader can see how many
        # there were.
        if loaded.repair is not None:
            base.update(loaded.repair.to_dict())
        if not loaded.quality.usable:
            base["extraction_status"] = "signal_qc_failed"
            base["error_type"] = "SignalQualityError"
            base["error_message"] = "|".join(loaded.quality.reasons)
            return base
        if loaded.quality.limb_lead_reversal_suspected:
            # Audit S6: with RA/LA reversed, lead I is inverted and II/III
            # swap; the Kors transform weights I and II heavily, so the
            # frontal-plane VCG is mirrored and every loop marker is wrong in
            # a systematic, not a noisy, way. 0.24 % of recordings. Excluded
            # from all analyses instead of carried with a 0.10 quality
            # deduction.
            base["extraction_status"] = LEAD_PLACEMENT_EXCLUDED_STATUS
            base["error_type"] = "LeadPlacementExclusion"
            base["error_message"] = "|".join(loaded.quality.reasons)
            return base

        paths = loader.prepare_paths(
            loaded,
            powerline_hz=float(options["powerline_hz"]),
            device_filtering=row.get("machine_filtering"),
        )
        wideband_supported, acquisition_upper_hz, wideband_reason = (
            _wideband_acquisition_support(row.get("machine_bandwidth"))
        )
        base.update(
            {
                "machine_bandwidth": str(row.get("machine_bandwidth") or ""),
                "acquisition_upper_bandwidth_hz": (
                    acquisition_upper_hz
                    if acquisition_upper_hz is not None
                    else np.nan
                ),
                "hf_qrs_effective_band_hz": (
                    f"{HF_QRS_LOWER_HZ:g}-{HF_QRS_UPPER_HZ:g}"
                    if wideband_supported
                    else ""
                ),
                "hf_qrs_acquisition_supported": wideband_supported,
                "hf_qrs_acquisition_status": wideband_reason,
            }
        )
        r_peaks = detect_r_peaks_multilead(
            paths["delineation"],
            loaded.sampling_rate,
        )
        if len(r_peaks) < 2:
            raise ValueError(f"Only {len(r_peaks)} R peak(s) detected")
        vcg_delineation = VCGTransformRegistry.transform(
            paths["delineation"],
            "kors",
        )
        record_vcg_morphology = VCGTransformRegistry.transform(
            paths["morphology"],
            "kors",
        )
        # HF-QRS is exposed only when the analogue front-end metadata support
        # the complete prespecified 80--150-Hz band.  Sampling rate alone is
        # not evidence of analogue bandwidth.
        record_vcg_wideband = (
            record_vcg_morphology
            if wideband_supported and "wideband" in paths
            else None
        )
        delineator = (
            FusionDelineator(
                weights=options["delineation_weights"],
                conflict_threshold=options["delineation_conflict_threshold_ms"],
            )
            if explicit_options or _FEATURE_WORKER_DELINEATOR is None
            else _FEATURE_WORKER_DELINEATOR
        )
        delineation = delineator.delineate_record(
            ecg_12lead=paths["delineation"],
            r_peaks=r_peaks,
            fs=loaded.sampling_rate,
            ecg_id=int(row["study_id"]),
            vcg_xyz=vcg_delineation.xyz,
        )
        if not delineation.beats:
            raise ValueError("No beats could be delineated")
        parsed_report = _REPORT_PARSER.parse(str(row.get("machine_report") or ""))
        # A definite atrial fibrillation/flutter statement from the recorder
        # overrides a P wave found by the delineation (conducted flutter; see
        # ``RecordDelineation.suppress_discrete_p``). None of the 84,744
        # definite statements in MIMIC-IV mentions sinus rhythm. "Possible" and
        # "probable" statements (10,403) are left to the delineation.
        base["context_report_af_suppressed_p"] = bool(
            parsed_report["af"].state == ReportMentionState.POSITIVE
            and delineation.suppress_discrete_p(
                "discrete_p_suppressed_by_report_atrial_fibrillation_flutter"
            )
        )
        robust_morphology, template_diagnostics = (
            construct_robust_morphology_record(
                paths["morphology"],
                delineation,
            )
        )
        vcg_morphology = VCGTransformRegistry.transform(
            robust_morphology,
            "kors",
        )
        ndat_acquisition = {
            "original_sampling_rate_hz": loaded.original_sampling_rate,
            "upper_bandwidth_hz": acquisition_upper_hz,
            "pacing_present": True if parsed_report["pacemaker"].state == ReportMentionState.POSITIVE else None,
            "pacing_source": "machine_report_positive_only_not_independent_adjudication",
        }
        biomarker_results = BiomarkerRegistry.compute_all(
            delineation=delineation,
            ecg_12lead=robust_morphology,
            vcg_xyz=vcg_morphology.xyz,
            fs=loaded.sampling_rate,
            names=options["vcg_feature_ids"],
            record_ecg_12lead=paths["morphology"],
            record_raw_ecg_12lead=loaded.signal_millivolts,
            record_acquisition=ndat_acquisition,
            record_vcg_xyz=record_vcg_morphology.xyz,
            record_vcg_wideband_xyz=(
                record_vcg_wideband.xyz
                if record_vcg_wideband is not None
                else None
            ),
        )
        feature_metadata: dict[str, dict] = {}
        # Audit S7: the 80-150 Hz markers are only reportable next to the
        # band's own signal-to-noise ratio.
        from .hf_qrs_quality import EMPTY_HF_QRS_QUALITY, hf_qrs_signal_quality
        base.update(
            hf_qrs_signal_quality(
                record_vcg_wideband.xyz,
                delineation.accepted_beats,
                loaded.sampling_rate,
            )
            if record_vcg_wideband is not None
            else EMPTY_HF_QRS_QUALITY
        )
        from ..core.temporal_research import record_descriptor
        base["context_temporal_research_json"] = json.dumps(record_descriptor(
            delineation, record_vcg_morphology.xyz, loaded.sampling_rate,
            source="mimic", subject_id=row["subject_id"], transform="kors",
        ), allow_nan=False, separators=(",", ":"))
        from ..core.measurement_capsule import capsule_json
        from ..core.measurement_robustness import angle_sensitivity
        base["context_measurement_sensitivity_json"] = json.dumps(angle_sensitivity(
            paths["morphology"], delineation, loaded.sampling_rate, source="mimic", record_id=row["study_id"]),
            allow_nan=False, separators=(",", ":"))
        base["context_measurement_capsule_json"] = capsule_json(
            paths["morphology"], delineation, loaded.sampling_rate,
            source="mimic", record_id=row["study_id"], raw_reference=row["waveform_path"],
            processing={"signal_path": "record_morphology", "powerline_hz": float(options["powerline_hz"]),
                        "original_sampling_rate_hz": float(loaded.original_sampling_rate),
                        "baseline": "prepared_signal_path_no_additional_capsule_correction",
                        "signal_quality": loaded.quality.score,
                        "template_diagnostics": template_diagnostics},
        )
        _store_biomarker_results(
            base,
            biomarker_results,
            calculation_failures,
            metadata=feature_metadata,
        )
        sensitivity_methods = tuple(options["sensitivity_vcg_methods"])
        if sensitivity_methods:
            sensitivity_transforms = compute_vcg_sensitivity_transforms(
                robust_morphology,
                methods=sensitivity_methods,
                kors_result=vcg_morphology,
            )
            record_sensitivity_transforms = (
                compute_vcg_sensitivity_transforms(
                    paths["morphology"],
                    methods=sensitivity_methods,
                    kors_result=record_vcg_morphology,
                )
            )
            for method in sensitivity_methods:
                if method not in sensitivity_transforms:
                    raise ValueError(
                        f"Unsupported VCG sensitivity transform: {method}"
                    )
                if method not in record_sensitivity_transforms:
                    raise ValueError(
                        "Missing record-level VCG sensitivity transform: "
                        f"{method}"
                    )
                transformed_results = BiomarkerRegistry.compute_all(
                    delineation=delineation,
                    ecg_12lead=robust_morphology,
                    vcg_xyz=sensitivity_transforms[method].xyz,
                    fs=loaded.sampling_rate,
                    names=options["sensitivity_feature_ids"],
                    record_ecg_12lead=paths["morphology"],
                    record_raw_ecg_12lead=loaded.signal_millivolts,
                    record_acquisition=ndat_acquisition,
                    record_vcg_xyz=(
                        record_sensitivity_transforms[method].xyz
                    ),
                )
                _store_biomarker_results(
                    base,
                    transformed_results,
                    calculation_failures,
                    prefix=f"sensitivity_{method}__",
                    metadata=feature_metadata,
                )

        paced_or_lbbb = (
            parsed_report["pacemaker"].state == ReportMentionState.POSITIVE
            or parsed_report["lbbb"].state == ReportMentionState.POSITIVE
        )
        # Only the exclusion above is taken from the parse here. The other
        # twelve findings are deliberately not written into the feature row:
        # the ``parse-reports`` stage already stores all fourteen as
        # ``report_*_state`` in ``machine_report_labels.parquet``, and
        # ``validation.py`` joins that table beside the features. A second
        # copy computed in a different stage, behind a different reuse key,
        # is how two sources of the same fact drift apart.
        lead_measurements: dict[str, float] = {}
        classical = ClassicalECGCriteria().compute(
            paths["diagnostic"],
            delineation,
            age=_optional_int(row.get("age")),
            sex=row.get("sex"),
            machine_qrs_axis_deg=_optional_float(
                row.get("machine_qrs_axis_deg")
            ),
            machine_t_axis_deg=_optional_float(
                row.get("machine_t_axis_deg")
            ),
            paced_or_lbbb=paced_or_lbbb,
            paced=parsed_report["pacemaker"].state == ReportMentionState.POSITIVE,
            measurement_sink=lead_measurements,
        )
        from .lead_measurements import LEAD_MEASUREMENT_VERSION
        base.update(lead_measurements)
        base["lead_measurement_version"] = LEAD_MEASUREMENT_VERSION
        base["lead_measurement_scope"] = "diagnostic_12lead_shared_fiducials"
        for feature_id in options["classical_feature_ids"]:
            criterion_id = feature_id.removeprefix("classical_")
            criterion = classical.get(criterion_id)
            base[feature_id] = (
                criterion.value
                if criterion is not None and criterion.evaluable
                else np.nan
            )
            base[f"{feature_id}_positive"] = (
                criterion.positive
                if criterion is not None and criterion.evaluable
                else None
            )
        # Heart rate as context, not as a marker.
        #
        # Rate belongs in every adjustment: in the last run's phenome screen
        # ``heart_rate`` was itself a screened marker and duly came out as the
        # best predictor of sepsis (0,50), acidosis (0,45), respiratory
        # failure (0,41), disseminated intravascular coagulation (0,46) and
        # rhabdomyolysis (0,30) -- true, and not a finding about the
        # electrocardiogram. It was then removed from the panels, which left
        # the opposite problem: no stage could adjust for it, because the
        # value was no longer written anywhere.
        #
        # ``context_`` names it for what it is. Context columns are not part
        # of any panel, are not screened and do not enlarge an FDR family;
        # they exist so that every model downstream can hold rate fixed. The
        # mean RR is the same one the interval markers use, so the context
        # column and ``heart_rate_bpm`` cannot disagree.
        rr_seconds = ventricular_rr_seconds(
            (beat.r_peak.sample for beat in delineation.beats),
            delineation.fs,
        )
        base["context_rr_interval_ms"] = (
            None if rr_seconds is None else rr_seconds * 1000.0
        )
        base["context_heart_rate_bpm"] = (
            None
            if rr_seconds is None or rr_seconds <= 0
            else 60.0 / rr_seconds
        )
        # The recorder's own intervals beside the measured ones, as context.
        # The own QRS duration is a union over leads and sits above the
        # global device value by design; a downstream reader of any duration
        # marker can hold the device reading next to it without a join.
        for interval in ("qrs_duration_ms", "qt_interval_ms", "pr_interval_ms"):
            base[f"context_machine_{interval}"] = _optional_float(
                row.get(f"machine_{interval}")
            )
        base["detected_beat_count"] = delineation.n_beats
        base["plausible_beat_count"] = delineation.n_plausible_beats
        base["accepted_beat_count"] = delineation.n_accepted_beats
        base["representative_beat_available"] = (
            delineation.representative_beat is not None
        )
        base["morphology_template_status"] = template_diagnostics["status"]
        base["morphology_template_reason"] = template_diagnostics.get(
            "reason",
            "",
        )
        base["morphology_template_candidate_beat_count"] = (
            template_diagnostics["candidate_beat_count"]
        )
        base["morphology_template_used_beat_count"] = (
            template_diagnostics["used_beat_count"]
        )
        base["morphology_template_outlier_count"] = template_diagnostics[
            "excluded_morphology_outlier_count"
        ]
        base["fiducial_confidence"] = delineation.quality_score
        base["delineation_review_required"] = delineation.review_required
        base["delineation_review_reasons"] = "|".join(
            delineation.review_reasons
        )
        base["rhythm_context"] = delineation.rhythm.rhythm_class.value
        base["rhythm_context_confidence"] = delineation.rhythm.confidence
        base["rr_cv"] = delineation.rhythm.rr_cv
        base["rr_successive_irregularity"] = (
            delineation.rhythm.rr_successive_irregularity
        )
        base["rr_lag1_correlation"] = (
            delineation.rhythm.rr_lag1_correlation
        )
        base["rr_turning_point_fraction"] = (
            delineation.rhythm.rr_turning_point_fraction
        )
        base["p_wave_coverage"] = delineation.rhythm.p_wave_coverage
        base["p_candidate_coverage"] = (
            delineation.rhythm.p_candidate_coverage
        )
        base["p_candidate_timing_mad_ms"] = (
            delineation.rhythm.p_candidate_timing_mad_ms
        )
        base["p_candidate_timing_p10_p90_ms"] = (
            delineation.rhythm.p_candidate_timing_p10_p90_ms
        )
        base["atrial_frequency_hz"] = (
            delineation.rhythm.atrial_frequency_hz
        )
        base["atrial_spectral_concentration"] = (
            delineation.rhythm.atrial_spectral_concentration
        )
        base["atrial_bandpower_ratio"] = (
            delineation.rhythm.atrial_bandpower_ratio
        )
        base["p_fiducial_confidence"] = delineation.segment_confidence("p")
        base["pr_fiducial_confidence"] = delineation.segment_confidence("pr")
        base["qrs_fiducial_confidence"] = delineation.segment_confidence("qrs")
        base["t_fiducial_confidence"] = delineation.segment_confidence("t")
        base["qrst_fiducial_confidence"] = delineation.segment_confidence("qrst")
        base["detector_conflict_fraction"] = delineation.diagnostics.get(
            "marker_conflict_fraction",
            np.nan,
        )
        _finalize_biomarker_failure_provenance(
            base,
            calculation_failures,
        )
        _store_feature_metadata(base, feature_metadata)
        base["extraction_status"] = "ok"
        return base
    except Exception as exc:
        _finalize_biomarker_failure_provenance(
            base,
            calculation_failures,
        )
        base["error_type"] = type(exc).__name__
        base["error_message"] = str(exc)[:1000]
        return base


def _store_biomarker_results(
    destination: dict,
    results: Iterable,
    calculation_failures: list[dict[str, str]],
    *,
    prefix: str = "",
    metadata: Optional[dict[str, dict]] = None,
) -> None:
    """Persist each marker value together with the metadata behind it.

    Only ``value`` used to survive this function. Everything that explains a
    value -- the marker's own confidence, its per-lead breakdown, and above
    all the *reason* a value is missing ("QRS segment too short" versus "no
    wideband source" versus "T/U fusion") -- was discarded, so a NaN in the
    feature table was indistinguishable from any other NaN. Recomputing the
    waveforms is the expensive step in this pipeline; throwing away the
    metadata means paying it again to answer questions that were already
    answered once.

    Non-scalar per-lead detail is serialised once per record into a compact
    JSON column rather than exploded into thousands of parquet columns.
    """

    per_feature_confidence: dict[str, float] = {}
    per_lead_detail: dict[str, dict] = {}
    unevaluable_reasons: dict[str, str] = {}
    computation_details: dict[str, str] = {}
    for result in results:
        feature_id = f"{prefix}{result.name}"
        destination[feature_id] = result.value
        confidence = getattr(result, "confidence", None)
        if confidence is not None:
            try:
                per_feature_confidence[feature_id] = float(confidence)
            except (TypeError, ValueError):
                pass
        per_lead = getattr(result, "per_lead", None)
        if isinstance(per_lead, dict) and per_lead:
            per_lead_detail[feature_id] = {
                str(key): _optional_float(value)
                for key, value in per_lead.items()
            }
        computation_notes = str(result.computation_notes or "").strip()
        lowered = computation_notes.lower()
        if lowered.startswith("error:"):
            calculation_failures.append(
                {
                    "feature_id": feature_id,
                    "message": computation_notes[6:].strip()[:1000],
                }
            )
        elif computation_notes:
            # "not_evaluable: <reason>" and any other diagnostic note. This is
            # the abstention reason, which is a scientific result in itself.
            if _optional_float(result.value) is None:
                unevaluable_reasons[feature_id] = computation_notes[:300]
            else:
                computation_details[feature_id] = computation_notes[:300]
    if metadata is None:
        return
    metadata.setdefault("confidence", {}).update(per_feature_confidence)
    metadata.setdefault("per_lead", {}).update(per_lead_detail)
    metadata.setdefault("unevaluable_reason", {}).update(unevaluable_reasons)
    metadata.setdefault("computation_notes", {}).update(computation_details)


def _store_feature_metadata(
    destination: dict,
    metadata: dict[str, dict],
) -> None:
    """Persist per-feature confidence, per-lead detail and abstention reasons.

    Serialised as three compact JSON columns rather than one column per
    feature: the catalog holds over a thousand markers, and exploding the
    metadata would triple the column count of every feature partition for
    data that is read selectively during review, not during screening.
    """

    confidence = metadata.get("confidence", {})
    per_lead = metadata.get("per_lead", {})
    reasons = metadata.get("unevaluable_reason", {})
    destination["feature_computation_notes_json"] = json.dumps(
        metadata.get("computation_notes", {}), ensure_ascii=True, sort_keys=True,
        separators=(",", ":"),
    )
    destination["feature_confidence_json"] = json.dumps(
        {key: value for key, value in sorted(confidence.items())},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    destination["feature_per_lead_json"] = json.dumps(
        {key: value for key, value in sorted(per_lead.items())},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    destination["feature_unevaluable_reason_json"] = json.dumps(
        {key: value for key, value in sorted(reasons.items())},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    destination["feature_unevaluable_count"] = len(reasons)
    destination["feature_low_confidence_count"] = sum(
        1
        for value in confidence.values()
        if isinstance(value, (int, float)) and float(value) < 0.5
    )


def _finalize_biomarker_failure_provenance(
    destination: dict,
    calculation_failures: list[dict[str, str]],
) -> None:
    destination["biomarker_calculation_failure_count"] = len(
        calculation_failures
    )
    destination["biomarker_calculation_failure_names"] = "|".join(
        failure["feature_id"] for failure in calculation_failures
    )
    destination["biomarker_calculation_failure_details"] = json.dumps(
        calculation_failures,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _optional_float(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _optional_int(value) -> Optional[int]:
    converted = _optional_float(value)
    return int(converted) if converted is not None else None
