from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, cast

import numpy as np
import pandas as pd

from ..core.biomarkers import BiomarkerRegistry
from ..core.data_loader import LEAD_NAMES_12
from ..core.delineation.base import (
    BeatDelineation,
    MarkerPoint,
    RecordDelineation,
    WaveStatus,
)
from ..core.delineation.fusion_delineator import FusionDelineator
from ..core.external_loader import (
    ExternalECGMeta,
    load_external_ecg_result,
)
from ..core.preprocessing import (
    construct_robust_morphology_record,
    detect_r_peaks_multilead,
    prepare_signal_paths,
    resample_signal,
)
from ..core.vcg_transform import VCGTransformRegistry
from .classical import ClassicalECGCriteria
from .baseline_registry import BaselineRegistry
from .config import MimicProjectConfig
from .feature_roles import (
    ROLE_CANDIDATE,
    canonical_feature_id,
    feature_profile,
)
from .models import (
    AnnotationRevision,
    CriterionStage,
    EvidenceGrade,
    Finding,
    FrozenScoreSpec,
    ImportedECGCase,
    ScoreReleaseStage,
    ThresholdBand,
)
from .panels import default_feature_panels
from .projection_hypotheses import projection_case_eligibility
from .signals import assess_signal_quality
from ..core.delineation.base import measurement_confidence_threshold
from .storage import (
    StageManifest,
    atomic_replace,
    atomic_write_json,
    content_hash,
    file_sha256,
    source_tree_hash,
)
from .user_guidance import (
    biomarker_display_name,
    biomarker_tooltip,
    clinical_display_term,
)
from .validation_provenance import validation_implementation_hash


CASE_LIBRARY_VERSION = "2.24.2"

_RESEARCH_WARNING_DE = (
    "Nur Forschungsbefund: kein klinisch freigegebenes Diagnosekriterium "
    "und keine Diagnosewahrscheinlichkeit. Signalqualitaet, Wellengrenzen "
    "und klinischer Kontext muessen fachlich geprueft werden."
)
_RESEARCH_DIAGNOSIS_STAGES = frozenset(
    {
        CriterionStage.VALIDATION_REPLICATED.value,
        CriterionStage.FROZEN.value,
        CriterionStage.LOCKED_TEST_VALIDATED.value,
        CriterionStage.EXTERNAL_VALIDATED.value,
    }
)

_FIDUCIAL_ALIASES = {
    "qrs_onset": "r_onset",
    "qrs_offset": "r_offset",
}
_EDITABLE_FIDUCIALS = (
    "p_onset",
    "p_peak",
    "p_offset",
    "q_peak",
    "qrs_onset",
    "r_peak",
    "qrs_offset",
    "t_onset",
    "t_peak",
    "t_offset",
    "u_onset",
    "u_peak",
    "u_offset",
)


class ECGCaseLibrary:
    def __init__(self, config: MimicProjectConfig):
        self.config = config
        self.root = config.cases_root
        self.root.mkdir(parents=True, exist_ok=True)

    def import_case(
        self,
        source_path: str | Path,
        *,
        metadata: Optional[ExternalECGMeta] = None,
        pseudonym: Optional[str] = None,
        retain_source_path: bool = False,
        copy_original: bool = False,
    ) -> ImportedECGCase:
        source = Path(source_path).resolve()
        import_metadata = metadata or ExternalECGMeta()
        result = load_external_ecg_result(source, import_metadata)
        source_files = _source_bundle_paths(source, result.source_format)
        source_hash = _source_bundle_fingerprint(source_files)
        case_id = f"case-{source_hash[:16]}"
        case_root = self.root / case_id
        case_json = case_root / "case.json"
        if case_json.is_file():
            return self.load_case(case_id)
        case_root.mkdir(parents=True, exist_ok=True)
        signal_path = case_root / "signal.npz"
        _atomic_write_npz(
            signal_path,
            signal_millivolts=result.signal_millivolts.astype(np.float32),
            sampling_rate_hz=np.asarray(result.sampling_rate_hz),
            lead_names=np.asarray(result.lead_names),
            source_units=np.asarray(result.source_units),
        )
        if copy_original:
            original_root = case_root / "source"
            original_root.mkdir(parents=True, exist_ok=True)
            _copy_source_bundle_atomic(
                source_files,
                source_root=(
                    source.with_suffix("").parent
                    if result.source_format == "wfdb"
                    else source.parent
                ),
                destination_root=original_root,
            )
        case = ImportedECGCase(
            case_id=case_id,
            pseudonym=pseudonym or f"ECG-{source_hash[:8].upper()}",
            source_format=result.source_format,
            source_path=str(source) if retain_source_path else "<local-redacted>",
            source_sha256=source_hash,
            sampling_rate_hz=result.sampling_rate_hz,
            lead_names=result.lead_names,
            units=tuple("mV" for _ in LEAD_NAMES_12),
            metadata={
                "library_version": CASE_LIBRARY_VERSION,
                "source_metadata": dict(result.source_metadata),
                "age": import_metadata.age,
                "sex": import_metadata.sex,
                "original_copied": bool(copy_original),
                "source_file_count": len(source_files),
                "source_fingerprint_kind": (
                    "wfdb_record_bundle_sha256"
                    if result.source_format == "wfdb"
                    else "single_file_sha256"
                ),
                "dua_notice": (
                    "Local research case. Do not transmit MIMIC-derived or "
                    "patient-identifying content to external services."
                ),
            },
        )
        atomic_write_json(case_json, case.to_dict())
        return case

    def load_case(self, case_id: str) -> ImportedECGCase:
        payload = json.loads(
            (self.root / case_id / "case.json").read_text(encoding="utf-8")
        )
        payload["lead_names"] = tuple(payload.get("lead_names", ()))
        payload["units"] = tuple(payload.get("units", ()))
        payload["metadata"] = dict(payload.get("metadata", {}))
        return ImportedECGCase(**payload)

    def load_signal(self, case_id: str) -> tuple[np.ndarray, float]:
        path = self.root / case_id / "signal.npz"
        with np.load(path, allow_pickle=False) as archive:
            signal = np.asarray(archive["signal_millivolts"], dtype=float)
            sampling_rate = float(np.asarray(archive["sampling_rate_hz"]).item())
        return signal, sampling_rate

    def list_cases(self) -> tuple[ImportedECGCase, ...]:
        cases = []
        for path in sorted(self.root.glob("case-*/case.json")):
            try:
                cases.append(self.load_case(path.parent.name))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return tuple(cases)

    def add_annotation(
        self,
        case_id: str,
        *,
        fiducials: dict[str, int],
        notes: str = "",
        author: str = "local_user",
        parent_revision_id: str = "",
    ) -> AnnotationRevision:
        self.load_case(case_id)
        revision = AnnotationRevision(
            revision_id=f"annotation-{uuid.uuid4().hex}",
            case_id=case_id,
            parent_revision_id=parent_revision_id,
            fiducials={str(key): int(value) for key, value in fiducials.items()},
            notes=notes,
            author=author,
        )
        path = (
            self.root
            / case_id
            / "annotations"
            / f"{revision.created_at.replace(':', '')}-{revision.revision_id}.json"
        )
        atomic_write_json(path, revision.to_dict())
        return revision

    def annotations(self, case_id: str) -> tuple[AnnotationRevision, ...]:
        root = self.root / case_id / "annotations"
        revisions = []
        for path in sorted(root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["fiducials"] = dict(payload.get("fiducials", {}))
            revisions.append(AnnotationRevision(**payload))
        return tuple(revisions)

    def analyze_case(
        self,
        case_id: str,
        *,
        panel_ids: tuple[str, ...] = ("primary_core",),
        vcg_method: str = "kors",
        powerline_hz: float = 0.0,
        acquisition_metadata: Optional[dict] = None,
        ischemia_context: Optional[dict] = None,
        resource_checkpoint: Optional[Callable[[], None]] = None,
    ) -> dict[str, Any]:
        checkpoint = resource_checkpoint or (lambda: None)
        checkpoint()
        case = self.load_case(case_id)
        from .ischemia_patterns import validate_lab_ischemia_context
        ischemia_context = validate_lab_ischemia_context(ischemia_context, case_id)
        signal, original_fs = self.load_signal(case_id)
        target_fs = 500
        if not np.isclose(original_fs, target_fs, rtol=0, atol=1e-12):
            signal = resample_signal(
                signal,
                original_fs,
                target_fs,
            )
        quality = assess_signal_quality(signal, target_fs)
        if not quality.usable:
            return self._write_blocked_quality_analysis(
                case,
                signal,
                target_fs,
                panel_ids=panel_ids,
                vcg_method=vcg_method,
                powerline_hz=powerline_hz,
                quality=quality,
            )
        paths = prepare_signal_paths(
            signal,
            target_fs,
            powerline=powerline_hz,
        )
        r_peaks = detect_r_peaks_multilead(paths["delineation"], target_fs)
        if len(r_peaks) < 2:
            raise ValueError(
                f"Research analysis requires at least two detected beats; got {len(r_peaks)}"
            )
        vcg_delineation = VCGTransformRegistry.transform(
            paths["delineation"],
            vcg_method,
        )
        record_vcg_morphology = VCGTransformRegistry.transform(
            paths["morphology"],
            vcg_method,
        )
        delineator = FusionDelineator(
            weights={"neurokit": 0.4, "multilead": 0.35, "vcg": 0.25},
            conflict_threshold=20.0,
            capture_candidates=True,
        )
        checkpoint()
        delineation = delineator.delineate_record(
            ecg_12lead=paths["delineation"],
            r_peaks=r_peaks,
            fs=target_fs,
            ecg_id=0,
            vcg_xyz=vcg_delineation.xyz,
        )
        annotation = self._latest_annotation(case_id)
        if annotation is not None:
            _apply_annotation_revision(delineation, annotation)
            delineator.refresh_record_quality(delineation)
        robust_morphology, template_diagnostics = (
            construct_robust_morphology_record(
                paths["morphology"],
                delineation,
            )
        )
        vcg_morphology = VCGTransformRegistry.transform(
            robust_morphology,
            vcg_method,
        )
        panels = {item.panel_id: item for item in default_feature_panels()}
        unknown = set(panel_ids) - set(panels)
        if unknown:
            raise ValueError(f"Unknown feature panels: {sorted(unknown)}")
        evidence_run_root, evidence_run_inventory = (
            _select_case_evidence_run(self.config)
        )
        all_releases = _load_score_releases(
            self.config,
            run_root=evidence_run_root,
        )
        compatible_releases = [
            release
            for release in all_releases
            if release.get("implementation_compatible") is True
        ]
        releases = _recommended_score_releases(compatible_releases)
        (
            single_marker_candidates,
            single_marker_inventory,
        ) = _load_recommended_single_marker_candidates(
            self.config,
            run_root=evidence_run_root,
        )
        criterion_candidates, criterion_inventory = (
            _load_case_criterion_candidates(
                self.config,
                run_root=evidence_run_root,
            )
        )
        release_components = _release_component_names(releases)
        single_marker_components = tuple(
            str(item.get("feature_id", ""))
            for item in single_marker_candidates
            if item.get("feature_id")
        )
        selected = tuple(
            dict.fromkeys(
                [
                    feature_id
                    for panel_id in panel_ids
                    for feature_id in panels[panel_id].feature_ids
                ]
                + list(release_components)
                + list(single_marker_components)
                + list(
                    _criterion_component_names(criterion_candidates)
                )
            )
        )
        vcg_features = tuple(
            item for item in selected if not item.startswith("classical_")
        )
        from ..core.biomarkers.ndat_biomarkers import NDAT_MARKER_IDS
        direct8_inspection_ids = tuple(
            f"qrs_t_{method}_angle_direct_8lead_{baseline}_deg"
            for method in ("area", "integral") for baseline in ("preqrs", "tp")) + tuple(
            f"qrs_t_{method}_angle_direct_8lead_beat_{summary}{suffix}_deg"
            for method in ("area", "integral") for summary in ("median", "mad") for suffix in ('', '_preqrs'))
        acquisition = dict(case.metadata.get("source_metadata", {}).get("acquisition", {}))
        if acquisition_metadata is not None:
            acquisition.update(acquisition_metadata)
        # The stored waveform rate is authoritative for this case; an explicitly
        # retained pre-resampling rate may be lower and must not be overwritten.
        acquisition.setdefault("original_sampling_rate_hz", original_fs)
        checkpoint()
        feature_results = BiomarkerRegistry.compute_all(
            delineation=delineation,
            ecg_12lead=robust_morphology,
            vcg_xyz=vcg_morphology.xyz,
            fs=target_fs,
            names=list(dict.fromkeys([*vcg_features, *NDAT_MARKER_IDS, *direct8_inspection_ids])),
            record_ecg_12lead=paths["morphology"],
            record_raw_ecg_12lead=signal,
            record_acquisition=dict(acquisition, lab_capture_measurement_windows=True),
            record_vcg_xyz=record_vcg_morphology.xyz,
        )
        ndat_result = {}
        for result in feature_results:
            if result.name == NDAT_MARKER_IDS[0]:
                try:
                    ndat_result = json.loads(result.computation_notes)
                except (ValueError, TypeError):
                    ndat_result = {"status": "not_evaluable", "reason": "ndat_computation_failed",
                                   "detail": str(result.computation_notes)}
                break
        feature_values = {
            result.name: (
                float(result.value)
                if result.value is not None and np.isfinite(result.value)
                else None
            )
            for result in feature_results
        }
        from ..core.biomarkers.catalog_biomarkers import FeatureContext
        from ..core.biomarkers.qrs_t_method_comparison import compare_qrs_t_methods
        comparison_context = FeatureContext(delineation, robust_morphology, vcg_morphology.xyz, target_fs)
        method_comparison = compare_qrs_t_methods(robust_morphology,
            qrs_bounds=comparison_context.bounds("qrs"), t_bounds=comparison_context.bounds("t"),
            baseline_bounds=comparison_context.bounds("pr"),
            segment_evaluable={name: bool(comparison_context.is_segment_evaluable(name))
                               for name in ("qrs", "t", "pr")})
        lead_measurements: dict[str, float] = {}
        from ..core.biomarkers.qrs_t_segment_matrix import qrs_t_segment_matrix
        segment_matrix = qrs_t_segment_matrix(robust_morphology,
            qrs_bounds=comparison_context.bounds("qrs"), t_bounds=comparison_context.bounds("t"),
            baseline_bounds=comparison_context.bounds("pr"),
            segment_evaluable={name: bool(comparison_context.is_segment_evaluable(name))
                               for name in ("qrs", "t", "pr")})
        classical = ClassicalECGCriteria().compute(
            paths["diagnostic"],
            delineation,
            age=case.metadata.get("age"),
            sex=case.metadata.get("sex"),
            measurement_sink=lead_measurements,
            extended_measurements=True,
            ischemia_context=ischemia_context['context'] if ischemia_context else None,
            paced_or_lbbb=bool(ischemia_context),
            paced=bool(ischemia_context)
            and ischemia_context['context'] == 'ventricular_pacing',
        )
        classical_payload = {
            key: value.to_dict() for key, value in classical.items()
        }
        baseline_registry = BaselineRegistry()
        from .lab_classical_provenance import capture_classical_provenance
        classical_payload = capture_classical_provenance(classical_payload, baseline_registry)
        # This laboratory call does not pass a machine axis to compute().
        for axis_id in ('left_axis_deviation', 'right_axis_deviation'):
            if axis_id in classical_payload:
                classical_payload[axis_id]['axis_measurement_method'] = 'I/aVF peak-amplitude surrogate; aVF scaled by 2/sqrt(3)'
        score_component_values = dict(feature_values)
        score_component_values.update(
            {
                f"classical_{key}": value.value
                for key, value in classical.items()
            }
        )
        criterion_findings = _evaluate_case_criteria(
            criterion_candidates,
            feature_values=feature_values,
            score_component_values=score_component_values,
            classical_payload=classical_payload,
            baseline_registry=baseline_registry,
            lead_names=case.lead_names,
            signal_quality=float(quality.score),
            delineation=delineation,
        )
        scores = []
        for release in releases:
            components = tuple(
                str(item)
                for item in release.get("score_spec", {}).get(
                    "components",
                    (),
                )
            )
            missing_components = tuple(
                item
                for item in components
                if _float_or_none(score_component_values.get(item)) is None
            )
            value = _apply_single_case_score(
                score_component_values,
                release["score_spec"],
            )
            projection_eligible, projection_restriction = (
                _laboratory_projection_eligibility(
                    components,
                    delineation=delineation,
                )
            )
            if not projection_eligible:
                value = None
            stage = ScoreReleaseStage(
                release.get("stage", ScoreReleaseStage.DEVELOPMENT.value)
            )
            marker_model = dict(release.get("marker_model", {}))
            component_details = _score_component_details(
                score_component_values,
                release.get("score_spec", {}),
            )
            operating_points = _single_case_operating_points(
                value,
                marker_model,
            )
            primary_operating_point = operating_points.get(
                "specificity_95",
                {},
            )
            endpoint_id = str(release.get("endpoint_id", ""))
            classical_comparators = _classical_comparators_for_endpoint(
                endpoint_id,
                classical_payload,
                registry=baseline_registry,
            )
            validation_metrics = dict(
                release.get("validation_metrics", {})
            )
            scores.append(
                {
                    "release_id": release.get("release_id"),
                    "endpoint_id": endpoint_id,
                    "stage": stage.value,
                    "research_score": value,
                    "recommended_release": True,
                    "missing_components": list(missing_components),
                    "component_details": component_details,
                    "operating_points": operating_points,
                    "primary_operating_point": primary_operating_point,
                    "research_pattern_status": _research_pattern_status(
                        value,
                        primary_operating_point,
                    ),
                    "measurement_confidence": float(
                        min(quality.score, delineation.quality_score)
                    ),
                    "validation_metrics": validation_metrics,
                    "evidence_status": _release_evidence_status(
                        stage,
                        validation_metrics,
                    ),
                    "classical_comparators": classical_comparators,
                    "classical_positive_count": sum(
                        item.get("positive") is True
                        for item in classical_comparators
                    ),
                    "not_evaluable_reason": (
                        projection_restriction["abstention_reason"]
                        if not projection_eligible
                        else
                        "missing score components: "
                        + ", ".join(missing_components)
                        if missing_components
                        else ""
                    ),
                    "analysis_restriction": projection_restriction,
                    "primary_display_allowed": stage in (
                        ScoreReleaseStage.LOCKED_TEST_VALIDATED,
                        ScoreReleaseStage.EXTERNAL_VALIDATED,
                    ),
                    "display_tier": (
                        "validated_research"
                        if stage in (
                            ScoreReleaseStage.LOCKED_TEST_VALIDATED,
                            ScoreReleaseStage.EXTERNAL_VALIDATED,
                        )
                        else "experimental_development"
                    ),
                }
            )
        single_marker_patterns = _single_case_marker_patterns(
            feature_values,
            single_marker_candidates,
            measurement_confidence=float(
                min(quality.score, delineation.quality_score)
            ),
            delineation=delineation,
        )
        pattern_summary = _research_pattern_summary(
            scores,
            single_marker_patterns=single_marker_patterns,
        )
        research_diagnosis_summary = _research_diagnosis_summary(
            criterion_findings
        )
        checkpoint()
        implementation_hash = source_tree_hash(Path(__file__).resolve().parents[1])
        analysis_id = content_hash(
            {
                "acquisition_metadata": acquisition,
                "ischemia_context": ischemia_context,
                "case_sha256": case.source_sha256,
                "panel_ids": panel_ids,
                "vcg_method": vcg_method,
                "powerline_hz": powerline_hz,
                "annotation_revision_id": (
                    annotation.revision_id if annotation is not None else ""
                ),
                "annotation_hash": (
                    content_hash(annotation.to_dict())
                    if annotation is not None
                    else ""
                ),
                "score_releases": [
                    {
                        "release_id": item.get("release_id"),
                        "stage": item.get("stage"),
                        "score_spec_hash": content_hash(
                            item.get("score_spec", {})
                        ),
                    }
                    for item in releases
                ],
                "single_marker_candidates": [
                    {
                        "candidate_id": item.get("candidate_id"),
                        "marker_model_hash": content_hash(
                            item.get("marker_model", {})
                        ),
                        "validation_metrics_hash": content_hash(
                            item.get("validation_metrics", {})
                        ),
                    }
                    for item in single_marker_candidates
                ],
                "criterion_registry": {
                    "source_run_id": criterion_inventory.get(
                        "source_run_id",
                        "",
                    ),
                    "registry_hash": criterion_inventory.get(
                        "registry_hash",
                        "",
                    ),
                    "criterion_ids": [
                        item.get("criterion_id", "")
                        for item in criterion_candidates
                    ],
                },
                "implementation_hash": implementation_hash,
            }
        )
        output_root = self.root / case_id / "analyses" / analysis_id[:16]
        output_root.mkdir(parents=True, exist_ok=True)
        fiducials = _representative_fiducials(delineation)
        fiducial_details = _representative_fiducial_details(delineation)
        wave_segments = _representative_wave_segments(delineation)
        accepted_beat_wave_segments = _accepted_beat_wave_segments(
            delineation
        )
        measurable_wave_segments = [
            item
            for item in accepted_beat_wave_segments
            if item["status"] == "measurable"
        ]
        if not measurable_wave_segments:
            measurable_wave_segments = [
                item
                for item in wave_segments
                if item["status"] == "measurable"
            ]
        interval_measurements = _representative_intervals(delineation)
        from .lab_bbb_qt import bogossian_modified_qt
        interval_measurements.append(bogossian_modified_qt(interval_measurements, ischemia_context, case_id,
                                                         age=case.metadata.get('age')))
        qt_rr_beats = delineation.qt_rr_measurements()
        from ..core.svd_projection import project_ecg_components
        svd_projection = project_ecg_components(paths["morphology"])
        _atomic_write_npz(
            output_root / "visualization_arrays.npz",
            diagnostic_ecg=paths["diagnostic"].astype(np.float32),
            morphology_ecg=paths["morphology"].astype(np.float32),
            robust_morphology_ecg=robust_morphology.astype(np.float32),
            record_vcg_xyz=record_vcg_morphology.xyz.astype(np.float32),
            record_svd_components=svd_projection.coordinates.astype(np.float32),
            svd_component_loadings=svd_projection.loadings,
            vcg_xyz=vcg_morphology.xyz.astype(np.float32),
            r_peaks=np.asarray(r_peaks, dtype=np.int32),
            fiducial_names=np.asarray(tuple(fiducials), dtype="U24"),
            fiducial_samples=np.asarray(tuple(fiducials.values()), dtype=np.int32),
            fiducial_confidences=np.asarray(
                [
                    fiducial_details[name]["confidence"]
                    for name in fiducials
                ],
                dtype=np.float32,
            ),
            fiducial_uncertainty_ms=np.asarray(
                [
                    fiducial_details[name]["uncertainty_ms"]
                    for name in fiducials
                ],
                dtype=np.float32,
            ),
            fiducial_sources=np.asarray(
                [
                    fiducial_details[name]["source"]
                    for name in fiducials
                ],
                dtype="U160",
            ),
            fiducial_wave_statuses=np.asarray(
                [
                    fiducial_details[name]["wave_status"]
                    for name in fiducials
                ],
                dtype="U40",
            ),
            segment_names=np.asarray(
                [item["segment"] for item in measurable_wave_segments],
                dtype="U24",
            ),
            segment_start_samples=np.asarray(
                [
                    item["start_sample"]
                    for item in measurable_wave_segments
                ],
                dtype=np.int32,
            ),
            segment_end_samples=np.asarray(
                [
                    item["end_sample"]
                    for item in measurable_wave_segments
                ],
                dtype=np.int32,
            ),
            segment_confidences=np.asarray(
                [
                    item["confidence"]
                    for item in measurable_wave_segments
                ],
                dtype=np.float32,
            ),
            segment_beat_indices=np.asarray(
                [
                    int(item.get("beat_index", -1))
                    for item in measurable_wave_segments
                ],
                dtype=np.int32,
            ),
            segment_r_peak_samples=np.asarray(
                [
                    int(item.get("r_peak_sample", -1))
                    for item in measurable_wave_segments
                ],
                dtype=np.int32,
            ),
            sampling_rate_hz=np.asarray(target_fs),
            qt_rr_beat_index=np.asarray(
                [m.beat_index for m in qt_rr_beats], dtype=np.int32
            ),
            qt_rr_beat_rr_ms=np.asarray(
                [m.preceding_rr_seconds * 1000.0 for m in qt_rr_beats],
                dtype=np.float32,
            ),
            qt_rr_beat_qt_ms=np.asarray(
                [m.qt_ms for m in qt_rr_beats], dtype=np.float32
            ),
            qt_rr_beat_confidence=np.asarray(
                [m.confidence for m in qt_rr_beats], dtype=np.float32
            ),
        )
        accepted_beat_ids = {id(beat) for beat in delineation.accepted_beats}
        payload = {
            "analysis_id": analysis_id,
            "measurement_protocol": {"schema_version": "1.0.0", "implementation_hash": implementation_hash,
                                     "scope": "representative_and_explicit_beatwise_intervals"},
            "lab_processing": {"schema_version": "1.0.0", "original_sampling_rate_hz": original_fs,
                               "analysis_sampling_rate_hz": target_fs, "powerline_hz": powerline_hz,
                               "coordinate_system": "analysis_samples", "raw_signal_immutable": True},
            "signal_reference": {"source_sha256": case.source_sha256, "case_id": case_id,
                                 "native_sampling_rate_hz": original_fs},
            "beat_inventory": [{"beat_index": index, "r_peak_sample": beat.r_peak.sample,
                                "morphology_cluster_id": beat.morphology_cluster_id,
                                "beat_type": beat.beat_type, "beat_type_source": beat.beat_type_source,
                                "review_reasons": list(beat.review_reasons),
                                "accepted": id(beat) in accepted_beat_ids}
                               for index, beat in enumerate(delineation.beats)],
            "case_id": case_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "diagnosis_generated": False,
            "research_diagnosis_generated": bool(
                research_diagnosis_summary[
                    "probable_research_diagnoses"
                ]
            ),
            "analysis_status": "completed_research_analysis",
            "markers_calculated": True,
            "signal_quality": quality.to_dict(),
            "delineation": {
                "method": delineation.source_method,
                "minimum_measurement_confidence": 0.35,
                "detected_beat_count": delineation.n_beats,
                "accepted_beat_count": delineation.n_accepted_beats,
                "fiducial_confidence": delineation.quality_score,
                "morphology_template": template_diagnostics,
                "representative_fiducials": fiducials,
                "fiducial_details": fiducial_details,
                "method_candidates": {
                    "schema_version": "1.0.0", "sampling_rate_hz": target_fs,
                    "scope": "representative_beat_original_detector_candidates_before_fusion_and_manual_edits",
                    "capture_limit_beats": 2000,
                    "items": delineator.candidate_snapshots.get(delineation.representative_beat.r_peak.sample, [])
                        if delineation.representative_beat is not None else [],
                    "warning": "Temporal admissibility does not imply selection in the final consensus; empty means unavailable, not agreement.",
                },
                "wave_segments": wave_segments,
                "accepted_beat_wave_segments": accepted_beat_wave_segments,
                "interval_measurements": interval_measurements,
                "rhythm_context": {
                    "class": delineation.rhythm.rhythm_class.value,
                    "confidence": delineation.rhythm.confidence,
                    "rr_median_ms": delineation.rhythm.rr_median_ms,
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
                    "p_wave_coverage": delineation.rhythm.p_wave_coverage,
                    "p_timing_mad_ms": delineation.rhythm.p_timing_mad_ms,
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
                    "research_pattern_only": True,
                },
                "review_required": delineation.review_required,
                "review_reasons": list(delineation.review_reasons),
                "diagnostics": dict(delineation.diagnostics),
                "annotation_revision_id": (
                    annotation.revision_id if annotation is not None else ""
                ),
                "annotation_applied": annotation is not None,
            },
            "vcg_method": vcg_method,
            "feature_values": feature_values,
            "feature_measurements": {result.name: {
                "value": feature_values.get(result.name), "unit": result.unit,
                "formula": result.formula, "family": result.family,
                "confidence": result.confidence, "notes": result.computation_notes,
                "implementation_status": result.implementation_status,
                "description": result.description, "references": result.references,
                "measurement_windows": (
                    dict(result.measurement_windows, signal_path="robust_morphology_ecg")
                    if isinstance(result.measurement_windows, dict)
                    and result.measurement_windows.get("signal_path") == "feature_context_ecg_12lead"
                    else dict(result.measurement_windows, signal_path="morphology_ecg")
                    if isinstance(result.measurement_windows, dict)
                    and result.measurement_windows.get("signal_path") == "record_feature_context_ecg_12lead"
                    else result.measurement_windows),
            } for result in feature_results},
            "svd_component_projection": svd_projection.metadata,
            "qrs_t_method_comparison": method_comparison,
            "direct8_measurement_windows": {
                "signal_path": "robust_morphology_ecg", "coordinate_system": "analysis_samples_inclusive",
                "sampling_rate_hz": target_fs,
                "qrs": comparison_context.bounds("qrs"),
                "pr_baseline": comparison_context.bounds("pr"),
                "j_to_t_end": ((comparison_context.bounds("qrs")[1], comparison_context.bounds("t")[1])
                               if comparison_context.bounds("qrs") and comparison_context.bounds("t") else None),
                "leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
                "scope": "Original PR-based Direct-8 definitions only; no substitution for Pre-QRS/TP/beatwise variants",
            },
            "qrs_t_segment_matrix": segment_matrix,
            "ndat": ndat_result,
            "acquisition_metadata": acquisition,
            "ischemia_context": ischemia_context,
            "score_component_values": score_component_values,
            "classical_criteria": classical_payload,
            # Audit 5.3.5: the smallest detectable change per marker from the
            # evidence run's repeatability stage, so a follow-up difference in
            # this case can be read against measurement error.
            "marker_change_limits": _load_marker_change_limits(
                evidence_run_root, tuple(feature_values)
            ),
            "lead_measurements": lead_measurements,
            "lead_measurement_scope": "diagnostic_12lead_shared_fiducials",
            "score_releases": scores,
            "single_marker_patterns": single_marker_patterns,
            "research_pattern_summary": pattern_summary,
            "research_findings": criterion_findings,
            "research_diagnosis_summary": research_diagnosis_summary,
            "research_warning_de": _RESEARCH_WARNING_DE,
            "probability_available": False,
            "calibrated_probabilities": {},
            "criterion_inventory": criterion_inventory,
            "evidence_run_inventory": evidence_run_inventory,
            "score_release_inventory": {
                "available_release_count": len(all_releases),
                "compatible_release_count": len(compatible_releases),
                "stale_release_count": (
                    len(all_releases) - len(compatible_releases)
                ),
                "endpoint_count": len(releases),
                "selection_policy": (
                    "one exact compatible complete run; one release per "
                    "endpoint by validation stage, documented selection/"
                    "stability, then validation AUC"
                ),
                "source_run_id": (
                    evidence_run_root.name
                    if evidence_run_root is not None
                    else ""
                ),
            },
            "single_marker_inventory": single_marker_inventory,
            "language_guard": (
                "Train-fitted thresholds and Validation evidence may support "
                "a probable research diagnosis only when applicability and "
                "measurement quality pass. This remains no clinical diagnosis "
                "or disease probability."
            ),
        }
        atomic_write_json(output_root / "analysis.json", payload)
        return payload

    def _latest_annotation(
        self,
        case_id: str,
    ) -> Optional[AnnotationRevision]:
        revisions = self.annotations(case_id)
        return revisions[-1] if revisions else None

    def _write_blocked_quality_analysis(
        self,
        case: ImportedECGCase,
        signal: np.ndarray,
        sampling_rate: int,
        *,
        panel_ids: tuple[str, ...],
        vcg_method: str,
        powerline_hz: float,
        quality,
    ) -> dict[str, Any]:
        analysis_id = content_hash(
            {
                "case_sha256": case.source_sha256,
                "panel_ids": panel_ids,
                "vcg_method": vcg_method,
                "powerline_hz": powerline_hz,
                "analysis_status": "blocked_signal_quality",
                "signal_quality": quality.to_dict(),
                "implementation_hash": source_tree_hash(
                    Path(__file__).resolve().parents[1]
                ),
            }
        )
        output_root = (
            self.root / case.case_id / "analyses" / analysis_id[:16]
        )
        output_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_npz(
            output_root / "visualization_arrays.npz",
            diagnostic_ecg=signal.astype(np.float32),
            morphology_ecg=np.empty((0, 12), dtype=np.float32),
            vcg_xyz=np.empty((0, 3), dtype=np.float32),
            r_peaks=np.empty(0, dtype=np.int32),
            fiducial_names=np.empty(0, dtype="U24"),
            fiducial_samples=np.empty(0, dtype=np.int32),
            fiducial_confidences=np.empty(0, dtype=np.float32),
            fiducial_uncertainty_ms=np.empty(0, dtype=np.float32),
            fiducial_sources=np.empty(0, dtype="U160"),
            fiducial_wave_statuses=np.empty(0, dtype="U40"),
            segment_names=np.empty(0, dtype="U24"),
            segment_start_samples=np.empty(0, dtype=np.int32),
            segment_end_samples=np.empty(0, dtype=np.int32),
            segment_confidences=np.empty(0, dtype=np.float32),
            sampling_rate_hz=np.asarray(sampling_rate),
        )
        reasons = list(quality.reasons)
        payload = {
            "analysis_id": analysis_id,
            "case_id": case.case_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "diagnosis_generated": False,
            "research_diagnosis_generated": False,
            "analysis_status": "blocked_signal_quality",
            "markers_calculated": False,
            "block_reason": (
                "Signal quality is insufficient for reproducible marker "
                "calculation: " + ", ".join(reasons)
            ),
            "signal_quality": quality.to_dict(),
            "delineation": {
                "detected_beat_count": 0,
                "accepted_beat_count": 0,
                "fiducial_confidence": 0.0,
                "representative_fiducials": {},
                "fiducial_details": {},
                "wave_segments": [],
                "interval_measurements": [],
                "rhythm_context": {
                    "class": "not_assessable",
                    "confidence": 0.0,
                    "research_pattern_only": True,
                },
                "review_required": True,
                "review_reasons": ["blocked_signal_quality"],
                "diagnostics": {},
                "annotation_revision_id": "",
                "annotation_applied": False,
            },
            "vcg_method": vcg_method,
            "feature_values": {},
            "score_component_values": {},
            "classical_criteria": {},
            "marker_change_limits": {},
            "lead_measurements": {},
            "lead_measurement_scope": "unavailable",
            "score_releases": [],
            "single_marker_patterns": [],
            "research_pattern_summary": [],
            "research_findings": [],
            "research_diagnosis_summary": {
                "status": "not_assessable",
                "probable_research_diagnoses": [],
                "message_de": (
                    "Das Signal ist nicht ausreichend beurteilbar; es wurde "
                    "keine Forschungsbewertung erzeugt."
                ),
            },
            "research_warning_de": _RESEARCH_WARNING_DE,
            "probability_available": False,
            "calibrated_probabilities": {},
            "criterion_inventory": {
                "status": "not_evaluated_signal_quality_block",
                "candidate_count": 0,
            },
            "evidence_run_inventory": {
                "status": "not_evaluated_signal_quality_block",
            },
            "score_release_inventory": {
                "available_release_count": 0,
                "endpoint_count": 0,
                "selection_policy": "not applicable: signal-quality block",
            },
            "single_marker_inventory": {
                "available_candidate_count": 0,
                "selected_candidate_count": 0,
                "selection_policy": "not applicable: signal-quality block",
            },
            "language_guard": (
                "Signal-QC block: no research marker, comparison rule or "
                "patient-level diagnosis was calculated."
            ),
        }
        atomic_write_json(output_root / "analysis.json", payload)
        return payload


def _select_case_evidence_run(
    config: MimicProjectConfig,
) -> tuple[Optional[Path], dict[str, Any]]:
    """Select one exact complete development run for case interpretation."""

    reviewed = 0
    rejected: list[dict[str, str]] = []
    eligible: list[tuple[int, str, Path, StageManifest, dict]] = []
    for registry_path in sorted(
        config.runs_root.glob("*/criterion_registry.json")
    ):
        reviewed += 1
        run_root = registry_path.parent
        manifest_path = run_root / "validate.manifest.json"
        provenance_path = run_root / "development_provenance.json"
        try:
            manifest = StageManifest.load(manifest_path)
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            registry = json.loads(
                registry_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rejected.append(
                {
                    "run_id": run_root.name,
                    "reason": f"Manifest oder Register nicht lesbar: {exc}",
                }
            )
            continue
        compatible, compatibility_reason = (
            _run_implementation_compatibility(run_root)
        )
        exact_outputs = {
            str(Path(item).resolve()) for item in manifest.output_paths
        }
        reasons = []
        if manifest.status != "complete":
            reasons.append("Validation-Manifest ist nicht vollstaendig")
        if str(registry_path.resolve()) not in exact_outputs:
            reasons.append("Kriterienregister fehlt im exakten Manifest")
        if provenance.get("analysis_completeness") != "complete_snapshot":
            reasons.append("Lauf ist nur Vorschau oder partiell")
        if provenance.get("preview") is not False:
            reasons.append("Lauf ist als Vorschau markiert")
        if provenance.get("locked_test_labels_read") is not False:
            reasons.append("Locked-Test-Schutz ist nicht nachweisbar")
        if registry.get("locked_test_read") is not False:
            reasons.append("Kriterienregister meldet Locked-Test-Zugriff")
        if not compatible:
            reasons.append(compatibility_reason)
        if reasons:
            rejected.append(
                {
                    "run_id": run_root.name,
                    "reason": "; ".join(reasons),
                }
            )
            continue
        eligible.append(
            (
                registry_path.stat().st_mtime_ns,
                run_root.name,
                run_root,
                manifest,
                registry,
            )
        )
    if not eligible:
        return None, {
            "status": "no_compatible_complete_validation_run",
            "reviewed_run_count": reviewed,
            "eligible_run_count": 0,
            "selected_run_id": "",
            "rejected_runs": rejected[-20:],
            "selection_policy": (
                "latest exact complete snapshot with matching scientific "
                "implementation; artifacts from different runs are never mixed"
            ),
        }
    _, _, run_root, manifest, registry = max(
        eligible,
        key=lambda item: (item[0], item[1]),
    )
    return run_root, {
        "status": "compatible_complete_validation_run_selected",
        "reviewed_run_count": reviewed,
        "eligible_run_count": len(eligible),
        "selected_run_id": run_root.name,
        "snapshot_hash": manifest.input_hash,
        "criterion_registry_version": registry.get(
            "criterion_registry_version",
            "",
        ),
        "rejected_runs": rejected[-20:],
        "selection_policy": (
            "latest exact complete snapshot with matching scientific "
            "implementation; artifacts from different runs are never mixed"
        ),
    }


def _load_case_criterion_candidates(
    config: MimicProjectConfig,
    *,
    run_root: Optional[Path],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    if run_root is None:
        return (), {
            "status": "no_compatible_registry",
            "source_run_id": "",
            "candidate_count": 0,
            "evaluation_policy": (
                "No historical or partial criterion is used as fallback."
            ),
        }
    path = run_root / "criterion_registry.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (), {
            "status": "registry_unreadable",
            "source_run_id": run_root.name,
            "candidate_count": 0,
            "error": str(exc),
        }
    if (
        payload.get("locked_test_read") is not False
        or payload.get("clinical_claim_allowed") is not False
    ):
        return (), {
            "status": "registry_safety_guard_failed",
            "source_run_id": run_root.name,
            "candidate_count": 0,
        }
    threshold_lookup: dict[str, list[dict[str, Any]]] = {}
    evidence_lookup = {}
    evidence_path = run_root / "guideline" / "candidate_evidence.parquet"
    if evidence_path.is_file():
        evidence_frame = pd.read_parquet(evidence_path)
        if "criterion_id" in evidence_frame and not evidence_frame.criterion_id.duplicated().any():
            evidence_lookup = {str(row["criterion_id"]): row for row in evidence_frame.to_dict("records")}
    for row in payload.get("thresholds", ()):
        values = dict(row)
        threshold_lookup.setdefault(
            str(values.get("criterion_id", "")),
            [],
        ).append(values)
    candidates: list[dict[str, Any]] = []
    skipped = 0
    for raw in payload.get("criteria", ()):
        candidate = dict(raw)
        criterion_id = str(candidate.get("criterion_id", ""))
        feature_id = str(candidate.get("feature_id", ""))
        if not criterion_id or not feature_id:
            skipped += 1
            continue
        components = _parse_component_ids(candidate)
        if not components or len(components) > 4:
            skipped += 1
            continue
        applicability = _parse_json_object(
            candidate.get("applicability_json")
        )
        score_spec = _parse_json_object(
            candidate.get("score_spec_json")
        )
        bands = []
        threshold_metadata: dict[str, dict[str, Any]] = {}
        for threshold in threshold_lookup.get(criterion_id, ()):
            try:
                band = _threshold_band_from_payload(threshold)
                bands.append(band)
                threshold_metadata[band.operating_point] = dict(threshold)
            except (TypeError, ValueError):
                continue
        candidate["component_ids"] = components
        candidate["applicability"] = applicability
        candidate["score_spec"] = score_spec
        candidate["threshold_bands"] = tuple(bands)
        candidate["threshold_metadata"] = threshold_metadata
        candidate["source_run_id"] = run_root.name
        audited = evidence_lookup.get(criterion_id, {})
        if (str(audited.get("run_id", "")) == run_root.name
                and str(audited.get("snapshot_hash", "")) == str(candidate.get("snapshot_hash", ""))
                and audited.get("snapshot_hash")):
            candidate.update({key: value for key, value in audited.items()
                if key.endswith("_passed") or key in {"reference_kind", "analysis_completeness"}})
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            _criterion_stage_rank(str(item.get("stage", ""))),
            _evidence_grade_rank(str(item.get("evidence_grade", ""))),
            _float_or_none(item.get("roc_auc")) or -1.0,
            str(item.get("criterion_id", "")),
        ),
        reverse=True,
    )
    return tuple(candidates), {
        "status": "ready" if candidates else "no_reviewable_criteria",
        "source_run_id": run_root.name,
        "registry_path": str(path.resolve()),
        "registry_hash": file_sha256(path),
        "registry_version": payload.get("criterion_registry_version", ""),
        "snapshot_hash": payload.get("feature_snapshot_hash", ""),
        "candidate_count": len(candidates),
        "skipped_candidate_count": skipped,
        "threshold_source": payload.get("threshold_source", ""),
        "evaluation_split": payload.get("evaluation_split", ""),
        "clinical_claim_allowed": False,
    }


def _parse_component_ids(candidate: dict[str, Any]) -> tuple[str, ...]:
    raw_json = candidate.get("component_ids_json")
    try:
        parsed = json.loads(str(raw_json)) if raw_json else None
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed if str(item))
    raw = candidate.get("component_ids", "")
    if isinstance(raw, (tuple, list)):
        return tuple(str(item) for item in raw if str(item))
    return tuple(item for item in str(raw).split("|") if item)


def _parse_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, "", "null"):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _threshold_band_from_payload(values: dict[str, Any]) -> ThresholdBand:
    return ThresholdBand(
        operating_point=str(values["operating_point"]),
        threshold=float(values["threshold"]),
        gray_zone_low=float(values["gray_zone_low"]),
        gray_zone_high=float(values["gray_zone_high"]),
        direction=int(values["direction"]),
        learned_split=str(values.get("threshold_source", "train")),
        uncertainty_method=str(
            values.get("uncertainty_method", "not_estimated")
        ),
        bootstrap_iterations=int(values.get("bootstrap_iterations", 0)),
        successful_bootstrap_iterations=int(
            values.get("successful_bootstrap_iterations", 0)
        ),
    )


def _criterion_component_names(
    candidates: Iterable[dict[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            component
            for candidate in candidates
            for component in candidate.get("component_ids", ())
            if component and not str(component).startswith("classical_")
        )
    )


def _evaluate_case_criteria(
    candidates: Iterable[dict[str, Any]],
    *,
    feature_values: dict[str, Optional[float]],
    score_component_values: dict[str, Optional[float]],
    classical_payload: dict[str, dict],
    baseline_registry: BaselineRegistry,
    lead_names: Iterable[str],
    signal_quality: float,
    delineation: RecordDelineation,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    available_leads = {str(item) for item in lead_names}
    for candidate in candidates:
        components = tuple(candidate.get("component_ids", ()))
        score_spec = dict(candidate.get("score_spec", {}))
        feature_id = str(candidate.get("feature_id", ""))
        value = (
            _apply_single_case_score(score_component_values, score_spec)
            if score_spec
            else _float_or_none(feature_values.get(feature_id))
        )
        projection_eligible, projection_restriction = (
            _laboratory_projection_eligibility(
                components or (feature_id,),
                delineation=delineation,
            )
        )
        if not projection_eligible:
            value = None
        applicability_reasons = _criterion_applicability_reasons(
            candidate,
            value=value,
            available_leads=available_leads,
            signal_quality=signal_quality,
            delineation=delineation,
        )
        if not projection_eligible:
            applicability_reasons.append(
                "P-Loop-Projektion nicht beurteilbar: erforderlich sind "
                "ein gleichzeitig technisch als regulaer oder breit-QRS "
                "klassifiziertes Rhythmusmuster, Rhythmussicherheit >=60% "
                "und P-Wellen-Abdeckung >=50%."
            )
        bands = tuple(candidate.get("threshold_bands", ()))
        if not bands:
            applicability_reasons.append(
                "Kein reproduzierbarer Train-Grenzwert vorhanden."
            )
        operating_points = [
            _evaluate_threshold_band(
                band,
                value=value,
                applicable=not applicability_reasons,
                metadata=dict(candidate.get("threshold_metadata", {})).get(
                    band.operating_point,
                    {},
                ),
            )
            for band in sorted(
                bands,
                key=lambda item: _operating_point_rank(
                    item.operating_point
                ),
                reverse=True,
            )
        ]
        primary = next(
            (
                item
                for item in operating_points
                if item["operating_point"] == "specificity_95"
            ),
            operating_points[0] if operating_points else {},
        )
        endpoint_id = str(candidate.get("endpoint_id", ""))
        endpoint_term = clinical_display_term(endpoint_id)
        stage = str(candidate.get("stage", CriterionStage.HYPOTHESIS.value))
        evidence = str(
            candidate.get("evidence_grade", EvidenceGrade.UNSTABLE.value)
        )
        classification = str(primary.get("classification", "not_evaluable"))
        from .result_contract import scientific_result_contract
        scientific = scientific_result_contract(candidate)
        diagnosis_phrase_allowed = (
            stage in _RESEARCH_DIAGNOSIS_STAGES
            and classification == "positive"
            and not applicability_reasons
            and scientific.evidence_stage in {"guideline_candidate", "clinically_validated"}
        )
        interpretation = _criterion_interpretation_de(
            endpoint_name=endpoint_term.display_name_de,
            classification=classification,
            diagnosis_phrase_allowed=diagnosis_phrase_allowed,
            applicability_reasons=applicability_reasons,
        )
        comparators = _classical_comparators_for_endpoint(
            endpoint_id,
            classical_payload,
            registry=baseline_registry,
        )
        validation_sensitivity = _float_or_none(
            primary.get("sensitivity_validation")
        )
        validation_specificity = _float_or_none(
            primary.get("specificity_validation")
        )
        try:
            evidence_enum = EvidenceGrade(evidence)
        except ValueError:
            evidence_enum = EvidenceGrade.UNSTABLE
        primary_direction = _safe_direction(primary.get("direction", 1))
        finding = Finding(
            finding_id="finding_" + content_hash(
                {
                    "criterion_id": candidate.get("criterion_id", ""),
                    "value": value,
                    "classification": classification,
                    "snapshot_hash": candidate.get("snapshot_hash", ""),
                }
            )[:20],
            endpoint_id=endpoint_id,
            feature_id=feature_id,
            evidence_grade=evidence_enum,
            effect_direction=primary_direction,
            validation_auc=_float_or_none(candidate.get("roc_auc")),
            sensitivity=validation_sensitivity,
            specificity=validation_specificity,
            q_value=_float_or_none(candidate.get("q_value")),
            snapshot_hash=str(candidate.get("snapshot_hash", "")),
            interpretation_de=interpretation,
        ).to_dict()
        finding.update(
            {
                **scientific.to_dict(),
                "sensitivity_source": "validation_operating_point" if validation_sensitivity is not None else "unknown",
                "specificity_source": "validation_operating_point" if validation_specificity is not None else "unknown",
                "criterion_id": candidate.get("criterion_id", ""),
                "criterion_name_de": candidate.get(
                    "display_name_de",
                    endpoint_term.display_name_de,
                ),
                "endpoint_name_de": endpoint_term.display_name_de,
                "endpoint_category_de": endpoint_term.category_de,
                "endpoint_definition_de": endpoint_term.short_definition_de,
                "feature_name_de": candidate.get(
                    "feature_name_de",
                    biomarker_display_name(feature_id),
                ),
                "feature_tooltip_de": candidate.get(
                    "feature_tooltip_de",
                    biomarker_tooltip(feature_id),
                ),
                "formula": candidate.get("formula", ""),
                "unit": candidate.get("unit", ""),
                "value": value,
                "component_ids": list(components),
                "component_details": (
                    _score_component_details(
                        score_component_values,
                        score_spec,
                    )
                    if score_spec
                    else [
                        {
                            "feature_id": feature_id,
                            "raw_value": value,
                            "feature_name_de": biomarker_display_name(
                                feature_id
                            ),
                        }
                    ]
                ),
                "stage": stage,
                "stage_de": _criterion_stage_de(stage),
                "evidence_grade_de": _evidence_grade_de(evidence),
                "classification": classification,
                "classification_de": _classification_de(classification),
                "applicable": not applicability_reasons,
                "applicability_reasons_de": applicability_reasons,
                "applicability": dict(
                    candidate.get("applicability", {})
                ),
                "analysis_restriction": projection_restriction,
                "operating_points": operating_points,
                "primary_operating_point": primary,
                "distance_to_threshold": primary.get(
                    "distance_to_threshold"
                ),
                "distance_to_gray_zone": primary.get(
                    "distance_to_gray_zone"
                ),
                "threshold_source": "train",
                "source_run_id": candidate.get("source_run_id", ""),
                "research_diagnosis_phrase_allowed": (
                    diagnosis_phrase_allowed
                ),
                "clinical_diagnosis_allowed": False,
                "probability": None,
                "probability_available": False,
                "research_warning_de": _RESEARCH_WARNING_DE,
                "classical_comparators": comparators,
                "discordance_status": _criterion_discordance_status(
                    classification,
                    comparators,
                ),
                "limitations": _parse_json_list(
                    candidate.get("limitations_json")
                ),
            }
        )
        findings.append(finding)
    return sorted(
        findings,
        key=lambda item: (
            item.get("applicable") is True,
            item.get("classification") == "positive",
            _criterion_stage_rank(str(item.get("stage", ""))),
            _evidence_grade_rank(str(item.get("evidence_grade", ""))),
            _float_or_none(item.get("validation_auc")) or -1.0,
            _float_or_none(item.get("distance_to_gray_zone")) or -np.inf,
        ),
        reverse=True,
    )


def _criterion_applicability_reasons(
    candidate: dict[str, Any],
    *,
    value: Optional[float],
    available_leads: set[str],
    signal_quality: float,
    delineation: RecordDelineation,
) -> list[str]:
    applicability = dict(candidate.get("applicability", {}))
    reasons: list[str] = []
    if _float_or_none(value) is None:
        reasons.append(
            "Der Markerwert konnte fuer dieses Signal nicht zuverlaessig "
            "berechnet werden."
        )
    required_leads = {
        str(item) for item in applicability.get("required_leads", ())
    }
    missing_leads = sorted(required_leads - available_leads)
    if missing_leads:
        reasons.append(
            "Pflichtableitungen fehlen: " + ", ".join(missing_leads)
        )
    minimum_quality = float(
        applicability.get("minimum_signal_quality", 0.0) or 0.0
    )
    if signal_quality < minimum_quality:
        reasons.append(
            f"Signalqualitaet {signal_quality:.0%} liegt unter der "
            f"Kriteriumsgrenze von {minimum_quality:.0%}."
        )
    minimum_fiducial = float(
        applicability.get("minimum_fiducial_confidence", 0.0) or 0.0
    )
    if delineation.quality_score < minimum_fiducial:
        reasons.append(
            f"Sicherheit der Wellengrenzen {delineation.quality_score:.0%} "
            f"liegt unter {minimum_fiducial:.0%}."
        )
    rhythm_class = delineation.rhythm.rhythm_class.value
    allowed = {
        str(item) for item in applicability.get("allowed_rhythm_classes", ())
    }
    if allowed and rhythm_class not in allowed:
        reasons.append(
            "Rhythmusmuster ist fuer dieses Kriterium nicht freigegeben: "
            + rhythm_class
        )
    required_segments = _required_segments_for_components(
        tuple(candidate.get("component_ids", ()))
    )
    segment_minimum = max(0.35, minimum_fiducial)
    for segment in required_segments:
        confidence = delineation.segment_confidence(segment)
        if confidence < segment_minimum:
            reasons.append(
                f"Segment {segment.upper()} ist nicht sicher abgrenzbar "
                f"({confidence:.0%}, erforderlich {segment_minimum:.0%})."
            )
    if "p" in required_segments and rhythm_class in {
        "af_compatible_pattern",
        "flutter_compatible_pattern",
        "multiple_atrial_activity",
    }:
        reasons.append(
            "Diskrete P-Wellen-Morphologie ist bei diesem atrialen "
            "Rhythmusmuster nicht beurteilbar."
        )
    return list(dict.fromkeys(reasons))


def _required_segments_for_components(
    components: tuple[str, ...],
) -> tuple[str, ...]:
    required: list[str] = []
    for component in components:
        name = str(component).casefold()
        if name.startswith(("p_", "pwave", "atrial_p")) or "p_loop" in name:
            required.append("p")
        if name.startswith("pr_") or "pr_segment" in name:
            required.extend(("p", "pr"))
        if "qrs_t" in name or "qrst" in name or name.startswith("svg_"):
            required.extend(("qrs", "t"))
        if name.startswith("qrs_") or "qrs_loop" in name:
            required.append("qrs")
        if name.startswith("st_") or "st_loop" in name or "st_j_" in name:
            required.append("st")
        if name.startswith("t_") or "t_loop" in name:
            required.append("t")
        if name.startswith("u_") or "u_loop" in name:
            required.append("u")
        if name.startswith(("qt", "jt", "tpeak")):
            required.extend(("qrs", "t"))
    return tuple(dict.fromkeys(required))


def _evaluate_threshold_band(
    band: ThresholdBand,
    *,
    value: Optional[float],
    applicable: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    numeric = _float_or_none(value)
    classification = (
        band.classify(numeric)
        if applicable and numeric is not None
        else "not_evaluable"
    )
    aligned_value = band.direction * numeric if numeric is not None else None
    aligned_threshold = band.direction * band.threshold
    aligned_bounds = sorted(
        (
            band.direction * band.gray_zone_low,
            band.direction * band.gray_zone_high,
        )
    )
    distance = (
        aligned_value - aligned_threshold
        if aligned_value is not None
        else None
    )
    if aligned_value is None:
        gray_distance = None
    elif aligned_value < aligned_bounds[0]:
        gray_distance = aligned_value - aligned_bounds[0]
    elif aligned_value > aligned_bounds[1]:
        gray_distance = aligned_value - aligned_bounds[1]
    else:
        gray_distance = 0.0
    source = metadata or {}
    return {
        **band.to_dict(),
        "value": numeric,
        "aligned_value": aligned_value,
        "aligned_threshold": aligned_threshold,
        "distance_to_threshold": distance,
        "distance_to_gray_zone": gray_distance,
        "classification": classification,
        "triggered": (
            classification == "positive"
            if classification != "not_evaluable"
            else None
        ),
        "threshold_source": "train",
        "sensitivity_validation": _float_or_none(
            source.get("sensitivity_validation")
        ),
        "specificity_validation": _float_or_none(
            source.get("specificity_validation")
        ),
        "ppv_validation": _float_or_none(source.get("ppv_validation")),
        "npv_validation": _float_or_none(source.get("npv_validation")),
    }


def _criterion_interpretation_de(
    *,
    endpoint_name: str,
    classification: str,
    diagnosis_phrase_allowed: bool,
    applicability_reasons: list[str],
) -> str:
    if applicability_reasons or classification == "not_evaluable":
        return "Nicht beurteilbar: " + "; ".join(applicability_reasons)
    if classification == "gray_zone":
        return (
            f"Der Messwert fuer {endpoint_name} liegt in der "
            "vorab bestimmten Unsicherheits-Grauzone."
        )
    if classification == "positive" and diagnosis_phrase_allowed:
        return (
            f"Wahrscheinliche Forschungsdiagnose: {endpoint_name}. "
            + _RESEARCH_WARNING_DE
        )
    if classification == "positive":
        return (
            f"Exploratives Forschungsmuster fuer {endpoint_name} ist "
            "ausgeloest; die Evidenzstufe erlaubt noch keine wahrscheinliche "
            "Forschungsdiagnose."
        )
    return (
        f"Das vorab bestimmte Kriterium fuer {endpoint_name} ist nicht "
        "ausgeloest. Das schliesst die Pathologie nicht aus."
    )


def _research_diagnosis_summary(
    findings: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(findings)
    positives = []
    seen: set[str] = set()
    for row in rows:
        endpoint_id = str(row.get("endpoint_id", ""))
        if (
            row.get("research_diagnosis_phrase_allowed") is not True
            or endpoint_id in seen
        ):
            continue
        positives.append(
            {
                "endpoint_id": endpoint_id,
                "endpoint_name_de": row.get("endpoint_name_de", endpoint_id),
                "criterion_id": row.get("criterion_id", ""),
                "evidence_grade": row.get("evidence_grade", ""),
                "evidence_grade_de": row.get("evidence_grade_de", ""),
                "distance_to_threshold": row.get("distance_to_threshold"),
                "distance_to_gray_zone": row.get("distance_to_gray_zone"),
                "sensitivity": row.get("sensitivity"),
                "specificity": row.get("specificity"),
            }
        )
        seen.add(endpoint_id)
    if positives:
        status = "probable_research_diagnosis"
        message = (
            "Mindestens ein in Validation bestaetigtes Forschungskriterium "
            "ist ausserhalb seiner Grauzone ausgeloest."
        )
    elif rows:
        status = "no_probable_research_diagnosis"
        message = (
            "Kein anwendbares Kriterium mit ausreichender Evidenz ist "
            "ausserhalb seiner positiven Grauzone ausgeloest."
        )
    else:
        status = "no_compatible_criteria"
        message = (
            "Es gibt noch keinen vollstaendigen kompatiblen Validation-Lauf "
            "mit auswertbaren Kriterien."
        )
    return {
        "status": status,
        "probable_research_diagnoses": positives,
        "message_de": message,
        "research_warning_de": _RESEARCH_WARNING_DE,
        "probability_available": False,
    }


def _criterion_discordance_status(
    classification: str,
    comparators: Iterable[dict[str, Any]],
) -> str:
    evaluable = [item for item in comparators if item.get("evaluable") is True]
    classical_positive = any(item.get("positive") is True for item in evaluable)
    classical_negative = bool(evaluable) and all(
        item.get("positive") is False for item in evaluable
    )
    if classification == "positive" and classical_negative:
        return "research_positive_classical_negative"
    if classification == "positive" and classical_positive:
        return "concordant_positive"
    if classification == "negative" and classical_positive:
        return "research_negative_classical_positive"
    if classification == "negative" and classical_negative:
        return "concordant_negative"
    if not evaluable:
        return "no_evaluable_classical_baseline"
    return "indeterminate_or_gray_zone"


def _parse_json_list(value: object) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _operating_point_rank(name: str) -> int:
    return {
        "specificity_95": 3,
        "specificity_90": 2,
        "sensitivity_90": 1,
    }.get(str(name), 0)


def _criterion_stage_rank(stage: str) -> int:
    return {
        CriterionStage.HYPOTHESIS.value: 0,
        CriterionStage.TRAIN_STABLE.value: 1,
        CriterionStage.VALIDATION_REPLICATED.value: 2,
        CriterionStage.FROZEN.value: 3,
        CriterionStage.LOCKED_TEST_VALIDATED.value: 4,
        CriterionStage.EXTERNAL_VALIDATED.value: 5,
    }.get(str(stage), -1)


def _evidence_grade_rank(grade: str) -> int:
    return {
        EvidenceGrade.UNSTABLE.value: 0,
        EvidenceGrade.UNDERPOWERED.value: 1,
        EvidenceGrade.TECHNICAL.value: 2,
        EvidenceGrade.DEVELOPMENT.value: 3,
        EvidenceGrade.VALIDATION_REPLICATED.value: 4,
        EvidenceGrade.FROZEN.value: 5,
        EvidenceGrade.LOCKED_TEST_VALIDATED.value: 6,
        EvidenceGrade.EXTERNAL_VALIDATED.value: 7,
    }.get(str(grade), -1)


def _criterion_stage_de(stage: str) -> str:
    return {
        CriterionStage.HYPOTHESIS.value: "Hypothese",
        CriterionStage.TRAIN_STABLE.value: "Im Training stabil",
        CriterionStage.VALIDATION_REPLICATED.value: "In Validation bestaetigt",
        CriterionStage.FROZEN.value: "Eingefroren",
        CriterionStage.LOCKED_TEST_VALIDATED.value: "Im Locked Test bestaetigt",
        CriterionStage.EXTERNAL_VALIDATED.value: "Extern bestaetigt",
    }.get(str(stage), str(stage))


def _evidence_grade_de(grade: str) -> str:
    return {
        EvidenceGrade.TECHNICAL.value: "Technischer Vergleich",
        EvidenceGrade.DEVELOPMENT.value: "Entwicklung",
        EvidenceGrade.VALIDATION_REPLICATED.value: "Validation bestaetigt",
        EvidenceGrade.FROZEN.value: "Eingefroren",
        EvidenceGrade.LOCKED_TEST_VALIDATED.value: "Locked Test bestaetigt",
        EvidenceGrade.EXTERNAL_VALIDATED.value: "Extern bestaetigt",
        EvidenceGrade.UNDERPOWERED.value: "Unterpowert",
        EvidenceGrade.UNSTABLE.value: "Instabil",
    }.get(str(grade), str(grade))


def _classification_de(classification: str) -> str:
    return {
        "positive": "Kriterium ausgeloest",
        "negative": "Kriterium nicht ausgeloest",
        "gray_zone": "Grauzone",
        "not_evaluable": "Nicht beurteilbar",
    }.get(str(classification), str(classification))


def _safe_direction(value: object) -> int:
    try:
        direction = int(cast(Any, value))
    except (TypeError, ValueError):
        return 1
    return direction if direction in (-1, 1) else 1


def _load_score_releases(
    config: MimicProjectConfig,
    *,
    run_root: Optional[Path] = None,
) -> list[dict]:
    releases: list[dict] = []
    paths = (
        (run_root / "score_releases.json",)
        if run_root is not None
        else tuple(sorted(config.runs_root.glob("*/score_releases.json")))
    )
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_root = path.parent
        implementation_compatible, implementation_reason = (
            _run_implementation_compatibility(run_root)
        )
        candidate_lookup = _score_candidate_lookup(run_root)
        validation_lookup = _score_validation_lookup(run_root)
        for raw_release in payload.get("releases", ()):
            release = deepcopy(dict(raw_release))
            score_spec = dict(release.get("score_spec", {}))
            score_id = str(score_spec.get("score_id", ""))
            endpoint_id = str(release.get("endpoint_id", ""))
            candidate = candidate_lookup.get(score_id, {})
            if not release.get("marker_model"):
                release["marker_model"] = dict(
                    candidate.get("marker_model", {})
                )
            if not release.get("validation_metrics"):
                release["validation_metrics"] = dict(
                    validation_lookup.get((endpoint_id, score_id), {})
                )
            release["source_run_id"] = run_root.name
            release["implementation_compatible"] = (
                implementation_compatible
            )
            release["implementation_compatibility_reason"] = (
                implementation_reason
            )
            release["evidence_rank"] = _release_evidence_rank(release)
            releases.append(release)
    deduplicated: dict[str, dict] = {}
    for release in releases:
        release_id = str(release.get("release_id", ""))
        key = release_id or content_hash(release)
        previous = deduplicated.get(key)
        if previous is None or _release_sort_key(release) > _release_sort_key(
            previous
        ):
            deduplicated[key] = release
    return sorted(
        deduplicated.values(),
        key=_release_sort_key,
        reverse=True,
    )


def _load_marker_change_limits(
    run_root: Optional[Path], feature_ids: tuple[str, ...]
) -> dict[str, dict[str, Optional[float]]]:
    """ICC and smallest detectable change per marker from a run (audit 5.3.5)."""

    if run_root is None:
        return {}
    path = Path(run_root) / "marker_repeatability.parquet"
    if not path.is_file():
        return {}
    try:
        table = pd.read_parquet(path)
    except (OSError, ValueError):
        return {}
    key = "feature_id" if "feature_id" in table.columns else "marker" if "marker" in table.columns else None
    if key is None:
        return {}
    wanted = set(feature_ids)
    limits: dict[str, dict[str, Optional[float]]] = {}
    for row in table.drop_duplicates(key).itertuples(index=False):
        name = str(getattr(row, key))
        if name not in wanted:
            continue
        limits[name] = {
            "icc": _float_or_none(getattr(row, "icc", None)),
            "smallest_detectable_change": _float_or_none(
                getattr(row, "smallest_detectable_change", None)
            ),
            "source_run": Path(run_root).name,
        }
    return limits


def _load_recommended_single_marker_candidates(
    config: MimicProjectConfig,
    *,
    maximum_per_endpoint: int = 3,
    run_root: Optional[Path] = None,
) -> tuple[tuple[dict, ...], dict[str, Any]]:
    candidates: list[dict] = []
    compatible_complete_runs = 0
    stale_run_count = 0
    explorer_paths = (
        (run_root / "marker_pathology_explorer.parquet",)
        if run_root is not None
        else tuple(
            sorted(
                config.runs_root.glob(
                    "*/marker_pathology_explorer.parquet"
                )
            )
        )
    )
    for explorer_path in explorer_paths:
        if not explorer_path.is_file():
            continue
        run_root = explorer_path.parent
        compatible, _ = _run_implementation_compatibility(run_root)
        if not compatible:
            stale_run_count += 1
            continue
        try:
            provenance = json.loads(
                (run_root / "development_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            continue
        if (
            provenance.get("preview") is not False
            or provenance.get("analysis_completeness")
            != "complete_snapshot"
            or provenance.get("locked_test_labels_read") is not False
        ):
            continue
        model_path = run_root / "trained_marker_models.json"
        try:
            frame = pd.read_parquet(explorer_path)
            model_payload = json.loads(
                model_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        compatible_complete_runs += 1
        model_lookup = {
            (
                str(item.get("endpoint_id", "")),
                str(item.get("feature_id", "")),
            ): dict(item)
            for item in model_payload.get("models", ())
        }
        for row in frame.to_dict(orient="records"):
            endpoint_id = str(row.get("endpoint_id", ""))
            feature_id = str(row.get("feature_id", ""))
            profile = feature_profile(feature_id)
            if (
                not endpoint_id
                or profile.role != ROLE_CANDIDATE
                or canonical_feature_id(feature_id) != feature_id
                or str(row.get("candidate_type", ""))
                not in {"vcg_single", "ecg_single"}
                or str(row.get("panel_id", ""))
                not in {"primary_core", "sentinel"}
                or str(row.get("inference_status", ""))
                != "formal_inference"
                or str(row.get("reference_independence", ""))
                != "clinical_reference_comparison"
                or _strict_bool(row.get("historical_pre_audit_result"))
                or _strict_bool(row.get("post_outcome_hypothesis"))
                or not _strict_bool(row.get("direction_agreement"))
            ):
                continue
            auc = _float_or_none(row.get("roc_auc"))
            q_value = _float_or_none(row.get("q_value"))
            gap = _float_or_none(row.get("train_validation_auc_gap"))
            grade = str(row.get("evidence_grade", "D"))
            if (
                auc is None
                or auc < 0.60
                or q_value is None
                or q_value >= 0.05
                or gap is None
                or abs(gap) > 0.10
                or grade not in {"A", "B", "C"}
            ):
                continue
            marker_model = model_lookup.get((endpoint_id, feature_id))
            if not marker_model:
                continue
            candidates.append(
                {
                    "candidate_id": (
                        f"{run_root.name}:{endpoint_id}:{feature_id}"
                    ),
                    "source_run_id": run_root.name,
                    "endpoint_id": endpoint_id,
                    "feature_id": feature_id,
                    "feature_modality": profile.modality,
                    "conceptual_family": profile.conceptual_family,
                    "panel_id": str(row.get("panel_id", "")),
                    "evidence_grade": grade,
                    "evidence_rank": (
                        _float_or_none(row.get("evidence_rank")) or 0.0
                    ),
                    "validation_metrics": {
                        "roc_auc": auc,
                        "roc_auc_ci_low": _float_or_none(
                            row.get("roc_auc_ci_low")
                        ),
                        "roc_auc_ci_high": _float_or_none(
                            row.get("roc_auc_ci_high")
                        ),
                        "average_precision": _float_or_none(
                            row.get("average_precision")
                        ),
                        "sensitivity_at_95_specificity": _float_or_none(
                            row.get("sensitivity_at_95_specificity")
                        ),
                        "specificity_at_90_sensitivity": _float_or_none(
                            row.get("specificity_at_90_sensitivity")
                        ),
                        "q_value": q_value,
                        "train_validation_auc_gap": gap,
                    },
                    "marker_model": marker_model,
                    "result_scope": (
                        "development validation support; not locked-test "
                        "validated and not a patient-level diagnosis"
                    ),
                    "_source_mtime_ns": explorer_path.stat().st_mtime_ns,
                }
            )

    deduplicated: dict[tuple[str, str], dict] = {}
    for candidate in candidates:
        key = (
            str(candidate["endpoint_id"]),
            str(candidate["feature_id"]),
        )
        previous = deduplicated.get(key)
        if (
            previous is None
            or _single_marker_candidate_sort_key(candidate)
            > _single_marker_candidate_sort_key(previous)
        ):
            deduplicated[key] = candidate

    selected: list[dict] = []
    by_endpoint: dict[str, list[dict]] = {}
    for candidate in deduplicated.values():
        by_endpoint.setdefault(
            str(candidate["endpoint_id"]),
            [],
        ).append(candidate)
    for endpoint_id in sorted(by_endpoint):
        used_families: set[str] = set()
        for candidate in sorted(
            by_endpoint[endpoint_id],
            key=_single_marker_candidate_sort_key,
            reverse=True,
        ):
            family = str(candidate["conceptual_family"])
            if family in used_families:
                continue
            candidate = dict(candidate)
            candidate.pop("_source_mtime_ns", None)
            selected.append(candidate)
            used_families.add(family)
            if len(used_families) >= max(1, int(maximum_per_endpoint)):
                break
    return (
        tuple(selected),
        {
            "available_candidate_count": len(deduplicated),
            "selected_candidate_count": len(selected),
            "endpoint_count": len(
                {item["endpoint_id"] for item in selected}
            ),
            "compatible_complete_run_count": compatible_complete_runs,
            "stale_or_incompatible_run_count": stale_run_count,
            "maximum_per_endpoint": max(1, int(maximum_per_endpoint)),
            "selection_policy": (
                "audited canonical Core/Sentinel marker; independent "
                "clinical reference; formal Validation; q<0.05; AUC>=0.60; "
                "direction agreement; |Train-Validation gap|<=0.10; at most "
                "one marker per conceptual family"
            ),
        },
    )


def _single_marker_candidate_sort_key(candidate: dict) -> tuple:
    metrics = dict(candidate.get("validation_metrics", {}))
    grade = {"A": 3, "B": 2, "C": 1}.get(
        str(candidate.get("evidence_grade", "")),
        0,
    )
    return (
        grade,
        _float_or_none(candidate.get("evidence_rank")) or 0.0,
        _float_or_none(metrics.get("roc_auc")) or -1.0,
        _float_or_none(metrics.get("sensitivity_at_95_specificity"))
        or -1.0,
        int(candidate.get("_source_mtime_ns", 0)),
        str(candidate.get("feature_id", "")),
    )


def _single_case_marker_patterns(
    feature_values: dict[str, Optional[float]],
    candidates: Iterable[dict],
    *,
    measurement_confidence: float,
    delineation: Optional[RecordDelineation] = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        feature_id = str(candidate.get("feature_id", ""))
        raw_value = _float_or_none(feature_values.get(feature_id))
        projection_eligible, projection_restriction = (
            _laboratory_projection_eligibility(
                (feature_id,),
                delineation=delineation,
            )
        )
        if not projection_eligible:
            raw_value = None
        marker_model = dict(candidate.get("marker_model", {}))
        operating_points = _single_case_operating_points(
            raw_value,
            marker_model,
        )
        primary = operating_points.get("specificity_95", {})
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id", ""),
                "source_run_id": candidate.get("source_run_id", ""),
                "endpoint_id": candidate.get("endpoint_id", ""),
                "feature_id": feature_id,
                "feature_modality": candidate.get(
                    "feature_modality",
                    "",
                ),
                "conceptual_family": candidate.get(
                    "conceptual_family",
                    "",
                ),
                "panel_id": candidate.get("panel_id", ""),
                "raw_value": raw_value,
                "direction": marker_model.get("direction"),
                "operating_points": operating_points,
                "primary_operating_point": primary,
                "research_pattern_status": _research_pattern_status(
                    raw_value,
                    primary,
                ),
                "analysis_restriction": projection_restriction,
                "not_evaluable_reason": projection_restriction[
                    "abstention_reason"
                ],
                "measurement_confidence": measurement_confidence,
                "evidence_grade": candidate.get("evidence_grade", ""),
                "validation_metrics": dict(
                    candidate.get("validation_metrics", {})
                ),
                "result_scope": candidate.get("result_scope", ""),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row.get("primary_operating_point", {}).get("triggered") is True,
            {"A": 3, "B": 2, "C": 1}.get(
                str(row.get("evidence_grade", "")),
                0,
            ),
            _float_or_none(
                row.get("validation_metrics", {}).get("roc_auc")
            )
            or -1.0,
            _float_or_none(
                row.get("primary_operating_point", {}).get(
                    "distance_to_threshold"
                )
            )
            or -np.inf,
        ),
        reverse=True,
    )


def _laboratory_projection_eligibility(
    marker_ids: Iterable[str],
    *,
    delineation: Optional[RecordDelineation],
) -> tuple[bool, dict[str, Any]]:
    if delineation is None:
        return projection_case_eligibility(
            marker_ids,
            rhythm_context=None,
            rhythm_context_confidence=None,
            p_wave_coverage=None,
        )
    return projection_case_eligibility(
        marker_ids,
        rhythm_context=delineation.rhythm.rhythm_class.value,
        rhythm_context_confidence=getattr(
            delineation.rhythm,
            "confidence",
            None,
        ),
        p_wave_coverage=getattr(
            delineation.rhythm,
            "p_wave_coverage",
            None,
        ),
    )


def _run_implementation_compatibility(
    run_root: Path,
) -> tuple[bool, str]:
    current_hash = validation_implementation_hash()
    provenance_path = run_root / "development_provenance.json"
    try:
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False, "development provenance missing or unreadable"
    recorded_hash = str(provenance.get("code_hash", ""))
    if recorded_hash == current_hash:
        return True, "development implementation hash matches current code"
    attestation_path = run_root / "validation_code_scope_attestation.json"
    try:
        attestation = json.loads(
            attestation_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        attestation = {}
    attested_hash = str(attestation.get("implementation_hash", ""))
    if attested_hash == current_hash:
        return True, "legacy scientific-scope attestation matches current code"
    return (
        False,
        "historical score implementation differs from the audited current "
        f"code ({(attested_hash or recorded_hash)[:12]} != "
        f"{current_hash[:12]})",
    )


def _score_candidate_lookup(run_root: Path) -> dict[str, dict]:
    path = run_root / "frozen_score_candidates.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    output: dict[str, dict] = {}
    for candidate in payload.get("scores", ()):
        score_id = str(
            candidate.get("score_spec", {}).get("score_id", "")
        )
        if score_id:
            output[score_id] = dict(candidate)
    return output


def _score_validation_lookup(
    run_root: Path,
) -> dict[tuple[str, str], dict]:
    path = run_root / "combination_results.parquet"
    if not path.is_file():
        return {}
    wanted = (
        "endpoint_id",
        "feature_id",
        "evaluation_split",
        "analysis_type",
        "inference_status",
        "n_positive",
        "n_negative",
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "average_precision",
        "specificity_90_sensitivity",
        "specificity_95_sensitivity",
        "sensitivity_90_specificity",
        "validation_stable",
        "selected_for_freeze",
    )
    try:
        schema = pd.read_parquet(path).columns
        columns = [column for column in wanted if column in schema]
        frame = pd.read_parquet(path, columns=columns)
    except (OSError, ValueError):
        return {}
    if "evaluation_split" in frame:
        frame = frame[
            frame["evaluation_split"].astype(str).eq("validation")
        ]
    output: dict[tuple[str, str], dict] = {}
    for row in frame.itertuples(index=False):
        values = row._asdict()
        key = (
            str(values.get("endpoint_id", "")),
            str(values.get("feature_id", "")),
        )
        output[key] = {
            "analysis_type": str(values.get("analysis_type", "")),
            "inference_status": str(
                values.get("inference_status", "")
            ),
            "n_positive": _int_or_none(values.get("n_positive")),
            "n_negative": _int_or_none(values.get("n_negative")),
            "roc_auc": _float_or_none(values.get("roc_auc")),
            "roc_auc_ci_low": _float_or_none(
                values.get("roc_auc_ci_low")
            ),
            "roc_auc_ci_high": _float_or_none(
                values.get("roc_auc_ci_high")
            ),
            "average_precision": _float_or_none(
                values.get("average_precision")
            ),
            "sensitivity_at_90_specificity": _float_or_none(
                values.get("specificity_90_sensitivity")
            ),
            "sensitivity_at_95_specificity": _float_or_none(
                values.get("specificity_95_sensitivity")
            ),
            "specificity_at_90_sensitivity": _float_or_none(
                values.get("sensitivity_90_specificity")
            ),
            "validation_stable": _strict_bool(
                values.get("validation_stable")
            ),
            "selected_for_freeze": _strict_bool(
                values.get("selected_for_freeze")
            ),
        }
    return output


def _recommended_score_releases(
    releases: Iterable[dict],
) -> tuple[dict, ...]:
    selected: dict[str, dict] = {}
    for release in sorted(releases, key=_release_sort_key, reverse=True):
        endpoint_id = str(release.get("endpoint_id", ""))
        if endpoint_id and endpoint_id not in selected:
            selected[endpoint_id] = release
    return tuple(
        sorted(
            selected.values(),
            key=_release_sort_key,
            reverse=True,
        )
    )


def _release_sort_key(release: dict) -> tuple:
    metrics = dict(release.get("validation_metrics", {}))
    return (
        int(release.get("implementation_compatible") is True),
        _release_rank(str(release.get("stage", ""))),
        int(_strict_bool(metrics.get("selected_for_freeze"))),
        int(_strict_bool(metrics.get("validation_stable"))),
        _float_or_none(metrics.get("roc_auc")) or -1.0,
        _float_or_none(metrics.get("sensitivity_at_95_specificity"))
        or -1.0,
        str(release.get("created_at", "")),
        str(release.get("release_id", "")),
    )


def _release_evidence_rank(release: dict) -> float:
    metrics = dict(release.get("validation_metrics", {}))
    auc = _float_or_none(metrics.get("roc_auc")) or 0.5
    return float(
        100.0 * _release_rank(str(release.get("stage", "")))
        + 20.0 * int(_strict_bool(metrics.get("selected_for_freeze")))
        + 15.0 * int(_strict_bool(metrics.get("validation_stable")))
        + 100.0 * max(0.0, auc - 0.5)
    )


def _apply_annotation_revision(
    delineation: RecordDelineation,
    revision: AnnotationRevision,
) -> None:
    if not delineation.beats:
        raise ValueError(
            "A manual fiducial revision cannot be applied without detected beats"
        )
    normalized: dict[str, int] = {}
    for raw_name, raw_sample in revision.fiducials.items():
        name = _FIDUCIAL_ALIASES.get(str(raw_name), str(raw_name))
        if name not in {
            "p_onset",
            "p_peak",
            "p_offset",
            "q_peak",
            "r_onset",
            "r_peak",
            "r_offset",
            "t_onset",
            "t_peak",
            "t_offset",
            "u_onset",
            "u_peak",
            "u_offset",
        }:
            raise ValueError(f"Unsupported manual fiducial: {raw_name}")
        sample = int(raw_sample)
        if not 0 <= sample < delineation.n_samples:
            raise ValueError(
                f"Manual fiducial {raw_name}={sample} is outside the signal"
            )
        normalized[name] = sample
    if not normalized:
        raise ValueError("The manual fiducial revision contains no markers")

    requested_r = normalized.get("r_peak")
    if requested_r is None:
        representative = delineation.median_beat
        if representative is None:
            raise ValueError("No representative beat is available for correction")
        target_index = next(
            index
            for index, beat in enumerate(delineation.beats)
            if beat is representative
        )
    else:
        target_index = min(
            range(len(delineation.beats)),
            key=lambda index: abs(
                delineation.beats[index].r_peak.sample - requested_r
            ),
        )
    candidate = deepcopy(delineation.beats[target_index])
    for name, sample in normalized.items():
        setattr(
            candidate,
            name,
            MarkerPoint(
                sample=sample,
                confidence=1.0,
                source=f"manual:{revision.revision_id}",
            ),
        )
    if any(name.startswith("p_") for name in normalized):
        candidate.p_wave_absent = False
        candidate.p_status = WaveStatus.PRESENT
    if any(name.startswith("t_") for name in normalized):
        candidate.t_wave_absent = False
        candidate.t_status = WaveStatus.PRESENT
    if any(name.startswith("u_") for name in normalized):
        candidate.u_wave_absent = False
        candidate.u_status = WaveStatus.PRESENT
    elif any(
        name.startswith("t_")
        or name in {"r_onset", "r_peak", "r_offset"}
        for name in normalized
    ):
        # U-wave limits are derived relative to T and the following cycle.
        # A manual upstream revision invalidates, rather than silently keeps,
        # the old automatic U-wave estimate.
        candidate.u_onset = candidate.u_peak = candidate.u_offset = None
        candidate.u_wave_absent = False
        candidate.u_status = WaveStatus.INDETERMINATE
    if any(name in {"q_peak", "r_onset", "r_peak", "r_offset"} for name in normalized):
        candidate.qrs_status = WaveStatus.PRESENT
    if not delineation._is_physiologically_plausible(candidate):
        raise ValueError(
            "Manual fiducials violate signal bounds or physiological ordering"
        )
    delineation.beats[target_index] = candidate
    delineation.source_method = "fusion+manual_revision"
    delineation.invalidate_beat_cache()


def _representative_fiducials(
    delineation: RecordDelineation,
) -> dict[str, int]:
    beat = delineation.representative_beat
    if beat is None:
        return {}
    attributes = (
        "p_onset",
        "p_peak",
        "p_offset",
        "q_peak",
        "r_onset",
        "r_peak",
        "r_offset",
        "t_onset",
        "t_peak",
        "t_offset",
        "u_onset",
        "u_peak",
        "u_offset",
    )
    names = {
        "r_onset": "qrs_onset",
        "r_offset": "qrs_offset",
    }
    return {
        names.get(attribute, attribute): int(marker.sample)
        for attribute in attributes
        for marker in (getattr(beat, attribute),)
        if marker is not None
    }


def _representative_fiducial_details(
    delineation: RecordDelineation,
) -> dict[str, dict[str, Any]]:
    beat = delineation.representative_beat
    if beat is None:
        return {}
    attributes = (
        "p_onset",
        "p_peak",
        "p_offset",
        "q_peak",
        "r_onset",
        "r_peak",
        "r_offset",
        "t_onset",
        "t_peak",
        "t_offset",
        "u_onset",
        "u_peak",
        "u_offset",
    )
    names = {
        "r_onset": "qrs_onset",
        "r_offset": "qrs_offset",
    }
    wave_for_attribute = {
        "p_onset": "p",
        "p_peak": "p",
        "p_offset": "p",
        "q_peak": "qrs",
        "r_onset": "qrs",
        "r_peak": "qrs",
        "r_offset": "qrs",
        "t_onset": "t",
        "t_peak": "t",
        "t_offset": "t",
        "u_onset": "u",
        "u_peak": "u",
        "u_offset": "u",
    }
    details: dict[str, dict[str, Any]] = {}
    for attribute in attributes:
        marker = getattr(beat, attribute)
        if marker is None:
            continue
        wave = wave_for_attribute[attribute]
        details[names.get(attribute, attribute)] = {
            "sample": int(marker.sample),
            "time_seconds": marker.sample / delineation.fs,
            "confidence": float(marker.confidence),
            "uncertainty_ms": float(marker.uncertainty_ms),
            "source": marker.source,
            "support_count": int(marker.support_count),
            "lead_names": list(marker.lead_names),
            "evidence_sources": list(marker.evidence_sources),
            "conflict_ms": float(marker.conflict_ms),
            "wave_status": beat.resolved_wave_status(wave).value,
            "manually_reviewed": marker.manually_reviewed,
        }
    return details


def _representative_wave_segments(
    delineation: RecordDelineation,
) -> list[dict[str, Any]]:
    beat = delineation.representative_beat
    if beat is None:
        return []
    return _wave_segments_for_beat(beat, delineation.fs)


def _accepted_beat_wave_segments(
    delineation: RecordDelineation,
) -> list[dict[str, Any]]:
    accepted_ids = {id(beat) for beat in delineation.accepted_beats}
    segments: list[dict[str, Any]] = []
    for beat_index, beat in enumerate(delineation.beats):
        if id(beat) not in accepted_ids:
            continue
        for segment in _wave_segments_for_beat(beat, delineation.fs):
            segment["beat_index"] = int(beat_index)
            segment["r_peak_sample"] = int(beat.r_peak.sample)
            segments.append(segment)
    return sorted(
        segments,
        key=lambda item: (
            item["start_sample"] is None,
            item["start_sample"] if item["start_sample"] is not None else 0,
            item["segment"],
        ),
    )


def _wave_segments_for_beat(
    beat: BeatDelineation,
    fs: int,
) -> list[dict[str, Any]]:
    specifications = (
        ("p_wave", "p_onset", "p_offset", "p"),
        ("pr_segment", "p_offset", "r_onset", "pr"),
        ("qrs_complex", "r_onset", "r_offset", "qrs"),
        ("st_segment", "r_offset", "t_onset", "st"),
        ("t_wave", "t_onset", "t_offset", "t"),
        ("u_wave", "u_onset", "u_offset", "u"),
    )
    segments = []
    for (
        name,
        start_attribute,
        end_attribute,
        confidence_segment,
    ) in specifications:
        start = getattr(beat, start_attribute)
        end = getattr(beat, end_attribute)
        confidence = beat.segment_confidence(confidence_segment)
        measurable = not (
            start is None
            or end is None
            or end.sample <= start.sample
            or confidence < 0.35
        )
        segments.append(
            {
                "segment": name,
                "start_sample": (
                    int(start.sample) if start is not None else None
                ),
                "end_sample": (
                    int(end.sample) if end is not None else None
                ),
                "start_seconds": (
                    start.sample / fs
                    if start is not None
                    else None
                ),
                "end_seconds": (
                    end.sample / fs
                    if end is not None
                    else None
                ),
                "duration_ms": (
                    end.sample - start.sample
                ) / fs * 1000.0
                if (
                    start is not None
                    and end is not None
                    and end.sample > start.sample
                )
                else None,
                "confidence": float(confidence),
                "status": (
                    "measurable" if measurable else "not_evaluable"
                ),
                "reason": (
                    ""
                    if measurable
                    else _segment_not_evaluable_reason(
                        beat,
                        confidence_segment,
                        start,
                        end,
                        confidence,
                    )
                ),
            }
        )
    return segments


def _representative_intervals(
    delineation: RecordDelineation,
) -> list[dict[str, Any]]:
    beat = delineation.representative_beat
    if beat is None:
        return []
    specifications = (
        ("P duration", "p_onset", "p_offset", "p"),
        ("PR interval", "p_onset", "r_onset", "pr"),
        ("QRS duration", "r_onset", "r_offset", "qrs"),
        ("QT interval", "r_onset", "t_offset", "qrst"),
        ("JT interval", "r_offset", "t_offset", "st_t"),
        ("Tpeak-Tend", "t_peak", "t_offset", "t"),
        ("U duration", "u_onset", "u_offset", "u"),
        ("T-U separation", "t_offset", "u_onset", "tu"),
        ("QU interval", "r_onset", "u_offset", "qu"),
    )
    output = []
    for name, start_attribute, end_attribute, segment in specifications:
        start = getattr(beat, start_attribute)
        end = getattr(beat, end_attribute)
        confidence = delineation.segment_confidence(segment)
        if (
            start is None
            or end is None
            or end.sample <= start.sample
            or confidence < 0.35
        ):
            output.append(
                {
                    "interval": name,
                    "value_ms": None,
                    "confidence": float(confidence),
                    "status": "not_evaluable",
                    "reason": _segment_not_evaluable_reason(
                        beat,
                        segment,
                        start,
                        end,
                        confidence,
                    ),
                }
            )
            continue
        output.append(
            {
                "interval": name,
                "value_ms": (
                    end.sample - start.sample
                ) / delineation.fs * 1000.0,
                "measurement_method": "representative_beat_fiducials",
                "representative_r_sample": int(beat.r_peak.sample),
                "sampling_rate_hz": delineation.fs,
                "start_sample": int(start.sample),
                "end_sample": int(end.sample),
                "confidence": float(confidence),
                "uncertainty_ms": float(
                    np.hypot(start.uncertainty_ms, end.uncertainty_ms)
                ),
                "status": "measurable",
            }
        )
    measurements = delineation.qt_rr_measurements()
    required_pairs = (
        5 if delineation.rhythm.requires_beatwise_rate_correction else 1
    )
    correction_specs = (
        ("QTc Bazett", lambda qt, rr: qt / np.sqrt(rr)),
        ("QTc Fridericia", lambda qt, rr: qt / np.cbrt(rr)),
        ("QTc Framingham", lambda qt, rr: qt + 154.0 * (1.0 - rr)),
        ("QTc Hodges", lambda qt, rr: qt + 1.75 * (60.0 / rr - 60.0)),
    )
    for name, correction in correction_specs:
        if len(measurements) < required_pairs:
            output.append(
                {
                    "interval": name,
                    "value_ms": None,
                    "confidence": 0.0,
                    "status": "not_evaluable",
                    "reason": (
                        f"{len(measurements)}/{required_pairs} valid "
                        "beatwise QT-RR pairs"
                    ),
                }
            )
            continue
        qt_values = np.asarray(
            [item.qt_ms for item in measurements],
            dtype=float,
        )
        rr_values = np.asarray(
            [item.preceding_rr_seconds for item in measurements],
            dtype=float,
        )
        corrected = correction(qt_values, rr_values)
        output.append(
            {
                "interval": name,
                "value_ms": float(np.median(corrected)),
                "confidence": float(
                    np.median([item.confidence for item in measurements])
                    * min(1.0, 0.55 + 0.09 * len(measurements))
                ),
                "uncertainty_ms": float(
                    np.percentile(corrected, 75)
                    - np.percentile(corrected, 25)
                ),
                "status": "measurable",
                "measurement_method": "beatwise_preceding_rr_median",
                "n_pairs": len(measurements),
                "rr_seconds_median": float(np.median(rr_values)),
                "rr_seconds_min": float(np.min(rr_values)),
                "rr_seconds_max": float(np.max(rr_values)),
                "uncertainty_kind": "between_beat_iqr_not_measurement_ci",
                "formula": {
                    "QTc Bazett": "QT_ms / sqrt(RR_s)",
                    "QTc Fridericia": "QT_ms / cbrt(RR_s)",
                    "QTc Framingham": "QT_ms + 154*(1-RR_s)",
                    "QTc Hodges": "QT_ms + 1.75*(60/RR_s-60)",
                }[name],
                "interpretation_note": "Beatwise correction with preceding RR; no QT adaptation/hysteresis model. IQR is not a measurement confidence interval.",
            }
        )
    return output


def _segment_not_evaluable_reason(
    beat,
    segment: str,
    start: Optional[MarkerPoint],
    end: Optional[MarkerPoint],
    confidence: float,
) -> str:
    dependencies = {
        "p": ("p",),
        "pr": ("p", "qrs"),
        "qrs": ("qrs",),
        "st": ("qrs", "t"),
        "t": ("t",),
        "u": ("u",),
        "tu": ("t", "u"),
        "qu": ("qrs", "u"),
        "qrst": ("qrs", "t"),
        "st_t": ("qrs", "t"),
        "beat": ("qrs", "t"),
    }
    states = {
        wave: beat.resolved_wave_status(wave).value
        for wave in dependencies.get(segment, ())
        if beat.resolved_wave_status(wave) != WaveStatus.PRESENT
    }
    if states:
        return ", ".join(
            f"{wave.upper()} status {status}"
            for wave, status in states.items()
        )
    if start is None or end is None:
        return "required fiducial boundary missing"
    if end.sample <= start.sample:
        return "physiological boundary order invalid"
    if beat.has_excessive_segment_uncertainty(segment):
        return (
            "fiducial boundary uncertainty exceeds the validated "
            "segment-specific limit"
        )
    if confidence < measurement_confidence_threshold(segment):
        return (
            "fiducial confidence below the wave-specific measurement floor "
            f"({confidence:.0%})"
        )
    return "segment not assessable"


def _release_component_names(releases: Iterable[dict]) -> tuple[str, ...]:
    selected: list[str] = []
    for release in releases:
        score_spec = release.get("score_spec", {})
        for component in score_spec.get("components", ()):
            feature_id = str(component)
            if feature_id.startswith("classical_"):
                continue
            try:
                BiomarkerRegistry.definition(feature_id)
            except ValueError:
                continue
            if feature_id not in selected:
                selected.append(feature_id)
    return tuple(selected)


def _release_rank(stage: str) -> int:
    order = {
        ScoreReleaseStage.DEVELOPMENT.value: 0,
        ScoreReleaseStage.FROZEN.value: 1,
        ScoreReleaseStage.LOCKED_TEST_VALIDATED.value: 2,
        ScoreReleaseStage.EXTERNAL_VALIDATED.value: 3,
    }
    return order.get(str(stage), -1)


def _score_component_details(
    features: dict[str, Optional[float]],
    score_payload: dict,
) -> list[dict[str, Any]]:
    components = tuple(str(item) for item in score_payload.get("components", ()))
    directions = tuple(int(item) for item in score_payload.get("directions", ()))
    means = tuple(float(item) for item in score_payload.get("means", ()))
    standard_deviations = tuple(
        float(item)
        for item in score_payload.get("standard_deviations", ())
    )
    if len({len(components), len(directions), len(means), len(standard_deviations)}) != 1:
        return []
    output: list[dict[str, Any]] = []
    for feature_id, direction, mean, standard_deviation in zip(
        components,
        directions,
        means,
        standard_deviations,
    ):
        raw_value = _float_or_none(features.get(feature_id))
        contribution = (
            direction * (raw_value - mean) / standard_deviation
            if raw_value is not None and standard_deviation > 0
            else None
        )
        output.append(
            {
                "feature_id": feature_id,
                "raw_value": raw_value,
                "direction": direction,
                "train_mean": mean,
                "train_standard_deviation": standard_deviation,
                "aligned_z_contribution": contribution,
                "contribution_direction": (
                    "supports_positive_reference"
                    if contribution is not None and contribution > 0
                    else "supports_negative_reference"
                    if contribution is not None and contribution < 0
                    else "neutral_or_not_evaluable"
                ),
            }
        )
    return output


def _single_case_operating_points(
    research_score: Optional[float],
    marker_model: dict,
) -> dict[str, dict[str, Any]]:
    score = _float_or_none(research_score)
    direction = int(marker_model.get("direction", 1) or 1)
    if direction not in (-1, 1):
        direction = 1
    aligned_score = direction * score if score is not None else None
    output: dict[str, dict[str, Any]] = {}
    for name in ("specificity_95", "specificity_90", "sensitivity_90"):
        threshold = _float_or_none(
            dict(marker_model.get("thresholds", {})).get(name)
        )
        distance = (
            aligned_score - threshold
            if aligned_score is not None and threshold is not None
            else None
        )
        output[name] = {
            "target": name,
            "threshold_aligned": threshold,
            "score_direction": direction,
            "aligned_score": aligned_score,
            "distance_to_threshold": distance,
            "triggered": (
                bool(distance >= 0.0) if distance is not None else None
            ),
            "threshold_source": "train_only",
        }
    return output


def _research_pattern_status(
    research_score: Optional[float],
    operating_point: dict,
) -> str:
    if _float_or_none(research_score) is None:
        return "not_evaluable"
    triggered = operating_point.get("triggered")
    if triggered is True:
        return "above_train_95_specificity_threshold"
    if triggered is False:
        return "below_train_95_specificity_threshold"
    return "evaluable_without_threshold"


def _classical_comparators_for_endpoint(
    endpoint_id: str,
    classical_payload: dict[str, dict],
    *,
    registry: Optional[BaselineRegistry] = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    active_registry = registry or BaselineRegistry()
    for definition in active_registry.for_endpoint(endpoint_id):
        criterion_id = str(definition.criterion_id or "")
        if not criterion_id:
            output.append(
                {
                    "baseline_id": definition.baseline_id,
                    "display_name": definition.display_name,
                    "kind": definition.kind.value,
                    "criterion_id": "",
                    "evaluable": False,
                    "positive": None,
                    "value": None,
                    "exclusion_reason": "no established ECG diagnostic baseline",
                    "sollwert": definition.formula,
                    "notes": definition.notes,
                }
            )
            continue
        result = dict(classical_payload.get(criterion_id, {}))
        output.append(
            {
                "baseline_id": definition.baseline_id,
                "display_name": definition.display_name,
                "kind": definition.kind.value,
                "criterion_id": criterion_id,
                "evaluable": result.get("evaluable", False),
                "positive": result.get("positive"),
                "value": result.get("value"),
                "exclusion_reason": result.get("exclusion_reason", ""),
                # Literature-sourced expected/target rule ("Sollwert") for
                # this criterion -- the plausibility-check counterpart to
                # `value`: does the computed value/verdict land where the
                # guideline formula says it should?
                "sollwert": definition.formula,
                "notes": definition.notes,
            }
        )
    return output


def _release_evidence_status(
    stage: ScoreReleaseStage,
    metrics: dict,
) -> str:
    if stage == ScoreReleaseStage.EXTERNAL_VALIDATED:
        return "external_validated_research"
    if stage == ScoreReleaseStage.LOCKED_TEST_VALIDATED:
        return "locked_test_validated_research"
    if _strict_bool(metrics.get("validation_stable")):
        return "development_validation_stable"
    if metrics:
        return "development_validation_unstable_or_unselected"
    return "development_evidence_unavailable"


def _research_pattern_summary(
    scores: Iterable[dict],
    *,
    single_marker_patterns: Iterable[dict] = (),
) -> list[dict[str, Any]]:
    single_by_endpoint: dict[str, list[dict]] = {}
    for marker in single_marker_patterns:
        single_by_endpoint.setdefault(
            str(marker.get("endpoint_id", "")),
            [],
        ).append(marker)
    rows: list[dict[str, Any]] = []
    for score in scores:
        primary = dict(score.get("primary_operating_point", {}))
        metrics = dict(score.get("validation_metrics", {}))
        distance = _float_or_none(primary.get("distance_to_threshold"))
        endpoint_id = str(score.get("endpoint_id", ""))
        supporting = single_by_endpoint.pop(endpoint_id, [])
        triggered_support = [
            item
            for item in supporting
            if item.get("primary_operating_point", {}).get("triggered")
            is True
        ]
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "research_pattern_status": score.get(
                    "research_pattern_status",
                    "not_evaluable",
                ),
                "triggered_at_train_95_specificity": primary.get(
                    "triggered"
                ),
                "threshold_distance": distance,
                "measurement_confidence": score.get(
                    "measurement_confidence"
                ),
                "evidence_status": score.get("evidence_status", ""),
                "validation_roc_auc": metrics.get("roc_auc"),
                "validation_roc_auc_ci_low": metrics.get(
                    "roc_auc_ci_low"
                ),
                "validation_roc_auc_ci_high": metrics.get(
                    "roc_auc_ci_high"
                ),
                "validation_sensitivity_at_95_specificity": metrics.get(
                    "sensitivity_at_95_specificity"
                ),
                "classical_positive_count": score.get(
                    "classical_positive_count",
                    0,
                ),
                "supporting_single_marker_count": len(supporting),
                "triggered_single_marker_count": len(triggered_support),
                "triggered_single_marker_ids": [
                    item.get("feature_id", "")
                    for item in triggered_support
                ],
                "release_id": score.get("release_id", ""),
                "interpretation": (
                    "research-pattern compatibility only; no diagnosis or "
                    "disease probability"
                ),
            }
        )
    for endpoint_id, supporting in single_by_endpoint.items():
        triggered_support = [
            item
            for item in supporting
            if item.get("primary_operating_point", {}).get("triggered")
            is True
        ]
        best_auc = max(
            (
                _float_or_none(
                    item.get("validation_metrics", {}).get("roc_auc")
                )
                or -1.0
                for item in supporting
            ),
            default=-1.0,
        )
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "research_pattern_status": (
                    "single_marker_support_only"
                    if triggered_support
                    else "single_markers_below_train_95_specificity_threshold"
                ),
                "triggered_at_train_95_specificity": None,
                "threshold_distance": None,
                "measurement_confidence": max(
                    (
                        _float_or_none(
                            item.get("measurement_confidence")
                        )
                        or 0.0
                        for item in supporting
                    ),
                    default=0.0,
                ),
                "evidence_status": "development_single_marker_support",
                "validation_roc_auc": (
                    best_auc if best_auc >= 0.0 else None
                ),
                "validation_roc_auc_ci_low": None,
                "validation_roc_auc_ci_high": None,
                "validation_sensitivity_at_95_specificity": None,
                "classical_positive_count": 0,
                "supporting_single_marker_count": len(supporting),
                "triggered_single_marker_count": len(triggered_support),
                "triggered_single_marker_ids": [
                    item.get("feature_id", "")
                    for item in triggered_support
                ],
                "release_id": "",
                "interpretation": (
                    "single-marker support only; no validated combined score, "
                    "diagnosis or disease probability"
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row.get("triggered_at_train_95_specificity") is True,
            int(row.get("triggered_single_marker_count", 0)),
            _release_rank_from_evidence_status(
                str(row.get("evidence_status", ""))
            ),
            _float_or_none(row.get("validation_roc_auc")) or -1.0,
            _float_or_none(row.get("threshold_distance")) or -np.inf,
        ),
        reverse=True,
    )


def _release_rank_from_evidence_status(status: str) -> int:
    return {
        "development_evidence_unavailable": 0,
        "development_validation_unstable_or_unselected": 1,
        "development_single_marker_support": 1,
        "development_validation_stable": 2,
        "locked_test_validated_research": 3,
        "external_validated_research": 4,
    }.get(status, -1)


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return False


def _float_or_none(value: object) -> Optional[float]:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _int_or_none(value: object) -> Optional[int]:
    numeric = _float_or_none(value)
    return int(numeric) if numeric is not None else None


def _apply_single_case_score(
    features: dict[str, Optional[float]],
    score_payload: dict,
) -> Optional[float]:
    spec_payload = dict(score_payload)
    for key in (
        "components",
        "directions",
        "means",
        "standard_deviations",
    ):
        spec_payload[key] = tuple(spec_payload.get(key, ()))
    try:
        spec = FrozenScoreSpec(**spec_payload)
    except (TypeError, ValueError):
        return None
    components = [_float_or_none(features.get(item)) for item in spec.components]
    if any(value is None for value in components):
        return None
    numeric_components = [value for value in components if value is not None]
    aligned = [
        direction * (value - mean) / standard_deviation
        for value, direction, mean, standard_deviation in zip(
            numeric_components,
            spec.directions,
            spec.means,
            spec.standard_deviations,
        )
    ]
    return float(np.mean(aligned))


def _atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _source_bundle_paths(
    source: Path,
    source_format: str,
) -> tuple[Path, ...]:
    if source_format != "wfdb":
        return (source.resolve(),)
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install wfdb to inspect WFDB record provenance") from exc

    record_base = (
        source.with_suffix("")
        if source.suffix.lower() in (".hea", ".dat")
        else source
    ).resolve()
    collected: set[Path] = set()

    def collect(record, base: Path) -> None:
        header_path = base.with_suffix(".hea").resolve()
        if header_path.is_file():
            collected.add(header_path)
        for file_name in tuple(getattr(record, "file_name", ()) or ()):
            if not file_name or str(file_name) == "~":
                continue
            signal_path = (base.parent / str(file_name)).resolve()
            if signal_path.is_file():
                collected.add(signal_path)
        segment_names = tuple(getattr(record, "seg_name", ()) or ())
        segments = tuple(getattr(record, "segments", ()) or ())
        for index, segment in enumerate(segments):
            if segment is None:
                continue
            segment_name = (
                str(segment_names[index])
                if index < len(segment_names)
                else str(getattr(segment, "record_name", ""))
            )
            if not segment_name or segment_name == "~":
                continue
            collect(segment, (base.parent / segment_name).resolve())

    header = wfdb.rdheader(str(record_base), rd_segments=True)
    collect(header, record_base)
    if not collected:
        raise FileNotFoundError(
            f"No local files found for WFDB record {record_base}"
        )
    return tuple(sorted(collected, key=lambda item: item.as_posix().lower()))


def _source_bundle_fingerprint(paths: Iterable[Path]) -> str:
    resolved = tuple(Path(path).resolve() for path in paths)
    if len(resolved) == 1:
        return file_sha256(resolved[0])
    common_root = Path(os.path.commonpath([str(path.parent) for path in resolved]))
    return content_hash(
        {
            "fingerprint_kind": "wfdb_record_bundle_sha256_v1",
            "files": [
                {
                    "relative_path": path.relative_to(common_root).as_posix(),
                    "sha256": file_sha256(path),
                }
                for path in resolved
            ],
        }
    )


def _copy_source_bundle_atomic(
    paths: Iterable[Path],
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    for source in paths:
        source = Path(source).resolve()
        try:
            relative = source.relative_to(source_root)
        except ValueError:
            relative = Path(source.name)
        destination = (destination_root / relative).resolve()
        try:
            destination.relative_to(destination_root)
        except ValueError as exc:
            raise ValueError("WFDB source path escapes the case library") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        shutil.copy2(source, temporary)
        atomic_replace(temporary, destination)
