from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from jsonschema import Draft202012Validator

from .storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    capture_output_integrity,
    content_hash,
)


CLINICAL_UTILITY_SCHEMA_VERSION = "2.0.0"
CLINICAL_UTILITY_STAGE_VERSION = "2.0.0"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_SPLITS = frozenset(
    {
        "development_train",
        "development_validation",
        "external_validation",
        "prospective_impact",
        "governance",
    }
)


@dataclass(frozen=True)
class AuthoritativeSource:
    source_id: str
    title: str
    organization: str
    year: int
    url: str
    role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AUTHORITATIVE_SOURCES: tuple[AuthoritativeSource, ...] = (
    AuthoritativeSource(
        source_id="aha_biomarker_utility_2026",
        title=(
            "Criteria to Assess the Predictive and Clinical Utility of "
            "Novel Models, Biomarkers, and Tools for Risk of Cardiovascular "
            "Disease: A Scientific Statement From the American Heart "
            "Association"
        ),
        organization="American Heart Association",
        year=2026,
        url="https://doi.org/10.1161/CIR.0000000000001401",
        role=(
            "Primary framework for intended population, predictive utility, "
            "calibration, incremental value, clinically defined "
            "reclassification, decision curves, actionability, scalability, "
            "generalizability, cost and health equity."
        ),
    ),
    AuthoritativeSource(
        source_id="fda_nih_best_validation",
        title="BEST Resource: Validation",
        organization="FDA-NIH Biomarker Working Group",
        year=2020,
        url="https://www.ncbi.nlm.nih.gov/books/NBK464453/",
        role=(
            "Fit-for-purpose analytical and clinical validation tied to a "
            "prespecified context of use."
        ),
    ),
    AuthoritativeSource(
        source_id="tripod_ai_2024",
        title=(
            "TRIPOD+AI statement: updated guidance for reporting clinical "
            "prediction models that use regression or machine learning methods"
        ),
        organization="TRIPOD+AI Group / BMJ",
        year=2024,
        url="https://doi.org/10.1136/bmj-2023-078378",
        role=(
            "Reporting standard for diagnostic or prognostic multivariable "
            "prediction-model studies; it is not a quality appraisal tool."
        ),
    ),
    AuthoritativeSource(
        source_id="probast_ai_2025",
        title=(
            "PROBAST+AI: an updated quality, risk of bias, and applicability "
            "assessment tool for prediction models using regression or "
            "artificial intelligence methods"
        ),
        organization="PROBAST+AI Group / BMJ",
        year=2025,
        url="https://doi.org/10.1136/bmj-2024-082505",
        role=(
            "Quality, risk-of-bias and applicability appraisal for prediction "
            "model development and performance evaluation."
        ),
    ),
    AuthoritativeSource(
        source_id="stard_ai_2025",
        title=(
            "The STARD-AI reporting guideline for diagnostic accuracy studies "
            "using artificial intelligence"
        ),
        organization="STARD-AI Steering Committee / Nature Medicine",
        year=2025,
        url="https://doi.org/10.1038/s41591-025-03953-8",
        role=(
            "Reporting standard for AI-centred diagnostic-accuracy studies; "
            "it does not itself establish clinical utility."
        ),
    ),
)


class EvidenceStatus(StrEnum):
    NOT_ASSESSED = "not_assessed"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    SUPPORTED = "supported"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class FrameworkScope(StrEnum):
    """Scientific framework whose claims are being assessed.

    The AHA statement is directly applicable only to incident cardiovascular
    risk prediction.  Diagnostic ECG/VCG use therefore has an explicitly
    separate adaptation scope and must never be labelled as direct AHA
    compliance.
    """

    AHA_INCIDENT_CVD_RISK = "aha_incident_cvd_risk"
    DIAGNOSTIC_ADAPTATION = "diagnostic_adaptation"


class EvaluationFamily(StrEnum):
    DIAGNOSTIC_ACCURACY = "diagnostic_accuracy"
    PREDICTION_MODEL = "prediction_model"


class ClinicalTask(StrEnum):
    DIAGNOSIS = "diagnosis"
    SCREENING = "screening"
    TRIAGE = "triage"
    PROGNOSIS = "prognosis"
    RISK_STRATIFICATION = "risk_stratification"
    TREATMENT_SELECTION = "treatment_selection"
    MONITORING = "monitoring"


class ClaimLevel(StrEnum):
    HYPOTHESIS_ONLY = "hypothesis_only"
    ANALYTICAL_VALIDITY_SUPPORTED = "analytical_validity_supported"
    CLINICAL_ASSOCIATION_SUPPORTED = "clinical_association_supported"
    INTERNAL_VALIDATION_SUPPORTED = "internal_validation_supported"
    EXTERNAL_VALIDATION_SUPPORTED = "external_validation_supported"
    POTENTIAL_CLINICAL_UTILITY = "potential_clinical_utility"
    DECISION_PROCESS_UTILITY_DEMONSTRATED = (
        "decision_process_utility_demonstrated"
    )
    PATIENT_BENEFIT_DEMONSTRATED = "patient_benefit_demonstrated"


CLAIM_LEVEL_ORDER: tuple[ClaimLevel, ...] = (
    ClaimLevel.HYPOTHESIS_ONLY,
    ClaimLevel.ANALYTICAL_VALIDITY_SUPPORTED,
    ClaimLevel.CLINICAL_ASSOCIATION_SUPPORTED,
    ClaimLevel.INTERNAL_VALIDATION_SUPPORTED,
    ClaimLevel.EXTERNAL_VALIDATION_SUPPORTED,
    ClaimLevel.POTENTIAL_CLINICAL_UTILITY,
    ClaimLevel.DECISION_PROCESS_UTILITY_DEMONSTRATED,
    ClaimLevel.PATIENT_BENEFIT_DEMONSTRATED,
)


CLAIM_TEXT_DE: dict[ClaimLevel, str] = {
    ClaimLevel.HYPOTHESIS_ONLY: (
        "Hypothesengenerierender Zusammenhang; kein klinischer Nutzen belegt."
    ),
    ClaimLevel.ANALYTICAL_VALIDITY_SUPPORTED: (
        "Analytische Validität für den festgelegten Verwendungszweck "
        "unterstützt; kein klinischer Nutzen belegt."
    ),
    ClaimLevel.CLINICAL_ASSOCIATION_SUPPORTED: (
        "Klinische Assoziation unterstützt; diagnostische oder "
        "prognostische Leistung und klinischer Nutzen nicht belegt."
    ),
    ClaimLevel.INTERNAL_VALIDATION_SUPPORTED: (
        "Interne klinische Leistung unterstützt; externe "
        "Generalisierbarkeit und klinischer Nutzen nicht belegt."
    ),
    ClaimLevel.EXTERNAL_VALIDATION_SUPPORTED: (
        "Externe diagnostische oder prognostische Leistung unterstützt; "
        "klinischer Nutzen nicht belegt."
    ),
    ClaimLevel.POTENTIAL_CLINICAL_UTILITY: (
        "Potentieller klinischer Nutzen unterstützt; prospektiver Einfluss "
        "auf Entscheidungen oder patientenrelevante Ergebnisse nicht "
        "nachgewiesen."
    ),
    ClaimLevel.DECISION_PROCESS_UTILITY_DEMONSTRATED: (
        "Klinischer Entscheidungs- oder Prozessnutzen nachgewiesen; ein "
        "patientenrelevanter Nutzen ist damit nicht belegt."
    ),
    ClaimLevel.PATIENT_BENEFIT_DEMONSTRATED: (
        "Patientenrelevanter klinischer Nutzen in einer prospektiven, "
        "randomisierten Impact-Studie nachgewiesen."
    ),
}


def _required_text(value: str, label: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{label} is required")


def _status_value(status: EvidenceStatus | str) -> str:
    return status.value if isinstance(status, EvidenceStatus) else str(status)


def _is_supported(status: EvidenceStatus | str) -> bool:
    return _status_value(status) == EvidenceStatus.SUPPORTED.value


def _has_traceable_ids(values: tuple[str, ...]) -> bool:
    normalized = tuple(str(value).strip() for value in values)
    return (
        bool(normalized)
        and all(normalized)
        and len(set(normalized)) == len(normalized)
    )


def _has_nonempty_text_items(values: tuple[str, ...]) -> bool:
    return bool(values) and all(str(value).strip() for value in values)


def _json_metadata(value: Any) -> Any:
    """Return only native JSON containers for schema validation/output."""

    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _required_sha256(value: str, label: str) -> None:
    if not _SHA256_PATTERN.fullmatch(str(value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _required_iso_datetime(value: str, label: str) -> None:
    _required_text(value, label)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")


@dataclass(frozen=True)
class EvidenceRecord:
    """Tamper-evident metadata for one aggregate evidence artifact.

    ``artifact_uri`` and ``manifest_uri`` are identifiers only.  The utility
    evaluator never dereferences them.  Their SHA-256 values must have been
    produced by the evidence-generating stage and are bound into the signed
    registry metadata.  Patient-level and locked-test artifacts are explicitly
    outside this registry's admissible scope.
    """

    evidence_id: str
    evidence_kind: str
    artifact_uri: str
    artifact_sha256: str
    manifest_id: str
    manifest_uri: str
    manifest_sha256: str
    dataset_id: str
    site_id: str
    split: str
    assessor_id: str
    assessor_organization: str
    assessed_at: str
    domains: tuple[str, ...]
    independent_from_development: bool = False
    contains_patient_level_data: bool = False
    contains_locked_test_data: bool = False
    contains_locked_test_outcomes: bool = False
    record_sha256: str = ""

    def __post_init__(self) -> None:
        for label in (
            "evidence_id",
            "evidence_kind",
            "artifact_uri",
            "manifest_id",
            "manifest_uri",
            "dataset_id",
            "site_id",
            "split",
            "assessor_id",
            "assessor_organization",
        ):
            _required_text(str(getattr(self, label)), label)
        if not str(self.artifact_uri).startswith(("urn:", "evidence:", "https://")):
            raise ValueError("artifact_uri must be a non-file evidence identifier")
        if not str(self.manifest_uri).startswith(("urn:", "evidence:", "https://")):
            raise ValueError("manifest_uri must be a non-file manifest identifier")
        _required_sha256(self.artifact_sha256, "artifact_sha256")
        _required_sha256(self.manifest_sha256, "manifest_sha256")
        _required_iso_datetime(self.assessed_at, "assessed_at")
        if self.split not in _EVIDENCE_SPLITS:
            raise ValueError(
                "EvidenceRecord split must be an explicit development, "
                "external, prospective-impact, or governance split"
            )
        if not _has_nonempty_text_items(self.domains):
            raise ValueError("EvidenceRecord domains must be non-empty")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("EvidenceRecord domains must be unique")
        if (
            self.contains_patient_level_data
            or self.contains_locked_test_data
            or self.contains_locked_test_outcomes
        ):
            raise ValueError(
                "Clinical-utility evidence registry accepts aggregate, "
                "non-locked metadata only"
            )
        expected = content_hash(self._hash_payload())
        if self.record_sha256:
            _required_sha256(self.record_sha256, "record_sha256")
            if self.record_sha256 != expected:
                raise ValueError(
                    f"EvidenceRecord {self.evidence_id!r} failed integrity validation"
                )
        else:
            object.__setattr__(self, "record_sha256", expected)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_kind": self.evidence_kind,
            "artifact_uri": self.artifact_uri,
            "artifact_sha256": self.artifact_sha256,
            "manifest_id": self.manifest_id,
            "manifest_uri": self.manifest_uri,
            "manifest_sha256": self.manifest_sha256,
            "dataset_id": self.dataset_id,
            "site_id": self.site_id,
            "split": self.split,
            "assessor_id": self.assessor_id,
            "assessor_organization": self.assessor_organization,
            "assessed_at": self.assessed_at,
            "domains": list(self.domains),
            "independent_from_development": self.independent_from_development,
            "contains_patient_level_data": self.contains_patient_level_data,
            "contains_locked_test_data": self.contains_locked_test_data,
            "contains_locked_test_outcomes": self.contains_locked_test_outcomes,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self._hash_payload()
        data["record_sha256"] = self.record_sha256
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRecord":
        if not str(payload.get("record_sha256", "")):
            raise ValueError("Persisted EvidenceRecord requires record_sha256")
        return _construct_dataclass(
            cls,
            payload,
            tuple_fields=("domains",),
        )


@dataclass(frozen=True)
class EvidenceRegistry:
    registry_id: str
    development_site_id: str
    created_at: str
    created_by: str
    records: tuple[EvidenceRecord, ...]
    registry_sha256: str = ""

    def __post_init__(self) -> None:
        for label in ("registry_id", "development_site_id", "created_by"):
            _required_text(str(getattr(self, label)), label)
        _required_iso_datetime(self.created_at, "created_at")
        identifiers = tuple(record.evidence_id for record in self.records)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Evidence registry contains duplicate evidence IDs")
        expected = content_hash(self._hash_payload())
        if self.registry_sha256:
            _required_sha256(self.registry_sha256, "registry_sha256")
            if self.registry_sha256 != expected:
                raise ValueError("Evidence registry failed integrity validation")
        else:
            object.__setattr__(self, "registry_sha256", expected)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "development_site_id": self.development_site_id,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "records": [record.to_dict() for record in self.records],
        }

    def to_dict(self) -> dict[str, Any]:
        data = self._hash_payload()
        data["registry_sha256"] = self.registry_sha256
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRegistry":
        if not str(payload.get("registry_sha256", "")):
            raise ValueError("Persisted evidence registry requires registry_sha256")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown EvidenceRegistry fields: {unknown}")
        return cls(
            registry_id=str(payload["registry_id"]),
            development_site_id=str(payload["development_site_id"]),
            created_at=str(payload["created_at"]),
            created_by=str(payload["created_by"]),
            records=tuple(
                EvidenceRecord.from_dict(dict(item))
                for item in payload.get("records", ())
            ),
            registry_sha256=str(payload["registry_sha256"]),
        )

    def resolve(self, evidence_id: str) -> EvidenceRecord:
        for record in self.records:
            if record.evidence_id == evidence_id:
                return record
        raise KeyError(evidence_id)


class AhaMetricId(StrEnum):
    ASSOCIATION_RR_PER_SD = "association_rr_per_sd"
    ASSOCIATION_CATEGORICAL_RR = "association_categorical_rr"
    MEASUREMENT_COEFFICIENT_OF_VARIATION = (
        "measurement_coefficient_of_variation"
    )
    DISCRIMINATION_C_STATISTIC = "discrimination_c_statistic"
    CALIBRATION_IN_THE_LARGE = "calibration_in_the_large"
    CALIBRATION_SLOPE = "calibration_slope"
    CALIBRATION_SOCIODEMOGRAPHIC_SUBGROUPS = (
        "calibration_sociodemographic_subgroups"
    )
    INCREMENTAL_DELTA_C_STATISTIC = "incremental_delta_c_statistic"
    INCREMENTAL_DELTA_CALIBRATION = "incremental_delta_calibration"
    CATEGORICAL_NRI = "categorical_nri"
    EXTERNAL_VALIDATION = "external_validation"
    DECISION_CURVE = "decision_curve"
    COST_EFFECTIVENESS = "cost_effectiveness"
    ACTIONABILITY = "actionability"


AHA_REQUIRED_METRICS: tuple[AhaMetricId, ...] = tuple(AhaMetricId)
_AHA_NUMERIC_METRICS = frozenset(
    {
        AhaMetricId.ASSOCIATION_RR_PER_SD,
        AhaMetricId.ASSOCIATION_CATEGORICAL_RR,
        AhaMetricId.MEASUREMENT_COEFFICIENT_OF_VARIATION,
        AhaMetricId.DISCRIMINATION_C_STATISTIC,
        AhaMetricId.CALIBRATION_IN_THE_LARGE,
        AhaMetricId.CALIBRATION_SLOPE,
        AhaMetricId.INCREMENTAL_DELTA_C_STATISTIC,
        AhaMetricId.INCREMENTAL_DELTA_CALIBRATION,
        AhaMetricId.CATEGORICAL_NRI,
    }
)
_AHA_UNCERTAINTY_METRICS = _AHA_NUMERIC_METRICS - {
    AhaMetricId.MEASUREMENT_COEFFICIENT_OF_VARIATION,
}


@dataclass(frozen=True)
class AhaMetricRecord:
    metric_id: AhaMetricId
    status: EvidenceStatus
    estimate: Optional[float]
    unit: str
    confidence_interval_low: Optional[float]
    confidence_interval_high: Optional[float]
    uncertainty_reported: bool
    population: str
    time_horizon: str
    comparator_id: str
    threshold_definition: str
    external_validation: bool
    prespecified: bool
    interpretation: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label in ("unit", "population", "time_horizon", "interpretation"):
            _required_text(str(getattr(self, label)), label)
        if self.metric_id in _AHA_NUMERIC_METRICS:
            if self.estimate is None or not math.isfinite(float(self.estimate)):
                raise ValueError(f"{self.metric_id.value} requires a finite estimate")
        if self.estimate is not None and not math.isfinite(float(self.estimate)):
            raise ValueError("AHA metric estimate must be finite")
        for value in (self.confidence_interval_low, self.confidence_interval_high):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("AHA metric confidence interval must be finite")
        if (
            self.confidence_interval_low is not None
            and self.confidence_interval_high is not None
            and self.confidence_interval_low > self.confidence_interval_high
        ):
            raise ValueError("AHA metric confidence interval is inverted")
        if self.metric_id in _AHA_UNCERTAINTY_METRICS and self.uncertainty_reported:
            if (
                self.confidence_interval_low is None
                or self.confidence_interval_high is None
            ):
                raise ValueError(
                    f"{self.metric_id.value} declares uncertainty without a confidence interval"
                )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metric_id"] = self.metric_id.value
        data["status"] = self.status.value
        return _json_metadata(data)


@dataclass(frozen=True)
class ScopeFreezeAttestation:
    attestation_id: str
    candidate_id: str
    use_id: str
    intended_use_sha256: str
    comparator_sha256: str
    thresholds_sha256: str
    evidence_registry_sha256: str
    custody_evidence_id: str
    custody_manifest_sha256: str
    attested_by: str
    attested_at: str
    frozen_before_evaluation: bool
    locked_test_data_attested_unread: bool
    locked_test_outcomes_attested_unread: bool
    attestation_sha256: str = ""

    def __post_init__(self) -> None:
        for label in (
            "attestation_id",
            "candidate_id",
            "use_id",
            "custody_evidence_id",
            "attested_by",
        ):
            _required_text(str(getattr(self, label)), label)
        for label in (
            "intended_use_sha256",
            "comparator_sha256",
            "thresholds_sha256",
            "evidence_registry_sha256",
            "custody_manifest_sha256",
        ):
            _required_sha256(str(getattr(self, label)), label)
        _required_iso_datetime(self.attested_at, "attested_at")
        expected = content_hash(self._hash_payload())
        if self.attestation_sha256:
            _required_sha256(self.attestation_sha256, "attestation_sha256")
            if self.attestation_sha256 != expected:
                raise ValueError("Scope-freeze attestation failed integrity validation")
        else:
            object.__setattr__(self, "attestation_sha256", expected)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "attestation_sha256"
        }

    def to_dict(self) -> dict[str, Any]:
        data = self._hash_payload()
        data["attestation_sha256"] = self.attestation_sha256
        return data


@dataclass(frozen=True)
class IntendedUse:
    use_id: str
    evaluation_family: EvaluationFamily
    clinical_task: ClinicalTask
    target_population: str
    care_setting: str
    intended_user: str
    workflow_position: str
    index_test: str
    target_condition_or_outcome: str
    reference_standard: str
    clinical_decision: str
    action_if_positive: str
    action_if_negative: str
    unit_of_analysis: str
    time_horizon: str = ""
    outputs_absolute_risk: bool = False
    clinically_actionable_thresholds: tuple[float, ...] = ()
    anticipated_benefits: tuple[str, ...] = ()
    anticipated_harms: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    frozen_before_evaluation: bool = False

    def __post_init__(self) -> None:
        for label in (
            "use_id",
            "target_population",
            "care_setting",
            "intended_user",
            "workflow_position",
            "index_test",
            "target_condition_or_outcome",
            "reference_standard",
            "clinical_decision",
            "action_if_positive",
            "action_if_negative",
            "unit_of_analysis",
        ):
            _required_text(str(getattr(self, label)), label)
        if (
            self.evaluation_family == EvaluationFamily.PREDICTION_MODEL
            and not self.time_horizon.strip()
        ):
            raise ValueError("Prediction-model intended use requires a time_horizon")
        if (
            self.evaluation_family == EvaluationFamily.PREDICTION_MODEL
            and not self.outputs_absolute_risk
        ):
            raise ValueError(
                "Prediction-model clinical utility requires an absolute-risk output"
            )
        if any(
            not 0.0 < float(threshold) < 1.0
            for threshold in self.clinically_actionable_thresholds
        ):
            raise ValueError(
                "Clinically actionable probability thresholds must lie in (0, 1)"
            )
        if tuple(sorted(set(self.clinically_actionable_thresholds))) != tuple(
            self.clinically_actionable_thresholds
        ):
            raise ValueError(
                "Clinically actionable thresholds must be unique and sorted"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evaluation_family"] = self.evaluation_family.value
        data["clinical_task"] = self.clinical_task.value
        return _json_metadata(data)


@dataclass(frozen=True)
class ComparatorDefinition:
    comparator_id: str
    description: str
    current_standard_of_care: bool = False
    clinically_implemented: bool = False
    frozen_before_comparison: bool = False
    evaluated_head_to_head: bool = False
    evaluated_in_same_participants: bool = False
    includes_default_strategies: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.comparator_id, "comparator_id")
        _required_text(self.description, "comparator description")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticalValidityEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    measurement_procedure_frozen: bool = False
    preanalytic_and_signal_requirements_prespecified: bool = False
    repeatability_supported: bool = False
    reproducibility_across_devices_or_software_supported: bool = False
    failure_and_abstention_rates_reported: bool = False
    quality_control_acceptance_criteria_met: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClinicalAssociationEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    analysis_prespecified: bool = False
    reference_standard_appropriate: bool = False
    reference_assessment_blinded_to_index_test: bool = False
    incorporation_bias_absent: bool = False
    temporal_alignment_appropriate: bool = False
    uncertainty_reported: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class InternalValidationEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    candidate_frozen_before_evaluation: bool = False
    participant_level_separation: bool = False
    target_population_represented: bool = False
    sample_size_and_event_count_adequate: bool = False
    missing_data_handled_appropriately: bool = False
    discrimination_or_accuracy_reported: bool = False
    threshold_performance_reported: bool = False
    uncertainty_reported: bool = False
    prespecified_acceptance_criteria_met: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExternalValidationEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    external_dataset_count: int = 0
    distinct_institution_count: int = 0
    independent_investigators: bool = False
    candidate_applied_unchanged: bool = False
    intended_use_matches: bool = False
    target_population_represented: bool = False
    sample_size_and_event_count_adequate: bool = False
    discrimination_or_accuracy_reported: bool = False
    threshold_performance_reported: bool = False
    uncertainty_reported: bool = False
    prespecified_acceptance_criteria_met: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.external_dataset_count < 0 or self.distinct_institution_count < 0:
            raise ValueError("External validation counts cannot be negative")


@dataclass(frozen=True)
class CalibrationEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    evaluated_in_external_validation: bool = False
    calibration_plot_reported: bool = False
    calibration_in_the_large_reported: bool = False
    calibration_slope_reported: bool = False
    calibration_across_relevant_range_reported: bool = False
    subgroup_calibration_reported: bool = False
    acceptance_criteria_prespecified: bool = False
    acceptance_criteria_met: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class IncrementalValueEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    comparator_id: str = ""
    comparison_design: str = ""
    head_to_head_same_participants: bool = False
    discrimination_or_accuracy_compared: bool = False
    calibration_compared_when_applicable: bool = False
    threshold_performance_compared: bool = False
    uncertainty_reported: bool = False
    minimum_clinically_meaningful_improvement_prespecified: bool = False
    prespecified_minimum_met: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionCurveEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    thresholds_evaluated: tuple[float, ...] = ()
    primary_threshold: Optional[float] = None
    thresholds_prespecified: bool = False
    thresholds_clinically_actionable: bool = False
    evaluated_in_external_validation: bool = False
    threshold_coverage_complete: bool = False
    candidate_above_comparator_across_range: bool = False
    candidate_above_default_strategies_across_range: bool = False
    no_material_harm_across_range: bool = False
    uncertainty_reported: bool = False
    primary_threshold_delta_ci_positive: bool = False
    minimum_net_benefit_margin: float = 0.0
    minimum_margin_prespecified: bool = False
    evidence_scope: str = "not_assessed"
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not 0.0 < float(threshold) < 1.0
            for threshold in self.thresholds_evaluated
        ):
            raise ValueError("Decision-curve thresholds must lie in (0, 1)")
        if tuple(sorted(set(self.thresholds_evaluated))) != tuple(
            self.thresholds_evaluated
        ):
            raise ValueError("Decision-curve thresholds must be unique and sorted")
        if self.primary_threshold is not None and not 0.0 < float(
            self.primary_threshold
        ) < 1.0:
            raise ValueError("Decision-curve primary threshold must lie in (0, 1)")
        if self.minimum_net_benefit_margin < 0.0:
            raise ValueError("Decision-curve minimum net-benefit margin cannot be negative")
        if self.evidence_scope not in {
            "external_paired_prespecified_metadata",
            "not_assessed",
        }:
            raise ValueError(
                "Clinical-utility decision curves cannot use internal development artifacts"
            )


@dataclass(frozen=True)
class ReclassificationEvidence:
    performed: bool = False
    method: str = "not_used"
    clinically_defined_categories: bool = False
    categories_prespecified: bool = False
    event_and_nonevent_reclassification_reported_separately: bool = False
    uncertainty_reported: bool = False
    changes_clinical_management: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeasibilityActionabilityEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    result_changes_a_clinical_decision: bool = False
    resulting_action_supported_by_evidence: bool = False
    workflow_and_turnaround_feasible: bool = False
    failure_and_abstention_workflow_defined: bool = False
    downstream_harms_assessed: bool = False
    implementation_costs_and_resources_assessed: bool = False
    stakeholder_acceptability_assessed: bool = False
    scalability_assessed: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubgroupGeneralizabilityEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    subgroups_prespecified: bool = False
    clinically_relevant_subgroups_covered: bool = False
    subgroup_sample_sizes_reported: bool = False
    discrimination_or_accuracy_by_subgroup_reported: bool = False
    calibration_by_subgroup_reported_when_applicable: bool = False
    threshold_performance_by_subgroup_reported: bool = False
    device_site_and_era_heterogeneity_assessed: bool = False
    no_unacceptable_performance_or_harm_disparity: bool = False
    equity_implications_assessed: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportingBiasEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    reporting_guideline: str = ""
    reporting_checklist_complete: bool = False
    risk_of_bias_tool: str = ""
    low_risk_of_bias: bool = False
    low_concern_for_applicability: bool = False
    protocol_or_analysis_plan_public: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProspectiveImpactEvidence:
    status: EvidenceStatus = EvidenceStatus.NOT_ASSESSED
    study_design: str = ""
    prospectively_registered: bool = False
    protocol_and_primary_outcome_prespecified: bool = False
    candidate_frozen_before_enrolment: bool = False
    concurrent_standard_care_comparator: bool = False
    evaluated_in_intended_workflow: bool = False
    clinically_meaningful_primary_outcome: bool = False
    primary_outcome_type: str = ""
    primary_outcome_improved: bool = False
    uncertainty_reported: bool = False
    harms_and_unintended_consequences_acceptable: bool = False
    implementation_fidelity_reported: bool = False
    sample_size_adequate: bool = False
    low_risk_of_bias: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClinicalUtilityDossier:
    assessment_id: str
    candidate_id: str
    framework_scope: FrameworkScope
    intended_use: IntendedUse
    comparator: ComparatorDefinition
    scope_freeze_attestation: Optional[ScopeFreezeAttestation] = None
    aha_metric_records: tuple[AhaMetricRecord, ...] = ()
    analytical_validity: AnalyticalValidityEvidence = field(
        default_factory=AnalyticalValidityEvidence
    )
    clinical_association: ClinicalAssociationEvidence = field(
        default_factory=ClinicalAssociationEvidence
    )
    internal_validation: InternalValidationEvidence = field(
        default_factory=InternalValidationEvidence
    )
    external_validation: ExternalValidationEvidence = field(
        default_factory=ExternalValidationEvidence
    )
    calibration: CalibrationEvidence = field(default_factory=CalibrationEvidence)
    incremental_value: IncrementalValueEvidence = field(
        default_factory=IncrementalValueEvidence
    )
    decision_curve: DecisionCurveEvidence = field(
        default_factory=DecisionCurveEvidence
    )
    reclassification: ReclassificationEvidence = field(
        default_factory=ReclassificationEvidence
    )
    feasibility_actionability: FeasibilityActionabilityEvidence = field(
        default_factory=FeasibilityActionabilityEvidence
    )
    subgroup_generalizability: SubgroupGeneralizabilityEvidence = field(
        default_factory=SubgroupGeneralizabilityEvidence
    )
    reporting_and_bias: ReportingBiasEvidence = field(
        default_factory=ReportingBiasEvidence
    )
    prospective_impact: ProspectiveImpactEvidence = field(
        default_factory=ProspectiveImpactEvidence
    )

    def __post_init__(self) -> None:
        _required_text(self.assessment_id, "assessment_id")
        _required_text(self.candidate_id, "candidate_id")
        if (
            self.framework_scope == FrameworkScope.AHA_INCIDENT_CVD_RISK
            and self.intended_use.evaluation_family
            != EvaluationFamily.PREDICTION_MODEL
        ):
            raise ValueError(
                "Direct AHA incident-CVD scope requires a prediction-model "
                "intended use"
            )
        if (
            self.framework_scope == FrameworkScope.AHA_INCIDENT_CVD_RISK
            and self.intended_use.clinical_task
            not in {ClinicalTask.PROGNOSIS, ClinicalTask.RISK_STRATIFICATION}
        ):
            raise ValueError(
                "Direct AHA incident-CVD scope requires prognosis or risk "
                "stratification"
            )
        if (
            self.framework_scope == FrameworkScope.DIAGNOSTIC_ADAPTATION
            and self.intended_use.evaluation_family
            != EvaluationFamily.DIAGNOSTIC_ACCURACY
        ):
            raise ValueError(
                "Diagnostic adaptation requires a diagnostic-accuracy intended use"
            )
        metric_ids = tuple(item.metric_id for item in self.aha_metric_records)
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("AHA metric records must contain unique metric IDs")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["framework_scope"] = self.framework_scope.value
        data["intended_use"] = self.intended_use.to_dict()
        data["comparator"] = self.comparator.to_dict()
        data["scope_freeze_attestation"] = (
            self.scope_freeze_attestation.to_dict()
            if self.scope_freeze_attestation is not None
            else None
        )
        data["aha_metric_records"] = [
            item.to_dict() for item in self.aha_metric_records
        ]
        for key in (
            "analytical_validity",
            "clinical_association",
            "internal_validation",
            "external_validation",
            "calibration",
            "incremental_value",
            "decision_curve",
            "feasibility_actionability",
            "subgroup_generalizability",
            "reporting_and_bias",
            "prospective_impact",
        ):
            section = data[key]
            status = section.get("status")
            if isinstance(status, EvidenceStatus):
                section["status"] = status.value
        for key in (
            "comparator",
            "analytical_validity",
            "clinical_association",
            "internal_validation",
            "external_validation",
            "calibration",
            "incremental_value",
            "decision_curve",
            "reclassification",
            "feasibility_actionability",
            "subgroup_generalizability",
            "reporting_and_bias",
            "prospective_impact",
        ):
            section = data[key]
            if "evidence_ids" in section:
                section["evidence_ids"] = list(section["evidence_ids"])
        return _json_metadata(data)


@dataclass(frozen=True)
class UtilityGateResult:
    gate_id: str
    domain: str
    passed: bool
    missing_requirements: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    rationale: str

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status
        return _json_metadata(data)


@dataclass(frozen=True)
class ClinicalUtilityAssessment:
    assessment_id: str
    candidate_id: str
    claim_level: ClaimLevel
    gates: tuple[UtilityGateResult, ...]
    dossier: ClinicalUtilityDossier
    evidence_registry: EvidenceRegistry
    resolved_evidence: tuple[EvidenceRecord, ...]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def clinical_utility_demonstrated(self) -> bool:
        return self.decision_process_utility_demonstrated

    @property
    def decision_process_utility_demonstrated(self) -> bool:
        return CLAIM_LEVEL_ORDER.index(self.claim_level) >= CLAIM_LEVEL_ORDER.index(
            ClaimLevel.DECISION_PROCESS_UTILITY_DEMONSTRATED
        )

    @property
    def patient_benefit_demonstrated(self) -> bool:
        return self.claim_level == ClaimLevel.PATIENT_BENEFIT_DEMONSTRATED

    @property
    def claim_text_de(self) -> str:
        return CLAIM_TEXT_DE[self.claim_level]

    @property
    def dossier_hash(self) -> str:
        return content_hash(self.dossier.to_dict())

    def gate(self, gate_id: str) -> UtilityGateResult:
        for item in self.gates:
            if item.gate_id == gate_id:
                return item
        raise KeyError(gate_id)

    def to_dict(self) -> dict[str, Any]:
        process_required = set(
            CLAIM_REQUIREMENTS[
                ClaimLevel.DECISION_PROCESS_UTILITY_DEMONSTRATED
            ]
        )
        patient_required = set(
            CLAIM_REQUIREMENTS[ClaimLevel.PATIENT_BENEFIT_DEMONSTRATED]
        )
        process_blockers = [
            gate.gate_id
            for gate in self.gates
            if gate.gate_id in process_required and not gate.passed
        ]
        patient_blockers = [
            gate.gate_id
            for gate in self.gates
            if gate.gate_id in patient_required and not gate.passed
        ]
        current_index = CLAIM_LEVEL_ORDER.index(self.claim_level)
        next_level = (
            CLAIM_LEVEL_ORDER[current_index + 1]
            if current_index + 1 < len(CLAIM_LEVEL_ORDER)
            else None
        )
        scope_attested = self.gate("scope_freeze_attested").passed
        aha_complete = self.gate("aha_2026_metric_completeness").passed
        aha_direct = (
            self.dossier.framework_scope
            == FrameworkScope.AHA_INCIDENT_CVD_RISK
        )
        return _json_metadata({
            "schema_version": CLINICAL_UTILITY_SCHEMA_VERSION,
            "assessment_id": self.assessment_id,
            "candidate_id": self.candidate_id,
            "created_at": self.created_at,
            "claim_level": self.claim_level.value,
            "claim_rank": current_index,
            "claim_text_de": self.claim_text_de,
            "clinical_utility_demonstrated": self.clinical_utility_demonstrated,
            "clinical_utility_claim_allowed": self.clinical_utility_demonstrated,
            "decision_process_utility_demonstrated": (
                self.decision_process_utility_demonstrated
            ),
            "patient_benefit_demonstrated": self.patient_benefit_demonstrated,
            "patient_benefit_claim_allowed": self.patient_benefit_demonstrated,
            "clinical_use_authorized": False,
            "regulatory_authorization_assessed": False,
            "assessment_scope": "verified_declarative_evidence_metadata_only",
            "framework_scope": self.dossier.framework_scope.value,
            "aha_2026_direct_scope": aha_direct,
            "aha_2026_metric_completeness": aha_complete if aha_direct else None,
            "aha_2026_framework_alignment_supported": bool(
                aha_direct
                and aha_complete
                and CLAIM_LEVEL_ORDER.index(self.claim_level)
                >= CLAIM_LEVEL_ORDER.index(ClaimLevel.POTENTIAL_CLINICAL_UTILITY)
            ),
            # An AHA scientific statement is a methodological framework, not
            # a certification or regulatory conformity-assessment scheme.
            "aha_2026_direct_compliance_claim_allowed": False,
            "locked_test_access_status": (
                "attested_unread" if scope_attested else "not_attested"
            ),
            "locked_test_data_read": False if scope_attested else None,
            "locked_test_outcomes_read": False if scope_attested else None,
            "dossier_hash": self.dossier_hash,
            "evidence_registry_id": self.evidence_registry.registry_id,
            "evidence_registry_sha256": self.evidence_registry.registry_sha256,
            "evidence_registry": self.evidence_registry.to_dict(),
            "next_claim_level": next_level.value if next_level else None,
            "blocking_gates_for_demonstrated_utility": process_blockers,
            "blocking_gates_for_patient_benefit": patient_blockers,
            "gates": [gate.to_dict() for gate in self.gates],
            "evidence_dossier": self.dossier.to_dict(),
            "resolved_evidence_records": [
                record.to_dict() for record in self.resolved_evidence
            ],
            "authoritative_sources": [
                source.to_dict() for source in AUTHORITATIVE_SOURCES
            ],
            "limitations": [
                "This assessment does not itself authorize clinical use.",
                "Reporting-guideline compliance is not equivalent to low risk of bias.",
                "Internal or locked-test validation is not external validation.",
                "Decision-curve net benefit supports potential utility but does not replace a prospective comparative impact study.",
                "Direct AHA-2026 compliance is assessed only for incident-CVD risk prediction; diagnostic assessments are explicitly adaptations.",
                "The evaluator verifies signed evidence metadata but never opens the referenced aggregate artifacts.",
            ],
        })


def _gate(
    gate_id: str,
    domain: str,
    requirements: Iterable[tuple[str, bool]],
    *,
    evidence_ids: tuple[str, ...] = (),
    source_ids: tuple[str, ...] = (),
    success_rationale: str,
) -> UtilityGateResult:
    missing = tuple(name for name, satisfied in requirements if not satisfied)
    return UtilityGateResult(
        gate_id=gate_id,
        domain=domain,
        passed=not missing,
        missing_requirements=missing,
        evidence_ids=tuple(str(item) for item in evidence_ids),
        source_ids=source_ids,
        rationale=(
            success_rationale
            if not missing
            else "Missing or failed: " + ", ".join(missing)
        ),
    )


def _expected_reporting_guideline(family: EvaluationFamily) -> str:
    return (
        "STARD-AI"
        if family == EvaluationFamily.DIAGNOSTIC_ACCURACY
        else "TRIPOD+AI"
    )


def _normalized_framework_name(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _thresholds_match(
    first: tuple[float, ...],
    second: tuple[float, ...],
    *,
    tolerance: float = 1e-9,
) -> bool:
    return len(first) == len(second) and all(
        abs(float(left) - float(right)) <= tolerance
        for left, right in zip(first, second)
    )


def _referenced_evidence_ids(dossier: ClinicalUtilityDossier) -> tuple[str, ...]:
    values: list[str] = []
    if dossier.scope_freeze_attestation is not None:
        values.append(dossier.scope_freeze_attestation.custody_evidence_id)
    values.extend(dossier.comparator.evidence_ids)
    for section_name in (
        "analytical_validity",
        "clinical_association",
        "internal_validation",
        "external_validation",
        "calibration",
        "incremental_value",
        "decision_curve",
        "reclassification",
        "feasibility_actionability",
        "subgroup_generalizability",
        "reporting_and_bias",
        "prospective_impact",
    ):
        values.extend(getattr(dossier, section_name).evidence_ids)
    for metric in dossier.aha_metric_records:
        values.extend(metric.evidence_ids)
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def _resolve_evidence(
    dossier: ClinicalUtilityDossier,
    registry: EvidenceRegistry,
) -> tuple[EvidenceRecord, ...]:
    resolved: list[EvidenceRecord] = []
    missing: list[str] = []
    for evidence_id in _referenced_evidence_ids(dossier):
        try:
            resolved.append(registry.resolve(evidence_id))
        except KeyError:
            missing.append(evidence_id)
    if missing:
        raise ValueError(
            "Evidence registry cannot resolve referenced evidence IDs: "
            + ", ".join(sorted(missing))
        )
    return tuple(resolved)


def _evidence_supports(
    evidence_ids: tuple[str, ...],
    *,
    resolved_by_id: dict[str, EvidenceRecord],
    domain: str,
    registry: EvidenceRegistry,
    external: bool = False,
) -> bool:
    if not _has_traceable_ids(evidence_ids):
        return False
    records = [resolved_by_id.get(item) for item in evidence_ids]
    if any(record is None for record in records):
        return False
    typed_records = [record for record in records if record is not None]
    if any(domain not in record.domains for record in typed_records):
        return False
    if external:
        return any(
            record.split in {"external_validation", "prospective_impact"}
            and record.site_id != registry.development_site_id
            and record.independent_from_development
            for record in typed_records
        )
    return True


def _scope_freeze_requirements(
    dossier: ClinicalUtilityDossier,
    registry: EvidenceRegistry,
    resolved_by_id: dict[str, EvidenceRecord],
) -> tuple[tuple[str, bool], ...]:
    attestation = dossier.scope_freeze_attestation
    if attestation is None:
        return (("scope-freeze attestation present", False),)
    custody = resolved_by_id.get(attestation.custody_evidence_id)
    try:
        registry_created_at = datetime.fromisoformat(
            registry.created_at.replace("Z", "+00:00")
        )
        attested_at = datetime.fromisoformat(
            attestation.attested_at.replace("Z", "+00:00")
        )
        chronology_valid = registry_created_at <= attested_at
    except ValueError:
        chronology_valid = False
    return (
        ("scope-freeze attestation present", True),
        ("candidate identity bound", attestation.candidate_id == dossier.candidate_id),
        ("intended-use identity bound", attestation.use_id == dossier.intended_use.use_id),
        (
            "intended-use content hash bound",
            attestation.intended_use_sha256
            == content_hash(dossier.intended_use.to_dict()),
        ),
        (
            "standard-care comparator hash bound",
            attestation.comparator_sha256
            == content_hash(dossier.comparator.to_dict()),
        ),
        (
            "actionable thresholds hash bound",
            attestation.thresholds_sha256
            == content_hash(list(dossier.intended_use.clinically_actionable_thresholds)),
        ),
        (
            "evidence registry hash bound",
            attestation.evidence_registry_sha256 == registry.registry_sha256,
        ),
        (
            "custody evidence resolved as governance metadata",
            custody is not None
            and "governance" in custody.domains
            and custody.split == "governance"
            and custody.evidence_kind == "scope_freeze_custody_attestation",
        ),
        (
            "custody manifest digest bound",
            custody is not None
            and custody.manifest_sha256 == attestation.custody_manifest_sha256,
        ),
        (
            "attester identity bound to custody evidence assessor",
            custody is not None
            and custody.assessor_id == attestation.attested_by,
        ),
        (
            "freeze attestation is separate from registry creation",
            attestation.attested_by != registry.created_by,
        ),
        (
            "registry predates or coincides with freeze attestation",
            chronology_valid,
        ),
        ("scope frozen before evaluation", attestation.frozen_before_evaluation),
        (
            "locked-test data attested unread by custodian",
            attestation.locked_test_data_attested_unread,
        ),
        (
            "locked-test outcomes attested unread by custodian",
            attestation.locked_test_outcomes_attested_unread,
        ),
    )


def _aha_metric_requirements(
    dossier: ClinicalUtilityDossier,
    *,
    resolved_by_id: dict[str, EvidenceRecord],
    registry: EvidenceRegistry,
) -> tuple[tuple[str, bool], ...]:
    if dossier.framework_scope != FrameworkScope.AHA_INCIDENT_CVD_RISK:
        return (("AHA metric gate explicitly not applicable to diagnostic adaptation", True),)
    records = {item.metric_id: item for item in dossier.aha_metric_records}
    requirements: list[tuple[str, bool]] = []
    for metric_id in AHA_REQUIRED_METRICS:
        metric = records.get(metric_id)
        complete = bool(
            metric is not None
            and _is_supported(metric.status)
            and metric.prespecified
            and _evidence_supports(
                metric.evidence_ids,
                resolved_by_id=resolved_by_id,
                domain="aha_metric",
                registry=registry,
                external=metric.external_validation,
            )
        )
        if metric_id in _AHA_UNCERTAINTY_METRICS:
            complete = bool(
                complete
                and metric is not None
                and metric.uncertainty_reported
                and metric.confidence_interval_low is not None
                and metric.confidence_interval_high is not None
            )
        if metric_id in {
            AhaMetricId.INCREMENTAL_DELTA_C_STATISTIC,
            AhaMetricId.INCREMENTAL_DELTA_CALIBRATION,
            AhaMetricId.CATEGORICAL_NRI,
            AhaMetricId.DECISION_CURVE,
        }:
            complete = bool(
                complete
                and metric is not None
                and metric.comparator_id == dossier.comparator.comparator_id
            )
        if metric_id in {
            AhaMetricId.EXTERNAL_VALIDATION,
            AhaMetricId.CALIBRATION_IN_THE_LARGE,
            AhaMetricId.CALIBRATION_SLOPE,
            AhaMetricId.CALIBRATION_SOCIODEMOGRAPHIC_SUBGROUPS,
            AhaMetricId.INCREMENTAL_DELTA_C_STATISTIC,
            AhaMetricId.INCREMENTAL_DELTA_CALIBRATION,
            AhaMetricId.CATEGORICAL_NRI,
            AhaMetricId.DECISION_CURVE,
        }:
            complete = bool(complete and metric is not None and metric.external_validation)
        requirements.append((f"AHA metric complete: {metric_id.value}", complete))
    return tuple(requirements)


def _build_gates(
    dossier: ClinicalUtilityDossier,
    registry: EvidenceRegistry,
    resolved_evidence: tuple[EvidenceRecord, ...],
) -> tuple[UtilityGateResult, ...]:
    use = dossier.intended_use
    comparator = dossier.comparator
    analytical = dossier.analytical_validity
    association = dossier.clinical_association
    internal = dossier.internal_validation
    external = dossier.external_validation
    calibration = dossier.calibration
    incremental = dossier.incremental_value
    decision = dossier.decision_curve
    reclassification = dossier.reclassification
    feasibility = dossier.feasibility_actionability
    subgroup = dossier.subgroup_generalizability
    reporting = dossier.reporting_and_bias
    impact = dossier.prospective_impact
    resolved_by_id = {record.evidence_id: record for record in resolved_evidence}

    expected_guideline = _expected_reporting_guideline(use.evaluation_family)
    reporting_name_ok = _normalized_framework_name(
        reporting.reporting_guideline
    ) == _normalized_framework_name(expected_guideline)
    risk_tool_ok = bool(reporting.risk_of_bias_tool.strip())
    if use.evaluation_family == EvaluationFamily.PREDICTION_MODEL:
        risk_tool_ok = _normalized_framework_name(
            reporting.risk_of_bias_tool
        ) == _normalized_framework_name("PROBAST+AI")

    calibration_not_applicable = (
        not use.outputs_absolute_risk
        and _status_value(calibration.status)
        == EvidenceStatus.NOT_APPLICABLE.value
    )
    calibration_requirements = (
        (
            (
                "calibration explicitly not applicable to non-probability output",
                True,
            ),
            (
                "calibration rationale is traceable",
                _evidence_supports(
                    calibration.evidence_ids,
                    resolved_by_id=resolved_by_id,
                    domain="calibration",
                    registry=registry,
                ),
            ),
        )
        if calibration_not_applicable
        else (
            ("calibration evidence supported", _is_supported(calibration.status)),
            (
                "traceable calibration evidence",
                _evidence_supports(
                    calibration.evidence_ids,
                    resolved_by_id=resolved_by_id,
                    domain="calibration",
                    registry=registry,
                    external=True,
                ),
            ),
            (
                "calibration evaluated externally",
                calibration.evaluated_in_external_validation,
            ),
            ("calibration plot reported", calibration.calibration_plot_reported),
            (
                "calibration-in-the-large reported",
                calibration.calibration_in_the_large_reported,
            ),
            ("calibration slope reported", calibration.calibration_slope_reported),
            (
                "calibration across relevant range reported",
                calibration.calibration_across_relevant_range_reported,
            ),
            (
                "subgroup calibration reported",
                calibration.subgroup_calibration_reported,
            ),
            (
                "calibration acceptance criteria prespecified",
                calibration.acceptance_criteria_prespecified,
            ),
             (
                 "calibration acceptance criteria met",
                 calibration.acceptance_criteria_met,
             ),
         )
    )

    reclassification_method = _normalized_framework_name(reclassification.method)
    reclassification_not_used = not reclassification.performed
    reclassification_valid = reclassification_not_used or (
        reclassification_method in {"CATEGORICALNRI", "RECLASSIFICATIONTABLE"}
        and reclassification.clinically_defined_categories
        and reclassification.categories_prespecified
        and reclassification.event_and_nonevent_reclassification_reported_separately
        and reclassification.uncertainty_reported
        and reclassification.changes_clinical_management
    )

    accepted_impact_designs = {
        "randomized_individual",
        "cluster_randomized",
        "stepped_wedge_randomized",
        "prospective_controlled_impact",
    }
    accepted_impact_outcomes = {
        "patient_outcome",
        "clinical_management",
        "patient_safety",
    }
    randomized_impact_designs = {
        "randomized_individual",
        "cluster_randomized",
        "stepped_wedge_randomized",
    }
    framework_requirements = (
        (
            "direct AHA scope uses incident-CVD prediction family",
            use.evaluation_family == EvaluationFamily.PREDICTION_MODEL
            and use.clinical_task
            in {ClinicalTask.PROGNOSIS, ClinicalTask.RISK_STRATIFICATION},
        ),
    ) if dossier.framework_scope == FrameworkScope.AHA_INCIDENT_CVD_RISK else (
        (
            "diagnostic adaptation uses diagnostic-accuracy family",
            use.evaluation_family == EvaluationFamily.DIAGNOSTIC_ACCURACY,
        ),
        (
            "diagnostic adaptation is not represented as direct AHA compliance",
            True,
        ),
    )
    registered_external_sites = {
        record.site_id
        for evidence_id in external.evidence_ids
        for record in (resolved_by_id.get(evidence_id),)
        if record is not None
        and record.split == "external_validation"
        and record.site_id != registry.development_site_id
        and record.independent_from_development
    }

    return (
        _gate(
            "framework_scope_valid",
            "framework_scope",
            framework_requirements,
            source_ids=(
                ("aha_biomarker_utility_2026", "tripod_ai_2024", "probast_ai_2025")
                if dossier.framework_scope == FrameworkScope.AHA_INCIDENT_CVD_RISK
                else ("stard_ai_2025", "fda_nih_best_validation")
            ),
            success_rationale=(
                "The direct AHA incident-CVD framework is correctly scoped."
                if dossier.framework_scope == FrameworkScope.AHA_INCIDENT_CVD_RISK
                else "Diagnostic utility is explicitly assessed as an adaptation, not as direct AHA compliance."
            ),
        ),
        _gate(
            "scope_freeze_attested",
            "governance",
            _scope_freeze_requirements(dossier, registry, resolved_by_id),
            evidence_ids=(
                (dossier.scope_freeze_attestation.custody_evidence_id,)
                if dossier.scope_freeze_attestation is not None
                else ()
            ),
            source_ids=("fda_nih_best_validation",),
            success_rationale=(
                "A separately hash-bound custody attestation proves that the "
                "candidate, use, comparator, thresholds and registry were "
                "frozen while locked-test data and outcomes remained unread."
            ),
        ),
        _gate(
            "aha_2026_metric_completeness",
            "aha_predictive_utility",
            _aha_metric_requirements(
                dossier,
                resolved_by_id=resolved_by_id,
                registry=registry,
            ),
            evidence_ids=tuple(
                dict.fromkeys(
                    evidence_id
                    for metric in dossier.aha_metric_records
                    for evidence_id in metric.evidence_ids
                )
            ),
            source_ids=("aha_biomarker_utility_2026",),
            success_rationale=(
                "All machine-readable AHA-2026 Table-3 association, "
                "prediction, incremental-value and readiness metrics are "
                "complete for direct incident-CVD risk scope."
                if dossier.framework_scope == FrameworkScope.AHA_INCIDENT_CVD_RISK
                else "AHA Table-3 completeness is explicitly not claimed for the diagnostic adaptation."
            ),
        ),
        _gate(
            "intended_use_frozen",
            "context_of_use",
            (
                ("intended use frozen before evaluation", use.frozen_before_evaluation),
                (
                    "anticipated benefit specified",
                    _has_nonempty_text_items(use.anticipated_benefits),
                ),
                (
                    "anticipated harms specified",
                    _has_nonempty_text_items(use.anticipated_harms),
                ),
                (
                    "clinically actionable thresholds prespecified",
                    bool(use.clinically_actionable_thresholds),
                ),
            ),
            source_ids=(
                "aha_biomarker_utility_2026",
                "fda_nih_best_validation",
            ),
            success_rationale=(
                "Population, setting, user, workflow, decision, actions, "
                "reference and actionable thresholds were frozen a priori."
            ),
        ),
        _gate(
            "analytical_validity",
            "measurement",
            (
                ("analytical evidence supported", _is_supported(analytical.status)),
                (
                    "traceable analytical evidence",
                    _evidence_supports(
                        analytical.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="analytical_validity",
                        registry=registry,
                    ),
                ),
                (
                    "measurement procedure frozen",
                    analytical.measurement_procedure_frozen,
                ),
                (
                    "preanalytic and signal requirements prespecified",
                    analytical.preanalytic_and_signal_requirements_prespecified,
                ),
                ("repeatability supported", analytical.repeatability_supported),
                (
                    "cross-device/software reproducibility supported",
                    analytical.reproducibility_across_devices_or_software_supported,
                ),
                (
                    "failure and abstention rates reported",
                    analytical.failure_and_abstention_rates_reported,
                ),
                (
                    "quality-control acceptance criteria met",
                    analytical.quality_control_acceptance_criteria_met,
                ),
            ),
            evidence_ids=analytical.evidence_ids,
            source_ids=("fda_nih_best_validation",),
            success_rationale=(
                "The frozen measurement procedure is repeatable, "
                "reproducible and fit for the specified context of use."
            ),
        ),
        _gate(
            "clinical_association_validity",
            "clinical_validity",
            (
                ("clinical association supported", _is_supported(association.status)),
                (
                    "traceable clinical evidence",
                    _evidence_supports(
                        association.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="clinical_association",
                        registry=registry,
                    ),
                ),
                ("analysis prespecified", association.analysis_prespecified),
                (
                    "reference standard appropriate",
                    association.reference_standard_appropriate,
                ),
                (
                    "reference assessment blinded",
                    association.reference_assessment_blinded_to_index_test,
                ),
                ("incorporation bias absent", association.incorporation_bias_absent),
                (
                    "temporal alignment appropriate",
                    association.temporal_alignment_appropriate,
                ),
                ("uncertainty reported", association.uncertainty_reported),
            ),
            evidence_ids=association.evidence_ids,
            source_ids=(
                "aha_biomarker_utility_2026",
                "fda_nih_best_validation",
                "stard_ai_2025",
            ),
            success_rationale=(
                "Association was evaluated against an appropriate, blinded "
                "reference without incorporation or temporal bias."
            ),
        ),
        _gate(
            "internal_validation",
            "predictive_or_diagnostic_performance",
            (
                ("internal validation supported", _is_supported(internal.status)),
                (
                    "traceable internal-validation evidence",
                    _evidence_supports(
                        internal.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="internal_validation",
                        registry=registry,
                    ),
                ),
                (
                    "candidate frozen before evaluation",
                    internal.candidate_frozen_before_evaluation,
                ),
                (
                    "participant-level separation",
                    internal.participant_level_separation,
                ),
                (
                    "target population represented",
                    internal.target_population_represented,
                ),
                (
                    "sample size and event count adequate",
                    internal.sample_size_and_event_count_adequate,
                ),
                (
                    "missing data handled appropriately",
                    internal.missing_data_handled_appropriately,
                ),
                (
                    "discrimination or accuracy reported",
                    internal.discrimination_or_accuracy_reported,
                ),
                (
                    "threshold performance reported",
                    internal.threshold_performance_reported,
                ),
                ("uncertainty reported", internal.uncertainty_reported),
                (
                    "prespecified acceptance criteria met",
                    internal.prespecified_acceptance_criteria_met,
                ),
            ),
            evidence_ids=internal.evidence_ids,
            source_ids=(
                "aha_biomarker_utility_2026",
                "tripod_ai_2024",
                "probast_ai_2025",
                "stard_ai_2025",
            ),
            success_rationale=(
                "Frozen performance met prespecified criteria on unseen "
                "participants from the development data source."
            ),
        ),
        _gate(
            "external_validation",
            "generalizability",
            (
                ("external validation supported", _is_supported(external.status)),
                (
                    "traceable external-validation evidence",
                    _evidence_supports(
                        external.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="external_validation",
                        registry=registry,
                        external=True,
                    ),
                ),
                ("at least one external dataset", external.external_dataset_count >= 1),
                (
                    "at least one distinct institution",
                    external.distinct_institution_count >= 1,
                ),
                (
                    "declared external institutions represented in evidence registry",
                    len(registered_external_sites)
                    >= external.distinct_institution_count,
                ),
                ("independent investigators", external.independent_investigators),
                ("candidate applied unchanged", external.candidate_applied_unchanged),
                ("intended use matches", external.intended_use_matches),
                (
                    "target population represented",
                    external.target_population_represented,
                ),
                (
                    "sample size and event count adequate",
                    external.sample_size_and_event_count_adequate,
                ),
                (
                    "discrimination or accuracy reported",
                    external.discrimination_or_accuracy_reported,
                ),
                (
                    "threshold performance reported",
                    external.threshold_performance_reported,
                ),
                ("uncertainty reported", external.uncertainty_reported),
                (
                    "prespecified acceptance criteria met",
                    external.prespecified_acceptance_criteria_met,
                ),
            ),
            evidence_ids=external.evidence_ids,
            source_ids=(
                "aha_biomarker_utility_2026",
                "probast_ai_2025",
                "stard_ai_2025",
            ),
            success_rationale=(
                "The unchanged candidate met prespecified criteria in an "
                "independent external institution and intended-use population."
            ),
        ),
        _gate(
            "calibration",
            "predictive_or_diagnostic_performance",
            calibration_requirements,
            evidence_ids=calibration.evidence_ids,
            source_ids=("aha_biomarker_utility_2026",),
            success_rationale=(
                "External calibration and subgroup calibration met "
                "prespecified context-specific acceptance criteria."
                if not calibration_not_applicable
                else "Calibration is explicitly not applicable to a non-probability output."
            ),
        ),
        _gate(
            "reporting_and_risk_of_bias",
            "methodological_quality",
            (
                ("reporting evidence supported", _is_supported(reporting.status)),
                (
                    "traceable reporting and bias evidence",
                    _evidence_supports(
                        reporting.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="reporting_and_bias",
                        registry=registry,
                    ),
                ),
                (
                    f"{expected_guideline} reporting checklist complete",
                    reporting_name_ok and reporting.reporting_checklist_complete,
                ),
                ("design-appropriate risk-of-bias tool", risk_tool_ok),
                ("low risk of bias", reporting.low_risk_of_bias),
                (
                    "low concern for applicability",
                    reporting.low_concern_for_applicability,
                ),
                (
                    "protocol or analysis plan public",
                    reporting.protocol_or_analysis_plan_public,
                ),
            ),
            evidence_ids=reporting.evidence_ids,
            source_ids=(
                ("tripod_ai_2024", "probast_ai_2025")
                if use.evaluation_family == EvaluationFamily.PREDICTION_MODEL
                else ("stard_ai_2025",)
            ),
            success_rationale=(
                "Reporting is complete and a separate design-appropriate "
                "appraisal found low risk of bias and low applicability concern."
            ),
        ),
        _gate(
            "standard_care_comparator",
            "incremental_value",
            (
                ("current standard-of-care comparator", comparator.current_standard_of_care),
                (
                    "traceable comparator evidence",
                    _evidence_supports(
                        comparator.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="comparator",
                        registry=registry,
                    ),
                ),
                ("comparator clinically implemented", comparator.clinically_implemented),
                ("comparator frozen", comparator.frozen_before_comparison),
                ("head-to-head comparison", comparator.evaluated_head_to_head),
                (
                    "same-participant comparison",
                    comparator.evaluated_in_same_participants,
                ),
                (
                    "treat-all and treat-none strategies included",
                    comparator.includes_default_strategies,
                ),
            ),
            source_ids=("aha_biomarker_utility_2026",),
            success_rationale=(
                "Candidate was compared head to head with frozen current "
                "standard care and default decision strategies."
            ),
        ),
        _gate(
            "incremental_value",
            "incremental_value",
            (
                ("incremental evidence supported", _is_supported(incremental.status)),
                (
                    "traceable incremental-value evidence",
                    _evidence_supports(
                        incremental.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="incremental_value",
                        registry=registry,
                        external=True,
                    ),
                ),
                (
                    "comparator identity matches",
                    incremental.comparator_id == comparator.comparator_id,
                ),
                (
                    "comparison design specified",
                    incremental.comparison_design
                    in {"add_on", "replacement", "triage"},
                ),
                (
                    "head-to-head same-participant analysis",
                    incremental.head_to_head_same_participants,
                ),
                (
                    "discrimination or accuracy compared",
                    incremental.discrimination_or_accuracy_compared,
                ),
                (
                    "calibration compared when applicable",
                    incremental.calibration_compared_when_applicable
                    or not use.outputs_absolute_risk,
                ),
                (
                    "threshold performance compared",
                    incremental.threshold_performance_compared,
                ),
                ("uncertainty reported", incremental.uncertainty_reported),
                (
                    "minimum meaningful improvement prespecified",
                    incremental.minimum_clinically_meaningful_improvement_prespecified,
                ),
                (
                    "prespecified minimum improvement met",
                    incremental.prespecified_minimum_met,
                ),
            ),
            evidence_ids=incremental.evidence_ids,
            source_ids=("aha_biomarker_utility_2026",),
            success_rationale=(
                "Candidate added context-specific value beyond current "
                "standard care in a paired external comparison."
            ),
        ),
        _gate(
            "decision_curve_net_benefit",
            "clinical_utility",
            (
                ("decision-curve evidence supported", _is_supported(decision.status)),
                (
                    "absolute probability output available for decision analysis",
                    use.outputs_absolute_risk,
                ),
                (
                    "external paired prespecified evidence scope",
                    decision.evidence_scope
                    == "external_paired_prespecified_metadata",
                ),
                (
                    "traceable decision-curve evidence",
                    _evidence_supports(
                        decision.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="decision_curve",
                        registry=registry,
                        external=True,
                    ),
                ),
                ("thresholds prespecified", decision.thresholds_prespecified),
                (
                    "minimum net-benefit margin prespecified",
                    decision.minimum_margin_prespecified,
                ),
                (
                    "thresholds clinically actionable",
                    decision.thresholds_clinically_actionable,
                ),
                (
                    "thresholds match frozen intended use",
                    _thresholds_match(
                        decision.thresholds_evaluated,
                        use.clinically_actionable_thresholds,
                    ),
                ),
                (
                    "primary threshold belongs to frozen intended use",
                    decision.primary_threshold is not None
                    and any(
                        abs(
                            float(decision.primary_threshold)
                            - float(threshold)
                        )
                        <= 1e-9
                        for threshold in use.clinically_actionable_thresholds
                    ),
                ),
                (
                    "evaluated in external validation",
                    decision.evaluated_in_external_validation,
                ),
                (
                    "actionable threshold coverage complete",
                    decision.threshold_coverage_complete,
                ),
                (
                    "candidate above comparator across range",
                    decision.candidate_above_comparator_across_range,
                ),
                (
                    "candidate above default strategies across range",
                    decision.candidate_above_default_strategies_across_range,
                ),
                (
                    "no material harm across range",
                    decision.no_material_harm_across_range,
                ),
                ("net-benefit uncertainty reported", decision.uncertainty_reported),
                (
                    "primary-threshold net-benefit CI positive",
                    decision.primary_threshold_delta_ci_positive,
                ),
            ),
            evidence_ids=decision.evidence_ids,
            source_ids=("aha_biomarker_utility_2026",),
            success_rationale=(
                "External decision curves show positive incremental net "
                "benefit without material harm across the prespecified "
                "clinically actionable threshold range."
            ),
        ),
        _gate(
            "reclassification_interpretable_or_not_used",
            "incremental_value",
            (
                ("reclassification valid or not used", reclassification_valid),
                (
                    "traceable reclassification evidence when used",
                    reclassification_not_used
                    or _evidence_supports(
                        reclassification.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="reclassification",
                        registry=registry,
                        external=True,
                    ),
                ),
            ),
            evidence_ids=reclassification.evidence_ids,
            source_ids=("aha_biomarker_utility_2026",),
            success_rationale=(
                "Reclassification was not used, or categorical "
                "reclassification used prespecified clinically meaningful "
                "categories with separate event/non-event results and uncertainty."
            ),
        ),
        _gate(
            "feasibility_and_actionability",
            "clinical_utility",
            (
                ("feasibility evidence supported", _is_supported(feasibility.status)),
                (
                    "traceable feasibility evidence",
                    _evidence_supports(
                        feasibility.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="feasibility_actionability",
                        registry=registry,
                    ),
                ),
                (
                    "result changes a clinical decision",
                    feasibility.result_changes_a_clinical_decision,
                ),
                (
                    "resulting action supported by evidence",
                    feasibility.resulting_action_supported_by_evidence,
                ),
                (
                    "workflow and turnaround feasible",
                    feasibility.workflow_and_turnaround_feasible,
                ),
                (
                    "failure and abstention workflow defined",
                    feasibility.failure_and_abstention_workflow_defined,
                ),
                (
                    "downstream harms assessed",
                    feasibility.downstream_harms_assessed,
                ),
                (
                    "costs and resources assessed",
                    feasibility.implementation_costs_and_resources_assessed,
                ),
                (
                    "stakeholder acceptability assessed",
                    feasibility.stakeholder_acceptability_assessed,
                ),
                ("scalability assessed", feasibility.scalability_assessed),
            ),
            evidence_ids=feasibility.evidence_ids,
            source_ids=(
                "aha_biomarker_utility_2026",
                "fda_nih_best_validation",
            ),
            success_rationale=(
                "The result leads to an evidence-supported action and has a "
                "feasible, safe, acceptable, scalable and cost-assessed workflow."
            ),
        ),
        _gate(
            "subgroups_and_generalizability",
            "equity_and_generalizability",
            (
                ("subgroup evidence supported", _is_supported(subgroup.status)),
                (
                    "traceable subgroup evidence",
                    _evidence_supports(
                        subgroup.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="subgroup_generalizability",
                        registry=registry,
                        external=True,
                    ),
                ),
                ("subgroups prespecified", subgroup.subgroups_prespecified),
                (
                    "clinically relevant subgroups covered",
                    subgroup.clinically_relevant_subgroups_covered,
                ),
                (
                    "subgroup sample sizes reported",
                    subgroup.subgroup_sample_sizes_reported,
                ),
                (
                    "subgroup discrimination or accuracy reported",
                    subgroup.discrimination_or_accuracy_by_subgroup_reported,
                ),
                (
                    "subgroup calibration reported when applicable",
                    subgroup.calibration_by_subgroup_reported_when_applicable
                    or not use.outputs_absolute_risk,
                ),
                (
                    "subgroup threshold performance reported",
                    subgroup.threshold_performance_by_subgroup_reported,
                ),
                (
                    "device, site and era heterogeneity assessed",
                    subgroup.device_site_and_era_heterogeneity_assessed,
                ),
                (
                    "no unacceptable subgroup disparity",
                    subgroup.no_unacceptable_performance_or_harm_disparity,
                ),
                (
                    "equity implications assessed",
                    subgroup.equity_implications_assessed,
                ),
            ),
            evidence_ids=subgroup.evidence_ids,
            source_ids=(
                "aha_biomarker_utility_2026",
                "tripod_ai_2024",
                "probast_ai_2025",
                "stard_ai_2025",
            ),
            success_rationale=(
                "Prespecified subgroup, device, site, era and equity analyses "
                "support the intended-use population without unacceptable disparity."
            ),
        ),
        _gate(
            "prospective_decision_process_impact",
            "demonstrated_decision_process_utility",
            (
                ("prospective impact evidence supported", _is_supported(impact.status)),
                (
                    "traceable prospective-impact evidence",
                    _evidence_supports(
                        impact.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="prospective_impact",
                        registry=registry,
                        external=True,
                    ),
                ),
                (
                    "eligible prospective controlled design",
                    impact.study_design in accepted_impact_designs,
                ),
                ("study prospectively registered", impact.prospectively_registered),
                (
                    "protocol and primary outcome prespecified",
                    impact.protocol_and_primary_outcome_prespecified,
                ),
                (
                    "candidate frozen before enrolment",
                    impact.candidate_frozen_before_enrolment,
                ),
                (
                    "concurrent standard-care comparator",
                    impact.concurrent_standard_care_comparator,
                ),
                (
                    "evaluated in intended workflow",
                    impact.evaluated_in_intended_workflow,
                ),
                (
                    "clinically meaningful primary outcome",
                    impact.clinically_meaningful_primary_outcome,
                ),
                (
                    "eligible impact outcome type",
                    impact.primary_outcome_type in accepted_impact_outcomes,
                ),
                ("primary outcome improved", impact.primary_outcome_improved),
                ("uncertainty reported", impact.uncertainty_reported),
                (
                    "harms and unintended consequences acceptable",
                    impact.harms_and_unintended_consequences_acceptable,
                ),
                (
                    "implementation fidelity reported",
                    impact.implementation_fidelity_reported,
                ),
                ("sample size adequate", impact.sample_size_adequate),
                ("low risk of bias", impact.low_risk_of_bias),
            ),
            evidence_ids=impact.evidence_ids,
            source_ids=(
                "aha_biomarker_utility_2026",
                "fda_nih_best_validation",
            ),
            success_rationale=(
                "A registered, prespecified prospective controlled impact "
                "study of the frozen tool improved a clinically meaningful "
                "decision, process or safety outcome in the intended workflow "
                "without unacceptable harm."
            ),
        ),
        _gate(
            "prospective_patient_benefit",
            "demonstrated_patient_benefit",
            (
                ("prospective impact evidence supported", _is_supported(impact.status)),
                (
                    "traceable independent prospective-impact evidence",
                    _evidence_supports(
                        impact.evidence_ids,
                        resolved_by_id=resolved_by_id,
                        domain="prospective_impact",
                        registry=registry,
                        external=True,
                    ),
                ),
                (
                    "randomized impact design",
                    impact.study_design in randomized_impact_designs,
                ),
                ("study prospectively registered", impact.prospectively_registered),
                (
                    "protocol and primary outcome prespecified",
                    impact.protocol_and_primary_outcome_prespecified,
                ),
                (
                    "candidate frozen before enrolment",
                    impact.candidate_frozen_before_enrolment,
                ),
                (
                    "concurrent standard-care comparator",
                    impact.concurrent_standard_care_comparator,
                ),
                (
                    "evaluated in intended workflow",
                    impact.evaluated_in_intended_workflow,
                ),
                (
                    "patient-relevant primary outcome",
                    impact.primary_outcome_type
                    in {"patient_outcome", "patient_safety"}
                    and impact.clinically_meaningful_primary_outcome,
                ),
                ("patient-relevant primary outcome improved", impact.primary_outcome_improved),
                ("uncertainty reported", impact.uncertainty_reported),
                (
                    "harms and unintended consequences acceptable",
                    impact.harms_and_unintended_consequences_acceptable,
                ),
                (
                    "implementation fidelity reported",
                    impact.implementation_fidelity_reported,
                ),
                ("sample size adequate", impact.sample_size_adequate),
                ("low risk of bias", impact.low_risk_of_bias),
            ),
            evidence_ids=impact.evidence_ids,
            source_ids=(
                "aha_biomarker_utility_2026",
                "fda_nih_best_validation",
            ),
            success_rationale=(
                "A registered, prespecified randomized impact study of the "
                "frozen tool improved a patient-relevant outcome or safety "
                "endpoint in the intended workflow without unacceptable harm."
            ),
        ),
    )


CLAIM_REQUIREMENTS: dict[ClaimLevel, tuple[str, ...]] = {
    ClaimLevel.HYPOTHESIS_ONLY: ("framework_scope_valid",),
    ClaimLevel.ANALYTICAL_VALIDITY_SUPPORTED: (
        "framework_scope_valid",
        "scope_freeze_attested",
        "intended_use_frozen",
        "analytical_validity",
    ),
    ClaimLevel.CLINICAL_ASSOCIATION_SUPPORTED: (
        "framework_scope_valid",
        "scope_freeze_attested",
        "intended_use_frozen",
        "analytical_validity",
        "clinical_association_validity",
    ),
    ClaimLevel.INTERNAL_VALIDATION_SUPPORTED: (
        "framework_scope_valid",
        "scope_freeze_attested",
        "intended_use_frozen",
        "analytical_validity",
        "clinical_association_validity",
        "internal_validation",
    ),
    ClaimLevel.EXTERNAL_VALIDATION_SUPPORTED: (
        "framework_scope_valid",
        "scope_freeze_attested",
        "intended_use_frozen",
        "analytical_validity",
        "clinical_association_validity",
        "internal_validation",
        "external_validation",
        "calibration",
        "reporting_and_risk_of_bias",
    ),
    ClaimLevel.POTENTIAL_CLINICAL_UTILITY: (
        "framework_scope_valid",
        "scope_freeze_attested",
        "intended_use_frozen",
        "analytical_validity",
        "clinical_association_validity",
        "internal_validation",
        "external_validation",
        "calibration",
        "reporting_and_risk_of_bias",
        "standard_care_comparator",
        "incremental_value",
        "decision_curve_net_benefit",
        "reclassification_interpretable_or_not_used",
        "feasibility_and_actionability",
        "subgroups_and_generalizability",
        "aha_2026_metric_completeness",
    ),
    ClaimLevel.DECISION_PROCESS_UTILITY_DEMONSTRATED: (
        "framework_scope_valid",
        "scope_freeze_attested",
        "intended_use_frozen",
        "analytical_validity",
        "clinical_association_validity",
        "internal_validation",
        "external_validation",
        "calibration",
        "reporting_and_risk_of_bias",
        "standard_care_comparator",
        "incremental_value",
        "decision_curve_net_benefit",
        "reclassification_interpretable_or_not_used",
        "feasibility_and_actionability",
        "subgroups_and_generalizability",
        "aha_2026_metric_completeness",
        "prospective_decision_process_impact",
    ),
    ClaimLevel.PATIENT_BENEFIT_DEMONSTRATED: (
        "framework_scope_valid",
        "scope_freeze_attested",
        "intended_use_frozen",
        "analytical_validity",
        "clinical_association_validity",
        "internal_validation",
        "external_validation",
        "calibration",
        "reporting_and_risk_of_bias",
        "standard_care_comparator",
        "incremental_value",
        "decision_curve_net_benefit",
        "reclassification_interpretable_or_not_used",
        "feasibility_and_actionability",
        "subgroups_and_generalizability",
        "aha_2026_metric_completeness",
        "prospective_decision_process_impact",
        "prospective_patient_benefit",
    ),
}


def evaluate_clinical_utility(
    dossier: ClinicalUtilityDossier,
    evidence_registry: EvidenceRegistry,
) -> ClinicalUtilityAssessment:
    """Evaluate claim level from declarative evidence without reading data files.

    The evaluator has no path to MIMIC feature, label, or outcome artifacts. It
    can therefore be run while every locked-test partition remains sealed. The
    evidence IDs in the dossier are traceability labels, not files opened here.
    """

    resolved_evidence = _resolve_evidence(dossier, evidence_registry)
    gates = _build_gates(dossier, evidence_registry, resolved_evidence)
    by_id = {gate.gate_id: gate for gate in gates}
    claim_level = ClaimLevel.HYPOTHESIS_ONLY
    for candidate_level in CLAIM_LEVEL_ORDER[1:]:
        required = CLAIM_REQUIREMENTS[candidate_level]
        if all(by_id[gate_id].passed for gate_id in required):
            claim_level = candidate_level
        else:
            break
    return ClinicalUtilityAssessment(
        assessment_id=dossier.assessment_id,
        candidate_id=dossier.candidate_id,
        claim_level=claim_level,
        gates=gates,
        dossier=dossier,
        evidence_registry=evidence_registry,
        resolved_evidence=resolved_evidence,
    )


def _construct_dataclass(
    cls: type,
    payload: dict[str, Any],
    *,
    tuple_fields: tuple[str, ...] = (),
    enum_fields: Optional[dict[str, type[StrEnum]]] = None,
) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    transformed = dict(payload)
    for name in tuple_fields:
        if name in transformed:
            value = transformed[name]
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{cls.__name__}.{name} must be an array")
            transformed[name] = tuple(value)
    for name, enum_type in (enum_fields or {}).items():
        if name in transformed:
            transformed[name] = enum_type(transformed[name])
    return cls(**transformed)


def clinical_utility_dossier_from_dict(
    payload: dict[str, Any],
) -> ClinicalUtilityDossier:
    """Strictly decode a metadata-only v2 dossier."""

    if "evidence_dossier" in payload:
        payload = dict(payload["evidence_dossier"])
    allowed = {item.name for item in fields(ClinicalUtilityDossier)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown ClinicalUtilityDossier fields: {unknown}")
    required = {
        "assessment_id",
        "candidate_id",
        "framework_scope",
        "intended_use",
        "comparator",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"ClinicalUtilityDossier fields missing: {missing}")

    def status_section(name: str, cls: type) -> Any:
        raw = dict(payload.get(name, {}))
        return _construct_dataclass(
            cls,
            raw,
            tuple_fields=("evidence_ids",),
            enum_fields={"status": EvidenceStatus},
        )

    intended = _construct_dataclass(
        IntendedUse,
        dict(payload["intended_use"]),
        tuple_fields=(
            "clinically_actionable_thresholds",
            "anticipated_benefits",
            "anticipated_harms",
            "exclusions",
        ),
        enum_fields={
            "evaluation_family": EvaluationFamily,
            "clinical_task": ClinicalTask,
        },
    )
    comparator = _construct_dataclass(
        ComparatorDefinition,
        dict(payload["comparator"]),
        tuple_fields=("evidence_ids",),
    )
    freeze_payload = payload.get("scope_freeze_attestation")
    freeze = None
    if freeze_payload is not None:
        freeze_payload = dict(freeze_payload)
        if not str(freeze_payload.get("attestation_sha256", "")):
            raise ValueError(
                "Persisted scope-freeze attestation requires attestation_sha256"
            )
        freeze = _construct_dataclass(ScopeFreezeAttestation, freeze_payload)
    metric_records = tuple(
        _construct_dataclass(
            AhaMetricRecord,
            dict(item),
            tuple_fields=("evidence_ids",),
            enum_fields={
                "metric_id": AhaMetricId,
                "status": EvidenceStatus,
            },
        )
        for item in payload.get("aha_metric_records", ())
    )
    return ClinicalUtilityDossier(
        assessment_id=str(payload["assessment_id"]),
        candidate_id=str(payload["candidate_id"]),
        framework_scope=FrameworkScope(payload["framework_scope"]),
        intended_use=intended,
        comparator=comparator,
        scope_freeze_attestation=freeze,
        aha_metric_records=metric_records,
        analytical_validity=status_section(
            "analytical_validity", AnalyticalValidityEvidence
        ),
        clinical_association=status_section(
            "clinical_association", ClinicalAssociationEvidence
        ),
        internal_validation=status_section(
            "internal_validation", InternalValidationEvidence
        ),
        external_validation=status_section(
            "external_validation", ExternalValidationEvidence
        ),
        calibration=status_section("calibration", CalibrationEvidence),
        incremental_value=status_section(
            "incremental_value", IncrementalValueEvidence
        ),
        decision_curve=status_section("decision_curve", DecisionCurveEvidence),
        reclassification=_construct_dataclass(
            ReclassificationEvidence,
            dict(payload.get("reclassification", {})),
            tuple_fields=("evidence_ids",),
        ),
        feasibility_actionability=status_section(
            "feasibility_actionability", FeasibilityActionabilityEvidence
        ),
        subgroup_generalizability=status_section(
            "subgroup_generalizability", SubgroupGeneralizabilityEvidence
        ),
        reporting_and_bias=status_section(
            "reporting_and_bias", ReportingBiasEvidence
        ),
        prospective_impact=status_section(
            "prospective_impact", ProspectiveImpactEvidence
        ),
    )


def _read_metadata_json(path: Path | str, *, maximum_bytes: int = 10_000_000) -> dict:
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("Clinical-utility metadata inputs must be JSON files")
    size = source.stat().st_size
    if size > maximum_bytes:
        raise ValueError(
            f"Refusing non-metadata-sized clinical-utility input ({size} bytes)"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Clinical-utility JSON root must be an object")
    return payload


def assess_clinical_utility_from_files(
    dossier_path: Path | str,
    evidence_registry_path: Path | str,
    output_root: Path | str,
) -> dict[str, Path]:
    """Evaluate JSON metadata only; referenced evidence URIs are never opened."""

    dossier = clinical_utility_dossier_from_dict(
        _read_metadata_json(dossier_path)
    )
    registry = EvidenceRegistry.from_dict(
        _read_metadata_json(evidence_registry_path)
    )
    assessment = evaluate_clinical_utility(dossier, registry)
    return write_clinical_utility_assessment(assessment, output_root)


def assess_decision_curve(
    frame: pd.DataFrame,
    *,
    actionable_thresholds: tuple[float, ...],
    primary_threshold: float,
    evidence_ids: tuple[str, ...],
    thresholds_prespecified: bool,
    thresholds_clinically_actionable: bool,
    evaluated_in_external_validation: bool,
    minimum_net_benefit_margin: float = 0.0,
    minimum_margin_prespecified: bool = False,
    threshold_tolerance: float = 1e-9,
) -> DecisionCurveEvidence:
    """Assess paired net benefit over a prespecified actionable range.

    Required columns are ``threshold_probability``, ``candidate_net_benefit``,
    ``comparator_net_benefit``, ``treat_all_net_benefit``,
    ``treat_none_net_benefit`` and ``delta_net_benefit_ci_low``. The latter is
    the lower confidence bound for candidate minus comparator net benefit.
    This helper deliberately does not choose thresholds from the results.
    """

    required_columns = {
        "threshold_probability",
        "candidate_net_benefit",
        "comparator_net_benefit",
        "treat_all_net_benefit",
        "treat_none_net_benefit",
        "delta_net_benefit_ci_low",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise KeyError(
            "Decision-curve frame lacks required columns: "
            + ", ".join(missing_columns)
        )
    if not actionable_thresholds:
        raise ValueError("At least one actionable threshold is required")
    if any(not 0.0 < float(value) < 1.0 for value in actionable_thresholds):
        raise ValueError("Actionable thresholds must lie in (0, 1)")
    if tuple(sorted(set(actionable_thresholds))) != tuple(actionable_thresholds):
        raise ValueError("Actionable thresholds must be unique and sorted")
    if not 0.0 < float(primary_threshold) < 1.0:
        raise ValueError("primary_threshold must lie in (0, 1)")
    if not any(
        abs(float(primary_threshold) - float(value)) <= threshold_tolerance
        for value in actionable_thresholds
    ):
        raise ValueError("primary_threshold must be an actionable threshold")
    if minimum_net_benefit_margin < 0:
        raise ValueError("minimum_net_benefit_margin cannot be negative")

    numeric = frame.loc[:, sorted(required_columns)].copy()
    for column in required_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    if any(
        not math.isfinite(float(value))
        for value in numeric.to_numpy().ravel()
    ):
        raise ValueError("Decision-curve inputs must be finite numeric values")
    if numeric["threshold_probability"].duplicated().any():
        raise ValueError("Decision-curve thresholds must be unique")

    selected_rows: list[pd.Series] = []
    coverage_complete = True
    for threshold in actionable_thresholds:
        distance = (numeric["threshold_probability"] - float(threshold)).abs()
        if distance.empty or float(distance.min()) > threshold_tolerance:
            coverage_complete = False
            continue
        selected_rows.append(numeric.loc[distance.idxmin()])
    if selected_rows:
        selected = pd.DataFrame(selected_rows).reset_index(drop=True)
        comparator_delta = (
            selected["candidate_net_benefit"]
            - selected["comparator_net_benefit"]
        )
        default_maximum = selected[
            ["treat_all_net_benefit", "treat_none_net_benefit"]
        ].max(axis=1)
        default_delta = selected["candidate_net_benefit"] - default_maximum
        above_comparator = bool(
            (comparator_delta >= minimum_net_benefit_margin).all()
        )
        above_defaults = bool(
            (default_delta >= minimum_net_benefit_margin).all()
        )
        no_material_harm = bool(
            (comparator_delta >= 0.0).all() and (default_delta >= 0.0).all()
        )
    else:
        selected = pd.DataFrame()
        above_comparator = False
        above_defaults = False
        no_material_harm = False

    primary_distance = (
        numeric["threshold_probability"] - float(primary_threshold)
    ).abs()
    primary_present = bool(
        not primary_distance.empty
        and float(primary_distance.min()) <= threshold_tolerance
    )
    primary_ci_positive = bool(
        primary_present
        and float(
            numeric.loc[
                primary_distance.idxmin(),
                "delta_net_benefit_ci_low",
            ]
        )
        > 0.0
    )
    uncertainty_reported = bool(
        coverage_complete
        and len(selected) == len(actionable_thresholds)
        and selected["delta_net_benefit_ci_low"].notna().all()
    )
    supported = all(
        (
            thresholds_prespecified,
            thresholds_clinically_actionable,
            evaluated_in_external_validation,
            minimum_margin_prespecified,
            coverage_complete,
            above_comparator,
            above_defaults,
            no_material_harm,
            uncertainty_reported,
            primary_ci_positive,
        )
    )
    return DecisionCurveEvidence(
        status=(EvidenceStatus.SUPPORTED if supported else EvidenceStatus.FAILED),
        thresholds_evaluated=tuple(float(item) for item in actionable_thresholds),
        primary_threshold=float(primary_threshold),
        thresholds_prespecified=bool(thresholds_prespecified),
        thresholds_clinically_actionable=bool(thresholds_clinically_actionable),
        evaluated_in_external_validation=bool(evaluated_in_external_validation),
        threshold_coverage_complete=coverage_complete,
        candidate_above_comparator_across_range=above_comparator,
        candidate_above_default_strategies_across_range=above_defaults,
        no_material_harm_across_range=no_material_harm,
        uncertainty_reported=uncertainty_reported,
        primary_threshold_delta_ci_positive=primary_ci_positive,
        minimum_net_benefit_margin=float(minimum_net_benefit_margin),
        minimum_margin_prespecified=bool(minimum_margin_prespecified),
        evidence_scope="external_paired_prespecified_metadata",
        evidence_ids=tuple(str(item) for item in evidence_ids),
    )


def render_clinical_utility_markdown(
    assessment: ClinicalUtilityAssessment,
) -> str:
    data = assessment.to_dict()
    use = assessment.dossier.intended_use
    comparator = assessment.dossier.comparator
    lines = [
        "# Clinical-Utility-Assessment",
        "",
        f"- Assessment: `{assessment.assessment_id}`",
        f"- Kandidat: `{assessment.candidate_id}`",
        f"- Claim-Stufe: `{assessment.claim_level.value}`",
        f"- Aussage: {assessment.claim_text_de}",
        f"- Framework-Scope: `{assessment.dossier.framework_scope.value}`",
        f"- Locked-Test-Custody: `{data['locked_test_access_status']}`; das Modul liest ausschließlich verifizierte Evidenzmetadaten.",
        "- Klinische Anwendung zugelassen: nein; regulatorische Bewertung ist nicht Gegenstand dieses Assessments.",
        "",
        "## Intended use",
        "",
        f"- Population: {use.target_population}",
        f"- Setting / Nutzer: {use.care_setting} / {use.intended_user}",
        f"- Workflow: {use.workflow_position}",
        f"- Index-Test: {use.index_test}",
        f"- Ziel: {use.target_condition_or_outcome}",
        f"- Referenz: {use.reference_standard}",
        f"- Entscheidung: {use.clinical_decision}",
        f"- Standardvergleich: {comparator.description}",
        "",
        "## Harte Gates",
        "",
        "| Gate | Status | Fehlend/fehlgeschlagen |",
        "|---|---:|---|",
    ]
    for gate in assessment.gates:
        missing = "; ".join(gate.missing_requirements) or "-"
        lines.append(f"| `{gate.gate_id}` | {gate.status} | {missing} |")
    lines.extend(
        [
            "",
            "## Blocker bis zum Nutzennachweis",
            "",
        ]
    )
    blockers = data["blocking_gates_for_demonstrated_utility"]
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- Keine bis zum Entscheidungs-/Prozessnutzennachweis.")
    lines.extend(["", "## Blocker bis zum patientenrelevanten Nutzen", ""])
    patient_blockers = data["blocking_gates_for_patient_benefit"]
    if patient_blockers:
        lines.extend(f"- `{item}`" for item in patient_blockers)
    else:
        lines.append("- Keine; alle patientenrelevanten Impact-Gates sind erfüllt.")
    lines.extend(
        [
            "",
            "## Methodische Einordnung",
            "",
            "- AUC oder statistische Assoziation allein belegen keinen klinischen Nutzen.",
            "- Kalibrationsgrenzen und minimale Zusatzleistung werden für den konkreten Verwendungszweck vorab festgelegt; das Modul erfindet keine universellen Schwellen.",
            "- Reclassification ist optional. Wenn sie verwendet wird, akzeptiert das Gate nur präspezifizierte klinische Kategorien, getrennte Event-/Non-Event-Ergebnisse und Unsicherheit; continuous NRI trägt keinen Nutzennachweis.",
            "- Vorhandene interne Decision-Curve-Artefakte werden niemals automatisch übernommen. Utility-Evidenz muss extern, gepaart, präspezifiziert und registry-verifiziert sein.",
            "- Eine kontrollierte Impact-Studie kann Entscheidungs-/Prozessnutzen belegen; patientenrelevanter Nutzen benötigt zusätzlich ein randomisiertes Design mit verbessertem Patienten- oder Sicherheitsendpunkt.",
            "- Die AHA-2026-Metrikvollständigkeit gilt direkt nur für incident-CVD-Risikoprädiktion. Diagnostische ECG/VCG-Nutzung ist ausdrücklich eine Adaption.",
            "- TRIPOD+AI und STARD-AI sind Reporting-Standards und ersetzen keine Risk-of-Bias-Bewertung.",
            "",
            "## Autoritative Quellen",
            "",
        ]
    )
    for source in AUTHORITATIVE_SOURCES:
        lines.append(
            f"- [{source.title}]({source.url}) — {source.role}"
        )
    return "\n".join(lines) + "\n"


EXPECTED_UTILITY_GATE_IDS: tuple[str, ...] = (
    "framework_scope_valid",
    "scope_freeze_attested",
    "aha_2026_metric_completeness",
    "intended_use_frozen",
    "analytical_validity",
    "clinical_association_validity",
    "internal_validation",
    "external_validation",
    "calibration",
    "reporting_and_risk_of_bias",
    "standard_care_comparator",
    "incremental_value",
    "decision_curve_net_benefit",
    "reclassification_interpretable_or_not_used",
    "feasibility_and_actionability",
    "subgroups_and_generalizability",
    "prospective_decision_process_impact",
    "prospective_patient_benefit",
)


def clinical_utility_schema_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "resources"
        / "clinical_utility_assessment.schema.json"
    )


def validate_clinical_utility_payload(payload: dict[str, Any]) -> None:
    """Apply JSON-Schema and cross-field semantic integrity validation."""

    schema = json.loads(clinical_utility_schema_path().read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ValueError(f"Clinical-utility schema violation at {location}: {error.message}")
    _required_iso_datetime(str(payload["created_at"]), "created_at")

    dossier = clinical_utility_dossier_from_dict(
        dict(payload["evidence_dossier"])
    )
    registry = EvidenceRegistry.from_dict(dict(payload["evidence_registry"]))
    if payload["dossier_hash"] != content_hash(dossier.to_dict()):
        raise ValueError("Clinical-utility dossier_hash is inconsistent")
    if payload["evidence_registry_id"] != registry.registry_id:
        raise ValueError("Evidence registry identity is inconsistent")
    if payload["evidence_registry_sha256"] != registry.registry_sha256:
        raise ValueError("Evidence registry SHA-256 is inconsistent")

    resolved = tuple(
        EvidenceRecord.from_dict(dict(item))
        for item in payload["resolved_evidence_records"]
    )
    referenced_ids = set(_referenced_evidence_ids(dossier))
    resolved_ids = {item.evidence_id for item in resolved}
    if referenced_ids != resolved_ids:
        raise ValueError("Resolved evidence records do not exactly match dossier references")
    for record in resolved:
        if registry.resolve(record.evidence_id).to_dict() != record.to_dict():
            raise ValueError(
                f"Resolved evidence {record.evidence_id!r} differs from registry"
            )

    expected_assessment = evaluate_clinical_utility(dossier, registry)
    expected_payload = expected_assessment.to_dict()
    # Generation time is output metadata rather than a scientific input.  It
    # is validated independently above; every other field must be exactly
    # reproducible from the dossier and tamper-evident registry.
    expected_payload["created_at"] = payload["created_at"]
    if payload != expected_payload:
        differing = sorted(
            key
            for key in set(payload) | set(expected_payload)
            if payload.get(key) != expected_payload.get(key)
        )
        raise ValueError(
            "Clinical-utility payload is not reproducible from its dossier "
            f"and evidence registry; differing fields: {differing}"
        )

    gate_rows = list(payload["gates"])
    gate_ids = tuple(str(item["gate_id"]) for item in gate_rows)
    if len(set(gate_ids)) != len(gate_ids) or set(gate_ids) != set(
        EXPECTED_UTILITY_GATE_IDS
    ):
        raise ValueError("Clinical-utility gate set is incomplete or duplicated")
    by_id = {str(item["gate_id"]): item for item in gate_rows}
    for gate_id, gate in by_id.items():
        passed = bool(gate["passed"])
        if gate["status"] != ("pass" if passed else "fail"):
            raise ValueError(f"Gate status is inconsistent: {gate_id}")
        if passed == bool(gate["missing_requirements"]):
            raise ValueError(f"Gate missing requirements are inconsistent: {gate_id}")

    expected_level = ClaimLevel.HYPOTHESIS_ONLY
    for candidate_level in CLAIM_LEVEL_ORDER[1:]:
        if all(by_id[item]["passed"] for item in CLAIM_REQUIREMENTS[candidate_level]):
            expected_level = candidate_level
        else:
            break
    if payload["claim_level"] != expected_level.value:
        raise ValueError("Claim level is inconsistent with hard gates")
    expected_rank = CLAIM_LEVEL_ORDER.index(expected_level)
    if payload["claim_rank"] != expected_rank:
        raise ValueError("Claim rank is inconsistent")
    expected_next = (
        CLAIM_LEVEL_ORDER[expected_rank + 1].value
        if expected_rank + 1 < len(CLAIM_LEVEL_ORDER)
        else None
    )
    if payload["next_claim_level"] != expected_next:
        raise ValueError("Next claim level is inconsistent")
    process = expected_rank >= CLAIM_LEVEL_ORDER.index(
        ClaimLevel.DECISION_PROCESS_UTILITY_DEMONSTRATED
    )
    patient = expected_level == ClaimLevel.PATIENT_BENEFIT_DEMONSTRATED
    for key in (
        "clinical_utility_demonstrated",
        "clinical_utility_claim_allowed",
        "decision_process_utility_demonstrated",
    ):
        if payload[key] is not process:
            raise ValueError(f"{key} is inconsistent with claim level")
    for key in ("patient_benefit_demonstrated", "patient_benefit_claim_allowed"):
        if payload[key] is not patient:
            raise ValueError(f"{key} is inconsistent with claim level")
    scope_passed = bool(by_id["scope_freeze_attested"]["passed"])
    expected_access = "attested_unread" if scope_passed else "not_attested"
    if payload["locked_test_access_status"] != expected_access:
        raise ValueError("Locked-test access status is not attestation-derived")
    expected_read_value = False if scope_passed else None
    if (
        payload["locked_test_data_read"] is not expected_read_value
        or payload["locked_test_outcomes_read"] is not expected_read_value
    ):
        raise ValueError("Locked-test read status is not attestation-derived")


def _publish_clinical_utility_directory(
    staging_root: Path,
    output_root: Path,
) -> None:
    parent = output_root.parent.resolve()
    if staging_root.resolve().parent != parent or not staging_root.name.startswith("."):
        raise ValueError("Clinical-utility staging directory is outside output parent")
    backup: Optional[Path] = None
    if output_root.exists():
        backup = output_root.with_name(
            f".{output_root.name}.{uuid.uuid4().hex}.backup"
        )
        os.replace(output_root, backup)
    try:
        os.replace(staging_root, output_root)
    except Exception:
        if backup is not None and backup.exists() and not output_root.exists():
            os.replace(backup, output_root)
        raise
    if backup is not None and backup.exists():
        resolved = backup.resolve()
        if resolved.parent != parent or not resolved.name.startswith("."):
            raise ValueError("Refusing to remove unsafe clinical-utility backup")
        shutil.rmtree(resolved)


def write_clinical_utility_assessment(
    assessment: ClinicalUtilityAssessment,
    output_root: Path | str,
) -> dict[str, Path]:
    """Validate, stage and atomically publish an ID-free assessment bundle."""

    root = Path(output_root).resolve()
    if root == root.parent:
        raise ValueError("Clinical-utility output must be a named directory")
    if root.exists() and not root.is_dir():
        raise ValueError("Clinical-utility output must be a directory")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.",
            suffix=".staging",
            dir=root.parent,
        )
    )
    json_path = staging_root / "clinical_utility_assessment.json"
    gates_path = staging_root / "clinical_utility_gates.parquet"
    summary_path = staging_root / "clinical_utility_summary.parquet"
    markdown_path = staging_root / "clinical_utility_assessment.md"
    manifest_path = staging_root / "clinical_utility_assessment.manifest.json"
    payload = assessment.to_dict()
    validate_clinical_utility_payload(payload)
    try:
        atomic_write_json(json_path, payload)
        atomic_write_parquet(
            pd.DataFrame([gate.to_dict() for gate in assessment.gates]),
            gates_path,
        )
        atomic_write_parquet(
            pd.DataFrame(
                [
                    {
                        "assessment_status": "verified_metadata_assessment",
                        "candidate_id": payload["candidate_id"],
                        "use_id": assessment.dossier.intended_use.use_id,
                        "framework_scope": payload["framework_scope"],
                        "claim_level": payload["claim_level"],
                        "claim_rank": payload["claim_rank"],
                        "next_claim_level": payload["next_claim_level"],
                        "decision_process_utility_demonstrated": payload[
                            "decision_process_utility_demonstrated"
                        ],
                        "patient_benefit_demonstrated": payload[
                            "patient_benefit_demonstrated"
                        ],
                        "blocking_gates_for_demonstrated_utility": json.dumps(
                            payload[
                                "blocking_gates_for_demonstrated_utility"
                            ],
                            ensure_ascii=False,
                        ),
                        "blocking_gates_for_patient_benefit": json.dumps(
                            payload["blocking_gates_for_patient_benefit"],
                            ensure_ascii=False,
                        ),
                        "disclaimer": (
                            "Research-only evidence assessment; no clinical "
                            "use, regulatory authorization, AHA certification, "
                            "or formal AHA compliance is established."
                        ),
                    }
                ]
            ),
            summary_path,
        )
        atomic_write_text(markdown_path, render_clinical_utility_markdown(assessment))
        final_paths = (
            root / json_path.name,
            root / gates_path.name,
            root / summary_path.name,
            root / markdown_path.name,
        )
        integrity = []
        for record, final_path in zip(
            capture_output_integrity(
                (json_path, gates_path, summary_path, markdown_path),
                include_sha256=True,
            ),
            final_paths,
        ):
            updated = dict(record)
            updated["path"] = str(final_path.resolve())
            integrity.append(updated)
        access_status = payload["locked_test_access_status"]
        manifest = StageManifest.complete(
            stage=f"clinical-utility-assessment:{assessment.assessment_id}",
            stage_version=CLINICAL_UTILITY_STAGE_VERSION,
            input_hash=content_hash(
                {
                    "dossier_hash": assessment.dossier_hash,
                    "evidence_registry_sha256": (
                        assessment.evidence_registry.registry_sha256
                    ),
                }
            ),
            config_hash=content_hash(
                {
                    "schema_version": CLINICAL_UTILITY_SCHEMA_VERSION,
                    "schema": json.loads(
                        clinical_utility_schema_path().read_text(encoding="utf-8")
                    ),
                    "source_ids": [
                        source.source_id for source in AUTHORITATIVE_SOURCES
                    ],
                }
            ),
            output_paths=tuple(str(path.resolve()) for path in final_paths),
            row_counts={
                "gates": len(assessment.gates),
                "summary": 1,
                "resolved_evidence_records": len(assessment.resolved_evidence),
                "aha_metric_records": len(
                    assessment.dossier.aha_metric_records
                ),
            },
            notes=(
                "Verified declarative metadata assessment only; referenced artifacts were not opened.",
                f"locked_test_access_status={access_status}",
                "existing_internal_decision_curve_auto_import=false",
                f"framework_scope={assessment.dossier.framework_scope.value}",
            ),
            output_integrity=tuple(integrity),
        )
        manifest.save(manifest_path)
        _publish_clinical_utility_directory(staging_root, root)
    except Exception:
        if staging_root.exists():
            resolved = staging_root.resolve()
            if resolved.parent == root.parent.resolve() and resolved.name.startswith("."):
                shutil.rmtree(resolved)
        raise
    return {
        "assessment": root / json_path.name,
        "gates": root / gates_path.name,
        "summary": root / summary_path.name,
        "report": root / markdown_path.name,
        "manifest": root / manifest_path.name,
    }
