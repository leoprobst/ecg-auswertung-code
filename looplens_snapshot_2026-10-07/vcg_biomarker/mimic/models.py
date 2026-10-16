from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable, Mapping, Optional


class LabelState(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INDETERMINATE = "indeterminate"
    UNAVAILABLE = "unavailable"


class EligibilityState(StrEnum):
    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class TargetKind(StrEnum):
    DIAGNOSTIC = "diagnostic"
    RETROSPECTIVE_PROXY = "retrospective_proxy"
    PREDICTIVE = "predictive"
    # Reference standard and marker are two algorithms reading the *same*
    # waveform (e.g. the recorder's own QT measurement versus this project's
    # delineation of that identical record). Agreement there is analytical
    # concordance, not diagnostic accuracy against an independent standard,
    # and must never be reported or gated as if it were.
    ANALYTICAL_CONCORDANCE = "analytical_concordance"


class ConfidenceTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class PanelTier(StrEnum):
    PRIMARY_CORE = "primary_core"
    SENTINEL = "sentinel"
    DISCOVERY_RESERVE = "discovery_reserve"
    CLASSICAL = "classical"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class BaselineKind(StrEnum):
    FIXED_RULE = "fixed_rule"
    CONTINUOUS_MARKER = "continuous_marker"
    CALIBRATED_MODEL = "calibrated_model"
    EXPERT_PATTERN = "expert_pattern"
    NONE_ESTABLISHED = "none_established"


class ScoreReleaseStage(StrEnum):
    DEVELOPMENT = "development"
    FROZEN = "frozen"
    LOCKED_TEST_VALIDATED = "locked_test_validated"
    EXTERNAL_VALIDATED = "external_validated"


class CriterionStage(StrEnum):
    HYPOTHESIS = "hypothesis"
    TRAIN_STABLE = "train_stable"
    VALIDATION_REPLICATED = "validation_replicated"
    FROZEN = "frozen"
    LOCKED_TEST_VALIDATED = "locked_test_validated"
    EXTERNAL_VALIDATED = "external_validated"


class EvidenceGrade(StrEnum):
    TECHNICAL = "technical_same_ecg"
    DEVELOPMENT = "development"
    VALIDATION_REPLICATED = "validation_replicated"
    FROZEN = "frozen"
    LOCKED_TEST_VALIDATED = "locked_test_validated"
    EXTERNAL_VALIDATED = "external_validated"
    UNDERPOWERED = "underpowered"
    UNSTABLE = "unstable"


class ClinicalEvidenceTier(StrEnum):
    """Claim hierarchy exposed consistently by backend, exports and GUI."""

    EXPLORATORY = "exploratory"
    REPLICATED_ASSOCIATION = "replicated_association"
    GUIDELINE_CANDIDATE = "guideline_candidate"
    CLINICALLY_VALIDATED = "clinically_validated"


class ReportType(StrEnum):
    STUDY = "study"
    ENDPOINT = "endpoint"
    MARKER = "marker"
    CASE = "case"
    JOB_QC = "job_qc"


@dataclass(frozen=True)
class StageDescriptor:
    stage_id: str
    group_id: str
    display_name_de: str
    purpose_de: str
    result_de: str
    prerequisite_stage_ids: tuple[str, ...] = ()
    technical_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClinicalDisplayTerm:
    term_id: str
    display_name_de: str
    category_de: str
    short_definition_de: str
    source_summary_de: str = ""
    synonyms: tuple[str, ...] = ()

    @property
    def search_text(self) -> str:
        return " ".join(
            (
                self.term_id,
                self.display_name_de,
                self.category_de,
                self.short_definition_de,
                *self.synonyms,
            )
        ).casefold()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResultQuery:
    perspective: str = "pathologies"
    run_id: str = ""
    endpoint_ids: tuple[str, ...] = ()
    pathology_group_ids: tuple[str, ...] = ()
    search_text: str = ""
    panel_ids: tuple[str, ...] = ()
    feature_ids: tuple[str, ...] = ()
    feature_families: tuple[str, ...] = ()
    candidate_types: tuple[str, ...] = ()
    baseline_ids: tuple[str, ...] = ()
    evidence_grades: tuple[str, ...] = ()
    minimum_sensitivity: Optional[float] = None
    minimum_specificity: Optional[float] = None
    minimum_signal_quality: Optional[float] = None
    maximum_q_value: Optional[float] = None
    validation_only: bool = True
    include_historical: bool = False
    complete_runs_only: bool = True
    sort_by: str = "evidence_rank"
    sort_descending: bool = True
    offset: int = 0
    limit: int = 200

    def __post_init__(self) -> None:
        if self.perspective not in {
            "pathologies",
            "biomarkers",
            "criteria",
            "discovery",
        }:
            raise ValueError(f"Unsupported result perspective: {self.perspective}")
        for value, label in (
            (self.minimum_sensitivity, "minimum_sensitivity"),
            (self.minimum_specificity, "minimum_specificity"),
            (self.minimum_signal_quality, "minimum_signal_quality"),
            (self.maximum_q_value, "maximum_q_value"),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between zero and one")
        if self.offset < 0:
            raise ValueError("offset cannot be negative")
        if not 1 <= self.limit <= 5_000:
            raise ValueError("limit must be between 1 and 5000")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobGroup:
    group_id: str
    display_name_de: str
    job_ids: tuple[str, ...]
    state: str
    purpose_de: str
    result_de: str
    progress_current: int = 0
    progress_total: int = 0
    eta_seconds: Optional[float] = None
    action_required_de: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyRunSpec:
    study_id: str
    display_name_de: str
    preset_id: str
    run_id: str
    stage_ids: tuple[str, ...]
    include_phenome_scan: bool = False
    development_only: bool = True

    def __post_init__(self) -> None:
        if not self.study_id.strip() or not self.run_id.strip():
            raise ValueError("study_id and run_id are required")
        if not self.stage_ids:
            raise ValueError("A study run requires at least one stage")
        if not self.development_only:
            raise ValueError("Guided study runs must remain development-only")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DelineationResult:
    method: str
    rhythm_class: str
    signal_quality: float
    fiducial_confidence: float
    fiducials: Mapping[str, Optional[int]]
    boundary_uncertainty_ms: Mapping[str, Optional[float]]
    wave_status: Mapping[str, str]
    review_required: bool
    review_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fiducials"] = dict(self.fiducials)
        data["boundary_uncertainty_ms"] = dict(self.boundary_uncertainty_ms)
        data["wave_status"] = dict(self.wave_status)
        return data


@dataclass(frozen=True)
class ThresholdBand:
    operating_point: str
    threshold: float
    gray_zone_low: float
    gray_zone_high: float
    direction: int
    learned_split: str = "train"
    uncertainty_method: str = "not_estimated"
    bootstrap_iterations: int = 0
    successful_bootstrap_iterations: int = 0

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("Threshold direction must be +1 or -1")
        if self.gray_zone_low > self.threshold or self.gray_zone_high < self.threshold:
            raise ValueError("The threshold must lie inside its gray zone")
        if self.learned_split != "train":
            raise ValueError("Criterion thresholds may only be learned in train")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations cannot be negative")
        if not 0 <= self.successful_bootstrap_iterations <= self.bootstrap_iterations:
            raise ValueError(
                "successful_bootstrap_iterations must be between zero and "
                "bootstrap_iterations"
            )

    def classify(self, value: Optional[float]) -> str:
        if value is None:
            return "not_evaluable"
        aligned = self.direction * float(value)
        low = self.direction * self.gray_zone_low
        high = self.direction * self.gray_zone_high
        lower, upper = sorted((low, high))
        if lower <= aligned <= upper:
            return "gray_zone"
        return "positive" if aligned > upper else "negative"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApplicabilityRule:
    required_leads: tuple[str, ...] = ()
    minimum_signal_quality: float = 0.0
    minimum_fiducial_confidence: float = 0.0
    allowed_rhythm_classes: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_signal_quality <= 1.0:
            raise ValueError("minimum_signal_quality must be between zero and one")
        if not 0.0 <= self.minimum_fiducial_confidence <= 1.0:
            raise ValueError("minimum_fiducial_confidence must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Finding:
    finding_id: str
    endpoint_id: str
    feature_id: str
    evidence_grade: EvidenceGrade
    effect_direction: int
    validation_auc: Optional[float]
    sensitivity: Optional[float]
    specificity: Optional[float]
    q_value: Optional[float]
    snapshot_hash: str
    interpretation_de: str = ""

    def __post_init__(self) -> None:
        if self.effect_direction not in (-1, 1):
            raise ValueError("effect_direction must be +1 or -1")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_grade"] = self.evidence_grade.value
        return data


@dataclass(frozen=True)
class CriterionCandidate:
    criterion_id: str
    endpoint_id: str
    display_name_de: str
    formula: str
    component_ids: tuple[str, ...]
    thresholds: tuple[ThresholdBand, ...]
    applicability: ApplicabilityRule
    stage: CriterionStage
    evidence_grade: EvidenceGrade
    development_run_id: str
    snapshot_hash: str
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    roc_auc: Optional[float] = None
    confidence_interval: tuple[Optional[float], Optional[float]] = (None, None)
    baseline_id: str = ""
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.component_ids or len(self.component_ids) > 4:
            raise ValueError("A criterion requires one to four components")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        data["evidence_grade"] = self.evidence_grade.value
        return data


@dataclass(frozen=True)
class CriterionRelease:
    release_id: str
    candidate: CriterionCandidate
    release_hash: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def clinical_claim_allowed(self) -> bool:
        return self.candidate.stage == CriterionStage.EXTERNAL_VALIDATED

    @property
    def research_diagnosis_allowed(self) -> bool:
        return self.candidate.stage in {
            CriterionStage.VALIDATION_REPLICATED,
            CriterionStage.FROZEN,
            CriterionStage.LOCKED_TEST_VALIDATED,
            CriterionStage.EXTERNAL_VALIDATED,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceProfile:
    cpu_tokens: int = 1
    memory_gb: float = 1.0
    io_weight: int = 1
    writer_namespace: str = ""
    interactive: bool = False

    def __post_init__(self) -> None:
        if self.cpu_tokens < 1:
            raise ValueError("cpu_tokens must be at least one")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.io_weight < 0:
            raise ValueError("io_weight cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    stage: str
    project_file: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    resource_profile: ResourceProfile = field(default_factory=ResourceProfile)
    priority: int = 50
    requested_by: str = "local_user"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id is required")
        if not self.stage.strip():
            raise ValueError("stage is required")
        if not self.project_file.strip():
            raise ValueError("project_file is required")
        if not 0 <= self.priority <= 100:
            raise ValueError("priority must be between 0 and 100")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["arguments"] = dict(self.arguments)
        return data


@dataclass(frozen=True)
class JobRecord:
    spec: JobSpec
    state: JobState
    progress_current: int = 0
    progress_total: int = 0
    progress_message: str = ""
    worker_pid: Optional[int] = None
    heartbeat_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: str = ""
    artifact_hash: str = ""

    @property
    def progress_fraction(self) -> Optional[float]:
        if self.progress_total <= 0:
            return None
        return min(1.0, max(0.0, self.progress_current / self.progress_total))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


@dataclass(frozen=True)
class ArtifactSnapshot:
    snapshot_id: str
    artifact_kind: str
    source_root: str
    files: tuple[str, ...]
    file_hashes: Mapping[str, str]
    snapshot_hash: str
    complete: bool
    input_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return "complete" if self.complete else "partial"

    def assert_freezable(self) -> None:
        if not self.complete:
            raise RuntimeError(
                f"Snapshot {self.snapshot_id!r} is partial and cannot be frozen"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["file_hashes"] = dict(self.file_hashes)
        data["status"] = self.status
        return data


@dataclass(frozen=True)
class ParameterPolicy:
    diagnostic_band_hz: tuple[float, float] = (0.05, 150.0)
    delineation_band_hz: tuple[float, float] = (0.5, 40.0)
    morphology_band_hz: tuple[float, float] = (0.05, 100.0)
    powerline_hz: float = 60.0
    bootstrap_iterations: int = 1000
    threshold_targets: tuple[str, ...] = (
        "sensitivity_at_90_specificity",
        "sensitivity_at_95_specificity",
        "specificity_at_90_sensitivity",
    )
    max_development_score_components: int = 3
    grouped_folds: int = 5
    grouped_cv_repeats: int = 1
    maximum_component_abs_correlation: float = 0.90
    minimum_score_incremental_auc: float = 0.005
    score_representatives_per_family: int = 4
    score_validation_pairs: int = 5
    score_validation_triples: int = 3
    score_triple_seed_pairs: int = 25
    include_transform_sensitivity: bool = True
    include_time_window_sensitivity: bool = True
    include_subgroup_stability: bool = True
    include_broad_phenome_scan: bool = False
    allow_outcome_tuned_filters: bool = False
    allow_label_rule_optimization: bool = False
    allow_locked_test_optimization: bool = False

    def __post_init__(self) -> None:
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations cannot be negative")
        if self.max_development_score_components not in (1, 2, 3):
            raise ValueError("Development scores may contain at most three components")
        if self.grouped_folds < 2:
            raise ValueError("grouped_folds must be at least two")
        if self.grouped_cv_repeats < 1:
            raise ValueError("grouped_cv_repeats must be at least one")
        if not 0.0 < self.maximum_component_abs_correlation <= 1.0:
            raise ValueError(
                "maximum_component_abs_correlation must be in (0, 1]"
            )
        if self.minimum_score_incremental_auc < 0:
            raise ValueError(
                "minimum_score_incremental_auc cannot be negative"
            )
        if self.score_representatives_per_family < 1:
            raise ValueError(
                "score_representatives_per_family must be at least one"
            )
        if self.score_validation_pairs < 0 or self.score_validation_triples < 0:
            raise ValueError("score validation candidate counts cannot be negative")
        if self.score_triple_seed_pairs < 0:
            raise ValueError("score_triple_seed_pairs cannot be negative")
        if (
            self.allow_outcome_tuned_filters
            or self.allow_label_rule_optimization
            or self.allow_locked_test_optimization
        ):
            raise ValueError(
                "Scientific guard parameters cannot be enabled by an analysis preset"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisPreset:
    preset_id: str
    display_name_de: str
    panel_ids: tuple[str, ...]
    policy: ParameterPolicy
    description_de: str
    expert_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = self.policy.to_dict()
        return data


@dataclass(frozen=True)
class BaselineDefinition:
    baseline_id: str
    endpoint_ids: tuple[str, ...]
    display_name: str
    kind: BaselineKind
    source_title: str
    source_version: str
    source_url: str
    formula: str
    required_inputs: tuple[str, ...]
    exclusions: tuple[str, ...] = ()
    strata: tuple[str, ...] = ()
    criterion_id: str = ""
    notes: str = ""
    literature_reference_metrics: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass(frozen=True)
class BaselineResult:
    baseline_id: str
    subject_id: int
    study_id: int
    evaluable: bool
    value: Optional[float]
    positive: Optional[bool]
    exclusion_reason: str = ""
    source_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhenotypeDefinition:
    phenotype_id: str
    display_name: str
    body_system: str
    mapping_system: str
    mapping_version: str
    icd9_codes: tuple[str, ...] = ()
    icd10_codes: tuple[str, ...] = ()
    exclusion_phenotypes: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    child_ids: tuple[str, ...] = ()
    minimum_case_count: int = 30
    retrospective_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryHit:
    phenotype_id: str
    feature_id: str
    panel_id: str
    train_effect: float
    train_q_value: float
    validation_effect: Optional[float]
    validation_q_value: Optional[float]
    direction_consistent: bool
    replicated: bool
    inference_status: str
    snapshot_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionCandidate:
    candidate_id: str
    phenotype_id: str
    feature_ids: tuple[str, ...]
    evidence_snapshot_hash: str
    proposed_label_definition: Mapping[str, Any]
    status: str = "draft"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    effect_size: Optional[float] = None
    effect_ci_low: Optional[float] = None
    effect_ci_high: Optional[float] = None
    train_q_value: Optional[float] = None
    validation_q_value: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["proposed_label_definition"] = dict(self.proposed_label_definition)
        return data


@dataclass(frozen=True)
class ScoreRelease:
    release_id: str
    endpoint_id: str
    score_spec: FrozenScoreSpec
    stage: ScoreReleaseStage
    reference_type: str
    development_run_id: str
    freeze_hash: str = ""
    locked_test_artifact_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def primary_case_display_allowed(self) -> bool:
        return self.stage in (
            ScoreReleaseStage.LOCKED_TEST_VALIDATED,
            ScoreReleaseStage.EXTERNAL_VALIDATED,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        return data


@dataclass(frozen=True)
class ImportedECGCase:
    case_id: str
    pseudonym: str
    source_format: str
    source_path: str
    source_sha256: str
    sampling_rate_hz: float
    lead_names: tuple[str, ...]
    units: tuple[str, ...]
    imported_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class AnnotationRevision:
    revision_id: str
    case_id: str
    parent_revision_id: str
    fiducials: Mapping[str, int]
    notes: str = ""
    author: str = "local_user"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fiducials"] = dict(self.fiducials)
        return data


@dataclass(frozen=True)
class ReportSpec:
    report_id: str
    report_type: ReportType
    run_id: str
    language: str = "de"
    formats: tuple[str, ...] = ("pdf", "html", "xlsx", "parquet")
    endpoint_ids: tuple[str, ...] = ()
    feature_ids: tuple[str, ...] = ()
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel")
    sections: tuple[str, ...] = (
        "summary",
        "methods",
        "cohort",
        "performance",
        "baselines",
        "stability",
        "limitations",
        "provenance",
    )
    include_case_ids: bool = False
    include_case_appendix: bool = False
    page_size: str = "A4"

    def __post_init__(self) -> None:
        if self.language not in ("de", "en"):
            raise ValueError("Report language must be 'de' or 'en'")
        allowed = {"pdf", "html", "xlsx", "csv", "parquet"}
        if not self.formats or not set(self.formats).issubset(allowed):
            raise ValueError(f"Unsupported report format in {self.formats!r}")
        if self.page_size != "A4":
            raise ValueError("The clinical report renderer currently supports A4")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["report_type"] = self.report_type.value
        return data


# Tables whose content is the ECG recorder's own measurement. A label sourced
# from these alone has no reference outside the tracing it is measured on.
ECG_DERIVED_SOURCE_TABLES = frozenset({"machine_measurements"})


@dataclass(frozen=True)
class LabelDefinition:
    label_id: str
    display_name: str
    target_kind: TargetKind
    source_tables: tuple[str, ...]
    index_policy: str
    window_before_hours: Optional[float] = None
    window_after_hours: Optional[float] = None
    positive_rule: str = ""
    negative_rule: str = ""
    exclusions: tuple[str, ...] = ()
    confidence_rule: str = ""
    rule_version: str = "1.0.0"
    references: tuple[str, ...] = ()
    panel: str = "secondary"
    # What the binary contrast actually estimates, and which selection its
    # control definition imposes. Left empty for endpoints measured at the
    # index itself; set explicitly wherever the control state requires
    # something to be true *after* the index, because that conditions the
    # cohort on the future and cannot be read as a plain risk contrast.
    estimand: str = ""
    selection_mechanism: str = ""
    primary_analysis_arm: str = "binary"

    @property
    def reference_independent_of_ecg(self) -> bool:
        """Whether the outcome is defined by something other than this ECG.

        The two largest incremental AUCs the project has ever produced were
        ``qtc_gt_500`` (+0.129) and ``qtc_gt_480`` (+0.127) -- and both ask a
        model to predict a threshold on the recorder's own QTc from markers
        measured on the same tracing. That is a tautology wearing a
        performance figure, and it sat unmarked next to real findings.

        Derived from the sources rather than listed by hand, so a label added
        later is classified without anyone remembering to. **Only** when every
        source is the machine measurement is the reference the ECG itself: the
        new-onset AF labels also read it, but there it excludes prevalent AF
        at baseline while the outcome comes from a diagnosis code. That is a
        dependency in the eligibility rule, not in the reference standard, and
        conflating the two would retire three real endpoints.
        """

        sources = {str(table) for table in self.source_tables}
        if not sources:
            return True
        return not sources <= ECG_DERIVED_SOURCE_TABLES

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_kind"] = self.target_kind.value
        data["reference_independent_of_ecg"] = self.reference_independent_of_ecg
        return data


@dataclass(frozen=True)
class LabelEvidence:
    subject_id: int
    study_id: int
    label_id: str
    state: LabelState
    eligibility: EligibilityState
    confidence: ConfidenceTier
    rule_version: str
    evidence_ids: tuple[str, ...] = ()
    evidence_summary: str = ""
    time_delta_hours: Optional[float] = None
    time_precision: str = "timestamp"
    source_tables: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("state", "eligibility", "confidence"):
            data[key] = getattr(self, key).value
        return data


@dataclass(frozen=True)
class FeaturePanelSpec:
    panel_id: str
    version: str
    tier: PanelTier
    feature_ids: tuple[str, ...]
    description: str = ""
    frozen_before_outcome_review: bool = True
    required_complete: bool = False

    def __post_init__(self) -> None:
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError(f"Panel {self.panel_id!r} contains duplicate feature IDs")
        invalid = [
            name for name in self.feature_ids
            if any(token in name for token in ("<", ">", "*", "/"))
        ]
        if invalid:
            raise ValueError(
                f"Panel {self.panel_id!r} contains non-scalar/template IDs: {invalid[:5]}"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tier"] = self.tier.value
        return data


@dataclass(frozen=True)
class FrozenScoreSpec:
    score_id: str
    version: str
    origin: str
    endpoint_id: str
    components: tuple[str, ...]
    directions: tuple[int, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    training_split_hash: str
    aggregation: str = "unweighted_mean"
    exploratory_score: bool = True

    def __post_init__(self) -> None:
        lengths = {
            len(self.components),
            len(self.directions),
            len(self.means),
            len(self.standard_deviations),
        }
        if len(lengths) != 1 or not self.components:
            raise ValueError("Frozen score arrays must be non-empty and equally sized")
        if len(self.components) > 4:
            raise ValueError("Primary interpretable scores may contain at most four components")
        if any(direction not in (-1, 1) for direction in self.directions):
            raise ValueError("Score directions must be +1 or -1")
        if any(sd <= 0 for sd in self.standard_deviations):
            raise ValueError("Score standard deviations must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAssignment:
    subject_id: int
    split: str
    split_value: float
    split_hash: str


@dataclass(frozen=True)
class ValidationSpec:
    run_id: str
    split_hash: str
    endpoint_ids: tuple[str, ...]
    panel_versions: Mapping[str, str]
    threshold_targets: tuple[str, ...] = (
        "sensitivity_at_90_specificity",
        "sensitivity_at_95_specificity",
        "specificity_at_90_sensitivity",
    )
    development_bootstrap_iterations: int = 1000
    final_bootstrap_iterations: int = 2000
    random_seed: int = 20260725
    primary_fdr_alpha: float = 0.05
    discovery_fdr_alpha: float = 0.01
    min_class_inference: int = 30
    min_class_primary: int = 100
    grouped_score_folds: int = 5
    grouped_score_repeats: int = 1
    maximum_component_abs_correlation: float = 0.90
    minimum_score_incremental_auc: float = 0.005
    grouped_score_representatives_per_family: int = 2
    grouped_score_selected_pairs: int = 1
    grouped_score_selected_triples: int = 1
    grouped_score_triple_seed_pairs: int = 25

    def __post_init__(self) -> None:
        if self.grouped_score_folds < 2:
            raise ValueError("grouped_score_folds must be at least two")
        if self.grouped_score_repeats < 1:
            raise ValueError("grouped_score_repeats must be at least one")
        if not 0.0 < self.maximum_component_abs_correlation <= 1.0:
            raise ValueError(
                "maximum_component_abs_correlation must be in (0, 1]"
            )
        if self.minimum_score_incremental_auc < 0:
            raise ValueError(
                "minimum_score_incremental_auc cannot be negative"
            )
        if self.grouped_score_representatives_per_family < 1:
            raise ValueError(
                "grouped_score_representatives_per_family must be at least one"
            )
        if (
            self.grouped_score_selected_pairs < 0
            or self.grouped_score_selected_triples < 0
        ):
            raise ValueError("grouped score selection counts cannot be negative")
        if self.grouped_score_triple_seed_pairs < 0:
            raise ValueError("grouped_score_triple_seed_pairs cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["panel_versions"] = dict(self.panel_versions)
        return data


@dataclass
class TestLock:
    run_id: str
    freeze_hash: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    evaluation_started_at: Optional[str] = None
    consumed_at: Optional[str] = None
    consumed_artifact_hash: Optional[str] = None
    invalidated_at: Optional[str] = None
    invalidation_reason: Optional[str] = None

    @property
    def is_frozen(self) -> bool:
        return self.invalidated_at is None

    @property
    def is_consumed(self) -> bool:
        return self.consumed_at is not None

    @property
    def is_evaluation_started(self) -> bool:
        return self.evaluation_started_at is not None

    def assert_evaluable(self, current_hash: str) -> None:
        if not self.is_frozen:
            raise RuntimeError(
                f"Locked test specification is invalidated: {self.invalidation_reason}"
            )
        if current_hash != self.freeze_hash:
            raise RuntimeError(
                "Locked test refused: current code/config hash differs from the frozen hash"
            )
        if self.is_consumed:
            raise RuntimeError(
                f"Locked test was already consumed at {self.consumed_at}"
            )
        if self.is_evaluation_started:
            raise RuntimeError(
                "Locked test evaluation was already started at "
                f"{self.evaluation_started_at}; a second access is refused"
            )

    def begin_evaluation(self) -> None:
        if self.is_evaluation_started or self.is_consumed:
            raise RuntimeError("Locked test evaluation can only start once")
        self.evaluation_started_at = datetime.now(timezone.utc).isoformat()

    def consume(self, artifact_hash: str) -> None:
        if not self.is_evaluation_started:
            raise RuntimeError(
                "Locked test must be marked as started before it can be consumed"
            )
        if self.is_consumed:
            raise RuntimeError("Locked test can only be consumed once")
        self.consumed_at = datetime.now(timezone.utc).isoformat()
        self.consumed_artifact_hash = artifact_hash

    def invalidate(self, reason: str) -> None:
        if self.is_evaluation_started:
            raise RuntimeError(
                "A started locked-test evaluation cannot be invalidated"
            )
        self.invalidated_at = datetime.now(timezone.utc).isoformat()
        self.invalidation_reason = reason

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def records_to_dicts(records: Iterable[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        if hasattr(record, "to_dict"):
            output.append(record.to_dict())
        else:
            output.append(asdict(record))
    return output
