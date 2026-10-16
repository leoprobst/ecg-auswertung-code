from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

import vcg_biomarker.mimic.clinical_utility as utility_module
import vcg_biomarker.mimic_cli as mimic_cli
from vcg_biomarker.mimic.adjusted_models import (
    FrozenLogisticModel,
    decision_curve as internal_decision_curve,
)
from vcg_biomarker.mimic.clinical_utility import (
    AUTHORITATIVE_SOURCES,
    AHA_REQUIRED_METRICS,
    AhaMetricId,
    AhaMetricRecord,
    AnalyticalValidityEvidence,
    CalibrationEvidence,
    ClaimLevel,
    ClinicalAssociationEvidence,
    ClinicalTask,
    ClinicalUtilityDossier,
    ComparatorDefinition,
    DecisionCurveEvidence,
    EvaluationFamily,
    EvidenceRecord,
    EvidenceRegistry,
    EvidenceStatus,
    ExternalValidationEvidence,
    FeasibilityActionabilityEvidence,
    FrameworkScope,
    IncrementalValueEvidence,
    IntendedUse,
    InternalValidationEvidence,
    ProspectiveImpactEvidence,
    ReclassificationEvidence,
    ReportingBiasEvidence,
    ScopeFreezeAttestation,
    SubgroupGeneralizabilityEvidence,
    assess_decision_curve,
    clinical_utility_dossier_from_dict,
    evaluate_clinical_utility,
    validate_clinical_utility_payload,
    write_clinical_utility_assessment,
)
from vcg_biomarker.mimic.services import run_service_stage
from vcg_biomarker.mimic.storage import content_hash
from vcg_biomarker.mimic.reports import (
    _load_report_tables,
    resolve_clinical_utility_report_paths,
)


_AT = "2026-08-12T08:00:00+00:00"


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _record(
    evidence_id: str,
    *domains: str,
    external: bool = False,
    split: str | None = None,
    assessor_id: str = "assessor-1",
    evidence_kind: str = "aggregate_validation_metadata",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence_kind=evidence_kind,
        artifact_uri=f"urn:evidence:{evidence_id}",
        artifact_sha256=_sha(f"artifact:{evidence_id}"),
        manifest_id=f"manifest:{evidence_id}",
        manifest_uri=f"urn:manifest:{evidence_id}",
        manifest_sha256=_sha(f"manifest:{evidence_id}"),
        dataset_id=("external-cohort" if external else "development-cohort"),
        site_id=("external-site" if external else "development-site"),
        split=(split or ("external_validation" if external else "development_validation")),
        assessor_id=assessor_id,
        assessor_organization="Independent evidence office",
        assessed_at=_AT,
        domains=tuple(domains),
        independent_from_development=external,
    )


def _registry() -> EvidenceRegistry:
    custody = _record(
        "scope-freeze-custody-v1",
        "governance",
        split="governance",
        assessor_id="custodian-1",
        evidence_kind="scope_freeze_custody_attestation",
    )
    records = (
        custody,
        _record("comparator-protocol-v1", "comparator"),
        _record("analytical-validation-v1", "analytical_validity"),
        _record("clinical-association-v1", "clinical_association"),
        _record("internal-validation-v1", "internal_validation"),
        _record("external-validation-v1", "external_validation", external=True),
        _record("external-calibration-v1", "calibration", external=True),
        _record("incremental-value-v1", "incremental_value", external=True),
        _record("external-dca-analysis-v1", "decision_curve", external=True),
        _record("reporting-checklist-and-rob-v1", "reporting_and_bias"),
        _record("implementation-evaluation-v1", "feasibility_actionability"),
        _record(
            "subgroup-generalizability-v1",
            "subgroup_generalizability",
            external=True,
        ),
        _record(
            "registered-impact-study-v1",
            "prospective_impact",
            external=True,
            split="prospective_impact",
        ),
        _record("categorical-nri-v1", "reclassification", external=True),
        _record("aha-metrics-v1", "aha_metric", external=True),
    )
    return EvidenceRegistry(
        registry_id="clinical-evidence-registry-v2",
        development_site_id="development-site",
        created_at="2026-08-12T07:00:00+00:00",
        created_by="evidence-registry-service",
        records=records,
    )


def _intended_use(
    family: EvaluationFamily = EvaluationFamily.DIAGNOSTIC_ACCURACY,
) -> IntendedUse:
    return IntendedUse(
        use_id="ecg-triage-v1",
        evaluation_family=family,
        clinical_task=(
            ClinicalTask.PROGNOSIS
            if family == EvaluationFamily.PREDICTION_MODEL
            else ClinicalTask.TRIAGE
        ),
        target_population="Adults receiving a 12-lead ECG in acute care",
        care_setting="Emergency department",
        intended_user="Treating physician",
        workflow_position="After ECG acquisition and before disposition",
        index_test="Frozen ECG/VCG candidate version 1",
        target_condition_or_outcome=(
            "Incident cardiovascular disease"
            if family == EvaluationFamily.PREDICTION_MODEL
            else "Clinically adjudicated target pathology"
        ),
        reference_standard="Blinded clinical adjudication",
        clinical_decision="Escalate diagnostic work-up",
        action_if_positive="Immediate protocolized confirmatory work-up",
        action_if_negative="Continue current standard pathway",
        unit_of_analysis="Patient at index encounter",
        time_horizon="10 years" if family == EvaluationFamily.PREDICTION_MODEL else "",
        outputs_absolute_risk=True,
        clinically_actionable_thresholds=(0.1, 0.2, 0.3),
        anticipated_benefits=("Earlier appropriate escalation",),
        anticipated_harms=("Additional testing and false-positive escalation",),
        exclusions=("Uninterpretable ECG",),
        frozen_before_evaluation=True,
    )


def _comparator() -> ComparatorDefinition:
    return ComparatorDefinition(
        comparator_id="current-standard-v1",
        description="Current implemented standard-care pathway",
        current_standard_of_care=True,
        clinically_implemented=True,
        frozen_before_comparison=True,
        evaluated_head_to_head=True,
        evaluated_in_same_participants=True,
        includes_default_strategies=True,
        evidence_ids=("comparator-protocol-v1",),
    )


def _freeze(
    intended_use: IntendedUse,
    comparator: ComparatorDefinition,
    registry: EvidenceRegistry,
) -> ScopeFreezeAttestation:
    custody = registry.resolve("scope-freeze-custody-v1")
    return ScopeFreezeAttestation(
        attestation_id="scope-freeze-v2",
        candidate_id="candidate-v1",
        use_id=intended_use.use_id,
        intended_use_sha256=content_hash(intended_use.to_dict()),
        comparator_sha256=content_hash(comparator.to_dict()),
        thresholds_sha256=content_hash(
            list(intended_use.clinically_actionable_thresholds)
        ),
        evidence_registry_sha256=registry.registry_sha256,
        custody_evidence_id=custody.evidence_id,
        custody_manifest_sha256=custody.manifest_sha256,
        attested_by="custodian-1",
        attested_at=_AT,
        frozen_before_evaluation=True,
        locked_test_data_attested_unread=True,
        locked_test_outcomes_attested_unread=True,
    )


def _decision_curve() -> DecisionCurveEvidence:
    return assess_decision_curve(
        pd.DataFrame(
            {
                "threshold_probability": [0.1, 0.2, 0.3],
                "candidate_net_benefit": [0.08, 0.07, 0.06],
                "comparator_net_benefit": [0.05, 0.04, 0.03],
                "treat_all_net_benefit": [0.03, 0.02, 0.01],
                "treat_none_net_benefit": [0.0, 0.0, 0.0],
                "delta_net_benefit_ci_low": [0.01, 0.01, 0.005],
            }
        ),
        actionable_thresholds=(0.1, 0.2, 0.3),
        primary_threshold=0.2,
        evidence_ids=("external-dca-analysis-v1",),
        thresholds_prespecified=True,
        thresholds_clinically_actionable=True,
        evaluated_in_external_validation=True,
        minimum_margin_prespecified=True,
    )


def _aha_metrics() -> tuple[AhaMetricRecord, ...]:
    external_metrics = {
        AhaMetricId.EXTERNAL_VALIDATION,
        AhaMetricId.CALIBRATION_IN_THE_LARGE,
        AhaMetricId.CALIBRATION_SLOPE,
        AhaMetricId.CALIBRATION_SOCIODEMOGRAPHIC_SUBGROUPS,
        AhaMetricId.INCREMENTAL_DELTA_C_STATISTIC,
        AhaMetricId.INCREMENTAL_DELTA_CALIBRATION,
        AhaMetricId.CATEGORICAL_NRI,
        AhaMetricId.DECISION_CURVE,
    }
    return tuple(
        AhaMetricRecord(
            metric_id=metric_id,
            status=EvidenceStatus.SUPPORTED,
            estimate=1.0,
            unit="metric-specific unit",
            confidence_interval_low=0.8,
            confidence_interval_high=1.2,
            uncertainty_reported=True,
            population="Frozen intended-use population",
            time_horizon="10 years",
            comparator_id="current-standard-v1",
            threshold_definition="Prespecified actionable thresholds 0.1-0.3",
            external_validation=metric_id in external_metrics,
            prespecified=True,
            interpretation="Prespecified criterion met",
            evidence_ids=("aha-metrics-v1",),
        )
        for metric_id in AHA_REQUIRED_METRICS
    )


def _complete_bundle(
    *,
    framework_scope: FrameworkScope = FrameworkScope.DIAGNOSTIC_ADAPTATION,
    impact_design: str | None = None,
    impact_outcome: str = "clinical_management",
) -> tuple[ClinicalUtilityDossier, EvidenceRegistry]:
    family = (
        EvaluationFamily.PREDICTION_MODEL
        if framework_scope == FrameworkScope.AHA_INCIDENT_CVD_RISK
        else EvaluationFamily.DIAGNOSTIC_ACCURACY
    )
    registry = _registry()
    intended_use = _intended_use(family)
    comparator = _comparator()
    reporting = ReportingBiasEvidence(
        status=EvidenceStatus.SUPPORTED,
        reporting_guideline=(
            "TRIPOD+AI"
            if family == EvaluationFamily.PREDICTION_MODEL
            else "STARD-AI"
        ),
        reporting_checklist_complete=True,
        risk_of_bias_tool=(
            "PROBAST+AI"
            if family == EvaluationFamily.PREDICTION_MODEL
            else "documented design-appropriate diagnostic appraisal"
        ),
        low_risk_of_bias=True,
        low_concern_for_applicability=True,
        protocol_or_analysis_plan_public=True,
        evidence_ids=("reporting-checklist-and-rob-v1",),
    )
    impact = ProspectiveImpactEvidence()
    if impact_design is not None:
        impact = ProspectiveImpactEvidence(
            status=EvidenceStatus.SUPPORTED,
            study_design=impact_design,
            prospectively_registered=True,
            protocol_and_primary_outcome_prespecified=True,
            candidate_frozen_before_enrolment=True,
            concurrent_standard_care_comparator=True,
            evaluated_in_intended_workflow=True,
            clinically_meaningful_primary_outcome=True,
            primary_outcome_type=impact_outcome,
            primary_outcome_improved=True,
            uncertainty_reported=True,
            harms_and_unintended_consequences_acceptable=True,
            implementation_fidelity_reported=True,
            sample_size_adequate=True,
            low_risk_of_bias=True,
            evidence_ids=("registered-impact-study-v1",),
        )
    dossier = ClinicalUtilityDossier(
        assessment_id="utility-assessment-v2",
        candidate_id="candidate-v1",
        framework_scope=framework_scope,
        intended_use=intended_use,
        comparator=comparator,
        scope_freeze_attestation=_freeze(intended_use, comparator, registry),
        aha_metric_records=(
            _aha_metrics()
            if framework_scope == FrameworkScope.AHA_INCIDENT_CVD_RISK
            else ()
        ),
        analytical_validity=AnalyticalValidityEvidence(
            status=EvidenceStatus.SUPPORTED,
            measurement_procedure_frozen=True,
            preanalytic_and_signal_requirements_prespecified=True,
            repeatability_supported=True,
            reproducibility_across_devices_or_software_supported=True,
            failure_and_abstention_rates_reported=True,
            quality_control_acceptance_criteria_met=True,
            evidence_ids=("analytical-validation-v1",),
        ),
        clinical_association=ClinicalAssociationEvidence(
            status=EvidenceStatus.SUPPORTED,
            analysis_prespecified=True,
            reference_standard_appropriate=True,
            reference_assessment_blinded_to_index_test=True,
            incorporation_bias_absent=True,
            temporal_alignment_appropriate=True,
            uncertainty_reported=True,
            evidence_ids=("clinical-association-v1",),
        ),
        internal_validation=InternalValidationEvidence(
            status=EvidenceStatus.SUPPORTED,
            candidate_frozen_before_evaluation=True,
            participant_level_separation=True,
            target_population_represented=True,
            sample_size_and_event_count_adequate=True,
            missing_data_handled_appropriately=True,
            discrimination_or_accuracy_reported=True,
            threshold_performance_reported=True,
            uncertainty_reported=True,
            prespecified_acceptance_criteria_met=True,
            evidence_ids=("internal-validation-v1",),
        ),
        external_validation=ExternalValidationEvidence(
            status=EvidenceStatus.SUPPORTED,
            external_dataset_count=1,
            distinct_institution_count=1,
            independent_investigators=True,
            candidate_applied_unchanged=True,
            intended_use_matches=True,
            target_population_represented=True,
            sample_size_and_event_count_adequate=True,
            discrimination_or_accuracy_reported=True,
            threshold_performance_reported=True,
            uncertainty_reported=True,
            prespecified_acceptance_criteria_met=True,
            evidence_ids=("external-validation-v1",),
        ),
        calibration=CalibrationEvidence(
            status=EvidenceStatus.SUPPORTED,
            evaluated_in_external_validation=True,
            calibration_plot_reported=True,
            calibration_in_the_large_reported=True,
            calibration_slope_reported=True,
            calibration_across_relevant_range_reported=True,
            subgroup_calibration_reported=True,
            acceptance_criteria_prespecified=True,
            acceptance_criteria_met=True,
            evidence_ids=("external-calibration-v1",),
        ),
        incremental_value=IncrementalValueEvidence(
            status=EvidenceStatus.SUPPORTED,
            comparator_id=comparator.comparator_id,
            comparison_design="add_on",
            head_to_head_same_participants=True,
            discrimination_or_accuracy_compared=True,
            calibration_compared_when_applicable=True,
            threshold_performance_compared=True,
            uncertainty_reported=True,
            minimum_clinically_meaningful_improvement_prespecified=True,
            prespecified_minimum_met=True,
            evidence_ids=("incremental-value-v1",),
        ),
        decision_curve=_decision_curve(),
        reclassification=ReclassificationEvidence(performed=False),
        feasibility_actionability=FeasibilityActionabilityEvidence(
            status=EvidenceStatus.SUPPORTED,
            result_changes_a_clinical_decision=True,
            resulting_action_supported_by_evidence=True,
            workflow_and_turnaround_feasible=True,
            failure_and_abstention_workflow_defined=True,
            downstream_harms_assessed=True,
            implementation_costs_and_resources_assessed=True,
            stakeholder_acceptability_assessed=True,
            scalability_assessed=True,
            evidence_ids=("implementation-evaluation-v1",),
        ),
        subgroup_generalizability=SubgroupGeneralizabilityEvidence(
            status=EvidenceStatus.SUPPORTED,
            subgroups_prespecified=True,
            clinically_relevant_subgroups_covered=True,
            subgroup_sample_sizes_reported=True,
            discrimination_or_accuracy_by_subgroup_reported=True,
            calibration_by_subgroup_reported_when_applicable=True,
            threshold_performance_by_subgroup_reported=True,
            device_site_and_era_heterogeneity_assessed=True,
            no_unacceptable_performance_or_harm_disparity=True,
            equity_implications_assessed=True,
            evidence_ids=("subgroup-generalizability-v1",),
        ),
        reporting_and_bias=reporting,
        prospective_impact=impact,
    )
    return dossier, registry


def _write_metadata_inputs(
    tmp_path: Path,
    dossier: ClinicalUtilityDossier,
    registry: EvidenceRegistry,
) -> tuple[Path, Path]:
    dossier_path = tmp_path / "dossier.json"
    registry_path = tmp_path / "registry.json"
    dossier_path.write_text(
        json.dumps(dossier.to_dict(), indent=2), encoding="utf-8"
    )
    registry_path.write_text(
        json.dumps(registry.to_dict(), indent=2), encoding="utf-8"
    )
    return dossier_path, registry_path


def test_internal_development_curve_is_not_clinical_utility_evidence() -> None:
    model = FrozenLogisticModel(
        model_id="internal-model",
        endpoint_id="endpoint",
        predictor_names=("marker",),
        means=(0.0,),
        standard_deviations=(1.0,),
        standardize=(False,),
        coefficients=(0.0, 1.0),
        baseline_model_id=None,
        training_subject_hash="train-hash",
        events_per_parameter=20.0,
    )
    rows = internal_decision_curve(
        pd.DataFrame({"label": [0, 1], "marker": [-1.0, 1.0]}),
        model,
        thresholds=(0.2, 0.5),
    )

    assert rows
    assert all(not row["clinical_utility_evidence_eligible"] for row in rows)
    assert all(
        row["evidence_scope"] == "internal_development_descriptive"
        for row in rows
    )
    assert all(not row["incremental_standard_care_comparator"] for row in rows)
    assert all(not row["paired_net_benefit_uncertainty_reported"] for row in rows)


def test_default_dossier_requires_non_tautological_freeze_attestation() -> None:
    registry = EvidenceRegistry(
        registry_id="empty-registry",
        development_site_id="development-site",
        created_at=_AT,
        created_by="registry-service",
        records=(),
    )
    dossier = ClinicalUtilityDossier(
        assessment_id="assessment-empty",
        candidate_id="candidate-empty",
        framework_scope=FrameworkScope.DIAGNOSTIC_ADAPTATION,
        intended_use=replace(
            _intended_use(),
            anticipated_benefits=(),
            anticipated_harms=(),
            clinically_actionable_thresholds=(),
        ),
        comparator=ComparatorDefinition(
            comparator_id="unspecified",
            description="Not yet selected",
        ),
    )

    assessment = evaluate_clinical_utility(dossier, registry)

    assert assessment.claim_level == ClaimLevel.HYPOTHESIS_ONLY
    assert not assessment.gate("scope_freeze_attested").passed
    assert assessment.to_dict()["locked_test_data_read"] is None


def test_diagnostic_adaptation_reaches_only_potential_without_impact() -> None:
    dossier, registry = _complete_bundle()
    assessment = evaluate_clinical_utility(dossier, registry)

    assert assessment.claim_level == ClaimLevel.POTENTIAL_CLINICAL_UTILITY
    assert not assessment.decision_process_utility_demonstrated
    assert not assessment.patient_benefit_demonstrated
    assert assessment.gate("aha_2026_metric_completeness").passed
    payload = assessment.to_dict()
    assert payload["aha_2026_direct_scope"] is False
    assert payload["aha_2026_metric_completeness"] is None
    assert payload["aha_2026_direct_compliance_claim_allowed"] is False


def test_process_utility_is_distinct_from_patient_benefit() -> None:
    dossier, registry = _complete_bundle(
        impact_design="prospective_controlled_impact",
        impact_outcome="clinical_management",
    )
    assessment = evaluate_clinical_utility(dossier, registry)

    assert assessment.claim_level == ClaimLevel.DECISION_PROCESS_UTILITY_DEMONSTRATED
    assert assessment.decision_process_utility_demonstrated
    assert not assessment.patient_benefit_demonstrated
    assert assessment.gate("prospective_decision_process_impact").passed
    assert not assessment.gate("prospective_patient_benefit").passed


def test_patient_benefit_requires_randomized_patient_relevant_impact() -> None:
    dossier, registry = _complete_bundle(
        impact_design="cluster_randomized",
        impact_outcome="patient_safety",
    )
    assessment = evaluate_clinical_utility(dossier, registry)

    assert assessment.claim_level == ClaimLevel.PATIENT_BENEFIT_DEMONSTRATED
    assert assessment.decision_process_utility_demonstrated
    assert assessment.patient_benefit_demonstrated
    assert assessment.to_dict()["blocking_gates_for_patient_benefit"] == []


def test_framework_scope_combinations_are_explicitly_separated() -> None:
    with pytest.raises(ValueError, match="Direct AHA incident-CVD"):
        ClinicalUtilityDossier(
            assessment_id="invalid",
            candidate_id="candidate",
            framework_scope=FrameworkScope.AHA_INCIDENT_CVD_RISK,
            intended_use=_intended_use(EvaluationFamily.DIAGNOSTIC_ACCURACY),
            comparator=_comparator(),
        )
    with pytest.raises(ValueError, match="Diagnostic adaptation"):
        ClinicalUtilityDossier(
            assessment_id="invalid",
            candidate_id="candidate",
            framework_scope=FrameworkScope.DIAGNOSTIC_ADAPTATION,
            intended_use=_intended_use(EvaluationFamily.PREDICTION_MODEL),
            comparator=_comparator(),
        )


def test_direct_aha_scope_has_machine_readable_completeness_gate() -> None:
    dossier, registry = _complete_bundle(
        framework_scope=FrameworkScope.AHA_INCIDENT_CVD_RISK
    )
    complete = evaluate_clinical_utility(dossier, registry)
    incomplete = evaluate_clinical_utility(
        replace(dossier, aha_metric_records=dossier.aha_metric_records[:-1]),
        registry,
    )

    assert complete.claim_level == ClaimLevel.POTENTIAL_CLINICAL_UTILITY
    assert complete.to_dict()["aha_2026_metric_completeness"] is True
    assert complete.to_dict()["aha_2026_framework_alignment_supported"] is True
    assert complete.to_dict()["aha_2026_direct_compliance_claim_allowed"] is False
    assert incomplete.claim_level == ClaimLevel.EXTERNAL_VALIDATION_SUPPORTED
    assert not incomplete.gate("aha_2026_metric_completeness").passed
    assert incomplete.to_dict()["aha_2026_framework_alignment_supported"] is False
    assert incomplete.to_dict()["aha_2026_direct_compliance_claim_allowed"] is False


def test_fake_evidence_id_is_rejected_not_merely_reported_missing() -> None:
    dossier, registry = _complete_bundle()
    dossier = replace(
        dossier,
        external_validation=replace(
            dossier.external_validation,
            evidence_ids=("fabricated-evidence-id",),
        ),
    )
    with pytest.raises(ValueError, match="cannot resolve.*fabricated-evidence-id"):
        evaluate_clinical_utility(dossier, registry)


def test_resolved_evidence_must_support_the_declared_domain() -> None:
    dossier, registry = _complete_bundle()
    dossier = replace(
        dossier,
        analytical_validity=replace(
            dossier.analytical_validity,
            evidence_ids=("comparator-protocol-v1",),
        ),
    )
    assessment = evaluate_clinical_utility(dossier, registry)

    assert assessment.claim_level == ClaimLevel.HYPOTHESIS_ONLY
    assert "traceable analytical evidence" in assessment.gate(
        "analytical_validity"
    ).missing_requirements


def test_registry_record_and_registry_hashes_detect_tampering() -> None:
    payload = _registry().to_dict()
    payload["records"][5]["site_id"] = "development-site"
    with pytest.raises(ValueError, match="failed integrity validation"):
        EvidenceRegistry.from_dict(payload)

    payload = _registry().to_dict()
    payload["created_by"] = "tampered"
    with pytest.raises(ValueError, match="registry failed integrity"):
        EvidenceRegistry.from_dict(payload)


def test_scope_freeze_hash_detects_post_freeze_threshold_change() -> None:
    dossier, registry = _complete_bundle()
    changed = replace(
        dossier,
        intended_use=replace(
            dossier.intended_use,
            clinically_actionable_thresholds=(0.1, 0.2, 0.4),
        ),
    )
    assessment = evaluate_clinical_utility(changed, registry)

    assert assessment.claim_level == ClaimLevel.HYPOTHESIS_ONLY
    assert "intended-use content hash bound" in assessment.gate(
        "scope_freeze_attested"
    ).missing_requirements


def test_scope_freeze_requires_attester_to_match_custody_assessor() -> None:
    dossier, registry = _complete_bundle()
    assert dossier.scope_freeze_attestation is not None
    forged = replace(
        dossier.scope_freeze_attestation,
        attested_by="unregistered-attester",
        attestation_sha256="",
    )
    assessment = evaluate_clinical_utility(
        replace(dossier, scope_freeze_attestation=forged),
        registry,
    )

    assert assessment.claim_level == ClaimLevel.HYPOTHESIS_ONLY
    assert "attester identity bound to custody evidence assessor" in assessment.gate(
        "scope_freeze_attested"
    ).missing_requirements


def test_continuous_nri_cannot_support_potential_utility() -> None:
    dossier, registry = _complete_bundle()
    dossier = replace(
        dossier,
        reclassification=ReclassificationEvidence(
            performed=True,
            method="continuous NRI",
            event_and_nonevent_reclassification_reported_separately=True,
            uncertainty_reported=True,
            evidence_ids=("categorical-nri-v1",),
        ),
    )
    assessment = evaluate_clinical_utility(dossier, registry)

    assert assessment.claim_level == ClaimLevel.EXTERNAL_VALIDATION_SUPPORTED
    assert not assessment.gate("reclassification_interpretable_or_not_used").passed


def test_internal_or_unscoped_decision_curve_cannot_be_imported() -> None:
    with pytest.raises(ValueError, match="cannot use internal development"):
        replace(_decision_curve(), evidence_scope="internal_development_descriptive")


def test_decision_curve_helper_rejects_harm_in_actionable_range() -> None:
    frame = pd.DataFrame(
        {
            "threshold_probability": [0.1, 0.2, 0.3],
            "candidate_net_benefit": [0.08, 0.01, 0.06],
            "comparator_net_benefit": [0.05, 0.04, 0.03],
            "treat_all_net_benefit": [0.03, 0.02, 0.01],
            "treat_none_net_benefit": [0.0, 0.0, 0.0],
            "delta_net_benefit_ci_low": [0.01, -0.04, 0.005],
        }
    )
    evidence = assess_decision_curve(
        frame,
        actionable_thresholds=(0.1, 0.2, 0.3),
        primary_threshold=0.2,
        evidence_ids=("external-dca-analysis-v1",),
        thresholds_prespecified=True,
        thresholds_clinically_actionable=True,
        evaluated_in_external_validation=True,
        minimum_margin_prespecified=True,
    )

    assert evidence.status == EvidenceStatus.FAILED
    assert not evidence.no_material_harm_across_range
    assert not evidence.primary_threshold_delta_ci_positive


def test_dossier_decoder_is_strict_per_section() -> None:
    dossier, _ = _complete_bundle()
    payload = dossier.to_dict()
    payload["analytical_validity"]["invented_flag"] = True
    with pytest.raises(ValueError, match="Unknown AnalyticalValidityEvidence"):
        clinical_utility_dossier_from_dict(payload)


def test_schema_rejects_missing_section_field_and_unknown_top_level() -> None:
    dossier, registry = _complete_bundle()
    payload = evaluate_clinical_utility(dossier, registry).to_dict()
    del payload["evidence_dossier"]["analytical_validity"]["repeatability_supported"]
    with pytest.raises(ValueError, match="schema violation"):
        validate_clinical_utility_payload(payload)

    payload = evaluate_clinical_utility(dossier, registry).to_dict()
    payload["unreviewed_claim"] = True
    with pytest.raises(ValueError, match="schema violation"):
        validate_clinical_utility_payload(payload)


def test_semantic_validator_rejects_gate_tampering() -> None:
    dossier, registry = _complete_bundle()
    payload = evaluate_clinical_utility(dossier, registry).to_dict()
    impact = next(
        gate
        for gate in payload["gates"]
        if gate["gate_id"] == "prospective_decision_process_impact"
    )
    impact["passed"] = True
    impact["status"] = "pass"
    impact["missing_requirements"] = []
    with pytest.raises(ValueError):
        validate_clinical_utility_payload(payload)


def test_writer_emits_schema_valid_atomically_published_bundle(tmp_path: Path) -> None:
    dossier, registry = _complete_bundle(
        impact_design="cluster_randomized",
        impact_outcome="patient_safety",
    )
    assessment = evaluate_clinical_utility(dossier, registry)
    outputs = write_clinical_utility_assessment(assessment, tmp_path / "utility")

    assert set(outputs) == {
        "assessment",
        "gates",
        "summary",
        "report",
        "manifest",
    }
    payload = json.loads(outputs["assessment"].read_text(encoding="utf-8"))
    schema = json.loads(
        utility_module.clinical_utility_schema_path().read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(payload)
    validate_clinical_utility_payload(payload)
    assert payload["locked_test_access_status"] == "attested_unread"
    assert payload["patient_benefit_demonstrated"] is True
    assert len(pd.read_parquet(outputs["gates"])) == len(assessment.gates)
    summary = pd.read_parquet(outputs["summary"])
    assert len(summary) == 1
    assert summary.iloc[0]["candidate_id"] == "candidate-v1"
    assert bool(summary.iloc[0]["patient_benefit_demonstrated"])
    assert "no clinical use" in summary.iloc[0]["disclaimer"].lower()
    report = outputs["report"].read_text(encoding="utf-8")
    assert "diagnostic_adaptation" in report
    for source in AUTHORITATIVE_SOURCES:
        assert source.url in report
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert "existing_internal_decision_curve_auto_import=false" in manifest["notes"]
    assert manifest["row_counts"]["summary"] == 1
    assert len(manifest["output_integrity"]) == 4


def test_writer_failure_preserves_previous_published_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dossier, registry = _complete_bundle()
    assessment = evaluate_clinical_utility(dossier, registry)
    output_root = tmp_path / "utility"
    outputs = write_clinical_utility_assessment(assessment, output_root)
    original = outputs["assessment"].read_bytes()

    def fail_parquet(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic staging failure")

    monkeypatch.setattr(utility_module, "atomic_write_parquet", fail_parquet)
    with pytest.raises(RuntimeError, match="synthetic staging failure"):
        write_clinical_utility_assessment(assessment, output_root)
    assert outputs["assessment"].read_bytes() == original
    assert not list(tmp_path.glob(".utility.*.staging"))


def test_metadata_only_cli_bypasses_mimic_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dossier, registry = _complete_bundle()
    dossier_path, registry_path = _write_metadata_inputs(tmp_path, dossier, registry)

    def forbid_config(*args: object, **kwargs: object) -> None:
        raise AssertionError("MIMIC configuration must not be loaded")

    monkeypatch.setattr(mimic_cli, "_load_config", forbid_config)
    output_root = tmp_path / "cli-output"
    result = mimic_cli.main(
        [
            "assess-clinical-utility",
            "--dossier",
            str(dossier_path),
            "--evidence-registry",
            str(registry_path),
            "--output",
            str(output_root),
        ]
    )

    assert result == 0
    assert (output_root / "clinical_utility_assessment.json").is_file()


def test_metadata_only_service_stage_is_explicit_and_not_guided(tmp_path: Path) -> None:
    dossier, registry = _complete_bundle()
    dossier_path, registry_path = _write_metadata_inputs(tmp_path, dossier, registry)
    output_root = tmp_path / "service-output"
    progress: list[str] = []

    outputs = run_service_stage(
        object(),  # the explicit metadata stage never reads project config
        "assess-clinical-utility",
        {
            "dossier_path": str(dossier_path),
            "evidence_registry_path": str(registry_path),
            "output_root": str(output_root),
        },
        progress=progress.append,
    )

    assert outputs["assessment"].is_file()
    assert progress[-1] == "Clinical-utility metadata assessment complete"


def test_verified_clinical_utility_bundle_is_reportable(tmp_path: Path) -> None:
    dossier, registry = _complete_bundle()
    assessment = evaluate_clinical_utility(dossier, registry)
    run_root = tmp_path / "run"
    outputs = write_clinical_utility_assessment(
        assessment,
        run_root / "clinical_utility",
    )

    resolved = resolve_clinical_utility_report_paths(run_root)

    assert resolved is not None
    assert resolved["assessment"] == outputs["assessment"]
    assert resolved["summary"] == outputs["summary"]


def test_tampered_clinical_utility_bundle_is_not_reportable(tmp_path: Path) -> None:
    dossier, registry = _complete_bundle()
    run_root = tmp_path / "run"
    outputs = write_clinical_utility_assessment(
        evaluate_clinical_utility(dossier, registry),
        run_root / "clinical_utility",
    )
    outputs["summary"].write_bytes(outputs["summary"].read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="integrity"):
        resolve_clinical_utility_report_paths(run_root)


def test_missing_utility_bundle_reports_hypothesis_only() -> None:
    tables = _load_report_tables(
        object(),
        "run",
        {
            "clinical_utility_summary": None,
            "clinical_utility_gates": None,
        },
    )

    summary = tables["clinical_utility_summary"].iloc[0]
    assert summary["claim_level"] == "hypothesis_only"
    assert not bool(summary["decision_process_utility_demonstrated"])
    assert not bool(summary["patient_benefit_demonstrated"])
