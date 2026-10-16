"""Endpoint/reference contracts for treatment research, not treatment recommendations.

Mere procedures, ECG annotations or diagnosis codes cannot serve as proof of
occlusion or causal benefit. This contract is independent of cohort names.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta


VERSION = "1.1.0"


@dataclass(frozen=True)
class TimeInterval:
    earliest: datetime
    latest: datetime

    def __post_init__(self):
        if self.earliest.tzinfo is None or self.latest.tzinfo is None:
            raise ValueError("timezone_required")
        if self.latest < self.earliest:
            raise ValueError("inverted_time_interval")

    @classmethod
    def calendar_day(cls, day, timezone):
        start = datetime.combine(day, datetime.min.time(), tzinfo=timezone)
        return cls(start, datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone))


def temporal_relation(first: TimeInterval, second: TimeInterval):
    if first.latest < second.earliest:
        return "before"
    if first.earliest > second.latest:
        return "after"
    return "ambiguous"


def available_at_decision(result_available: TimeInterval | None, decision: TimeInterval):
    # Specimen collection/order time must not be passed in place of result availability.
    return result_available is not None and result_available.latest <= decision.earliest


def reference_label(*, observed_state, reference_type, evidence_id, independent,
                    verified, result_available=None, decision=None, design="association", as_predictor=False):
    reasons = []
    if observed_state not in ("positive", "negative"):
        reasons.append("unknown_reference_state")
    if not evidence_id or not verified:
        reasons.append("actual_verified_evidence_missing")
    if not independent or reference_type in ("ecg_annotation", "machine_report", "administrative_code"):
        reasons.append("not_independent_disease_reference")
    if design not in ("association", "prediction", "physiological_response"):
        reasons.append("unsupported_design")
    # A future outcome is legitimate in prognosis; only *predictor* information
    # must be available at the decision. Do not exclude all future endpoints.
    if as_predictor and (decision is None or not available_at_decision(result_available, decision)):
        reasons.append("not_available_at_decision")
    return {"state": "unknown" if reasons else observed_state, "reasons": reasons,
            "reference_type": reference_type, "evidence_id": evidence_id,
            "clinical_claim_allowed": False}


# Explicit semantic fields, not table names. No automatic mapping from ICD or
# a paper's supplementary dataset to unavailable local measured fields.
PROFILES = {
    "urgent_angiography": ("ecg_time", "angiography_time_interval", "culprit_adjudication", "reference_verification", "hemodynamics", "symptoms"),
    "pci": ("ecg_time", "angiography_time_interval", "culprit_adjudication", "pci_performed", "pci_indication", "reference_verification"),
    "pretreatment": ("ecg_time", "drug_name", "actual_administration_time", "dose", "contraindication", "bleeding_followup", "ischemic_followup"),
    "crt_response": ("lvef_before", "lvesv_before", "lvesv_after", "echo_times", "pacing_mode", "pacing_percentage", "guideline_therapy"),
    "icd_prognosis": ("vt_vf_event_time", "followup_end", "death_cause", "icd_therapy", "lvef"),
    "af_monitoring": ("atrial_marker_quality", "af_event_time", "monitoring_duration", "left_atrial_measurement"),
    "electrolyte_drug_response": ("ecg_time_before", "ecg_time_after", "drug_name", "actual_administration_time", "dose", "sodium", "potassium", "renal_function", "ph", "rhythm"),
    "pulmonary_embolism": ("ctpa_or_vq_result", "reference_time", "rv_measurement", "hemodynamics"),
    "amyloid_hcm_lvh": ("lv_mass", "echo_time", "amyloid_specific_reference", "hcm_specific_reference", "blood_pressure_history"),
    "dialysis_fluid_response": ("dialysis_start", "dialysis_end", "ecg_time_before", "ecg_time_after", "ultrafiltration_volume", "weight_before", "weight_after", "electrolytes", "hemodynamics"),
}

PROFILES.update({
    "af_rhythm_response": ("ecg_time", "treatment_time", "rhythm_strategy", "recurrence_time", "monitoring_duration"),
    "af_stroke_bleeding": ("ecg_time", "actual_administration_time", "anticoagulant", "stroke_time", "bleeding_time", "followup_end"),
    "heart_failure_drug_response": ("ecg_time", "drug_name", "actual_administration_time", "dose", "lvef_before", "lvef_after", "echo_times", "renal_function", "followup_end"),
    "structural_intervention_response": ("ecg_time", "procedure_time", "procedure_indication", "structural_reference", "followup_reference", "followup_end"),
})


@dataclass(frozen=True)
class TreatmentProfile:
    profile_id: str
    design: str
    treatment: str = ""
    comparator: str = ""
    benefit_endpoint: str = ""
    harm_endpoint: str = ""
    time_zero: str = ""
    horizon: str = ""
    protocol_hash: str = ""

    def readiness(self, *, verified_fields=(), record_level_verified=False,
                  overlap_verified=False, confounding_plan_verified=False):
        if self.profile_id not in PROFILES:
            raise ValueError("unknown_treatment_profile")
        allowed = {"diagnosis", "prognosis", "physiological_response", "treatment_benefit_research"}
        if self.design not in allowed:
            raise ValueError("unsupported_treatment_design")
        missing = sorted(set(PROFILES[self.profile_id]) - set(verified_fields))
        reasons = ["missing_field:" + name for name in missing]
        if record_level_verified is not True:
            reasons.append("record_level_reference_not_verified")
        if self.design == "prognosis":
            for name in ("benefit_endpoint", "time_zero", "horizon"):
                if not getattr(self, name):
                    reasons.append("missing_protocol:" + name)
        if self.design == "treatment_benefit_research":
            for name in ("treatment", "comparator", "benefit_endpoint", "harm_endpoint", "time_zero", "horizon", "protocol_hash"):
                if not getattr(self, name):
                    reasons.append("missing_protocol:" + name)
            if self.treatment == self.comparator:
                reasons.append("distinct_treatment_strategies_required")
            if overlap_verified is not True:
                reasons.append("positivity_not_verified")
            if confounding_plan_verified is not True:
                reasons.append("confounding_plan_not_verified")
        return {**asdict(self), "required_fields": list(PROFILES[self.profile_id]),
                "status": "research_ready" if not reasons else "non_estimable",
                "reasons": reasons, "clinical_claim_allowed": False,
                "performed_treatment_is_benefit_label": False}


def tracking_requirements(verified_fields=None):
    """A field counts only after an explicit semantic adapter has verified it.

    This is a readiness contract, never a claim of actual record-level availability.
    """
    fields = set(verified_fields or ())
    return {name: {"required_fields": list(required),
                   "missing_fields": sorted(set(required) - fields) if verified_fields is not None else [],
                   "unverified_fields": sorted(set(required) - fields),
                   "status": ("semantic_availability_not_assessed" if verified_fields is None else
                              "requires_record_level_validation" if set(required) <= fields else "not_estimable_missing_reference_fields"),
                   "clinical_claim_allowed": False}
            for name, required in PROFILES.items()}
