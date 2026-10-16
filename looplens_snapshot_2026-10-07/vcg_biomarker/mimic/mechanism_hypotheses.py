"""Explicit post-pilot research questions, not additional clinical criteria."""
from .run_log import profiled_phase

VERSION = "1.11.0"

# The family is closed. 77 marker x endpoint bindings, 308 planned tests
# (two development splits, two models). A new binding changes the family
# every existing test is corrected against, so it needs its own written
# rationale first and then this number -- never a silent append.
FROZEN_HYPOTHESIS_COUNT = 77

# Measured on the A117 10k pilot (2026-09-18), 9,905 recordings, Spearman:
# ``qrs_t_opposed_energy_fraction`` correlates 0,90 with the mean spatial
# QRS-T angle. The redundancy stage groups only above 0,99, so the family
# would count the two as separate findings. They are not; every result row
# of the opposed-energy bindings says so. A synthetic counterexample separates
# the formulas, the cohort does not.
MEASURED_NEAR_ALIASES = {
    "qrs_t_opposed_energy_fraction": {
        "near_alias_of": "spatial_qrs_t_angle_deg",
        "pilot_spearman_rho": 0.90,
        "measured_on": "A117-2026-09-16-736108-pilot-10000",
    },
}


@profiled_phase("postprocess.mechanism_annotation")
def annotate_mechanism_hypotheses(frame):
    """Attach provenance only to existing screening rows, preserving statistics.

    A row may itself be non-estimable; presence does not assert a successful
    model, replication or completion of the planned comparison analyses.
    """
    import json
    from ..core.biomarkers import BiomarkerRegistry

    result = frame.copy()
    column = "mechanism_hypothesis_json"
    result[column] = None
    if not {"feature_id", "endpoint_id"} <= set(result.columns):
        return result
    rows = mechanism_hypothesis_rows(
        [{"marker_id": name} for name in BiomarkerRegistry.available()]
    )
    lookup = {
        (row["marker_id"], row["endpoint_id"]): json.dumps(
            {**row, "analysis_execution_status": "screening_row_present_not_confirmation"},
            sort_keys=True, allow_nan=False,
        ) for row in rows
    }
    result[column] = [lookup.get((marker, endpoint))
                      for marker, endpoint in zip(result.feature_id, result.endpoint_id)]
    return result


def mechanism_hypothesis_rows(markers):
    """Bind planned valve questions to available frozen implementations.

    This is a provenance contract, not evidence of an executed analysis or
    permission to select markers from observed outcomes.
    """
    available = {row["marker_id"] for row in markers}
    rationales = {
        "qrs_t_opposed_energy_fraction": "opposed_repolarization_energy_as_remodeling_not_valve_area",
        "qrs_early_late_directional_operator_overlap": "temporal_depolarization_organization_not_local_fibrosis",
        "qrs_local24ms_cancellation_fraction": "local_temporal_cancellation_not_hidden_myocardial_mass",
    }
    endpoints = {
        "aortic_stenosis": "reported_echo_moderate_or_worse_vs_graded_mild_or_sclerotic_controls",
        "aortic_stenosis_severe": "reported_echo_severe",
        "moderate_aortic_stenosis_velocity": "vmax_threshold_3_m_per_s_not_anatomic_exclusion",
        "severe_aortic_stenosis_velocity": "vmax_threshold_4_m_per_s_not_anatomic_exclusion",
        "aortic_regurgitation": "reported_echo_regurgitation_moderate_or_worse_not_regurgitant_volume",
        "aortic_regurgitation_severe": "reported_echo_regurgitation_severe_not_regurgitant_volume",
    }
    missing = rationales.keys() - available
    if missing:
        raise ValueError(f"Unfrozen mechanism hypotheses: {sorted(missing)}")
    rows = [
        {"marker_id": marker, "endpoint_id": endpoint, "rationale": rationale,
         "reference_target": reference, "hypothesis_version": VERSION,
         "reference_limitations": (
             {"onset_verified": False, "cause_verified": False,
              "acute_disease_diagnosis_supported": False,
              "regurgitant_volume_measured_by_endpoint": False}
             if endpoint.startswith("aortic_regurgitation") else
             {"anatomic_severity_exclusion_supported": False,
              "flow_and_rhythm_applicability_verified": False}
             if endpoint.endswith("_velocity") else
             {"valve_area_measured_by_endpoint": False}),
         "origin": ("research_pathology_round_20260913/09B_AORTIC_REGURGITATION.md"
                    if endpoint.startswith("aortic_regurgitation") else
                    "research_pathology_round_20260913/09_AORTIC_STENOSIS.md"),
         "post_outcome_hypothesis": True, "evidence_tier": "exploratory",
         "expected_direction": "two_sided_no_prespecified_monotonic_direction",
         "analysis_execution_status": "not_asserted_by_inventory",
         "clinical_claim_allowed": False,
         "required_comparisons": (["classical_ecg", "lv_dilatation_and_mass",
                                   "rhythm_and_conduction_context", "coexisting_valve_disease"]
                                  if endpoint.startswith("aortic_regurgitation") else
                                  ["classical_ecg", "hypertrophy", "rhythm_and_conduction_context"]),
         "claim_limit": ("association_not_AR_specificity_or_regurgitant_volume_or_fibrosis_estimation"
                         if endpoint.startswith("aortic_regurgitation") else
                         "association_not_AS_specificity_or_valve_area_estimation")}
        for marker, rationale in rationales.items()
        for endpoint, reference in endpoints.items()
    ]
    rows = (rows + _mitral_hypothesis_rows(available)
            + _tricuspid_hypothesis_rows(available) + _rv_hypothesis_rows(available)
            + _copd_hypothesis_rows(available) + _priority_hypothesis_rows(available)
            + _priority_cardiac_rows(available))
    from .research_coverage import additional_research_hypotheses
    rows = rows + additional_research_hypotheses(
        markers, rows, hypothesis_version=VERSION)
    for row in rows:
        alias = MEASURED_NEAR_ALIASES.get(row["marker_id"])
        if alias is not None:
            row["measured_near_alias"] = dict(alias)
    if len(rows) != FROZEN_HYPOTHESIS_COUNT:
        raise ValueError(
            f"Mechanism hypothesis family has {len(rows)} bindings; the frozen "
            f"family has {FROZEN_HYPOTHESIS_COUNT}. A new binding needs its own "
            "rationale and an explicit change of FROZEN_HYPOTHESIS_COUNT."
        )
    return rows


def _priority_cardiac_rows(available):
    """Bind existing measurements to cardiac targets without inventing diagnoses."""
    groups = {
        "acute_mi_high_specificity": (
            "qrs_st_directional_operator_commutator",
            "qrs_st_directional_operator_overlap"),
        "reduced_lvef_strict": (
            "qrs_t_opposed_energy_fraction",
            "qrs_early_late_directional_operator_overlap",
            "qrs_local24ms_cancellation_fraction"),
    }
    if {marker for markers in groups.values() for marker in markers} - available:
        raise ValueError("Unfrozen priority cardiac markers")
    rows = []
    for endpoint, markers in groups.items():
        coronary = endpoint == "acute_mi_high_specificity"
        for marker in markers:
            rows.append({
                "marker_id": marker, "endpoint_id": endpoint,
                "hypothesis_version": VERSION,
                "origin": "research_pathology_round_20260913/PRIORITY_MARKER_IMPLEMENTATION.md",
                "rationale": "depolarization_ST_organization" if coronary else "ventricular_remodeling_organization",
                "reference_target": "troponin_ICD_same_day_PCI_composite_not_adjudicated_infarction" if coronary else "echo_LVEF_below_40_not_all_heart_failure",
                "reference_limitations": {"causal_mechanism_verified": False,
                                         "clinical_threshold_validated": False},
                "post_outcome_hypothesis": True, "evidence_tier": "exploratory",
                "expected_direction": "two_sided_no_prespecified_monotonic_direction",
                "analysis_execution_status": "not_asserted_by_inventory",
                "clinical_claim_allowed": False,
                "required_comparisons": ["classical_ecg", "rhythm_and_pacing",
                    "noncoronary_injury_and_treatment_selection" if coronary else "valve_disease_mass_and_loading"],
                "required_measurement_context": ["QRS_ST_quality" if coronary else "QRS_T_quality",
                    "reference_temporal_alignment", "transform_and_delineation_sensitivity"],
                "claim_limit": "composite_association_not_infarct_adjudication_or_prospective_detection" if coronary else "systolic_function_association_not_HFpEF_or_fibrosis_diagnosis",
            })
    return rows


def _priority_hypothesis_rows(available):
    groups = {
        "t_energy_temporal_concentration": (
            "hypokalemia_near_ecg", "severe_hypokalemia_near_ecg",
            "hyperkalemia_near_ecg", "severe_hyperkalemia_near_ecg"),
        "p_early_late_directional_operator_overlap": ("echo_lae", "echo_rae"),
    }
    if groups.keys() - available:
        raise ValueError("Unfrozen priority mechanism markers")
    rows = []
    for marker, endpoints in groups.items():
        atrial = marker.startswith("p_")
        for endpoint in endpoints:
            rows.append({
                "marker_id": marker, "endpoint_id": endpoint,
                "hypothesis_version": VERSION,
                "origin": "research_pathology_round_20260913/PRIORITY_MARKER_IMPLEMENTATION.md",
                "rationale": "atrial_temporal_organization" if atrial else "repolarization_energy_occupancy",
                "reference_target": "echo_size_not_histology" if atrial else "near_ECG_potassium_not_intracellular_concentration",
                "reference_limitations": {"causal_mechanism_verified": False,
                                         "clinical_threshold_validated": False},
                "post_outcome_hypothesis": True, "evidence_tier": "exploratory",
                "expected_direction": "two_sided_no_prespecified_monotonic_direction",
                "analysis_execution_status": "not_asserted_by_inventory",
                "clinical_claim_allowed": False,
                "required_comparisons": ["classical_ecg", "rhythm_and_pacing",
                    "valve_and_loading_context" if atrial else "renal_function_pH_other_electrolytes_medication"],
                "required_measurement_context": ["P_segment_quality" if atrial else "T_segment_quality",
                    "reference_temporal_alignment", "transform_and_delineation_sensitivity"],
                "claim_limit": "size_association_not_fibrosis_or_AF_prediction" if atrial else
                               "potassium_association_not_etiology_or_treatment_rule",
            })
    return rows


def _copd_hypothesis_rows(available):
    candidates = {
        "qrs_early_late_directional_operator_overlap": "relative_activation_organization_under_common_rotation",
        "qrs_local24ms_cancellation_fraction": "normalized_local_cancellation_under_common_gain",
    }
    missing = candidates.keys() - available
    if missing:
        raise ValueError(f"Unfrozen COPD mechanism hypotheses: {sorted(missing)}")
    return [{
        "marker_id": marker, "endpoint_id": "copd_icd", "rationale": rationale,
        "hypothesis_version": VERSION,
        "origin": "research_pathology_round_20260913/14_COPD_VOLUME_CONDUCTOR.md",
        "reference_target": "encounter_ICD_not_spirometry_or_CT_emphysema",
        "reference_limitations": {"post_bronchodilator_obstruction_verified": False,
                                  "CT_emphysema_verified": False,
                                  "hyperinflation_measured": False},
        "post_outcome_hypothesis": True, "evidence_tier": "exploratory",
        "expected_direction": "two_sided_no_prespecified_monotonic_direction",
        "analysis_execution_status": "not_asserted_by_inventory",
        "clinical_claim_allowed": False,
        "required_comparisons": ["classical_ecg_axes_and_voltage", "BMI_and_smoking",
                                 "spirometry_and_lung_volumes", "RV_load_and_left_heart_disease"],
        "required_measurement_context": ["QRS_segment_quality", "rhythm_and_pacing",
                                         "lead_placement_and_transform_sensitivity"],
        "claim_limit": "administrative_association_not_COPD_diagnosis_or_conductor_source_separation",
    } for marker, rationale in candidates.items()]


def _rv_hypothesis_rows(available):
    candidates = {
        "qrs_early_late_directional_operator_overlap": "activation_organization_not_isolated_RV_source",
        "qrs_local24ms_cancellation_fraction": "local_cancellation_not_pressure_measurement",
    }
    references = {
        "elevated_rvsp_or_pasp": "derived_echo_systolic_pressure_threshold_not_invasive_PH",
        "echo_rv_dilation": "echo_diameter_or_reported_RV_size",
        "echo_rv_systolic_dysfunction": "TAPSE_or_reported_RV_function_not_invasive_contractility",
    }
    missing = candidates.keys() - available
    if missing:
        raise ValueError(f"Unfrozen RV mechanism hypotheses: {sorted(missing)}")
    return [{
        "marker_id": marker, "endpoint_id": endpoint, "rationale": rationale,
        "hypothesis_version": VERSION,
        "origin": "research_pathology_round_20260913/12_PULMONARY_HYPERTENSION.md",
        "reference_target": reference,
        "reference_limitations": {"invasive_PH_confirmed": False,
                                  "PAH_etiology_confirmed": False,
                                  "isolated_RV_electrical_source_measured": False},
        "post_outcome_hypothesis": True, "evidence_tier": "exploratory",
        "expected_direction": "two_sided_no_prespecified_monotonic_direction",
        "analysis_execution_status": "not_asserted_by_inventory",
        "clinical_claim_allowed": False,
        "required_comparisons": ["classical_ecg", "left_sided_disease",
                                 "rhythm_and_conduction_context",
                                 "load_function_interaction_separate_from_main_effect"],
        "required_measurement_context": ["QRS_segment_quality", "pacing_and_capture_status",
                                         "echo_method_and_temporal_alignment"],
        "claim_limit": "association_not_invasive_PH_PAH_or_cellular_APD_estimation",
    } for marker, rationale in candidates.items() for endpoint, reference in references.items()]


def _tricuspid_hypothesis_rows(available):
    candidates = {
        "qrs_early_late_directional_operator_overlap": "activation_organization_not_RV_localization",
        "qrs_local24ms_cancellation_fraction": "local_cancellation_not_regurgitant_volume",
    }
    missing = candidates.keys() - available
    if missing:
        raise ValueError(f"Unfrozen tricuspid mechanism hypotheses: {sorted(missing)}")
    return [{
        "marker_id": marker, "endpoint_id": "tricuspid_regurgitation" + suffix,
        "rationale": rationale, "hypothesis_version": VERSION,
        "origin": "research_pathology_round_20260913/11_TRICUSPID_REGURGITATION.md",
        "reference_target": "reported_echo_severe" if suffix else "reported_echo_moderate_or_worse",
        "reference_limitations": {"cause_verified": False,
                                  "lead_induced_damage_verified": False,
                                  "invasive_pressure_measured_by_endpoint": False},
        "post_outcome_hypothesis": True, "evidence_tier": "exploratory",
        "expected_direction": "two_sided_no_prespecified_monotonic_direction",
        "analysis_execution_status": "not_asserted_by_inventory",
        "clinical_claim_allowed": False,
        "required_comparisons": ["classical_ecg", "RV_size_and_function",
                                 "pulmonary_pressure_context", "left_sided_valve_disease",
                                 "rhythm_and_conduction_context"],
        "required_measurement_context": ["QRS_segment_quality", "pacing_and_capture_status",
                                         "intrinsic_versus_paced_RBBB_morphology"],
        "claim_limit": "severity_association_not_pressure_cause_or_lead_damage",
    } for marker, rationale in candidates.items() for suffix in ("", "_severe")]


def _mitral_hypothesis_rows(available):
    """Reuse measured features without pretending MR is an MVP reference."""
    atrial = {
        "p_duration_ms": "atrial_activation_duration_not_LA_pressure",
        "p_loop_terminal_area_fraction": "terminal_swept_area_share_not_PTFV1_or_TDI_delay",
    }
    ventricular = {
        "qrs_early_late_directional_operator_overlap": "temporal_activation_organization_not_papillary_localization",
        "qrs_local24ms_cancellation_fraction": "local_cancellation_not_MVP_specific_fibrosis",
    }
    missing = (atrial.keys() | ventricular.keys()) - available
    if missing:
        raise ValueError(f"Unfrozen mitral mechanism hypotheses: {sorted(missing)}")
    rows = []
    for valve in ("mitral_stenosis", "mitral_regurgitation"):
        candidates = atrial if valve == "mitral_stenosis" else {**atrial, **ventricular}
        for suffix in ("", "_severe"):
            for marker, rationale in candidates.items():
                is_atrial = marker in atrial
                rows.append({
                    "marker_id": marker, "endpoint_id": valve + suffix,
                    "rationale": rationale, "hypothesis_version": VERSION,
                    "origin": "research_pathology_round_20260913/10_MITRAL_DISEASE.md",
                    "reference_target": "reported_echo_severe" if suffix else "reported_echo_moderate_or_worse",
                    "reference_limitations": {
                        "cause_verified": False, "LA_pressure_measured_by_endpoint": False,
                        "electromechanical_delay_measured_by_endpoint": False,
                        **({"prolapse_and_annular_disjunction_verified": False,
                            "arrhythmogenic_MVP_diagnosis_supported": False}
                           if valve == "mitral_regurgitation" else {}),
                    },
                    "post_outcome_hypothesis": True, "evidence_tier": "exploratory",
                    "expected_direction": "two_sided_no_prespecified_monotonic_direction",
                    "analysis_execution_status": "not_asserted_by_inventory",
                    "clinical_claim_allowed": False,
                    "required_comparisons": ["classical_ecg", "rhythm_and_conduction_context",
                                             "coexisting_valve_disease",
                                             "LA_volume_mediator_sensitivity" if is_atrial else "LV_function_and_mass"],
                    "required_measurement_context": (
                        ["discernible_P_wave", "qualified_atrial_rhythm", "P_segment_quality"]
                        if is_atrial else ["QRS_segment_quality", "pacing_and_bundle_branch_context"]
                    ),
                    "claim_limit": "severity_association_not_cause_pressure_TDI_delay_or_arrhythmia_prediction",
                })
    return rows
