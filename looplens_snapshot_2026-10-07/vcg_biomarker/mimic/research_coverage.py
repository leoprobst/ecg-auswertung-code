"""Auditable mapping of researched mechanisms to measurements and data gaps.

This inventory does not turn a proposed analysis into an executed model. Only
explicit endpoint bindings below enter the existing exploratory model adapter.
No extra aliases or disease cutoffs are introduced.
"""

VERSION = "1.2.0"
ROOT = "research_pathology_round_20260913/"

# document, reusable measurements, permissible endpoints, unresolved analyses
CHAPTERS = (
    ("01_ISCHEMIA.md", ("qrs_st_directional_operator_commutator", "qrs_st_directional_operator_overlap"),
     ("acute_mi_high_specificity",), ("local_ionic_currents_require_independent_tissue_measurements",)),
    ("02_SCAR_REENTRY.md", ("qrs_early_late_directional_operator_overlap",),
     ("old_mi_icd",), ("scar_topology_and_reentry_require_CMR_or_mapping",)),
    ("03_HYPERTENSIVE_LVH.md", ("qrs_t_opposed_energy_fraction",),
     ("echo_lvh_strict", "concentric_remodeling_proxy"), ("fibrosis_and_hypertension_etiology_not_verified_by_geometry",)),
    ("04_HCM.md", ("t_direction_time_linear_coupling", "t_direction_time_quadratic_coupling"),
     ("hypertrophic_cardiomyopathy_icd",), ("genotype_CMR_and_arrhythmic_risk_not_verified_by_ICD",)),
    ("05_AMYLOID_INFILTRATION.md", ("qrs_local24ms_cancellation_fraction",),
     ("cardiac_amyloidosis",), ("cardiac_involvement_and_amyloid_typing_need_independent_reference",)),
    ("05B_IRON_OVERLOAD.md", ("qtc_fridericia_ms", "qtc_framingham_ms", "qtc_bazett_ms",
      "t_direction_time_linear_coupling", "t_direction_time_quadratic_coupling",
      "qrs_local24ms_cancellation_fraction", "qrs_fragmentation_high_frequency_energy", "terminal_qrs_hf_energy_fraction"),
     (), ("organ_specific_CMR_T2star_reference_missing", "electromechanical_discordance_requires_independent_clinical_target_validation")),
    ("05C_STORAGE_PREEXCITATION.md", ("delta_wave_present_flag", "delta_wave_frontal_axis_deg",
      "accessory_pathway_quadrant_estimate", "qrs_initial_terminal_energy_balance"),
     (), ("genotype_and_EPS_pathway_reference_missing", "PR_slope_requires_independently_qualified_beats_and_calibrated_error_inputs")),
    ("05D_POMPE.md", ("qtc_fridericia_ms", "t_direction_time_linear_coupling", "delta_wave_present_flag"),
     (), ("independent_pediatric_age_HR_PR_reference_missing", "enzyme_genotype_and_serial_structure_reference_missing")),
    ("06_DCM_SYSTOLIC_HF.md", ("qrs_early_late_directional_operator_overlap", "qrs_local24ms_cancellation_fraction"),
     ("cardiomyopathy_icd", "reduced_lvef_strict", "lv_dilatation_proxy"),
     ("DCM_genotype_and_substrate_not_identifiable_from_LVEF",)),
    ("06B_DCM_CELLULAR.md", ("beat_to_beat_qt_variability_ms", "qt_variability_index_vector"),
     ("cardiomyopathy_icd",), ("cellular_ROS_calcium_and_drug_mechanism_not_observable_in_surface_ECG",)),
    ("07_HFPEF_DIASTOLIC.md", ("p_duration_ms", "p_loop_terminal_area_fraction"),
     ("diastolic_relaxation_abnormality", "elevated_filling_pressure_proxy"),
     ("echo_rhythm_applicability_not_verified", "serial_echo_method_comparability_and_time_varying_confounding_not_verified")),
    ("07B_ATRIAL_CONDUCTION.md", ("p_duration_ms", "p_loop_terminal_area_fraction"),
     ("echo_lae",), ("invasive_pressure_and_conduction_mapping_missing",)),
    ("08_VOLUME_DIALYSIS.md", (), (),
     ("dialysis_session_identity_completion_and_independent_volume_reference_not_verified",)),
    ("08B_SERIAL_HEART_FAILURE.md", (), (),
     ("serial_echo_method_comparability_and_time_varying_confounding_not_verified",)),
    ("09_AORTIC_STENOSIS.md", ("qrs_t_opposed_energy_fraction", "qrs_early_late_directional_operator_overlap", "qrs_local24ms_cancellation_fraction"),
     ("moderate_aortic_stenosis_velocity", "aortic_stenosis"), ("flow_rhythm_and_anatomic_severity_context_required",
      "graded_AS_field_controls_are_mild_or_sclerotic_valves_not_healthy",)),
    ("09B_AORTIC_REGURGITATION.md", ("qrs_t_opposed_energy_fraction", "qrs_early_late_directional_operator_overlap", "qrs_local24ms_cancellation_fraction"),
     ("aortic_regurgitation",), ("mass_loading_and_regurgitant_volume_comparison_not_fully_connected",)),
    ("09C_ACUTE_AR_ROOT.md", ("qrs_st_directional_operator_commutator", "qrs_st_directional_operator_overlap"),
     (), ("acute_AR_onset_root_etiology_and_coronary_involvement_reference_missing",)),
    ("10_MITRAL_DISEASE.md", ("p_duration_ms", "p_loop_terminal_area_fraction"),
     ("mitral_stenosis", "mitral_regurgitation"), ("MVP_annular_disjunction_TDI_and_rhythm_context_missing",)),
    ("11_TRICUSPID_REGURGITATION.md", ("qrs_early_late_directional_operator_overlap", "qrs_local24ms_cancellation_fraction"),
     ("tricuspid_regurgitation",), ("pacing_transition_and_lead_induced_damage_not_verified",)),
    ("12_PULMONARY_HYPERTENSION.md", ("qrs_early_late_directional_operator_overlap", "qrs_local24ms_cancellation_fraction"),
     ("elevated_rvsp_or_pasp", "echo_rv_systolic_dysfunction"), ("invasive_PH_etiology_and_load_function_interaction_not_verified",)),
    ("13_PULMONARY_EMBOLISM.md", ("qrs_early_late_directional_operator_overlap", "qrs_local24ms_cancellation_fraction",
      "t_direction_time_linear_coupling", "t_direction_time_quadratic_coupling"),
     ("pulmonary_embolism_icd",), ("CTPA_confirmation_negative_imaging_controls_and_treatment_timing_missing",)),
    ("14_COPD_VOLUME_CONDUCTOR.md", ("qrs_early_late_directional_operator_overlap", "qrs_local24ms_cancellation_fraction"),
     ("copd_icd",), ("spirometry_CT_lung_volumes_BMI_smoking_and_anisotropy_context_required",)),
    ("15_ATRIAL_CARDIOMYOPATHY.md", ("p_duration_ms", "p_loop_terminal_area_fraction", "p_early_late_directional_operator_overlap"),
     ("echo_lae", "echo_rae"), ("amplified_P_duration_requires_validated_multilead_delineation", "histology_staging_and_incident_AF_not_inferred_from_atrial_size")),
)

REFERENCES = {
    "old_mi_icd": "administrative_history_not_CMR_scar_or_reentry",
    "echo_lvh_strict": "echo_geometry_not_hypertensive_etiology_or_histology",
    "concentric_remodeling_proxy": "echo_relative_wall_thickness_not_fibrosis",
    "hypertrophic_cardiomyopathy_icd": "administrative_HCM_not_genotype_or_CMR_adjudication",
    "cardiac_amyloidosis": "systemic_amyloidosis_ICD_not_confirmed_cardiac_amyloid",
    "cardiomyopathy_icd": "administrative_cardiomyopathy_not_DCM_etiology",
    "lv_dilatation_proxy": "echo_dilatation_not_DCM_etiology",
    "diastolic_relaxation_abnormality": "echo_relaxation_proxy_not_HFpEF_diagnosis",
    "elevated_filling_pressure_proxy": "echo_pressure_proxy_not_invasive_pressure_or_HFpEF",
    "echo_lae": "echo_LA_size_not_fibrosis_or_AF_prediction",
    "echo_rae": "echo_RA_size_not_fibrosis_or_AF_prediction",
    "pulmonary_embolism_icd": "administrative_PE_not_CTPA_confirmation",
}


def research_coverage_rows(markers):
    available = {row["marker_id"] for row in markers}
    result = []
    for chapter, names, endpoints, gaps in CHAPTERS:
        missing = set(names) - available
        if missing:
            raise ValueError(f"Unfrozen research measurements in {chapter}: {sorted(missing)}")
        serial = chapter.startswith(("08_", "08B_"))
        result.append({
            "origin": ROOT + chapter, "coverage_version": VERSION,
            "marker_ids": list(names), "endpoint_ids": list(endpoints),
            "measurement_status": "registered_reused_measurements" if names else "existing_serial_analysis",
            "serial_analysis_ids": ["serial_similarity_decomposition", "serial_arc_clock"] if serial else [],
            "available_context_helpers": (["mechanism_echo_context.compare_echo_pair_context"]
                if chapter in ("07_HFPEF_DIASTOLIC.md", "08B_SERIAL_HEART_FAILURE.md") else []),
            "production_analysis_adapter": ("extended_mechanism_workflow.run_extended_mechanisms"
                if chapter in ("05B_IRON_OVERLOAD.md", "05C_STORAGE_PREEXCITATION.md",
                               "07_HFPEF_DIASTOLIC.md", "08B_SERIAL_HEART_FAILURE.md") else None),
            "unimplemented_or_unverified_requirements": list(gaps),
            "analysis_complete": False, "clinical_claim_allowed": False,
        })
    return result


def additional_research_hypotheses(markers, existing, *, hypothesis_version):
    """One test per marker/endpoint, even when several chapters motivate it."""
    covered = {(row["marker_id"], row["endpoint_id"]) for row in existing}
    collected = {}
    for chapter in research_coverage_rows(markers):
        for marker in chapter["marker_ids"]:
            for endpoint in chapter["endpoint_ids"]:
                key = marker, endpoint
                if key in covered:
                    continue
                if endpoint not in REFERENCES:
                    raise ValueError(f"Missing researched endpoint reference: {endpoint}")
                row = collected.setdefault(key, {
                    "marker_id": marker, "endpoint_id": endpoint,
                    "origin": chapter["origin"], "research_origins": [],
                    "hypothesis_version": hypothesis_version,
                    "rationale": "reused_measurement_for_researched_mechanism",
                    "reference_target": REFERENCES[endpoint],
                    "reference_limitations": {"causal_mechanism_verified": False,
                                             "clinical_threshold_validated": False},
                    "post_outcome_hypothesis": True, "evidence_tier": "exploratory",
                    "expected_direction": "two_sided_no_prespecified_monotonic_direction",
                    "analysis_execution_status": "not_asserted_by_inventory",
                    "clinical_claim_allowed": False,
                    "required_comparisons": ["classical_ecg", "rhythm_and_pacing",
                                             "disease_specific_confounders_and_mediators"],
                    "required_measurement_context": [],
                    "claim_limit": REFERENCES[endpoint] + "_association_only",
                })
                row["research_origins"].append(chapter["origin"])
                row["required_measurement_context"] = sorted(set(
                    row["required_measurement_context"] + ["reference_temporal_alignment",
                    "transform_and_delineation_sensitivity"] + chapter["unimplemented_or_unverified_requirements"]
                ))
    return list(collected.values())
