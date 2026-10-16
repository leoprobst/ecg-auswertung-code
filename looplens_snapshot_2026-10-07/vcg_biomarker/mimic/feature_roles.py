from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


ROLE_CANDIDATE = "candidate_biomarker"
ROLE_CLASSICAL = "classical_ecg_baseline"
ROLE_CONTEXT = "ecg_context_comparator"
ROLE_QC = "methodological_qc_control"
ROLE_REVIEW = "manual_review_heuristic"
ROLE_COMPOSITE = "exploratory_composite"
ROLE_ALIAS = "redundant_alias"
ROLE_COHORT = "cohort_only"
ROLE_NON_SCALAR = "non_scalar_foundation"
ROLE_RETIRED = "retired_invalid_proxy"
FEATURE_ROLE_POLICY_VERSION = "2.5.0"


# Three markers of the catalog's ``Beat-Variabilitaet`` family that describe
# beat-to-beat *morphology* rather than rate or rhythm. The family maps to the
# ``rhythm`` category, and the rhythm rule below routes that whole category to
# rate/rhythm context -- which is right for the RR measures beside them and
# wrong for these, whose subject is the shape of the complex. Routed to
# context, they belong to no panel, and a marker in no panel is never computed:
# the extraction asks only for the panels' feature ids.
#
# They are exempted from that rule here rather than by changing the catalog's
# family-to-category map, which would move the RR markers with them. They enter
# as post-outcome discovery, not as ordinary candidates: they were never
# measured, so admitting them now is a decision taken with the interim results
# in hand, and the curated panels have fixed sizes -- a marker admitted to the
# candidate pool displaces one already in the sentinel panel.
BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "qrs_alternans_vector",
        "t_wave_alternans_vector",
        "qrs_t_angle_variability",
    }
)

METHODOLOGICAL_CONTROL_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "fiducial_confidence",
        "lead_to_vcg_consistency_score",
        "quality_score",
        "signal_quality_score",
        "vcg_trust_score",
        "p_fiducial_confidence",
        "pr_fiducial_confidence",
        "qrs_fiducial_confidence",
        "t_fiducial_confidence",
        "qrst_fiducial_confidence",
        "rhythm_context_confidence",
        "rr_cv",
        "p_wave_coverage",
        "detector_conflict_fraction",
        "einthoven_relation_error_rms",
        "kors_vs_pca_axis_difference_deg",
        "chest_lead_misplacement_probability_proxy",
        "loop_missingness_risk",
        "loop_fiducial_margin",
        "loop_confidence_weighted_value",
        "manual_review_required",
    }
)

MANUAL_REVIEW_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "qrs_loop_abnormal_score",
        "t_loop_abnormal_score",
        "p_loop_abnormal_score",
        "st_loop_abnormal_score",
    }
)

COHORT_ONLY_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "loop_quality_stratified_delta",
        "sex_age_adjusted_residual_candidate",
    }
)

NON_SCALAR_FOUNDATION_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "context_temporal_research_json",
        "context_measurement_capsule_json",
        "context_measurement_sensitivity_json",
        "delineation_lead",
        "delineation_method",
        "spatial_magnitude_m_t",
        "vcg_axes_xyz",
        "spatial_ventricular_gradient_vector",
        "external_norm/pathology_template_bank_available_flag",
    }
)

# These names claimed an algorithm that was not actually implemented. They are
# kept in the catalog for provenance, but must not enter a scientific panel.
RETIRED_INVALID_PROXY_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "persistent_homology_loop_lifetime",
        "topological_signature_distance_qrs_t",
    }
)

EXPLORATORY_COMPOSITE_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "acute_ischemia_vector_score",
        "lbbb_like_vector_score",
        "rbbb_like_terminal_vector_score",
        "lvh_vector_score / rvh_vector_score",
        "scar_vector_index",
        "qt_prolongation/shortening_surrogate",
        "medication_sensitive_repolarization_score",
        "pacemaker_capture_vector_signature",
        "p_wave_mvp_vector_score",
    }
)

RHYTHM_CONTEXT_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "heart_rate",
        "heart_rate_bpm",
        "rr_interval_ms",
        "rr_variance_ms2",
        "rr_entropy/irregularity_index",
        "rr_mean_ms/sd_ms/rmssd_ms/pnn50",
        "atrial_fibrillation_rr_surrogate",
    }
)

# Exact implementation aliases found by deterministic synthetic checks and by a
# 20,000-record train-only correlation audit. Only the canonical feature is
# tested, preventing duplicate FDR tests and duplicate score components.
FEATURE_CANONICAL_ALIASES: dict[str, str] = {
    "spatial_qrs_t_angle_deg_catalog": "spatial_qrs_t_angle_deg",
    "spatial_mean_qrs_t_angle_deg": "spatial_qrs_t_angle_deg",
    "spatial_qrs_t_angle_mean": "spatial_qrs_t_angle_deg",
    "depolarization_repolarization_mismatch": "spatial_qrs_t_angle_deg",
    "qrs_axis_to_t_axis_angle_deg": "spatial_qrs_t_angle_deg",
    "max_qrs_to_max_t_angle_deg": "spatial_peak_qrs_t_angle_deg",
    "spatial_qrs_t_angle_peak": "spatial_peak_qrs_t_angle_deg",
    "qrs_t_loop_shape_distance": "qrs_t_procrustes_distance",
    "qrs_t_angle_lability_index": "qrs_t_angle_variability",
    "p_duration": "p_duration_ms",
    "pr_interval": "pr_interval_ms",
    "qrs_duration": "qrs_duration_ms",
    "qt_interval": "qt_interval_ms",
    "electric_systole_ms": "qt_interval_ms",
    "qtc_bazett": "qtc_bazett_ms",
    "qtc_fridericia": "qtc_fridericia_ms",
    "p_qrs_loop_plane_angle": "p_qrs_loop_plane_angle_deg",
    "p_t_loop_plane_angle": "p_t_loop_plane_angle_deg",
    "qrs_area_to_t_area_angle_deg_axis_relation": (
        "qrs_area_to_t_area_angle_deg"
    ),
    "st_qrs_discordance_angle_deg": "st_qrs_discordance_deg",
    "qrs_st_angle_deg": "st_qrs_discordance_deg",
    "p_loop_max_point_distance": "p_loop_diameter",
    "qrs_loop_max_point_distance": "qrs_loop_diameter",
    "qrst_loop_max_point_distance": "qrst_loop_diameter",
    "st_loop_max_point_distance": "st_loop_diameter",
    "t_loop_max_point_distance": "t_loop_diameter",
    "p_velocity_integral": "p_loop_path_length",
    "qrs_velocity_integral": "qrs_loop_path_length",
    "qrst_velocity_integral": "qrst_loop_path_length",
    "st_velocity_integral": "st_loop_path_length",
    "t_velocity_integral": "t_loop_path_length",
    "p_loop_return_to_baseline_distance": "p_loop_start_end_distance",
    "qrs_loop_return_to_baseline_distance": "qrs_loop_start_end_distance",
    "qrst_loop_return_to_baseline_distance": "qrst_loop_start_end_distance",
    "st_loop_return_to_baseline_distance": "st_loop_start_end_distance",
    "t_loop_return_to_baseline_distance": "t_loop_start_end_distance",
    "p_mean_vector": "p_vector_magnitude",
    "p_loop_centroid_shift_magnitude": "p_vector_magnitude",
    "st_loop_centroid_shift_magnitude": "st_vector_magnitude",
    "p_wave_terminal_force_vector_vcg": "p_loop_terminal_force_vector",
    "ectopic_beat_vector_deviation": "qrs_internal_cluster_distance_median",
    "loop_interbeat_reproducibility": "qrs_internal_cluster_distance_median",
    "qrs_loop_folding_index": "loop_total_self_intersections",
    "qrs_loop_dihedral_angle_deg": "loop_dihedral_angle_mean",
    "curvature_torsion_phase_coupling": "loop_curvature_torsion_ratio",
    "curvature_velocity_coupling": "loop_curvature_speed_correlation",
    "repolarization_front_spread_velocity_proxy": "t_path_length_per_time",
    "activation_front_spread_velocity_proxy": "qrs_path_length_per_time",
    "qrs_loop_notch_burden": "qrs_velocity_peak_count",
    "topological_signature_distance_qrs_t": "qrs_t_hausdorff_distance",
    "early_late_t_angle_deg": "t_loop_upstroke_downstroke_angle",
    "local_plane_drift_index": "loop_sliding_plane_angle_mean",
    "loop_backtracking_fraction": "trajectory_reversal_fraction",
    "loop_convexity_ratio": "qrs_loop_actual_area_to_convex_hull_ratio",
    "loop_ellipse_major_axis": "qrs_loop_width_major",
    "loop_ellipse_minor_axis": "qrs_loop_width_minor",
    "loop_graph_bottleneck_distance": "loop_min_nonlocal_distance",
    "loop_mean_curvature": "qrs_curvature_mean",
    "loop_self_intersection_xy": (
        "qrs_loop_frontal_xy_self_intersection_count"
    ),
    "loop_self_intersection_xz": (
        "qrs_loop_horizontal_xz_self_intersection_count"
    ),
    "loop_self_intersection_yz": (
        "qrs_loop_sagittal_yz_self_intersection_count"
    ),
    "p_loop_double_rotation_index": (
        "p_loop_frontal_xy_rotation_change_count"
    ),
    "pr_vector": "pr_drift_magnitude",
    "qrs_area_t_area_crt_response_surrogate": (
        "qrs_loop_area_t_loop_area_ratio"
    ),
    "st_loop_drift": "st_loop_start_end_distance",
}

# These hypotheses were added after inspecting MIMIC development results. They
# are deliberately Discovery-only and cannot be promoted into a frozen panel by
# changing a GUI setting.
POST_OUTCOME_DISCOVERY_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "ndat_v1_ms", "ndat_v2_ms", "ndat_v3_ms", "ndat_v4_ms",
        "ndat_v5_ms", "ndat_v6_ms", "ndat_precordial_spread_ms",
        "qrs_total_turning_angle_deg",
        "qrs_turning_angle_p95_deg",
        "qrs_terminal_energy_fraction_20ms",
        "qrs_initial_terminal_energy_balance",
        "qrs_t_normalized_hausdorff_distance",
        "qrs_t_normalized_dtw_distance",
        "qrs_t_mirror_phase_angle_median_deg",
        "qrs_t_mirror_phase_angle_p90_deg",
        "qrs_t_mirror_discordant_fraction",
        "svg_cancellation_index",
        "t_terminal_path_fraction",
        "t_terminal_direction_dispersion_deg",
        "st_t_transition_angle_deg",
        "qrs_beat_area_axis_coherence",
        "t_beat_area_axis_coherence",
        "beatwise_svg_direction_dispersion_deg",
        "u_peak_vector_magnitude_mv",
        "u_loop_area_vector_magnitude_mv2",
        "u_to_t_peak_magnitude_ratio",
        "t_u_area_vector_angle_deg",
        "u_loop_planarity",
        "t_u_transition_angle_deg",
        "secondary_morphology_beat_fraction",
        "secondary_morphology_centroid_distance_mv",
        "secondary_morphology_mean_coupling_ratio",
        "secondary_morphology_frontal_axis_delta_deg",
        "t_amplitude_beat_to_beat_variability",
        "st_level_beat_to_beat_variability",
        "qt_beat_to_beat_variability_ms",
        "secondary_morphology_frontal_axis_deg",
        "secondary_morphology_compensatory_pause_ratio",
        "secondary_morphology_t_amplitude_beat_to_beat_variability",
        "secondary_morphology_qt_beat_to_beat_variability_ms",
        "secondary_morphology_st_level_beat_to_beat_variability",
        "secondary_morphology_qt_delta_ms",
        "secondary_morphology_qrs_duration_delta_ms",
        "secondary_morphology_qrs_area_vector_magnitude_delta",
        "post_ectopic_t_amplitude_delta_mv",
        "post_ectopic_t_axis_delta_deg",
        "secondary_morphology_coupling_variability_cv",
        "secondary_morphology_common_denominator_fit",
        "primary_morphology_maximum_deflection_index",
        "secondary_morphology_maximum_deflection_index",
        "secondary_morphology_maximum_deflection_index_delta",
        "sum_absolute_qrst_integral_mv_ms",
        "t_loop_biphasic_index",
        "primary_morphology_qrs_octant_id",
        "secondary_morphology_qrs_octant_id",
        "qrs_axis_kors_vs_dower_delta_deg",
        "t_axis_kors_vs_dower_delta_deg",
        "rr_poincare_sd1_ms",
        "rr_poincare_sd2_ms",
        "qrs_magnitude_poincare_sd1_mv",
        "qrs_magnitude_poincare_sd2_mv",
        "t_magnitude_poincare_sd1_mv",
        "t_magnitude_poincare_sd2_mv",
        "within_record_qrs_axis_drift_deg",
        "within_record_qrs_amplitude_drift_mv",
        "within_record_qtc_drift_ms",
        "fragmented_qrs_present_flag",
        "fragmented_qrs_lead_count",
        "qrs_multiscale_entropy",
        "t_multiscale_entropy",
        "qrs_higuchi_fractal_dimension",
        "t_higuchi_fractal_dimension",
        "qrs_xy_cross_correlation_peak_lag_ms",
        "qrs_xy_cross_correlation_peak_value",
        "qrs_time_reversal_asymmetry",
        "t_time_reversal_asymmetry",
        "narrow_complex_tachycardia_pattern_flag",
        "wide_complex_tachycardia_pattern_flag",
        "accelerated_idioventricular_pattern_flag",
        "junctional_rhythm_pattern_flag",
        "precordial_rs_absent_flag",
        "precordial_rs_duration_max_ms",
        "precordial_concordance_flag",
        "avr_vi_vt_ratio",
        "av_dissociation_supports_vt_flag",
        "independent_atrial_rate_bpm",
        "independent_atrial_ventricular_rate_ratio",
        "rp_prime_interval_ms",
        "retrograde_p_qrs_axis_angle_deg",
        "p_loop_qrs_loop_axis_angle_deg",
        "p_loop_t_loop_axis_angle_deg",
        "beatwise_p_loop_axis_stability_deg",
        "terminal_qrs_pseudo_r_prime_v1_flag",
        "terminal_qrs_pseudo_s_inferior_flag",
        "beatwise_qrs_axis_alternation_score",
        "qrs_axis_progressive_rotation_deg_per_beat",
        "distinct_p_morphology_cluster_count",
        "multifocal_atrial_tachycardia_pattern_flag",
        "delta_wave_present_flag",
        "delta_wave_frontal_axis_deg",
        "accessory_pathway_quadrant_estimate",
        "flutter_av_block_ratio_estimate",
    }
)


@dataclass(frozen=True)
class FeatureProfile:
    feature_id: str
    canonical_feature_id: str
    role: str
    modality: str
    conceptual_family: str
    scientific_tier: str
    core_sentinel_eligible: bool
    discovery_eligible: bool
    combination_eligible: bool
    post_outcome_hypothesis: bool
    reason: str = ""

    @property
    def is_canonical(self) -> bool:
        return self.feature_id == self.canonical_feature_id

    def to_dict(self) -> dict:
        return asdict(self)



_POST_OUTCOME_MODULE_CACHE: Optional[frozenset[str]] = None


def _post_outcome_module_feature_ids() -> frozenset[str]:
    """Feature ids from modules written after development results were seen.

    Resolved lazily and cached: importing the marker modules at module import
    time would create a cycle with the biomarker registry.
    """

    global _POST_OUTCOME_MODULE_CACHE
    if _POST_OUTCOME_MODULE_CACHE is not None:
        return _POST_OUTCOME_MODULE_CACHE
    names: set[str] = set()
    from ..core.biomarkers.advanced_vcg_biomarkers import ADVANCED_SPECS
    names.update(spec.name for spec in ADVANCED_SPECS if spec.family == "A108 explorative Messrobustheit")
    from ..core.biomarkers.pathology_mechanism_biomarkers import SPECS as MECHANISM_SPECS
    names.update(spec.name for spec in MECHANISM_SPECS)
    try:
        from ..core.biomarkers.electrophysiology_markers import (
            ELECTROPHYSIOLOGY_SPECS,
        )

        names.update(str(spec.name) for spec in ELECTROPHYSIOLOGY_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        from ..core.biomarkers.discovery_v2_biomarkers import (
            DISCOVERY_V2_SPECS,
        )

        names.update(str(spec.name) for spec in DISCOVERY_V2_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        from ..core.biomarkers.discovery_v4_biomarkers import (
            DISCOVERY_V4_SPECS,
        )

        names.update(str(spec.name) for spec in DISCOVERY_V4_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        from ..core.biomarkers.discovery_v3_biomarkers import (
            DISCOVERY_V3_SPECS,
        )

        names.update(str(spec.name) for spec in DISCOVERY_V3_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The TQ-window and strip markers. They were written after the
        # development run showed that the P segment is unusable in 61 % of
        # recordings and, worse, in 96,6 % of the recordings that show atrial
        # fibrillation -- so they are hypotheses formed from results, and they
        # belong in Discovery Reserve like every other such module.
        #
        # This is not bookkeeping. The curated panels have fixed sizes and are
        # filled by a family-balanced round robin, so a marker admitted to the
        # candidate pool does not merely join the sentinel panel -- it pushes
        # another marker out of it. Registering these 13 without this entry
        # displaced eight sentinel markers, ``p_loop_linearity`` among them,
        # which would have removed an atrial marker in the name of adding
        # atrial markers.
        from ..core.biomarkers.segment_free_biomarkers import (
            SEGMENT_FREE_SPECS,
        )

        names.update(str(spec.name) for spec in SEGMENT_FREE_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The beat-to-beat morphology variability markers. Written after the
        # development run had shown what the P segment's missingness costs, so
        # -- like the TQ-window module -- hypotheses formed from results.
        from ..core.biomarkers.morphology_variability_biomarkers import (
            MORPHOLOGY_VARIABILITY_SPECS,
        )

        names.update(str(spec.name) for spec in MORPHOLOGY_VARIABILITY_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The three markers taken from named 2024-2026 publications. Written
        # after the development run, so post-outcome like every other module
        # in that position.
        from ..core.biomarkers.literature_2026_biomarkers import (
            LITERATURE_2026_SPECS,
        )

        names.update(str(spec.name) for spec in LITERATURE_2026_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The five markers from the extended 2026 search. Same position as
        # every other module written after development results were seen.
        from ..core.biomarkers.literature_2026b_biomarkers import (
            LITERATURE_2026B_SPECS,
        )

        names.update(str(spec.name) for spec in LITERATURE_2026B_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The activation-sequence markers. Written after the propagation
        # literature was searched and the registry measured, so post-outcome
        # like every module in this position.
        from ..core.biomarkers.propagation_sequence_biomarkers import (
            PROPAGATION_SEQUENCE_SPECS,
        )

        names.update(str(spec.name) for spec in PROPAGATION_SEQUENCE_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The loop-parametrisation markers. Written after the spatial-velocity
        # literature was searched and the registry measured, so post-outcome
        # like every module in this position.
        from ..core.biomarkers.loop_parametrisation_biomarkers import (
            LOOP_PARAMETRISATION_SPECS,
        )

        names.update(str(spec.name) for spec in LOOP_PARAMETRISATION_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The thesis-derived markers. Each follows from a thesis written down
        # with its refuting observation first; post-outcome like every module
        # in this position.
        from ..core.biomarkers.thesis_derived_biomarkers import (
            THESIS_DERIVED_SPECS,
        )

        names.update(str(spec.name) for spec in THESIS_DERIVED_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The acute-ischaemia thesis markers; post-outcome like the others.
        from ..core.biomarkers.ischemia_thesis_biomarkers import (
            ISCHEMIA_THESIS_SPECS,
        )

        names.update(str(spec.name) for spec in ISCHEMIA_THESIS_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The physiology/pathology thesis markers of 23.09.2026 (T6, T10,
        # T15); post-outcome like the others.
        from ..core.biomarkers.physiology_thesis_biomarkers import (
            PHYSIOLOGY_THESIS_SPECS,
        )

        names.update(str(spec.name) for spec in PHYSIOLOGY_THESIS_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The treatment-decision thesis markers; post-outcome like the others.
        from ..core.biomarkers.decision_thesis_biomarkers import (
            DECISION_THESIS_SPECS,
        )

        names.update(str(spec.name) for spec in DECISION_THESIS_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The uniform-slow-conduction markers of the third research round.
        from ..core.biomarkers.conduction_uniformity_biomarkers import (
            CONDUCTION_UNIFORMITY_SPECS,
        )

        names.update(str(spec.name) for spec in CONDUCTION_UNIFORMITY_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The QRS-T coupling extensions. Written after the A117 pilot showed
        # the shape-coupling family as the strongest of the run; post-outcome
        # like every module in this position.
        from ..core.biomarkers.qrs_t_coupling_extension_biomarkers import (
            QRS_T_COUPLING_EXTENSION_SPECS,
        )

        names.update(str(spec.name) for spec in QRS_T_COUPLING_EXTENSION_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    try:
        # The circular projections. Written after a probe over the existing
        # partitions measured that a quarter of the marker-endpoint pairs
        # involving an azimuth separate better under a faithful scale -- so a
        # hypothesis formed from results, and post-outcome for the same reason
        # every module in this position is. The displacement rule matters here
        # more than anywhere: forty-eight markers admitted to the candidate
        # pool would push forty-eight others out of the fixed-size sentinel
        # panel, which is not a trade any measurement supports.
        from ..core.biomarkers.circular_projection_biomarkers import (
            CIRCULAR_PROJECTION_SPECS,
        )

        names.update(str(spec.name) for spec in CIRCULAR_PROJECTION_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    names.update(BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS)
    try:
        # The rotation invariants and the families beside them. Written after
        # the audit measured that three catalog markers changed value when the
        # coordinate frame turned, and that the octant sector markers inherit
        # the same weakness -- so, like the TQ-window module, hypotheses formed
        # from results. The same displacement rule applies: the curated panels
        # have fixed sizes, and a marker admitted to the candidate pool pushes
        # another one out of the sentinel panel rather than joining it.
        from ..core.biomarkers.invariant_biomarkers import INVARIANT_SPECS

        names.update(str(spec.name) for spec in INVARIANT_SPECS)
    except Exception:  # noqa: BLE001 - policy must not break on import order
        pass
    _POST_OUTCOME_MODULE_CACHE = frozenset(names)
    return _POST_OUTCOME_MODULE_CACHE


def canonical_feature_id(feature_id: object) -> str:
    normalized = str(feature_id or "").strip()
    visited: set[str] = set()
    while normalized in FEATURE_CANONICAL_ALIASES and normalized not in visited:
        visited.add(normalized)
        normalized = FEATURE_CANONICAL_ALIASES[normalized]
    return normalized


def feature_profile(
    feature_id: object,
    *,
    family: Optional[str] = None,
    category: Optional[str] = None,
    unit: Optional[str] = None,
) -> FeatureProfile:
    normalized = str(feature_id or "").strip()
    canonical = canonical_feature_id(normalized)
    definition = _definition_metadata(normalized)
    family = str(family if family is not None else definition.get("family", ""))
    category = str(
        category if category is not None else definition.get("category", "")
    )
    unit = str(unit if unit is not None else definition.get("unit", ""))
    family_folded = family.casefold()
    category_folded = category.casefold()
    unit_folded = unit.casefold()
    # Markers conceived after inspecting development results are
    # post-outcome hypotheses by construction, whatever their physiological
    # motivation. Listing them individually was already error-prone; the
    # electrophysiology gap batch is therefore recognised by its registry
    # source, so a new marker in that module cannot accidentally enter the
    # frozen Core or Sentinel panels.
    post_outcome = (
        normalized.startswith("ndat_")
        or
        normalized in POST_OUTCOME_DISCOVERY_FEATURE_IDS
        or normalized in _post_outcome_module_feature_ids()
    )

    if normalized.startswith("classical_"):
        return _profile(
            normalized,
            canonical,
            ROLE_CLASSICAL,
            "classical_ecg",
            "classical_ecg_rule",
            "baseline",
            core=True,
            discovery=False,
            combination=False,
            reason="Versioned classical ECG criterion; comparator, not a novel VCG marker.",
        )
    if normalized in RETIRED_INVALID_PROXY_FEATURE_IDS:
        return _profile(
            normalized,
            canonical,
            ROLE_RETIRED,
            "vcg",
            "retired_topology_proxy",
            "retired",
            reason=(
                "The historical name overclaimed the implemented calculation; "
                "retained only for provenance."
            ),
        )
    if normalized in COHORT_ONLY_FEATURE_IDS:
        return _profile(
            normalized,
            canonical,
            ROLE_COHORT,
            "cohort",
            "normative_or_stratified",
            "cohort_only",
            reason="Requires a fitted cohort reference and is not a per-record scalar.",
        )
    if (
        normalized in NON_SCALAR_FOUNDATION_FEATURE_IDS
        or unit_folded in {"categorical", "text", "vector", "array"}
        or any(token in normalized for token in ("<", ">", "*"))
        or "/" in normalized and normalized not in RHYTHM_CONTEXT_FEATURE_IDS
    ):
        return _profile(
            normalized,
            canonical,
            ROLE_NON_SCALAR,
            _infer_modality(normalized, category_folded),
            "foundation_or_bundle",
            "metadata",
            reason="Foundation, bundled output or placeholder rather than one scalar hypothesis.",
        )
    if (
        normalized in METHODOLOGICAL_CONTROL_FEATURE_IDS
        or family_folded in {"qualitaet/elektroden", "signalbasis"}
        or category_folded == "quality"
        or normalized.startswith(("residual_per_lead_", "jackknife_influence_"))
    ):
        return _profile(
            normalized,
            canonical,
            ROLE_QC,
            "quality_control",
            "signal_and_measurement_quality",
            "control",
            reason="Methodological quality/control variable; audit and stratify, never claim as pathology marker.",
        )
    if (
        normalized in MANUAL_REVIEW_FEATURE_IDS
        or family_folded == "manuelle review"
        or category_folded == "review"
    ):
        return _profile(
            normalized,
            canonical,
            ROLE_REVIEW,
            "review",
            "manual_review_summary",
            "review_only",
            reason="Heuristic review summary; not an independently validated biomarker.",
        )
    if normalized != canonical:
        return _profile(
            normalized,
            canonical,
            ROLE_ALIAS,
            _infer_modality(normalized, category_folded),
            _conceptual_family(canonical, family_folded),
            "alias",
            reason=f"Exact or semantic duplicate of {canonical}.",
        )
    if normalized in EXPLORATORY_COMPOSITE_FEATURE_IDS or (
        family_folded == "pathophysiologische surrogate"
        and (
            "score" in normalized
            or "surrogate" in normalized
            or "signature" in normalized
        )
    ):
        return _profile(
            normalized,
            canonical,
            ROLE_COMPOSITE,
            _infer_modality(normalized, category_folded),
            _conceptual_family(normalized, family_folded),
            "discovery",
            discovery=True,
            combination=False,
            reason="Hand-built exploratory composite; cannot be nested in an automatically optimized score.",
        )
    if (
        normalized in RHYTHM_CONTEXT_FEATURE_IDS or category_folded == "rhythm"
    ) and normalized not in BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS:
        return _profile(
            normalized,
            canonical,
            ROLE_CONTEXT,
            "ecg_context",
            "rate_and_rhythm_context",
            "comparator",
            combination=False,
            reason="Rate/rhythm context is reported separately from morphology and VCG discovery.",
        )

    modality = _infer_modality(normalized, category_folded)
    tier = "post_outcome_discovery" if post_outcome else "candidate"
    return _profile(
        normalized,
        canonical,
        ROLE_CANDIDATE,
        modality,
        _conceptual_family(normalized, family_folded),
        tier,
        core=not post_outcome,
        discovery=True,
        combination=not post_outcome,
        post_outcome=post_outcome,
        reason=(
            "Post-development physiology hypothesis; independent replication required."
            if post_outcome
            else ""
        ),
    )


def feature_analysis_role(feature_id: object) -> str:
    return feature_profile(feature_id).role


def is_candidate_biomarker(feature_id: object) -> bool:
    return feature_analysis_role(feature_id) == ROLE_CANDIDATE


def is_core_sentinel_candidate(feature_id: object) -> bool:
    return feature_profile(feature_id).core_sentinel_eligible


def is_discovery_candidate(feature_id: object) -> bool:
    return feature_profile(feature_id).discovery_eligible


def is_combination_candidate(feature_id: object) -> bool:
    return feature_profile(feature_id).combination_eligible


def conceptual_feature_family(feature_id: object) -> str:
    return feature_profile(feature_id).conceptual_family


def endpoint_feature_role(feature_id: object, endpoint_id: object) -> str:
    """Identify same-source comparator measurements for endpoint reporting.

    QT/QTc-derived values remain useful technical comparators, but testing a
    machine/ECG QTc measurement against a QTc endpoint is not independent
    clinical validation and must be labeled accordingly.
    """

    feature = str(feature_id or "").casefold()
    endpoint = str(endpoint_id or "").casefold()
    if endpoint.startswith(("qtc_", "qt_")) and (
        feature.startswith(("qt_", "qtc_", "jt_"))
        or feature in {
            "electric_systole_ms",
            "tpeak_tend_ms",
            "classical_qtc_bazett",
            "classical_qtc_fridericia",
            "classical_qtc_framingham",
            "classical_qtc_gt_480",
            "classical_qtc_gt_500",
            "classical_jtc_fridericia",
        }
    ):
        return "same_ecg_source_comparator"
    return feature_analysis_role(feature_id)


def _profile(
    feature_id: str,
    canonical: str,
    role: str,
    modality: str,
    conceptual_family: str,
    tier: str,
    *,
    core: bool = False,
    discovery: bool = False,
    combination: bool = False,
    post_outcome: bool = False,
    reason: str = "",
) -> FeatureProfile:
    return FeatureProfile(
        feature_id=feature_id,
        canonical_feature_id=canonical,
        role=role,
        modality=modality,
        conceptual_family=conceptual_family,
        scientific_tier=tier,
        core_sentinel_eligible=bool(core),
        discovery_eligible=bool(discovery),
        combination_eligible=bool(combination),
        post_outcome_hypothesis=bool(post_outcome),
        reason=reason,
    )


def _definition_metadata(feature_id: str) -> dict[str, str]:
    try:
        from ..core.biomarkers import BiomarkerRegistry

        definition = BiomarkerRegistry.definition(feature_id)
    except (ImportError, ValueError):
        return {}
    return {
        "family": definition.family,
        "category": definition.category,
        "unit": definition.unit,
    }


def _infer_modality(feature_id: str, category: str) -> str:
    normalized = feature_id.casefold()
    if normalized in {"v6_r_peak_time_ms", "v6_v1_interpeak_interval_ms"}:
        # Input is the native ECG, not an orthogonal VCG transformation.
        return "ecg"
    if normalized.startswith("classical_"):
        return "classical_ecg"
    if category in {"ecg", "rhythm"} or normalized.startswith(
        ("qt_", "qtc_", "jt_", "pr_", "p_duration", "qrs_duration", "heart_rate")
    ):
        return "ecg"
    if "report" in normalized:
        return "machine_report"
    return "vcg"


def _conceptual_family(feature_id: str, catalog_family: str = "") -> str:
    name = feature_id.casefold()
    if name.startswith("classical_"):
        return "classical_ecg_rule"
    if "qrs_t" in name or "qrs-to-t" in name or "depolarization_repolarization" in name:
        return "qrs_t_coupling"
    if name.startswith("svg_") or "ventricular_gradient" in name:
        return "spatial_ventricular_gradient"
    if name.startswith(("qt_", "qtc_", "jt_")) or "tpeak" in name:
        return "repolarization_intervals"
    if any(
        token in name
        for token in (
            "variability",
            "instability",
            "alternans",
            "interbeat",
            "beatwise",
            "coherence",
            # A beat-to-beat template distance is a variability measure whatever
            # it is called. Without these tokens the scale-free counterpart of
            # ``p_loop_template_instability`` would land in a different family
            # from the marker it exists to be compared against, and the
            # redundancy stage compares within families. "similarity" also
            # catches "dissimilarity"; no marker outside this group carries
            # either token.
            "similarity",
            "bimodality",
        )
    ):
        return "beat_variability"
    if any(
        token in name
        for token in ("pca", "svd", "lambda", "nondipolar", "residual_energy")
    ):
        return "dipolarity_pca"
    if any(
        token in name
        for token in ("spectral", "wavelet", "entropy", "frequency", "power_band")
    ):
        return "spectral_wavelet"
    if any(
        token in name
        for token in (
            "topolog",
            "recurrence",
            "self_intersection",
            "winding",
            "hole",
        )
    ):
        return "topology_and_recurrence"
    if any(
        token in name
        for token in (
            "velocity",
            "acceleration",
            "jerk",
            "curvature",
            "torsion",
            "turning_angle",
        )
    ):
        return "kinematics"
    if any(token in name for token in ("magnitude", "voltage", "amplitude")):
        return "magnitude_voltage"
    # Test QRST before ST: ``qrst_loop`` contains the substring ``st_loop``.
    if "qrst_loop" in name or name.startswith("qrst_"):
        return "integrated_qrst_geometry"
    if "p_loop" in name or name.startswith(("p_", "pr_")):
        return "atrial_activation"
    if "st_loop" in name or name.startswith("st_"):
        return "st_ischemia_geometry"
    if "t_loop" in name or name.startswith("t_") or "repolarization" in name:
        return "t_repolarization"
    if "qrs_loop" in name or "qrs_" in name or "initial_qrs" in name or "terminal_qrs" in name:
        return "qrs_activation_geometry"
    if catalog_family:
        normalized_family = (
            catalog_family.casefold()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )
        family_aliases = {
            "kinematik": "kinematics",
            "kinematische_landmarken": "kinematics",
            "beat_variabilitaet": "beat_variability",
            "pca_svd": "dipolarity_pca",
            "experimentell_topologisch": "topology_and_recurrence",
            "ventricular_gradient": "spatial_ventricular_gradient",
            "qrs_t_repolarisation": "qrs_t_coupling",
            "zeitintervalle": "ecg_intervals",
            "raeumliche_magnitude": "magnitude_voltage",
        }
        return family_aliases.get(normalized_family, normalized_family)
    return "other_vcg"


__all__ = [
    "COHORT_ONLY_FEATURE_IDS",
    "EXPLORATORY_COMPOSITE_FEATURE_IDS",
    "FEATURE_CANONICAL_ALIASES",
    "FEATURE_ROLE_POLICY_VERSION",
    "FeatureProfile",
    "MANUAL_REVIEW_FEATURE_IDS",
    "METHODOLOGICAL_CONTROL_FEATURE_IDS",
    "NON_SCALAR_FOUNDATION_FEATURE_IDS",
    "POST_OUTCOME_DISCOVERY_FEATURE_IDS",
    "RETIRED_INVALID_PROXY_FEATURE_IDS",
    "canonical_feature_id",
    "conceptual_feature_family",
    "endpoint_feature_role",
    "feature_analysis_role",
    "feature_profile",
    "is_candidate_biomarker",
    "is_combination_candidate",
    "is_core_sentinel_candidate",
    "is_discovery_candidate",
]
