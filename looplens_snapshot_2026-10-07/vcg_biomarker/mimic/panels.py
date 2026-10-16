from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..core.biomarkers import BiomarkerRegistry
from .feature_roles import (
    canonical_feature_id,
    conceptual_feature_family,
    feature_profile,
    is_core_sentinel_candidate,
    is_discovery_candidate,
)
from .models import FeaturePanelSpec, PanelTier
from .storage import atomic_write_json, content_hash


PANEL_VERSION = "mimic-v2.11.0-reference-scores"

# The panel set a full study extracts and screens, named once.
#
# It was a four-place literal repeated in the CLI, the job builder and six
# service entries, and it also decides the store path
# (``features/<panel_ids joined by _>/development``), so a tuple that differs
# in one caller does not select different markers -- it selects a different,
# empty directory. ``classical_criteria`` is appended rather than inserted so
# the existing prefix stays readable in the path.
STUDY_PANEL_IDS: tuple[str, ...] = (
    "primary_core",
    "sentinel",
    "discovery_reserve",
    "classical_criteria",
)

NON_RECORD_SCALAR_FEATURE_IDS: frozenset[str] = frozenset(
    {
        # Cohort-level or normative-model outputs cannot be computed per ECG.
        "loop_quality_stratified_delta",
        "sex_age_adjusted_residual_candidate",
    }
)

CLASSICAL_FEATURE_IDS: tuple[str, ...] = (
    "classical_st_elevation_contiguous",
    "classical_st_elevation_contiguous_mm",
    "classical_st_depression_contiguous",
    "classical_st_depression_contiguous_mm",
    "classical_t_inversion_contiguous",
    "classical_posterior_mi_pattern",
    "classical_wellens_ecg_pattern",
    "classical_de_winter_ecg_pattern",
    "classical_sokolow_lyon",
    "classical_cornell_voltage",
    "classical_cornell_product",
    "classical_peguero_lo_presti",
    "classical_pathological_q_waves",
    "classical_qs_complexes",
    "classical_poor_r_wave_progression",
    "classical_right_axis_deviation",
    "classical_left_axis_deviation",
    "classical_dominant_r_v1",
    "classical_dominant_s_v5_v6",
    "classical_s1q3t3_pattern",
    "classical_p_pulmonale",
    "classical_p_duration_ge_120",
    "classical_p_terminal_force_v1",
    "classical_bifid_p_wave",
    "classical_qtc_bazett",
    "classical_qtc_fridericia",
    "classical_qtc_framingham",
    "classical_qtc_gt_480",
    "classical_qtc_gt_500",
    "classical_jtc_fridericia",
    "classical_tpeak_tend",
    "classical_peaked_t_wave_heuristic",
    "classical_qrs_widening_ge_120",
    "classical_pr_prolongation_gt_200",
    "classical_p_wave_attenuation",
    "classical_hyperkalemia_ecg_heuristic",
    "classical_low_voltage_qrs",
    "classical_rbbb_pattern",
    "classical_lbbb_pattern",
)


# The criteria, as criteria. Every ``classical_<x>`` above is the *continuous*
# quantity a criterion is cut from -- an axis in degrees, an interval in
# milliseconds -- and the extraction has always written the criterion's own
# verdict beside it under ``classical_<x>_positive``. Not one of those columns
# belonged to a panel, so no criterion has ever been evaluated as a criterion,
# although the project exists to compare new markers against exactly those.
#
# The continuous columns cannot stand in for them. Several are the same number
# under two names, because two criteria are cut from one quantity at different
# thresholds. Measured over 5.540 to 5.851 records of the current store, these
# pairs are byte-identical:
#
#   classical_right_axis_deviation   == classical_left_axis_deviation
#       (both are the frontal axis in degrees, -180..179)
#   classical_pr_prolongation_gt_200 == classical_first_degree_av_block
#   classical_qtc_gt_480             == classical_qtc_gt_500
#                                    == classical_qtc_fridericia
#   classical_qrs_widening_ge_120    == qrs_duration_ms
#
# So "right axis deviation" and "left axis deviation" are numerically the same
# marker, and any finding reported under either name says nothing about the
# direction. The ``_positive`` columns are the ones that differ, because that
# is where the threshold and its sign live.
CLASSICAL_CRITERION_FEATURE_IDS: tuple[str, ...] = tuple(
    f"{feature_id}_positive" for feature_id in CLASSICAL_FEATURE_IDS
)


# Published reference scores and indices (audit 2026-09-23, section 5.1).
# They join the criteria panel, not the primary core: the core is fixed at 80
# markers, and every classical entry there displaces a VCG marker. The
# continuous value is screened for all of them; the verdict only where the
# source gives a cut-off (upright T in aVR, Romhilt-Estes >=5, Daniel >=10,
# the diffuse ST pattern). The others have no validated threshold, and an
# invented one would break the provenance rule for thresholds.
REFERENCE_SCORE_FEATURE_IDS: tuple[str, ...] = (
    "classical_t_amplitude_avr",
    "classical_iceb_qt_qrs_ratio",
    "classical_tpeak_tend_qt_ratio",
    "classical_frontal_qrs_t_angle_device",
    "classical_qrs_voltage_sum_12lead",
    "classical_qrs_voltage_duration_product_12lead",
    "classical_romhilt_estes_points",
    "classical_daniel_pe_score",
    "classical_macruz_index",
    "classical_lab_diffuse_st_depression_avr_v1_pattern",
)
REFERENCE_SCORE_VERDICT_IDS: tuple[str, ...] = (
    "classical_t_amplitude_avr_positive",
    "classical_romhilt_estes_points_positive",
    "classical_daniel_pe_score_positive",
    "classical_lab_diffuse_st_depression_avr_v1_pattern_positive",
)
# Every classical quantity an extraction writes, for code that must know them
# all (the criteria audit, the external cohorts).
ALL_CLASSICAL_FEATURE_IDS: tuple[str, ...] = (
    CLASSICAL_FEATURE_IDS + REFERENCE_SCORE_FEATURE_IDS
)


# Markers for which a second VCG transform is not a sensitivity analysis.
#
# The sensitivity arm recomputes core-panel markers under inverse Dower beside
# Kors, to show how much a value depends on the transform. For a marker read
# off the twelve-lead ECG and the delineation rather than off the VCG vector,
# both arms compute the same number from the same input. Measured over the
# current store, these four ``sensitivity_inverse_dower__*`` columns are
# byte-identical to their base -- 10 % of the forty-marker sensitivity set:
#
#   qt_interval_ms, qtc_fridericia_ms, tpeak_tend_ms, cornell_voltage_surrogate
#
# Computing them twice costs a tenth of the sensitivity arm, and it is worse
# than wasteful: a table of transform sensitivity that lists four markers with
# a correlation of exactly 1,000 reports robustness where it has only reported
# that the transform was never involved.
#
# The extraction re-checks this after every run rather than trusting the list
# (see ``_transform_sensitivity_report``), so a marker that becomes
# transform-independent later is found rather than assumed.
TRANSFORM_INDEPENDENT_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "qt_interval_ms",
        "qtc_fridericia_ms",
        "tpeak_tend_ms",
        "cornell_voltage_surrogate",
    }
)


CORE_REQUIRED_VCG: tuple[str, ...] = (
    "qrs_t_hausdorff_distance",
    "qrs_t_vector_correlation_distance",
    "frontal_xy_qrs_t_angle_deg",
    "horizontal_xz_qrs_t_angle_deg",
    "spatial_qrs_t_angle_deg",
    "svg_magnitude",
    "svg_qrs_area_ratio",
    "spatial_magnitude_max",
    "qrs_terminal_20ms_vector",
    "qrs_loop_area_best_fit_plane",
    "qrst_loop_frontal_xy_winding_number",
    "st_loop_area_best_fit_plane",
    "st_loop_convex_hull_volume_3d",
    "t_loop_planarity",
    "t_loop_tortuosity",
    "t_loop_width_major",
    "t_downstroke_velocity_energy",
    "p_loop_width_major",
    "p_loop_terminal_force_vector",
    "p_loop_template_instability",
    "p_loop_interbeat_axis_variability",
    "qt_interval_ms",
    "qtc_fridericia_ms",
    "tpeak_tend_ms",
    "cornell_voltage_surrogate",
)


CORE_VCG_SEEDS: tuple[str, ...] = (
    "qrs_t_hausdorff_distance",
    "qrs_t_dtw_distance",
    "qrs_t_procrustes_distance",
    "qrs_t_cosine_distance",
    "qrs_t_vector_correlation_distance",
    "qrs_t_directional_wasserstein",
    "qrs_t_morphological_discordance",
    "frontal_xy_qrs_t_angle_deg",
    "horizontal_xz_qrs_t_angle_deg",
    "sagittal_yz_qrs_t_angle_deg",
    "spatial_qrs_t_angle_deg",
    "spatial_peak_qrs_t_angle_deg",
    "svg_magnitude",
    "svg_qrs_area_ratio",
    "svg_t_area_ratio",
    "svg_angle_to_qrs_area_deg",
    "svg_angle_to_t_area_deg",
    "spatial_magnitude_max",
    "magnitude_area_qrs",
    "magnitude_area_qrst",
    "qrs_initial_20ms_vector",
    "qrs_terminal_20ms_vector",
    "qrs_loop_area_best_fit_plane",
    "qrs_loop_convex_hull_volume_3d",
    "qrs_loop_planarity",
    "qrs_loop_tortuosity",
    "qrs_loop_frontal_xy_winding_number",
    "qrs_loop_horizontal_xz_winding_number",
    "qrs_loop_sagittal_yz_winding_number",
    "qrst_loop_frontal_xy_winding_number",
    "qrst_loop_area_best_fit_plane",
    "st_loop_area_best_fit_plane",
    "st_loop_convex_hull_volume_3d",
    "st_loop_planarity",
    "t_loop_area_best_fit_plane",
    "t_loop_convex_hull_volume_3d",
    "t_loop_planarity",
    "t_loop_width_major",
    "t_loop_width_minor",
    "t_loop_tortuosity",
    "t_downstroke_velocity_energy",
    "t_downstroke_curvature_mean",
    "t_downstroke_vector",
    "p_loop_area_best_fit_plane",
    "p_loop_width_major",
    "p_loop_width_minor",
    "p_loop_planarity",
    "p_loop_terminal_force_vector",
    "p_loop_template_instability",
    "p_loop_interbeat_axis_variability",
    "qt_interval_ms",
    "qtc_bazett_ms",
    "qtc_fridericia_ms",
    "qtc_framingham_ms",
    "jt_interval_ms",
    "tpeak_tend_ms",
)


SENTINEL_PRIORITY_TOKENS: tuple[str, ...] = (
    "qrs_loop",
    "t_loop",
    "p_loop",
    "st_loop",
    "curvature",
    "torsion",
    "velocity",
    "acceleration",
    "jerk",
    "entropy",
    "self_intersection",
    "nondipolar",
    "lambda",
    "residual",
    "wavelet",
    "spectral",
    "winding",
    "tortuosity",
    "planarity",
    "instability",
    "alternans",
)

SENTINEL_FAMILY_ORDER: tuple[str, ...] = (
    "qrs_t_coupling",
    "spatial_ventricular_gradient",
    "qrs_activation_geometry",
    "t_repolarization",
    "atrial_activation",
    "st_ischemia_geometry",
    "repolarization_intervals",
    "magnitude_voltage",
    "kinematics",
    "dipolarity_pca",
    "topology_and_recurrence",
    "spectral_wavelet",
    "beat_variability",
)


def default_feature_panels() -> tuple[FeaturePanelSpec, ...]:
    available = set(BiomarkerRegistry.available())
    safe_available = sorted(
        name
        for name in available
        if _safe_scalar_name(name)
        and canonical_feature_id(name) == name
    )
    development_eligible = [
        name for name in safe_available if is_core_sentinel_candidate(name)
    ]

    ordered_core_seeds = CORE_REQUIRED_VCG + tuple(
        name for name in CORE_VCG_SEEDS if name not in CORE_REQUIRED_VCG
    )
    core_vcg = [
        name for name in ordered_core_seeds
        if (
            name in available
            and _safe_scalar_name(name)
            and canonical_feature_id(name) == name
            and is_core_sentinel_candidate(name)
        )
    ]
    core_vcg = _fill_to_size(
        core_vcg,
        development_eligible,
        80 - len(CLASSICAL_FEATURE_IDS),
        preferred_tokens=(
            "qrs_t", "svg", "spatial", "qrs_loop", "t_loop",
            "st_loop", "p_loop", "qt", "jt", "tpeak",
        ),
    )
    core_features = tuple(core_vcg + list(CLASSICAL_FEATURE_IDS))
    if len(core_features) != 80:
        raise RuntimeError(
            f"Core panel must contain exactly 80 features, got {len(core_features)}"
        )
    missing_required = sorted(set(CORE_REQUIRED_VCG) - set(core_vcg))
    if missing_required:
        raise RuntimeError(
            "Core panel is missing score/model dependencies: "
            f"{missing_required}"
        )

    excluded = set(core_vcg)
    sentinel_candidates = [
        name
        for name in development_eligible
        if name not in excluded
    ]
    sentinel = _family_balanced_selection(
        sentinel_candidates,
        240,
    )
    if len(sentinel) != 240:
        raise RuntimeError(
            f"Sentinel panel must contain exactly 240 features, got {len(sentinel)}"
        )
    discovery = tuple(
        name
        for name in safe_available
        if (
            name not in excluded
            and name not in set(sentinel)
            and is_discovery_candidate(name)
        )
    )

    return (
        FeaturePanelSpec(
            panel_id="primary_core",
            version=PANEL_VERSION,
            tier=PanelTier.PRIMARY_CORE,
            feature_ids=core_features,
            description=(
                "Scientifically audited interpretable VCG and classical ECG "
                "development core. Requires a new freeze before any final test."
            ),
            frozen_before_outcome_review=False,
        ),
        FeaturePanelSpec(
            panel_id="sentinel",
            version=PANEL_VERSION,
            tier=PanelTier.SENTINEL,
            feature_ids=tuple(sentinel),
            description=(
                "Role-filtered, physiologically family-balanced VCG sentinel "
                "panel. Requires a new freeze."
            ),
            frozen_before_outcome_review=False,
        ),
        FeaturePanelSpec(
            panel_id="classical_criteria",
            version=PANEL_VERSION,
            tier=PanelTier.CLASSICAL,
            feature_ids=(
                CLASSICAL_CRITERION_FEATURE_IDS
                + REFERENCE_SCORE_FEATURE_IDS
                + REFERENCE_SCORE_VERDICT_IDS
            ),
            description=(
                "The published criteria evaluated as criteria: the binary "
                "verdict of each classical rule, which is the comparison "
                "every new marker is supposed to beat, and the published "
                "reference scores (TaVR, iCEB, Tp-e/QT, frontal QRS-T "
                "angle, 12-lead voltage, Romhilt-Estes, Daniel, Macruz, "
                "diffuse ST pattern) with their verdict where the source "
                "gives a cut-off."
            ),
            frozen_before_outcome_review=True,
            required_complete=True,
        ),
        FeaturePanelSpec(
            panel_id="discovery_reserve",
            version=PANEL_VERSION,
            tier=PanelTier.DISCOVERY_RESERVE,
            feature_ids=discovery,
            description=(
                "Audited exploratory scalars, including explicitly marked "
                "post-development hypotheses; never a primary claim source."
            ),
            frozen_before_outcome_review=False,
        ),
    )


def freeze_panel_registry(path: Path | str) -> dict:
    panels = default_feature_panels()
    payload = {
        "panel_version": PANEL_VERSION,
        "panels": [panel.to_dict() for panel in panels],
        "sentinel_family_counts": _family_counts(
            next(panel for panel in panels if panel.panel_id == "sentinel").feature_ids
        ),
        "role_counts": {
            panel.panel_id: _role_counts(panel.feature_ids)
            for panel in panels
        },
    }
    payload["panel_hash"] = content_hash(payload)
    atomic_write_json(path, payload)
    return payload


def _safe_scalar_name(name: str) -> bool:
    if any(token in name for token in ("<", ">", "*", "/")):
        return False
    if name in NON_RECORD_SCALAR_FEATURE_IDS:
        return False
    definition = BiomarkerRegistry.definition(name)
    return (
        definition.unit.casefold()
        not in {"categorical", "text", "vector", "array", "mixed"}
    )


def _fill_to_size(
    selected: list[str],
    candidates: Iterable[str],
    size: int,
    *,
    preferred_tokens: tuple[str, ...],
) -> list[str]:
    output = list(dict.fromkeys(selected))
    ranked = sorted(
        set(candidates) - set(output),
        key=lambda name: (
            min(
                (
                    index
                    for index, token in enumerate(preferred_tokens)
                    if token in name
                ),
                default=len(preferred_tokens),
            ),
            name,
        ),
    )
    output.extend(ranked[: max(0, size - len(output))])
    return output[:size]


def _family_balanced_selection(
    candidates: Iterable[str],
    size: int,
) -> list[str]:
    candidate_set = set(candidates)
    families = list(SENTINEL_FAMILY_ORDER)
    observed = sorted({_feature_family(name) for name in candidate_set})
    families.extend(family for family in observed if family not in families)
    buckets = {family: [] for family in families}
    for name in candidate_set:
        buckets[_feature_family(name)].append(name)
    for family in buckets:
        buckets[family].sort(key=_scientific_candidate_rank)

    selected: list[str] = []
    offsets = {family: 0 for family in buckets}
    while len(selected) < size:
        added = False
        for family in families:
            index = offsets[family]
            if index >= len(buckets[family]):
                continue
            selected.append(buckets[family][index])
            offsets[family] += 1
            added = True
            if len(selected) == size:
                break
        if not added:
            break
    return selected


def _feature_family(name: str) -> str:
    return conceptual_feature_family(name)


def _family_counts(feature_ids: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in feature_ids:
        family = _feature_family(name)
        counts[family] = counts.get(family, 0) + 1
    return counts


def _scientific_candidate_rank(name: str) -> tuple:
    definition = BiomarkerRegistry.definition(name)
    token_rank = min(
        (
            index
            for index, token in enumerate(SENTINEL_PRIORITY_TOKENS)
            if token in name.casefold()
        ),
        default=len(SENTINEL_PRIORITY_TOKENS),
    )
    category_rank = {
        "vcg": 0,
        "loop": 0,
        "repolarization": 0,
        "literature": 1,
        "ecg": 1,
        "exploratory": 2,
        "pathophysiology": 3,
    }.get(definition.category.casefold(), 4)
    return (
        category_rank,
        token_rank,
        definition.source_index if definition.source_index is not None else 10_000,
        name,
    )


def _role_counts(feature_ids: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature_id in feature_ids:
        role = feature_profile(feature_id).role
        counts[role] = counts.get(role, 0) + 1
    return counts
