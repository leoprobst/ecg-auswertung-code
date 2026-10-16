from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.mimic.biomarker_audit import (
    _merge_dynamic_summary,
    _synthetic_cases,
    run_biomarker_audit,
)
from vcg_biomarker.mimic.ecg_lab import (
    _load_recommended_single_marker_candidates,
    _research_pattern_summary,
    _score_component_details,
    _single_case_marker_patterns,
    _single_case_operating_points,
)
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.feature_roles import (
    BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS,
    POST_OUTCOME_DISCOVERY_FEATURE_IDS,
    ROLE_ALIAS,
    ROLE_CONTEXT,
    ROLE_QC,
    FEATURE_ROLE_POLICY_VERSION,
    canonical_feature_id,
    feature_profile,
)
from vcg_biomarker.mimic.optimization import grouped_score_search
from vcg_biomarker.core.biomarkers.electrophysiology_markers import (
    ELECTROPHYSIOLOGY_SPECS,
)
from vcg_biomarker.core.biomarkers.discovery_v2_biomarkers import (
    DISCOVERY_V2_SPECS,
)
from vcg_biomarker.core.biomarkers.discovery_v3_biomarkers import (
    DISCOVERY_V3_SPECS,
)
from vcg_biomarker.core.biomarkers.discovery_v4_biomarkers import (
    DISCOVERY_V4_SPECS,
)
from vcg_biomarker.core.biomarkers.invariant_biomarkers import (
    INVARIANT_SPECS,
)
from vcg_biomarker.core.biomarkers.segment_free_biomarkers import (
    SEGMENT_FREE_SPECS,
)
from vcg_biomarker.mimic.panels import STUDY_PANEL_IDS, default_feature_panels
from vcg_biomarker.mimic.presets import get_analysis_preset
from vcg_biomarker.mimic.result_explorer import (
    build_new_association_explorer,
    build_result_explorer_tables,
    rank_broad_discovery,
)


def _values(case, names: tuple[str, ...]) -> dict[str, float | None]:
    _, delineation, ecg, vcg = case
    return {
        result.name: result.value
        for result in BiomarkerRegistry.compute_all(
            delineation,
            ecg,
            vcg,
            delineation.fs,
            names=list(names),
        )
    }


def test_new_association_explorer_exposes_strength_and_noncardiac_scope() -> None:
    pathology_status = pd.DataFrame(
        {
            "phenotype_id": ["neurology.demo"],
            "phenotype_name": ["Neurologische Beispielerkrankung"],
            "body_system": ["neurological"],
            "feature_id": ["spatial_qrs_t_angle_deg"],
            "panel_id": ["primary_core"],
            "train_q_value": [0.001],
            "validation_q_value": [0.01],
            "validation_aligned_auc": [0.80],
            "validation_effect": [0.60],
            "association_strength": [0.60],
            "n_positive": [180],
            "n_negative": [720],
            "inference_status": ["formal_inference"],
            "direction_consistent": [True],
            "unadjusted_replicated": [True],
            "adjustment_status": ["evaluable"],
            "adjusted_q_value": [0.02],
            "adjusted_or_per_sd": [1.4],
            "adjustment_robust": [True],
            "status_id": ["replicated"],
            "status_de": ["repliziert"],
            "evidence_rank": [4],
            "status_reason_de": ["Validation und Adjustierung bestanden."],
        }
    )
    result = build_new_association_explorer(
        pathology_status,
        pd.DataFrame(),
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row["association_strength"] == pytest.approx(0.60)
    assert row["statistical_strength"] == pytest.approx(2.0)
    assert row["cardiac_scope_id"] == "non_cardiovascular"
    assert row["followup_priority"] == "high"
    assert row["n_positive"] == 180
    assert row["locked_test_scanned"] is False or not bool(
        row["locked_test_scanned"]
    )
    # spatial_qrs_t_angle_deg is a literature-known mortality marker, but
    # this row's target is an unrelated neurological hypothesis, so it
    # must not be flagged as a known literature association.
    assert row["novel_vs_literature"] == "not_yet_described"


def test_new_association_explorer_flags_known_literature_associations() -> None:
    pathology_status = pd.DataFrame(
        {
            "phenotype_id": ["outcome_1y_mortality"],
            "phenotype_name": ["1-year mortality"],
            "body_system": ["circulatory"],
            "feature_id": ["spatial_qrs_t_angle_deg"],
            "panel_id": ["primary_core"],
            "train_q_value": [0.001],
            "validation_q_value": [0.01],
            "validation_aligned_auc": [0.80],
            "validation_effect": [0.60],
            "association_strength": [0.60],
            "n_positive": [180],
            "n_negative": [720],
            "inference_status": ["formal_inference"],
            "direction_consistent": [True],
            "unadjusted_replicated": [True],
            "adjustment_status": ["evaluable"],
            "adjusted_q_value": [0.02],
            "adjusted_or_per_sd": [1.4],
            "adjustment_robust": [True],
            "status_id": ["replicated"],
            "status_de": ["repliziert"],
            "evidence_rank": [4],
            "status_reason_de": ["Validation und Adjustierung bestanden."],
        }
    )
    result = build_new_association_explorer(pathology_status, pd.DataFrame())
    assert result.iloc[0]["novel_vs_literature"] == (
        "known_literature_association"
    )


def test_new_vcg_markers_are_invariant_and_react_to_morphology() -> None:
    cases = {case[0]: case for case in _synthetic_cases()}
    names = (
        "qrs_t_normalized_hausdorff_distance",
        "qrs_t_normalized_dtw_distance",
        "qrs_t_mirror_discordant_fraction",
        "svg_cancellation_index",
        "qrs_beat_area_axis_coherence",
        "t_beat_area_axis_coherence",
        "beatwise_svg_direction_dispersion_deg",
        "p_loop_interbeat_axis_variability",
    )
    normal = _values(cases["normal"], names)
    rotated = _values(cases["rotated"], names)
    scaled = _values(cases["amplitude_scaled"], names)
    inverted = _values(cases["inverted_t"], names)
    outlier = _values(cases["single_morphology_outlier"], names)
    af_like = _values(cases["irregular_af_like"], names)

    for feature_id in names[:-1]:
        assert normal[feature_id] is not None
        assert rotated[feature_id] == pytest.approx(
            normal[feature_id],
            rel=1e-6,
            abs=5e-7,
        )
        assert scaled[feature_id] == pytest.approx(
            normal[feature_id],
            rel=1e-6,
            abs=2e-6,
        )
    assert inverted["qrs_t_mirror_discordant_fraction"] != pytest.approx(
        normal["qrs_t_mirror_discordant_fraction"]
    )
    assert (
        outlier["qrs_beat_area_axis_coherence"]
        < normal["qrs_beat_area_axis_coherence"]
    )
    assert (
        outlier["beatwise_svg_direction_dispersion_deg"]
        > normal["beatwise_svg_direction_dispersion_deg"] + 5.0
    )
    assert af_like["p_loop_interbeat_axis_variability"] is None


def test_corrected_catalog_metrics_have_non_degenerate_contracts() -> None:
    cases = {case[0]: case for case in _synthetic_cases()}
    names = (
        "qrs_loop_tortuosity",
        "qrs_t_directional_wasserstein",
        "qrs_t_loop_mirror_symmetry",
        "t_loop_repolarization_tail_index",
        "vcg_phase_space_recurrence_rate",
        "vcg_phase_space_determinism",
        "conduction_detour_index",
        "qrs_t_morphological_discordance",
    )
    normal = _values(cases["normal"], names)
    noisy = _values(cases["noisy"], names)
    inverted = _values(cases["inverted_t"], names)

    assert normal["qrs_loop_tortuosity"] >= 1.0
    assert 0.0 <= normal["qrs_t_directional_wasserstein"] <= 1.0
    assert 0.0 <= normal["qrs_t_loop_mirror_symmetry"] <= 1.0
    assert 0.0 <= normal["t_loop_repolarization_tail_index"] <= 1.0
    assert 0.0 <= normal["vcg_phase_space_recurrence_rate"] <= 1.0
    assert 0.0 <= normal["vcg_phase_space_determinism"] <= 1.0
    assert normal["conduction_detour_index"] >= 0.0
    assert normal["qrs_t_morphological_discordance"] != pytest.approx(
        inverted["qrs_t_morphological_discordance"]
    )
    assert noisy["qrs_loop_tortuosity"] > normal["qrs_loop_tortuosity"]


def test_u_wave_biomarkers_have_an_explicit_synthetic_contract() -> None:
    cases = {case[0]: case for case in _synthetic_cases()}
    names = (
        "u_peak_vector_magnitude_mv",
        "u_loop_area_vector_magnitude_mv2",
        "u_loop_planarity",
        "u_to_t_peak_magnitude_ratio",
        "t_u_area_vector_angle_deg",
        "t_u_transition_angle_deg",
    )

    normal = _values(cases["normal"], names)
    measured = _values(cases["u_wave"], names)

    assert all(normal[name] is None for name in names)
    assert all(measured[name] is not None for name in names)
    assert measured["u_peak_vector_magnitude_mv"] > 0.05
    assert measured["u_loop_area_vector_magnitude_mv2"] > 0.0
    assert 0.0 <= measured["u_loop_planarity"] <= 1.0
    assert 0.0 < measured["u_to_t_peak_magnitude_ratio"] < 1.0
    assert 0.0 <= measured["t_u_area_vector_angle_deg"] <= 180.0
    assert 0.0 <= measured["t_u_transition_angle_deg"] <= 180.0


def test_default_panels_have_no_alias_qc_or_post_outcome_marker() -> None:
    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    core = panels["primary_core"]
    sentinel = panels["sentinel"]
    discovery = panels["discovery_reserve"]
    for panel in (core, sentinel):
        for feature_id in panel.feature_ids:
            profile = feature_profile(feature_id)
            assert profile.role not in {ROLE_ALIAS, ROLE_QC}
            assert profile.canonical_feature_id == feature_id
            assert not profile.post_outcome_hypothesis
    post_outcome = {
        feature_id
        for feature_id in discovery.feature_ids
        if feature_profile(feature_id).post_outcome_hypothesis
    }
    # Post-outcome status is declared two ways: an explicit id list, and
    # membership of a marker module written after development results were
    # seen. Both must resolve to exactly the same set the discovery panel
    # carries -- nothing extra, nothing missing.
    from vcg_biomarker.core.biomarkers.morphology_variability_biomarkers import MORPHOLOGY_VARIABILITY_SPECS
    from vcg_biomarker.core.biomarkers.literature_2026_biomarkers import LITERATURE_2026_SPECS
    from vcg_biomarker.core.biomarkers.circular_projection_biomarkers import (
        CIRCULAR_PROJECTION_SPECS,
    )
    from vcg_biomarker.core.biomarkers.literature_2026b_biomarkers import (
        LITERATURE_2026B_SPECS,
    )
    from vcg_biomarker.core.biomarkers.propagation_sequence_biomarkers import (
        PROPAGATION_SEQUENCE_SPECS,
    )
    from vcg_biomarker.core.biomarkers.loop_parametrisation_biomarkers import (
        LOOP_PARAMETRISATION_SPECS,
    )
    from vcg_biomarker.core.biomarkers.thesis_derived_biomarkers import (
        THESIS_DERIVED_SPECS,
    )
    from vcg_biomarker.core.biomarkers.ischemia_thesis_biomarkers import (
        ISCHEMIA_THESIS_SPECS,
    )
    from vcg_biomarker.core.biomarkers.decision_thesis_biomarkers import (
        DECISION_THESIS_SPECS,
    )
    from vcg_biomarker.core.biomarkers.conduction_uniformity_biomarkers import (
        CONDUCTION_UNIFORMITY_SPECS,
    )
    from vcg_biomarker.core.biomarkers.pathology_mechanism_biomarkers import (
        SPECS as PATHOLOGY_MECHANISM_SPECS,
    )
    from vcg_biomarker.core.biomarkers.qrs_t_coupling_extension_biomarkers import (
        QRS_T_COUPLING_EXTENSION_SPECS,
    )
    from vcg_biomarker.core.biomarkers.physiology_thesis_biomarkers import (
        PHYSIOLOGY_THESIS_SPECS,
    )
    declared = set(POST_OUTCOME_DISCOVERY_FEATURE_IDS) | set(BEAT_MORPHOLOGY_VARIABILITY_FEATURE_IDS) | {
        spec.name for spec in MORPHOLOGY_VARIABILITY_SPECS
    } | {
        spec.name for spec in LITERATURE_2026_SPECS
    } | {
        spec.name for spec in CIRCULAR_PROJECTION_SPECS
    } | {
        spec.name for spec in LITERATURE_2026B_SPECS
    } | {
        spec.name for spec in PROPAGATION_SEQUENCE_SPECS
    } | {
        spec.name for spec in LOOP_PARAMETRISATION_SPECS
    } | {
        spec.name for spec in THESIS_DERIVED_SPECS
    } | {
        spec.name for spec in ISCHEMIA_THESIS_SPECS
    } | {
        spec.name for spec in DECISION_THESIS_SPECS
    } | {
        spec.name for spec in CONDUCTION_UNIFORMITY_SPECS
    } | {
        spec.name for spec in PATHOLOGY_MECHANISM_SPECS
    } | {
        spec.name for spec in QRS_T_COUPLING_EXTENSION_SPECS
    } | {
        spec.name for spec in PHYSIOLOGY_THESIS_SPECS
    } | {
        spec.name for spec in ELECTROPHYSIOLOGY_SPECS
    } | {
        spec.name for spec in DISCOVERY_V2_SPECS
    } | {
        spec.name for spec in DISCOVERY_V3_SPECS
    } | {
        spec.name for spec in DISCOVERY_V4_SPECS
    } | {
        spec.name for spec in SEGMENT_FREE_SPECS
    } | {
        spec.name for spec in INVARIANT_SPECS
    }
    from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import ADVANCED_SPECS
    declared.update(s.name for s in ADVANCED_SPECS if s.family == "A108 explorative Messrobustheit")
    assert post_outcome == declared
    assert post_outcome.isdisjoint(core.feature_ids)
    assert post_outcome.isdisjoint(sentinel.feature_ids)


def test_rate_context_and_exact_calculator_aliases_are_not_tested_twice() -> None:
    for feature_id in ("heart_rate", "heart_rate_bpm"):
        profile = feature_profile(feature_id)
        assert profile.role == ROLE_CONTEXT
        assert not profile.core_sentinel_eligible
        assert not profile.discovery_eligible
        assert not profile.combination_eligible
    all_panel_features = {
        feature_id
        for panel in default_feature_panels()
        for feature_id in panel.feature_ids
    }
    assert "heart_rate" not in all_panel_features
    assert canonical_feature_id("qrs_axis_to_t_axis_angle_deg") == (
        "spatial_qrs_t_angle_deg"
    )
    assert canonical_feature_id("ectopic_beat_vector_deviation") == (
        "qrs_internal_cluster_distance_median"
    )
    assert canonical_feature_id("st_loop_drift") == (
        "st_loop_start_end_distance"
    )


def test_grouped_score_search_can_require_a_vcg_component() -> None:
    rng = np.random.default_rng(20260730)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
    frame = pd.DataFrame(
        {
            "marker_a": labels + rng.normal(0.0, 0.05, len(labels)),
            "marker_b": -labels + rng.normal(0.0, 0.05, len(labels)),
        }
    )
    kwargs = {
        "labels": labels,
        "subject_ids": np.arange(len(labels)),
        "feature_ids": tuple(frame.columns),
        "family_for_feature": {
            "marker_a": "ecg_interval",
            "marker_b": "vcg_geometry",
        }.__getitem__,
        "folds": 5,
        "max_components": 2,
    }
    blocked = grouped_score_search(
        frame,
        **kwargs,
        required_component_predicate=lambda _: False,
    )
    accepted = grouped_score_search(
        frame,
        **kwargs,
        required_component_predicate=lambda feature_id: (
            feature_id == "marker_b"
        ),
    )
    assert blocked == ()
    assert len(accepted) == 1
    assert "marker_b" in accepted[0].components


def test_static_biomarker_audit_keeps_defined_novel_markers_hypothesis_only(
    tmp_path: Path,
) -> None:
    result = run_biomarker_audit(tmp_path / "audit", dynamic=False)
    catalog = pd.read_parquet(result.catalog_path)
    assert len(catalog) == len(BiomarkerRegistry.available())
    blockers = catalog.loc[catalog["default_panel_blocker"].fillna(False)]
    assert blockers.empty
    novel_core = catalog.loc[
        catalog["panel_id"].eq("primary_core")
        & catalog["reference_count"].eq(0)
    ]
    assert not novel_core.empty
    assert novel_core["prespecified_novel_definition"].eq(True).all()
    assert novel_core["claim_eligible"].eq(False).all()
    assert novel_core["allowed_claim_scope"].eq(
        "prespecified_novel_hypothesis"
    ).all()
    assert novel_core["audit_issues"].str.contains(
        "prespecified_novel_without_literature_reference",
        regex=False,
    ).all()
    assert not catalog["implementation_status"].ne("implemented").any()
    assert catalog["calculator_contract_status"].eq("not_run").all()


def test_synthetic_constant_candidate_is_audited_as_warning() -> None:
    catalog = pd.DataFrame(
        [
            {
                "feature_id": "constant_candidate",
                "role": "candidate_biomarker",
                "panel_id": "sentinel",
                "default_panel_blocker": False,
                "audit_severity": "ok",
                "audit_issues": "",
            }
        ]
    )
    dynamic = pd.DataFrame(
        [
            {
                "feature_id": "constant_candidate",
                "status": "finite",
                "value": 2.0,
                "invariance_violation": False,
            }
            for _ in range(8)
        ]
    )

    audited = _merge_dynamic_summary(
        catalog,
        dynamic,
        pd.DataFrame(),
    ).iloc[0]

    assert bool(audited["synthetic_constant_candidate"])
    assert audited["audit_severity"] == "warning"
    assert "constant_across_synthetic_audit_cases" in audited["audit_issues"]
    assert not bool(audited["default_panel_blocker"])


def test_result_explorer_uses_only_endpoint_applicable_baselines() -> None:
    common = {
        "assessability_gate_passed": True,
        "endpoint_id": "acute_mi_high_specificity",
        "evaluation_split": "validation",
        "analysis_type": "single_marker",
        "panel_id": "primary_core",
        "inference_status": "formal_inference",
        "n_positive": 150,
        "n_negative": 300,
        "missingness": 0.0,
        "roc_auc_ci_low": 0.55,
        "roc_auc_ci_high": 0.70,
        "average_precision": 0.30,
        "q_value": 0.01,
        "specificity_90_sensitivity": 0.30,
        "specificity_95_sensitivity": 0.20,
        "sensitivity_90_specificity": 0.25,
    }
    screening = pd.DataFrame(
        [
            {
                **common,
                "feature_id": "classical_p_pulmonale",
                "roc_auc": 0.80,
            },
            {
                **common,
                "feature_id": "classical_st_elevation_contiguous",
                "roc_auc": 0.61,
            },
            {
                **common,
                "feature_id": "qrs_t_hausdorff_distance",
                "roc_auc": 0.66,
            },
        ]
    )

    tables = build_result_explorer_tables(screening)

    assert "classical_p_pulmonale" not in set(
        tables.marker_pathology["feature_id"]
    )
    assert "classical_st_elevation_contiguous" in set(
        tables.marker_pathology["feature_id"]
    )
    summary = tables.endpoint_summary.iloc[0]
    assert summary["best_classical_rule"] == (
        "classical_st_elevation_contiguous"
    )


def test_result_explorer_labels_mixed_audited_scores_honestly() -> None:
    combinations = pd.DataFrame(
        [
            {
                "endpoint_id": "reduced_lvef_strict",
                "feature_id": "small_score",
                "evaluation_split": "validation",
                "analysis_type": "exploratory_mimic_combination",
                "components": "qrs_duration_ms|qrs_t_hausdorff_distance",
                "component_modalities": "ecg|vcg",
                "feature_modality": "mixed_ecg_vcg_score",
                "feature_role_policy_version": FEATURE_ROLE_POLICY_VERSION,
                "panel_id": "exploratory_score",
                "n_positive": 150,
                "n_negative": 250,
                "roc_auc": 0.72,
                "specificity_95_sensitivity": 0.30,
                "specificity_95_specificity": 0.95,
                "validation_stable": True,
            }
        ]
    )

    tables = build_result_explorer_tables(
        pd.DataFrame(),
        combinations=combinations,
    )

    row = tables.pathology_marker.iloc[0]
    assert row["feature_modality"] == "mixed_ecg_vcg_score"
    assert not bool(row["historical_pre_audit_result"])


def test_broad_discovery_ranking_collapses_families_and_preserves_counts() -> None:
    frame = pd.DataFrame(
        {
            "discovery_class": ["pathology_hypothesis"] * 2,
            "discovery_domain": ["Respiratory"] * 2,
            "target_id": ["P1"] * 2,
            "target_name": ["Pulmonary phenotype"] * 2,
            "target_kind": ["retrospective_proxy"] * 2,
            "feature_id": [
                "qrs_loop_tortuosity",
                "qrs_loop_planarity",
            ],
            "panel_id": ["primary_core"] * 2,
            "n_total": [500, 500],
            "n_positive": [150, 150],
            "n_negative": [350, 350],
            "train_effect": [0.20, 0.18],
            "train_q_value": [1e-5, 2e-5],
            "validation_effect": [0.22, 0.17],
            "validation_q_value": [1e-4, 2e-4],
            "direction_consistent": [True, True],
            "replicated": [True, True],
            "inference_status": ["formal_inference"] * 2,
            "validation_stable": [True, False],
            "locked_test_scanned": [False, False],
            "feature_role_policy_version": [
                FEATURE_ROLE_POLICY_VERSION,
                FEATURE_ROLE_POLICY_VERSION,
            ],
        }
    )

    ranked = rank_broad_discovery(frame)

    assert len(ranked) == 1
    assert ranked.iloc[0]["n_positive"] == 150
    assert ranked.iloc[0]["evidence_tier"] == (
        "validation_replicated_formal"
    )
    assert not bool(ranked.iloc[0]["locked_test_scanned"])
    assert not bool(ranked.iloc[0]["historical_implementation"])

    historical = rank_broad_discovery(
        frame.drop(columns="feature_role_policy_version")
    )
    assert historical.iloc[0]["evidence_tier"] == (
        "historical_recompute_required"
    )
    assert bool(historical.iloc[0]["historical_implementation"])


def test_ecg_lab_exposes_contributions_and_threshold_distance() -> None:
    spec = {
        "components": ("marker_up", "marker_down"),
        "directions": (1, -1),
        "means": (1.0, 4.0),
        "standard_deviations": (2.0, 1.0),
    }
    details = _score_component_details(
        {"marker_up": 3.0, "marker_down": 2.0},
        spec,
    )
    assert [row["aligned_z_contribution"] for row in details] == [1.0, 2.0]
    points = _single_case_operating_points(
        1.5,
        {
            "direction": 1,
            "thresholds": {
                "specificity_95": 1.2,
                "specificity_90": 1.0,
                "sensitivity_90": -0.5,
            },
        },
    )
    assert points["specificity_95"]["triggered"] is True
    assert points["specificity_95"]["distance_to_threshold"] == pytest.approx(
        0.3
    )
    summary = _research_pattern_summary(
        [
            {
                "endpoint_id": "endpoint",
                "primary_operating_point": points["specificity_95"],
                "research_pattern_status": (
                    "above_train_95_specificity_threshold"
                ),
                "measurement_confidence": 0.9,
                "evidence_status": "development_validation_stable",
                "validation_metrics": {
                    "roc_auc": 0.72,
                    "sensitivity_at_95_specificity": 0.4,
                },
                "release_id": "score@1",
            }
        ]
    )
    assert summary[0]["triggered_at_train_95_specificity"] is True
    assert "no diagnosis" in summary[0]["interpretation"]


def test_ecg_lab_single_marker_selection_is_stable_and_family_limited(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "iv",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "studies.csv",
        echo_record_list_path=tmp_path / "records.csv",
        work_root=tmp_path / "work",
    )
    run_root = config.runs_root / "development"
    run_root.mkdir(parents=True)
    (run_root / "development_provenance.json").write_text(
        json.dumps(
            {
                "preview": False,
                "analysis_completeness": "complete_snapshot",
                "locked_test_labels_read": False,
            }
        ),
        encoding="utf-8",
    )
    feature_ids = (
        "qrs_t_hausdorff_distance",
        "qrs_t_dtw_distance",
        "t_loop_area_best_fit_plane",
    )
    pd.DataFrame(
        [
            {
                "endpoint_id": "reduced_lvef_strict",
                "feature_id": feature_id,
                "candidate_type": "vcg_single",
                "panel_id": "sentinel",
                "inference_status": "formal_inference",
                "reference_independence": "clinical_reference_comparison",
                "historical_pre_audit_result": False,
                "post_outcome_hypothesis": False,
                "direction_agreement": True,
                "roc_auc": auc,
                "roc_auc_ci_low": auc - 0.04,
                "roc_auc_ci_high": auc + 0.04,
                "average_precision": 0.5,
                "sensitivity_at_95_specificity": 0.3,
                "specificity_at_90_sensitivity": 0.4,
                "q_value": 0.001,
                "train_validation_auc_gap": 0.02,
                "evidence_grade": "A",
                "evidence_rank": auc * 100.0,
            }
            for feature_id, auc in zip(
                feature_ids,
                (0.72, 0.70, 0.69),
            )
        ]
    ).to_parquet(run_root / "marker_pathology_explorer.parquet")
    models = [
        {
            "endpoint_id": "reduced_lvef_strict",
            "panel_id": "sentinel",
            "feature_id": feature_id,
            "direction": 1,
            "mean": 0.0,
            "standard_deviation": 1.0,
            "thresholds": {
                "specificity_95": 1.0,
                "specificity_90": 0.8,
                "sensitivity_90": -0.5,
            },
            "n_train": 400,
            "n_positive_train": 150,
            "n_negative_train": 250,
        }
        for feature_id in feature_ids
    ]
    (run_root / "trained_marker_models.json").write_text(
        json.dumps({"models": models}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "vcg_biomarker.mimic.ecg_lab._run_implementation_compatibility",
        lambda _: (True, "test"),
    )

    candidates, inventory = _load_recommended_single_marker_candidates(
        config
    )

    assert len(candidates) == 2
    assert {
        item["conceptual_family"] for item in candidates
    } == {"qrs_t_coupling", "t_repolarization"}
    assert inventory["endpoint_count"] == 1
    patterns = _single_case_marker_patterns(
        {
            "qrs_t_hausdorff_distance": 1.2,
            "t_loop_area_best_fit_plane": 0.2,
        },
        candidates,
        measurement_confidence=0.9,
    )
    assert patterns[0]["feature_id"] == "qrs_t_hausdorff_distance"
    assert patterns[0]["primary_operating_point"]["triggered"] is True


def test_single_marker_p_projection_abstains_without_case_rhythm_context() -> None:
    feature_id = "p_loop_frontal_xy_area"
    patterns = _single_case_marker_patterns(
        {feature_id: 3.0},
        ({
            "candidate_id": "p-projection",
            "endpoint_id": "echo_lae",
            "feature_id": feature_id,
            "marker_model": {
                "direction": 1,
                "thresholds": {"specificity_95": 1.0},
            },
        },),
        measurement_confidence=.95,
    )
    assert patterns[0]["raw_value"] is None
    assert patterns[0]["research_pattern_status"] == "not_evaluable"
    assert patterns[0]["primary_operating_point"]["triggered"] is None
    assert patterns[0]["analysis_restriction"]["abstention_reason"]


def test_exhaustive_preset_is_high_compute_but_keeps_scientific_guards() -> None:
    preset = get_analysis_preset("exhaustive_research")
    # Lowered from 2000 on purpose: the subject-cluster bootstrap was measured
    # to saturate near 500 resamples, so the extra thousand bought weeks of
    # arithmetic and no change in any interval. More can still be added
    # afterwards through refinement, which is where that decision belongs.
    assert preset.policy.bootstrap_iterations == 1000
    assert preset.policy.bootstrap_iterations >= 500
    assert preset.policy.grouped_cv_repeats == 5
    assert preset.policy.include_broad_phenome_scan
    # The criteria panel joined the study set: the classical rules are now
    # screened as rules rather than only as the continuous quantity they are
    # cut from.
    assert preset.panel_ids == STUDY_PANEL_IDS
    assert "classical_criteria" in preset.panel_ids
    assert not preset.policy.allow_outcome_tuned_filters
    assert not preset.policy.allow_label_rule_optimization
    assert not preset.policy.allow_locked_test_optimization
