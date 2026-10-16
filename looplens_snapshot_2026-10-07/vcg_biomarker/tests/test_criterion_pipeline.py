from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.core.delineation import benchmark_suite
from vcg_biomarker.core.delineation.benchmark_suite import (
    BENCHMARK_SPECS,
    inventory_delineation_benchmarks,
    run_delineation_benchmark_suite,
)
from vcg_biomarker.mimic.criteria import (
    build_endpoint_criterion_catalog,
    finalize_criterion_catalog_evidence,
)
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.baseline_registry import BaselineRegistry
from vcg_biomarker.mimic.ecg_lab import (
    _evaluate_case_criteria,
    _research_diagnosis_summary,
)
from vcg_biomarker.mimic.models import CriterionStage, ThresholdBand
from vcg_biomarker.mimic.models import ResultQuery
from vcg_biomarker.mimic.locking import (
    _load_and_validate_locked_test_marker_shortlist,
    _models_for_locked_shortlist,
)
from vcg_biomarker.mimic.optimization import grouped_score_search
from vcg_biomarker.mimic.phenome_adjustment import (
    PhenomeAdjustmentPolicy,
    build_phenome_discovery_status,
    confirm_replicated_phenome_hits,
)
from vcg_biomarker.mimic.statistics import (
    apply_fdr_by_family,
    bootstrap_operating_threshold_bands,
    fit_marker_model,
)
from vcg_biomarker.mimic.result_query import (
    ResultRepository,
    _sort_column,
    _where_clause,
)
from vcg_biomarker.mimic.storage import StageManifest
from vcg_biomarker.mimic.validation import (
    VALIDATION_STAGE_VERSION,
    _build_locked_test_marker_shortlist,
)


def test_discovery_query_filters_scope_and_sorts_by_effect_strength() -> None:
    query = ResultQuery(
        perspective="discovery",
        candidate_types=("pathology_hypothesis",),
        baseline_ids=("non_cardiovascular",),
        evidence_grades=("replicated",),
        search_text="neurologisch",
        sort_by="association_strength",
    )
    schema = {
        "discovery_class",
        "cardiac_scope_id",
        "status_id",
        "target_name",
        "discovery_domain",
        "cardiac_scope_de",
        "association_strength",
    }
    where, parameters, unsupported = _where_clause(query, schema)
    assert '"discovery_class"' in where
    assert '"cardiac_scope_id"' in where
    assert '"status_id"' in where
    assert parameters[:3] == [
        "pathology_hypothesis",
        "non_cardiovascular",
        "replicated",
    ]
    assert not unsupported
    assert _sort_column(query, schema) == "association_strength"


def _technical_delineation(
    rhythm_class: str = "regular",
    *,
    segment_confidences: dict[str, float] | None = None,
    rhythm_confidence: float = .90,
    p_wave_coverage: float = .80,
):
    confidences = {
        "p": 0.92,
        "pr": 0.92,
        "qrs": 0.95,
        "st": 0.90,
        "t": 0.93,
        "u": 0.0,
        **(segment_confidences or {}),
    }
    return SimpleNamespace(
        quality_score=0.94,
        rhythm=SimpleNamespace(
            rhythm_class=SimpleNamespace(value=rhythm_class),
            confidence=rhythm_confidence,
            p_wave_coverage=p_wave_coverage,
        ),
        segment_confidence=lambda segment: confidences.get(segment, 0.90),
    )


def test_negative_marker_threshold_band_preserves_original_coordinate() -> None:
    rng = np.random.default_rng(7101)
    subject_ids = np.repeat(np.arange(80), 2)
    labels = np.repeat(np.asarray([0] * 40 + [1] * 40, dtype=np.int8), 2)
    values = np.where(labels == 1, -4.0, 4.0) + rng.normal(
        0.0, 0.45, len(labels)
    )
    model = fit_marker_model(values, labels, "lower_is_pathologic")
    assert model.direction == -1

    uncertainty = bootstrap_operating_threshold_bands(
        subject_ids,
        values,
        labels,
        direction=model.direction,
        point_thresholds={"specificity_90": model.thresholds["specificity_90"]},
        iterations=100,
        seed=44,
    )["specificity_90"]
    band = ThresholdBand(
        operating_point="specificity_90",
        threshold=float(uncertainty["threshold_original"]),
        gray_zone_low=float(uncertainty["gray_zone_original_low"]),
        gray_zone_high=float(uncertainty["gray_zone_original_high"]),
        direction=-1,
        uncertainty_method=str(uncertainty["uncertainty_method"]),
        bootstrap_iterations=int(uncertainty["bootstrap_iterations"]),
        successful_bootstrap_iterations=int(
            uncertainty["successful_bootstrap_iterations"]
        ),
    )

    assert band.gray_zone_low <= band.threshold <= band.gray_zone_high
    assert band.successful_bootstrap_iterations >= 80
    assert band.classify(-10.0) == "positive"
    assert band.classify(10.0) == "negative"


def test_ecg_lab_applies_negative_direction_gray_zone_and_research_guard() -> None:
    candidate = {
        "criterion_id": "criterion-low-marker",
        "endpoint_id": "acute_mi_high_specificity",
        "display_name_de": "Akuter Herzinfarkt: niedriger VCG-Marker",
        "feature_id": "qrs_t_hausdorff_distance",
        "feature_name_de": "QRS-T-Hausdorff-Abstand",
        "component_ids": ("qrs_t_hausdorff_distance",),
        "score_spec": {},
        "threshold_bands": (
            ThresholdBand(
                operating_point="specificity_95",
                threshold=-2.0,
                gray_zone_low=-2.2,
                gray_zone_high=-1.8,
                direction=-1,
                uncertainty_method="subject_cluster_bootstrap",
                bootstrap_iterations=1000,
                successful_bootstrap_iterations=980,
            ),
        ),
        "threshold_metadata": {
            "specificity_95": {
                "sensitivity_validation": 0.71,
                "specificity_validation": 0.95,
            }
        },
        "applicability": {
            "required_leads": (),
            "minimum_signal_quality": 0.60,
            "minimum_fiducial_confidence": 0.60,
            "allowed_rhythm_classes": ("regular",),
        },
        "stage": CriterionStage.VALIDATION_REPLICATED.value,
        "evidence_grade": "validation_replicated",
        "snapshot_hash": "snapshot-lab",
        "source_run_id": "run-lab",
        "roc_auc": 0.83,
        "q_value": 0.004,
    }
    findings = _evaluate_case_criteria(
        (candidate,),
        feature_values={"qrs_t_hausdorff_distance": -3.0},
        score_component_values={"qrs_t_hausdorff_distance": -3.0},
        classical_payload={},
        baseline_registry=BaselineRegistry(),
        lead_names=(),
        signal_quality=0.96,
        delineation=_technical_delineation(),
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding["classification"] == "positive"
    assert finding["distance_to_threshold"] == 1.0
    assert finding["distance_to_gray_zone"] == pytest.approx(0.8)
    assert not finding["research_diagnosis_phrase_allowed"]
    assert "Wahrscheinliche Forschungsdiagnose" not in finding["interpretation_de"]
    assert finding["evidence_stage"] == "exploratory"
    assert finding["probability"] is None
    assert finding["specificity"] == 0.95
    summary = _research_diagnosis_summary(findings)
    assert summary["status"] != "probable_research_diagnosis"
    assert not summary["probability_available"]


def test_ecg_lab_abstains_from_p_wave_criterion_for_af_pattern() -> None:
    candidate = {
        "criterion_id": "criterion-p-loop",
        "endpoint_id": "new_onset_af_90d",
        "feature_id": "p_loop_width",
        "component_ids": ("p_loop_width",),
        "score_spec": {},
        "threshold_bands": (
            ThresholdBand(
                operating_point="specificity_95",
                threshold=1.0,
                gray_zone_low=0.9,
                gray_zone_high=1.1,
                direction=1,
            ),
        ),
        "applicability": {
            "required_leads": (),
            "minimum_signal_quality": 0.60,
            "minimum_fiducial_confidence": 0.60,
            "allowed_rhythm_classes": (
                "regular",
                "irregular",
                "wide_qrs_pattern",
            ),
        },
        "stage": CriterionStage.VALIDATION_REPLICATED.value,
        "evidence_grade": "validation_replicated",
        "snapshot_hash": "snapshot-af",
    }
    findings = _evaluate_case_criteria(
        (candidate,),
        feature_values={"p_loop_width": 3.0},
        score_component_values={"p_loop_width": 3.0},
        classical_payload={},
        baseline_registry=BaselineRegistry(),
        lead_names=(),
        signal_quality=0.95,
        delineation=_technical_delineation(
            "af_compatible_pattern",
            segment_confidences={"p": 0.0},
        ),
    )

    assert findings[0]["classification"] == "not_evaluable"
    assert not findings[0]["research_diagnosis_phrase_allowed"]
    assert any(
        "P-Wellen-Morphologie" in reason
        for reason in findings[0]["applicability_reasons_de"]
    )


def test_ecg_lab_uses_exact_p_projection_restriction_before_threshold() -> None:
    feature_id = "p_loop_frontal_xy_area"
    candidate = {
        "criterion_id": "criterion-p-projection",
        "endpoint_id": "echo_lae",
        "feature_id": feature_id,
        "component_ids": (feature_id,),
        "score_spec": {},
        "threshold_bands": (
            ThresholdBand(
                operating_point="specificity_95",
                threshold=1.0,
                gray_zone_low=.9,
                gray_zone_high=1.1,
                direction=1,
            ),
        ),
        "applicability": {},
        "stage": CriterionStage.VALIDATION_REPLICATED.value,
        "evidence_grade": "validation_replicated",
        "snapshot_hash": "snapshot-p-projection",
        "sensitivity": .99,  # Unqualified/training metric is NOT validation performance.
        "specificity": .98,
    }
    finding = _evaluate_case_criteria(
        (candidate,),
        feature_values={feature_id: 3.0},
        score_component_values={feature_id: 3.0},
        classical_payload={},
        baseline_registry=BaselineRegistry(),
        lead_names=(),
        signal_quality=.95,
        delineation=_technical_delineation(
            rhythm_confidence=.59,
            p_wave_coverage=.99,
        ),
    )[0]
    assert finding["classification"] == "not_evaluable"
    assert finding["sensitivity"] is None
    assert finding["specificity"] is None
    assert finding["sensitivity_source"] == "unknown"
    assert finding["specificity_source"] == "unknown"
    assert finding["value"] is None
    assert finding["analysis_restriction"]["restricted_marker_ids"] == [
        feature_id
    ]
    assert finding["analysis_restriction"]["abstention_reason"]
    assert any(
        "Rhythmussicherheit" in reason
        for reason in finding["applicability_reasons_de"]
    )


def test_criterion_catalog_keeps_formula_and_train_threshold_provenance() -> None:
    rng = np.random.default_rng(7102)
    labels = np.asarray([0] * 80 + [1] * 80, dtype=np.int8)
    values = labels + rng.normal(0.0, 0.12, len(labels))
    feature_id = "qrs_t_hausdorff_distance"
    train = pd.DataFrame(
        {
            "subject_id": np.arange(len(labels)),
            "label": labels,
            feature_id: values,
        }
    )
    model = fit_marker_model(values, labels, feature_id)
    screening = apply_fdr_by_family(
        pd.DataFrame(
            [
                {
                    "endpoint_id": "acute_mi_high_specificity",
                    "evaluation_split": split,
                    "panel_id": "primary_core",
                    "analysis_type": "single_marker",
                    "feature_id": feature_id,
                    "feature_role": "candidate_biomarker",
                    "scientific_tier": "candidate",
                    "endpoint_feature_role": "candidate_biomarker",
                    "post_outcome_hypothesis": False,
                    "exploratory_score": False,
                    "conceptual_family": "qrs_t_distance",
                    "roc_auc": 0.97,
                    "roc_auc_ci_low": 0.94,
                    "roc_auc_ci_high": 0.99,
                    "average_precision": 0.96,
                    "p_value": 0.000001,
                    "inference_status": "formal_inference",
                    "missingness": 0.01,
                    "test_yield": 0.99,
                    "assessability_gate_passed": True,
                    "missingness_auc_lower_bound": 0.94,
                    "missingness_auc_upper_bound": 0.99,
                    "missingness_sensitivity_gate_passed": True,
                    "specificity_90_sensitivity": 0.91,
                    "specificity_90_specificity": 0.90,
                }
                for split in ("train", "validation")
            ]
        ),
        family_columns=("endpoint_id", "panel_id", "evaluation_split"),
    )
    adjusted = pd.DataFrame(
        [
            {
                "endpoint_id": "acute_mi_high_specificity",
                "evaluation_split": "validation",
                "vcg_feature_id": feature_id,
                "model_id": "vcg_adjusted",
                "metric_scope": "validation_holdout",
                "incremental_comparison_status": "formal_inference",
                "baseline_model_id": "covariates",
                "automatic_feature_selection": False,
                "events_per_parameter_train": 30.0,
                "incremental_auc": 0.04,
                "incremental_auc_ci_low": 0.01,
                "incremental_auc_ci_high": 0.07,
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "endpoint_id": "acute_mi_high_specificity",
                "analysis_type": "single_marker",
                "feature_id": feature_id,
                "direction_agreement": True,
            }
        ]
    )

    catalog = build_endpoint_criterion_catalog(
        endpoint_id="acute_mi_high_specificity",
        train=train,
        screening=screening,
        combinations=pd.DataFrame(),
        stability=stability,
        marker_models={feature_id: model},
        score_payloads=(),
        adjusted_results=adjusted,
        development_run_id="development-test",
        snapshot_hash="snapshot-123",
        bootstrap_iterations=80,
        maximum_single_candidates=1,
    )

    assert len(catalog.candidates) == 1
    pending = catalog.candidates.iloc[0]
    assert pending["stage"] == CriterionStage.TRAIN_STABLE.value
    assert not pending["research_diagnosis_allowed"]
    finalized = finalize_criterion_catalog_evidence(
        catalog.candidates,
        screening,
        cross_fit_results=pd.DataFrame(
            [
                {
                    "endpoint_id": "acute_mi_high_specificity",
                    "panel_id": "primary_core",
                    "feature_id": feature_id,
                    "folds_available": 2,
                    "direction_consistent_across_folds": True,
                    "confirmatory_assessability_gate_passed": True,
                    "cross_fit_confirmed": True,
                    "p_value": 0.002,
                    "q_value": 0.004,
                    "global_q_value": 0.006,
                    "hierarchical_q_value": 0.008,
                    "pooled_effect": 0.72,
                    "pooled_effect_ci_low": 0.68,
                    "pooled_effect_ci_high": 0.76,
                }
            ]
        ),
        verification_bias_results=pd.DataFrame(
            [
                {
                    "endpoint_id": "acute_mi_high_specificity",
                    "panel_id": "primary_core",
                    "feature_id": feature_id,
                    "evaluation_split": "validation",
                    "operating_point": "specificity_95",
                    "verification_bias_gate_passed": True,
                    "verification_rate": 0.92,
                    "sensitivity_lower_bound": 0.70,
                    "sensitivity_upper_bound": 0.80,
                    "specificity_lower_bound": 0.91,
                    "specificity_upper_bound": 0.97,
                }
            ]
        ),
    )
    candidate = finalized.iloc[0]
    assert candidate["stage"] == CriterionStage.VALIDATION_REPLICATED.value
    assert candidate["core_replication_evidence_passed"]
    assert candidate["project_wide_multiplicity_confirmed"]
    assert candidate["cross_fit_confirmed"]
    assert candidate["verification_bias_gate_passed"]
    assert candidate["adjustment_confirmed"]
    assert candidate["locked_test_eligible"]
    assert candidate["formula"]
    assert candidate["unit"] == "mV"
    assert candidate["threshold_source"] == "train"
    assert candidate["research_diagnosis_allowed"]
    assert len(catalog.thresholds) == 3
    assert set(catalog.thresholds["threshold_source"]) == {"train"}
    assert (
        catalog.thresholds["gray_zone_high"]
        >= catalog.thresholds["gray_zone_low"]
    ).all()
    assert (
        catalog.thresholds["successful_bootstrap_iterations"] > 0
    ).all()
    model_row = {
        "endpoint_id": "acute_mi_high_specificity",
        "panel_id": "primary_core",
        **model.to_dict(),
    }
    shortlist = _build_locked_test_marker_shortlist(
        finalized,
        [model_row],
        development_run_id="development-test",
        snapshot_hash="snapshot-123",
    )
    assert shortlist["entry_count"] == 1
    assert shortlist["entries"][0]["feature_id"] == feature_id
    broad_registry = [
        model_row,
        {
            **model_row,
            "feature_id": "unselected_marker",
        },
    ]
    locked_models = _models_for_locked_shortlist(shortlist, broad_registry)
    assert [
        locked_model.feature_id
        for _, locked_model in locked_models["acute_mi_high_specificity"]
    ] == [feature_id]


def test_locked_test_freeze_contract_refuses_an_empty_shortlist(tmp_path: Path) -> None:
    payload = _build_locked_test_marker_shortlist(
        pd.DataFrame(),
        [],
        development_run_id="empty-development",
        snapshot_hash="snapshot-empty",
    )
    (tmp_path / "locked_test_marker_shortlist.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (tmp_path / "trained_marker_models.json").write_text(
        json.dumps({"models": []}),
        encoding="utf-8",
    )
    (tmp_path / "cross_fit_marker_models.json").write_text(
        json.dumps({"models": [], "cross_fitting_version": "2.0.0"}),
        encoding="utf-8",
    )
    pd.DataFrame().to_parquet(tmp_path / "criterion_candidates.parquet")

    with pytest.raises(RuntimeError, match="Test remains sealed"):
        _load_and_validate_locked_test_marker_shortlist(
            tmp_path,
            expected_run_id="empty-development",
            require_nonempty=True,
        )


def test_deep_grouped_search_returns_distinct_pairs_and_triples() -> None:
    rng = np.random.default_rng(7103)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 100)
    subjects = np.arange(len(labels), dtype=np.int64)
    names = (
        "qrs_t_hausdorff_distance",
        "t_loop_area",
        "svg_magnitude",
        "qrs_loop_volume",
    )
    frame = pd.DataFrame(
        {
            names[0]: labels + rng.normal(0.0, 0.08, len(labels)),
            names[1]: -labels + rng.normal(0.0, 0.08, len(labels)),
            names[2]: labels + rng.normal(0.0, 0.08, len(labels)),
            names[3]: -labels + rng.normal(0.0, 0.08, len(labels)),
        }
    )
    families = dict(zip(names, ("qrs_t", "t_loop", "svg", "qrs_loop")))
    candidates = grouped_score_search(
        frame,
        labels=labels,
        subject_ids=subjects,
        feature_ids=names,
        family_for_feature=families.__getitem__,
        folds=5,
        max_components=3,
        representatives_per_family=1,
        selected_pairs=4,
        selected_triples=3,
        triple_seed_pairs=6,
    )

    component_sets = [candidate.components for candidate in candidates]
    assert len(component_sets) >= 5
    assert len(component_sets) == len(set(component_sets))
    assert {len(components) for components in component_sets} == {2, 3}
    assert all(candidate.stable for candidate in candidates)
    assert all(candidate.mean_auc > 0.98 for candidate in candidates)


def test_external_benchmark_registry_never_pools_incompatible_tasks(
    tmp_path,
) -> None:
    assert {spec.dataset_id for spec in BENCHMARK_SPECS} == {
        "ludb",
        "qtdb",
        "nstdb",
        "afdb",
        "incart",
        "staffiii",
    }
    fake_ludb = tmp_path / "ludb-1.0.1"
    fake_ludb.mkdir()
    (fake_ludb / "RECORDS").write_text("", encoding="utf-8")
    (fake_ludb / "ludb.csv").write_text("", encoding="utf-8")
    (fake_ludb / "data").mkdir()
    inventory = inventory_delineation_benchmarks(tmp_path)
    statuses = dict(zip(inventory["dataset_id"], inventory["status"]))
    assert statuses["ludb"] == "ready"
    assert statuses["qtdb"] == "missing"
    assert statuses["afdb"] == "missing"
    assert statuses["staffiii"] == "missing"

    output = tmp_path / "suite.json"
    payload = run_delineation_benchmark_suite(
        tmp_path,
        output,
        dataset_ids=("qtdb",),
    )
    assert payload["status"] == "incomplete_missing_datasets"
    assert payload["locked_test_read"] is False
    assert payload["implementation_hash"] == (
        benchmark_suite.delineation_implementation_hash()
    )
    assert payload["release_core_complete"] is False
    assert "No pooled score" in payload["aggregation_policy"]
    assert output.is_file()


def test_qtdb_parser_preserves_manual_p_qrs_t_u_triplets(monkeypatch) -> None:
    annotation = SimpleNamespace(
        sample=np.asarray(
            [10, 20, 30, 40, 50, 60, 70, 90, 110, 120, 135, 150]
        ),
        symbol=("(", "p", ")", "(", "N", ")", "(", "t", ")", "(", "u", ")"),
        num=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    )
    monkeypatch.setattr(benchmark_suite.wfdb, "rdann", lambda *_args: annotation)
    parsed = benchmark_suite._read_qtdb_references(Path("synthetic-qtdb-record"))

    assert parsed == {
        "p": [(10, 20, 30)],
        "qrs": [(40, 50, 60)],
        "t": [(70, 90, 110)],
        "u": [(120, 135, 150)],
    }


def test_qtdb_parser_preserves_missing_t_u_onsets_and_qrs_beat_class(
    monkeypatch,
) -> None:
    annotation = SimpleNamespace(
        sample=np.asarray([10, 20, 30, 50, 60, 80, 100, 115, 130]),
        symbol=("(", "N", ")", "t", ")", "u", ")", "(", "V"),
        num=np.asarray([1, 6, 1, 0, 2, 0, 3, 1, 0]),
    )
    monkeypatch.setattr(benchmark_suite.wfdb, "rdann", lambda *_args: annotation)

    parsed = benchmark_suite._read_qtdb_references(Path("synthetic-qtdb-record"))

    assert parsed["qrs"] == [(10, 20, 30)]
    assert parsed["t"] == [(None, 50, 60)]
    assert parsed["u"] == [(None, 80, 100)]


def test_qtdb_reference_windows_are_padded_merged_and_non_overlapping() -> None:
    references = {
        "p": [(100, 110, 120), (500, 510, 520)],
        "qrs": [(140, 150, 160), (540, 550, 560)],
        "t": [],
        "u": [],
    }

    windows = benchmark_suite._qtdb_reference_windows(
        references,
        fs=100,
        signal_length=2_000,
        margin_seconds=1.0,
        merge_gap_seconds=0.5,
    )

    assert windows == [(0, 261), (400, 661)]
    assert all(left[1] < right[0] for left, right in zip(windows, windows[1:]))


def test_qtdb_window_reference_coordinates_are_shifted_exactly_once() -> None:
    references = {
        "p": [(100, 110, 120), (500, 510, 520)],
        "qrs": [(140, 150, 160)],
        "t": [],
        "u": [],
    }

    first = benchmark_suite._qtdb_references_in_window(
        references,
        start=90,
        end=200,
    )
    second = benchmark_suite._qtdb_references_in_window(
        references,
        start=400,
        end=600,
    )

    assert first["p"] == [(10, 20, 30)]
    assert first["qrs"] == [(50, 60, 70)]
    assert second["p"] == [(100, 110, 120)]
    assert sum(len(first[wave]) + len(second[wave]) for wave in references) == 3


def test_noise_stress_record_names_map_to_explicit_snr_strata() -> None:
    assert benchmark_suite._nstdb_noise_snr_db("118e_6") == -6
    assert benchmark_suite._nstdb_noise_snr_db("119e00") == 0
    assert benchmark_suite._nstdb_noise_snr_db("118e24") == 24
    assert benchmark_suite._nstdb_noise_snr_db("clean") is None


def test_phenome_adjustment_separates_robust_and_confounded_hits() -> None:
    rng = np.random.default_rng(7104)
    rows = []
    current = []
    history = []
    for index in range(600):
        positive = index < 300
        subject_id = index + 1
        age = rng.normal(74.0 if positive else 55.0, 7.0)
        rows.append(
            {
                "subject_id": subject_id,
                "study_id": subject_id + 10_000,
                "split": "validation",
                "is_control_index": not positive,
                "is_case_index": positive,
                "age": age,
                "sex": "M" if index % 2 else "F",
                "race": "audit-only",
                "signal_quality_score": rng.uniform(0.70, 1.0),
                "robust_marker": (
                    0.85 * positive + rng.normal(0.0, 0.65)
                ),
                "age_confounded_marker": age + rng.normal(0.0, 2.5),
            }
        )
        if positive:
            current.append(
                {
                    "subject_id": subject_id,
                    "study_id": subject_id + 10_000,
                    "split": "validation",
                    "phenotype_id": "P1",
                }
            )
            history.append(
                {"subject_id": subject_id, "body_system": "circulatory"}
            )
    replicated = pd.DataFrame(
        [
            {
                "phenotype_id": "P1",
                "phenotype_name": "Synthetic condition",
                "body_system": "circulatory",
                "panel_id": "primary_core",
                "feature_id": feature_id,
                "train_direction": 1,
                "replicated": True,
            }
            for feature_id in ("robust_marker", "age_confounded_marker")
        ]
    )
    adjusted = confirm_replicated_phenome_hits(
        replicated,
        pd.DataFrame(rows),
        pd.DataFrame(current),
        pd.DataFrame(history),
        policy=PhenomeAdjustmentPolicy(
            minimum_events_per_parameter=5,
            fdr_alpha=0.05,
        ),
    ).set_index("feature_id")

    assert bool(adjusted.loc["robust_marker", "adjustment_robust"])
    assert adjusted.loc["robust_marker", "adjusted_or_ci_low"] > 1.0
    assert not bool(
        adjusted.loc["age_confounded_marker", "adjustment_robust"]
    )
    assert bool(adjusted.loc["age_confounded_marker", "likely_confounded"])

    train_hits = replicated.assign(q_value=0.001, aligned_auc=0.72)
    validation = replicated.assign(
        q_value=0.001,
        roc_auc=0.70,
        validation_aligned_auc=0.70,
        direction_consistent=True,
        inference_status="formal_inference",
    )
    statuses = build_phenome_discovery_status(
        train_hits,
        validation,
        adjusted.reset_index(),
    ).set_index("feature_id")
    assert statuses.loc["robust_marker", "status_de"] == "repliziert"
    assert statuses.loc[
        "robust_marker", "association_strength"
    ] == pytest.approx(0.40)
    assert (
        statuses.loc["age_confounded_marker", "status_de"]
        == "wahrscheinlich confounded"
    )


def test_phenome_adjustment_refuses_locked_test_rows() -> None:
    replicated = pd.DataFrame(
        [
            {
                "phenotype_id": "P1",
                "body_system": "circulatory",
                "panel_id": "primary_core",
                "feature_id": "marker",
                "train_direction": 1,
            }
        ]
    )
    features = pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [2],
            "split": ["locked_test"],
        }
    )
    import pytest

    with pytest.raises(RuntimeError, match="Locked-test"):
        confirm_replicated_phenome_hits(
            replicated,
            features,
            pd.DataFrame(),
            pd.DataFrame(),
        )


def test_result_repository_filters_full_run_without_cross_run_fallback(
    tmp_path,
) -> None:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic-iv",
        mimic_ed_root=tmp_path / "mimic-ed",
        mimic_ecg_root=tmp_path / "mimic-ecg",
        echo_structured_path=tmp_path / "echo.csv.gz",
        echo_study_list_path=tmp_path / "echo-study.csv",
        echo_record_list_path=tmp_path / "echo-record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    run_a = config.runs_root / "run-a"
    run_a.mkdir()
    screening = run_a / "screening_results.parquet"
    explorer = run_a / "pathology_marker_explorer.parquet"
    pd.DataFrame(
        {"analysis_completeness": ["complete_snapshot"]}
    ).to_parquet(screening, index=False)
    pd.DataFrame(
        {
            "endpoint_id": [
                "acute_mi_high_specificity",
                "echo_lvh_strict",
            ],
            "endpoint_name_de": [
                "Akuter Herzinfarkt, strenger Rückblick-Proxy",
                "Linksherzmuskelverdickung im Echo, streng",
            ],
            "endpoint_definition_de": ["MI proxy", "LVH echo"],
            "pathology_group": ["ACS / OMI", "LVH / Remodeling"],
            "pathology_group_id": [
                "acs_omi_stemi_equivalent",
                "lvh_remodeling",
            ],
            "feature_id": [
                "qrs_t_hausdorff_distance",
                "spatial_qrs_t_angle_deg",
            ],
            "feature_name_de": ["QRS-T-Distanz", "Räumlicher QRS-T-Winkel"],
            "conceptual_family": ["qrs_t", "qrs_t"],
            "panel_id": ["primary_core", "primary_core"],
            "candidate_type": ["vcg_single", "vcg_single"],
            "evaluation_split": ["validation", "validation"],
            "evidence_grade": ["A", "B"],
            "evidence_rank": [450.0, 320.0],
            "roc_auc": [0.82, 0.73],
            "sensitivity_at_95_specificity": [0.81, 0.44],
            "operating_specificity": [0.95, 0.95],
            "mean_signal_quality": [0.91, 0.89],
            "q_value": [0.001, 0.01],
            "historical_pre_audit_result": [False, False],
            "analysis_completeness": [
                "complete_snapshot",
                "complete_snapshot",
            ],
        }
    ).to_parquet(explorer, index=False)
    StageManifest.complete(
        stage="validate",
        stage_version=VALIDATION_STAGE_VERSION,
        input_hash="a" * 64,
        config_hash="b" * 64,
        output_paths=(str(screening.resolve()), str(explorer.resolve())),
    ).save(run_a / "validate.manifest.json")
    provenance = {
        "code_hash": "validation-current",
        "feature_integrity": {"source_code_hash": "feature-current"},
    }
    (run_a / "development_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )

    repository = ResultRepository(
        config,
        expected_feature_code_hash="feature-current",
        expected_validation_code_hash="validation-current",
    )
    page = repository.query(
        ResultQuery(
            perspective="pathologies",
            run_id="run-a",
            search_text="Herzinfarkt",
            pathology_group_ids=("acs_omi_stemi_equivalent",),
            minimum_sensitivity=0.80,
            minimum_signal_quality=0.90,
            maximum_q_value=0.01,
        )
    )
    assert page.status == "ok"
    assert page.total_count == 1
    assert page.rows.iloc[0]["endpoint_id"] == "acute_mi_high_specificity"
    assert page.unsupported_filters == ()

    pd.DataFrame(
        {"analysis_completeness": ["partial_preview"]}
    ).to_parquet(screening, index=False)
    pilot_blocked = repository.query(
        ResultQuery(perspective="pathologies", run_id="run-a")
    )
    assert pilot_blocked.status == "partial_or_incompatible_run"
    pilot_visible = repository.query(
        ResultQuery(
            perspective="pathologies",
            run_id="run-a",
            complete_runs_only=False,
        )
    )
    assert pilot_visible.status == "ok"
    assert pilot_visible.analysis_completeness == "partial_preview"

    run_b = config.runs_root / "run-b"
    run_b.mkdir()
    screening_b = run_b / "screening_results.parquet"
    pd.DataFrame(
        {"analysis_completeness": ["complete_snapshot"]}
    ).to_parquet(screening_b, index=False)
    StageManifest.complete(
        stage="validate",
        stage_version=VALIDATION_STAGE_VERSION,
        input_hash="c" * 64,
        config_hash="d" * 64,
        output_paths=(str(screening_b.resolve()),),
    ).save(run_b / "validate.manifest.json")
    (run_b / "development_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    missing = repository.query(
        ResultQuery(perspective="pathologies", run_id="run-b")
    )
    assert missing.status == "artifact_not_available"
    assert missing.total_count == 0
    assert missing.artifact_path is None
