from __future__ import annotations

import pandas as pd

from vcg_biomarker.mimic.evidence_contract import (
    build_candidate_evidence_contract,
)


def _criterion(**updates):
    row = {
        "criterion_id": "c1",
        "endpoint_id": "endpoint",
        "feature_id": "marker",
        "stage": "validation_replicated",
        "inference_status": "formal_inference",
        "assessability_gate_passed": True,
        "missingness_sensitivity_gate_passed": True,
        "adjustment_confirmed": True,
        "threshold_source": "train",
        "specificity_95_threshold": 1.2,
        "roc_auc": 0.72,
        "roc_auc_ci_low": 0.67,
        "roc_auc_ci_high": 0.77,
        "q_value": 0.01,
        "n_observations": 500,
        "n_positive": 125,
        "development_run_id": "run-1",
        "snapshot_hash": "snapshot-1",
        "analysis_completeness": "complete_snapshot",
        "reference_kind": "timed_quantitative",
        "independent_reference_passed": True,
    }
    row.update(updates)
    return pd.DataFrame([row])


def _reporting():
    return pd.DataFrame(
        [
            {
                "marker_id": "marker",
                "estimable_stratum_share": 0.8,
                "icc": 0.9,
                "repeatability_coefficient": 0.3,
                "transform_agreement_class": "interchangeable_after_conversion",
            }
        ]
    )


def _adjusted():
    return pd.DataFrame(
        [
            {
                "endpoint_id": "endpoint",
                "vcg_feature_id": "marker",
                "evaluation_split": "validation",
                "model_id": "classical_plus_vcg",
                "incremental_comparison_status": "formal_inference",
                "incremental_auc": 0.03,
                "incremental_auc_ci_low": 0.01,
            },
            {
                "endpoint_id": "endpoint",
                "vcg_feature_id": "marker",
                "evaluation_split": "validation",
                "model_id": "vcg_adjusted",
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            },
        ]
    )


def _subgroups():
    result = pd.DataFrame(
        [
            {"endpoint_id": "endpoint", "feature_id": "marker", "analysis_type": "subgroup_stability", "evaluation_split": "validation", "subgroup_variable": "sex", "subgroup_level": "F", "roc_auc": 0.70},
            {"endpoint_id": "endpoint", "feature_id": "marker", "analysis_type": "subgroup_stability", "evaluation_split": "validation", "subgroup_variable": "sex", "subgroup_level": "M", "roc_auc": 0.74},
        ]
    )
    result["n_positive"] = 150
    result["n_negative"] = 500
    result["roc_auc_ci_low"] = result["roc_auc"] - .02
    result["roc_auc_ci_high"] = result["roc_auc"] + .02
    return result


def test_all_internal_gates_promote_only_to_guideline_candidate():
    result = build_candidate_evidence_contract(
        criteria=_criterion(),
        reporting_set=_reporting(),
        adjusted_models=_adjusted(),
        screening=_subgroups(),
        decision_curves=pd.DataFrame(),
    ).iloc[0]
    assert result["evidence_tier"] == "guideline_candidate"
    assert not result["external_validation_passed"]
    assert not result["net_benefit_passed"]
    assert result["model_id"] == "single_marker_roc_auc"
    assert result["events"] == 125
    assert result["gate_status"] == "all_internal_candidate_gates_passed"
    assert result["run_id"] == "run-1"


def test_replication_never_hides_missing_candidate_gates():
    result = build_candidate_evidence_contract(
        criteria=_criterion(adjustment_confirmed=False),
        reporting_set=pd.DataFrame(),
        adjusted_models=pd.DataFrame(),
        screening=pd.DataFrame(),
        decision_curves=pd.DataFrame(),
    ).iloc[0]
    assert result["evidence_tier"] == "replicated_association"
    assert "adjustment" in result["failed_gates"]
    assert "repeatability" in result["failed_gates"]


def test_external_label_alone_cannot_promote_without_net_benefit():
    result = build_candidate_evidence_contract(
        criteria=_criterion(stage="external_validated"),
        reporting_set=_reporting(),
        adjusted_models=_adjusted(),
        screening=_subgroups(),
        decision_curves=pd.DataFrame(),
    ).iloc[0]
    assert result["evidence_tier"] == "guideline_candidate"
    assert not result["external_validation_passed"]
    assert not result["net_benefit_passed"]
