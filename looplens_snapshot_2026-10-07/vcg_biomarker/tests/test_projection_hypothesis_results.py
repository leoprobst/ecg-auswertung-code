import numpy as np
import pandas as pd

from vcg_biomarker.mimic.projection_hypotheses import (
    PROJECTION_RESTRICTION_VERSION,
    P_PROJECTION_RESTRICTION_STATUS,
    annotate_projection_hypothesis_frame,
    projection_analysis_eligibility,
    projection_case_eligibility,
)


def test_frozen_projection_family_is_annotated_and_corrected_per_split():
    frame = pd.DataFrame([
        {"feature_id": "t_loop_frontal_xy_area", "endpoint_id": "hyperkalemia_near_ecg",
         "evaluation_split": "validation", "p_value": .001, "inference_status": "formal_inference"},
        {"feature_id": "t_loop_horizontal_xz_area", "endpoint_id": "hyperkalemia_near_ecg",
         "evaluation_split": "validation", "p_value": .04, "inference_status": "formal_inference"},
        {"feature_id": "t_loop_sagittal_yz_area", "endpoint_id": "hyperkalemia_near_ecg",
         "evaluation_split": "validation", "p_value": .02, "inference_status": "descriptive_only"},
        {"feature_id": "t_loop_frontal_xy_area", "endpoint_id": "hyperkalemia_near_ecg",
         "evaluation_split": "train", "p_value": .9, "inference_status": "formal_inference"},
        {"feature_id": "qrs_loop_frontal_xy_area", "endpoint_id": "hyperkalemia_near_ecg",
         "evaluation_split": "validation", "p_value": .001, "inference_status": "formal_inference"},
        {"feature_id": "heart_rate_bpm", "endpoint_id": "echo_lae",
         "evaluation_split": "validation", "p_value": .001, "inference_status": "formal_inference"},
    ])
    result = annotate_projection_hypothesis_frame(frame)
    validation = result.iloc[:3]
    assert validation.projection_hypothesis_prespecified.all()
    assert validation.projection_multiplicity_family.eq("fixed_2d_projection__hyperkalemia_near_ecg").all()
    assert validation.projection_reference_modality.eq("laboratory").all()
    assert result.iloc[0].projection_hypothesis_q_value == .002
    assert result.iloc[1].projection_hypothesis_q_value == .04
    assert np.isnan(result.iloc[2].projection_hypothesis_q_value)
    assert result.iloc[3].projection_hypothesis_q_value == .9
    assert not result.iloc[4:].projection_hypothesis_prespecified.any()
    assert result.iloc[4:].projection_hypothesis_q_value.isna().all()


def test_p_loop_hypothesis_requires_proof_that_rhythm_restriction_was_applied():
    frame = pd.DataFrame([{"feature_id": "p_loop_frontal_xy_area", "endpoint_id": "echo_lae",
        "evaluation_split": "validation", "p_value": 1e-8, "inference_status": "formal_inference"}])
    result = annotate_projection_hypothesis_frame(frame).iloc[0]
    assert bool(result.projection_hypothesis_prespecified)
    assert result.projection_hypothesis_restriction_operationalized == P_PROJECTION_RESTRICTION_STATUS
    assert np.isnan(result.projection_hypothesis_q_value)
    assert not bool(result.projection_hypothesis_fdr_significant)


def test_p_loop_hypothesis_is_formal_only_after_matching_restriction_contract():
    frame = pd.DataFrame([{
        "feature_id": "p_loop_frontal_xy_area",
        "endpoint_id": "echo_lae",
        "evaluation_split": "validation",
        "p_value": 1e-8,
        "inference_status": "formal_inference",
        "analysis_restriction_status": P_PROJECTION_RESTRICTION_STATUS,
        "analysis_restriction_version": PROJECTION_RESTRICTION_VERSION,
    }])
    result = annotate_projection_hypothesis_frame(frame).iloc[0]
    assert result.projection_hypothesis_q_value == 1e-8
    assert bool(result.projection_hypothesis_fdr_significant)


def test_p_loop_eligibility_uses_positive_concurrent_technical_evidence():
    frame = pd.DataFrame({
        "rhythm_context": [
            "regular",
            "wide_qrs_pattern",
            "irregular",
            "af_compatible_pattern",
            "regular",
            "regular",
        ],
        "rhythm_context_confidence": [.60, .90, .95, .95, .59, .90],
        "p_wave_coverage": [.50, .80, .90, .90, .90, .49],
    })
    eligible, contract = projection_analysis_eligibility(
        frame,
        "p_loop_frontal_xy_area",
    )
    assert eligible.tolist() == [True, True, False, False, False, False]
    assert contract["status"] == P_PROJECTION_RESTRICTION_STATUS
    assert contract["required_columns_present"] is True


def test_p_loop_eligibility_abstains_when_context_columns_are_missing():
    eligible, contract = projection_analysis_eligibility(
        pd.DataFrame({"rhythm_context": ["regular"]}),
        "p_loop_frontal_xy_area",
    )
    assert eligible.tolist() == [False]
    assert contract["status"] == "not_applied_missing_required_columns"
    assert set(contract["missing_columns"]) == {
        "rhythm_context_confidence",
        "p_wave_coverage",
    }


def test_single_case_projection_contract_matches_cohort_restriction():
    eligible, contract = projection_case_eligibility(
        ("p_loop_frontal_xy_area", "qrs_loop_frontal_xy_area"),
        rhythm_context="regular",
        rhythm_context_confidence=.60,
        p_wave_coverage=.50,
    )
    assert eligible
    assert contract["status"] == P_PROJECTION_RESTRICTION_STATUS
    assert contract["version"] == PROJECTION_RESTRICTION_VERSION
    assert contract["restricted_marker_ids"] == [
        "p_loop_frontal_xy_area"
    ]
    assert not contract["abstention_reason"]

    eligible, contract = projection_case_eligibility(
        ("p_loop_frontal_xy_area",),
        rhythm_context="af_compatible_pattern",
        rhythm_context_confidence=.99,
        p_wave_coverage=.99,
    )
    assert not eligible
    assert contract["abstention_reason"]


def test_single_case_non_p_projection_is_not_restricted():
    eligible, contract = projection_case_eligibility(
        ("qrs_loop_frontal_xy_area",),
        rhythm_context=None,
        rhythm_context_confidence=None,
        p_wave_coverage=None,
    )
    assert eligible
    assert contract["status"] == "not_applicable"


def test_empty_or_legacy_result_gets_safe_negative_role_without_fake_q_values():
    empty = annotate_projection_hypothesis_frame(pd.DataFrame())
    assert "projection_hypothesis_prespecified" in empty
    legacy = annotate_projection_hypothesis_frame(pd.DataFrame({"feature_id": ["p_loop_frontal_xy_area"]}))
    assert not legacy.projection_hypothesis_prespecified.iloc[0]
    assert np.isnan(legacy.projection_hypothesis_q_value.iloc[0])
