import pytest

from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
from vcg_biomarker.mimic.mechanism_hypotheses import mechanism_hypothesis_rows
from vcg_biomarker.mimic.labels import _label_rules


def test_frozen_as_questions_have_real_marker_and_endpoint_identities():
    payload = marker_freeze_payload()
    rows = payload["mechanism_hypotheses"]
    names = {row["marker_id"] for row in payload["markers"]}
    assert len(rows) == len({(r["marker_id"], r["endpoint_id"]) for r in rows}) == 77
    for row in rows:
        assert row["marker_id"] in names
        assert row["endpoint_id"] in _label_rules()
        assert row["post_outcome_hypothesis"]
        assert not row["clinical_claim_allowed"]
        assert row["analysis_execution_status"] == "not_asserted_by_inventory"
        assert row["evidence_tier"] == "exploratory"


def test_ar_questions_use_distinct_provenance_and_clinical_limits():
    import json
    import pandas as pd
    from vcg_biomarker.mimic.mechanism_hypotheses import annotate_mechanism_hypotheses
    frame = pd.DataFrame({"feature_id": ["qrs_t_opposed_energy_fraction"]*2,
                          "endpoint_id": ["aortic_regurgitation", "aortic_stenosis"]})
    result = annotate_mechanism_hypotheses(frame)
    ar, aortic_stenosis = [json.loads(value) for value in result.mechanism_hypothesis_json]
    assert ar["origin"].endswith("09B_AORTIC_REGURGITATION.md")
    assert aortic_stenosis["origin"].endswith("09_AORTIC_STENOSIS.md")
    assert "not_regurgitant_volume" in ar["reference_target"]
    assert "coexisting_valve_disease" in ar["required_comparisons"]
    assert ar["claim_limit"] != aortic_stenosis["claim_limit"]


def test_missing_implementation_is_not_silently_dropped():
    with pytest.raises(ValueError, match="Unfrozen"):
        mechanism_hypothesis_rows([])


def test_mitral_questions_separate_atrial_and_ventricular_context():
    payload = marker_freeze_payload()
    rows = [row for row in payload["mechanism_hypotheses"]
            if row["endpoint_id"].startswith("mitral_")]
    assert len(rows) == 12
    for row in rows:
        assert row["origin"].endswith("10_MITRAL_DISEASE.md")
        if row["marker_id"].startswith("p_"):
            assert "P_segment_quality" in row["required_measurement_context"]
            assert "LA_volume_mediator_sensitivity" in row["required_comparisons"]
        else:
            assert row["endpoint_id"].startswith("mitral_regurgitation")
            assert "LV_function_and_mass" in row["required_comparisons"]
        assert "arrhythmia_prediction" in row["claim_limit"]
    markers = [row for row in payload["markers"]
               if row["marker_id"] != "p_loop_terminal_area_fraction"]
    with pytest.raises(ValueError, match="Unfrozen mitral"):
        mechanism_hypothesis_rows(markers)


def test_reference_limits_survive_freeze_to_screening_without_promotion():
    import json
    import pandas as pd
    from vcg_biomarker.mimic.mechanism_hypotheses import annotate_mechanism_hypotheses

    rows = marker_freeze_payload()["mechanism_hypotheses"]
    frame = pd.DataFrame({"feature_id": [r["marker_id"] for r in rows],
                          "endpoint_id": [r["endpoint_id"] for r in rows]})
    output = annotate_mechanism_hypotheses(frame)
    for frozen, encoded in zip(rows, output.mechanism_hypothesis_json):
        actual = json.loads(encoded)
        limits = actual["reference_limitations"]
        assert limits == frozen["reference_limitations"]
        assert all(value is False for value in limits.values())
        assert actual["evidence_tier"] == "exploratory"
        assert actual["clinical_claim_allowed"] is False
        if actual.get("research_origins"):
            assert actual["origin"] in actual["research_origins"]
            assert limits == {"causal_mechanism_verified": False, "clinical_threshold_validated": False}
            assert "rhythm_and_pacing" in actual["required_comparisons"]
        elif actual["origin"].endswith("PRIORITY_MARKER_IMPLEMENTATION.md"):
            assert limits == {"causal_mechanism_verified": False, "clinical_threshold_validated": False}
            assert "rhythm_and_pacing" in actual["required_comparisons"]
        elif frozen["endpoint_id"] == "copd_icd":
            assert "post_bronchodilator_obstruction_verified" in limits
            assert "BMI_and_smoking" in actual["required_comparisons"]
            assert "administrative_association" in actual["claim_limit"]
        elif actual["origin"].endswith("12_PULMONARY_HYPERTENSION.md"):
            assert set(limits) == {"invasive_PH_confirmed", "PAH_etiology_confirmed",
                                   "isolated_RV_electrical_source_measured"}
            assert "echo_method_and_temporal_alignment" in actual["required_measurement_context"]
        elif frozen["endpoint_id"].startswith("tricuspid_"):
            assert set(limits) == {"cause_verified", "lead_induced_damage_verified",
                                   "invasive_pressure_measured_by_endpoint"}
            assert "pacing_and_capture_status" in actual["required_measurement_context"]
            assert "RV_size_and_function" in actual["required_comparisons"]
        elif frozen["endpoint_id"].startswith("mitral_"):
            assert "cause_verified" in limits
            assert actual["required_measurement_context"]
        elif frozen["endpoint_id"].startswith("aortic_regurgitation"):
            assert set(limits) == {"onset_verified", "cause_verified",
                                   "acute_disease_diagnosis_supported",
                                   "regurgitant_volume_measured_by_endpoint"}
        elif frozen["endpoint_id"].endswith("_velocity"):
            assert "anatomic_severity_exclusion_supported" in limits
        else:
            assert set(limits) == {"valve_area_measured_by_endpoint"}


def test_annotation_preserves_rows_statistics_and_evidence_even_for_duplicate_indices():
    import json
    import pandas as pd
    from vcg_biomarker.mimic.mechanism_hypotheses import annotate_mechanism_hypotheses
    frame = pd.DataFrame({"feature_id": ["qrs_t_opposed_energy_fraction", "other"],
                          "endpoint_id": ["aortic_stenosis", "aortic_stenosis"],
                          "p_value": [.2, .01], "evidence_stage": ["exploratory"]*2,
                          "status": ["non_estimable", "ok"]}, index=[7, 7])
    result = annotate_mechanism_hypotheses(frame)
    pd.testing.assert_frame_equal(result.drop(columns="mechanism_hypothesis_json"), frame)
    assert result.iloc[1].mechanism_hypothesis_json is None
    metadata = json.loads(result.iloc[0].mechanism_hypothesis_json)
    assert metadata["analysis_execution_status"] == "screening_row_present_not_confirmation"
    assert not metadata["clinical_claim_allowed"]
    pd.testing.assert_frame_equal(annotate_mechanism_hypotheses(result), result)


def test_empty_or_incomplete_results_do_not_generate_hypothesis_results():
    import pandas as pd
    from vcg_biomarker.mimic.mechanism_hypotheses import annotate_mechanism_hypotheses
    assert annotate_mechanism_hypotheses(pd.DataFrame()).empty
    result = annotate_mechanism_hypotheses(pd.DataFrame({"endpoint_id": ["aortic_stenosis"]}))
    assert len(result) == 1 and result.mechanism_hypothesis_json.isna().all()


def test_family_is_closed_and_the_measured_near_alias_travels_with_its_rows():
    from vcg_biomarker.mimic.mechanism_hypotheses import (
        FROZEN_HYPOTHESIS_COUNT,
        MEASURED_NEAR_ALIASES,
    )

    payload = marker_freeze_payload()
    rows = payload["mechanism_hypotheses"]
    assert len(rows) == FROZEN_HYPOTHESIS_COUNT
    opposed = [row for row in rows if row["marker_id"] == "qrs_t_opposed_energy_fraction"]
    assert opposed
    for row in opposed:
        assert row["measured_near_alias"]["near_alias_of"] == "spatial_qrs_t_angle_deg"
        assert row["measured_near_alias"]["pilot_spearman_rho"] == 0.90
        # The verification flags stay what they are: flags.
        assert all(value is False for value in row["reference_limitations"].values())
    others = [row for row in rows if row["marker_id"] not in MEASURED_NEAR_ALIASES]
    assert all("measured_near_alias" not in row for row in others)
    # An appended binding must fail loudly.
    markers = [{"marker_id": row["marker_id"]} for row in payload["markers"]]
    import vcg_biomarker.mimic.mechanism_hypotheses as module
    original = module.FROZEN_HYPOTHESIS_COUNT
    module.FROZEN_HYPOTHESIS_COUNT = original + 1
    try:
        with pytest.raises(ValueError, match="frozen family"):
            mechanism_hypothesis_rows(markers)
    finally:
        module.FROZEN_HYPOTHESIS_COUNT = original
