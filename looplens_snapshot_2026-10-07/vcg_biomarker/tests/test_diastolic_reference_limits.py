import duckdb
import pytest

from vcg_biomarker.mimic.labels import _echo_rules


@pytest.mark.parametrize("label", ["diastolic_relaxation_abnormality",
    "restrictive_filling_pattern", "elevated_filling_pressure_proxy"])
def test_diastolic_evidence_exposes_unverified_rhythm(label):
    rule = _echo_rules()[label]
    with duckdb.connect() as connection:
        result = connection.execute(f"""SELECT {rule.evidence_summary} FROM (
            SELECT 0.6 AS mitral_e_a_ratio, 220. AS mitral_e_deceleration_time_ms,
                   16. AS mitral_e_e_prime_ratio)""").fetchone()[0]
    assert "echo_rhythm_applicability=unverified" in result
    assert "not_HFpEF_diagnosis" in result


@pytest.mark.parametrize("endpoint", ["diastolic_relaxation_abnormality",
    "restrictive_filling_pattern", "elevated_filling_pressure_proxy"])
def test_unqualified_echo_proxy_cannot_pass_clinical_candidate_gate(endpoint):
    import pandas as pd
    from vcg_biomarker.mimic.evidence_contract import build_candidate_evidence_contract
    result = build_candidate_evidence_contract(
        criteria=pd.DataFrame([{"endpoint_id": endpoint, "feature_id": "test_marker"}]),
        reporting_set=pd.DataFrame(), adjusted_models=pd.DataFrame(),
        screening=pd.DataFrame(), decision_curves=pd.DataFrame())
    assert "echo_reference_applicability" in result.iloc[0]["failed_gates"].split("|")
    assert result.iloc[0]["gate_status"] == "required_gates_missing"


def test_statistical_success_cannot_override_echo_applicability():
    from vcg_biomarker.mimic.result_contract import (
        CANDIDATE_GATES, CLINICAL_GATES, scientific_result_contract)
    row = {f"{gate}_passed": True for gate in (*CANDIDATE_GATES, *CLINICAL_GATES)}
    row.update(endpoint_id="elevated_filling_pressure_proxy", reference_kind="timed_quantitative",
               analysis_completeness="complete_snapshot")
    contract = scientific_result_contract(row)
    assert contract.evidence_stage == "replicated_association"
    assert not contract.clinical_claim_allowed
    assert contract.failed_gates == "echo_reference_applicability"
