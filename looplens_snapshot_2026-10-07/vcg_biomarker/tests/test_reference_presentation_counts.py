from vcg_biomarker.mimic.result_contract import scientific_result_contract


def test_independent_modality_does_not_assert_clock_alignment_or_release():
    result = scientific_result_contract({"reference_kind": "timed_quantitative",
        "n_independent_modality_observed": 120, "n_reference_alignment_confirmed": 0})
    assert result.independent_modality_observed_n == 120
    assert result.clock_alignment_confirmed_n == 0
    assert result.timing_evidence_note == "no_clock_alignment_confirmed"
    assert not result.clinical_claim_allowed
    assert "independent_reference" in result.failed_gates


def test_legacy_missing_counts_are_not_zero_or_synchronization_evidence():
    result = scientific_result_contract({"n_reference_alignment_confirmed": float("nan")})
    assert result.clock_alignment_confirmed_n is None
    assert result.independent_modality_observed_n is None
    assert result.timing_evidence_note == "not_reported"
