from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.result_contract import scientific_result_contract, CANDIDATE_GATES, CLINICAL_GATES
from vcg_biomarker.mimic.reference_tiers import (annotate_reference_rows, classify_reference_tier,
    TIER_A, TIER_UNKNOWN, draw_adjudication_sample, blinded_reviewer_packet, AdjudicationSamplePolicy)
from vcg_biomarker.mimic.external_validation import external_agreement, EXTERNAL_ENDPOINT_MAP
from vcg_biomarker.mimic.pilot_workspace import confined, WORKSPACE_VERSION
from vcg_biomarker.mimic.pilot_viewmodel import PilotViewModel
from vcg_biomarker.mimic.storage import file_sha256
from vcg_biomarker.mimic.evidence_contract import _net_benefit_passed


def test_auc_and_missing_boolean_can_never_create_evidence():
    row = {"roc_auc": .99, "q_value": 1e-90, "evidence_grade": "A",
           "analysis_completeness": "complete_snapshot", "reference_kind": "timed_quantitative"}
    row.update({f"{name}_passed": np.nan for name in (*CANDIDATE_GATES, *CLINICAL_GATES)})
    result = scientific_result_contract(row)
    assert result.evidence_stage == "exploratory"
    assert not result.clinical_claim_allowed


def test_all_gates_are_required_and_pilot_ceiling_is_unconditional():
    row = {f"{name}_passed": True for name in (*CANDIDATE_GATES, *CLINICAL_GATES)}
    row.update(analysis_completeness="complete_snapshot", reference_kind="timed_quantitative")
    assert scientific_result_contract(row).evidence_stage == "clinically_validated"
    for gate in CANDIDATE_GATES:
        assert scientific_result_contract({**row, f"{gate}_passed": False}).evidence_stage not in {"guideline_candidate", "clinically_validated"}
    assert scientific_result_contract({**row, "analysis_completeness": "partial_preview"}).evidence_stage == "exploratory"
    assert not scientific_result_contract({**row, "reference_kind": "same_ecg_expert"}).clinical_claim_allowed


def test_reference_dictionary_is_not_a_measurement_and_tables_are_not_independent():
    assert classify_reference_tier(["d_labitems"], evidence_available=True, temporally_eligible=True) == TIER_UNKNOWN
    assert classify_reference_tier(["structured-measurement"]) == TIER_UNKNOWN
    assert classify_reference_tier(["labevents", "d_labitems"], evidence_available=True, temporally_eligible=True) == TIER_A
    assert classify_reference_tier(["labevents", "d_labitems"], confidence="high") == TIER_UNKNOWN


def test_reference_tier_depends_on_actual_timed_eligible_evidence():
    base = {"source_tables": "labevents,d_labitems", "label_state": "positive", "evidence_ids_json": '["measurement-1"]',
            "time_delta_hours": -1, "time_precision": "timestamp", "eligibility": "eligible"}
    rows = [base, {**base, "evidence_ids_json": "[]"}, {**base, "time_delta_hours": np.nan},
            {**base, "eligibility": "ineligible"}, {**base, "source_tables": "machine_measurements"}]
    annotated = annotate_reference_rows(pd.DataFrame(rows))
    assert annotated.reference_kind.tolist() == ["timed_quantitative", "unverified_or_unknown", "unverified_or_unknown", "unverified_or_unknown", "same_ecg_device"]
    pd.testing.assert_frame_equal(annotated, annotate_reference_rows(pd.DataFrame(rows)))
    assert annotated.independent_reference_modality_available.tolist() == [True, False, True, True, False]
    assert not annotated.reference_timing_claim_allowed.any()
    confirmed = annotate_reference_rows(pd.DataFrame([{**base, "ehr_time_alignment_confirmatory_eligible": True}]))
    assert confirmed.reference_clock_status.iloc[0] == "confirmed"
    assert confirmed.reference_timing_claim_allowed.iloc[0]


def test_review_strata_are_disjoint_and_not_exposed_to_reviewers():
    frame = pd.DataFrame({"subject_id": range(20), "study_id": range(20),
        "label_state": ["positive"]*10 + ["negative"]*10,
        "agreement": ["discordant"]*5 + ["concordant"]*15})
    selection = draw_adjudication_sample(frame, endpoint_id="e", concordance_column="agreement",
        policy=AdjudicationSamplePolicy(positives=3, negatives=3, discordant=3, uncertain=0))
    assert not selection.duplicated(["review_case_id", "reviewer_slot"]).any()
    assert selection.selection_probability.between(0, 1).all()
    assert not selection.blinded.any()
    packet = blinded_reviewer_packet(selection)
    assert packet.blinded.all()
    assert not {"review_stratum", "selection_probability", "label_state", "subject_id", "study_id"} & set(packet)


def test_external_missing_references_remain_unknown_and_measurement_yield_is_not_filtered():
    endpoint = next(iter(EXTERNAL_ENDPOINT_MAP))
    label = EXTERNAL_ENDPOINT_MAP[endpoint][0]
    n = 100
    frame = pd.DataFrame({"source_id": "site", "record_id": range(n), "patient_id": np.arange(n)//2,
        "labels": [label]*50+[""]*50, "marker": np.arange(n, dtype=float),
        "extraction_status": ["ok"]*90+["failed"]*10})
    result = external_agreement(frame, marker_id="marker", endpoint_id=endpoint, direction=1).iloc[0]
    assert result.n_reference_unknown == 50
    assert result.measurement_yield == .9
    assert result.status == "not_evaluable"
    frame["annotation_exhaustive"] = True
    result = external_agreement(frame, marker_id="marker", endpoint_id=endpoint, direction=1)
    assert result.iloc[0].test_yield == .9
    assert result.iloc[0].n_unique_patients == 45
    assert result.p_value.isna().all()  # repeated patients and pooled cohorts


def test_net_benefit_cannot_be_borrowed_from_another_marker():
    row = {"endpoint_id": "e", "vcg_feature_id": "other", "model_id": "classical_plus_vcg", "evaluation_split": "validation",
           "clinical_utility_evidence_eligible": True, "net_benefit": .2, "treat_all_net_benefit": .1, "treat_none_net_benefit": 0}
    assert not _net_benefit_passed(pd.DataFrame([row]), "e", "wanted")
    assert _net_benefit_passed(pd.DataFrame([row]), "e", "other")
    assert not _net_benefit_passed(pd.DataFrame([{**row, "clinical_utility_evidence_eligible": np.nan}]), "e", "other")


def test_workspace_path_and_integrity_are_fail_closed(tmp_path):
    for path in ("../production", str(tmp_path.parent.resolve())):
        with pytest.raises(ValueError): confined(tmp_path, path)
    original = tmp_path/"original"; original.mkdir()
    artifact = original/"test.parquet"; pd.DataFrame({"x": [1]}).to_parquet(artifact)
    manifest = {"version": WORKSPACE_VERSION, "production_resume_allowed": False, "locked_test_read": False,
                "run_id": "test", "pilot_sizes": [1000, 10000],
                "files": [{"path": "original/test.parquet", "sha256": file_sha256(artifact)}]}
    (tmp_path/"workspace.json").write_text(json.dumps(manifest))
    assert PilotViewModel(tmp_path).read_frame("original/test.parquet").x.iloc[0] == 1
    with pytest.raises(ValueError): PilotViewModel(tmp_path).read_frame("../production.parquet")
    pd.DataFrame({"x": [2]}).to_parquet(artifact)
    with pytest.raises(ValueError, match="integrity"): PilotViewModel(tmp_path).read_frame("original/test.parquet")
    assert not (tmp_path/"jobs").exists()


def test_scientific_release_needs_bound_evidence_for_every_gate(tmp_path):
    from vcg_biomarker.mimic.scientific_release import scientific_release_valid, SCIENTIFIC_RELEASE_VERSION, REQUIRED_RELEASE_GATES
    assert not scientific_release_valid(tmp_path)
    evidence = tmp_path/"assessment.json"; evidence.write_text('{"passed":true}')
    payload = {"version": SCIENTIFIC_RELEASE_VERSION, "run_id": tmp_path.name,
        "run_fingerprint": "source-v1", "status": "approved_for_development_full_run",
        "clinical_approval": False, "locked_test_read": False,
        "gates": {name: True for name in REQUIRED_RELEASE_GATES},
        "gate_evidence": {name: {"relative_path": evidence.name, "sha256": file_sha256(evidence)} for name in REQUIRED_RELEASE_GATES}}
    target = tmp_path/"scientific_release.json"; target.write_text(json.dumps(payload))
    # A hash-bound file with a generic flag is not a passing scientific gate.
    assert not scientific_release_valid(tmp_path, "source-v1")
    from vcg_biomarker.mimic.scientific_release import REQUIRED_GATE_CHECKS
    for name in REQUIRED_RELEASE_GATES:
        checks = dict.fromkeys(REQUIRED_GATE_CHECKS[name], True)
        evidence = tmp_path / f"{name}.json"
        evidence.write_text(json.dumps({
            "gate": name, "status": "PASS", "run_fingerprint": "source-v1",
            "checks": checks,
        }), encoding="utf-8")
        payload["gate_evidence"][name] = {
            "relative_path": evidence.name, "sha256": file_sha256(evidence),
        }
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert scientific_release_valid(tmp_path, "source-v1")
    assert not scientific_release_valid(tmp_path, "source-v2")
    evidence.write_text('{"passed":false}')
    assert not scientific_release_valid(tmp_path, "source-v1")


def test_gate_content_requires_actual_production_execution():
    from vcg_biomarker.mimic.scientific_release import gate_content_passed

    evidence = {
        "gate": "production_path", "status": "PASS", "run_fingerprint": "source",
        "checks": {"artifacts_exist": True},
    }
    assert not gate_content_passed(evidence, "production_path", "source")
    assert not gate_content_passed(evidence, "software_suite", "source")
    assert not gate_content_passed(evidence, "production_path", "different_source")


def test_each_release_gate_requires_its_own_complete_checklist():
    from vcg_biomarker.mimic.scientific_release import REQUIRED_GATE_CHECKS, gate_content_passed
    for gate, required in REQUIRED_GATE_CHECKS.items():
        payload = {"gate": gate, "status": "PASS", "run_fingerprint": "source",
                   "checks": dict.fromkeys(required, True)}
        assert gate_content_passed(payload, gate, "source")
        for name in required:
            incomplete = {**payload, "checks": {k: v for k, v in payload["checks"].items() if k != name}}
            assert not gate_content_passed(incomplete, gate, "source")
        assert not gate_content_passed({**payload, "checks": {"file_exists": True}}, gate, "source")
    assert not gate_content_passed([], "software_suite", "source")


def test_only_one_isolated_lab_writer_at_a_time(tmp_path):
    from vcg_biomarker.mimic.pilot_lab_worker import single_lab_lock
    with single_lab_lock(tmp_path):
        with pytest.raises(OSError):
            with single_lab_lock(tmp_path):
                pytest.fail("Second writer acquired the same workspace lock")
    with single_lab_lock(tmp_path):
        pass


def test_survival_all_abstentions_have_a_complete_output_contract():
    from vcg_biomarker.mimic.population_cohorts import _survival_associations
    n = 100
    frame = pd.DataFrame({"cohort": "sami_trop", "patient_id": range(n),
        "death": [True]*5+[False]*(n-5), "follow_up_years": 2., "extraction_status": "ok",
        "age": np.arange(n)+20., "sex": ["M", "F"]*(n//2), "marker": np.linspace(0, 1, n)})
    result = _survival_associations(frame, ["marker"], incremental_marker_ids=["marker"])
    assert result.status.eq("non_estimable").all()
    assert result.q_value_cohort.isna().all()
    assert "insufficient_events" in result.iloc[0].abstention_reason


def test_external_ecg_age_joint_comparison_uses_same_complete_patients():
    from vcg_biomarker.mimic.population_cohorts import _survival_associations
    rng = np.random.default_rng(873)
    n = 800
    marker = rng.normal(size=n)
    age = rng.uniform(25, 80, n)
    event_time = rng.exponential(2*np.exp(-.45*marker), size=n)
    censor = rng.uniform(1, 5, n)
    gap = rng.normal(size=n); gap[:100] = np.nan
    frame = pd.DataFrame({"cohort": "sami_trop", "patient_id": range(n), "death": event_time < censor,
        "follow_up_years": np.minimum(event_time, censor), "extraction_status": "ok", "age": age,
        "sex": rng.choice(["M", "F"], n), "reported_normal_ecg": rng.choice([True, False], n),
        "ecg_age_gap_years": gap, "marker": marker})
    row = _survival_associations(frame, ["marker"], incremental_marker_ids=["marker"]).set_index("marker_id").loc["marker"]
    assert row.ecg_age_comparison_status == "estimable"
    assert row.ecg_age_comparison_n == 700
    assert row.hr_plus_classical_ecg_age > 1
    assert row.incremental_evaluation_scope.startswith("in_sample")
    assert not row.clinical_claim_allowed


def test_cohort_value_separates_records_patients_events_and_model_improvement():
    from vcg_biomarker.mimic.cohort_value import cohort_value_table
    frame = pd.DataFrame({"cohort": ["code15"]*4, "patient_id": [1, 1, 2, None],
        "extraction_status": ["ok", "ok", "failed", "ok"], "marker": [1., 2., np.nan, 3.],
        "death": [True, None, False, None], "follow_up_years": [1., None, 2., None]})
    row = cohort_value_table(frame, cohort_column="cohort", marker_ids=("marker",)).iloc[0]
    assert row.n_selected == 4
    assert row.n_unique_known_patients == 2
    assert row.n_records_unknown_patient == 1
    assert row.n_outcome_bound_patients == 2
    assert row.n_observed_deaths == 1
    assert row.model_improvement.startswith("not_established")


def test_cohort_value_rejects_invalid_outcomes_and_failed_qc_measurements():
    from vcg_biomarker.mimic.cohort_value import cohort_value_table
    frame = pd.DataFrame({"cohort": ["code15"]*5, "patient_id": range(5),
        "extraction_status": ["failed"]*5, "marker": [1.]*5,
        "death": ["False", True, 2, False, True], "follow_up_years": [1., -1., 2., np.inf, 2.]})
    row = cohort_value_table(frame, cohort_column="cohort", marker_ids=("marker",)).iloc[0]
    assert row.n_markers_with_any_measurement == 0
    assert row.marker_cell_availability == 0
    assert row.n_outcome_bound_patients == 1
    assert row.n_observed_deaths == 1
    assert row.n_identified_records_without_valid_outcome == 4


def test_lab_segment_names_match_backend_and_preserve_gaps():
    from vcg_biomarker.mimic.pilot_viewmodel import laboratory_segment_mask
    arrays = {"segment_names": ["p_wave", "qrs_complex", "t_wave", "qrs_complex"],
        "segment_start_samples": [0, 2, 5, 8], "segment_end_samples": [1, 3, 6, 9]}
    assert np.flatnonzero(laboratory_segment_mask(10, arrays, "QRS")).tolist() == [2, 3, 8, 9]
    assert np.flatnonzero(laboratory_segment_mask(10, arrays, "P")).tolist() == [0, 1]
    assert np.flatnonzero(laboratory_segment_mask(10, arrays, "T")).tolist() == [5, 6]
    assert laboratory_segment_mask(10, arrays, "Gesamtsignal").all()
    assert not laboratory_segment_mask(10, {}, "QRS").any()


def test_lab_segment_contract_rejects_mismatched_arrays_and_clips_bounds():
    from vcg_biomarker.mimic.pilot_viewmodel import laboratory_segment_mask
    arrays = {"segment_names": ["qrs_complex"], "segment_start_samples": [-8], "segment_end_samples": [-2]}
    assert not laboratory_segment_mask(10, arrays, "QRS").any()
    arrays["segment_end_samples"] = []
    with pytest.raises(ValueError, match="Inconsistent"):
        laboratory_segment_mask(10, arrays, "QRS")
