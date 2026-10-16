import pandas as pd

from vcg_biomarker.mimic.external_cohorts import eligible_for_diagnostic_validation, external_claim_scope
from vcg_biomarker.mimic.external_validation import ExternalExtractionOptions, extract_external_features, external_agreement


def inventory():
    return pd.DataFrame({"source_id": ["georgia"]*2, "source_country": ["USA"]*2,
        "record_id": ["a", "b"], "header_path": ["a.hea", "b.hea"], "sampling_rate_hz": [500]*2,
        "independent_external": [True]*2, "n_leads": [12]*2,
        "duration_seconds": [10.]*2, "n_labels": [1, 0]})


def test_extraction_summary_separates_success_failure_and_copies():
    from vcg_biomarker.mimic.external_validation import extraction_summary
    frame = pd.DataFrame({
        "extraction_status": ["ok", "excluded_exact_copy", "failed"],
        "fingerprint_status": ["measured", "measured", "failed"],
        "exact_copy_representative": [True, False, False],
        "copy_overlaps_development_source": [False, False, False],
    })
    result = extraction_summary(frame)
    assert result["n_records_processed"] == 3
    assert result["n_records_extracted"] == 1
    assert result["n_records_exact_copy_excluded"] == 1
    assert result["n_records_measurement_denominator"] == 2
    assert result["n_records_without_successful_extraction"] == 1
    assert result["extraction_success_rate"] == .5


def test_extraction_summary_empty_is_unavailable_not_zero_success():
    from vcg_biomarker.mimic.external_validation import extraction_summary
    result = extraction_summary(pd.DataFrame())
    assert result["n_records_extracted"] == 0
    assert result["extraction_success_rate"] is None
    assert result["extraction_success_rate_status"] == "no_measurement_denominator"


def test_unlabelled_signals_remain_analytically_eligible():
    scoped = eligible_for_diagnostic_validation(inventory())
    assert scoped.eligible_for_analytical_validation.tolist() == [True, True]
    assert scoped.eligible_for_diagnostic_validation.tolist() == [True, False]
    assert not scoped.patient_independence_established_by_source_role.any()
    scope = external_claim_scope(scoped)
    assert scope["n_records_analytical_role_eligible_before_copy_checks"] == 2
    assert scope["n_records_annotation_role_eligible_before_copy_checks"] == 1
    assert scope["n_verified_independent_patients"] is None


def test_empty_collection_does_not_claim_validation_capabilities():
    scope = external_claim_scope(pd.DataFrame())
    assert not scope["supports_external_analytical_validation"]
    assert not scope["supports_external_agreement_with_expert_reading"]


def test_extraction_attempts_unlabelled_signal_without_assigning_negative_label(monkeypatch):
    seen = []
    def record_attempt(path):
        seen.append(path.stem)
        raise ValueError("Synthetic stop after confirming extraction selection")
    monkeypatch.setattr("vcg_biomarker.core.external_loader.load_external_ecg_result", record_attempt)
    result = extract_external_features(eligible_for_diagnostic_validation(inventory()),
        ExternalExtractionOptions(marker_ids=("heart_rate",), workers=1))
    assert seen == ["a", "b"]
    assert result.eligible_for_diagnostic_validation.tolist() == [True, False]
    assert result.extraction_status.eq("failed").all()


def test_external_serial_and_bounded_parallel_extraction_contracts_match(monkeypatch):
    def synthetic_failure(path):
        raise ValueError("synthetic missing signal")
    monkeypatch.setattr("vcg_biomarker.core.external_loader.load_external_ecg_result", synthetic_failure)
    scoped = eligible_for_diagnostic_validation(inventory())
    serial = extract_external_features(scoped, ExternalExtractionOptions(marker_ids=("heart_rate",), workers=1))
    parallel = extract_external_features(scoped, ExternalExtractionOptions(marker_ids=("heart_rate",), workers=2))
    pd.testing.assert_frame_equal(serial, parallel)


def test_analytical_only_rows_do_not_enter_annotation_accuracy():
    frame = pd.DataFrame({"source_id": ["georgia"]*2, "labels": ["left_ventricular_hypertrophy"]*2,
        "extraction_status": ["ok"]*2, "marker": [1., 2.], "eligible_for_diagnostic_validation": [True, False]})
    result = external_agreement(frame, marker_id="marker", endpoint_id="echo_lvh_strict", direction=1)
    assert result.n_selected.eq(1).all()
    assert result.n_analytical_only_excluded_from_annotation.eq(1).all()


def test_coverage_does_not_treat_copy_exclusions_as_measurement_failures():
    from vcg_biomarker.mimic.external_validation import extraction_coverage
    from vcg_biomarker.mimic.external_identity_audit import duplicate_contract
    frame = pd.DataFrame({"source_id": ["a"]*2, "record_id": ["1", "2"],
        "source_role": ["independent_external"]*2, "fingerprint_status": ["measured"]*2,
        "waveform_sha256": ["same"]*2, "extraction_status": ["ok", "excluded_exact_copy"], "marker": [1., float("nan")]})
    row = extraction_coverage(duplicate_contract(frame), ["marker"]).iloc[0]
    assert row.n_selected_before_copy_filter == 2 and row.n_records == 1
    assert row.n_exact_copy_excluded == 1 and row.coverage == 1


def test_coverage_uses_finite_measurements_and_preserves_marker_generators():
    from vcg_biomarker.mimic.external_validation import extraction_coverage
    frame = pd.DataFrame({"source_id": ["a", "b"], "extraction_status": ["ok"]*2, "marker": [float("inf"), 1.]})
    result = extraction_coverage(frame, (name for name in ["marker"]))
    assert result.coverage.tolist() == [0., 1.]


def test_identical_marginal_distributions_do_not_validate_threshold_transfer():
    import numpy as np
    from vcg_biomarker.mimic.external_validation import distribution_shift
    values = np.linspace(0, 1, 100)
    frame = pd.DataFrame({"source_id": ["a"]*100 + ["b"]*100,
        "extraction_status": ["ok"]*200, "marker": np.tile(values, 2),
        "label": np.r_[values > .5, values <= .5]})
    result = distribution_shift(frame, {"marker": (values.mean(), values.std(ddof=1))}, ["marker"])
    assert result.distribution_shift_within_screening_tolerance.all()
    assert not result.operating_point_transferable.any()
    assert not result.clinical_claim_allowed.any()
