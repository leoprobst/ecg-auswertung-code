import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.external_identity_audit import waveform_fingerprint, duplicate_contract


def test_fingerprint_ignores_storage_not_physiological_samples():
    signal = np.arange(120, dtype=float).reshape(10, 12)
    reference = waveform_fingerprint(signal, 500)
    assert reference == waveform_fingerprint(np.asfortranarray(signal.astype(">f8")), 500)
    signal[0, 0] = -0.0
    assert reference == waveform_fingerprint(signal, 500)
    assert reference != waveform_fingerprint(signal, 250)
    signal[1, 0] += .001
    assert reference != waveform_fingerprint(signal, 500)


@pytest.mark.parametrize("signal", [np.ones((10, 8)), np.zeros((0, 12)), np.full((10, 12), np.nan)])
def test_invalid_signals_do_not_prove_identity(signal):
    with pytest.raises(ValueError): waveform_fingerprint(signal, 500)


def test_copies_are_not_patients_and_overlap_is_not_independence():
    frame = pd.DataFrame({"source_id": ["a", "b", "c"], "record_id": ["2", "1", "1"],
        "source_role": ["independent_external", "development_reused_not_independent", "independent_external"],
        "fingerprint_status": ["measured", "measured", "failed"], "waveform_sha256": ["same", "same", None]})
    result = duplicate_contract(frame)
    assert result.exact_copy_group_size.tolist() == [2, 2, 0]
    assert result.copy_overlaps_development_source.tolist() == [True, True, False]
    assert result.exact_copy_representative.tolist() == [True, False, False]
    assert not result.patient_identity_established_by_fingerprint.any()
    assert not result.independence_proven_by_no_exact_copy.any()
    shuffled = duplicate_contract(frame.iloc[::-1]).sort_index()
    pd.testing.assert_frame_equal(result, shuffled)


def test_agreement_deduplicates_before_quality_or_label_selection():
    from vcg_biomarker.mimic.external_validation import external_agreement
    frame = pd.DataFrame({"source_id": ["site"]*3, "record_id": ["a", "b", "c"],
        "source_role": ["independent_external"]*3,
        "fingerprint_status": ["measured"]*3, "waveform_sha256": ["same", "same", "other"],
        "labels": ["", "left_ventricular_hypertrophy", ""],
        "extraction_status": ["failed", "ok", "ok"], "marker": [np.nan, 4., 5.]})
    features = duplicate_contract(frame)
    rows = external_agreement(features, marker_id="marker", endpoint_id="echo_lvh_strict", direction=1)
    row = rows.loc[rows.source_id.eq("site")].iloc[0]
    assert row.n_selected_before_copy_filter == 3
    assert row.n_selected == 2
    assert row.n_exact_copy_excluded == 1
    assert row.n_reference_unknown == 2  # Do not prefer the labelled successful copy.
    assert row.measurement_yield == .5


def test_all_copies_of_development_source_are_excluded_without_empty_mean_warning():
    from vcg_biomarker.mimic.external_validation import external_agreement
    frame = pd.DataFrame({"source_id": ["a", "b"], "record_id": ["1", "1"],
        "source_role": ["independent_external", "development_reused_not_independent"],
        "fingerprint_status": ["measured"]*2, "waveform_sha256": ["same"]*2,
        "labels": [""]*2, "extraction_status": ["ok"]*2, "marker": [1., 1.]})
    rows = external_agreement(duplicate_contract(frame), marker_id="marker", endpoint_id="echo_lvh_strict", direction=1)
    assert rows.n_selected.eq(0).all()
    assert rows.measurement_yield.isna().all()


def test_patient_placeholders_cannot_support_patient_level_inference():
    from vcg_biomarker.mimic.patient_identity import known_patient_mask
    values = pd.Series([None, np.nan, "", " ", "unknown", "NaN", 0, -1, True, "abc", 12])
    assert known_patient_mask(values).tolist() == [False]*6 + [True, True, False, True, True]
