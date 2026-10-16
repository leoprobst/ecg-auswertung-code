import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.mechanism_echo_context import compare_echo_pair_context


def inputs():
    a = pd.DataFrame({"pair_id": [1, 2], "subject_id": [10, 20], "split": ["train", "validation"],
        "ecg_time": ["2020-01-01"]*2, "nearest_echo_measurement_id": [101, 201], "nearest_echo_delta_hours": [0., 0.]})
    b = a.assign(ecg_time="2020-01-02", nearest_echo_measurement_id=[102, 202])
    measurements = pd.DataFrame({"measurement_id": [101, 102, 201, 202], "subject_id": [10, 10, 20, 20],
        "echo_time": ["2020-01-01", "2020-01-02"]*2, "identity_conflict": False, "lavi": [30., 35., 40., 40.]})
    return a, b, measurements


def test_real_zero_delta_and_key_alignment_preserve_inputs():
    a, b, m = inputs()
    original = a.copy(deep=True)
    out = compare_echo_pair_context(a, b.iloc[::-1], m, columns=["lavi"])
    assert out.lavi_delta.tolist() == [5., 0.]
    assert out.status.eq("descriptive").all()
    assert out.echo_interval_hours.eq(24).all()
    assert not out.clinical_claim_allowed.any()
    pd.testing.assert_frame_equal(a, original)


@pytest.mark.parametrize("problem,reason", [
    ("same", "same_echo_measurement"), ("patient", "patient_mismatch_or_missing"),
    ("clock", "echo_identity_or_alignment_unqualified"), ("reverse", "nonforward_ecg_pair"),
    ("identity", "echo_identity_or_alignment_unqualified")])
def test_bad_pairs_abstain_without_selecting_replacement(problem, reason):
    a, b, m = inputs()
    if problem == "same":
        b.loc[0, "nearest_echo_measurement_id"] = 101
        b.loc[0, "nearest_echo_delta_hours"] = -24
    elif problem == "patient":
        b.loc[0, "subject_id"] = 99
    elif problem == "clock":
        b.loc[0, "nearest_echo_delta_hours"] = 1
    elif problem == "reverse":
        b.loc[0, "ecg_time"] = "2019-12-31"
    else:
        m.loc[1, "identity_conflict"] = True
    out = compare_echo_pair_context(a, b, m, columns=["lavi"])
    assert out.reason.iloc[0] == reason
    assert np.isnan(out.lavi_delta.iloc[0])
    assert out.status.iloc[1] == "descriptive"


@pytest.mark.parametrize("problem", ["locked", "duplicate", "different_pairs", "cross_split"])
def test_invalid_cohort_contract_raises(problem):
    a, b, m = inputs()
    if problem == "locked":
        a.loc[0, "split"] = "locked_test"
    elif problem == "duplicate":
        a.loc[1, "pair_id"] = 1
    elif problem == "different_pairs":
        b.loc[0, "pair_id"] = 3
    else:
        b.loc[0, "split"] = "validation"
    with pytest.raises(ValueError):
        compare_echo_pair_context(a, b, m, columns=["lavi"])


def test_missing_value_is_not_a_zero_change():
    a, b, m = inputs()
    m.loc[1, "lavi"] = np.nan
    out = compare_echo_pair_context(a, b, m, columns=["lavi"])
    assert not out.lavi_available.iloc[0]
    assert np.isnan(out.lavi_delta.iloc[0])


def test_forward_ecgs_with_reversed_echoes_are_not_a_response():
    a, b, m = inputs()
    a.loc[0, "nearest_echo_measurement_id"] = 102
    a.loc[0, "nearest_echo_delta_hours"] = 24
    b.loc[0, "nearest_echo_measurement_id"] = 101
    b.loc[0, "nearest_echo_delta_hours"] = -24
    out = compare_echo_pair_context(a, b, m, columns=["lavi"])
    assert out.reason.iloc[0] == "nonforward_echo_pair"
    assert np.isnan(out.lavi_delta.iloc[0])


def test_helper_is_declared_without_claiming_production_analysis():
    from vcg_biomarker.mimic.marker_freeze import marker_freeze_payload
    chapters = marker_freeze_payload()["research_coverage"]["chapters"]
    selected = [c for c in chapters if c["available_context_helpers"]]
    assert len(selected) == 2
    assert all(not c["analysis_complete"] for c in selected)
