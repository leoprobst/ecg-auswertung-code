import numpy as np
import pandas as pd

from vcg_biomarker.mimic.lead_endpoint_analysis import prepare_lead_frame, analyze_endpoint_leads


def test_projection_matches_full_copy_and_does_not_mutate_input(monkeypatch):
    from vcg_biomarker.mimic import lead_endpoint_analysis as module
    from vcg_biomarker.tests.test_lead_comparison import sample

    frame = sample()
    frame["eligibility_state"] = "eligible"
    frame["reference_kind"] = "simulation"
    frame["sex"] = "M"
    frame["context_heart_rate_bpm"] = 70.
    frame["signal_quality_score"] = .9
    frame["ed_stay_id"] = np.nan
    frame["hadm_id"] = np.nan
    frame.index = np.arange(len(frame)) * 3 + 7
    frame.loc[frame.index[4], "age"] = np.nan
    before = frame.copy(deep=True)
    projected = prepare_lead_frame(frame, lead_only=True)
    actual_rows, actual_models = analyze_endpoint_leads(
        frame, endpoint_id="test", iterations=0, seed=1, preview=True)
    monkeypatch.setattr(module, "_project_lead_inputs", lambda value: value.copy())
    legacy = prepare_lead_frame(frame)
    expected_rows, expected_models = analyze_endpoint_leads(
        frame, endpoint_id="test", iterations=0, seed=1, preview=True)
    pd.testing.assert_frame_equal(projected, legacy.loc[:, projected.columns])
    pd.testing.assert_frame_equal(pd.DataFrame(actual_rows), pd.DataFrame(expected_rows))
    assert actual_models == expected_models
    pd.testing.assert_frame_equal(frame, before)
    assert "sex_encoded" not in projected


def test_projection_does_not_copy_unrelated_wide_marker_block():
    from vcg_biomarker.mimic.lead_endpoint_analysis import _project_lead_inputs
    base = pd.DataFrame({"subject_id": np.arange(100), "eligibility_state": "eligible"})
    wide = pd.DataFrame(np.zeros((100, 1000)), columns=[f"other_marker_{i}" for i in range(1000)])
    frame = pd.concat([base, wide], axis=1)
    result = _project_lead_inputs(frame)
    pd.testing.assert_frame_equal(result, base)
    assert result.memory_usage(deep=True).sum() < frame.memory_usage(deep=True).sum() / 50


def test_adapter_does_not_turn_unknown_sex_or_setting_into_controls():
    frame = pd.DataFrame({"subject_id": [1, 2], "eligibility_state": ["eligible", "unknown"],
                          "sex": ["M", "unknown"], "age": [40, 50]})
    result = prepare_lead_frame(frame)
    assert result.reference_eligible.tolist() == [True, False]
    assert result.lead_adjust_sex_male.iloc[0] == 1
    assert np.isnan(result.lead_adjust_sex_male.iloc[1])
    assert result.lead_adjust_ed.isna().all()
    assert result.lead_adjust_heart_rate.isna().all()


def test_historical_features_without_lead_values_produce_explicit_abstentions():
    frame = pd.DataFrame({"subject_id": [1, 2], "eligibility_state": ["eligible"]*2,
        "split": ["train", "validation"], "label_state": ["positive", "negative"],
        "sex": ["M", "F"], "age": [40, 50], "extraction_status": ["ok"]*2})
    rows, models = analyze_endpoint_leads(frame, endpoint_id="test", iterations=0, seed=1, preview=True)
    assert len(rows) == 48
    assert all(row["status"] == "non_estimable" for row in rows)
    assert all(row["n_lead_measured"] == 0 for row in rows)
    assert all(not row["calibrated_probability_claim_allowed"] for row in rows)
    assert len(models) == 4
