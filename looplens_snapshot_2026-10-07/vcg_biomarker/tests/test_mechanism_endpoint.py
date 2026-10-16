import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.mechanism_endpoint import (
    analyze_mechanism_endpoint, adjust_mechanism_family,
)


def hypothesis():
    return {"marker_id": "marker", "endpoint_id": "endpoint", "hypothesis_version": "test",
            "reference_target": "echo", "required_comparisons": ["rhythm"],
            "required_measurement_context": ["QRS_quality"]}


def universe():
    rng = np.random.default_rng(42)
    n = 1000
    marker, age, rate, quality, duration = rng.normal(size=(5, n))
    labels = rng.random(n) < 1 / (1 + np.exp(-(.4*marker + .3*duration)))
    return pd.DataFrame({"subject_id": np.arange(n),
        "split": np.repeat(["train", "validation"], n//2),
        "eligibility_state": "eligible", "label_id": "endpoint", "label_state": np.where(labels, "positive", "negative"),
        "extraction_status": "ok", "marker": marker, "age": 60 + age*10,
        "sex": rng.choice(["M", "F"], n), "context_heart_rate_bpm": 70 + rate*10,
        "signal_quality_score": .8 + quality*.03, "qrs_duration_ms": 100 + duration*10,
        "ed_stay_id": np.nan, "hadm_id": np.nan})


def test_actual_fits_common_cohort_and_no_promotion():
    frame = universe()
    frame.loc[:49, "qrs_duration_ms"] = np.nan
    before = frame.copy(deep=True)
    rows = analyze_mechanism_endpoint(frame, hypotheses=[hypothesis()], preview=False)
    pd.testing.assert_frame_equal(frame, before)
    assert len(rows) == 4
    assert all(row["status"] == "estimable" for row in rows)
    assert [row["n"] for row in rows] == [450, 450, 500, 500]
    assert all(not row["clinical_claim_allowed"] and not row["mechanism_requirements_fulfilled"]
               for row in rows)
    assert all("rhythm" in row["remaining_requirements_json"] for row in rows)


def test_unknown_reference_and_failed_extraction_are_not_controls():
    frame = universe()
    frame.loc[:19, "label_state"] = "unknown"
    frame.loc[20:29, "extraction_status"] = "failed"
    rows = analyze_mechanism_endpoint(frame, hypotheses=[hypothesis()], preview=False)
    assert rows[0]["n_index_universe"] == 500
    assert rows[0]["n_reference_and_extraction_eligible"] == 470
    assert rows[0]["n"] == rows[1]["n"] == 470


@pytest.mark.parametrize("problem", ["locked", "duplicate", "cross_split_duplicate"])
def test_index_contract_rejected(problem):
    frame = universe()
    if problem == "locked":
        frame.loc[0, "split"] = "locked_test"
    else:
        frame.loc[500 if problem == "cross_split_duplicate" else 1, "subject_id"] = 0
    with pytest.raises(ValueError):
        analyze_mechanism_endpoint(frame, hypotheses=[hypothesis()], preview=False)


@pytest.mark.parametrize("missing", ["marker", "qrs_duration_ms"])
def test_missing_measurement_yields_rows_not_fallback(missing):
    rows = analyze_mechanism_endpoint(universe().drop(columns=missing),
                                      hypotheses=[hypothesis()], preview=False)
    assert len(rows) == 4
    assert all(row["status"] == "non_estimable" for row in rows)


def test_pilot_does_not_fit(monkeypatch):
    from vcg_biomarker.mimic import mechanism_endpoint
    def forbidden(*args, **kwargs):
        raise AssertionError("pilot must not fit")
    monkeypatch.setattr(mechanism_endpoint, "compare_mechanism_adjustment", forbidden)
    rows = analyze_mechanism_endpoint(universe(), hypotheses=[hypothesis()], preview=True)
    assert len(rows) == 4 and all(row["reason"] == "pilot_no_confirmation" for row in rows)


def test_fdr_retains_uncomputed_tests_and_abstention():
    frame = pd.DataFrame([
        {"feature_id": "m", "endpoint_id": "e", "split": "train", "model_id": "base",
         "status": "estimable", "p_value": .01},
        {"feature_id": "m", "endpoint_id": "e", "split": "train", "model_id": "extra",
         "status": "non_estimable", "p_value": np.nan}])
    result = adjust_mechanism_family(frame, planned_hypotheses=1)
    assert result.q_value.iloc[0] == pytest.approx(.04)
    assert np.isnan(result.q_value.iloc[1])
    with pytest.raises(ValueError):
        adjust_mechanism_family(pd.concat([frame, frame]), planned_hypotheses=1)


def test_empty_result_is_exportable():
    result = adjust_mechanism_family(pd.DataFrame(), planned_hypotheses=40)
    assert result.empty and "q_value" in result


@pytest.mark.parametrize("problem", ["wrong", "mixed", "missing", "duplicate_hypothesis", "mixed_hypotheses"])
def test_reference_endpoint_identity_is_enforced(problem):
    frame = universe()
    hypotheses = [hypothesis()]
    if problem == "wrong":
        frame["label_id"] = "another_endpoint"
    elif problem == "mixed":
        frame.loc[0, "label_id"] = "another_endpoint"
    elif problem == "missing":
        frame.loc[0, "label_id"] = None
    elif problem == "duplicate_hypothesis":
        hypotheses.append(hypothesis())
    else:
        hypotheses.append({**hypothesis(), "endpoint_id": "another_endpoint"})
    with pytest.raises(ValueError):
        analyze_mechanism_endpoint(frame, hypotheses=hypotheses, preview=False)


def test_real_checkpoint_roundtrip_and_parquet_export(tmp_path):
    from vcg_biomarker.mimic.validation import (
        _write_endpoint_validation_checkpoint, _load_endpoint_validation_checkpoint,
        _extend_validation_result_groups,
    )
    rows = analyze_mechanism_endpoint(universe(), hypotheses=[hypothesis()], preview=False)
    path = tmp_path / "endpoint.json"
    _write_endpoint_validation_checkpoint(path, input_hash="synthetic", endpoint_id="endpoint",
        result_groups={"mechanism_records": rows}, group_offsets={"mechanism_records": 0})
    payload = _load_endpoint_validation_checkpoint(path, input_hash="synthetic", endpoint_id="endpoint")
    restored = {"mechanism_records": []}
    _extend_validation_result_groups(restored, payload["result_groups"])
    result = adjust_mechanism_family(pd.DataFrame(restored["mechanism_records"]), planned_hypotheses=1)
    assert len(result) == 4
    assert result.status.eq("estimable").all()
    assert np.allclose(result.estimate, [row["estimate"] for row in rows])
    result.to_parquet(tmp_path / "result.parquet", index=False)
    reloaded = pd.read_parquet(tmp_path / "result.parquet")
    assert np.allclose(reloaded.q_value, result.q_value)
    assert not reloaded.mechanism_requirements_fulfilled.any()
