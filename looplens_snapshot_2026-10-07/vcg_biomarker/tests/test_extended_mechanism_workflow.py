from types import SimpleNamespace
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.extended_mechanism_workflow import (
    MARKERS, analyze_extended_records, run_extended_mechanisms,
)


def data():
    rng = np.random.default_rng(55)
    n = 400
    patient = np.repeat(np.arange(n), 2)
    times = pd.to_datetime(np.tile(["2020-01-01", "2020-01-03"], n))
    records = pd.DataFrame({"subject_id": patient, "study_id": np.arange(n*2),
        "split": np.where(patient < 300, "train", "validation"), "ecg_time": times,
        "nearest_echo_measurement_id": np.arange(n*2), "nearest_echo_delta_hours": 0.,
        "age": np.repeat(rng.uniform(30, 80, n), 2), "sex": np.repeat(rng.choice(["M", "F"], n), 2),
        "extraction_status": "ok", "signal_quality_score": rng.uniform(.7, 1, n*2),
        "context_heart_rate_bpm": rng.uniform(50, 110, n*2), "hadm_id": patient,
        "ed_stay_id": np.where(patient % 3 == 0, patient, np.nan)})
    for name in MARKERS:
        records[name] = rng.normal(size=n*2)
    measurements = pd.DataFrame({"measurement_id": records.study_id, "subject_id": patient,
        "echo_time": times, "identity_conflict": False,
        "la_volume_index_direct": 30 + records[MARKERS[0]] + rng.normal(size=n*2),
        "tapse_cm": 2 + .1*records[MARKERS[2]] + rng.normal(size=n*2)*.1})
    return records, measurements


def test_connected_numeric_pipeline_and_fixed_family():
    records, measurements = data()
    pairs, models, residuals, fit = analyze_extended_records(records, measurements)
    assert len(pairs) == 400 and pairs.status.eq("descriptive").all()
    assert len(models) == 12 and models.status.eq("estimable").all()
    assert fit["status"] == "descriptive" and len(residuals) == 400
    for encoded in models.details_json:
        item = json.loads(encoded)
        assert item["fdr_planned_tests"] == 24
        assert all("q_value" in term for term in item["terms"])


def test_index_is_not_reselected_for_better_reference():
    records, measurements = data()
    records.loc[0, "nearest_echo_measurement_id"] = 99999
    pairs, _, residuals, _ = analyze_extended_records(records, measurements)
    assert pairs.status.iloc[0] == "non_estimable"
    row = residuals.loc[residuals.subject_id.eq(0)].iloc[0]
    assert row.status == "non_estimable"


def test_actual_duckdb_and_parquet_adapter(tmp_path):
    records, measurements = data()
    config = SimpleNamespace(evidence_root=tmp_path / "evidence", master_root=tmp_path / "master")
    config.evidence_root.mkdir()
    config.master_root.mkdir()
    feature_names = ["subject_id", "study_id", "split", "extraction_status", "signal_quality_score", "context_heart_rate_bpm", *MARKERS]
    records.drop(columns=["extraction_status", "signal_quality_score", "context_heart_rate_bpm", *MARKERS]).to_parquet(
        config.master_root / "ecg_master_index.parquet")
    measurements.drop(columns="identity_conflict").to_parquet(config.evidence_root / "echo_measurements_wide.parquet")
    measurements[["measurement_id", "subject_id", "identity_conflict"]].to_parquet(config.evidence_root / "echo_study_index.parquet")
    features = records[feature_names]
    with duckdb.connect() as con:
        con.register("features", features)
        outputs, summary = run_extended_mechanisms(con, config, "features", set(features))
    assert len(outputs["echo_pairs"]) == 400
    assert summary["electromechanical_model"]["status"] == "descriptive"
    assert summary["pr_geometry"]["reason"] == "qualified_beat_pairs_and_error_calibration_missing"
    for name, frame in outputs.items():
        frame.to_parquet(tmp_path / (name + ".parquet"))


def test_real_qualified_PR_adapter_and_locked_rejection(tmp_path):
    config = SimpleNamespace(evidence_root=tmp_path)
    x = np.linspace(100., 200., 30)
    pr = pd.DataFrame({"subject_id": 1, "study_id": 2, "split": "train",
        "beat_time_s": np.arange(30), "pr_ms": x, "geometry": x*2,
        "pr_error_variance_ms2": 0., "error_covariance": 0.,
        "qualification_verified": True, "calibration_reference": "synthetic_exact_control",
        "geometry_marker_id": "qrs_initial_terminal_energy_balance", "transform": "synthetic",
        "measurement_version": "test"})
    path = tmp_path / "pr_geometry_qualified.parquet"
    pr.to_parquet(path)
    from vcg_biomarker.tests.test_pr_calibration_contract import certificate_for
    certificate_for(tmp_path, path)
    out, _ = run_extended_mechanisms(None, config, "unused", set())
    result = json.loads(out["pr_geometry_models"].details_json.iloc[0])
    assert result["slope_per_ms"] == pytest.approx([2.])
    pr.loc[0, "split"] = "locked_test"
    pr.to_parquet(path)
    certificate_for(tmp_path, path)
    with pytest.raises(ValueError, match="isolation"):
        run_extended_mechanisms(None, config, "unused", set())


def test_preview_does_not_read_inputs_or_fit(tmp_path):
    config = SimpleNamespace(evidence_root=tmp_path)
    out, summary = run_extended_mechanisms(None, config, "invalid", set(), preview=True)
    assert all(frame.empty for frame in out.values())
    assert summary["reason"] == "pilot_no_confirmation"


def test_large_integer_identities_are_not_rounded_during_pair_shift():
    records, measurements = data()
    offset = 2**55
    records["study_id"] += offset
    records["nearest_echo_measurement_id"] += offset
    measurements["measurement_id"] += offset
    pairs, _, _, _ = analyze_extended_records(records, measurements)
    assert pairs.second_study_id.iloc[0] == offset + 1
    assert pairs.status.eq("descriptive").all()
