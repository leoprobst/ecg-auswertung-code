import pandas as pd
import pytest

from vcg_biomarker.mimic.pr_calibration_contract import load_approved_calibration
from vcg_biomarker.mimic.storage import atomic_write_json, file_sha256


def certificate_for(root, beat_path, **overrides):
    from vcg_biomarker.mimic.feature_extraction import feature_implementation_hash
    report = root / "pr_calibration_review_report.json"
    atomic_write_json(report, {"scope": "synthetic_test_only_not_patient_approval"})
    certificate = {"approved_for_correction": True, "reference_error_identified": True,
        "target_transportability_verified": True, "development_overlap_excluded": True,
        "beat_artifact_sha256": file_sha256(beat_path), "feature_implementation_hash": feature_implementation_hash(),
        "measurement_version": "test", "transform": "synthetic", "review_report_sha256": file_sha256(report),
        "pr_error_variance_ms2": 0., "error_covariance": 0., **overrides}
    path = root / "pr_geometry_calibration.json"
    atomic_write_json(path, certificate)
    return path


@pytest.mark.parametrize("gate", ["approved_for_correction", "reference_error_identified",
                                  "target_transportability_verified", "development_overlap_excluded"])
def test_reference_discrepancy_is_not_automatic_error_calibration(tmp_path, gate):
    beat = tmp_path / "beats.parquet"
    pd.DataFrame({"pr": [1]}).to_parquet(beat)
    path = certificate_for(tmp_path, beat, **{gate: False})
    certificate, reason = load_approved_calibration(path, beat)
    assert certificate is None and reason == "independent_calibration_gates_unmet"


def test_certificate_binds_actual_data_and_code(tmp_path):
    beat = tmp_path / "beats.parquet"
    pd.DataFrame({"pr": [1]}).to_parquet(beat)
    path = certificate_for(tmp_path, beat)
    certificate, reason = load_approved_calibration(path, beat)
    assert certificate and not reason
    pd.DataFrame({"pr": [2]}).to_parquet(beat)
    assert load_approved_calibration(path, beat)[1] == "qualified_beat_artifact_hash_mismatch"
    path = certificate_for(tmp_path, beat, feature_implementation_hash="stale")
    assert load_approved_calibration(path, beat)[1] == "calibration_measurement_implementation_mismatch"


def test_missing_certificate_cannot_be_replaced_by_row_flags(tmp_path):
    assert load_approved_calibration(tmp_path / "missing.json", tmp_path / "beats.parquet")[1] == "independent_calibration_certificate_missing"
