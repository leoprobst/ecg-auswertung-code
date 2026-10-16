import pandas as pd
import pytest

from vcg_biomarker.mimic import voltage_mass_endpoint as module


@pytest.mark.parametrize("endpoint", ["aortic_regurgitation", "aortic_regurgitation_severe"])
def test_ar_dispatch_uses_existing_components_without_reselecting_patients(monkeypatch, endpoint):
    assert endpoint in module.VOLTAGE_MASS_ENDPOINTS
    original = pd.DataFrame({"subject_id": [42], "nearest_echo_measurement_id": [7],
                             "nearest_echo_delta_hours": [12.]})
    geometry = pd.DataFrame({"measurement_id": [7], "septal_thickness_cm": [1.],
                             "posterior_wall_thickness_cm": [1.],
                             "lv_internal_diameter_diastole_cm": [5.]})
    seen = []
    monkeypatch.setattr(module, "prepare_lead_frame", lambda frame: frame.copy())
    def compare(frame, **kwargs):
        seen.append(frame.subject_id.tolist())
        assert frame.lv_mass_g.notna().all()
        return pd.DataFrame({"model": ["joint_quadratic_plus_ratio"], "validation_brier": [.2]}), {}
    monkeypatch.setattr(module, "compare_voltage_mass_models", compare)
    rows, models = module.analyze_voltage_mass_endpoint(
        original, geometry, endpoint_id=endpoint, iterations=0, seed=1, preview=True)
    assert len(rows) == len(models) == len(module.VOLTAGE_MARKERS)
    assert seen == [[42]] * len(module.VOLTAGE_MARKERS)
    assert "lv_mass_g" not in original
    for row in rows:
        assert row["analysis_temporality"] == "cross_sectional_not_ecg_time_prediction"
        assert "not_CMR" in row["component_interpretation"]
        assert not row["clinical_claim_allowed"]
        assert pd.isna(row["validation_brier"])
