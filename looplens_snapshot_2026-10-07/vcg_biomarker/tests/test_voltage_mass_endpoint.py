import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic import voltage_mass_endpoint as module


def test_adapter_preserves_index_and_missing_echo(monkeypatch):
    universe = pd.DataFrame(dict(subject_id=[1, 2, 3], study_id=[11, 22, 33],
        eligibility_state=["eligible"] * 3, label_state=["positive", "negative", "unknown"],
        nearest_echo_measurement_id=[100, 200, 300], nearest_echo_delta_hours=[0, 1000, 0],
        split=["train", "validation", "validation"]))
    geometry = pd.DataFrame(dict(measurement_id=[100, 200, 400], septal_thickness_cm=[1.] * 3,
        posterior_wall_thickness_cm=[1.] * 3, lv_internal_diameter_diastole_cm=[5.] * 3))
    seen = []

    def compare(frame, **kwargs):
        seen.append(frame.copy())
        return pd.DataFrame([dict(status="non_estimable")]), {}

    monkeypatch.setattr(module, "compare_voltage_mass_models", compare)
    rows, _ = module.analyze_voltage_mass_endpoint(universe, geometry,
        endpoint_id="reduced_lvef_strict", iterations=0, seed=1, preview=True)
    for frame in seen:
        assert frame.study_id.tolist() == [11, 22, 33]
        assert np.isfinite(frame.lv_mass_g.iloc[0])
        assert frame.lv_mass_g.iloc[1:].isna().all()
        assert frame.label_state.iloc[2] == "unknown"
    assert rows[0]["patient_index_policy"] == "unchanged_reference_blind_endpoint_index"
    assert np.isnan(rows[0]["validation_brier"])


def test_duplicate_echo_ids_rejected():
    universe = pd.DataFrame({"eligibility_state": ["eligible"]})
    geometry = pd.DataFrame({name: [1, 1] for name in module.ECHO_GEOMETRY_COLUMNS})
    with pytest.raises(ValueError, match="unique"):
        module.analyze_voltage_mass_endpoint(universe, geometry,
            endpoint_id="reduced_lvef", iterations=0, seed=1, preview=True)


def test_geometry_target_not_used_as_its_own_outcome():
    assert module.analyze_voltage_mass_endpoint(pd.DataFrame(), pd.DataFrame(),
        endpoint_id="echo_lvh_strict", iterations=0, seed=1, preview=False) == ([], [])
