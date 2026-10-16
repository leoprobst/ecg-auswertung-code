import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.voltage_mass_confirmation import adjust_voltage_mass_confirmation
from vcg_biomarker.mimic.voltage_mass import VOLTAGE_MARKERS
from vcg_biomarker.mimic.voltage_mass_endpoint import VOLTAGE_MASS_ENDPOINTS


def row(**kwargs):
    return dict(endpoint_id=VOLTAGE_MASS_ENDPOINTS[0], voltage_marker=VOLTAGE_MARKERS[0],
                model="joint_quadratic_plus_ratio", confirmation_status="estimable",
                confirmation_p_value=.001, **kwargs)


def test_missing_planned_tests_count_without_reported_replacement_p():
    first = row()
    failed = {**row(), "voltage_marker": VOLTAGE_MARKERS[1],
              "confirmation_status": "non_estimable", "confirmation_p_value": np.nan}
    result = adjust_voltage_mass_confirmation(pd.DataFrame([first, failed]))
    assert result.confirmation_q_global_bh.iloc[0] == pytest.approx(.001 * len(VOLTAGE_MARKERS) * len(VOLTAGE_MASS_ENDPOINTS))
    assert np.isnan(result.confirmation_p_value.iloc[1])
    assert np.isnan(result.confirmation_q_global_bh.iloc[1])
    assert result.confirmation_q_global_by.iloc[0] >= result.confirmation_q_global_bh.iloc[0]


def test_duplicate_or_unplanned_models_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        adjust_voltage_mass_confirmation(pd.DataFrame([row(), row()]))
    with pytest.raises(ValueError, match="not planned"):
        adjust_voltage_mass_confirmation(pd.DataFrame([{**row(), "model": "ratio"}]))


def test_preview_and_empty_do_not_generate_evidence():
    assert adjust_voltage_mass_confirmation(pd.DataFrame()).empty
    result = adjust_voltage_mass_confirmation(pd.DataFrame([{**row(), "confirmation_status": "descriptive_only"}]))
    assert result.confirmation_q_global_bh.isna().all()
