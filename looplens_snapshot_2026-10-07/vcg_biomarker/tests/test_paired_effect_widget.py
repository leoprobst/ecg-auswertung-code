import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication

from vcg_biomarker.gui.widgets.paired_effect_widget import PairedEffectWidget


def result():
    return dict(design="dialysis_potassium", feature_id="example", association_status="estimable",
        spearman_rho=.4, rank_slope_ci_low=.2, rank_slope_ci_high=.6, n_pairs=400, n_patients=100, global_q_value=.01,
        adjusted_model="timing_quality_sensitivity_only", adjusted_status="estimable",
        adjusted_rank_slope=.25, adjusted_ci_low=.1, adjusted_ci_high=.4, adjusted_n_complete=300,
        adjusted_n_patients=80, adjusted_global_q_value=.03, unadjusted_same_sample_rank_slope=.35)


def test_effects_keep_distinct_samples_and_missing_common_sample_ci():
    app = QApplication.instance() or QApplication([])
    widget = PairedEffectWidget()
    widget.set_result(result())
    app.processEvents()
    assert len(widget.rendered_effects) == 3
    assert not widget.plot.getAxis("bottom").autoSIPrefix
    assert widget.rendered_effects[1]["ci"] is None
    assert widget.table.item(0, 2).text() == "400 / 100"
    assert widget.table.item(1, 2).text() == "300 / 80"
    assert widget.table.item(2, 3).text() == "0.03"
    assert "Separate" in widget.table.item(2, 4).text()
    widget.set_result({})
    assert widget.rendered_effects == []
    assert widget.plot.isHidden()
    assert all(widget.table.item(i, j).text() == "—" for i in range(3) for j in range(5))
    widget.close()


@pytest.mark.parametrize("row", [None, {}, {"adjusted_status": pd.NA, "adjusted_model": pd.NA},
    {**result(), "adjusted_status": "non_estimable", "adjusted_abstention_reason": "collinear_design"},
    {**result(), "adjusted_model": "unknown_future_model"}])
def test_missing_unknown_or_abstained_adjustment_never_draws_stale_effect(row):
    app = QApplication.instance() or QApplication([])
    widget = PairedEffectWidget()
    widget.set_result(result())
    widget.set_result(row)
    app.processEvents()
    assert not any(item["row"] == 2 for item in widget.rendered_effects)
    widget.close()


def test_nonfinite_boolean_and_reversed_ci_are_not_plotted_as_uncertainty():
    app = QApplication.instance() or QApplication([])
    widget = PairedEffectWidget()
    widget.set_result({**result(), "spearman_rho": np.bool_(True), "adjusted_ci_low": .5,
                       "adjusted_ci_high": -.5, "adjusted_global_q_value": np.inf})
    app.processEvents()
    assert not any(item["row"] == 0 for item in widget.rendered_effects)
    assert widget.rendered_effects[-1]["ci"] is None
    assert widget.table.item(2, 3).text() == "—"
    widget.close()


def test_model_selector_uses_clinical_sample_and_never_falls_back_to_timing():
    app = QApplication.instance() or QApplication([])
    widget = PairedEffectWidget()
    row = {**result(), "clinical_adjusted_model": "baseline_clinical_timing_quality_sensitivity",
        "clinical_adjusted_status": "estimable", "clinical_adjusted_rank_slope": .12,
        "clinical_adjusted_ci_low": -.03, "clinical_adjusted_ci_high": .27,
        "clinical_adjusted_n_complete": 200, "clinical_adjusted_n_patients": 55,
        "clinical_adjusted_global_q_value": .2, "clinical_unadjusted_same_sample_rank_slope": .3}
    widget.set_result(row)
    widget.model_selector.setCurrentIndex(1)
    app.processEvents()
    assert widget.rendered_effects[-1]["effect"] == .12
    assert widget.table.item(1, 2).text() == "200 / 55"
    assert widget.table.item(2, 3).text() == "0.2"
    widget.set_result(result())
    assert not any(item["row"] == 2 for item in widget.rendered_effects)
    widget.model_selector.setCurrentIndex(0)
    assert widget.rendered_effects[-1]["effect"] == .25
    widget.set_result({})
    widget.model_selector.setCurrentIndex(1)
    assert widget.rendered_effects == []
    widget.close()


def test_joint_model_is_distinct_from_clinical_and_missing_does_not_fallback():
    app = QApplication.instance() or QApplication([])
    widget = PairedEffectWidget()
    widget.set_result({**result(), "joint_adjusted_model": "joint_physiology_baseline_timing_quality_sensitivity",
        "joint_adjusted_status": "estimable", "joint_adjusted_rank_slope": .07,
        "joint_adjusted_n_complete": 90, "joint_adjusted_n_patients": 40,
        "joint_adjusted_ci_low": -.04, "joint_adjusted_ci_high": .18})
    widget.model_selector.setCurrentIndex(2)
    app.processEvents()
    assert widget.rendered_effects[-1]["effect"] == .07
    assert widget.table.item(2, 2).text() == "90 / 40"
    assert "Volumen" in widget.note.text()
    widget.set_result(result())
    assert not any(point["row"] == 2 for point in widget.rendered_effects)
    widget.close()
