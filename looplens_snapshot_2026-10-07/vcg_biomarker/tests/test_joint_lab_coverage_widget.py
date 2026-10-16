import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication

from vcg_biomarker.gui.widgets.joint_lab_coverage_widget import JointLabCoverageWidget, coverage_rows


def summary():
    return dict(status="computed", n_pairs=100, n_pairs_complete_core=10, n_patients_complete_core=7,
        analytes=dict(potassium=dict(n_pairs=90, n_patients=60),
            calcium_total=dict(n_pairs=70, n_patients=50), calcium_ionized=dict(n_pairs=20, n_patients=15),
            magnesium=dict(n_pairs=60, n_patients=40), ph=dict(n_pairs=30, n_patients=20)))


@pytest.mark.parametrize("data", [None, {}, {"status": pd.NA}, {"status": "computed", "n_pairs": True}])
def test_missing_contract_is_not_zero_coverage(data):
    assert coverage_rows(data) == (None, [])


def test_coverage_counts_and_invalid_cells():
    total, rows = coverage_rows(summary())
    assert total == 100 and rows[0]["percent"] == 90 and rows[-1]["percent"] == 10
    assert rows[1]["n_pairs"] == 70 and rows[2]["n_pairs"] == 20
    data = summary()
    data["analytes"]["potassium"]["n_pairs"] = 101
    data["analytes"]["ph"]["n_patients"] = 31
    rows = coverage_rows(data)[1]
    assert rows[0]["percent"] is None and rows[4]["percent"] is None
    data = summary()
    data["n_pairs_complete_core"] = 25  # More than the 20 ionized-calcium pairs.
    assert coverage_rows(data)[1][-1]["percent"] is None


def test_widget_and_real_project_refresh_clear_previous_coverage(tmp_path, monkeypatch):
    from vcg_biomarker.gui.widgets import mimic_project_widget
    from vcg_biomarker.tests.test_provenance_integrity import _config
    config = _config(tmp_path/"isolated")
    monkeypatch.setattr(mimic_project_widget, "load_or_discover_config", lambda: config)
    app = QApplication.instance() or QApplication([])
    component = JointLabCoverageWidget()
    component.set_summary(summary())
    assert len(component.rows) == 6
    component.set_summary(None)
    assert component.rows == [] and component.plot.isHidden()
    component.close()
    root = tmp_path/"run"
    (root/"within_subject").mkdir(parents=True)
    (root/"within_subject"/"within_subject_summary.json").write_text(json.dumps(dict(joint_lab_coverage=summary())))
    widget = mimic_project_widget.MimicProjectWidget()
    widget._poll_timer.stop()
    try:
        widget._refresh_targeted_designs(root)
        app.processEvents()
        assert widget.joint_lab_coverage_view.rows[-1]["n_pairs"] == 10
        widget._refresh_targeted_designs(tmp_path/"missing_run")
        assert widget.joint_lab_coverage_view.rows == []
    finally:
        widget.close()
