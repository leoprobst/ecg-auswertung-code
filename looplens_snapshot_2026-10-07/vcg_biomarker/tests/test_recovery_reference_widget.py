import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PyQt6.QtWidgets import QApplication, QAbstractItemView

from vcg_biomarker.gui.widgets.recovery_reference_widget import RecoveryReferenceWidget


def test_five_references_status_gates_and_reset():
    app = QApplication.instance() or QApplication([])
    widget = RecoveryReferenceWidget()
    row = dict(marker_id="example", episode_id="episode", status="evaluable",
        normalization_status="evaluable", reference_normalization_status="observed",
        normalization_residual=-2., normalization_ecg_lag_hours=1.,
        reference_unit="mEq/L", reference_itemid=50971,
        reference_lower=3.5, reference_upper=5.,
        magnesium__normalization_status="not_evaluable",
        magnesium__normalization_residual=123.,
        reference_magnesium_normalization_reason="no_baseline_lab_in_window")
    widget.set_result(row)
    app.processEvents()
    assert widget.table.rowCount() == 5
    assert widget.table.item(0, 3).text() == "-2"
    assert "mEq/L" in widget.table.item(0, 3).toolTip()
    assert widget.table.item(3, 3).text() == "—"
    assert "no_baseline_lab" in widget.table.item(3, 4).text()
    assert "Nicht berechnet" in widget.table.item(1, 4).text()
    assert widget.table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    widget.set_result({**row, "status": "not_evaluable"})
    assert widget.table.item(0, 3).text() == "—"
    widget.set_result({})
    assert all(widget.table.item(i, 3).text() == "—" for i in range(5))
    widget.close()


def test_partial_legacy_and_missing_values_are_not_normal():
    app = QApplication.instance() or QApplication([])
    widget = RecoveryReferenceWidget()
    for row in (None, {"status": pd.NA, "normalization_reason": pd.NA},
                {"status": "evaluable", "normalization_status": "evaluable",
                 "normalization_residual": 1.}):
        widget.set_result(row)
        app.processEvents()
        assert all(widget.table.item(i, 3).text() == "—" for i in range(5))
    widget.close()
