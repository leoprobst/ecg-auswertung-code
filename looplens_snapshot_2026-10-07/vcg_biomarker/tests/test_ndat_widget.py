import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
from vcg_biomarker.gui.widgets.ndat_widget import NdatWidget


def test_partial_and_empty_views_clear_old_points():
    app = QApplication.instance() or QApplication([])
    widget = NdatWidget()
    widget.set_result({"status": "not_evaluable", "reason": "missing_lead",
        "leads": {"V1": {"time_ms": 50., "status": "evaluable", "n_assessable": 3,
                          "n_beats": 4, "mad_ms": 2.}}})
    assert widget.table.item(0, 1).text() == "50.0"
    assert widget.table.item(1, 1).text() == "nicht messbar"
    assert len(widget.plot.listDataItems()) == 1
    assert "missing_lead" in widget.note.text()
    widget.set_result({})
    assert not widget.plot.listDataItems()
    assert widget.table.item(0, 1).text() == "nicht messbar"
    widget.close()
    app.processEvents()
