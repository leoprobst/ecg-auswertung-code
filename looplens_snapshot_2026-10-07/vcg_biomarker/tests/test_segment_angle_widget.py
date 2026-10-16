import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QAbstractItemView
from vcg_biomarker.gui.widgets.segment_angle_widget import SegmentAngleWidget


def test_matrix_cells_missing_reset_and_read_only():
    app = QApplication.instance() or QApplication([])
    widget = SegmentAngleWidget()
    cells = [dict(qrs_third=i, repolarization_third=j, angle_deg=30.*i,
                  status="evaluable", magnitude_pair_fraction=1/9)
             for i in range(1, 4) for j in range(1, 4)]
    widget.set_result(dict(status="evaluable", cells=cells))
    app.processEvents()
    assert widget.table.item(2, 2).text() == "90.0°"
    assert widget.table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    widget.set_result({})
    assert all(widget.table.item(i, j).text() == "—" for i in range(3) for j in range(3))
    widget.close()


def test_malformed_duplicate_and_abstained_cells_do_not_show_angles():
    app = QApplication.instance() or QApplication([])
    widget = SegmentAngleWidget()
    cell = dict(qrs_third=1, repolarization_third=2, angle_deg=90., status="evaluable")
    widget.set_result(dict(status="evaluable", cells=[cell, cell]))
    assert widget.table.item(0, 1).text() == "mehrdeutig"
    for payload in (None, {"cells": None}, {"cells": [None]},
                    {"cells": [{"qrs_third": "1"}]},
                    dict(status="not_evaluable", cells=[cell]),
                    dict(status="evaluable", cells=[{**cell, "angle_deg": "bad"}])):
        widget.set_result(payload)
        app.processEvents()
        assert all("°" not in widget.table.item(i, j).text() for i in range(3) for j in range(3))
    widget.close()
