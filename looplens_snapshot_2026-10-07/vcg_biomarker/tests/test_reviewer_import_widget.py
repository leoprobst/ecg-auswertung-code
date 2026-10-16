import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from vcg_biomarker.gui.widgets.reviewer_import_widget import ReviewerImportWidget
from vcg_biomarker.mimic.reviewer_export import export_reviewer_packets
from vcg_biomarker.mimic.reviewer_packet import ReviewerPacket
from vcg_biomarker.mimic.reviewer_responses import save_review
from vcg_biomarker.tests.test_reviewer_export import selection


def test_async_import_reports_missing_without_label_changes(tmp_path):
    app = QApplication.instance() or QApplication([])
    export = tmp_path/"overlay"/"review_exports"/"example"
    export_reviewer_packets(selection(), export, signal_loader=lambda _: (np.ones((100,12)),250.))
    packet = ReviewerPacket(export/"reviewer_packets"/"slot-1")
    response = save_review(packet, case_id=packet.cases.review_case_id.iloc[0], verdict="uncertain",
        reviewer_id="reader", response_root=tmp_path/"responses")
    before = response.read_bytes()
    widget = ReviewerImportWidget(tmp_path)
    widget.start_import(export, [response])
    deadline = time.monotonic()+10
    while (widget.busy or not widget.button.isEnabled()) and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.01)
    assert not widget.busy
    app.processEvents()
    assert widget.model.rowCount() == 4
    assert widget.model.frame.status.eq("missing_response").sum() == 3
    assert widget.model.frame.reviewer_verdict.dropna().tolist() == ["uncertain"]
    assert response.read_bytes() == before
    # A rejected follow-up batch must not leave old successful judgments visible.
    widget.start_import(export, [response,response])
    deadline = time.monotonic()+10
    while (widget.busy or not widget.button.isEnabled()) and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert widget.model.rowCount() == 0
    assert "abgelehnt" in widget.status.text()
    widget.close()


def test_outside_export_is_rejected(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = ReviewerImportWidget(tmp_path)
    with pytest.raises(ValueError, match="Originalexport"):
        widget.start_import(tmp_path, [])
    app.processEvents()
    assert not widget.busy
    widget.close()
