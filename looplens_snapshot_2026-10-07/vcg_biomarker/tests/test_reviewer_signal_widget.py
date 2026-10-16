import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from PyQt6.QtWidgets import QApplication
from vcg_biomarker.mimic.reviewer_packet import ReviewerPacket
from vcg_biomarker.gui.widgets.reviewer_signal_widget import ReviewerSignalWidget
from vcg_biomarker.tests.test_reviewer_packet import packet


def test_twelve_leads_shared_scale_explicit_save_and_clear_on_bad_signal(tmp_path):
    app = QApplication.instance() or QApplication([])
    model = ReviewerPacket(packet(tmp_path))
    widget = ReviewerSignalWidget(model,tmp_path/"answers")
    app.processEvents()
    assert len(widget.plots) == 12
    assert all(len(p.listDataItems()) == 1 for p in widget.plots)
    assert widget.loaded_case is not None
    widget.save_verdict()
    assert "Nicht gespeichert" in widget.status.text()
    assert not (tmp_path/"answers").exists()
    widget.reviewer.setText("synthetic-reader")
    widget.verdict.setCurrentIndex(3)
    widget.save_verdict()
    assert not widget.save.isEnabled()
    assert len(list((tmp_path/"answers").rglob("*.json"))) == 1
    next_case = model.cases.review_case_id.iloc[1]
    (model.folder/f"{next_case}.npz").write_bytes(b"changed")
    widget.case.setCurrentIndex(1)
    app.processEvents()
    assert widget.loaded_case is None
    assert not widget.save.isEnabled()
    assert all(not p.listDataItems() for p in widget.plots)
    widget.close()
