import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time

import pytest
from PyQt6.QtWidgets import QApplication

from vcg_biomarker.gui.widgets.reviewer_export_widget import ReviewerExportWidget
from vcg_biomarker.tests.test_pilot_reviewer_export import archive


def test_gui_exports_in_real_child_process_without_blocking(tmp_path):
    selection = archive(tmp_path)
    overlay = tmp_path/"overlay"
    overlay.mkdir()
    path = overlay/"coordinator.parquet"
    selection.to_parquet(path)
    app = QApplication.instance() or QApplication([])
    widget = ReviewerExportWidget(tmp_path)
    widget.start_export(path)
    assert widget.busy
    assert not widget.button.isEnabled()
    with pytest.raises(ValueError,match="bereits"):
        widget.start_export(path)
    deadline = time.monotonic()+25
    while widget.busy and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    try:
        assert not widget.busy, widget.status.text()
        assert widget.completed, widget.status.text()
        assert widget.button.isEnabled()
        assert (widget.destination/"export_complete.json").is_file()
        assert "2 Beurteiler" in widget.status.text()
    finally:
        if widget.busy:
            widget.process.kill()
            widget.process.waitForFinished(3000)
        widget.close()


def test_gui_rejects_selection_outside_overlay_before_launch(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = ReviewerExportWidget(tmp_path)
    source = tmp_path/"private.parquet"
    source.touch()
    with pytest.raises(ValueError,match="Overlay"):
        widget.start_export(source)
    assert not widget.busy
    assert not (tmp_path/"overlay").exists()
    widget.close()
    app.processEvents()
