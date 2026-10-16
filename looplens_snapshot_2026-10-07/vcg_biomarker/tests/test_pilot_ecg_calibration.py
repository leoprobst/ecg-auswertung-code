import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from vcg_biomarker.gui.widgets.pilot_workspace_widget import CompactPlotWidget, PilotWorkspaceDialog


@pytest.mark.parametrize("height", [100,180,320])
def test_calibration_label_has_layout_space_and_pulse_preserves_units(height):
    app = QApplication.instance() or QApplication([])
    plot = CompactPlotWidget()
    plot.resize(900,height)
    owner = SimpleNamespace(ecg_plot=plot)
    PilotWorkspaceDialog.draw_ecg_calibration(owner)
    plot.show()
    app.processEvents()
    x,y = plot.listDataItems()[0].getData()
    assert np.ptp(y) == 1.
    assert x[3]-x[2] == pytest.approx(.2)
    assert plot.plotItem.titleLabel.text == "Referenzpuls: 1 mV / 200 ms"
    assert plot.plotItem.titleLabel.isVisible()
    assert plot.plotItem.titleLabel.geometry().height() > 0
    # Zoom changes visual size, never the stated physical calibration values.
    plot.setXRange(-.5,.5)
    app.processEvents()
    np.testing.assert_array_equal(plot.listDataItems()[0].getData()[1], y)
    plot.close()
