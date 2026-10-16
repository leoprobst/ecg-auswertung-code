import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QComboBox, QLabel

from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget


def test_transform_plot_never_pools_different_markers_on_one_axis(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    host = SimpleNamespace(transform_agreement_plot=pg.PlotWidget(), transform_marker_combo=QComboBox(),
                           transform_agreement_summary=QLabel())
    path = tmp_path / "agreement.parquet"
    rows = [
        {"marker_id": marker, "bias": bias, "limit_of_agreement_low": bias-1,
         "limit_of_agreement_high": bias+1, "primary_transform": "kors",
         "comparison_transform": "dower", "paired_count": 300}
        for marker, bias in [("angle", 20), ("voltage", .1)]
    ]
    rows.append({"marker_id": "sparse", "bias": float("nan"),
                 "limit_of_agreement_low": float("nan"), "limit_of_agreement_high": float("nan"),
                 "primary_transform": "kors", "comparison_transform": "dower", "paired_count": 40,
                 "index_record_count": 100, "paired_availability_fraction": .4, "status": "too_few_pairs"})
    pd.DataFrame(rows).to_parquet(path)
    draw = lambda: MimicProjectWidget._draw_transform_agreement(host, path)
    draw()
    assert host.transform_marker_combo.count() == 3
    assert host.transform_marker_combo.currentText() == "angle"
    points = [item for item in host.transform_agreement_plot.items() if isinstance(item, pg.ScatterPlotItem)]
    assert len(points) == 1 and points[0].getData()[0].tolist() == [20.]
    assert host.transform_agreement_plot.viewRange()[1] == [-.5, .5]
    host.transform_marker_combo.setCurrentText("sparse")
    MimicProjectWidget._draw_transform_agreement(host, path, reload=False)
    assert "too_few_pairs" in host.transform_agreement_summary.text()
    assert "n=40 von 100" in host.transform_agreement_summary.text()
    assert not any(isinstance(item, pg.ScatterPlotItem) for item in host.transform_agreement_plot.items())
    host.transform_marker_combo.setCurrentText("voltage")
    original_read = pd.read_parquet
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Marker navigation must not reread Parquet")
    monkeypatch.setattr(pd, "read_parquet", unexpected_read)
    MimicProjectWidget._draw_transform_agreement(host, path, reload=False)
    points = [item for item in host.transform_agreement_plot.items() if isinstance(item, pg.ScatterPlotItem)]
    assert len(points) == 1 and points[0].getData()[0].tolist() == [.1]
    monkeypatch.setattr(pd, "read_parquet", original_read)
    updated = original_read(path)
    updated.loc[updated.marker_id.eq("voltage"), ["bias", "limit_of_agreement_low", "limit_of_agreement_high"]] = [.2, -.8, 1.2]
    updated.to_parquet(path)
    draw()
    points = [item for item in host.transform_agreement_plot.items() if isinstance(item, pg.ScatterPlotItem)]
    assert points[0].getData()[0].tolist() == [.2]
    MimicProjectWidget._draw_transform_agreement(host, tmp_path / "missing.parquet")
    assert host.transform_marker_combo.count() == 0
    assert not any(isinstance(item, pg.ScatterPlotItem) for item in host.transform_agreement_plot.items())
    host.transform_agreement_plot.close()
    app.processEvents()
