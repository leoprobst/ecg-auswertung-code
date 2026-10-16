import json

import pandas as pd
import pytest

from vcg_biomarker.gui.pilot_export import export_pilot_bundle


def fixture(tmp_path):
    (tmp_path / "workspace.json").write_text("{}")
    selection = dict(endpoint="../../unsafe", marker="a/b", pilot_size=1000)
    frame = pd.DataFrame(dict(endpoint_id=[selection["endpoint"]], feature_id=[selection["marker"]]))
    return selection, frame


def render(bundle):
    (bundle / "plot.png").write_bytes(b"test-renderer-png")
    (bundle / "plot.svg").write_text("<svg/>")


def test_unique_bundles_and_safe_names_preserve_exact_selection(tmp_path):
    selection, frame = fixture(tmp_path)
    first = export_pilot_bundle(tmp_path, selection, frame, None, render)
    second = export_pilot_bundle(tmp_path, selection, frame, None, render)
    assert first != second and first.parent == tmp_path / "overlay/exports"
    manifest = json.loads((first / "export_manifest.json").read_text())
    assert manifest["selection"] == selection and manifest["status"] == "complete"
    assert set(manifest["files"]) == {"results.csv", "plot.png", "plot.svg"}
    assert not manifest["clinical_claim_allowed"]


def test_failed_render_is_not_a_complete_export(tmp_path):
    selection, frame = fixture(tmp_path)
    def fail(_):
        raise RuntimeError("renderer failed")
    with pytest.raises(RuntimeError):
        export_pilot_bundle(tmp_path, selection, frame, None, fail)
    manifests = list((tmp_path / "overlay/exports").glob("*/export_manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["status"] == "failed"


def test_mismatched_result_writes_nothing(tmp_path):
    selection, frame = fixture(tmp_path)
    frame["feature_id"] = "other"
    with pytest.raises(ValueError, match="does not match"):
        export_pilot_bundle(tmp_path, selection, frame, None, render)
    assert not (tmp_path / "overlay").exists()


def test_native_svg_contains_vector_graphics(tmp_path):
    import os
    import xml.etree.ElementTree as ET
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.pilot_export import render_plot_svg
    app = QApplication.instance() or QApplication([])
    plot = pg.PlotWidget()
    try:
        plot.resize(640, 320)
        plot.plot([0, 1, 2, 3], [1, 3, 2], stepMode="center", fillLevel=0)
        app.processEvents()
        path = tmp_path / "plot.svg"
        render_plot_svg(plot.plotItem, path)
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        assert any(element.tag.endswith(("path", "polyline")) for element in root.iter())
        assert not any(element.tag.endswith("image") for element in root.iter())
    finally:
        plot.close()


def test_readable_bundle_keeps_source_and_labels(tmp_path):
    import os
    import xml.etree.ElementTree as ET
    import numpy as np
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.pilot_export import render_readable_plot_bundle
    app = QApplication.instance() or QApplication([])
    source = pg.PlotWidget(background="black")
    try:
        source.setLabel("bottom", "Markerwert")
        source.setLabel("left", "Dichte")
        item = source.plot([0, 1, 2, 3], [1, 3, 2], stepMode="center",
                           fillLevel=0, pen="#D55E00", name="Referenz positiv")
        app.processEvents()
        x, y = item.xData.copy(), item.yData.copy()
        bounds = source.viewRange()
        render_readable_plot_bundle(source.plotItem, tmp_path,
            endpoint="echo_endpoint", marker="direct_8lead_marker")
        np.testing.assert_array_equal(item.xData, x)
        np.testing.assert_array_equal(item.yData, y)
        assert source.viewRange() == bounds
        assert source.backgroundBrush().color().name() == "#000000"
        root = ET.parse(tmp_path / "plot.svg").getroot()
        texts = " ".join(root.itertext())
        assert all(text in texts for text in ("echo_endpoint", "direct_8lead_marker", "Markerwert", "Referenz positiv"))
        assert "#d55e00" in (tmp_path / "plot.svg").read_text().lower()
        raster = QImage(str(tmp_path / "plot.png"))
        assert raster.width() == 1200 and raster.height() >= 700
        assert raster.pixelColor(5, 5).lightness() > 240
    finally:
        source.close()
