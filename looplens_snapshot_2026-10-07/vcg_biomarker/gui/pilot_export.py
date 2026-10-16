"""Unique overlay-only export bundles with explicit completion manifests."""
import hashlib
from pathlib import Path
import tempfile

from ..mimic.storage import atomic_write_json, file_sha256


def render_readable_plot_bundle(source, bundle, *, endpoint, marker):
    """Export a separate light figure without changing the live plot or data."""
    import html
    import textwrap

    import pyqtgraph as pg
    import pyqtgraph.exporters
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    figure = pg.PlotWidget(background="white")
    figure.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    try:
        figure.setFixedSize(1200, 800)
        plot = figure.plotItem
        plot.setContentsMargins(24, 20, 24, 24)
        title = "<br>".join(html.escape(line) for value in
            (endpoint, marker, "Ausgewaehlte Development-Stichprobe | Pilot-Hypothese | keine klinische Freigabe")
            for line in textwrap.wrap(value, width=64))
        plot.setTitle(title, color="#222222", size="12pt")
        # pyqtgraph reserves a single-line title height by default, even for
        # HTML line breaks. Reserve the measured text height before layout.
        plot.titleLabel.setFixedHeight(plot.titleLabel.item.boundingRect().height() + 20)
        plot.addLegend(labelTextColor="#222222", labelTextSize="11pt")
        for side in ("left", "bottom"):
            original = source.getAxis(side)
            axis = plot.getAxis(side)
            axis.setPen(pg.mkPen("#333333"))
            axis.setTextPen(pg.mkPen("#222222"))
            axis.setLabel(original.labelText, units=original.labelUnits, color="#222222")
        plot.getAxis("bottom").setHeight(64)
        plot.getAxis("left").setWidth(100)
        for item in source.listDataItems():
            options = {key: item.opts[key] for key in
                ("stepMode", "fillLevel", "connect", "symbol", "symbolSize") if key in item.opts}
            color = pg.mkPen(item.opts["pen"]).color().name()
            fill = pg.mkColor(color)
            fill.setAlpha(55)
            plot.plot(item.xData, item.yData, name=item.name(),
                      pen=pg.mkPen(color, width=2), brush=pg.mkBrush(fill),
                      symbolPen=pg.mkPen(color), symbolBrush=pg.mkBrush(color), **options)
        bounds = source.viewRange()
        plot.setRange(xRange=bounds[0], yRange=bounds[1], padding=0)
        figure.show()
        QApplication.instance().processEvents()
        exporter = pg.exporters.ImageExporter(plot)
        exporter.parameters()["width"] = 1200
        exporter.export(str(Path(bundle) / "plot.png"))
        render_plot_svg(plot, Path(bundle) / "plot.svg")
    finally:
        figure.close()


def render_plot_svg(plot_item, path):
    """Render Qt scene directly; avoid SVG re-parsing of Qt-generated paths."""
    import math
    from PyQt6.QtCore import QRectF, QSize
    from PyQt6.QtGui import QPainter
    from PyQt6.QtSvg import QSvgGenerator

    bounds = plot_item.sceneBoundingRect()
    width, height = math.ceil(bounds.width()), math.ceil(bounds.height())
    if width <= 0 or height <= 0:
        raise ValueError("Plot has no exportable geometry")
    generator = QSvgGenerator()
    generator.setFileName(str(path))
    generator.setSize(QSize(width, height))
    target = QRectF(0, 0, width, height)
    generator.setViewBox(target)
    painter = QPainter()
    if not painter.begin(generator):
        raise OSError("Cannot start SVG export")
    try:
        plot_item.scene().render(painter, target, bounds)
    finally:
        painter.end()


def export_pilot_bundle(workspace, selection, frame, timeline, render_plots):
    root = Path(workspace).resolve()
    for column, key in (("endpoint_id", "endpoint"), ("feature_id", "marker")):
        if frame.empty or column not in frame or not frame[column].eq(selection[key]).all():
            raise ValueError("Export selection does not match loaded results")
    overlay = root / "overlay"
    exports = overlay / "exports"
    if overlay.resolve() != overlay or exports.resolve() != exports:
        raise ValueError("Export overlay must not be redirected")
    exports.mkdir(parents=True, exist_ok=True)
    bundle = Path(tempfile.mkdtemp(prefix="selection-", dir=exports))
    manifest = dict(version="1.0.0", selection=dict(selection), clinical_claim_allowed=False,
                    source_workspace_manifest_sha256=file_sha256(root / "workspace.json"),
                    status="incomplete", files={})
    atomic_write_json(bundle / "export_manifest.json", manifest)
    try:
        frame.to_csv(bundle / "results.csv", index=False)
        if timeline is not None:
            timeline.to_csv(bundle / "timeline.csv", index=False)
        render_plots(bundle)
        required = ["results.csv", "plot.png", "plot.svg"]
        if timeline is not None:
            required.append("timeline.csv")
        for name in required:
            path = bundle / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Export missing or empty: {name}")
            manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["status"] = "complete"
        atomic_write_json(bundle / "export_manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        atomic_write_json(bundle / "export_manifest.json", manifest)
        raise
    return bundle
