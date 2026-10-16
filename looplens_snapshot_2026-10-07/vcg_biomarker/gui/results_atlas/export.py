"""Local, user-initiated exports of atlas aggregates and plots."""
from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPainter
from PyQt6.QtSvg import QSvgGenerator
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget
from pyqtgraph.exporters import ImageExporter, SVGExporter


LOGGER = logging.getLogger(__name__)


def _write_provenance(path: Path, metadata: dict[str, object]) -> Path:
    """Never silently replace a pre-existing provenance sidecar."""
    digest = hashlib.sha256()
    with path.open("rb") as exported:
        for chunk in iter(lambda: exported.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = dict(metadata)
    provenance["_export"] = {"filename": path.name, "sha256": digest.hexdigest()}
    base = path.with_name(path.name + ".provenance.json")
    candidate = base
    index = 1
    while candidate.exists():
        candidate = path.with_name(path.name + f".provenance-{index}.json")
        index += 1
    candidate.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return candidate


def save_plot(plot: pg.PlotWidget, path: Path, metadata: dict[str, object]) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".png":
        ImageExporter(plot.plotItem).export(str(path))
    elif suffix == ".svg":
        try:
            SVGExporter(plot.plotItem).export(str(path))
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            # Some pyqtgraph/Qt combinations cannot serialize ViewBox
            # transforms.  Qt's own vector renderer remains a valid SVG.
            LOGGER.warning("pyqtgraph SVG export failed; using Qt renderer: %s", exc)
            generator = QSvgGenerator()
            generator.setFileName(str(path))
            generator.setSize(QSize(max(plot.width(), 640), max(plot.height(), 400)))
            painter = QPainter(generator)
            try:
                plot.render(painter)
            finally:
                painter.end()
    else:
        raise ValueError("Grafikexport unterstützt nur PNG und SVG")
    return _write_provenance(path, metadata)


def save_table(frame: pd.DataFrame, path: Path, metadata: dict[str, object]) -> Path:
    if path.suffix.lower() != ".csv":
        raise ValueError("Tabellenexport unterstützt nur CSV")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return _write_provenance(path, metadata)


def choose_plot_export(
    parent: QWidget, plot: pg.PlotWidget, metadata: dict[str, object],
    *, proposed_name: str = "ergebnisatlas.png",
) -> None:
    filename, selected = QFileDialog.getSaveFileName(
        parent, "Diagramm lokal exportieren", proposed_name,
        "PNG-Bild (*.png);;SVG-Grafik (*.svg)",
    )
    if not filename:
        return
    path = Path(filename)
    if not path.suffix:
        path = path.with_suffix(".svg" if "SVG" in selected else ".png")
    try:
        sidecar = save_plot(plot, path, metadata)
    except (OSError, ValueError, RuntimeError) as exc:
        QMessageBox.warning(parent, "Export fehlgeschlagen", str(exc))
        return
    QMessageBox.information(parent, "Export fertig", f"Grafik: {path}\nHerkunft: {sidecar}")


def choose_table_export(
    parent: QWidget, frame: pd.DataFrame, metadata: dict[str, object],
    *, proposed_name: str = "ergebnisatlas.csv",
) -> None:
    filename, _selected = QFileDialog.getSaveFileName(
        parent, "Tabelle lokal exportieren", proposed_name,
        "CSV-Tabelle (*.csv)",
    )
    if not filename:
        return
    path = Path(filename)
    if not path.suffix:
        path = path.with_suffix(".csv")
    try:
        sidecar = save_table(frame, path, metadata)
    except (OSError, ValueError, RuntimeError) as exc:
        QMessageBox.warning(parent, "Export fehlgeschlagen", str(exc))
        return
    QMessageBox.information(parent, "Export fertig", f"Tabelle: {path}\nHerkunft: {sidecar}")
