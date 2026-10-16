"""Render production transform plotting with supplied real result artifacts."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtGui import QFont, QFontDatabase
    from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QLabel
    import pyqtgraph as pg
    from ..gui.widgets.mimic_project_widget import MimicProjectWidget
    from .storage import file_sha256, atomic_write_json
    before = file_sha256(args.results)
    app = QApplication([])
    if QFontDatabase.addApplicationFont(str(Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts/segoeui.ttf")) < 0:
        raise RuntimeError("Required readable font unavailable")
    app.setStyle("Fusion"); app.setFont(QFont("Segoe UI", 10))
    pg.setConfigOptions(background="w", foreground="#17212b")
    window = QWidget(); layout = QVBoxLayout(window)
    layout.addWidget(QLabel("Historische A023-Trainingsmessungen — keine Validierung aktueller Formeln"))
    host = SimpleNamespace(transform_marker_combo=QComboBox(), transform_agreement_summary=QLabel(),
                           transform_agreement_plot=pg.PlotWidget())
    layout.addWidget(host.transform_marker_combo); layout.addWidget(host.transform_agreement_summary)
    layout.addWidget(host.transform_agreement_plot)
    MimicProjectWidget._draw_transform_agreement(host, args.results)
    window.resize(1366, 768); window.show(); app.processEvents()
    args.output.mkdir(parents=True, exist_ok=True)
    images = []
    for marker in ("spatial_qrs_t_angle_deg", "p_loop_width_major"):
        if host.transform_marker_combo.findText(marker) < 0:
            raise ValueError(f"Required real audit marker absent: {marker}")
        host.transform_marker_combo.setCurrentText(marker)
        MimicProjectWidget._draw_transform_agreement(host, args.results, reload=False)
        app.processEvents()
        target = args.output / f"{marker}.png"
        if not window.grab().save(str(target)):
            raise RuntimeError("Screenshot write failed")
        images.append({"marker": marker, "path": str(target), "sha256": file_sha256(target)})
    window.close()
    if file_sha256(args.results) != before:
        raise RuntimeError("Source changed during rendering")
    atomic_write_json(args.output/"manifest.json", {"source_sha256": before, "screenshots": images,
        "scope": "production_plot_method_in_isolated_container_real_historical_features",
        "full_production_gui_accepted": False, "source_modified": False})


if __name__ == "__main__":
    main()
