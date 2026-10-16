"""Render the actual lead-result widget; synthetic fixtures are explicitly labelled."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", choices=("light", "dark"), required=True)
    parser.add_argument("--dpi", choices=("1", "1.25"), default="1")
    args = parser.parse_args()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = args.dpi
    import pandas as pd
    from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette
    from PyQt6.QtWidgets import QApplication
    import pyqtgraph as pg
    from ..gui.widgets.lead_comparison_widget import LeadComparisonWidget
    from ..core.data_loader import LEAD_NAMES_12
    from .storage import atomic_write_json, file_sha256

    app = QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
    if QFontDatabase.addApplicationFont(str(font_path)) < 0:
        raise RuntimeError("Audit renderer could not load Segoe UI; do not accept unreadable screenshots")
    app.setStyle("Fusion"); app.setFont(QFont("Segoe UI", 10))
    dark = args.theme == "dark"
    base, text = ("#101820", "#edf2f7") if dark else ("#ffffff", "#17212b")
    palette = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(role, QColor(base))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(role, QColor(text))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#202d38" if dark else "#edf2f7"))
    app.setPalette(palette)
    pg.setConfigOptions(background=base, foreground=text)
    if args.synthetic:
        rows = []
        for index, lead in enumerate(LEAD_NAMES_12):
            rows.append({"endpoint_id": "SYNTHETISCH — keine Pilotbefunde", "family": "qrs", "lead": lead,
                "status": "estimable" if index < 10 else "non_estimable", "reference_kind": "synthetic_fixture",
                "n_validation_common": 240, "n_validation_positive": 80, "n_validation_negative": 160,
                "n_index": 1000, "n_train_common": 560, "n_common_complete": 800,
                "n_excluded_reference": 100, "n_excluded_extraction_after_reference": 20,
                "n_excluded_covariates_after_extraction": 30, "n_excluded_leads_after_covariates": 50,
                "delta_auc_clinical_baseline": .01*(index-3),
                "delta_auc_clinical_baseline_ci_low": .01*(index-3)-.04,
                "delta_auc_clinical_baseline_ci_high": .01*(index-3)+.04 if index < 8 else None,
                "abstention_reason": "synthetic_non_estimable" if index >= 10 else ""})
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_parquet(args.results)
    widget = LeadComparisonWidget(); widget.set_frame(frame); widget.show()
    args.output.mkdir(parents=True, exist_ok=True)
    screenshots = []
    for width, height in ((1366, 768), (1440, 900), (1920, 1080)):
        widget.resize(width, height); app.processEvents(); widget.repaint(); app.processEvents()
        target = args.output / f"lead_{args.theme}_{args.dpi}_{width}x{height}.png"
        if not widget.grab().save(str(target)):
            raise RuntimeError("Screenshot could not be saved")
        screenshots.append({"path": str(target.resolve()), "sha256": file_sha256(target),
            "logical_size": [widget.width(), widget.height()], "device_pixel_ratio": widget.devicePixelRatioF(),
            "plot_size": [widget.plot.width(), widget.plot.height()],
            "rendered_effects": widget.rendered_effects, "summary": widget.summary.text()})
    widget.close()
    atomic_write_json(args.output / f"lead_{args.theme}_{args.dpi}.manifest.json", {
        "scope": "synthetic_widget_layout_not_clinical_or_full_application_validation" if args.synthetic else "provided_lead_results_widget_layout",
        "source_sha256": None if args.synthetic else file_sha256(args.results),
        "screenshots": screenshots, "theme": args.theme, "dpi": args.dpi,
        "full_production_gui_accepted": False})
    print(f"Rendered {len(screenshots)} screenshots; synthetic={args.synthetic}")


if __name__ == "__main__":
    main()
