"""Render the isolated real-pilot UI; outputs never touch production artifacts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import numpy as np

from PyQt6.QtGui import QFont, QFontDatabase, QPalette, QColor
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg

from .storage import atomic_write_json
from ..core.vcg_transform import VCGTransformRegistry


def audit_tab_indices(tabs):
    """Find named views, never silently relabel a screenshot after tab insertion."""
    result = []
    for name in ("results", "lab", "context", "lead_measurements", "ndat", "segment_matrix"):
        found = [i for i in range(tabs.count()) if tabs.widget(i).objectName() == "pilot_" + name]
        if len(found) != 1:
            raise ValueError(f"Audit view missing or ambiguous: {name}")
        result.append((found[0], name))
    return result


def run(workspace: Path, output: Path, theme: str, analyze: bool,
        method: str = "kors", case_index: int = 0, components: bool = False,
        hidden_native: bool = False, require_3d: bool = False):
    if method not in VCGTransformRegistry.available() or case_index < 0:
        raise ValueError("Invalid laboratory audit selection")
    from ..gui.widgets.pilot_workspace_widget import PilotWorkspaceDialog
    app = QApplication.instance() or QApplication([])
    if require_3d and (not analyze or app.platformName() in {"offscreen", "minimal"}):
        raise ValueError("3D audit requires native OpenGL and real laboratory analysis")
    if os.name == "nt" and app.platformName() == "offscreen":
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
        if QFontDatabase.addApplicationFont(str(font_path)) < 0:
            raise RuntimeError("Audit font could not be loaded; screenshots would not prove readability")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")
    palette = QPalette()
    dark = theme == "dark"
    for role, value in ((QPalette.ColorRole.Window, "#17212b" if dark else "#f5f7fa"),
                        (QPalette.ColorRole.Base, "#101820" if dark else "#ffffff"),
                        (QPalette.ColorRole.AlternateBase, "#202d38" if dark else "#e8edf2"),
                        (QPalette.ColorRole.Text, "#edf2f7" if dark else "#17212b"),
                        (QPalette.ColorRole.WindowText, "#edf2f7" if dark else "#17212b"),
                        (QPalette.ColorRole.Button, "#243342" if dark else "#e8edf2"),
                        (QPalette.ColorRole.ButtonText, "#edf2f7" if dark else "#17212b")):
        palette.setColor(role, QColor(value))
    app.setPalette(palette)
    pg.setConfigOptions(background="#101820" if dark else "#ffffff", foreground="#edf2f7" if dark else "#17212b")
    dialog = PilotWorkspaceDialog(workspace)
    if hidden_native:
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    audit_tabs = audit_tab_indices(dialog.tabs)
    dialog.show(); app.processEvents()
    if case_index >= dialog.case.count():
        dialog.close()
        raise ValueError("Requested pilot case is not available in the selected view")
    dialog.case.setCurrentIndex(case_index)
    dialog.method.setCurrentText(method)
    projection_checks = []
    if analyze:
        dialog.tabs.setCurrentIndex(dict((name, i) for i, name in audit_tabs)["lab"])
        dialog.analyze_case()
        deadline = time.monotonic()+180
        while dialog.task is not None and dialog.task.isRunning() and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.03)
        app.processEvents()
        if dialog.task is None or dialog.task.isRunning() or not len(dialog.xyz):
            raise RuntimeError("Real-pilot laboratory analysis did not produce synchronized arrays: " + dialog.lab_status.text())
        if components:
            dialog.coordinates.setCurrentIndex(1)
            app.processEvents()
            if dialog.svd_projection_metadata.get("status") != "available":
                raise RuntimeError("Requested SVD component view is not estimable")
            if dialog.projection_status.height() > 52:
                raise RuntimeError("Projection warning consumes excessive ECG display space")
        dialog.time_slider.setValue(min(500, len(dialog.xyz)-1))
        for plane in range(dialog.plane.count()):
            dialog.plane.setCurrentIndex(plane)
            for segment in range(dialog.segment.count()):
                dialog.segment.setCurrentIndex(segment)
                dialog.draw_loop()
                visible = np.flatnonzero(dialog.loop_visible_mask)
                expected_name = {"P": "p_wave", "QRS": "qrs_complex", "T": "t_wave"}.get(dialog.segment.currentText())
                backend_names = set(dialog.array_data.get("segment_names", []))
                if expected_name in backend_names and not len(visible):
                    raise RuntimeError("Backend provides measured segments but the laboratory hides all samples")
                if len(visible):
                    sample = int(visible[len(visible)//2])
                    dialog.time_slider.setValue(sample)
                    dialog.time_changed(sample)
                    x, y = dialog.loop_cursor.getData()
                    expected = dialog.xyz[sample, list(dialog.plane.currentData())]
                    if not np.allclose([x[0], y[0]], expected, rtol=0, atol=1e-12):
                        raise RuntimeError("2D cursor is not synchronized to the selected projection")
                    if dialog.gl and not np.allclose(dialog.cursor3d.pos, dialog.xyz[sample].reshape(1, 3), rtol=0, atol=1e-6):
                        raise RuntimeError("3D cursor is not synchronized to the backend coordinates")
                hidden = np.flatnonzero(~dialog.loop_visible_mask)
                if len(hidden):
                    dialog.time_changed(int(hidden[0]))
                    x, _ = dialog.loop_cursor.getData()
                    if x is not None and len(x):
                        raise RuntimeError("Cursor shows a point outside the selected segment")
                    if dialog.gl and len(dialog.cursor3d.pos):
                        raise RuntimeError("3D cursor shows a point outside the selected segment")
                projection_checks.append({"plane": dialog.plane.currentText(), "segment": dialog.segment.currentText(),
                                          "visible_samples": len(visible), "cursor_check": "passed"})
        dialog.plane.setCurrentIndex(0); dialog.segment.setCurrentIndex(0)
        dialog.time_slider.setValue(min(500, len(dialog.xyz)-1))
        dialog.time_changed(dialog.time_slider.value())
    output.mkdir(parents=True, exist_ok=True)
    screenshots = []
    framebuffers = []
    for width, height in ((1366, 768), (1440, 900), (1920, 1080)):
        dialog.resize(width, height)
        for tab, name in audit_tabs:
            dialog.tabs.setCurrentIndex(tab)
            app.processEvents()
            for _ in range(4):
                app.processEvents(); time.sleep(.05)
            if name == "context" and dialog.timeline_plot.geometry().bottom() >= dialog.context_table.geometry().top():
                raise RuntimeError("Clinical timeline overlaps the context table")
            if dialog.width() != width or dialog.height() != height:
                raise RuntimeError("Window cannot fit the requested logical resolution")
            if name == "lab" and require_3d:
                if dialog.gl is None or not dialog.loop3d.isValid():
                    raise RuntimeError("Native 3D context is not valid")
                framebuffer = dialog.loop3d.grabFramebuffer()
                if framebuffer.isNull():
                    raise RuntimeError("Native 3D framebuffer is empty")
                gl_path = output/f"{theme}-{width}x{height}-opengl.png"
                if not framebuffer.save(str(gl_path)):
                    raise OSError("3D framebuffer could not be saved")
                framebuffers.append({"path": str(gl_path), "width": framebuffer.width(),
                                     "height": framebuffer.height(), "context_valid": True})
            image = dialog.grab()
            path = output/f"{theme}-{width}x{height}-{name}.png"
            if not image.save(str(path)): raise OSError(f"Screenshot write failed: {path}")
            screenshots.append({"path": str(path), "requested_logical_size": [width, height],
                "actual_logical_size": [dialog.width(), dialog.height()], "pixels": [image.width(), image.height()],
                "device_pixel_ratio": image.devicePixelRatio(), "tab": name})
    atomic_write_json(output/f"{theme}-visual_manifest.json", {"workspace": str(workspace), "real_pilot": True,
        "method": method, "study_id": int(dialog.case.currentData()), "case_index": case_index,
        "coordinate_view": "svd_components_nonanatomical" if components else "anatomical_vcg",
        "laboratory_recalculated": analyze, "synchronized_samples": len(dialog.xyz), "screenshots": screenshots,
        "opengl_3d_available": dialog.gl is not None,
        "opengl_3d_visual_acceptance": "not_verified_by_this_audit",
        "native_framebuffers": framebuffers, "native_hidden_window": hidden_native,
        "qt_platform": app.platformName(),
        "projection_cursor_checks": projection_checks,
        "clinical_claim_allowed": False, "locked_test_read": False})
    dialog.close(); app.processEvents()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--method", choices=VCGTransformRegistry.available(), default="kors")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--hidden-native", action="store_true")
    parser.add_argument("--require-3d", action="store_true")
    args = parser.parse_args()
    run(args.workspace, args.output, args.theme, args.analyze, args.method, args.case_index, args.components,
        args.hidden_native, args.require_3d)
