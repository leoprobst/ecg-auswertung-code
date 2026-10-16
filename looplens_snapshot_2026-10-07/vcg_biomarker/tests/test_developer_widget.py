import os

import pytest


def test_developer_diagnostics_widget_smoke():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.core.pipeline import Pipeline
    from vcg_biomarker.gui.widgets.developer_widget import DeveloperDiagnosticsWidget

    app = QApplication.instance() or QApplication([])
    config = AppConfig()
    widget = DeveloperDiagnosticsWidget(config, Pipeline(config))
    widget.synthetic_records_spin.setValue(1)
    widget.project_records_spin.setValue(25)
    widget.bootstrap_spin.setValue(1)

    widget.run_diagnostics()
    app.processEvents()

    text = widget.output.toPlainText()
    assert "VCG Biomarker Developer Diagnostics" in text
    assert "Warnings captured:" in text
    assert "Predictions" in text
    assert "Combination sweeps requested/effective" in text
    assert "Keep them separate" in text


def test_developer_prediction_counts_match_combination_skip_logic():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.core.pipeline import Pipeline
    from vcg_biomarker.gui.widgets.developer_widget import DeveloperDiagnosticsWidget

    app = QApplication.instance() or QApplication([])
    config = AppConfig(max_combination_tasks=20)
    widget = DeveloperDiagnosticsWidget(config, Pipeline(config))

    requested, effective, skipped = widget._combination_sweep_counts(
        biomarker_count=5,
        pathology_count=4,
        max_combination=3,
    )

    assert requested == 80
    assert effective == 0
    assert skipped == 80


def test_developer_sweep_estimate_scales_with_record_count():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.core.pipeline import Pipeline
    from vcg_biomarker.gui.widgets.developer_widget import DeveloperDiagnosticsWidget

    app = QApplication.instance() or QApplication([])
    config = AppConfig()
    widget = DeveloperDiagnosticsWidget(config, Pipeline(config))

    estimate = widget._estimated_sweep_seconds_for_records(
        {"n": 600, "estimated_seconds_per_sweep": 2.0},
        target_records=1800,
    )

    assert estimate == pytest.approx(6.0)


def test_main_window_uses_single_visible_analysis_entrypoint():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication, QPushButton, QToolBar

    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(AppConfig())
    try:
        tab_texts = [window.tab_widget.tabText(i) for i in range(window.tab_widget.count())]
        toolbar_analysis_actions = [
            action for toolbar in window.findChildren(QToolBar)
            for action in toolbar.actions()
            if "Analyze Processed" in action.text().replace("&", "")
        ]
        duplicate_analysis_buttons = [
            button for button in window.findChildren(QPushButton)
            if "Analyze Processed" in button.text()
        ]

        assert all("Dev" not in text for text in tab_texts)
        assert "5 - Performance" in tab_texts
        assert "5 - Matrix" not in tab_texts
        assert len(toolbar_analysis_actions) == 1
        assert duplicate_analysis_buttons == []
    finally:
        window.close()
