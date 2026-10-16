import os

import numpy as np
import pytest

from vcg_biomarker.core.analysis.results import AnalysisRun
from vcg_biomarker.core.analysis.threshold_sweep import threshold_sweep


def test_evidence_widget_smoke():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.gui.widgets.evidence_widget import EvidenceExplorerWidget

    app = QApplication.instance() or QApplication([])

    values = np.concatenate([np.zeros(20), np.ones(20)])
    labels = np.concatenate([np.zeros(20), np.ones(20)])
    result = threshold_sweep(values, labels, "candidate", "target", bootstrap_iterations=10)
    run = AnalysisRun(
        run_id="gui_smoke",
        biomarker_names=["candidate"],
        pathology_names=["target"],
        vcg_method="kors",
        results=[result],
    )
    run.apply_corrections()

    widget = EvidenceExplorerWidget(AppConfig())
    widget.set_analysis_run(run)

    assert widget.evidence_table.rowCount() == 1
    assert widget.workbench_metrics.rowCount() > 0
    app.processEvents()


def test_biomarker_widget_catalog_filters():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.gui.widgets.biomarker_widget import BiomarkerWidget

    app = QApplication.instance() or QApplication([])
    widget = BiomarkerWidget(AppConfig())

    assert widget.summary_table.rowCount() >= 1155

    widget.search_edit.setText("qrs_late_acceleration_max")
    app.processEvents()
    assert widget.summary_table.rowCount() == 1

    widget.search_edit.clear()
    widget.status_combo.setCurrentText("candidate")
    app.processEvents()
    assert widget.summary_table.rowCount() >= 200


def test_biomarker_quality_notes_are_visible_and_searchable():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.core.biomarkers.base import BiomarkerResult
    from vcg_biomarker.gui.widgets.biomarker_widget import BiomarkerWidget, _classify_biomarker
    app = QApplication.instance() or QApplication([])
    widget = BiomarkerWidget(AppConfig())
    note = "absolute_snr_gate=not_validated"
    widget.display_biomarkers([BiomarkerResult("quality_test", None, "", "research", computation_notes=note)])
    widget.search_edit.setText(note)
    app.processEvents()
    assert widget.summary_table.rowCount() == 1
    assert widget.summary_table.item(0, 11).text() == note
    assert _classify_biomarker(float("nan"), (0, 1)) == "unknown"
    widget.close()


def test_evidence_widget_caps_large_tables():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.core.analysis.threshold_sweep import SweepResult
    from vcg_biomarker.gui.widgets.evidence_widget import EvidenceExplorerWidget

    app = QApplication.instance() or QApplication([])
    empty = np.array([])
    results = [
        SweepResult(
            biomarker_name=f"b{i}",
            pathology_name="target",
            n_positive=20,
            n_negative=20,
            n_total=40,
            auc=0.7,
            optimal_threshold=1.0,
            optimal_sensitivity=0.8,
            optimal_specificity=0.7,
            optimal_ppv=0.7,
            optimal_npv=0.8,
            optimal_accuracy=0.75,
            youden_j=0.5,
            roc_thresholds=empty,
            roc_tpr=empty,
            roc_fpr=empty,
            p_value=0.01,
        )
        for i in range(EvidenceExplorerWidget.TABLE_ROW_LIMIT + 1)
    ]
    run = AnalysisRun(
        run_id="large_gui_smoke",
        biomarker_names=[r.biomarker_name for r in results],
        pathology_names=["target"],
        vcg_method="kors",
        results=results,
    )

    widget = EvidenceExplorerWidget(AppConfig())
    widget.set_analysis_run(run)
    app.processEvents()

    assert widget.evidence_table.rowCount() == EvidenceExplorerWidget.TABLE_ROW_LIMIT
    assert "/" in widget.result_count_label.text()


def test_performance_widget_smoke_and_no_baseline_message():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.gui.widgets.performance_widget import PerformanceDashboardWidget

    app = QApplication.instance() or QApplication([])

    values = np.concatenate([np.zeros(20), np.ones(20)])
    labels = np.concatenate([np.zeros(20), np.ones(20)])
    result = threshold_sweep(values, labels, "candidate", "target", bootstrap_iterations=5)
    run = AnalysisRun(
        run_id="performance_smoke",
        biomarker_names=["candidate"],
        pathology_names=["target"],
        vcg_method="kors",
        results=[result],
    )

    widget = PerformanceDashboardWidget(AppConfig())
    widget.set_analysis_run(run)
    app.processEvents()

    assert widget.overview_table.rowCount() == 1
    assert widget.clinical_table.rowCount() == 1
    assert widget.clinical_table.item(0, 0).text() == "Keine klinische Referenzregel konfiguriert"


def test_performance_widget_renders_clinical_baseline():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.core.analysis.clinical_baseline import ClinicalRuleResult
    from vcg_biomarker.gui.widgets.performance_widget import PerformanceDashboardWidget

    app = QApplication.instance() or QApplication([])

    values = np.concatenate([np.zeros(20), np.ones(20)])
    labels = np.concatenate([np.zeros(20), np.ones(20)])
    result = threshold_sweep(values, labels, "candidate", "_AVB", bootstrap_iterations=5)
    run = AnalysisRun(
        run_id="performance_baseline",
        biomarker_names=["candidate"],
        pathology_names=["_AVB"],
        vcg_method="kors",
        results=[result],
        clinical_baseline_results=[
            ClinicalRuleResult(
                rule_id="avb_pr_gt_200",
                pathology_name="_AVB",
                biomarker_name="pr_interval",
                display_name="PR interval",
                threshold_label="> 200",
                source_label="PR reference",
                source_url="https://example.test",
                note="context",
                status="ok",
                n_total=40,
                n_evaluable=40,
                n_positive=20,
                n_negative=20,
                tp=10,
                fp=5,
                tn=15,
                fn=10,
                sensitivity=0.5,
                specificity=0.75,
                ppv=0.66,
                npv=0.6,
                accuracy=0.625,
                balanced_accuracy=0.625,
            )
        ],
    )

    widget = PerformanceDashboardWidget(AppConfig())
    widget.set_analysis_run(run)
    app.processEvents()

    assert widget.clinical_table.rowCount() == 1
    assert widget.clinical_table.item(0, 0).text() == "PR interval"
    assert widget.overview_table.item(0, 3).text() == "PR interval"
