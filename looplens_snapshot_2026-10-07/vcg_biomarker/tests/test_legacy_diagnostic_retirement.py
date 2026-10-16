import pytest

from vcg_biomarker.core.diagnostic import DiagnosticEngine, LegacyDiagnosticDisabledError
from vcg_biomarker.core.diagnostic import DiagnosticReport, PathologyScore


def historical_report():
    return DiagnosticReport({}, None, None, [PathologyScore('test', 'test', .99)], [], [], [], [])


def test_historical_report_does_not_reissue_diagnostic_claims():
    report = historical_report()
    assert report.pathology_scores[0].probability == .99
    assert report.pathology_scores[0].confidence_label == 'UNVALIDATED_LEGACY'
    assert '99%' not in report.summary
    assert 'deaktiviert' in report.summary
    with pytest.raises(LegacyDiagnosticDisabledError):
        _ = report.top_pathology
    with pytest.raises(LegacyDiagnosticDisabledError):
        _ = report.significant_pathologies


def test_legacy_engine_cannot_emit_pseudo_probability():
    engine = DiagnosticEngine()
    engine._learned_thresholds = {'x': {'disease': {'ppv': .99}}}
    with pytest.raises(LegacyDiagnosticDisabledError):
        engine.diagnose({'x': 100})
    with pytest.raises(LegacyDiagnosticDisabledError):
        engine._compute_pathology_scores({'x': 100}, [], 0)


def test_legacy_gui_cannot_trigger_engine(monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.config import AppConfig
    from vcg_biomarker.gui.widgets.diagnostic_widget import DiagnosticWidget
    app = QApplication.instance() or QApplication([])
    widget = DiagnosticWidget(AppConfig())
    def forbidden(*args, **kwargs):
        raise AssertionError('Retired engine called')
    monkeypatch.setattr(widget.diagnostic_engine, 'diagnose', forbidden)
    widget.set_biomarker_results([])
    widget._run_diagnosis()
    assert widget._current_report is None
    assert not widget.run_diag_btn.isEnabled()
    assert widget.diag_table.rowCount() == 0
    assert widget.diag_table.isHidden() or not widget.diag_table.isVisible()
    widget._current_report = historical_report()
    widget._populate_diagnostic_table()
    assert widget.diag_table.rowCount() == 0
    assert '99%' not in widget.summary_text.toPlainText()
    widget.close()
    app.processEvents()
