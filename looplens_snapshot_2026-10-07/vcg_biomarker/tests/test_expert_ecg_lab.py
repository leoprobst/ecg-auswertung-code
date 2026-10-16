from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.mimic.lab_workspace import (
    SignalReference, MeasurementAnnotation, LabReviewStore, ReportRevision, resolve_powerline,
)


def test_research_details_are_bounded_and_spatial_preview_is_optional(lab):
    from PyQt6.QtWidgets import QScrollArea

    assert lab.tabs.maximumHeight() == 540
    assert isinstance(lab.tabs.widget(lab.endpoint_review_tab), QScrollArea)
    assert isinstance(lab.tabs.widget(lab.research_tab), QScrollArea)
    assert not isinstance(lab.tabs.widget(0), QScrollArea)
    lab.mode.setCurrentText("Forschung")
    assert not lab.spatial_preview.isChecked()
    lab.spatial_preview.click()
    assert lab.spatial_preview.isChecked()
    lab.mode.setCurrentText("Befundung")
    lab.mode.setCurrentText("Forschung")
    assert lab.spatial_preview.isChecked()  # explicit user preference wins


def test_wpw_worksheet_requires_explicit_clinical_gates(lab):
    from vcg_biomarker.gui.widgets.wpw_dialog import WPWLocalizationDialog

    dialog = WPWLocalizationDialog(lab)
    dialog.evaluate()
    assert "Keine Lokalisation" in dialog.output.toPlainText()
    assert not dialog.manifest.isChecked()


def test_pvc_worksheet_does_not_localize_without_review(lab):
    from vcg_biomarker.gui.widgets.pvc_localization_dialog import PVCLocalizationDialog

    dialog = PVCLocalizationDialog(lab)
    dialog.evaluate()
    assert "Keine eindeutige Region" in dialog.output.toPlainText()
    assert dialog.pvc_v2_s.value() == -1


def test_atrial_focus_worksheet_requires_focal_rhythm_and_p_quality(lab):
    from vcg_biomarker.gui.widgets.atrial_focus_dialog import AtrialFocusDialog

    dialog = AtrialFocusDialog(lab)
    dialog.evaluate()
    assert "Kein regionaler Hinweis" in dialog.output.toPlainText()


@pytest.mark.parametrize('fail_write', [False, True])
def test_beat_annotation_precise_save_and_failure(lab, monkeypatch, fail_write):
    import json
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTableWidgetItem, QInputDialog
    from vcg_biomarker.mimic import storage
    lab.gallery.blockSignals(True)
    lab.gallery.setRowCount(1)
    time = QTableWidgetItem('0.100000')
    time.setData(Qt.ItemDataRole.UserRole, .100123456)
    lab.gallery.setItem(0, 1, time)
    lab.gallery.setItem(0, 4, QTableWidgetItem('unbekannt'))
    lab.gallery.setCurrentCell(0, 1)
    lab.gallery.blockSignals(False)
    lab.author.setText('Reviewer')
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *a, **kw: ('Artefakt', True))
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **kw: ('Signal geprüft', True))
    if fail_write:
        def fail(*args, **kwargs):
            raise OSError('Datenträger nicht beschreibbar')
        monkeypatch.setattr(storage, 'atomic_write_json', fail)
    lab.label_beat()
    files = list((lab.store.root / 'beats').glob('*.json'))
    if fail_write:
        assert not files
        assert lab.gallery.item(0, 4).text() == 'unbekannt'
        assert 'nicht gespeichert' in lab.beat_annotation_note.text()
    else:
        assert len(files) == 1
        saved = json.loads(files[0].read_text(encoding='utf-8'))
        assert saved['r_time_seconds'] == .100123456
        assert saved['native_r_sample'] == lab.reference.sample(.100123456)
        assert saved['author'] == 'Reviewer'
        assert lab.gallery.item(0, 4).text() == 'Artefakt'


def test_device_interval_import_is_signal_bound_and_local(lab, tmp_path, monkeypatch):
    import json
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    doc = dict(schema_version='1.0.0', case_id=lab.reference.case_id,
               source_sha256=lab.reference.source_sha256, device_source='Test report',
               measurements=[dict(interval='QT interval', value_ms=400, unit='ms')])
    source = tmp_path / 'device.json'
    source.write_text(json.dumps(doc), encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(source), ''))
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: warnings.append(a))
    lab.payload = {'delineation': {'interval_measurements': [
        dict(interval='QT interval', value_ms=410, status='measurable')]}}
    lab.import_device_intervals()
    assert 'Test report' in lab.device_interval_note.text()
    assert lab.device_interval_table.rowCount() == 1
    assert lab.device_interval_table.item(0, 3).text() == '10'
    exported = tmp_path / 'device_comparison.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(exported), ''))
    lab.export_device_comparison()
    comparison = json.loads(exported.read_text(encoding='utf-8'))
    assert comparison['rows'][0]['difference_ms'] == 10
    assert comparison['clinical_claim_allowed'] is False
    assert comparison['signal_reference']['source_sha256'] == lab.reference.source_sha256
    import csv
    csv_target = tmp_path / 'device_comparison.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(csv_target), ''))
    lab.export_device_comparison()
    with csv_target.open(encoding='utf-8-sig', newline='') as stream:
        exported_rows = list(csv.DictReader(stream))
    assert exported_rows[0]['difference_ms'] == '10.0'
    assert exported_rows[0]['unit'] == 'ms'
    assert exported_rows[0]['source_sha256'] == lab.reference.source_sha256
    before_export = csv_target.read_bytes()
    lab.blind.setChecked(True)
    lab.export_device_comparison()
    assert csv_target.read_bytes() == before_export
    lab.blind.setChecked(False)
    assert len(list((lab.store.root / 'device_intervals').glob('*.json'))) == 1
    saved = next((lab.store.root / 'device_intervals').glob('*.json'))
    lab.device_interval_document = None
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(saved), ''))
    lab.reopen_device_intervals()
    assert lab.device_interval_document == doc
    assert len(list((lab.store.root / 'device_intervals').glob('*.json'))) == 1
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(source), ''))
    lab.reopen_device_intervals()
    assert warnings
    warnings.clear()
    doc['source_sha256'] = 'wrong-signal'
    source.write_text(json.dumps(doc), encoding='utf-8')
    lab.import_device_intervals()
    assert warnings and lab.device_interval_document['source_sha256'] == lab.reference.source_sha256
    assert len(list((lab.store.root / 'device_intervals').glob('*.json'))) == 1
    lab.blind.setChecked(True)
    warnings.clear()
    lab.import_device_intervals()
    assert not warnings
    lab.blind.setChecked(False)
    next_case = SimpleNamespace(case_id='case-next', source_sha256='next-hash',
                                pseudonym='Next', metadata={})
    library = SimpleNamespace(root=tmp_path, load_case=lambda _: next_case,
        load_signal=lambda _: (lab.signal, lab.reference.sampling_rate_hz),
        list_cases=lambda: [next_case])
    lab.load_case(library, next_case.case_id)
    assert lab.device_interval_document is None
    assert lab.device_interval_table.rowCount() == 0
    assert 'Kein fallgebundener' in lab.device_interval_note.text()


def test_native_measurement_has_no_screen_dependency():
    reference = SignalReference("case-test", "abc", 250, 1000)
    item = MeasurementAnnotation(reference, "II", 10, 35, -.1, .9)
    assert item.duration_ms == 100
    assert item.delta_mv == 1
    assert reference.sample(.101) == 25
    with pytest.raises(ValueError):
        reference.sample(4)


@pytest.mark.parametrize('suffix', ['json', 'csv'])
def test_automatic_interval_export_actual_gui_path(lab, tmp_path, monkeypatch, suffix):
    import json
    import csv
    from PyQt6.QtWidgets import QFileDialog
    target = tmp_path / ('intervals.' + suffix)
    lab.payload = {'analysis_id': 'export-test', 'delineation': {'interval_measurements': [
        {'interval': 'QT interval', 'value_ms': 421.123456789, 'status': 'measurable',
         'measurement_method': 'representative_beat_fiducials', 'start_sample': 100, 'end_sample': 300},
        {'interval': 'QTm', 'value_ms': None, 'status': 'not_evaluable', 'reason': 'no_lbbb'}]}}
    lab.manual_overrides = {'QT interval': {'manual_value': '450 ms'}}
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(target), ''))
    lab.export_automatic_intervals()
    if suffix == 'json':
        document = json.loads(target.read_text(encoding='utf-8'))
        assert document['manual_overrides_applied'] is False
        assert document['signal_reference']['source_sha256'] == lab.reference.source_sha256
        rows = document['rows']
    else:
        with target.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]['analysis_id'] == 'export-test'
        assert rows[0]['signal_sha256'] == lab.reference.source_sha256
    assert float(rows[0]['value_ms']) == 421.123456789
    assert rows[1]['reason'] == 'no_lbbb'
    assert all(r['review_status'] == 'automatic_not_clinically_reviewed' for r in rows)
    original = target.read_bytes()
    lab.blind.setChecked(True)
    lab.export_automatic_intervals()
    assert target.read_bytes() == original


def test_qt_review_acknowledgement_is_local_and_blinded(lab, monkeypatch):
    import json
    from PyQt6.QtWidgets import QInputDialog
    lab.payload = {'analysis_id': 'qt-test', 'delineation': {'interval_measurements': [
        {'interval': 'QTc Fridericia', 'value_ms': 510, 'status': 'measurable'}]}}
    lab._refresh_qt_alert()
    assert not lab.qt_alert_note.isHidden()
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **k: ('Messung überprüft', True))
    lab.acknowledge_qt_alert()
    files = list((lab.store.root/'alert_reviews').glob('*.json'))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding='utf-8'))
    assert saved['analysis_id'] == 'qt-test' and saved['confirms_treatment'] is False
    assert saved['items'][0]['value_ms'] == 510
    from PyQt6.QtWidgets import QDialog, QTextEdit
    history = []
    monkeypatch.setattr(QDialog, 'exec', lambda dialog: history.append(dialog.findChild(QTextEdit).toPlainText()))
    lab.show_qt_review_history()
    assert len(history) == 1
    assert 'Messung überprüft' in history[0]
    assert 'ersetzen keine klinische Abklärung' in history[0]
    lab.blind.setChecked(True)
    assert lab.qt_alert_note.isHidden()
    assert lab.qt_history_button.isHidden()
    lab.show_qt_review_history()
    assert len(history) == 1
    lab.acknowledge_qt_alert()
    assert len(list((lab.store.root/'alert_reviews').glob('*.json'))) == 1


def test_rule_search_uses_recorded_definition_and_clears_hidden_selection(lab):
    from PyQt6.QtWidgets import QTableWidgetItem
    lab.payload = {'classical_criteria': {'one': {'definition': 'Cornell Voltage', 'source_id': 'old'},
                                         'two': {'definition': 'QT interval'}}}
    lab.suggestions.setRowCount(2)
    lab.suggestions.setItem(0, 0, QTableWidgetItem('one'))
    lab.suggestions.setItem(1, 0, QTableWidgetItem('two'))
    lab.suggestions.setCurrentCell(0, 0)
    lab.rule_search.setText('qt')
    assert lab.suggestions.isRowHidden(0)
    assert not lab.suggestions.isRowHidden(1)
    assert lab.suggestions.currentRow() == -1
    lab.rule_search.setText('cornell OLD')
    assert not lab.suggestions.isRowHidden(0)
    assert lab.suggestions.isRowHidden(1)
    lab.rule_search.clear()
    assert not any(lab.suggestions.isRowHidden(i) for i in range(2))


def test_catalog_source_registry_and_missing_case_do_not_create_result(lab):
    catalog = lab.criteria_catalog
    assert len(catalog.entries) > 20
    catalog.table.setCurrentCell(0, 0)
    assert 'Definition:' in catalog.details.toPlainText()
    before = dict(lab.payload)
    lab.open_catalog_case_criterion('not_recorded')
    assert 'ersetzt kein' in catalog.status.text()
    assert lab.payload == before
    catalog.search.setText('surely_no_definition_matches_937')
    assert all(catalog.table.isRowHidden(i) for i in range(catalog.table.rowCount()))
    assert not catalog.open_case.isEnabled()
    lab.blind.setChecked(True)
    assert not lab.tabs.isTabVisible(lab.criteria_catalog_tab)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_bad_sampling_rate_is_rejected(rate):
    with pytest.raises(ValueError):
        SignalReference("case-test", "abc", rate, 10)


@pytest.mark.parametrize("choice,metadata,expected", [
    ("metadata", {}, 0), ("metadata", {"country": "Austria"}, 0),
    ("metadata", {"powerline_hz": 60}, 60),
    ("metadata", {"source_metadata": {"powerline_hz": 50}}, 50),
    ("off", {"powerline_hz": 60}, 0), ("50", {}, 50), ("60", {}, 60),
])
def test_powerline_no_country_guess(choice, metadata, expected):
    assert resolve_powerline(choice, metadata) == expected


def test_report_review_completion_is_immutable(tmp_path):
    store = LabReviewStore(tmp_path, "case-test")
    draft = ReportRevision("case-test", "abc", "reviewer", sections={"Zusammenfassung": "Manuell geprüft"})
    store.save_report(draft)
    reviewed = replace(draft, status="reviewed", revision_id="reviewed", parent_revision_id=draft.revision_id)
    store.save_report(reviewed)
    with pytest.raises(ValueError, match="Changed content"):
        store.save_report(replace(reviewed, status="completed", revision_id="bad", parent_revision_id="reviewed", sections={"Zusammenfassung": "Geändert"}))
    completed = replace(reviewed, status="completed", revision_id="completed", parent_revision_id="reviewed")
    store.save_report(completed)
    with pytest.raises(ValueError):
        store.save_report(replace(completed, revision_id="bad2", parent_revision_id="completed"))
    store.save_report(replace(completed, status="draft", revision_id="reopen", parent_revision_id="completed"))
    assert len(store.reports()) == 4


def test_review_store_rejects_path_escape(tmp_path):
    for name in ("../case-other", "case-../x", "case-C:evil"):
        with pytest.raises(ValueError):
            LabReviewStore(tmp_path, name)


def test_retracted_suggestion_blocks_review_until_text_bound_reconciliation(tmp_path):
    from vcg_biomarker.mimic.lab_workspace import report_text_digest
    store = LabReviewStore(tmp_path, 'case-test')
    sections = {'Zusammenfassung': 'Manuell korrigierter Befund'}
    review = {'rule': {'decision': 'rejected', 'requires_text_reconciliation': True}}
    draft = ReportRevision('case-test', 'abc', 'reviewer', sections=sections, suggestion_review=review)
    store.save_report(draft)
    candidate = replace(draft, status='reviewed', parent_revision_id=draft.revision_id, revision_id='reviewed')
    with pytest.raises(ValueError, match='Unresolved'):
        store.save_report(candidate)
    review['rule']['text_reconciliation'] = {'sections_sha256': report_text_digest(sections),
        'author': 'reviewer', 'reason': 'Alte Aussage entfernt', 'reviewed_at': draft.created_at}
    store.save_report(candidate)
    changed = replace(candidate, revision_id='changed', parent_revision_id='reviewed',
                      sections={'Zusammenfassung': 'Wieder geänderter Text'})
    with pytest.raises(ValueError, match='Unresolved'):
        store.save_report(changed)


def test_old_analysis_acceptance_needs_new_explicit_review(tmp_path):
    from vcg_biomarker.mimic.lab_workspace import report_text_digest, unresolved_review_conflicts
    store = LabReviewStore(tmp_path, 'case-test')
    sections = {'Zusammenfassung': 'Manuell geprüft'}
    reviews = {'rule': {'decision': 'accepted', 'analysis_id': 'old'}}
    candidate = ReportRevision('case-test', 'abc', 'reviewer', status='reviewed',
                              sections=sections, suggestion_review=reviews, analysis_id='new')
    with pytest.raises(ValueError, match='Unresolved'):
        store.save_report(candidate)
    reviews['rule']['text_reconciliation'] = {'sections_sha256': report_text_digest(sections),
        'author': 'reviewer', 'reason': 'Mit aktueller Berechnung abgeglichen',
        'reviewed_at': candidate.created_at, 'analysis_id': 'new'}
    store.save_report(candidate)
    assert reviews['rule']['analysis_id'] == 'old'  # original provenance stays intact
    assert unresolved_review_conflicts(reviews, sections, 'new') == []
    assert unresolved_review_conflicts(reviews, sections, 'third') == ['rule']
    assert unresolved_review_conflicts(reviews, sections, '') == ['rule']


def test_gui_reconciles_historical_acceptance_without_relabeling_origin(lab, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    from vcg_biomarker.mimic.lab_workspace import unresolved_review_conflicts
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('Neu geprüft', True))
    lab.suggestion_review = {'rule': {'decision': 'accepted', 'analysis_id': 'old'}}
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'new'})
    lab.reconcile_suggestion_text()
    sections = {name: edit.toPlainText() for name, edit in lab.sections.items()}
    assert unresolved_review_conflicts(lab.suggestion_review, sections, 'new') == []
    assert lab.suggestion_review['rule']['analysis_id'] == 'old'


def test_old_override_requires_current_analysis_reconciliation(lab, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    from vcg_biomarker.mimic.lab_workspace import unresolved_override_conflicts
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('Messwert kontrolliert', True))
    lab.manual_overrides = {'QT': {'analysis_id': 'old', 'manual_value': '400 ms', 'reason': 'Manuell'}}
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'new'})
    lab.sections['Zusammenfassung'].setPlainText('Geprüft')
    sections = {name: edit.toPlainText() for name, edit in lab.sections.items()}
    candidate = ReportRevision(lab.case.case_id, lab.case.source_sha256, 'reviewer', status='reviewed',
        sections=sections, analysis_id='new', manual_overrides=lab.manual_overrides)
    with pytest.raises(ValueError, match='Unresolved manual'):
        lab.store.save_report(candidate)
    lab.reconcile_suggestion_text()
    assert unresolved_override_conflicts(lab.manual_overrides, sections, 'new') == []
    assert lab.manual_overrides['QT']['analysis_id'] == 'old'
    lab.store.save_report(candidate)


def test_retraction_records_history_and_requires_explicit_text_review(lab, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox, QInputDialog
    from vcg_biomarker.mimic.lab_workspace import unresolved_review_conflicts
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: None)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('Text abgeglichen', True))
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'a', 'classical_criteria': {
        'rule': {'exploratory': False, 'evaluable': True, 'positive': True, 'value': 1, 'definition': 'Test'}}})
    lab.suggestions.selectRow(0)
    lab.review_suggestion('accepted')
    lab.review_suggestion('rejected')
    def sections():
        return {name: edit.toPlainText() for name, edit in lab.sections.items()}
    assert lab.suggestion_review['rule']['history'][0]['decision'] == 'accepted'
    assert unresolved_review_conflicts(lab.suggestion_review, sections()) == ['rule']
    lab.reconcile_suggestion_text()
    assert unresolved_review_conflicts(lab.suggestion_review, sections()) == []
    lab.sections['Zusammenfassung'].append('Weitere Änderung')
    assert unresolved_review_conflicts(lab.suggestion_review, sections()) == ['rule']


@pytest.fixture
def app():
    from PyQt6.QtWidgets import QApplication
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def lab(app, tmp_path):
    from vcg_biomarker.gui.widgets.expert_ecg_lab import ExpertECGLab
    signal = np.tile(np.sin(np.arange(2500)*2*np.pi/250)[:, None], (1, 12))
    case = SimpleNamespace(case_id="case-test", source_sha256="abc", pseudonym="Synthetic", metadata={})
    library = SimpleNamespace(root=tmp_path, load_case=lambda _: case, load_signal=lambda _: (signal, 250.), list_cases=lambda: [case])
    widget = ExpertECGLab()
    widget.resize(1200, 900)
    widget.load_case(library, case.case_id)
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('fail_signal', [False, True])
def test_same_case_id_other_workspace_not_silently_ignored(lab, tmp_path, fail_signal):
    previous_signal, previous_reference = lab.signal, lab.reference
    case = SimpleNamespace(case_id=lab.case.case_id, source_sha256='different-source',
                           pseudonym='Different recording', metadata={})
    signal = np.zeros((1200, 12))
    def load_signal(_):
        if fail_signal:
            raise OSError('Signal unavailable')
        return signal, 300.
    library = SimpleNamespace(root=tmp_path / 'other_workspace', load_case=lambda _: case,
                              load_signal=load_signal, list_cases=lambda: [case])
    if fail_signal:
        with pytest.raises(OSError):
            lab.load_case(library, case.case_id)
        assert lab.reference == previous_reference
        assert lab.signal is previous_signal
    else:
        lab.load_case(library, case.case_id)
        assert lab.reference.source_sha256 == 'different-source'
        assert lab.reference.sampling_rate_hz == 300
        assert lab.signal is signal
        assert lab.library is library


def test_layouts_calipers_and_landmark_undo(lab):
    lab.add_caliper()
    before = lab.measurement_rows()
    assert len(before) == 1
    for layout in range(4):
        lab.layout_choice.setCurrentIndex(layout)
        assert lab.measurement_rows()[0]["duration_ms"] == before[0]["duration_ms"]
    lab.landmark_time.setValue(.1)
    lab.set_landmark()
    assert lab.fiducials["p_onset"] == 25
    lab.undo()
    assert "p_onset" not in lab.fiducials
    lab.redo()
    assert lab.fiducials["p_onset"] == 25


def test_phrase_search_is_substring_case_insensitive_and_never_auto_inserts(lab):
    from PyQt6.QtCore import Qt
    completer = lab.phrase.completer()
    assert completer.filterMode() == Qt.MatchFlag.MatchContains
    assert completer.caseSensitivity() == Qt.CaseSensitivity.CaseInsensitive
    completer.setCompletionPrefix('RHYTHMUS')
    assert completer.completionCount() == 1
    lab.phrase.setEditText('Rhythmus')
    assert all(not edit.toPlainText() for edit in lab.sections.values())


def test_manual_phrase_preserves_literal_text_and_single_undo(lab):
    name = lab.section_choice.currentText()
    edit = lab.sections[name]
    edit.setPlainText('Vorhandener Befund')
    lab.phrase.setEditText(' <b>Manuell</b> & QT < 400 ms ')
    lab.insert_phrase()
    assert edit.toPlainText() == 'Vorhandener Befund\n<b>Manuell</b> & QT < 400 ms'
    assert all(not other.toPlainText() for key, other in lab.sections.items() if key != name)
    edit.undo()
    assert edit.toPlainText() == 'Vorhandener Befund'
    lab.phrase.setEditText('  ')
    lab.insert_phrase()
    assert edit.toPlainText() == 'Vorhandener Befund'


def test_user_phrase_library_requires_confirmation_and_is_section_scoped(lab, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from PyQt6.QtCore import QSettings
    from vcg_biomarker.gui.widgets import expert_ecg_lab as module
    stored = {}
    settings = SimpleNamespace(value=lambda key: stored.get(key),
        setValue=lambda key, value: stored.update({key: value}),
        sync=lambda: None, status=lambda: QSettings.Status.NoError)
    def factory(*_args):
        return settings
    factory.Status = QSettings.Status
    monkeypatch.setattr(module, 'QSettings', factory)
    lab.phrase.setEditText('Eigene manuell zu prüfende Beobachtung.')
    before = {k: v.toPlainText() for k, v in lab.sections.items()}
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.No)
    lab.save_user_phrase()
    assert not stored
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    lab.save_user_phrase()
    assert lab.phrase.count() == 7
    lab.section_choice.setCurrentIndex(1)
    assert lab.phrase.count() == 6
    lab.section_choice.setCurrentIndex(0)
    assert lab.phrase.count() == 7
    assert {k: v.toPlainText() for k, v in lab.sections.items()} == before
    raw = stored['library_v1']
    stored['library_v1'] = '{bad'
    lab.refresh_user_phrases()
    assert not lab.phrase_save.isEnabled()
    assert stored['library_v1'] == '{bad'
    stored['library_v1'] = raw
    from PyQt6.QtWidgets import QInputDialog
    choices = iter([('Eigene manuell zu prüfende Beobachtung.', True), ('Bearbeiten', True)])
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: next(choices))
    monkeypatch.setattr(QInputDialog, 'getMultiLineText', lambda *args: ('Neue Vorlage', True))
    lab.manage_user_phrase()
    assert 'Neue Vorlage' in stored['library_v1']
    choices = iter([('Neue Vorlage', True), ('Entfernen', True)])
    lab.manage_user_phrase()
    from vcg_biomarker.gui.lab_phrase_library import decode_phrases
    assert decode_phrases(stored['library_v1']) == []
    assert {k: v.toPlainText() for k, v in lab.sections.items()} == before


def test_recorded_lead_matrix_links_to_native_lead_and_resets(lab):
    lab.set_analysis({'case_id': 'case-test', 'lead_measurements': {'lead_measurement_aVR_st_j_mv': -.15}})
    lab.avr_polarity.setCurrentIndex(1)
    assert lab.lead_measurement_table.item(3, 5).text() == '-0.15'
    lab.lead_measurement_table.cellClicked.emit(3, 5)
    assert lab.lead_choice.currentText() == 'aVR'
    assert lab.lead_measurement_table.item(0, 5).text() == 'nicht verfügbar'
    lab.set_analysis({'case_id': 'case-test'})
    assert lab.lead_measurement_table.item(3, 5).text() == 'nicht verfügbar'
    lab.set_analysis({'case_id': 'case-test', 'lead_measurements': {'lead_measurement_aVR_st_j_mv': -.15}})
    lab.clear_analysis()
    assert lab.lead_measurement_table.item(3, 5).text() == 'nicht verfügbar'


@pytest.mark.parametrize('suffix', ['json', 'csv'])
def test_lead_matrix_export_keeps_native_polarity_and_missingness(lab, tmp_path, monkeypatch, suffix):
    import csv
    import json
    from PyQt6.QtWidgets import QFileDialog
    path = tmp_path/f'matrix.{suffix}'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'matrix-source',
        'lead_measurements': {'lead_measurement_aVR_st_j_mv': -.123456789}})
    lab.avr_polarity.setCurrentIndex(1)
    lab.export_lead_matrix()
    if suffix == 'json':
        document = json.loads(path.read_text(encoding='utf-8'))
        assert document['analysis_id'] == 'matrix-source'
        assert document['signal_reference']['source_sha256'] == 'abc'
        rows = document['rows']
    else:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert all(row['analysis_id'] == 'matrix-source' for row in rows)
        assert all(row['source_sha256'] == 'abc' for row in rows)
    assert len(rows) == 156
    item = next(row for row in rows if row['source_key'] == 'lead_measurement_aVR_st_j_mv')
    assert float(item['value']) == -.123456789
    missing = next(row for row in rows if row['source_key'] == 'lead_measurement_I_st_j_mv')
    assert missing['value'] in ('', None)
    assert missing['status'] == 'unavailable'


def test_auto_analysis_native_coordinate_mapping_and_blinding(lab):
    lab.set_analysis({"case_id": "case-test", "analysis_id": "aaa", "delineation": {"representative_fiducials": {"r_peak": 500}, "interval_measurements": [{"interval": "QRS", "value_ms": 100, "status": "measurable"}]}, "feature_values": {"missing_marker": None}})
    assert lab.fiducials["r_peak"] == 250
    assert lab.measurements.rowCount() == 3
    assert lab.measurements.item(0, 0).text() == 'QRS'
    assert lab.measurements.item(1, 0).text() == 'P-Dauer · weitere akzeptierte Schläge'
    assert lab.measurements.item(1, 1).text() == 'nicht auswertbar'
    assert lab.measurements.item(2, 0).text() == 'Frontale QRS-Achseneinordnung'
    assert lab.measurements.item(2, 1).text() == 'QRS-Achse nicht beurteilbar.'
    lab.mode.setCurrentIndex(1)
    assert lab.tabs.isTabVisible(lab.research_tab)
    lab.blind.setChecked(True)
    assert not lab.tabs.isTabVisible(lab.research_tab)
    assert not lab.tabs.isTabVisible(1)
    assert not lab._landmark_items


@pytest.mark.parametrize('rate,peaks', [(0, [1, 2]), (float('nan'), [1, 2]), (500, [2, 1]), (500, [1.5])])
def test_invalid_visualization_clears_previous_arrays(lab, tmp_path, rate, peaks):
    path = tmp_path / 'arrays.npz'
    np.savez(path, sampling_rate_hz=rate, r_peaks=np.asarray(peaks))
    lab.analysis_arrays = {'old': np.ones(2)}
    lab.gallery.setRowCount(1)
    lab.transform_canvas.addPlot().plot([0, 1], [0, 1])
    lab.beat_plot.plot([0, 1], [0, 1])
    with pytest.raises(ValueError):
        lab.set_analysis_arrays(path)
    assert lab.analysis_arrays is None
    assert lab.gallery.rowCount() == 0
    assert not lab.transform_canvas.ci.items
    assert not lab.beat_plot.listDataItems()
    assert 'Keine kompatiblen' in lab.transform_note.text()


def test_rejected_late_result_leaves_signal_views_unchanged(lab, monkeypatch):
    calls = []
    monkeypatch.setattr(lab.scientific_displays, 'set_data', lambda *a: calls.append('reset'))
    previous = lab.payload
    assert lab.set_analysis({'case_id': 'other'}) is False
    assert lab.payload is previous and not calls
    with pytest.raises(ValueError, match='Originalsignal'):
        lab.set_analysis({'case_id': lab.case.case_id, 'signal_reference': {'source_sha256': 'other'}})
    assert lab.payload is previous and not calls


def test_provenance_tree_keeps_roots_when_beat_history_is_large(lab):
    tree = lab.marker_provenance_tree
    detail = {'analysis_id': 'case-analysis', 'measurement_provenance': {'beats': list(range(10000))},
              'current_registry_reference_not_historical_proof': {'unit': 'mV'}}
    tree.set_evidence(detail)
    assert tree.topLevelItemCount() == 3
    assert tree.topLevelItem(2).child(0).text(1) == 'mV'
    def count(item):
        return 1 + sum(count(item.child(i)) for i in range(item.childCount()))
    assert sum(count(tree.topLevelItem(i)) for i in range(3)) <= 400
    assert len(detail['measurement_provenance']['beats']) == 10000
    tree.set_evidence({})
    assert tree.topLevelItem(1).text(1) == 'nicht dokumentiert'


def test_marker_unknown_definition_does_not_crash(lab):
    lab.set_analysis({"case_id": "case-test", "feature_values": {"missing_marker": None}})
    lab.markers.selectRow(0)
    tree = lab.marker_provenance_tree
    assert tree.topLevelItemCount() == 3
    assert 'kein historischer Nachweis' in tree.topLevelItem(2).text(0)
    assert 'Historisches Artefakt' in tree.topLevelItem(1).child(0).text(1)
    lab.inspect_marker()
    assert "nicht vorhanden" in lab.marker_detail.toPlainText()


def test_missing_patient_identity_does_not_offer_comparison(lab):
    assert lab.comparison_id.count() == 0


def test_fiducial_evidence_keeps_recorded_time_and_is_read_only(lab):
    payload = {'case_id':'case-test', 'delineation': {'fiducial_details': {
        't_offset': {'time_seconds': .75, 'confidence': .8, 'source': 'recorded',
                     'conflict_ms': 6., 'uncertainty_ms': 4., 'support_count': 2}}}}
    lab.set_analysis(payload)
    before = dict(lab.fiducials)
    lab.mode.setCurrentIndex(1)
    assert lab.tabs.isTabVisible(lab.fiducial_evidence_tab)
    lab._inspect_fiducial_evidence(0, 0)
    assert lab.landmark_time.value() == .75
    assert lab.fiducials == before
    assert lab.fiducial_evidence.item(0, 4).text() == '6.0'
    lab.blind.setChecked(True)
    assert not lab.tabs.isTabVisible(lab.fiducial_evidence_tab)
    lab.clear_analysis()
    assert lab.fiducial_evidence.rowCount() == 0


def test_original_method_candidates_are_overlaid_without_editing(lab):
    import pyqtgraph as pg
    payload = {'case_id':'case-test', 'delineation': {
        'fiducial_details': {'t_offset': {'time_seconds': .8}},
        'method_candidates': {'schema_version':'1.0.0', 'items': [
            {'method':'neurokit', 'landmark':'t_offset', 'time_seconds':.78, 'temporally_admissible':True},
            {'method':'multilead', 'landmark':'t_offset', 'time_seconds':.82, 'temporally_admissible':True}]}}}
    lab.set_analysis(payload)
    original = dict(lab.fiducials)
    lab._inspect_fiducial_evidence(0,0)
    lines = [i for i in lab.fiducial_overlay.items() if isinstance(i, pg.InfiniteLine)]
    assert sorted(i.value() for i in lines) == [.78,.8,.82]
    assert all(not i.movable for i in lines)
    assert lab.fiducials == original
    lab.clear_analysis()
    assert not lab.fiducial_overlay.listDataItems()


def test_review_requires_explicit_evaluability_and_is_idempotent(lab, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: None)
    payload = {'case_id': 'case-test', 'analysis_id': 'review-test', 'classical_criteria': {
        'rule': {'exploratory': False, 'evaluable': 'unknown', 'positive': False,
                 'value': 0, 'unit': 'ms', 'definition': 'Test rule'}}}
    lab.set_analysis(payload)
    lab.suggestions.selectRow(0)
    lab.review_suggestion('accepted')
    assert not lab.suggestion_review
    payload['classical_criteria']['rule']['evaluable'] = True
    lab.set_analysis(payload)
    lab.suggestions.selectRow(0)
    lab.blind.setChecked(True)
    lab.review_suggestion('accepted')
    assert not lab.suggestion_review
    lab.blind.setChecked(False)
    lab.review_suggestion('accepted')
    text = lab.sections[lab.section_choice.currentText()].toPlainText()
    lab.review_suggestion('accepted')
    assert lab.sections[lab.section_choice.currentText()].toPlainText() == text
    assert lab.suggestion_review['rule']['reviewed_at']
    from vcg_biomarker.mimic.storage import content_hash
    evidence = lab.suggestion_review['rule']['evidence_snapshot']
    assert evidence['recorded_rule']['definition'] == 'Test rule'
    lab.inspect_suggestion()
    assert 'evidence_sha256' in lab.suggestion_detail.toPlainText()
    assert evidence['signal_reference']['source_sha256'] == lab.reference.source_sha256
    assert content_hash(evidence) == lab.suggestion_review['rule']['evidence_sha256']
    payload['classical_criteria']['rule']['definition'] = 'Later changed definition'
    assert evidence['recorded_rule']['definition'] == 'Test rule'
    payload['classical_criteria']['rule']['definition'] = 'Test rule'
    payload['classical_criteria']['rule']['source_id'] = 'updated-source'
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: None)
    lab.review_suggestion('accepted')
    assert lab.suggestion_review['rule']['evidence_snapshot']['recorded_rule']['source_id'] == 'updated-source'
    assert lab.suggestion_review['rule']['history']
    assert lab.suggestion_review['rule']['requires_text_reconciliation']
    assert lab.save_report('draft')
    stored = lab.store.reports()[-1]['suggestion_review']['rule']
    assert stored['evidence_snapshot']['recorded_rule']['source_id'] == 'updated-source'
    assert content_hash(stored['evidence_snapshot']) == stored['evidence_sha256']
    assert stored['history'][0]['evidence_snapshot']['recorded_rule']['definition'] == 'Test rule'


def test_invalid_marker_preferences_are_preserved_and_write_locked(lab, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: None)
    from vcg_biomarker.mimic.storage import atomic_write_json
    paths = [lab.store.root / 'marker_groups.json', lab.store.root / 'marker_favorites.json']
    for path in paths:
        atomic_write_json(path, ['invalid top-level structure'])
    before = [p.read_bytes() for p in paths]
    lab.case = None
    lab.load_case(lab.library, 'case-test')
    assert lab._marker_groups_read_error
    assert lab._marker_favorites_read_error
    lab._write_marker_groups({'new': ['x']})
    lab.toggle_marker_favorites()
    assert [p.read_bytes() for p in paths] == before


def test_lbbb_confirmation_is_explicit_case_bound_and_reset(lab, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    assert lab.next_ischemia_context() is None
    lab.lbbb_context.setChecked(True)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('', False))
    with pytest.raises(ValueError):
        lab.next_ischemia_context()
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('Manuell geprüfte Morphologie', True))
    context = lab.next_ischemia_context()
    assert context['case_id'] == 'case-test'
    assert context['author'] == lab.author.text()
    assert context['revision_id']
    # Reopen the case; selecting the already loaded case is intentionally a no-op.
    lab.case = None
    lab.load_case(lab.library, 'case-test')
    assert lab.next_ischemia_context() is None


def test_local_marker_comparison_group_survives_reopen(lab, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    payload = {'case_id': 'case-test', 'feature_values': {'a': 1., 'b': 2.}}
    lab.set_analysis(payload)
    lab.markers.selectRow(0)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('Review-Gruppe', True))
    lab.save_marker_group()
    assert lab.marker_groups == {'Review-Gruppe': ['a']}
    assert not lab.markers.isRowHidden(0)
    assert lab.markers.isRowHidden(1)
    lab.case = None
    lab.marker_groups = {}
    lab.load_case(lab.library, 'case-test')
    lab.set_analysis(payload)
    lab.marker_group.setCurrentIndex(lab.marker_group.findData('Review-Gruppe'))
    assert lab.markers.isRowHidden(1)
    lab.remove_marker_group()
    assert not lab.markers.isRowHidden(1)
    assert payload['feature_values'] == {'a': 1., 'b': 2.}


def test_cabrera_inverts_display_not_native_measurements(lab):
    original = lab.signal.copy()
    lab.layout_choice.setCurrentIndex(2)
    lab.lead_order.setCurrentIndex(1)
    assert lab.canvas.height() >= 1200
    assert [item[1] for item in lab.plots] == [4, 0, 3, 1, 5, 2, 6, 7, 8, 9, 10, 11]
    plot = lab.plots[2][0]
    _, displayed = plot.listDataItems()[0].getData()
    np.testing.assert_allclose(displayed, -original[:len(displayed), 3])
    lab.lead_choice.setCurrentText('aVR')
    lab.add_caliper()
    native = lab.measurement_rows()[0]
    assert native['lead'] == 'aVR'
    assert native['delta_mv'] == .5
    assert lab.calipers[0][1][3].value() == -.5
    lab.lead_order.setCurrentIndex(0)
    restored = lab.measurement_rows()[0]
    assert restored['measurement_id'] == native['measurement_id']
    assert restored['delta_mv'] == native['delta_mv']
    assert lab.calipers[0][1][3].value() == .5
    np.testing.assert_array_equal(lab.signal, original)


@pytest.mark.parametrize('layout', [0, 1, 2, 3])
def test_avr_toggle_in_all_layouts_preserves_other_leads_and_signal(lab, layout):
    original = lab.signal.copy()
    lab.lead_choice.setCurrentText('aVR')
    lab.layout_choice.setCurrentIndex(layout)
    lab.avr_polarity.setCurrentIndex(1)
    found = 0
    for plot, lead, lo, hi in lab.plots:
        x, y = plot.listDataItems()[0].getData()
        indices = np.rint(x * lab.reference.sampling_rate_hz).astype(int)
        np.testing.assert_allclose(y, original[indices, lead] * (-1 if lead == 3 else 1))
        if lead == 3:
            found += 1
            assert '−aVR' in plot.titleLabel.text
    assert found
    lab.avr_polarity.setCurrentIndex(0)
    np.testing.assert_array_equal(lab.signal, original)


@pytest.mark.parametrize('layout, expected', [
    (0, [4, 1, 6, 9, 0, 5, 7, 10, 3, 2, 8, 11]),
    (1, [4, 6, 0, 7, 3, 8, 1, 9, 5, 10, 2, 11]),
    (2, [4, 0, 3, 1, 5, 2, 6, 7, 8, 9, 10, 11]),
])
def test_cabrera_order_independent_of_layout_and_restores_manual_polarity(lab, layout, expected):
    lab.layout_choice.setCurrentIndex(layout)
    lab.lead_order.setCurrentIndex(0)  # gespeicherter Ansichtsstandard darf den Ausgangszustand nicht vorgeben
    lab.avr_polarity.setCurrentIndex(1)
    lab.lead_order.setCurrentIndex(1)
    assert [item[1] for item in lab.plots[:12]] == expected
    assert not lab.avr_polarity.isEnabled()
    lab.lead_order.setCurrentIndex(0)
    assert lab.avr_polarity.isEnabled()
    assert lab.avr_polarity.currentIndex() == 1


@pytest.mark.parametrize('cabrera,inverted', [(False, False), (False, True), (True, False)])
def test_pdf_cabrera_and_polarity_match_requested_display(lab, tmp_path, monkeypatch, cabrera, inverted):
    from vcg_biomarker.gui import lab_export
    from PyQt6.QtPdf import QPdfDocument
    calls = []
    original_draw = lab_export._draw_strip
    def capture(*args, **kwargs):
        calls.append((args[3], kwargs.get('display_sign', 1)))
        return original_draw(*args, **kwargs)
    monkeypatch.setattr(lab_export, '_draw_strip', capture)
    signal = lab.signal.copy()
    path = tmp_path/'lead-display.pdf'
    lab_export.export_ecg_pdf(path, '<p>Test</p>', signal, lab.reference.sampling_rate_hz,
                             cabrera=cabrera, invert_avr=inverted)
    assert [lead for lead, sign in calls[:12]] == ([4, 1, 6, 9, 0, 5, 7, 10, 3, 2, 8, 11] if cabrera else [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    assert all(sign == (-1 if lead == 3 and (cabrera or inverted) else 1) for lead, sign in calls)
    pdf = QPdfDocument(lab)
    assert pdf.load(str(path)).value == 0
    assert pdf.pageCount() >= 2
    # QtPdf 6.9 does not extract QPdfWriter text reliably; test the exact
    # production label helper and rasterized page rather than empty text.
    assert lab_export._lead_display_label(3, -1 if cabrera or inverted else 1) == (
        '−aVR (invertiert)' if cabrera or inverted else 'aVR'
    )
    from PyQt6.QtCore import QSize
    page = pdf.render(1, QSize(600, 420))
    assert not page.isNull()
    np.testing.assert_array_equal(signal, lab.signal)
    pdf.close()


def test_local_marker_favorites_survive_reopen_without_changing_analysis(lab):
    payload = {"case_id":"case-test", "feature_values":{"one":1., "two":2.}}
    lab.set_analysis(payload)
    lab.markers.selectRow(0)
    lab.toggle_marker_favorites()
    lab.only_favorites.setChecked(True)
    assert not lab.markers.isRowHidden(0)
    assert lab.markers.isRowHidden(1)
    assert payload['feature_values'] == {'one':1., 'two':2.}
    lab.case = None
    lab.marker_favorites = set()
    lab.load_case(lab.library, 'case-test')
    lab.set_analysis(payload)
    assert lab.marker_favorites == {'one'}
    assert lab.markers.isRowHidden(1)
    lab.markers.selectRow(0)
    lab.toggle_marker_favorites()
    assert lab.marker_favorites == set()


def test_scientific_displays_are_on_demand_and_blinded(lab):
    widget = lab.scientific_displays
    widget.set_data(lab.signal, 250, np.arange(1., 9.))
    widget.render()
    assert '8/8' in widget.note.text()
    widget.mode.setCurrentIndex(1)
    widget.render()
    assert 'Hann' in widget.note.text()
    lab.mode.setCurrentIndex(1)
    assert lab.tabs.isTabVisible(lab.scientific_display_tab)
    lab.blind.setChecked(True)
    assert not lab.tabs.isTabVisible(lab.scientific_display_tab)
    lab.clear_analysis()
    assert lab.scientific_displays.r_seconds is None


def test_comparison_shared_scales_and_blinded_measurements(lab):
    from vcg_biomarker.tests.test_lab_comparison import payload
    lab.set_analysis({"case_id": "case-test", **payload(120)})
    lab.prior = lab.signal.copy() * .7
    lab.prior_fs = lab.reference.sampling_rate_hz
    lab.comparison_analysis.addItem("Explicit selected analysis", payload(100))
    lab._comparison_analysis_changed()
    assert lab.comparison_values.item(0, 3).text() == "20"
    assert lab.comparison_changes.displayed_rows[0]['delta_current_minus_comparator'] == 20
    assert len(lab.comparison_changes.plot.listDataItems()) == 2
    assert len(lab.comparison_plot.ci.items) == 2
    lab.blind.setChecked(True)
    assert lab.comparison_values.isHidden()
    assert lab.comparison_changes.isHidden()
    lab.blind.setChecked(False)
    assert not lab.comparison_values.isHidden()
    lab.set_analysis({"case_id": "case-test", **payload(110)})
    assert lab.comparison_values.item(0, 3).text() == "10"
    assert lab.comparison_changes.displayed_rows[0]['delta_current_minus_comparator'] == 10
    invalid = payload(100)
    invalid.pop('measurement_protocol')
    lab.comparison_analysis.addItem('Incompatible', invalid)
    lab.comparison_analysis.setCurrentIndex(lab.comparison_analysis.count()-1)
    assert not lab.comparison_changes.displayed_rows
    assert not lab.comparison_changes.plot.listDataItems()


def test_native_annotations_roundtrip(lab):
    lab.landmark_time.setValue(.2)
    lab.set_landmark()
    lab.add_caliper()
    lab.save_annotations()
    assert len(list((lab.store.root / "annotations").glob("*.json"))) == 1


def test_pdf_is_vector_and_uses_paper_units(app, tmp_path):
    from vcg_biomarker.gui.lab_export import export_ecg_pdf, paper_coordinates
    x, y = paper_coordinates([.2, 1], [1, -2])
    np.testing.assert_allclose(x, [5, 25])
    np.testing.assert_allclose(y, [-10, 20])
    path = tmp_path / "report.pdf"
    signal = np.tile(np.sin(np.arange(500)*2*np.pi/250)[:, None], (1, 12))
    export_ecg_pdf(path, "<h1>Prüfbefund</h1>", signal, 250)
    assert path.read_bytes().startswith(b"%PDF")
    assert path.stat().st_size > 2000


def test_optional_research_appendix_is_separate_and_escaped(app, tmp_path):
    from PyQt6.QtPdf import QPdfDocument
    from vcg_biomarker.gui.lab_export import export_ecg_pdf, research_appendix_html
    data = {"feature_values": {"<marker>": None, "not_selected": 1}, "analysis_id": "a&b"}
    appendix = research_appendix_html(data, ['<marker>', '<marker>'])
    assert '&lt;marker&gt;' in appendix
    assert 'not_selected' not in appendix
    assert 'nicht berechenbar' in appendix
    assert 'a&amp;b' in appendix
    signal = np.zeros((500, 12))
    counts = []
    for name, extra in [('plain', None), ('research', appendix)]:
        path = tmp_path / f'{name}.pdf'
        export_ecg_pdf(path, '<h1>Befund</h1>', signal, 500, research_html=extra)
        document = QPdfDocument(None)
        document.load(str(path))
        counts.append(document.pageCount())
        document.close()
    assert counts[1] > counts[0]


def test_blinding_disables_research_pdf(lab):
    lab.pdf_research.setChecked(True)
    lab.blind.setChecked(True)
    assert not lab.pdf_research.isEnabled()
    assert not lab.pdf_research.isChecked()


def test_async_controller_reads_only_matching_result(app, tmp_path, monkeypatch):
    import json
    import sys
    from PyQt6.QtCore import QEventLoop, QTimer
    from vcg_biomarker.gui.lab_controller import LabProcessController
    from vcg_biomarker.mimic.storage import atomic_write_json
    controller = LabProcessController()
    import psutil
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=8 * 1024**3))
    identifier = "a"*64
    path = tmp_path / "cases/case-test/analyses" / identifier[:16] / "analysis.json"
    path.parent.mkdir(parents=True)
    atomic_write_json(path, {"case_id": "case-test", "analysis_id": identifier})
    config = SimpleNamespace(cases_root=tmp_path / "cases", to_dict=lambda: {"work_root": str(tmp_path)})
    original_start = controller.process.start
    response = json.dumps({"case_id": "case-test", "analysis_id": identifier})
    code = f"import sys,time; sys.stdin.readline(); time.sleep(.1); print('LAB_RESULT='+{response!r},flush=True)"
    monkeypatch.setattr(controller.process, "start", lambda *_: original_start(sys.executable, ["-c", code]))
    results, errors = [], []
    loop = QEventLoop()
    controller.completed.connect(lambda value: (results.append(value), loop.quit()))
    controller.failed.connect(lambda value: (errors.append(value), loop.quit()))
    QTimer.singleShot(5000, loop.quit)
    controller.start(config, "case-test", "off")
    loop.exec()
    controller.shutdown()
    assert not errors
    assert results == [{"case_id": "case-test", "analysis_id": identifier}]
    assert not controller.busy


def test_controller_low_ram_does_not_spawn_or_create_case_files(app, tmp_path, monkeypatch):
    import psutil
    from vcg_biomarker.gui.lab_controller import LabProcessController
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=3 * 1024**3))
    controller = LabProcessController()
    def forbidden(*args):
        pytest.fail('Insufficient RAM must not start a subprocess')
    monkeypatch.setattr(controller.process, 'start', forbidden)
    config = SimpleNamespace(cases_root=tmp_path / 'cases')
    with pytest.raises(RuntimeError, match='Laborstart zurückgestellt'):
        controller.start(config, 'case-test', 'off')
    assert not config.cases_root.exists()
    assert not controller.busy
    assert controller.request is None


def test_native_annotations_survive_layouts_and_restore(lab):
    lab.add_caliper()
    expected = lab.measurement_rows()
    lab.lead_choice.setCurrentText("V6")
    lab.layout_choice.setCurrentIndex(3)
    assert lab.measurement_rows() == expected
    lab.layout_choice.setCurrentIndex(0)
    assert lab.measurement_rows() == expected


def test_reloading_analysis_does_not_keep_stale_auto_fiducials(lab):
    payload = {"case_id": "case-test", "delineation": {"representative_fiducials": {"r_peak": 500}}}
    lab.set_analysis(payload)
    assert lab.fiducials["r_peak"] == 250
    payload["delineation"]["representative_fiducials"]["r_peak"] = 1000
    lab.set_analysis(payload)
    assert lab.fiducials["r_peak"] == 500
    lab.clear_analysis()
    assert not lab.fiducials


def test_patient_links_need_source_namespace_and_evidence(tmp_path):
    store = LabReviewStore(tmp_path, "case-test")
    with pytest.raises(ValueError):
        store.save_patient_link("abc", "", "123", "checked", "reviewer")
    store.save_patient_link("abc", "source-a", "123", "checked chart", "reviewer")
    assert store.patient_link("abc")["source"] == "source-a"
    with pytest.raises(ValueError):
        store.patient_link("other-signal")
    store.save_patient_link("abc", "source-b", "123", "corrected source", "reviewer")
    assert len(list((store.root / "patient_links").glob("*.json"))) == 2


def test_report_rejects_reused_revision_and_changed_source(tmp_path):
    store = LabReviewStore(tmp_path, "case-test")
    report = ReportRevision("case-test", "abc", "reviewer")
    store.save_report(report)
    with pytest.raises(ValueError, match="overwritten"):
        store.save_report(replace(report, parent_revision_id=report.revision_id))
    with pytest.raises(ValueError, match="Source signal"):
        store.save_report(replace(report, source_sha256="def", revision_id="new", parent_revision_id=report.revision_id))


def test_metric_scale_uses_actual_viewbox(lab, app):
    lab.layout_choice.setCurrentIndex(3)
    lab.fit.setChecked(False)
    lab.show()
    app.processEvents()
    plot = lab.plots[0][0]
    lab._metric_scale(plot, 0)
    x, y = plot.viewRange()
    assert (x[1]-x[0]) == pytest.approx(plot.vb.width()/(lab.pixels_per_mm.value()*25))
    assert (y[1]-y[0]) == pytest.approx(plot.vb.height()/(lab.pixels_per_mm.value()*10))


def test_ecg_zoom_unclipped_signal_and_rule_evidence_popup(lab, app):
    lab.signal[120, 0] = 8.0
    lab.layout_choice.setCurrentIndex(3)
    lab.lead_choice.setCurrentText('I')
    lab.view_duration.setCurrentIndex(lab.view_duration.findData(1.0))
    lab.show()
    app.processEvents()


    plot = lab.plots[0][0]
    x, y = plot.viewRange()
    assert x[1] - x[0] < 1.4
    assert y[1] > 8.0 and y[0] < -8.0
    assert plot.vb.state['mouseEnabled'] == [False, False]
    assert lab.heart_overlay.parentWidget() is lab.viewer_strip
    assert not lab.canvas.geometry().intersects(lab.viewer_strip.geometry())

    lab.set_analysis({'case_id': lab.case.case_id, 'analysis_id': 'visual-test',
                      'classical_criteria': {'test_rule': {
                          'value': 1.0, 'unit': 'mV', 'positive': True,
                          'evaluable': True, 'exploratory': False,
                          'definition': 'Stored test criterion'}},
                      'delineation': {'representative_fiducials': {'r_peak': 100},
                                      'interval_measurements': []}})
    assert '1 beurteilbare' in lab.quick_suggestion_status.text()
    lab.show_rule_evidence('test_rule')
    app.processEvents()
    assert lab.rule_evidence_dialog.isVisible()
    assert 'gespeicherte Regelevidenz' in lab.rule_evidence_dialog.windowTitle()
    lab.show_landmark_editor()
    app.processEvents()
    assert lab.landmark_editor.isVisible()
    lab.rule_evidence_dialog.close()
    lab.landmark_editor.close()


def test_lab_signature_workspace_keeps_analysis_setup_explicit(lab):
    original_signal = lab.signal
    assert lab.lab_hero.objectName() == "ecgLabHero"
    assert lab.signal_stage.objectName() == "ecgLabSignalStage"
    assert lab.measurement_dock.objectName() == "ecgLabMeasurementDock"
    assert lab.setup_panel.isHidden()
    assert lab.minimumSizeHint().width() <= 1366
    lab.setup_toggle.setChecked(True)
    assert not lab.setup_panel.isHidden()
    lab.setup_toggle.setChecked(False)
    assert lab.setup_panel.isHidden()
    assert lab.signal is original_signal


def test_edit_completed_report_immediately_invalidates_display_status(lab):
    lab.sections['Zusammenfassung'].setPlainText('Manuell geprüft')
    assert lab.save_report('draft')
    assert lab.save_report('reviewed')
    assert lab.save_report('completed')
    revision = lab.parent_revision
    lab.sections['Zusammenfassung'].setPlainText('Nachträglich geändert')
    assert lab._dirty
    assert 'Ungesicherte Änderungen' in lab.report_status.text()
    assert 'erneute Prüfung' in lab.report_status.text()
    assert lab.parent_revision == revision
    assert lab.save_report('draft')
    assert not lab._dirty
    assert lab.parent_revision != revision
    assert lab.report_status.text().startswith('draft')


def test_median_export_local_json(lab, monkeypatch, tmp_path):
    import json
    from PyQt6.QtWidgets import QFileDialog
    from vcg_biomarker.tests.test_lab_representative_beat import median_inputs
    data, inventory = median_inputs()
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'a', 'beat_inventory': inventory})
    lab.analysis_arrays = data
    lab.template_cluster.setCurrentIndex(lab.template_cluster.findData(0))
    lab.build_median_template()
    target = tmp_path / 'median.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(target), 'JSON'))
    lab.export_median_template()
    document = json.loads(target.read_text(encoding='utf-8'))
    assert document['analysis_id'] == 'a'
    assert document['signal_reference']['source_sha256'] == 'abc'
    assert len(document['provenance']['used_beats']) == 3


def test_manual_template_interval_stays_separate_and_invalidates(lab):
    from vcg_biomarker.tests.test_lab_representative_beat import median_inputs
    data, inventory = median_inputs()
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'a', 'beat_inventory': inventory})
    lab.analysis_arrays = data
    lab.template_cluster.setCurrentIndex(lab.template_cluster.findData(0))
    lab.build_median_template()
    lab.add_template_measurement()
    lab.template_region.setRegion((1., 39.))
    lab.save_template_measurement()
    assert len(lab.template_annotations) == 1
    assert lab.template_annotations[0]['duration_ms'] == 40
    assert not lab.manual_fiducials
    assert lab.template_region.getRegion() == (0., 40.)
    lab.set_dark(True)
    assert lab.template_region.getRegion() == (0., 40.)
    lab.representative_path.setCurrentIndex(1)
    assert lab.template_annotations == []
    assert lab.template_region is None
    lab.representative_path.setCurrentIndex(0)
    lab.build_median_template()
    assert len(lab.template_annotations) == 1
    assert lab.template_annotations[0]['duration_ms'] == 40


def test_template_table_selects_exact_measurement_for_withdrawal(lab, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    from vcg_biomarker.tests.test_lab_representative_beat import median_inputs
    data, inventory = median_inputs()
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'a', 'beat_inventory': inventory})
    lab.analysis_arrays = data
    lab.template_cluster.setCurrentIndex(lab.template_cluster.findData(0))
    lab.build_median_template()
    lab.add_template_measurement()
    lab.template_region.setRegion((0., 40.))
    lab.save_template_measurement()
    first = lab.template_annotations[0]['local_event_id']
    lab.template_region.setRegion((0., 80.))
    lab.save_template_measurement()
    assert lab.template_measurement_table.rowCount() == 2
    lab.template_measurement_table.selectRow(0)
    lab.show_template_measurement(0, 0)
    assert lab.template_region.getRegion() == (0., 40.)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('Grenze korrigiert', True))
    lab.withdraw_template_measurement()
    assert len(lab.template_annotations) == 1
    assert lab.template_annotations[0]['local_event_id'] != first
    assert lab.template_annotations[0]['duration_ms'] == 80
    assert lab.template_measurement_table.rowCount() == 1


def test_median_explicit_cluster_selection_and_invalidation(lab):
    from vcg_biomarker.tests.test_lab_representative_beat import median_inputs
    data, inventory = median_inputs()
    lab.set_analysis({'case_id': 'case-test', 'beat_inventory': inventory})
    lab.analysis_arrays = data
    lab.build_median_template()
    assert not lab.median_template['available']
    lab.template_cluster.setCurrentIndex(lab.template_cluster.findData(0))
    lab.build_median_template()
    assert lab.median_template['available']
    assert len(lab.representative_plot.listDataItems()) == 12
    assert 'used_beats' in lab.representative_note.toolTip()
    lab.representative_path.setCurrentIndex(1)
    assert lab.median_template is None
    assert not lab.representative_plot.listDataItems()


def test_representative_saved_beat_and_reset(lab):
    from vcg_biomarker.tests.test_lab_representative_beat import arrays
    lab.analysis_arrays = arrays()
    lab.draw_representative_beat()
    assert len(lab.representative_plot.listDataItems()) == 12
    lab.representative_lead.setCurrentIndex(4)
    assert len(lab.representative_plot.listDataItems()) == 1
    assert lab.representative_plot.listDataItems()[0].name() == 'aVR'
    lab.representative_path.setCurrentIndex(1)
    assert not lab.representative_plot.listDataItems()
    assert 'nicht gespeichert' in lab.representative_note.text()
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'new'})
    assert lab.analysis_arrays is None
    assert not lab.representative_plot.listDataItems()


def test_endpoint_review_loaded_and_cleared(lab):
    lab.set_analysis({'case_id': 'case-test', 'research_findings': [
        dict(endpoint_id='ef', classification='positive', applicable=True)]})
    assert lab.endpoint_review.rowCount() == 1
    assert lab.endpoint_review.item(0, 0).text() == 'ef'
    assert lab.endpoint_review.item(0, 1).text() == 'Nicht beurteilbar'
    lab.clear_analysis()
    assert lab.endpoint_review.rowCount() == 0


def test_inconsistent_classical_value_cannot_be_accepted(lab, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[-1]))
    lab.set_analysis({'case_id': 'case-test', 'classical_criteria': {
        'bad_rule': dict(exploratory=False, evaluable=True, value=None, positive=True)}})
    lab.suggestions.selectRow(0)
    lab.review_suggestion('accepted')
    assert warnings
    assert 'bad_rule' not in lab.suggestion_review
    assert 'Messwert fehlt' in lab.suggestions.item(0, 2).text()
    lab.review_suggestion('rejected')
    assert lab.suggestion_review['bad_rule']['decision'] == 'rejected'


def test_rhythm_note_reset_and_blinded(lab):
    lab.set_analysis({'case_id': 'case-test', 'delineation': {'rhythm_context': {
        'class': 'af_compatible_pattern', 'rr_median_ms': 800, 'confidence': .8}}})
    assert 'keine bestätigte AF-Diagnose' in lab.rhythm_review_note.text()
    assert '75.0/min' in lab.rhythm_review_note.text()
    lab.blind.setChecked(True)
    assert not lab.tabs.isTabVisible(lab.suggestions_tab)
    lab.clear_analysis()
    assert 'Kein kompatibler' in lab.rhythm_review_note.text()


def test_review_detail_and_literal_report_insertion(lab):
    definition = 'QRS <120 ms; <b>wörtlicher Text</b>'
    lab.set_analysis({'case_id': 'case-test', 'analysis_id': 'old', 'classical_criteria': {
        'rule': dict(exploratory=False, evaluable=True, value=110., positive=True,
                     definition=definition, unit='ms')}})
    lab.suggestions.selectRow(0)
    assert definition in lab.suggestion_detail.toPlainText()
    assert 'Quellen-ID: nicht dokumentiert' in lab.suggestion_detail.toPlainText()
    lab.review_suggestion('accepted')
    inserted = lab.sections[lab.section_choice.currentText()].toPlainText()
    assert definition in inserted
    assert lab.suggestion_review['rule']['text'] == inserted
    lab.clear_analysis()
    assert not lab.suggestion_detail.toPlainText()


def test_changed_analysis_requires_new_report_review(lab):
    lab.sections["Zusammenfassung"].setPlainText("Manuell geprüft")
    assert lab.save_report("draft")
    assert lab.save_report("reviewed")
    assert lab.save_report("completed")
    lab.set_analysis({"case_id": "case-test", "analysis_id": "different"})
    assert lab._dirty
    assert "erneute Prüfung" in lab.report_status.text()


def test_caliper_json_and_plot_exports(lab, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog
    lab.add_caliper()
    for suffix in ("png", "svg"):
        path = tmp_path / f"plot.{suffix}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_, p=path: (str(p), ""))
        lab.export_plot()
        assert path.stat().st_size > 100


def test_direct8_inspector_uses_recorded_windows(lab):
    name = "qrs_t_area_angle_direct_8lead_deg"
    lab.set_analysis({"case_id": "case-test", "feature_values": {name: 60.0},
        "direct8_measurement_windows": {"signal_path": "robust_morphology_ecg", "coordinate_system": "analysis_samples_inclusive", "sampling_rate_hz": 250,
            "leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
            "pr_baseline": [10, 20], "qrs": [30, 60], "j_to_t_end": [60, 120]}})
    lab.analysis_arrays = {"robust_morphology_ecg": lab.signal.copy()}
    lab.markers.selectRow(0)
    lab.inspect_marker()
    assert len(lab.marker_signal.listDataItems()) == 8
    assert "kein 3D" in lab.marker_signal.plotItem.titleLabel.text
    assert "nicht vollständig belegt" in lab.marker_evidence.summaries[0].text()
    assert "Messfenster mit lokalem Signal prüfbar" in lab.marker_evidence.summaries[2].text()
    assert "keine Diagnose" in lab.marker_evidence.summaries[3].text()


def test_marker_evidence_trail_distinguishes_missing_formula_and_reference():
    from vcg_biomarker.gui.widgets.marker_evidence_trail import marker_evidence_steps

    rows = marker_evidence_steps({"marker": "research_x", "value": 0.7,
                                  "recorded_definition": {"unit": "mV"}},
                                 None, None, signal_bound=False)
    assert "nicht vollständig belegt" in rows[0][1]
    assert "Formel im geladenen Fallartefakt nicht enthalten" in rows[1][2]
    assert "Keine exakten Messfenster" in rows[2][1]
    assert "keine unabhängige" in rows[3][2]


def test_marker_evidence_uses_stored_abstention_without_inventing_a_reason():
    from vcg_biomarker.gui.widgets.marker_evidence_trail import marker_evidence_steps

    base = {"marker": "p_loop_width_major", "value": None}
    unknown = marker_evidence_steps(dict(base, recorded_definition={}), None, None,
                                    signal_bound=True)
    assert "Keine exakten Messfenster" in unknown[2][1]
    assert "Gespeicherter Grund" not in unknown[2][2]
    p_gate = marker_evidence_steps(dict(base, recorded_definition={
        "notes": "not_evaluable: p fiducials below the pre-specified confidence threshold (0.35)"}),
        None, None, signal_bound=True)
    assert "Nicht auswertbar" in p_gate[2][1]
    assert "P-Wellengrenzen" in p_gate[2][2]
    beat_gate = marker_evidence_steps(dict(base, recorded_definition={
        "notes": '{"status":"not_evaluable","reason":"insufficient_quality_eligible_beats"}'}),
        None, None, signal_bound=True)
    assert "Zu wenige qualitätsgeeignete Schläge" in beat_gate[2][2]


@pytest.mark.parametrize('variant', ['preqrs', 'tp'])
def test_direct8_variant_windows_are_not_replaced_by_pr(lab, variant):
    from vcg_biomarker.gui.lab_measurement_windows import DIRECT8_LEADS
    name = f'qrs_t_area_angle_direct_8lead_{variant}_deg'
    window = {'schema_version': '1.0.0', 'signal_path': 'robust_morphology_ecg',
        'coordinate_system': 'analysis_samples_inclusive', 'sampling_rate_hz': 250,
        'baseline_kind': variant, 'leads': DIRECT8_LEADS,
        'baseline': [5, 12], 'qrs': [30, 60], 'j_to_t_end': [60, 120]}
    lab.set_analysis({'case_id': 'case-test', 'feature_values': {name: 60.},
                      'feature_measurements': {name: {'measurement_windows': window}}})
    lab.analysis_arrays = {'robust_morphology_ecg': lab.signal.copy()}
    lab.markers.selectRow(0)
    lab.inspect_marker()
    assert len(lab.marker_signal.listDataItems()) == 8
    assert variant in lab.marker_signal.plotItem.titleLabel.text
    window['sampling_rate_hz'] = 0
    lab.inspect_marker()
    assert not lab.marker_signal.listDataItems()


def test_other_signal_analysis_is_rejected(lab):
    with pytest.raises(ValueError, match="Originalsignal"):
        lab.set_analysis({"case_id": "case-test", "signal_reference": {"source_sha256": "other"}})


def test_direct8_beat_inspector_uses_record_signal_and_resets(lab):
    from vcg_biomarker.gui.lab_measurement_windows import DIRECT8_LEADS
    name = 'qrs_t_area_angle_direct_8lead_beat_median_preqrs_deg'
    contract = {'schema_version': '1.1.0', 'scope': 'beat_summary', 'signal_path': 'morphology_ecg',
        'coordinate_system': 'analysis_samples_inclusive', 'sampling_rate_hz': 250,
        'baseline_kind': 'preqrs', 'leads': DIRECT8_LEADS, 'eligible_beats': 3, 'stored_beats': 2,
        'record_beats': 5, 'minimum_beats': 3, 'truncated': True,
        'beat_audit': [{'beat_index': 0, 'included': True, 'reason': 'included'},
                       {'beat_index': 4, 'included': False, 'reason': 'segment_quality_below_threshold'}],
        'items': [{'beat_index': i, 'value': 60. + i, 'baseline': [10 + i*100, 20 + i*100],
                   'qrs': [30+i*100, 60+i*100], 'j_to_t_end': [60+i*100, 90+i*100]}
                  for i in range(2)]}
    lab.set_analysis({'case_id': 'case-test', 'feature_values': {name: 60.},
                      'feature_measurements': {name: {'measurement_windows': contract}}})
    lab.analysis_arrays = {'morphology_ecg': lab.signal.copy()}
    lab.markers.selectRow(0)
    lab.inspect_marker()
    assert lab.marker_beat.count() == 2
    assert lab.marker_beat_audit.rowCount() == 2
    assert lab.marker_beat_audit.item(1, 1).text() == 'nein'
    assert 'Segmentqualität' in lab.marker_beat_audit.item(1, 2).text()
    assert len(lab.marker_signal.listDataItems()) == 8
    first_time = lab.marker_signal.listDataItems()[0].xData[0]
    lab.marker_beat.setCurrentIndex(1)
    assert lab.marker_beat.currentData() == 1
    assert lab.marker_signal.listDataItems()[0].xData[0] > first_time
    assert 'nicht der Median/MAD' in lab.marker_signal.plotItem.titleLabel.text
    lab.clear_analysis()
    assert lab.marker_beat.count() == 0
    assert lab.marker_beat_audit.rowCount() == 0


def test_unreviewed_draft_recovery(lab, app):
    from vcg_biomarker.gui.widgets.expert_ecg_lab import ExpertECGLab
    lab.sections["Zusammenfassung"].setPlainText("Noch nicht fertig befundet")
    lab.manual_overrides = {'QT': {'manual_value': '410 ms', 'author': '<b>Reviewer</b>',
                                  'analysis_id': 'original-analysis'}}
    lab._save_recovery()
    other = ExpertECGLab()
    try:
        other.load_case(lab.library, lab.case.case_id)
        assert other.sections["Zusammenfassung"].toPlainText() == "Noch nicht fertig befundet"
        assert other._dirty
        assert "nicht geprüft" in other.report_status.text()
        assert other.store.reports() == []
        assert '410 ms' in other.override_note.text()
        assert 'original-analysis' in other.override_note.text()
        from PyQt6.QtCore import Qt
        assert other.override_note.textFormat() == Qt.TextFormat.PlainText
        other.manual_overrides = {}
        other.refresh_manual_overrides()
        assert '410 ms' not in other.override_note.text()
    finally:
        other.deleteLater()
        app.processEvents()
def test_manual_override_requires_author_and_preserves_source(lab, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog, QMessageBox, QTableWidgetItem
    from datetime import datetime
    lab.measurements.setRowCount(1)
    for col, text in enumerate(('QT', '400 ms', 'Automatisch')):
        lab.measurements.setItem(0, col, QTableWidgetItem(text))
    lab.measurements.selectRow(0)
    lab.payload = {'analysis_id': 'original-analysis'}
    lab.author.setText(' ')
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args))
    def forbidden(*args):
        pytest.fail('No correction dialog without author')
    monkeypatch.setattr(QInputDialog, 'getText', forbidden)
    lab.override_measurement()
    assert warnings and not lab.manual_overrides
    lab.author.setText(' reviewer ')
    answers = iter([(' 410 ms ', True), (' T-Ende manuell geprüft ', True)])
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: next(answers))
    lab.override_measurement()
    entry = lab.manual_overrides['QT']
    assert entry['author'] == 'reviewer'
    assert entry['manual_value'] == '410 ms'
    assert entry['automatic_value'] == '400 ms'
    assert entry['analysis_id'] == 'original-analysis'
    assert entry['signal_reference']['source_sha256'] == 'abc'
    assert datetime.fromisoformat(entry['created_at']).tzinfo is not None
    assert lab.measurements.item(0, 1).text() == '400 ms'
    assert 'Ungesicherte Änderungen' in lab.report_status.text()
def test_keyboard_undo_targets_text_not_landmarks(lab, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    editor = next(iter(lab.sections.values()))
    editor.insertPlainText('Befundtext')
    monkeypatch.setattr(QApplication, 'focusWidget', lambda: editor)
    calls = []
    monkeypatch.setattr(lab, 'undo', lambda: calls.append('landmark undo'))
    monkeypatch.setattr(lab, 'redo', lambda: calls.append('landmark redo'))
    lab.undo_focused()
    assert editor.toPlainText() == ''
    lab.undo_focused(True)
    assert editor.toPlainText() == 'Befundtext'
    assert not calls
    editor.setReadOnly(True)
    lab.undo_focused()
    assert not calls
    monkeypatch.setattr(QApplication, 'focusWidget', lambda: lab.canvas)
    lab.undo_focused()
    assert calls == ['landmark undo']
def test_report_controls_do_not_expand_into_large_vertical_gaps(lab):
    from PyQt6.QtWidgets import QApplication
    page = lab.tabs.widget(0)
    lab.tabs.setCurrentIndex(0)
    lab.show()
    page.resize(1300, 2000)
    page.layout().activate()
    QApplication.processEvents()
    # The phrase controls must remain close to the author and first field,
    # even when another tab forces a tall shared content area.
    phrase_bottom = lab.phrase.mapTo(page, lab.phrase.rect().bottomLeft()).y()
    save_top = lab.phrase_save.mapTo(page, lab.phrase_save.rect().topLeft()).y()
    first = next(iter(lab.sections.values()))
    first_top = first.mapTo(page, first.rect().topLeft()).y()
    save_bottom = lab.phrase_save.mapTo(page, lab.phrase_save.rect().bottomLeft()).y()
    assert save_top - phrase_bottom < 60
    assert first_top - save_bottom < 60
    last = list(lab.sections.values())[-1]
    status_top = lab.report_status.mapTo(page, lab.report_status.rect().topLeft()).y()
    last_bottom = last.mapTo(page, last.rect().bottomLeft()).y()
    assert status_top - last_bottom < 60
@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True, '0'])
def test_research_appendix_invalid_values_cannot_look_measured(value):
    from vcg_biomarker.gui.lab_export import research_appendix_html
    payload = {'feature_values': {'marker': value},
               'feature_measurements': {'marker': {'status': 'ok'}}}
    result = research_appendix_html(payload, ['marker'])
    assert 'nicht berechenbar / ungültig' in result
    assert 'gespeicherter Status: ok' in result


def test_research_appendix_partial_metadata_and_real_zero():
    from vcg_biomarker.gui.lab_export import research_appendix_html
    result = research_appendix_html({'feature_values': {'x': 0.}, 'feature_measurements': None}, ['x'])
    assert '<td>0.0</td>' in result
    assert 'historischer Stand ohne Messstatus' in result
    assert 'nicht dokumentiert' in result
    assert 'Forschungsanhang' in research_appendix_html({'feature_values': None}, ['x'])
def test_variable_phrase_preview_confirmation_and_blinding(lab, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    lab.payload = {'analysis_id': 'saved', 'delineation': {'interval_measurements': [
        {'interval': 'QTc Fridericia', 'value_ms': 410., 'status': 'measurable'}]}}
    lab.phrase.setEditText('QTc {qtc_fridericia} ms nach Fridericia.')
    edit = lab.sections[lab.section_choice.currentText()]
    monkeypatch.setattr(QMessageBox, 'exec', lambda self: QMessageBox.StandardButton.No)
    lab.insert_phrase()
    assert not edit.toPlainText()
    monkeypatch.setattr(QMessageBox, 'exec', lambda self: QMessageBox.StandardButton.Yes)
    lab.insert_phrase()
    assert edit.toPlainText() == 'QTc 410 ms nach Fridericia.'
    lab.blind.setChecked(True)
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args))
    lab.insert_phrase()
    assert warnings
    assert edit.toPlainText() == 'QTc 410 ms nach Fridericia.'
def test_march_out_preserves_width_and_identity_at_signal_edges(lab):
    lab.add_caliper(True)
    before = lab.measurement_rows()[0]
    lines = lab.calipers[0][1]
    width = lines[1].value() - lines[0].value()
    lab.caliper_march.setChecked(True)
    lines[0].setValue(lines[0].value() + .1)
    after = lab.measurement_rows()[0]
    assert after['measurement_id'] == before['measurement_id']
    assert after['end_sample'] - after['start_sample'] == before['end_sample'] - before['start_sample']
    lines[1].setValue(1000)
    assert lines[1].value() <= (lab.reference.sample_count-1)/lab.reference.sampling_rate_hz
    assert lines[1].value() - lines[0].value() == pytest.approx(width)
    lines[0].setValue(-1000)
    assert lines[0].value() == 0
    assert lines[1].value() - lines[0].value() == pytest.approx(width)
    lab.caliper_march.setChecked(False)
    lines[1].setValue(width / 2)
    assert lines[1].value() - lines[0].value() == pytest.approx(width / 2)
def test_screen_calibration_accept_cancel_preserve_signal(lab, monkeypatch):
    from vcg_biomarker.gui.lab_screen_calibration import ScreenCalibrationDialog
    original = lab.signal.copy()
    old = lab.pixels_per_mm.value()
    monkeypatch.setattr(ScreenCalibrationDialog, 'exec', lambda self: self.DialogCode.Rejected)
    lab.calibrate_screen()
    assert lab.pixels_per_mm.value() == old
    monkeypatch.setattr(ScreenCalibrationDialog, 'exec', lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(ScreenCalibrationDialog, 'calibration', lambda self: 4.0)
    lab.calibrate_screen()
    assert lab.pixels_per_mm.value() == 4.
    assert not lab.fit.isChecked()
    np.testing.assert_array_equal(original, lab.signal)
def test_measurement_variable_picker_updates_and_never_inserts_result_automatically(lab):
    lab.refresh_measurement_variables({'delineation': {'interval_measurements': [
        {'interval': 'QTc Fridericia', 'value_ms': 410., 'status': 'measurable'}]}})
    index = lab.phrase_variable.findData('qtc_fridericia')
    lab.phrase_variable.setCurrentIndex(index)
    assert '410 ms' in lab.phrase_variable.currentText()
    lab.phrase.setEditText('QTc ')
    lab.phrase.lineEdit().setCursorPosition(4)
    lab.insert_measurement_variable()
    assert lab.phrase.currentText() == 'QTc {qtc_fridericia}'
    assert all(not edit.toPlainText() for edit in lab.sections.values())
    lab.refresh_measurement_variables({})
    assert 'nicht verfügbar' in lab.phrase_variable.currentText()
    lab.blind.setChecked(True)
    assert 'verborgen' in lab.phrase_variable.currentText()
    assert not lab.phrase_variable_insert.isEnabled()
