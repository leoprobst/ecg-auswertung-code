import json
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox
from vcg_biomarker.gui.lab_laddergram import LaddergramWidget, validate_diagram

@pytest.fixture
def diagram(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = LaddergramWidget()
    widget.set_context({'case_id': 'case', 'source_sha256': 'abc', 'sampling_rate_hz': 250,
                        'sample_count': 2500}, tmp_path)
    yield widget
    widget.close()
    app.processEvents()


def test_manual_events_edges_undo_and_native_samples(diagram):
    diagram.time.setValue(.1)
    diagram.add_event()
    diagram.level.setCurrentText('V')
    diagram.time.setValue(.25)
    diagram.add_event()
    assert [item['sample'] for item in diagram.events] == [25, 62]
    diagram.target.setCurrentIndex(1)
    diagram.add_edge()
    assert len(diagram.edges) == 1
    diagram.remove_event()
    assert len(diagram.events) == 1 and not diagram.edges
    diagram.undo()
    assert len(diagram.events) == 2 and len(diagram.edges) == 1
    assert validate_diagram(diagram.document(), diagram.reference)['clinical_claim_allowed'] is False


def test_revision_and_recovery_are_separate_and_case_bound(diagram, tmp_path, monkeypatch):
    diagram.add_event()
    draft = diagram.root / 'draft_recovery.json'
    assert draft.exists()
    diagram.save()
    assert diagram.revisions.count() == 0  # author required for named revision
    diagram.author.setText('Reviewer')
    diagram.save()
    assert diagram.revisions.count() == 1
    saved = diagram.revisions.currentData()
    before = Path(saved).read_text(encoding='utf-8')
    diagram.time.setValue(1)
    diagram.add_event()
    assert Path(saved).read_text(encoding='utf-8') == before
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    diagram.load()
    assert len(diagram.events) == 1
    document = json.loads(draft.read_text(encoding='utf-8'))
    with pytest.raises(ValueError):
        validate_diagram(document, dict(diagram.reference, source_sha256='different'))
    diagram.set_context(diagram.reference, tmp_path)
    assert len(diagram.events) == 1


def test_backward_time_and_invalid_samples_rejected(diagram):
    diagram.time.setValue(1)
    diagram.add_event()
    diagram.time.setValue(.5)
    diagram.add_event()
    diagram.target.setCurrentIndex(1)
    diagram.add_edge()
    assert not diagram.edges
    document = diagram.document()
    document['events'][0]['sample'] = True
    with pytest.raises(ValueError):
        validate_diagram(document, diagram.reference)
def test_move_event_snaps_and_rejects_time_reversal(diagram):
    diagram.add_event()
    diagram.time.setValue(1)
    diagram.add_event()
    diagram.target.setCurrentIndex(1)
    diagram.add_edge()
    first = diagram.events[0]['id']
    diagram.move_event(first, .101)
    assert diagram.events[0]['sample'] == 25
    diagram.move_event(first, 2.)
    assert diagram.events[0]['sample'] == 25
    assert 'abgelehnt' in diagram.status.text()
    diagram.undo()
    assert diagram.events[0]['sample'] == 0


@pytest.mark.parametrize('format', ['PNG', 'SVG', 'PDF'])
def test_local_graphic_export_with_matching_json(diagram, format):
    diagram.add_event()
    diagram.author.setText('Manual reviewer')
    diagram.export_format.setCurrentText(format)
    diagram.export()
    paths = list((diagram.root / 'exports').glob('*.' + format.lower()))
    assert len(paths) == 1 and paths[0].stat().st_size > 100
    document = json.loads(paths[0].with_suffix('.json').read_text(encoding='utf-8'))
    assert document['signal_reference'] == diagram.reference
    assert document['clinical_claim_allowed'] is False
    if format == 'PDF':
        from PyQt6.QtPdf import QPdfDocument
        pdf = QPdfDocument(diagram)
        assert pdf.load(str(paths[0])).value == 0 and pdf.pageCount() == 1
        # The offscreen platform (set by whichever GUI test created the shared
        # QApplication first) has no embeddable system fonts, so its PDFs carry
        # no text layer. The native platform must embed the caption as text.
        if QApplication.platformName() != 'offscreen':
            assert 'Manuelles Laddergram' in pdf.getAllText(0).text()
def test_ecg_strip_uses_native_samples_and_clears_on_context_change(diagram, tmp_path):
    import numpy as np
    signal = np.arange(2500*12, dtype=float).reshape(2500, 12) / 1000
    original = signal.copy()
    diagram.set_context(diagram.reference, tmp_path, signal)
    curve = diagram.ecg_plot.listDataItems()[0]
    np.testing.assert_array_equal(curve.yData, signal[:, 1])
    np.testing.assert_allclose(curve.xData, np.arange(2500)/250)
    diagram.lead.setCurrentText('aVR')
    np.testing.assert_array_equal(diagram.ecg_plot.listDataItems()[0].yData, signal[:, 3])
    diagram.time.setValue(.4)
    assert diagram.ecg_cursor.value() == .4
    assert diagram.cursor.value() == .4
    np.testing.assert_array_equal(signal, original)
    diagram.set_context(diagram.reference, tmp_path)
    assert not diagram.ecg_plot.listDataItems()


def test_ecg_strip_wrong_shape_rejected_without_changing_context(diagram, tmp_path):
    import numpy as np
    with pytest.raises(ValueError):
        diagram.set_context(diagram.reference, tmp_path, np.zeros((10, 12)))
    assert diagram.signal is None


@pytest.mark.parametrize('format', ['PNG', 'SVG', 'PDF'])
def test_export_native_ecg_strip_with_scale_metadata(diagram, tmp_path, format):
    import numpy as np
    signal = np.zeros((2500, 12))
    signal[:, 3] = np.linspace(-.3, .7, 2500)
    signal[100, 3] = np.nan
    original = signal.copy()
    diagram.set_context(diagram.reference, tmp_path, signal)
    diagram.lead.setCurrentText('aVR')
    diagram.author.setText('Reviewer')
    diagram.export_ecg.setChecked(True)
    diagram.export_format.setCurrentText(format)
    diagram.export()
    path, = (diagram.root / 'exports').glob('*.' + format.lower())
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['export_ecg']
    assert metadata['lead'] == 'aVR'
    assert metadata['minimum'] == -.3 and metadata['maximum'] == .7
    assert metadata['amplitude_scale'] == 'automatic_not_calibrated_ecg_paper'
    np.testing.assert_array_equal(signal, original)


def test_export_requested_strip_without_signal_is_explicit_error(diagram):
    diagram.author.setText('Reviewer')
    diagram.export_ecg.setChecked(True)
    diagram.export()
    assert 'Kein Originalsignal' in diagram.status.text()
    assert not list((diagram.root / 'exports').glob('*'))
