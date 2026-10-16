import json
from vcg_biomarker.gui.lab_qt_alert import qt_alert_items, qt_review_history
from vcg_biomarker.mimic.storage import content_hash


def test_history_binds_signal_analysis_and_measurements(tmp_path):
    reference = {'case_id': 'one', 'source_sha256': 'hash'}
    payload = {'analysis_id': 'analysis', 'delineation': {'interval_measurements': [
        {'interval': 'QTc Fridericia', 'value_ms': 510, 'status': 'measurable'}]}}
    items = qt_alert_items(payload)
    doc = dict(schema_version='1.0.0', kind='qt_review_acknowledgement', signal_reference=reference,
               analysis_id='analysis', items=items, items_sha256=content_hash(items),
               author='Reviewer', review_note='Reviewed manually', created_at='2026-09-22T10:00:00Z')
    path = tmp_path / 'one.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    rows, errors = qt_review_history(tmp_path, reference, payload)
    assert rows == [doc] and not errors
    assert qt_review_history(tmp_path, {'case_id': 'other'}, payload) == ([], [])
    assert qt_review_history(tmp_path, reference, dict(payload, analysis_id='other')) == ([], [])
    doc['items'][0]['value_ms'] = 520
    path.write_text(json.dumps(doc), encoding='utf-8')
    rows, errors = qt_review_history(tmp_path, reference, payload)
    assert not rows and len(errors) == 1


def test_corrupt_history_is_visible_as_error(tmp_path):
    (tmp_path / 'broken.json').write_text('broken', encoding='utf-8')
    rows, errors = qt_review_history(tmp_path, {}, {})
    assert not rows and len(errors) == 1
