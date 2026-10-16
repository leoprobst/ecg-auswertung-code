import json
import pytest

from vcg_biomarker.gui.lab_phrase_library import add_phrase, encode_phrases, decode_phrases
from vcg_biomarker.gui.lab_phrase_library import change_phrase
from vcg_biomarker.mimic.lab_workspace import REPORT_SECTIONS


def test_roundtrip_literal_text_and_deduplication():
    entries = add_phrase([], REPORT_SECTIONS[0], 'QT < 400 & <b>kein HTML</b>')
    assert add_phrase(entries, REPORT_SECTIONS[0], entries[0]['text']) == entries
    assert decode_phrases(encode_phrases(entries)) == entries
    assert len(add_phrase(entries, REPORT_SECTIONS[1], entries[0]['text'])) == 2


@pytest.mark.parametrize('raw', ['null', '[]', '{}', '{broken',
    json.dumps({'schema_version': '9', 'entries': []}),
    json.dumps({'schema_version': '1.0.0', 'entries': [{'section': 'unknown', 'text': 'x'}]})])
def test_bad_library_rejected(raw):
    with pytest.raises((ValueError, TypeError)):
        decode_phrases(raw)


def test_limits_and_no_mutation():
    entries = []
    with pytest.raises(ValueError):
        add_phrase(entries, REPORT_SECTIONS[0], 'x' * 4001)
    assert entries == []
    assert decode_phrases(None) == []


def test_edit_remove_preserve_other_sections_and_reject_stale_targets():
    section, other = REPORT_SECTIONS[:2]
    entries = add_phrase(add_phrase([], section, 'Original'), other, 'Original')
    updated = change_phrase(entries, section, 'Original', 'Neu')
    assert entries[0]['text'] == 'Original'
    assert updated == [{'section': section, 'text': 'Neu'}, {'section': other, 'text': 'Original'}]
    assert change_phrase(updated, section, 'Neu') == updated[1:]
    with pytest.raises(ValueError):
        change_phrase(updated, section, 'Original', 'Veraltet')
    with pytest.raises(ValueError):
        change_phrase(updated, section, 'Neu', ' ')


def test_edit_cannot_create_duplicate():
    entries = add_phrase(add_phrase([], REPORT_SECTIONS[0], 'A'), REPORT_SECTIONS[0], 'B')
    with pytest.raises(ValueError, match='Doppelter'):
        change_phrase(entries, REPORT_SECTIONS[0], 'A', 'B')
def test_measurement_template_uses_exact_saved_formula_without_recalculation():
    from vcg_biomarker.gui.lab_phrase_library import render_measurement_phrase
    item = {'interval': 'QTc Fridericia', 'value_ms': 412.345, 'status': 'measurable', 'formula': 'saved formula'}
    payload = {'analysis_id': 'saved', 'delineation': {'interval_measurements': [item]}}
    result = render_measurement_phrase('QTc {qtc_fridericia} ms nach Fridericia.', payload)
    assert result['text'] == 'QTc 412.345 ms nach Fridericia.'
    assert result['measurements']['qtc_fridericia'] == item
    assert result['analysis_id'] == 'saved'
    assert result['missing'] == []


@pytest.mark.parametrize('value', [None, True, '400', float('inf'), -1])
def test_measurement_template_invalid_value_stays_unknown(value):
    from vcg_biomarker.gui.lab_phrase_library import render_measurement_phrase
    result = render_measurement_phrase('{qt_interval}', {'delineation': {'interval_measurements': [
        {'interval': 'QT interval', 'value_ms': value, 'status': 'measurable'}]}})
    assert result['text'] == '[nicht verfügbar]'
    assert result['missing'] == ['qt_interval']


@pytest.mark.parametrize('template', ['{unknown}', '{qt_interval.__class__}', '{qt_interval!r}', '{qt_interval:.2f}'])
def test_measurement_template_rejects_unapproved_syntax(template):
    from vcg_biomarker.gui.lab_phrase_library import render_measurement_phrase
    with pytest.raises(ValueError):
        render_measurement_phrase(template, {})


def test_duplicate_measurements_are_not_silently_selected():
    from vcg_biomarker.gui.lab_phrase_library import render_measurement_phrase
    item = {'interval': 'QT interval', 'value_ms': 400., 'status': 'measurable'}
    result = render_measurement_phrase('{qt_interval}', {'delineation': {'interval_measurements': [item, item]}})
    assert result['missing'] == ['qt_interval']
