"""User-authored reusable prose, not an automatic diagnostic statement engine."""
import json
import math
from numbers import Real
from string import Formatter

from ..mimic.lab_workspace import REPORT_SECTIONS

VERSION = '1.0.0'
MEASUREMENT_VARIABLES = {
    'p_duration': 'P duration', 'pr_interval': 'PR interval',
    'qrs_duration': 'QRS duration', 'qt_interval': 'QT interval', 'jt_interval': 'JT interval',
    'qtc_bazett': 'QTc Bazett', 'qtc_fridericia': 'QTc Fridericia',
    'qtc_framingham': 'QTc Framingham', 'qtc_hodges': 'QTc Hodges',
}


def render_measurement_phrase(template, payload):
    """Restricted placeholders, stored measurements only, no Python evaluation."""
    parsed = list(Formatter().parse(template))
    delineation = payload.get('delineation')
    intervals = delineation.get('interval_measurements', []) if isinstance(delineation, dict) else []
    intervals = intervals if isinstance(intervals, list) else []
    output, used, missing = [], {}, []
    for literal, variable, spec, conversion in parsed:
        output.append(literal)
        if variable is None:
            continue
        if variable not in MEASUREMENT_VARIABLES or spec or conversion:
            raise ValueError(f'Unbekannte oder unzulässig formatierte Variable: {variable}')
        matches = [item for item in intervals if isinstance(item, dict)
                   and item.get('interval') == MEASUREMENT_VARIABLES[variable]]
        item = matches[0] if len(matches) == 1 else {}
        value = item.get('value_ms')
        valid = (item.get('status') == 'measurable' and isinstance(value, Real)
                 and not isinstance(value, bool) and math.isfinite(value) and value >= 0)
        if not valid:
            missing.append(variable)
            output.append('[nicht verfügbar]')
        else:
            output.append(f'{value:.6g}')
            used[variable] = dict(item)
    return {'text': ''.join(output), 'measurements': used, 'missing': list(dict.fromkeys(missing)),
            'analysis_id': payload.get('analysis_id'), 'unit': 'ms',
            'note': 'Automatische gespeicherte Werte; keine manuellen Überschreibungen oder Neuberechnung.'}


def decode_phrases(raw):
    if raw is None:
        return []
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema_version') != VERSION:
        raise ValueError('Unbekanntes Bausteinformat')
    entries = value.get('entries')
    if not isinstance(entries, list) or len(entries) > 200:
        raise ValueError('Ungültige Bausteinliste')
    seen = set()
    for item in entries:
        if not isinstance(item, dict) or set(item) != {'section', 'text'}:
            raise ValueError('Ungültiger Baustein')
        section, text = item['section'], item['text']
        if not isinstance(section, str) or section not in REPORT_SECTIONS:
            raise ValueError('Unbekannter Befundabschnitt')
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError('Bausteintext fehlt oder ist zu lang')
        key = (section, text)
        if key in seen:
            raise ValueError('Doppelter Baustein')
        seen.add(key)
    return entries


def encode_phrases(entries):
    raw = json.dumps({'schema_version': VERSION, 'entries': entries}, ensure_ascii=False)
    decode_phrases(raw)
    return raw


def add_phrase(entries, section, text):
    item = {'section': section, 'text': text.strip()}
    updated = list(entries)
    if item not in updated:
        updated.append(item)
    encode_phrases(updated)
    return updated


def change_phrase(entries, section, original, replacement=None):
    """Exact scoped target; stale selection cannot edit another template."""
    encode_phrases(entries)
    target = {'section': section, 'text': original}
    if target not in entries:
        raise ValueError('Vorlage nicht mehr vorhanden; Bibliothek neu laden')
    updated = [dict(item) for item in entries]
    index = updated.index(target)
    if replacement is None:
        updated.pop(index)
    else:
        updated[index] = {'section': section, 'text': replacement.strip()}
    encode_phrases(updated)
    return updated
