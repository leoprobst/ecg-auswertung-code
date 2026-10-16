"""Versioned display defaults only; never store clinical or signal state."""
import json

LAYOUTS = ('3x4', '6x2', '12x1', 'single')
LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
KEY = 'display_defaults_v1'


def decode_display_defaults(raw):
    if raw is None:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('version') != '1.0.0':
        raise ValueError('Unbekanntes Format der Anzeigevorgaben')
    choices = {'layout': LAYOUTS, 'lead': LEADS, 'order': ('standard', 'cabrera'),
               'standard_avr': ('native', 'inverted'), 'speed': ('12.5', '25', '50'), 'gain': ('5', '10', '20')}
    if set(value) != {'version', 'fit', *choices} or type(value.get('fit')) is not bool:
        raise ValueError('Unvollständige oder ungültige Anzeigevorgaben')
    if any(not isinstance(value[key], str) or value[key] not in allowed for key, allowed in choices.items()):
        raise ValueError('Ungültiger Anzeigewert')
    return value


def encode_display_defaults(value):
    raw = json.dumps(value, ensure_ascii=False)
    decode_display_defaults(raw)
    return raw
