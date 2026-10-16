"""Explicit same-signal device interval contract; no inferred patient linkage."""
import math
from numbers import Real

METHODS = ('QT interval', 'QRS duration', 'PR interval', 'QTc Bazett',
           'QTc Fridericia', 'QTc Framingham', 'QTc Hodges')


def validate_device_intervals(document, reference):
    if not isinstance(document, dict) or document.get('schema_version') != '1.0.0':
        raise ValueError('Geräteintervalle: Schema 1.0.0 erforderlich')
    if not isinstance(reference, dict) or any(
        not isinstance(document.get(key), str) or not document[key].strip()
        or document[key] != reference.get(key) for key in ('case_id', 'source_sha256')
    ):
        raise ValueError('Gerätewerte gehören nicht nachweislich zu diesem Signal')
    if not isinstance(document.get('device_source'), str) or not document['device_source'].strip():
        raise ValueError('Geräte-/Befundquelle fehlt')
    measurements = document.get('measurements')
    if not isinstance(measurements, list) or not measurements or len(measurements) > len(METHODS):
        raise ValueError('Ungültige Geräteintervallliste')
    seen = set()
    for row in measurements:
        if not isinstance(row, dict) or row.get('interval') not in METHODS or row['interval'] in seen:
            raise ValueError('Unbekannte Formel/Intervallart oder doppelter Gerätewert')
        value = row.get('value_ms')
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError('Gerätewert muss positiv, endlich und in ms angegeben sein')
        if row.get('unit') != 'ms':
            raise ValueError('Explizite Einheit ms erforderlich; keine stillschweigende Umrechnung')
        seen.add(row['interval'])
    return document


def device_interval_comparison(payload, document, reference):
    validate_device_intervals(document, reference)
    delineation = payload.get('delineation', {}) if isinstance(payload, dict) else {}
    rows = delineation.get('interval_measurements', []) if isinstance(delineation, dict) else []
    rows = rows if isinstance(rows, list) else []
    result = []
    for device in document['measurements']:
        matches = [r for r in rows if isinstance(r, dict) and r.get('interval') == device['interval']]
        own = matches[0].get('value_ms') if len(matches) == 1 and matches[0].get('status') == 'measurable' else None
        valid = isinstance(own, Real) and not isinstance(own, bool) and math.isfinite(own) and own > 0
        result.append(dict(interval=device['interval'], own_ms=float(own) if valid else None,
            device_ms=device['value_ms'], difference_ms=float(own-device['value_ms']) if valid else None,
            status='descriptive_difference_not_accuracy_validation' if valid else 'own_measurement_unavailable',
            device_source=document['device_source']))
    return result
