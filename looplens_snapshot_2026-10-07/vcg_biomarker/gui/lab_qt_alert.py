"""Recorded QTc review flag; not a real-time alarm or syndrome diagnosis."""
import math
from numbers import Real
import json

SOURCE = 'https://www.jacc.org/doi/10.1016/j.jacc.2010.01.001'


def qt_review_history(directory, reference, payload):
    """Read only matching acknowledgements; never dismiss the clinical flag."""
    from ..mimic.storage import content_hash
    expected = content_hash(qt_alert_items(payload))
    entries, errors = [], []
    for path in sorted(directory.glob('*.json')):
        try:
            if path.stat().st_size > 1_000_000:
                raise ValueError('Datei zu groß')
            item = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(item, dict) or item.get('schema_version') != '1.0.0' or item.get('kind') != 'qt_review_acknowledgement':
                raise ValueError('Unbekanntes Quittierungsformat')
            if item.get('signal_reference') != reference or item.get('analysis_id') != payload.get('analysis_id'):
                continue
            if item.get('items_sha256') != expected or content_hash(item.get('items')) != expected:
                raise ValueError('Messstand stimmt nicht überein')
            if not all(isinstance(item.get(k), str) and item[k].strip() for k in ('author', 'review_note', 'created_at')):
                raise ValueError('Unvollständige Quittierung')
            entries.append(item)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f'{path.name}: {exc}')
    return sorted(entries, key=lambda r: r['created_at']), errors


def qt_alert_items(payload):
    delineation = payload.get('delineation')
    rows = delineation.get('interval_measurements', []) if isinstance(delineation, dict) else []
    rows = rows if isinstance(rows, list) else []
    items = []
    for method in ('QTc Bazett', 'QTc Fridericia', 'QTc Framingham', 'QTc Hodges'):
        matches = [r for r in rows if isinstance(r, dict) and r.get('interval') == method]
        if len(matches) != 1:
            continue
        row = matches[0]
        value = row.get('value_ms')
        if row.get('status') == 'measurable' and isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 500:
            items.append({'method': method, 'value_ms': float(value), 'measurement': dict(row)})
    return items
