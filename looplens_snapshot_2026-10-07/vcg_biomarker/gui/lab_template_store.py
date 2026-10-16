"""Append-only local measurement events, bound to an exact synthetic template."""
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .lab_representative_beat import median_export, _manual_interval_from_grid
from ..mimic.storage import atomic_write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


class TemplateMeasurementStore:
    def __init__(self, root, template, reference, analysis_id):
        self.analysis_id = analysis_id
        self.template_hash = median_export(template, reference, analysis_id)['template_sha256']
        self._times = np.array(template['time_ms'], dtype=float, copy=True)
        self._times.setflags(write=False)
        self._sampling_rate = float(template['sampling_rate_hz'])
        self.root = Path(root) / 'template_measurements' / self.template_hash

    def _validate_measurement(self, item):
        if not isinstance(item, dict):
            raise ValueError('Ungültige Template-Messung.')
        try:
            if any(type(item.get(key)) is not int for key in ('start_sample', 'end_sample')):
                raise ValueError('Samplekoordinaten müssen ganze Zahlen sein.')
            expected = _manual_interval_from_grid(self._times, self._sampling_rate, self.template_hash, self.analysis_id,
                item['start_ms'], item['end_ms'], item['label'], item['author'], item['lead_scope'])
            for key, value in expected.items():
                if key != 'created_at' and item.get(key) != value:
                    raise ValueError('Messung stimmt nicht mit Template/Koordinaten überein.')
            stamp = datetime.fromisoformat(item['created_at'])
            if stamp.tzinfo is None:
                raise ValueError('Messzeitpunkt ohne Zeitzone.')
        except (KeyError, TypeError) as exc:
            raise ValueError('Unvollständige Template-Messung.') from exc

    def events(self):
        events = []
        for path in sorted(self.root.glob('*.json')):
            try:
                envelope = json.loads(path.read_text(encoding='utf-8'))
                event = envelope['event']
                if envelope['sha256'] != digest(event):
                    raise ValueError('Integritätsprüfung fehlgeschlagen.')
                if event['schema_version'] != '1.0.0' or event['template_sha256'] != self.template_hash:
                    raise ValueError('Inkompatibler Template-Verlauf.')
                if not re.fullmatch(r'[0-9a-f]{32}', event['event_id']) or path.stem != event['event_id']:
                    raise ValueError('Ungültige Ereigniskennung.')
                if datetime.fromisoformat(event['created_at']).tzinfo is None:
                    raise ValueError('Ereigniszeit ohne Zeitzone.')
                if event['operation'] == 'add':
                    self._validate_measurement(event['measurement'])
                elif event['operation'] == 'withdraw':
                    if any(not isinstance(event.get(key), str) or not event[key].strip() for key in ('author', 'reason')):
                        raise ValueError('Rücknahme ohne Bearbeiter/Begründung.')
                    if not isinstance(event.get('target_event_id'), str):
                        raise ValueError('Rücknahme ohne Zielkennung.')
                else:
                    raise ValueError('Unbekannte Verlaufsoperation.')
                events.append(event)
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f'Ungültiger lokaler Messverlauf: {path.name}') from exc
        adds = {event['event_id'] for event in events if event['operation'] == 'add'}
        if any(event['target_event_id'] not in adds for event in events if event['operation'] == 'withdraw'):
            raise ValueError('Rücknahme verweist auf eine fehlende Messung.')
        return sorted(events, key=lambda event: (event['created_at'], event['event_id']))

    def active(self):
        events = self.events()
        withdrawn = {event['target_event_id'] for event in events if event['operation'] == 'withdraw'}
        return [dict(event['measurement'], local_event_id=event['event_id']) for event in events
                if event['operation'] == 'add' and event['event_id'] not in withdrawn]

    def _append(self, **content):
        event = dict(schema_version='1.0.0', template_sha256=self.template_hash,
            event_id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc).isoformat(), **content)
        atomic_write_json(self.root / (event['event_id'] + '.json'), {'event': event, 'sha256': digest(event)})

    def add(self, measurement):
        self.events()  # Never silently append to a damaged history.
        self._validate_measurement(measurement)
        self._append(operation='add', measurement=dict(measurement))
        return self.active()

    def withdraw(self, event_id, author, reason):
        if not isinstance(author, str) or not author.strip() or not isinstance(reason, str) or not reason.strip():
            raise ValueError('Bearbeiter und Begründung erforderlich.')
        if event_id not in {item['local_event_id'] for item in self.active()}:
            raise ValueError('Messung ist nicht aktiv.')
        self._append(operation='withdraw', target_event_id=event_id, author=author.strip(), reason=reason.strip())
        return self.active()
