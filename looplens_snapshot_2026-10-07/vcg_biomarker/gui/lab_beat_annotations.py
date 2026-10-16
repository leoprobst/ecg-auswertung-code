"""Read local beat reviews without treating malformed files as evidence."""
import json
import math
import logging

LOGGER = logging.getLogger(__name__)


def read_beat_labels(directory, reference, analysis_id):
    annotations, errors = [], []
    for path in sorted(directory.glob('*.json')):
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(item, dict):
                raise ValueError('Annotation ist kein Objekt')
            if item.get('signal') != reference or item.get('analysis_id') != analysis_id:
                continue
            if item.get('schema_version') not in {'1.0.0', '1.1.0'}:
                raise ValueError('Unbekannte Annotationsversion')
            if not isinstance(item.get('label'), str) or not item['label'].strip():
                raise ValueError('Schlaglabel fehlt')
            if item['schema_version'] == '1.1.0':
                sample = item.get('native_r_sample')
                if not isinstance(item.get('author'), str) or not item['author'].strip():
                    raise ValueError('Bearbeiter fehlt')
            else:
                seconds = item.get('r_time_seconds')
                if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds):
                    raise ValueError('Ungültige historische Schlagzeit')
                sample = round(seconds * reference['sampling_rate_hz'])
            if type(sample) is not int or not 0 <= sample < reference['sample_count']:
                raise ValueError('Schlag außerhalb des Signals')
            timestamp = item.get('created_at')
            if not isinstance(timestamp, str):
                raise ValueError('Zeitstempel fehlt')
            annotations.append((timestamp, path.name, sample, item['label']))
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f'{path.name}: {exc}')
            LOGGER.warning('Local beat annotation rejected: %s: %s', path.name, exc)
    labels = {sample: label for _, _, sample, label in sorted(annotations)}
    return labels, errors
