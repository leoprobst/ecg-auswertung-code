"""Validate visualization coordinates without altering or repairing signals."""
import numpy as np


def validate_visualization_arrays(arrays):
    rate = np.asarray(arrays.get('sampling_rate_hz'))
    if rate.ndim != 0 or rate.dtype.kind not in 'ifu' or not np.isfinite(rate) or float(rate) <= 0:
        raise ValueError('Visualisierung benötigt eine explizite positive Abtastrate')
    peaks = np.asarray(arrays.get('r_peaks', []))
    if (peaks.ndim != 1 or peaks.dtype.kind not in 'ifu'
            or not np.all(np.isfinite(peaks)) or np.any(peaks < 0)
            or np.any(peaks != np.floor(peaks)) or np.any(np.diff(peaks.astype(float)) <= 0)):
        raise ValueError('Ungültige oder ungeordnete R-Samplekoordinaten')
    lengths = set()
    for name, channels in (('diagnostic_ecg', 12), ('morphology_ecg', 12),
                           ('robust_morphology_ecg', 12), ('record_svd_components', 3)):
        if name not in arrays:
            continue
        signal = np.asarray(arrays[name])
        if signal.ndim != 2 or signal.shape[1] != channels or signal.shape[0] == 0 or signal.dtype.kind not in 'ifu':
            raise ValueError(f'Ungültige Signalform für {name}: Zeit × {channels} Kanäle erforderlich')
        lengths.add(signal.shape[0])
    if len(lengths) > 1:
        raise ValueError('Visualisierungssignale besitzen widersprüchliche Zeitachsen')
    if lengths and len(peaks) and peaks[-1] >= next(iter(lengths)):
        raise ValueError('R-Samplekoordinate liegt außerhalb des Visualisierungssignals')
    for name_key, coordinate_keys in (
        ('fiducial_names', ('fiducial_samples',)),
        ('segment_names', ('segment_start_samples', 'segment_end_samples')),
    ):
        fields = (name_key,) + coordinate_keys
        if not any(key in arrays for key in fields):
            continue
        if not all(key in arrays for key in fields):
            raise ValueError(f'Unvollständige Koordinatenfelder: {name_key}')
        names = np.asarray(arrays[name_key])
        if names.ndim != 1 or names.dtype.kind not in 'US' or any(not str(name).strip() for name in names):
            raise ValueError(f'Ungültige Namen: {name_key}')
        if name_key == 'fiducial_names' and len(set(names.tolist())) != len(names):
            raise ValueError('Doppelte repräsentative Landmarkennamen')
        for key in coordinate_keys:
            samples = np.asarray(arrays[key])
            if (samples.ndim != 1 or len(samples) != len(names) or samples.dtype.kind not in 'ifu'
                    or not np.all(np.isfinite(samples)) or np.any(samples < 0)
                    or np.any(samples != np.floor(samples))
                    or (lengths and np.any(samples >= next(iter(lengths))))):
                raise ValueError(f'Ungültige Samplezuordnung: {key}')
        if len(coordinate_keys) == 2 and np.any(np.asarray(arrays[coordinate_keys[1]]) < np.asarray(arrays[coordinate_keys[0]])):
            raise ValueError('Segmentende liegt vor Segmentbeginn')
    return float(rate), peaks
