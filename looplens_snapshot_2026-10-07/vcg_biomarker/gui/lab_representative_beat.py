"""Saved representative beat and explicitly requested research median template."""
import numpy as np
import hashlib
import json
from datetime import datetime, timezone
from numbers import Real


def median_export(template, signal_reference, analysis_id):
    """Full-precision local research artifact, not clinical measurements."""
    if not isinstance(template, dict) or template.get('available') is not True:
        raise ValueError('Kein berechnetes Median-Template vorhanden.')
    if not isinstance(analysis_id, str) or not analysis_id or not isinstance(signal_reference, dict) or not signal_reference.get('source_sha256'):
        raise ValueError('Signal- oder Analysereferenz fehlt; kein reproduzierbarer Export.')
    document = dict(schema_version='1.0.0', kind='exploratory_cluster_median',
        analysis_id=analysis_id, signal_reference=dict(signal_reference),
        clinical_claim_allowed=False, unit='mV', time_unit='ms',
        lead_order=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        coordinate_convention='native_leads; R-aligned analysis samples; no display inversion',
        values=template['signal'].tolist(), time_ms=template['time_ms'].tolist(),
        provenance={key: template[key] for key in ('used_beats', 'excluded_beats', 'method', 'cluster',
                    'path', 'sampling_rate_hz', 'amplitude_normalization', 'time_warping')},
        limitations=template['note'], fiducials=None)
    document['template_sha256'] = hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()
    return document


def manual_template_interval(template, signal_reference, analysis_id, start_ms, end_ms, label, author, lead_scope):
    """Sample-grid manual measurement on a synthetic template, not the record."""
    document = median_export(template, signal_reference, analysis_id)
    return _manual_interval_from_grid(template['time_ms'], template['sampling_rate_hz'],
        document['template_sha256'], analysis_id, start_ms, end_ms, label, author, lead_scope)


def _manual_interval_from_grid(times, sampling_rate_hz, template_hash, analysis_id,
                               start_ms, end_ms, label, author, lead_scope):
    """Internal numeric operation on a previously verified template context."""
    if label not in ('P', 'PR', 'QRS', 'QT', 'JT', 'freies Intervall'):
        raise ValueError('Unbekannte manuelle Messdefinition.')
    if not isinstance(author, str) or not author.strip():
        raise ValueError('Bearbeiter für manuelle Template-Messung erforderlich.')
    if lead_scope not in ('12-Kanal-Überlagerung', 'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'):
        raise ValueError('Messableitung fehlt.')
    if any(not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) for value in (start_ms, end_ms)) or not times[0] <= start_ms < end_ms <= times[-1]:
        raise ValueError('Intervallgrenzen außerhalb des Templates oder nicht aufsteigend.')
    # Nearest recorded grid point; exact ties resolve to the earlier sample.
    start = int(np.argmin(np.abs(times-start_ms)))
    end = int(np.argmin(np.abs(times-end_ms)))
    if start >= end:
        raise ValueError('Intervall kleiner als ein Sample; keine höhere Auflösung behaupten.')
    return dict(schema_version='1.0.0', kind='manual_synthetic_template_interval',
        template_sha256=template_hash, analysis_id=analysis_id,
        label=label, author=author.strip(), created_at=datetime.now(timezone.utc).isoformat(),
        lead_scope=lead_scope, start_sample=start, end_sample=end,
        start_ms=float(times[start]), end_ms=float(times[end]),
        duration_ms=(end-start)*1000/sampling_rate_hz,
        sampling_rate_hz=sampling_rate_hz, coordinate_system='synthetic_template_samples',
        method='manual_nearest_sample_earlier_tie', clinical_claim_allowed=False,
        warning='Manuelle Forschungstemplatemessung; keine validierte globale Delineation, kein Originalsignalintervall.')


def cluster_median(arrays, inventory, cluster, path='diagnostic_ecg'):
    """Samplewise median of accepted beats from one explicitly selected cluster.

    R-only alignment, no time warping, normalization, baseline substitution or
    inferred global fiducials. Bounded local display calculation for short ECGs.
    """
    unavailable = lambda reason: {'available': False, 'reason': reason}
    try:
        if not isinstance(arrays, dict) or path not in arrays:
            return unavailable('Gewählter Signalpfad nicht gespeichert.')
        raw_fs = np.asarray(arrays['sampling_rate_hz'])
        if raw_fs.size != 1 or raw_fs.dtype.kind not in 'fiu':
            raise ValueError('invalid fs')
        fs = float(raw_fs.reshape(-1)[0])
        signal = np.asarray(arrays[path])
        if not np.isfinite(fs) or fs <= 0 or signal.ndim != 2 or signal.shape[1] != 12 or signal.dtype.kind not in 'fiu':
            raise ValueError('invalid signal')
        if len(signal)/fs > 60 or fs > 2000:
            return unavailable('Lokale Medianansicht auf kurze EKGs ≤60 s und ≤2000 Hz begrenzt.')
        if not isinstance(inventory, list) or not inventory or len(inventory) > 100:
            return unavailable('Schlaginventar fehlt oder überschreitet 100 Schläge.')
        if cluster is None or isinstance(cluster, bool):
            return unavailable('Morphologiecluster ausdrücklich auswählen.')
        positions = []
        seen = set()
        for item in inventory:
            if not isinstance(item, dict):
                raise ValueError('invalid inventory')
            sample, index = item.get('r_peak_sample'), item.get('beat_index')
            if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or not 0 <= sample < len(signal):
                raise ValueError('invalid beat sample')
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0 or index in seen:
                raise ValueError('invalid beat identity')
            seen.add(index)
            positions.append((int(sample), item))
        positions.sort(key=lambda pair: pair[0])
        peaks = [sample for sample, _ in positions]
        if len(set(peaks)) != len(peaks):
            raise ValueError('duplicate R sample')
    except (KeyError, TypeError, ValueError, OverflowError):
        return unavailable('Signalzeitbasis oder Schlaginventar inkonsistent; keine Medianbildung.')
    selected, excluded = [], []
    for i, (peak, item) in enumerate(positions):
        reason = None
        if item.get('morphology_cluster_id') != cluster:
            reason = 'anderes_oder_unbekanntes_cluster'
        elif item.get('accepted') is not True:
            reason = 'nicht_akzeptiert'
        if reason:
            excluded.append({'beat_index': item['beat_index'], 'reason': reason})
        else:
            # Do not include neighbouring R complexes, even from other clusters.
            left = min(round(.35*fs), peak, (peak-peaks[i-1])//2 if i else peak)
            right = min(round(.65*fs), len(signal)-peak-1, (peaks[i+1]-peak-1)//2 if i+1 < len(peaks) else len(signal)-peak-1)
            selected.append((peak, item['beat_index'], left, right))
    if len(selected) < 3:
        return unavailable('Weniger als drei akzeptierte Schläge im gewählten Cluster; kein Median.')
    left, right = min(item[2] for item in selected), min(item[3] for item in selected)
    if left < 1 or right < 1:
        return unavailable('Kein gemeinsames beidseitiges Schlagfenster verfügbar.')
    windows, used = [], []
    for peak, index, _, _ in selected:
        window = signal[peak-left:peak+right+1]
        if not np.isfinite(window).all():
            excluded.append({'beat_index': index, 'reason': 'nicht_endliche_samples'})
        else:
            windows.append(window)
            used.append({'beat_index': index, 'r_peak_sample': peak})
    if len(used) < 3:
        return unavailable('Weniger als drei vollständige endliche Schlagfenster; kein Median.')
    stack = np.stack(windows)
    median = np.median(stack, axis=0)
    return dict(available=True, signal=median, time_ms=np.arange(-left, right+1)*1000/fs,
        fiducials_ms={}, sampling_rate_hz=fs, path=path, cluster=cluster,
        used_beats=used, excluded_beats=excluded, method='samplewise_median_r_aligned_v1',
        amplitude_normalization=False, time_warping=False,
        note=f'Exploratives Median-Template: {len(used)} akzeptierte Schläge, Cluster {cluster}. R-ausgerichtet; keine Amplitudennormierung. Gemeinsames Fenster kann P/T abschneiden. Cluster ist keine Schlagdiagnose. Keine übertragenen globalen Grenzen und keine klinischen Intervallmessungen.')


def representative_window(arrays, path='diagnostic_ecg'):
    """Return sample-accurate saved context around the recorded R fiducial."""
    if not isinstance(arrays, dict) or path not in arrays:
        return {'available': False, 'reason': 'Gewählter Signalpfad nicht gespeichert.'}
    try:
        fs_array = np.asarray(arrays['sampling_rate_hz'])
        if fs_array.size != 1 or fs_array.dtype.kind not in 'fiu':
            raise ValueError('invalid sampling rate')
        fs = float(fs_array.reshape(-1)[0])
        signal = np.asarray(arrays[path])
        names = np.asarray(arrays['fiducial_names'])
        samples = np.asarray(arrays['fiducial_samples'])
        if not np.isfinite(fs) or fs <= 0 or signal.ndim != 2 or signal.shape[1] != 12 or signal.dtype.kind not in 'fiu':
            raise ValueError('invalid signal')
        if names.ndim != 1 or samples.ndim != 1 or len(names) != len(samples) or len(set(map(str, names))) != len(names):
            raise ValueError('invalid fiducials')
        points = {}
        for name, sample in zip(names, samples):
            if isinstance(sample, (bool, np.bool_)) or not np.isfinite(sample) or float(sample) != int(sample) or not 0 <= sample < len(signal):
                raise ValueError('invalid sample')
            points[str(name)] = int(sample)
        r = points['r_peak']
    except (KeyError, TypeError, ValueError, OverflowError):
        return {'available': False, 'reason': 'Zeitbasis, Signal oder repräsentative Fiduzialpunkte fehlen / sind inkonsistent.'}
    # Display context, not inferred P/T boundaries. Never used for measurement.
    first = max(0, min([r-round(.35*fs), *points.values()]))
    last = min(len(signal), max([r+round(.65*fs)+1, *(value+1 for value in points.values())]))
    return {'available': True, 'signal': signal[first:last],
            'time_ms': (np.arange(first, last)-r)*1000/fs,
            'fiducials_ms': {name: (value-r)*1000/fs for name, value in points.items()},
            'sampling_rate_hz': fs, 'r_sample': r, 'path': path,
            'note': 'Gespeicherter repräsentativer Schlag mit Kontext; kein synthetischer Medianschlag. Zeit relativ zum gespeicherten R-Punkt. Keine neue Delineation.'}
