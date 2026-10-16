"""Validate stored measurement coordinates; never reconstruct missing windows."""
import math


DIRECT8_LEADS = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

BEAT_REASON_LABELS = {
    'included': 'Verwendet', 'rhythm_no_discrete_p': 'Keine diskrete P-Welle für PR-Baseline',
    'record_signal_unavailable_or_incompatible': 'Vollständiges Signal fehlt / inkompatibel',
    'r_peak_order_invalid': 'R-Peaks nicht streng zeitlich geordnet',
    'not_primary_morphology_cluster': 'Nicht im primären Morphologiecluster',
    'segment_quality_below_threshold': 'Segmentqualität unter der Messgrenze',
    'preqrs_window_outside_record': 'Pre-QRS-Fenster außerhalb der Aufzeichnung',
    'baseline_short_or_unstable': 'Baseline zu kurz oder instabil',
    'segment_boundaries': 'Ungültige Segmentgrenzen', 'signal_nonfinite': 'Ungültige Signalwerte',
    'degenerate_area_vector': 'Degenerierter Flächenvektor',
    'integral_unavailable_or_degenerate': 'Integralwinkel nicht auswertbar / degeneriert',
    'reason_unavailable': 'Grund nicht gespeichert',
}


def beat_audit_rows(contract):
    if (not isinstance(contract, dict) or contract.get('schema_version') != '1.1.0'
            or contract.get('scope') != 'beat_summary'):
        return []
    rows = contract.get('beat_audit')
    if not isinstance(rows, list):
        return []
    return [item for item in rows[:2000] if isinstance(item, dict)
            and type(item.get('beat_index')) is int and item['beat_index'] >= 0
            and type(item.get('included')) is bool and isinstance(item.get('reason'), str)]


def validated_direct8_windows(contract, sample_count):
    if not isinstance(contract, dict):
        return None
    if contract.get('schema_version') not in (None, '1.0.0'):
        return None
    if (contract.get('signal_path') not in ('robust_morphology_ecg', 'morphology_ecg')
            or contract.get('coordinate_system') != 'analysis_samples_inclusive'
            or contract.get('leads') != DIRECT8_LEADS):
        return None
    fs = contract.get('sampling_rate_hz')
    if isinstance(fs, bool) or not isinstance(fs, (int, float)) or not math.isfinite(fs) or fs <= 0:
        return None
    baseline_key = 'baseline' if contract.get('schema_version') == '1.0.0' else 'pr_baseline'
    if baseline_key == 'baseline' and contract.get('baseline_kind') not in ('preqrs', 'tp', 'pr'):
        return None
    bounds = {}
    for key in (baseline_key, 'qrs', 'j_to_t_end'):
        pair = contract.get(key)
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in pair)
                or not 0 <= pair[0] < pair[1] < sample_count):
            return None
        bounds[key] = pair
    if bounds['qrs'][1] != bounds['j_to_t_end'][0]:
        return None
    return fs, baseline_key, bounds


def beat_window_items(contract):
    """Return only versioned stored beat windows, never synthesize a representative beat."""
    if (not isinstance(contract, dict) or contract.get('schema_version') != '1.1.0'
            or contract.get('scope') != 'beat_summary' or contract.get('signal_path') != 'morphology_ecg'):
        return []
    items = contract.get('items')
    if not isinstance(items, list):
        return []
    return [item for item in items[:2000] if isinstance(item, dict)
            and isinstance(item.get('beat_index'), int) and not isinstance(item.get('beat_index'), bool)
            and item['beat_index'] >= 0 and isinstance(item.get('value'), (float, int))
            and not isinstance(item.get('value'), bool) and math.isfinite(item['value'])]
