"""Separate, non-rate-corrected modified QT for documented LBBB."""
import math
from numbers import Real
from .ischemia_patterns import validate_lab_ischemia_context


def bogossian_modified_qt(intervals, context, case_id, *, age=None):
    result = dict(interval='QTm Bogossian 2014 (nicht frequenzkorrigiert)',
        value_ms=None, status='not_evaluable', reason='',
        measurement_method='same_representative_beat_qt_minus_0_485_qrs',
        formula='QTm_ms = QT_ms - 0.485 * QRS_ms',
        source='https://pubmed.ncbi.nlm.nih.gov/25149024/',
        definition_version='Bogossian_2014_original_0.485',
        interpretation_note='Modifiziertes QT, nicht QTc. Keine Normklassifikation oder LQTS-Diagnose. '
            'Keine Kombination mit RR-Median anderer Schläge. Original-QT/QTc bleiben unverändert.')
    if isinstance(age, bool) or not isinstance(age, Real) or not math.isfinite(age) or age < 18:
        result['reason'] = 'adult_age_not_documented_no_pediatric_extrapolation'
        return result
    try:
        verified = validate_lab_ischemia_context(context, case_id)
    except (ValueError, TypeError):
        verified = None
    if not verified or verified['context'] != 'lbbb':
        result['reason'] = 'documented_case_bound_lbbb_required_not_generic_wide_qrs_or_pacing'
        return result
    values = []
    for name in ('QT interval', 'QRS duration'):
        matches = [item for item in intervals if isinstance(item, dict) and item.get('interval') == name]
        if len(matches) != 1 or matches[0].get('status') != 'measurable':
            result['reason'] = 'missing_ambiguous_or_unmeasurable_qt_qrs'
            return result
        row = matches[0]
        value = row.get('value_ms')
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            result['reason'] = 'invalid_interval'
            return result
        start, end, rate = row.get('start_sample'), row.get('end_sample'), row.get('sampling_rate_hz')
        if (type(start) is not int or type(end) is not int or not 0 <= start < end
                or isinstance(rate, bool) or not isinstance(rate, Real) or not math.isfinite(rate) or rate <= 0
                or not math.isclose(value, (end-start)*1000/rate, rel_tol=1e-9, abs_tol=1e-6)):
            result['reason'] = 'interval_value_and_signal_coordinates_inconsistent'
            return result
        values.append(row)
    qt, qrs = values
    identity = ('measurement_method', 'representative_r_sample', 'sampling_rate_hz', 'start_sample')
    if any(qt.get(key) is None or qt.get(key) != qrs.get(key) for key in identity) or qt['measurement_method'] != 'representative_beat_fiducials':
        result['reason'] = 'common_beat_and_onset_not_documented'
        return result
    if qrs['value_ms'] < 120 or qt['value_ms'] <= qrs['value_ms']:
        result['reason'] = 'incompatible_lbbb_intervals'
        return result
    if type(qt['representative_r_sample']) is not int or not qrs['start_sample'] <= qt['representative_r_sample'] <= qrs['end_sample']:
        result['reason'] = 'representative_r_outside_qrs'
        return result
    result.update(value_ms=float(qt['value_ms']-.485*qrs['value_ms']), status='measurable',
        input_qt_ms=qt['value_ms'], input_qrs_ms=qrs['value_ms'],
        context_revision_id=verified['revision_id'],
        representative_r_sample=qt['representative_r_sample'], sampling_rate_hz=qt['sampling_rate_hz'])
    return result
