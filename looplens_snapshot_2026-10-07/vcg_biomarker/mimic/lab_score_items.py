"""Source-versioned single score items, NOT complete diagnostic scores."""
from __future__ import annotations

import math

LQTS_SOURCE = 'https://academic.oup.com/view-large/553238290'


def lqts_qtc_item(intervals, sex=None):
    """ESC 2022 Table 10 QTc item only, using Bazett, without rounding bins.

    Never turn an incomplete ECG-only score into a low-risk classification.
    Clinical eligibility (including secondary causes) remains unconfirmed.
    """
    result = {'score_id': 'modified_lqts_esc_2022', 'item_id': 'resting_qtc_bazett',
              'source': LQTS_SOURCE, 'definition_version': 'ESC_2022_table_10',
              'points': None, 'status': 'not_evaluable', 'total_score': None,
              'diagnosis': None, 'probability': None,
              'remaining_items': ['exercise_recovery', 'torsades', 't_wave_alternans',
                                  'notched_t_three_leads', 'age_adjusted_bradycardia',
                                  'syncope', 'family_history', 'pathogenic_variant'],
              'clinical_eligibility': 'not_verified',
              'warning_de': 'Nur QTc-Einzelitem, kein vollständiger LQTS-Score. Medikamente, Elektrolyte, '
                            'Messkonventionen und klinische Vorgeschichte prüfen. Auch 0 Punkte schließen LQTS nicht aus.'}
    matches = [r for r in intervals if isinstance(r, dict) and r.get('interval') == 'QTc Bazett']
    if len(matches) != 1 or matches[0].get('status') != 'measurable':
        result['reason'] = 'missing_ambiguous_or_unmeasurable_bazett'
        return result
    value = matches[0].get('value_ms')
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        result['reason'] = 'invalid_qtc'
        return result
    result['qtc_ms'] = value
    if value >= 480:
        points = 3.5
    elif value >= 460:
        points = 2.
    elif value >= 450:
        code = str(sex or '').upper()
        if code not in {'M', 'MALE', 'F', 'FEMALE'}:
            result['reason'] = 'sex_required_for_450_to_460_ms'
            return result
        points = 1. if code in {'M', 'MALE'} else 0.
    else:
        points = 0.
    result.update(points=points, status='partial_item_requires_review', reason='other_items_and_secondary_causes_unknown')
    return result
