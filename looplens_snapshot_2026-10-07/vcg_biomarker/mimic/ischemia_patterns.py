"""Source-bound morphology components; never automatic treatment advice."""
import math

from ..core.data_loader import LEAD_NAMES_12

SMITH_SOURCE = 'https://pubmed.ncbi.nlm.nih.gov/22939607/'
PACING_SOURCE = 'https://pubmed.ncbi.nlm.nih.gov/34172301/'
DIFFUSE_ST_SOURCE = 'https://academic.oup.com/eurheartj/article/44/38/3720/7243210'


def diffuse_st_pattern(st, *, quality_ok, excluded=False):
    """ESC 2023 pattern, operationalized at the baseline-referenced J point.

    This is a morphology screen, not a culprit-vessel or ACS diagnosis.
    The source specifies no numeric minimum for the accompanying elevation;
    strictly positive J values are used and must be inspected for noise.
    """
    from numbers import Real
    result = dict(evaluable=False, positive=None, depressed_leads=[], elevated_leads=[],
                  missing_leads=[], source=DIFFUSE_ST_SOURCE, reason='',
                  clinical_diagnosis=False, definition_version='lab_diffuse_st_j_v1')
    if quality_ok is not True or excluded is not False:
        result['reason'] = 'Reliable J point/baseline required; LBBB or pacing excludes this standard ST screen'
        return result
    for lead in LEAD_NAMES_12:
        item = st.get(lead) if isinstance(st, dict) else None
        value = item.get('j') if isinstance(item, dict) else None
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            result['missing_leads'].append(lead)
            continue
        if value <= -.1:
            result['depressed_leads'].append(lead)
        if lead in ('aVR', 'V1') and value > 0:
            result['elevated_leads'].append(lead)
    if len(result['depressed_leads']) >= 6 and result['elevated_leads']:
        result.update(evaluable=True, positive=True)
    elif result['missing_leads']:
        result['reason'] = 'Incomplete leads: no negative classification from missing data'
    else:
        result.update(evaluable=True, positive=False)
    return result


def validate_lab_ischemia_context(context, case_id):
    if context is None:
        return None
    if not isinstance(context, dict) or context.get('case_id') != case_id:
        raise ValueError('Clinical context must belong to the selected case')
    if context.get('context') not in ('lbbb', 'ventricular_pacing') or context.get('schema_version') != '1.0.0':
        raise ValueError('Unsupported clinical context')
    if not all(isinstance(context.get(k), str) and context[k].strip()
               for k in ('author', 'reason', 'created_at', 'revision_id')):
        raise ValueError('Clinical context requires author, reason, time and revision')
    return dict(context)


def modified_sgarbossa(qrs, st, *, context, quality_ok):
    """Smith 2012 LBBB rule, with explicit applicability and missingness.

    Context is deliberately not inferred from wide QRS or a generic pacing
    flag. Ventricular pacing uses the separate Dodd 2021 study reference;
    its exploratory V4-V6 depression extension is not included.
    Inputs: baseline-referenced J point, nonnegative R height/S depth in mV.
    """
    result = {'schema_version': '1.0.0', 'positive': None, 'evaluable': False,
              'reason': '', 'triggered': [], 'missing_leads': [],
              'source': PACING_SOURCE if context == 'ventricular_pacing' else SMITH_SOURCE, 'context': context,
              'clinical_diagnosis': False, 'treatment_recommendation': None}
    if context not in ('lbbb', 'ventricular_pacing'):
        result['reason'] = 'Documented LBBB or ventricular paced morphology required; generic device presence is insufficient'
        return result
    if quality_ok is not True:
        result['reason'] = 'Reliable QRS bounds, J point and baseline required'
        return result
    for lead in LEAD_NAMES_12:
        try:
            r, s, j = qrs[lead]['r_height'], qrs[lead]['s_depth'], st[lead]['j']
            if not all(math.isfinite(v) for v in (r, s, j)) or min(r, s) < 0 or r == s:
                raise ValueError('QRS polarity unavailable')
        except (KeyError, TypeError, ValueError):
            result['missing_leads'].append(lead)
            continue
        arms = []
        if r > s and j >= .1:
            arms.append('concordant_st_elevation')
        if s > r and lead in ('V1', 'V2', 'V3') and j <= -.1:
            arms.append('concordant_anterior_st_depression')
        if s > r and j >= .1 and j >= .25 * s:
            arms.append('proportional_discordant_st_elevation')
        for arm in arms:
            result['triggered'].append({'component': arm, 'lead': lead,
                'j_mv': float(j), 'r_mv': float(r), 's_depth_mv': float(s),
                'signed_st_s_ratio': -j/s if s > 0 else None})
    if result['triggered']:
        result.update(positive=True, evaluable=True)
    elif result['missing_leads']:
        result['reason'] = 'Incomplete morphology: absence of a positive component is not a negative result'
    else:
        result.update(positive=False, evaluable=True)
    return result
