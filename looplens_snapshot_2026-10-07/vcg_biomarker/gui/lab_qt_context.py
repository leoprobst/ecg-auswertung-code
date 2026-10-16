"""Interpretation context for recorded measurements, never a QT recalculation."""
import math
from numbers import Real

SOURCE = 'https://www.jacc.org/doi/10.1016/j.jacc.2008.12.014'
SQTS_SOURCE = 'https://doi.org/10.1093/eurheartj/ehac262'


def qt_context(intervals, *, age=None, sex=None, rhythm=None):
    """Preserve unknown/duplicate measurements; do not infer a syndrome."""
    rows = intervals if isinstance(intervals, list) else []

    def measurement(name):
        matches = [r for r in rows if isinstance(r, dict) and r.get('interval') == name]
        if len(matches) != 1 or matches[0].get('status') != 'measurable':
            return None
        value = matches[0].get('value_ms')
        return matches[0] if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else None

    notes = []
    qrs = measurement('QRS duration')
    if qrs is None:
        notes.append('QRS-Dauer unbekannt: Einfluss der Depolarisation auf QT nicht beurteilbar.')
    elif qrs['value_ms'] >= 120:
        notes.append('Breiter QRS: verlängertes QT kann durch verlängerte Depolarisation entstehen. '
                     'JT/JTc und Messmethode gesondert prüfen; keine automatische BBB-Korrektur oder LQTS-Diagnose.')
    bazett = measurement('QTc Bazett')
    if bazett:
        rr = bazett.get('rr_seconds_median')
        if not isinstance(rr, Real) or isinstance(rr, bool) or not math.isfinite(rr) or rr <= 0:
            notes.append('Bazett: zugehöriges RR unbekannt; Frequenzabhängigkeit nicht überprüfbar.')
        else:
            rate = 60 / rr
            if rate < 60 or rate > 100:
                notes.append(f'Bazett: zugehöriges RR entspricht {rate:.1f}/min; außerhalb 60–100/min '
                             'verstärkte Frequenzabhängigkeit beachten und andere QTc-Formeln vergleichen. '
                             'Dieser Bereich ist ein lokaler Warnbereich, keine diagnostische Grenze.')
    classifications = []
    sex_code = str(sex or '').upper()
    adult = isinstance(age, Real) and not isinstance(age, bool) and math.isfinite(age) and age >= 18
    cv = rhythm.get('rr_cv') if isinstance(rhythm, dict) else None
    stable_rr = isinstance(cv, Real) and not isinstance(cv, bool) and math.isfinite(cv) and 0 <= cv <= .1
    for name in ('QTc Framingham', 'QTc Hodges'):
        measured = measurement(name)
        reasons = []
        if not adult:
            reasons.append('Erwachsenenalter nicht belegt')
        if sex_code not in {'M', 'MALE', 'F', 'FEMALE'}:
            reasons.append('Geschlechtsreferenz unbekannt')
        if qrs is None or qrs['value_ms'] >= 120:
            reasons.append('schmaler QRS nicht belegt')
        if not stable_rr:
            reasons.append('geringe RR-Variabilität nicht belegt (lokales Gate: RR-CV ≤0,10)')
        if measured is None:
            reasons.append('Messung fehlt oder nicht eindeutig auswertbar')
        result = {'method': name, 'status': 'not_evaluable', 'reasons': reasons,
                  'value_ms': measured['value_ms'] if measured else None,
                  'source': SOURCE, 'diagnosis': None}
        if not reasons:
            threshold = 450 if sex_code in {'M', 'MALE'} else 460
            value = measured['value_ms']
            status = 'prolonged' if value >= threshold else 'short' if value <= 390 else 'within_reference_limits'
            result.update(status=status, upper_threshold_ms=threshold, lower_threshold_ms=390)
            label = {'prolonged': 'verlängert', 'short': 'verkürzt',
                     'within_reference_limits': 'innerhalb dieser Referenzgrenzen'}[status]
            notes.append(f'{name}: {value:.6g} ms – {label} (Erwachsenenreferenz AHA 2009; '
                         f'verlängert ≥{threshold} ms, verkürzt ≤390 ms). Kein Syndromnachweis oder -ausschluss.')
        else:
            notes.append(f'{name}: Referenzeinordnung nicht beurteilbar – ' + '; '.join(reasons) + '.')
        classifications.append(result)
    notes.append('QTc-Formeln sind nicht austauschbar. Medikamente, Elektrolyte, Rhythmus und '
                 'T-Ende-Messung prüfen; ein QTc-Wert allein bestätigt oder widerlegt kein QT-Syndrom.')
    # Reference text, not an automatic threshold applied indiscriminately to
    # all correction formulae. Clinical SQTS inputs are not present here.
    short_qt_context = {
        'status': 'clinical_context_not_assessed', 'diagnosis': None,
        'source': SQTS_SOURCE, 'recommendation_table': 46,
        'automatic_threshold_applied': False,
        'required_review': ['QT/RR measurement and correction method',
                            'pathogenic variant', 'family history of SQTS',
                            'survived VT/VF and absence of heart disease',
                            'arrhythmic syncope', 'acquired QT shortening'],
    }
    notes.append('Kurz-QT-Kontext (ESC 2022, Tabelle 46): QTc ≤320 ms ist Anlass, '
                 'ein Short-QT-Syndrom zu erwägen (IIa); QTc ≤360 ms zusammen mit '
                 'pathogener Mutation, familiärem SQTS oder überlebter VT/VF ohne '
                 'Herzerkrankung begründet den diagnostischen Kriterienarm (I). '
                 'Bei QTc 320–360 ms ist eine arrhythmische Synkope zusätzlicher Abklärungskontext. '
                 'Diese Anamnese-/Genetik-/Strukturbefunde sind hier nicht geprüft. '
                 'Keine automatische Anwendung auf alle QTc-Formeln; die AHA-Referenzgrenze '
                 '390 ms ist keine SQTS-Diagnosegrenze. Quelle: ' + SQTS_SOURCE)
    return {'notes': notes, 'classifications': classifications, 'source': SOURCE,
            'diagnosis': None, 'recalculated': False, 'short_qt_context': short_qt_context}
