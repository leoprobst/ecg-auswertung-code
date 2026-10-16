"""Explain recorded technical rhythm classes without diagnostic upgrading."""
import math
from numbers import Real

RHYTHM_TEXT = {
    'regular': 'Regelmäßiges RR-Muster; Sinusursprung nicht allein dadurch belegt.',
    'irregular': 'Unregelmäßiges RR-Muster; Ursache nicht bestimmt.',
    'af_compatible_pattern': 'Technisches Muster mit Vorhofflimmern vereinbar; keine bestätigte AF-Diagnose.',
    'flutter_compatible_pattern': 'Technisches Muster mit Vorhofflattern vereinbar; keine bestätigte Flutter-Diagnose.',
    'multiple_atrial_activity': 'Mehrfache atriale Aktivitätskandidaten; Rhythmus und Überleitung manuell prüfen.',
    'wide_qrs_pattern': 'Breites QRS-Muster; keine automatische Zuordnung zu VT, Schenkelblock oder Schrittmacher.',
    'mixed_or_ectopic_pattern': 'Gemischtes/ektopiekompatibles Muster; keine gesicherte supraventrikuläre oder ventrikuläre Schlagdiagnose.',
    'unknown': 'Rhythmus technisch nicht zugeordnet; nicht als Sinusrhythmus oder Normalbefund interpretieren.',
}


def rhythm_review(payload):
    delineation = payload.get('delineation')
    delineation = delineation if isinstance(delineation, dict) else {}
    context = delineation.get('rhythm_context')
    context = context if isinstance(context, dict) else {}
    kind = context.get('class')
    known = isinstance(kind, str) and kind in RHYTHM_TEXT
    text = RHYTHM_TEXT[kind] if known else 'Kein kompatibler gespeicherter Rhythmuskontext vorhanden.'

    def number(key, lower=None, upper=None):
        value = context.get(key)
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
            return None
        if lower is not None and value < lower or upper is not None and value > upper:
            return None
        return float(value)

    rr = number('rr_median_ms', lower=0)
    rate = 60000/rr if rr and math.isfinite(60000/rr) else None
    confidence = number('confidence', 0, 1)
    coverage = number('p_wave_coverage', 0, 1)
    flags = context.get('flags')
    flags = [flag for flag in flags if isinstance(flag, str)] if isinstance(flags, list) else []
    restrictions = []
    if kind in ('af_compatible_pattern', 'flutter_compatible_pattern', 'multiple_atrial_activity'):
        restrictions.append('Einzelne P-/PR-Messungen nicht ohne gesonderte Prüfung übernehmen.')
    if kind in ('irregular', 'af_compatible_pattern', 'flutter_compatible_pattern', 'multiple_atrial_activity', 'mixed_or_ectopic_pattern'):
        restrictions.append('QT-Korrektur und Schlagmittelung benötigen dokumentierte schlagweise QT/RR-Paare.')
    if delineation.get('review_required') is True:
        restrictions.append('Gespeicherte Delineation fordert eine manuelle Prüfung.')
    display = (text + ' Frequenz aus medianem RR: ' + (f'{rate:.1f}/min' if rate is not None else 'unbekannt')
               + '; kein separat bestimmter Sinusknotenrhythmus. Technische Konfidenz: '
               + (f'{confidence:.3g}' if confidence is not None else 'nicht dokumentiert')
               + ' (keine Erkrankungswahrscheinlichkeit). ' + ' '.join(restrictions))
    return dict(schema_version='1.0.0', statement_code='technical_rhythm.' + (kind if known else 'unavailable'),
                text=display, recorded_class=kind, rate_from_median_rr_bpm=rate,
                p_wave_coverage=coverage, technical_confidence=confidence,
                flags=flags, restrictions=restrictions, analysis_id=payload.get('analysis_id'),
                probability=None, clinical_claim_allowed=False,
                methodology_source='https://academic.oup.com/eurheartj/article/45/36/3314/7738779',
                source_scope='AF requires ECG confirmation and expert interpretation; not validation of this detector')
