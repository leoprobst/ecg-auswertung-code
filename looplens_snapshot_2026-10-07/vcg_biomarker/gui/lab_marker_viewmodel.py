"""Display recorded marker metadata without substituting current definitions."""
import math
from numbers import Real
import json


def classical_rule_details(name, rule, analysis_id):
    """Readable evidence from the artifact, without consulting today's registry."""
    recorded = rule if isinstance(rule, dict) else {}
    def display(key):
        value = recorded.get(key)
        if value is None or value == '':
            return 'nicht dokumentiert'
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)
    view = classical_rule_view(recorded)
    text = '\n'.join((
        f'Regel: {name}', f'Analysestand: {analysis_id or "nicht dokumentiert"}',
        f'Gespeicherte Definition: {display("definition")}',
        f'Messwert: {display("value")} · Einheit: {display("unit")}',
        f'Beurteilung: {view["status"]}',
        f'Ausschlussgrund: {display("exclusion_reason")}',
        f'Ableitungen: {display("leads")}',
        f'Quellen-ID: {display("source_id")}',
        f'Regelversion: {display("version")}',
        f'Gespeicherte Quellenverträge: {display("source_records")}',
        f'Gespeicherte Ableitungsgruppen: {display("contiguity_groups")}',
        f'Gruppenregel: {display("contiguity_scope")}',
        f'Registerhash: {display("registry_hash")}',
        f'Quellenlock-Hash: {display("source_lock_sha256")}',
        'Fehlende historische Quellen-/Ableitungsangaben werden nicht aus dem aktuellen Register ergänzt.',
        'Übernahme ist eine lokale manuelle Prüfung dieser Regel, keine unabhängige klinische Validierung.'
    ))
    if name == 'rbbb_pattern':
        text += ('\n\nRSB-Surrogat: Der hier verwendete S-Komponentenpfad prüft die Tiefe '
                 'in I/V6, nicht die terminale S-Dauer. Historische Beschreibungen mit '
                 '„wide terminal S“ belegen deshalb keine gemessene breite S-Zacke. '
                 'Keine vollständige RSB-Diagnose aus diesem explorativen Komponentenwert.')
    if name == 't_inversion_contiguous':
        text += ('\n\nAussagegrenze: Der aktuelle quantitative Teilpfad verlangt T-Inversion >0,10 mV '
                 'und R/S >1 in den dokumentierten Ableitungsgruppen. Die alternative qualitative '
                 'UDMI-Bedingung einer prominenten R-Welle ist damit nicht vollständig implementiert. '
                 'Eine einzelne Aufnahme belegt außerdem nicht, dass die Veränderung neu ist. '
                 'Positiv bedeutet ein erfülltes Messmuster, nicht automatisch akute Ischämie; '
                 'negativ schließt Ischämie nicht aus. Historische Definition oben beachten. '
                 'Quelle: https://www.jacc.org/doi/10.1016/j.jacc.2018.08.1038')
    if name == 'pathological_q_waves':
        text += ('\n\nAussagegrenze: Dieser gespeicherte Messpfad prüft nur den Q-Zacken-Arm, '
                 'nicht die vollständige UDMI-Tabelle für einen früheren Infarkt. '
                 'Die V2-/V3-Sondergrenze der JACC-Fassung lautet >20 ms, nicht ≥20 ms. '
                 'QS-Morphologie und das posteriore R-Wellen-Äquivalent sind getrennte Komponenten. '
                 'Ein negatives Q-Zacken-Ergebnis schließt weder diese Komponenten noch einen Infarkt aus. '
                 'Quelle: https://www.jacc.org/doi/10.1016/j.jacc.2018.08.1038')
    if name in ('st_depression_contiguous', 'st_depression_contiguous_mm'):
        text += ('\n\nAbgrenzung ST-Senkungsprofile: Die gespeicherte Definition oben ist maßgeblich. '
                 'Ein J+60-ms-/Zwei-Ableitungs-Kriterium ist nicht identisch mit ESC 2023, Abb. S3. '
                 'Das korrigierte ESC-Profil nennt am J-Punkt ≥0,05 mV Senkung in V2/V3, sonst ≥0,10 mV, '
                 'mit horizontalem/abfallendem ST-Verlauf über mindestens 80 ms in mindestens einer Ableitung (außer aVR). '
                 'Die Profile unterscheiden sich auch in Messzeitpunkt und Ableitungsanzahl; eine höhere diagnostische '
                 'Spezifität ist dadurch nicht automatisch belegt.\n'
                 'Erratum 2024: Die ursprünglichen 0,05 mm wurden zu 0,5 mm korrigiert (bei 10 mm/mV = 0,05 mV). '
                 'Quelle: https://academic.oup.com/eurheartj/article/45/13/1145/7611911\n'
                 'Dieser Quellenhinweis berechnet kein zusätzliches ESC-Ergebnis und verändert den gespeicherten Befund nicht.')
    return text


# German display names for criteria that have no guideline baseline entry
# (the baseline registry also defines per-endpoint comparators, so adding a
# name there would change the analysis; the label belongs to the display).
_CLASSICAL_LABELS_DE = {
    'st_elevation_contiguous': 'ST-Hebung in benachbarten Ableitungen',
    'st_elevation_contiguous_mm': 'ST-Hebung in benachbarten Ableitungen (mm)',
    'st_depression_contiguous': 'ST-Senkung in benachbarten Ableitungen',
    'st_depression_contiguous_mm': 'ST-Senkung in benachbarten Ableitungen (mm)',
    't_inversion_contiguous': 'T-Negativierung in benachbarten Ableitungen',
    't_amplitude_avr': 'T-Amplitude in aVR',
    'iceb_qt_qrs_ratio': 'iCEB (QT/QRS)',
    'tpeak_tend_qt_ratio': 'Tpeak-Tend/QT',
    'frontal_qrs_t_angle_device': 'Frontaler QRS-T-Winkel (Geräteachsen)',
    'qrs_voltage_sum_12lead': 'QRS-Spannungssumme (12 Ableitungen)',
    'qrs_voltage_duration_product_12lead': 'Spannungs-Dauer-Produkt (12 Ableitungen)',
    'romhilt_estes_points': 'Romhilt-Estes-Punkte',
    'daniel_pe_score': 'Daniel-Score (Lungenembolie)',
    'macruz_index': 'Macruz-Index',
    'lab_diffuse_st_depression_avr_v1_pattern': 'Diffuse ST-Senkung mit Hebung in aVR/V1',
}


def classical_criterion_label(name):
    """German label of a classical criterion id (with or without prefix)."""
    criterion = str(name or '').removeprefix('classical_')
    if criterion in _CLASSICAL_LABELS_DE:
        return _CLASSICAL_LABELS_DE[criterion]
    from ..mimic.user_guidance import biomarker_display_name
    return biomarker_display_name('classical_' + criterion)


def classical_rule_view(rule):
    """One recorded-rule contract for display and explicit manual acceptance."""
    if not isinstance(rule, dict):
        return dict(eligible=False, status='nicht beurteilbar: ungültiger Regelvertrag')
    value = rule.get('value')
    finite = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    if rule.get('exploratory') is not False:
        return dict(eligible=False, status='Forschungsstatus oder Regelherkunft nicht geklärt')
    if rule.get('evaluable') is not True:
        return dict(eligible=False, status='nicht beurteilbar: ' + str(rule.get('exclusion_reason') or 'Grund nicht dokumentiert'))
    if not finite:
        return dict(eligible=False, status='nicht beurteilbar: Messwert fehlt oder ist ungültig trotz Auswertbarkeitsflag')
    positive = rule.get('positive')
    if positive is not None and not isinstance(positive, bool):
        return dict(eligible=False, status='nicht beurteilbar: ungültiges binäres Urteil')
    status = 'erfüllt (Kriterium, keine Diagnose)' if positive is True else 'nicht erfüllt (kein Krankheitsausschluss)' if positive is False else 'Messwert ohne binäres Urteil'
    return dict(eligible=True, status=status)


def classical_measurement_cells(name, rule):
    """Same assessability contract in the measurement table and rule inspector."""
    rule = rule if isinstance(rule, dict) else {}
    value = rule.get('value')
    finite = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    measured = str(value) if finite else 'nicht verfügbar'
    unit = rule.get('unit') or 'Einheit nicht dokumentiert'
    status = classical_rule_view(rule)['status']
    if rule.get('exploratory') is True:
        if rule.get('evaluable') is True and finite and type(rule.get('positive')) is bool:
            status += ' · exploratives Muster ' + ('erfüllt' if rule['positive'] else 'nicht erfüllt (kein Ausschluss)')
        else:
            status += ' · nicht auswertbar: ' + str(rule.get('exclusion_reason') or 'Messnachweis fehlt')
        status += ' · Gespeicherte Definition: ' + str(rule.get('definition') or 'nicht dokumentiert')
    if name in {'pr_prolongation_gt_200', 'first_degree_av_block'}:
        status += (' · PR-Messbefund, kein automatischer AV-Block-Nachweis: '
                   '1:1-P–QRS-Überleitung gesondert prüfen.')
        if finite and value > 300 and rule.get('evaluable') is True and rule.get('unit') == 'ms':
            status += ' Ausgeprägte PR-Verlängerung >300 ms; klinische Bedeutung kontextabhängig.'
    return (name, f'{measured} · {unit}', f'Automatische Regel · {status}')


def finding_validation_metrics(finding):
    """Read validation-only operating metrics, including legacy capsules.

    Unqualified legacy top-level values may contain a training fallback. Never
    infer their split from the enclosing table or from the artifact's age.
    """
    primary = finding.get('primary_operating_point')
    primary = primary if isinstance(primary, dict) else {}
    result = {}
    for metric in ('sensitivity', 'specificity'):
        explicit = primary.get(metric + '_validation')
        declared = finding.get(metric + '_source')
        top = finding.get(metric)

        def valid(value):
            return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1

        value, reason = None, 'Validierungsherkunft nicht dokumentiert; kein Trainings-Fallback.'
        if metric + '_validation' in primary:
            if not valid(explicit):
                reason = 'Expliziter Validierungswert fehlt oder ist ungültig.'
            elif declared == 'validation_operating_point' and (not valid(top) or not math.isclose(top, explicit, rel_tol=1e-9, abs_tol=1e-12)):
                reason = 'Widersprüchliche Validierungswerte im gespeicherten Artefakt.'
            else:
                value = float(explicit)
                reason = 'Expliziter Wert der Validierung am gespeicherten Trainingsgrenzwert.'
        elif declared == 'validation_operating_point' and valid(top):
            value = float(top)
            reason = 'Im Artefakt als Validierungswert am Trainingsgrenzwert ausgewiesen.'
        result[metric] = value
        result[metric + '_note'] = reason
    return result


def marker_row(name, value, measurement):
    recorded = measurement if isinstance(measurement, dict) else {}
    finite = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    unit = recorded.get('unit')
    unit_text = str(unit) if unit else 'Einheit nicht dokumentiert'
    status = 'Messwert vorhanden' if finite else 'nicht berechenbar / ungültig'
    notes = str(recorded.get('notes') or 'Kein Messhinweis im Artefakt')
    family = str(recorded.get('family') or '')
    description = str(recorded.get('description') or '')
    return {
        'name': name, 'value_text': f'{value:.6g}' if finite else 'nicht berechenbar',
        'unit_status': f'{unit_text} · {status} · Forschung',
        'tooltip': f'{description}\nFamilie: {family or "nicht dokumentiert"}\n{notes}\n'
                   'Metadaten des gespeicherten Analysestands; keine Diagnose.',
        'search_text': ' '.join((name, family, description)).casefold(),
    }
