"""Display only the recorded single-lead primitives, never derive new markers."""
import math
from numbers import Real
from ..mimic.lead_measurements import LAB_LEAD_MEASUREMENT_SPEC as LEAD_MEASUREMENT_SPEC, measurement_column
from ..core.data_loader import LEAD_NAMES_12

HEADERS = ('R-Höhe [mV]', 'S-Tiefe [mV]', 'P positiv [mV]', 'P negativ [mV]',
           'P terminale negative Fläche [mV·s]', 'ST-J [mV]', 'T Maximum [mV]', 'T Minimum [mV]',
           'J+20 ms [mV]', 'J+60 ms [mV]', 'J+80 ms [mV]',
           'R-Gipfelzeit, einfache Morphologie [ms]', 'S-Dauer, einfache Morphologie [ms]')


def recorded_lead_rows(payload):
    raw = payload.get('lead_measurements')
    raw = raw if isinstance(raw, dict) else {}
    rows = []
    for lead in LEAD_NAMES_12:
        cells = []
        for family, primitive, unit in LEAD_MEASUREMENT_SPEC:
            key = measurement_column(lead, family, primitive, unit)
            value = raw.get(key)
            finite = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
            cells.append(dict(text=f'{value:.5g}' if finite else 'nicht verfügbar',
                value=float(value) if finite else None, key=key,
                tooltip=f'{key}\nGespeicherter diagnostischer Signalpfad; gemeinsame Mehrkanal-Delineation. '
                        'Keine automatische Diagnose und kein Normwertvergleich.\n'
                        + ('Nur einfache Morphologie: einzelne positive Komponente, eindeutiger R-Gipfel; '
                           'lokale Mindestamplitude 0,05 mV. S-Dauer mit beobachteten Baselinekreuzungen, '
                           'linear interpoliert. Keine vollständige RSB-/LSB-Regel; gemeinsame QRS-Grenzen.\n'
                           if primitive in ('r_peak_time', 's_duration') else '')
                        + ('Zeitpunkt nach J; vorhandene PQ-/Pre-QRS-Baseline des klassischen Rechners. '
                           'Kann bei kurzer Repolarisation in der T-Welle liegen; keine ST-Morphologieklassifikation.\n'
                           if primitive in ('st20', 'st60', 'st80') else '')
                        + ('Messwert vorhanden.' if finite else 'Nicht gespeichert oder nicht auswertbar; kein Ersatz durch null.')))
        rows.append(dict(lead=lead, cells=cells))
    return rows


def lead_matrix_export(payload, signal_reference):
    """Long-form full-precision values, including every missing measurement."""
    rows = []
    for entry in recorded_lead_rows(payload):
        for cell, (family, primitive, unit) in zip(entry['cells'], LEAD_MEASUREMENT_SPEC):
            rows.append(dict(lead=entry['lead'], family=family, measurement=primitive,
                value=cell['value'], unit={'mv': 'mV', 'mvs': 'mV*s', 'ms': 'ms'}[unit],
                status='recorded' if cell['value'] is not None else 'unavailable',
                source_key=cell['key']))
    return dict(schema_version='1.2.0', kind='recorded_lead_measurements',
        analysis_id=payload.get('analysis_id'), signal_reference=dict(signal_reference),
        coordinate_convention='native_12_lead; aVR not display-inverted',
        measurement_scope='stored diagnostic signal path; joint multilead delineation',
        missingness='unavailable conflates not recorded and not evaluable; no cause invented',
        clinical_claim_allowed=False, rows=rows)


def st_j_review(payload):
    """Descriptive J-point measurements, NOT an ESC ischemia classifier.

    Full ST-segment shape, secondary repolarization and clinical context are
    not established by this artifact. Even complete J values do not remove
    that limitation. Native aVR is used, irrespective of display polarity.
    """
    values = {}
    for row in recorded_lead_rows(payload):
        key = measurement_column(row['lead'], 'st', 'j', 'mv')
        values[row['lead']] = next(cell['value'] for cell in row['cells'] if cell['key'] == key)
    depressed = [lead for lead, value in values.items() if value is not None and value <= -.1]
    missing = [lead for lead, value in values.items() if value is None]
    def display(lead):
        return 'nicht verfügbar' if values[lead] is None else f'{values[lead]:.4g} mV'
    return dict(schema_version='1.0.0', scope='descriptive_st_j_review',
        source='https://academic.oup.com/eurheartj/article/44/38/3720/7243210',
        depression_threshold_mv=-.1, observed_depression_leads=depressed,
        observed_depression_count=len(depressed), missing_leads=missing,
        native_avr_mv=values['aVR'], v1_mv=values['V1'],
        clinical_pattern_status='not_assessed', clinical_claim_allowed=False,
        text=(f'ST-J ≤ −0,10 mV: {len(depressed)} von {12-len(missing)} messbaren Ableitungen '
              f'({", ".join(depressed) or "keine"}); {len(missing)} nicht verfügbar. '
              f'Natives aVR: {display("aVR")}; V1: {display("V1")}. '
              'Nur Messübersicht. Das vollständige ESC-ST-Muster wurde nicht bewertet: '
              'ST-Verlauf, sekundäre Repolarisation und klinischer Kontext fehlen. '
              'Keine Hauptstamm-/Infarktdiagnose, kein Ischämieausschluss, keine Therapieempfehlung.'))
