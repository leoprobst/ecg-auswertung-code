"""Describe recorded frontal QRS axes without claiming a new measurement."""
import math
from numbers import Real


def axis_review(payload, age=None):
    result = {'value_deg': None, 'category': 'unknown', 'text': 'QRS-Achse nicht beurteilbar.',
              'method': 'nicht dokumentiert', 'source': 'https://www.jacc.org/doi/10.1016/j.jacc.2008.12.013'}
    criteria = payload.get('classical_criteria')
    criteria = criteria if isinstance(criteria, dict) else {}
    records = [criteria[k] for k in ('left_axis_deviation', 'right_axis_deviation') if isinstance(criteria.get(k), dict)]
    candidates = []
    for item in records:
        value = item.get('value')
        if item.get('evaluable') is not True or isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not -180 <= value <= 180:
            continue
        candidates.append(item)
    if not candidates:
        return result
    if any(not math.isclose(item['value'], candidates[0]['value'], abs_tol=1e-8) for item in candidates[1:]):
        result['text'] = 'Widersprüchliche gespeicherte Achsenwerte; keine automatische Auswahl.'
        return result
    value = float(candidates[0]['value'])
    methods = {item.get('axis_measurement_method', 'nicht dokumentiert') for item in candidates}
    method = next(iter(methods)) if len(methods) == 1 else 'widersprüchliche Methodenangaben'
    result.update(value_deg=value, method=method)
    if isinstance(age, bool) or not isinstance(age, Real) or not math.isfinite(age) or age < 18:
        result['text'] = f'{value:g}°; Erwachsenenreferenz nicht anwendbar oder Alter unbekannt.'
        return result
    # +180 and -180 denote the same direction; assign one boundary category.
    if value == -180:
        value = 180
    category = 'normal_range' if -30 <= value <= 90 else 'left' if -90 <= value < -30 else 'right' if value > 90 else 'extreme'
    label = {'normal_range': 'im Erwachsenenreferenzbereich', 'left': 'Linksabweichung',
             'right': 'Rechtsabweichung', 'extreme': 'extreme Achse (nicht gleichbedeutend mit fehlender Messbarkeit)'}[category]
    result.update(category=category, text=f"{result['value_deg']:g}° · {label}. Methode: {method}. "
        'Amplitudenschätzer ist keine integrierte Flächenachse; keine eigenständige Diagnose.')
    return result
