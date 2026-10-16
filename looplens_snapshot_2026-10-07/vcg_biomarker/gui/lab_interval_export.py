"""Export recorded automatic intervals without presenting them as reviewed."""
import math
from numbers import Integral, Real


def _positive_real(value):
    return (float(value) if isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0 else None)


def _sample(value):
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0 else None


def interval_export(payload, reference):
    delineation = payload.get('delineation')
    items = delineation.get('interval_measurements') if isinstance(delineation, dict) else None
    rows = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        value = item.get('value_ms')
        finite = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
        rate = _positive_real(item.get('sampling_rate_hz'))
        start, end = _sample(item.get('start_sample')), _sample(item.get('end_sample'))
        coordinate_status = 'recorded' if rate is not None and start is not None and end is not None and end > start else 'unavailable_or_invalid'
        rows.append(dict(interval=item.get('interval', 'unknown'), value_ms=float(value) if finite else None,
            recorded_status=item.get('status', 'unknown'), value_status='recorded' if finite else 'unavailable_or_invalid',
            method=item.get('measurement_method', 'not_recorded'), formula=item.get('formula', 'not_recorded'),
            sampling_rate_hz=rate, start_sample=start,
            end_sample=end, representative_r_sample=_sample(item.get('representative_r_sample')),
            coordinate_status=coordinate_status,
            rr_seconds_median=_positive_real(item.get('rr_seconds_median')), n_pairs=_sample(item.get('n_pairs')),
            reason=item.get('reason', ''), interpretation_note=item.get('interpretation_note', ''),
            source=item.get('source', 'not_recorded'), review_status='automatic_not_clinically_reviewed'))
    return dict(schema_version='1.1.0', kind='recorded_automatic_intervals',
        analysis_id=payload.get('analysis_id'), signal_reference=dict(reference), rows=rows,
        clinical_claim_allowed=False, manual_overrides_applied=False,
        scope='Recorded automatic intervals; separate from manual calipers and reviewed report values')
